"""Evaluation-only batching with candidate-independent FP32 compute shapes.

The trained architecture and state dict are unchanged. Each prompt is assigned
to a length bucket independently of the other candidates. Every forward uses
the configured pair batch size, repeating the last valid row to fill a partial
batch. Dummy scores are discarded. Thus subgrouping/reordering cannot change a
real candidate's sequence or batch dimensions. All evaluation conditions use
this same path; the frozen E2 permutation audit is unchanged.
"""
from __future__ import annotations

import torch

from models.smolvlm_candidate import SmolVLMCandidateScorer

SCORING_PROFILE = 'fp32_per_prompt_length_buckets_128_256_512_1024_fixed_pair_batch_v1'


class FixedShapeCandidateScorer(SmolVLMCandidateScorer):
    @torch.inference_mode()
    def classify_features(self, image_features, questions, candidates, candidate_mask):
        if self.training:
            raise ValueError('fixed-shape scoring is for evaluation only')
        if questions.ndim != 2 or questions.shape[1] != 1 or candidates.shape != (*candidate_mask.shape, 1):
            raise ValueError('question/candidate inputs must be registered text lookup IDs')
        if not torch.all(candidate_mask.sum(1) >= 2):
            raise ValueError('every sample needs at least two candidates')
        owners, positions = candidate_mask.nonzero(as_tuple=True)
        qids = questions[:, 0].detach().cpu().tolist()
        cids = candidates[..., 0].detach().cpu().tolist()
        buckets = {length: [] for length in (128, 256, 512, 1024)}
        for index, (owner, position) in enumerate(zip(owners.cpu().tolist(), positions.cpu().tolist())):
            sequence = self._prompt(qids[owner], cids[owner][position])
            if not 0 < len(sequence) <= 1024:
                raise ValueError('candidate prompt must contain 1 through 1024 tokens')
            bucket = next(length for length in buckets if len(sequence) <= length)
            buckets[bucket].append((index, sequence))
        device = image_features.device
        logits = torch.full(candidate_mask.shape, float('-inf'), dtype=torch.float32, device=device)
        for length, entries in buckets.items():
            for start in range(0, len(entries), self.score_batch_size):
                group = entries[start:start + self.score_batch_size]
                count = len(group)
                padded = group + [group[-1]] * (self.score_batch_size - count)
                indices = torch.tensor([entry[0] for entry in padded], device=device)
                lengths = torch.tensor([len(entry[1]) for entry in padded], device=device)
                ids = torch.full((self.score_batch_size, length), self.prompts.pad_token_id, dtype=torch.long)
                for row, (_, sequence) in enumerate(padded):
                    ids[row, :len(sequence)] = sequence
                ids = ids.to(device)
                attention = torch.arange(length, device=device)[None] < lengths[:, None]
                output = self.backbone(
                    input_ids=ids, attention_mask=attention,
                    image_hidden_states=image_features.index_select(0, owners[indices]),
                    use_cache=False, return_dict=True)
                hidden = output.last_hidden_state[
                    torch.arange(self.score_batch_size, device=device), lengths - 1]
                # Keep the scalar head's batch shape fixed too.
                scores = self.scorer(hidden.float()).squeeze(-1)[:count]
                real_indices = indices[:count]
                logits[owners[real_indices], positions[real_indices]] = scores.float()
        return logits
