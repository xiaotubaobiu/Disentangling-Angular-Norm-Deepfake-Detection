"""Pretrained multimodal token interaction with one shared candidate scorer.

There is no vocabulary projection, next-token loss, or generation loop. Every
candidate gets an independent image/question/candidate context. Candidate IDs
are only lookup keys for text; labels and semantic IDs never enter the model.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence


class CandidatePrompts:
    """Use the official chat template and unsplit 64-token image expansion."""

    def __init__(self, processor):
        self.processor = processor
        self.pad_token_id = processor.tokenizer.pad_token_id
        if self.pad_token_id is None:
            raise ValueError("pretrained tokenizer needs a padding token")

    def encode_pair(self, question: str, candidate: str) -> list[int]:
        text = f"Question: {question}\nCandidate answer: {candidate}\nEvaluate whether this candidate correctly answers the question about the image."
        chat = self.processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}],
            add_generation_prompt=True, tokenize=False,
        )
        p = self.processor
        image_tokens = p.fake_image_token + p.global_image_tag + p.image_token * p.image_seq_len + p.fake_image_token
        if chat.count(p.image_token) != 1:
            raise ValueError("candidate context must contain exactly one image")
        expanded = chat.replace(p.image_token, image_tokens)
        ids = p.tokenizer(expanded, add_special_tokens=True)["input_ids"]
        if ids.count(p.image_token_id) != p.image_seq_len:
            raise ValueError("candidate context and pretrained image-token count disagree")
        if len(ids) > 1024:
            raise ValueError("candidate prompt exceeds the frozen context limit")
        return ids


class SmolVLMCandidateScorer(nn.Module):
    def __init__(self, backbone, prompts, *, score_batch_size: int = 32):
        super().__init__()
        if any(name == "lm_head" or name.endswith(".lm_head") for name, _ in backbone.named_modules()):
            raise ValueError("candidate scorer requires the base model without a vocabulary head")
        if score_batch_size < 1:
            raise ValueError("score_batch_size must be positive")
        self.backbone = backbone
        self.prompts = prompts
        self.score_batch_size = score_batch_size
        hidden = backbone.config.text_config.hidden_size
        self.scorer = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
        self._texts: dict[int, str] = {}
        self._prompt_cache: dict[tuple[int, int], torch.Tensor] = {}
        for module in (backbone.vision_model, backbone.connector):
            module.requires_grad_(False)
            module.eval()

    @classmethod
    def from_pretrained(cls, path: str | Path, *, score_batch_size=32,
                        gradient_checkpointing=False):
        from transformers import AutoModel, AutoProcessor

        # The official 256M checkpoint declares model_type=idefics3.
        # AutoModel selects its base multimodal implementation, not a generator.
        backbone, info = AutoModel.from_pretrained(
            path, local_files_only=True, dtype=torch.float32,
            attn_implementation="sdpa", output_loading_info=True,
        )
        if info.get("missing_keys") or info.get("mismatched_keys") or info.get("error_msgs"):
            raise ValueError(f"incomplete pretrained backbone load: {info}")
        unexpected = [name for name in info.get("unexpected_keys", []) if name != "lm_head.weight"]
        if unexpected:
            raise ValueError(f"unexpected pretrained backbone keys: {unexpected}")
        processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        processor.tokenizer.padding_side = "right"
        model = cls(backbone, CandidatePrompts(processor), score_batch_size=score_batch_size)
        if gradient_checkpointing:
            backbone.text_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        model.pretrained_loading_info = {
            key: sorted(value) if isinstance(value, set) else value
            for key, value in info.items()
        }
        return model, processor

    def train(self, mode=True):
        super().train(mode)
        self.backbone.vision_model.eval()
        self.backbone.connector.eval()
        return self

    def register_texts(self, entries):
        for entry in entries:
            identity = int(entry.text_id)
            if identity in self._texts and self._texts[identity] != entry.text:
                raise ValueError("a text ID was rebound to a different string")
            self._texts[identity] = entry.text

    @torch.no_grad()
    def extract_features(self, images):
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("expected a Bx3xHxW preprocessed image batch")
        features = self.backbone.get_image_features(
            pixel_values=images[:, None], return_dict=True).pooler_output
        if features.shape[:2] != (images.shape[0], self.backbone.image_seq_len):
            raise ValueError("pretrained vision output lost an image or token sequence")
        return features

    def _prompt(self, question_id, candidate_id):
        key = (question_id, candidate_id)
        if key not in self._prompt_cache:
            self._prompt_cache[key] = torch.tensor(self.prompts.encode_pair(
                self._texts[question_id], self._texts[candidate_id]), dtype=torch.long)
        return self._prompt_cache[key]

    def classify_features(self, image_features, questions, candidates, candidate_mask):
        if questions.ndim != 2 or questions.shape[1] != 1 or candidates.shape != (*candidate_mask.shape, 1):
            raise ValueError("question/candidate inputs must be registered text lookup IDs")
        if not torch.all(candidate_mask.sum(1) >= 2):
            raise ValueError("every sample needs at least two candidates")
        owners, positions = candidate_mask.nonzero(as_tuple=True)
        qids = questions[:, 0].detach().cpu().tolist()
        cids = candidates[..., 0].detach().cpu().tolist()
        pairs = [(qids[i], cids[i][j]) for i, j in zip(owners.cpu().tolist(), positions.cpu().tolist())]
        scores = []
        device = image_features.device
        for start in range(0, len(pairs), self.score_batch_size):
            group = pairs[start:start + self.score_batch_size]
            sequences = [self._prompt(q, c) for q, c in group]
            lengths = torch.tensor([len(seq) for seq in sequences], device=device)
            ids = pad_sequence(sequences, batch_first=True, padding_value=self.prompts.pad_token_id).to(device)
            attention = torch.arange(ids.shape[1], device=device)[None, :] < lengths[:, None]
            output = self.backbone(
                input_ids=ids, attention_mask=attention,
                image_hidden_states=image_features.index_select(0, owners[start:start + len(group)]),
                use_cache=False, return_dict=True,
            )
            # Last non-padding input position has attended to the full image/Q/C.
            hidden = output.last_hidden_state[torch.arange(len(group), device=device), lengths - 1]
            scores.append(self.scorer(hidden.float()).squeeze(-1))
        logits = torch.full(candidate_mask.shape, float("-inf"), dtype=torch.float32, device=device)
        logits[owners, positions] = torch.cat(scores).float()
        return logits

    def forward(self, images, questions, candidates, candidate_mask):
        return self.classify_features(self.extract_features(images), questions, candidates, candidate_mask)


@dataclass
class CandidateTextLookup:
    """Enforce text split isolation without exposing IDs as learned features."""

    allowed: torch.Tensor

    @classmethod
    def build(cls, model, bank, allowed_splits, device):
        entries = tuple(entry for entry in bank.text_entries if entry.split in allowed_splits)
        if not entries:
            raise ValueError("empty permitted text split")
        model.register_texts(entries)
        allowed = torch.zeros(max(e.text_id for e in bank.text_entries) + 1, dtype=torch.bool, device=device)
        allowed[[entry.text_id for entry in entries]] = True
        return cls(allowed)

    def lookup(self, question_ids, candidate_ids, candidate_mask):
        for values in (question_ids, candidate_ids[candidate_mask]):
            if ((values < 0) | (values >= len(self.allowed))).any():
                raise KeyError("text ID is outside the permitted split")
            if not self.allowed[values].all():
                raise KeyError("text ID is outside the permitted split")
        return question_ids[:, None], candidate_ids[..., None]
