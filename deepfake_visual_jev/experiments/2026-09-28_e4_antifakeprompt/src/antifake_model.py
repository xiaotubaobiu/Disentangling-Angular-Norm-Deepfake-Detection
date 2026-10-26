"""One pretrained backbone; original vocabulary vs shared scalar decisions."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def last_attended(hidden, mask):
    if mask.shape != hidden.shape[:2] or not mask.bool().any(dim=1).all():
        raise ValueError('invalid or empty attention mask')
    positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
    indices = positions.masked_fill(~mask.bool(), -1).max(dim=1).values
    return hidden[torch.arange(len(hidden), device=hidden.device), indices]


def answer_nll(logits, labels):
    targets = labels[:, 1:]
    counts = (targets != -100).sum(1)
    if (counts == 0).any():
        raise ValueError('empty answer token labels')
    losses = F.cross_entropy(logits[:, :-1].float().transpose(1, 2), targets,
                             ignore_index=-100, reduction='none')
    return losses.sum(1), counts


def soft_prompt_question(question):
    if '[*]' in question:
        return question
    question = question.rstrip()
    return question[:-1] + ' [*]?' if question.endswith('?') else question + ' [*]'


class AntifakeScorer(nn.Module):
    def __init__(self, backbone, *, mode, gradient_checkpointing=True):
        super().__init__()
        if mode not in {'lm', 'scalar'}:
            raise ValueError(mode)
        self.backbone, self.mode = backbone, mode
        for param in backbone.parameters():
            param.requires_grad_(False)
        for name, param in backbone.named_parameters():
            if name.endswith(('train_word_embeddings.weight', 'train_embed_tokens.weight')):
                param.data = param.data.float()
                param.requires_grad_(True)
        hidden = backbone.llm_model.config.hidden_size
        if mode == 'scalar':
            backbone.llm_model.lm_head = None
            self.score_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
            self.score_head.to(next(backbone.parameters()).device)
        self.gradient_checkpointing = gradient_checkpointing
        backbone.llm_model.model.gradient_checkpointing = gradient_checkpointing
        backbone.llm_model.config.use_cache = False

    def train(self, mode=True):
        super().train(mode)
        # Disable frozen vision/Q-former dropout in both arms. A frozen language
        # decoder still needs training=True for activation checkpointing.
        self.backbone.visual_encoder.eval()
        self.backbone.Qformer.eval()
        return self

    def trainable_state_dict(self):
        return {n: p.detach().cpu().clone() for n, p in self.named_parameters() if p.requires_grad}

    def load_trainable(self, state):
        expected = {n for n, p in self.named_parameters() if p.requires_grad}
        if set(state) != expected:
            raise ValueError('checkpoint trainable key mismatch')
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name in state:
                    param.copy_(state[name])

    @torch.no_grad()
    def extract_features(self, images):
        b = self.backbone
        with b.maybe_autocast():
            return b.ln_vision(b.visual_encoder(images))

    def _visual_prefix(self, visual, questions):
        b = self.backbone
        query = b.query_tokens.expand(len(visual), -1, -1)
        qt = b.tokenizer(questions, padding=True, return_tensors='pt').to(visual.device)
        if qt.input_ids.shape[1] > b.max_txt_len:
            raise ValueError('question exceeds Q-former limit; no silent truncation')
        query_mask = torch.ones(query.shape[:2], device=visual.device, dtype=torch.long)
        with b.maybe_autocast():
            qout = b.Qformer.bert(qt.input_ids,
                attention_mask=torch.cat([query_mask, qt.attention_mask], 1),
                query_embeds=query, encoder_hidden_states=visual,
                encoder_attention_mask=torch.ones(visual.shape[:2], device=visual.device, dtype=torch.long),
                return_dict=True)
            return b.llm_proj(qout.last_hidden_state[:, :query.shape[1]])

    def _language_inputs(self, prefix, questions, answers, *, eos=False):
        b = self.backbone
        tok = b.llm_tokenizer
        tok.padding_side = 'right'
        prompt_tokens = tok(questions, return_tensors='pt', padding=True).to(prefix.device)
        output_tokens = tok([a + tok.eos_token if eos else a for a in answers],
                            return_tensors='pt', padding=True).to(prefix.device)
        packed, lengths = b.concat_text_input_output(prompt_tokens.input_ids, prompt_tokens.attention_mask,
                                                      output_tokens.input_ids, output_tokens.attention_mask)
        ids = packed['input_ids']
        embeddings = b.llm_model.get_input_embeddings()
        pseudo = ids >= embeddings.num_embeddings
        if (ids[pseudo] != embeddings.num_embeddings).any():
            raise ValueError('unexpected added token beyond one soft prompt')
        text = embeddings(ids.masked_fill(pseudo, 0))
        text = torch.where(pseudo.unsqueeze(-1),
                           b.llm_model.model.train_embed_tokens.weight[0].to(text.dtype), text)
        p_mask = torch.ones(prefix.shape[:2], device=prefix.device, dtype=torch.long)
        labels = ids.masked_fill(packed['attention_mask'] == 0, -100).clone()
        for i, length in enumerate(lengths):
            labels[i, :length] = -100
        labels = torch.cat([torch.full_like(p_mask, -100), labels], 1)
        attention = torch.cat([p_mask, packed['attention_mask']], 1)
        inputs = torch.cat([prefix.to(text.dtype), text], 1)
        return inputs, attention, labels

    def _score_pairs(self, prefix, questions, answers, *, eos=False):
        inputs, attention, labels = self._language_inputs(prefix, questions, answers, eos=eos)
        b = self.backbone
        with b.maybe_autocast():
            hidden = b.llm_model.model(inputs_embeds=inputs, attention_mask=attention,
                                       use_cache=False, return_dict=True).last_hidden_state
            if self.mode == 'scalar':
                return self.score_head(last_attended(hidden, attention).float()).squeeze(-1).float(), None
            logits = b.llm_model.lm_head(hidden)
        sums, counts = answer_nll(logits, labels)
        return -sums, counts

    @staticmethod
    def _validate(questions, candidates, mask):
        if mask.ndim != 2 or mask.dtype != torch.bool or len(questions) != len(candidates) or len(mask) != len(questions):
            raise ValueError('invalid candidate batch')
        if (mask.sum(1) < 2).any():
            raise ValueError('need at least two candidates')
        if any(len(row) != mask.shape[1] for row in candidates):
            raise ValueError('candidate shape mismatch')
        if any(not candidates[i][j].strip() for i, j in mask.nonzero().tolist()):
            raise ValueError('empty valid candidate')

    def forward(self, images, questions, candidates, mask):
        return self.classify_features(self.extract_features(images), questions, candidates, mask)

    def classify_features(self, visual, questions, candidates, mask):
        self._validate(questions, candidates, mask)
        questions = [soft_prompt_question(q) for q in questions]
        prefix = self._visual_prefix(visual, questions)
        pairs = mask.nonzero(as_tuple=False)
        scores, _ = self._score_pairs(prefix[pairs[:, 0]],
            [questions[i] for i, _ in pairs.tolist()],
            [candidates[i][j] for i, j in pairs.tolist()])
        logits = scores.new_full(mask.shape, -torch.inf)
        logits[pairs[:, 0], pairs[:, 1]] = scores
        return logits

    def training_loss(self, images, questions, candidates, mask, targets):
        self._validate(questions, candidates, mask)
        if not mask[torch.arange(len(targets), device=mask.device), targets].all():
            raise ValueError('invalid target')
        if self.mode == 'scalar':
            return F.cross_entropy(self(images, questions, candidates, mask), targets)
        questions = [soft_prompt_question(q) for q in questions]
        prefix = self._visual_prefix(self.extract_features(images), questions)
        answers = [candidates[i][target] for i, target in enumerate(targets.tolist())]
        scores, counts = self._score_pairs(prefix, questions, answers, eos=True)
        return -scores.sum() / counts.sum()
