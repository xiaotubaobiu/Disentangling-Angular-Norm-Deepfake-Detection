"""Native BridgeTower ITM scoring shared across every candidate."""
from __future__ import annotations

from pathlib import Path

import torch
from transformers import (BridgeTowerForImageAndTextRetrieval,
                          BridgeTowerImageProcessor, BridgeTowerProcessor,
                          RobertaTokenizer)


def compose(task: str, question: str, candidate: str) -> str:
    if task not in {'T1', 'T2', 'T3', 'T4', 'T5'}:
        raise ValueError(f'unknown task {task}')
    q, c = question.strip(), candidate.strip()
    if not q or not c:
        raise ValueError('question and candidate must be nonempty')
    if task in {'T2', 'T4'}:
        return f'Question: {q} Answer: {c}'
    return f'Question: {q} Answer: This is {c}'


class SharedBridgeScorer(torch.nn.Module):
    def __init__(self, checkpoint: str | Path):
        super().__init__()
        checkpoint = Path(checkpoint)
        tokenizer_path = checkpoint.parent / 'roberta-base-tokenizer'
        if not tokenizer_path.is_dir():
            raise FileNotFoundError(f'pinned RoBERTa tokenizer missing: {tokenizer_path}')
        # The pinned legacy config has no model_type; instantiate processor parts
        # explicitly while retaining the original checkpoint bytes. The fast
        # tokenizer bundled with Transformers 5.5 splits words into characters;
        # the slow tokenizer with pinned RoBERTa merges restores pretrained BPE.
        self.processor = BridgeTowerProcessor(
            BridgeTowerImageProcessor.from_pretrained(str(checkpoint)),
            RobertaTokenizer.from_pretrained(str(tokenizer_path)))
        self.model = BridgeTowerForImageAndTextRetrieval.from_pretrained(str(checkpoint))
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        for name, module in self.model.bridgetower.named_children():
            if name not in {'vision_model', 'text_model'}:
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
        for parameter in self.model.itm_score.parameters():
            parameter.requires_grad_(True)

    def counts(self) -> dict[str, int]:
        return {'total': sum(p.numel() for p in self.model.parameters()),
                'trainable': sum(p.numel() for p in self.model.parameters() if p.requires_grad)}

    def forward(self, images, tasks, questions, candidates):
        """Return padded candidate logits; semantic labels never enter the model."""
        texts, repeated_images, spans = [], [], []
        for image, task, question, options in zip(images, tasks, questions, candidates, strict=True):
            start = len(texts)
            for option in options:
                texts.append(compose(task, question, option))
                repeated_images.append(image)
            spans.append((start, len(texts)))
        if not texts:
            raise ValueError('empty candidate batch')
        encoding = self.processor(images=repeated_images, text=texts,
                                  return_tensors='pt', padding=True, truncation=True)
        device = next(self.model.parameters()).device
        encoding = {key: value.to(device) for key, value in encoding.items()}
        # The match column is the checkpoint's native shared ITM score.
        match = self.model(**encoding).logits[:, 1]
        width = max(end - start for start, end in spans)
        rows = []
        for start, end in spans:
            row = match[start:end]
            if end - start < width:
                row = torch.nn.functional.pad(row, (0, width - (end - start)), value=-1e4)
            rows.append(row)
        return torch.stack(rows)
