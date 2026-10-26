"""Frozen-CLIP semantic Visual-Jev decision model."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import clip
import torch
import torch.nn as nn

from data.semantic_language import LanguageBank


class Projection(nn.Module):
    """Independent layer-normalized projection into decision space."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(input_dim)
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(self.normalization(inputs))


class CLIPSemanticVisualJev(nn.Module):
    """Cross-attention decision model over image, question, and candidates."""

    def __init__(
        self,
        clip_model_name: str = "ViT-B/16",
        d_model: int = 256,
        nhead: int = 8,
        ffn_dim: int = 1024,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.clip, _ = clip.load(clip_model_name, device="cpu")
        self._freeze_clip()

        feature_dim = int(self.clip.visual.output_dim)
        self.image_projection = Projection(feature_dim, d_model)
        self.question_projection = Projection(feature_dim, d_model)
        self.candidate_projection = Projection(feature_dim, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
        )
        self.scorer = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

    def _freeze_clip(self) -> None:
        for parameter in self.clip.parameters():
            parameter.requires_grad_(False)
        self.clip.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.clip.eval()
        return self

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.clip.encode_image(images).float()

    def encode_texts(self, texts: Sequence[str]) -> torch.Tensor:
        tokens = clip.tokenize(list(texts), truncate=True).to(
            next(self.parameters()).device
        )
        with torch.no_grad():
            return self.clip.encode_text(tokens).float()

    def classify_features(
        self,
        image_features: torch.Tensor,
        question_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        memory = torch.stack(
            (
                self.image_projection(image_features.float()),
                self.question_projection(question_embeddings.float()),
            ),
            dim=1,
        )
        candidate_tokens = self.candidate_projection(
            candidate_embeddings.float()
        )
        decoded = self.decoder(
            tgt=candidate_tokens,
            memory=memory,
            tgt_key_padding_mask=~candidate_mask,
        )
        logits = self.scorer(decoded).squeeze(-1)
        return logits.masked_fill(~candidate_mask, float("-inf"))

    def forward(
        self,
        images: torch.Tensor,
        question_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.classify_features(
            self.extract_features(images),
            question_embeddings,
            candidate_embeddings,
            candidate_mask,
        )

    def load_h1_visual_checkpoint(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict):
            raise ValueError("H1 checkpoint must contain a state dictionary")
        state = checkpoint.get("model_state_dict", checkpoint)
        if not isinstance(state, dict):
            raise ValueError("H1 checkpoint must contain model_state_dict")
        visual_state = {
            key.removeprefix("visual."): value
            for key, value in state.items()
            if key.startswith("visual.")
        }
        if not visual_state:
            raise ValueError("H1 checkpoint contains no visual weights")
        self.clip.visual.load_state_dict(visual_state, strict=True)
        self._freeze_clip()


@dataclass
class EncodedTextLookup:
    """Dense, split-scoped mapping from language-bank IDs to CLIP vectors."""

    row_for_text_id: torch.Tensor
    embeddings: torch.Tensor

    @classmethod
    def build(
        cls,
        model: CLIPSemanticVisualJev,
        bank: LanguageBank,
        allowed_splits: Sequence[str],
        device: torch.device,
    ) -> EncodedTextLookup:
        allowed = frozenset(allowed_splits)
        entries = tuple(
            entry for entry in bank.text_entries if entry.split in allowed
        )
        if not entries:
            raise ValueError("allowed_splits selected no language-bank text")

        embeddings = model.encode_texts([entry.text for entry in entries]).to(
            device
        )
        largest_text_id = max(entry.text_id for entry in bank.text_entries)
        row_for_text_id = torch.full(
            (largest_text_id + 1,),
            -1,
            dtype=torch.long,
            device=device,
        )
        text_ids = torch.tensor(
            [entry.text_id for entry in entries],
            dtype=torch.long,
            device=device,
        )
        row_for_text_id[text_ids] = torch.arange(
            len(entries), dtype=torch.long, device=device
        )
        return cls(
            row_for_text_id=row_for_text_id,
            embeddings=embeddings,
        )

    def lookup(
        self,
        question_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text_id_count = self.row_for_text_id.shape[0]
        invalid_questions = (question_ids < 0) | (question_ids >= text_id_count)
        invalid_candidates = candidate_mask & (
            (candidate_ids < 0) | (candidate_ids >= text_id_count)
        )
        if invalid_questions.any() or invalid_candidates.any():
            raise KeyError("text ID is not encoded in this split-scoped lookup")

        question_rows = self.row_for_text_id[question_ids]
        safe_candidate_ids = candidate_ids.masked_fill(~candidate_mask, 0)
        candidate_rows = self.row_for_text_id[safe_candidate_ids]
        if (question_rows < 0).any() or (
            candidate_rows[candidate_mask] < 0
        ).any():
            raise KeyError("text ID is not encoded in this split-scoped lookup")
        questions = self.embeddings[question_rows]
        candidates = self.embeddings[candidate_rows.clamp_min(0)]
        candidates = candidates.masked_fill(
            ~candidate_mask[..., None],
            0.0,
        )
        return questions, candidates
