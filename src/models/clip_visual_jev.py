from __future__ import annotations

from collections.abc import Sequence

import clip
import torch
import torch.nn as nn


DEFAULT_QUESTION = "Which description best matches this facial image?"
DEFAULT_CANDIDATES = (
    "an authentic facial image.",
    "a manipulated deepfake facial image.",
)


def format_candidate_prompt(question: str, candidate: str) -> str:
    """Format the fixed question/candidate input specified by the H1 protocol."""
    return f"Question: {question} Candidate: {candidate}"


class CLIPVisualJevClassifier(nn.Module):
    """Minimal Visual-Jev with one shared scorer over semantic candidates.

    The CLIP visual tower is expected to be initialized from baseline A after
    construction. Both CLIP encoders remain frozen. Since the H1 protocol uses
    two fixed candidate strings, their frozen text embeddings are computed once
    at construction time and stored as a non-trainable buffer.
    """

    def __init__(
        self,
        clip_model_name: str = "ViT-B/16",
        question: str = DEFAULT_QUESTION,
        candidates: Sequence[str] = DEFAULT_CANDIDATES,
        hidden_dims: tuple[int, int] = (512, 128),
    ) -> None:
        super().__init__()
        if len(candidates) != 2:
            raise ValueError(f"H1 requires exactly two candidates, got {len(candidates)}")

        clip_model, _ = clip.load(clip_model_name, device="cpu")
        self.visual = clip_model.visual
        self.question = question
        self.candidates = tuple(candidates)

        prompts = [format_candidate_prompt(question, candidate) for candidate in candidates]
        tokens = clip.tokenize(prompts, truncate=True)
        with torch.no_grad():
            candidate_embeddings = clip_model.encode_text(tokens).float()

        feature_dim = int(self.visual.output_dim)
        if candidate_embeddings.shape != (2, feature_dim):
            raise ValueError(
                "CLIP image/text embedding dimensions must match: "
                f"text={tuple(candidate_embeddings.shape)}, image_dim={feature_dim}"
            )
        self.feature_dim = feature_dim
        self.register_buffer("candidate_embeddings", candidate_embeddings, persistent=True)

        self.scorer = nn.Sequential(
            nn.Linear(4 * feature_dim, hidden_dims[0]),
            nn.GELU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.GELU(),
            nn.Linear(hidden_dims[1], 1),
        )

        for parameter in self.visual.parameters():
            parameter.requires_grad_(False)
        self.visual.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        # The frozen encoder must not change behavior when the head is trained.
        self.visual.eval()
        return self

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.visual(images)

    def classify_features(
        self,
        image_features: torch.Tensor,
        candidate_order: Sequence[int] = (0, 1),
    ) -> torch.Tensor:
        image_features = image_features.float()
        order = torch.as_tensor(
            tuple(candidate_order),
            dtype=torch.long,
            device=self.candidate_embeddings.device,
        )
        if order.shape != (2,) or set(order.detach().cpu().tolist()) != {0, 1}:
            raise ValueError("candidate_order must be a permutation of (0, 1)")

        text_features = self.candidate_embeddings.index_select(0, order).float()
        batch_size = image_features.shape[0]
        image_by_candidate = image_features[:, None, :].expand(batch_size, 2, self.feature_dim)
        text_by_image = text_features[None, :, :].expand(batch_size, 2, self.feature_dim)
        joint_features = torch.cat(
            (
                image_by_candidate,
                text_by_image,
                image_by_candidate * text_by_image,
                torch.abs(image_by_candidate - text_by_image),
            ),
            dim=-1,
        )
        return self.scorer(joint_features).squeeze(-1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classify_features(self.extract_features(images))
