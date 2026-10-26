from __future__ import annotations

from models.efficientnet import EfficientNetBinaryClassifier
from models.dinov2 import DINOv2BinaryClassifier
from models.xception import XceptionBinaryClassifier
from models.clip_classifier import CLIPFineTuneBinaryClassifier
from models.clip_prompt import CLIPPromptBinaryClassifier
from models.clip_bgcontrast import CLIPBgFaceContrastModel
from models.clip_semantic_jev import CLIPSemanticVisualJev
from models.clip_visual_jev import CLIPVisualJevClassifier


def build_model(model_cfg: dict):
    name = model_cfg["name"]
    clip_name = model_cfg.get("clip_model_name", "ViT-B/16")
    if name == "efficientnet_b0":
        return EfficientNetBinaryClassifier(
            normalize_features=model_cfg.get("normalize_features", False),
        )
    if name == "dinov2_vitb14":
        return DINOv2BinaryClassifier(
            normalize_features=model_cfg.get("normalize_features", False),
        )
    if name == "xception":
        return XceptionBinaryClassifier(
            normalize_features=model_cfg.get("normalize_features", False),
        )
    if name == "clip_finetune":
        return CLIPFineTuneBinaryClassifier(
            clip_name,
            model_cfg.get("normalize_features", False),
        )
    if name == "clip_prompt":
        tau = model_cfg.get("tau", 0.07)
        freeze_visual = model_cfg.get("freeze_visual", "none")
        freeze_visual_layers = model_cfg.get("freeze_visual_layers", 9)
        projection_dim = model_cfg.get("projection_dim", 512)
        classifier_normalize_features = model_cfg.get("classifier_normalize_features", True)
        return CLIPPromptBinaryClassifier(
            clip_name,
            tau,
            freeze_visual,
            freeze_visual_layers,
            projection_dim,
            classifier_normalize_features,
        )
    if name == "clip_prompt_bgcontrast":
        proj_dim = model_cfg.get("projection_dim", 256)
        return CLIPBgFaceContrastModel(clip_name, proj_dim)
    if name == "clip_visual_jev":
        hidden_dims = tuple(model_cfg.get("hidden_dims", [512, 128]))
        return CLIPVisualJevClassifier(
            clip_model_name=clip_name,
            question=model_cfg.get(
                "question",
                "Which description best matches this facial image?",
            ),
            candidates=model_cfg.get(
                "candidates",
                [
                    "an authentic facial image.",
                    "a manipulated deepfake facial image.",
                ],
            ),
            hidden_dims=hidden_dims,
        )
    if name == "clip_semantic_jev":
        return CLIPSemanticVisualJev(
            clip_model_name=clip_name,
            d_model=model_cfg.get("d_model", 256),
            nhead=model_cfg.get("nhead", 8),
            ffn_dim=model_cfg.get("ffn_dim", 1024),
            num_layers=model_cfg.get("num_layers", 2),
            dropout=model_cfg.get("dropout", 0.1),
        )
    raise ValueError(f"Unknown model name: {name}")
