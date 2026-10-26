from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class EfficientNetBinaryClassifier(nn.Module):
    def __init__(self, normalize_features: bool = False):
        super().__init__()
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.feature_dim = backbone.classifier[1].in_features  # 1280
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Linear(self.feature_dim, 2)
        self.normalize_features = normalize_features

    def extract_features(self, images):
        return self.backbone(images)

    def forward(self, images):
        features = self.extract_features(images)
        if self.normalize_features:
            features = F.normalize(features, dim=-1)
        return self.classifier(features)
