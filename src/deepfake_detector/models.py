"""Public pretrained visual encoders with one newly initialized linear head."""
from pathlib import Path
import torch
from torch import nn


class Detector(nn.Module):
    def __init__(self,encoder,dimension):
        super().__init__();self.encoder=encoder;self.classifier=nn.Linear(dimension,2)
    def forward(self,x):return self.classifier(self.encoder(x).float())


def build_model(backbone,weight_path):
    p=Path(weight_path)
    if not p.is_file():raise FileNotFoundError('Explicit offline initialization weight required')
    if backbone=='clip_vit_b16':
        import clip
        original,_=clip.load(str(p),device='cpu',jit=False)
        encoder=original.visual.float();dimension=encoder.output_dim
    elif backbone=='dinov2_vit_b14':
        import timm
        from timm.models.vision_transformer import checkpoint_filter_fn
        encoder=timm.create_model('vit_base_patch14_dinov2',pretrained=False,num_classes=0,dynamic_img_size=True)
        state=torch.load(p,map_location='cpu',weights_only=True)
        encoder.load_state_dict(checkpoint_filter_fn(state,encoder),strict=True);dimension=encoder.num_features
        # DINOv2 positional embeddings were learned on518px; preserve the original
        # grid and interpolate at224 input instead of pretending weights are224.
    elif backbone=='efficientnet_b0':
        from torchvision.models import efficientnet_b0
        encoder=efficientnet_b0(weights=None)
        encoder.load_state_dict(torch.load(p,map_location='cpu',weights_only=True),strict=True)
        dimension=encoder.classifier[1].in_features;encoder.classifier=nn.Identity()
    else:raise ValueError(backbone)
    return Detector(encoder,dimension)
