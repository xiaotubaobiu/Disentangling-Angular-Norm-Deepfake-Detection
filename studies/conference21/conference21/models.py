"""Public pretrained visual encoders with one newly initialized linear head."""
from pathlib import Path
import torch
from torch import nn


class Detector(nn.Module):
    def __init__(self,encoder,dimension):
        super().__init__();self.encoder=encoder;self.classifier=nn.Linear(dimension,2)
    def forward(self,x):return self.classifier(self.encoder(x).float())


def official_position_cpu(self,x,w,h):
    """Run the unchanged official interpolation on CPU, retaining autograd.

    Strict deterministic torch2.6 rejects CUDA bicubic backward. Preserve
    official scale-factor offset .1 and antialias=False; no timm conversion.
    The empty tensor carries only shape/dtype (the official method reads no x data).
    """
    from types import SimpleNamespace
    from dinov2.models.vision_transformer import DinoVisionTransformer
    proxy=SimpleNamespace(pos_embed=self.pos_embed.to('cpu'),patch_size=self.patch_size,
      interpolate_offset=self.interpolate_offset,interpolate_antialias=self.interpolate_antialias)
    shape=torch.empty(x.shape,device='cpu',dtype=x.dtype)
    return DinoVisionTransformer.interpolate_pos_encoding(proxy,shape,w,h).to(x.device)


def build_model(backbone,weight_path):
    p=Path(weight_path)
    if not p.is_file():raise FileNotFoundError('Explicit offline initialization weight required')
    if backbone=='clip_vit_b16':
        import clip
        original,_=clip.load(str(p),device='cpu',jit=False)
        encoder=original.visual.float();dimension=encoder.output_dim
    elif backbone=='dinov2_vit_b14':
        import sys, os
        # Disable optional xformers so determinism does not depend on its presence.
        os.environ['XFORMERS_DISABLED']='1'
        official=Path(__file__).resolve().parents[1]/'third_party/dinov2'
        sys.path.insert(0,str(official))
        from dinov2.hub.backbones import dinov2_vitb14
        encoder=dinov2_vitb14(pretrained=False)
        encoder.load_state_dict(torch.load(p,map_location='cpu',weights_only=True),strict=True)
        dimension=encoder.embed_dim
        import types
        encoder.interpolate_pos_encoding=types.MethodType(official_position_cpu,encoder)
    elif backbone=='efficientnet_b0':
        from torchvision.models import efficientnet_b0
        encoder=efficientnet_b0(weights=None)
        encoder.load_state_dict(torch.load(p,map_location='cpu',weights_only=True),strict=True)
        dimension=encoder.classifier[1].in_features;encoder.classifier=nn.Identity()
    else:raise ValueError(backbone)
    return Detector(encoder,dimension)
