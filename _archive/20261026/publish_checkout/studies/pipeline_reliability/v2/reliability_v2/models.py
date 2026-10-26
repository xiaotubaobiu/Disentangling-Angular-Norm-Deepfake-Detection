"""Public pretrained visual encoders with one newly initialized linear head."""
from pathlib import Path
import torch
from torch import nn


class Detector(nn.Module):
    def __init__(self,encoder,dimension):
        super().__init__();self.encoder=encoder;self.classifier=nn.Linear(dimension,2)
    def forward(self,x):return self.classifier(self.encoder(x).float())


def resample_dino_positions(posemb,new_size,old_size,num_prefix_tokens):
    # CUDA AA bicubic backward is unsupported under strict deterministic algorithms
    # in pinnedtorch2.6. Preserve the same interpolation and learned37x37 parameter.
    from timm.layers import resample_abs_pos_embed
    return resample_abs_pos_embed(posemb.to('cpu'),new_size,old_size,num_prefix_tokens=num_prefix_tokens).to(posemb.device)

def dino_position_embedding(self,x):
    to_cat=[]
    if self.cls_token is not None:to_cat.append(self.cls_token.expand(x.shape[0],-1,-1))
    if self.reg_token is not None:to_cat.append(self.reg_token.expand(x.shape[0],-1,-1))
    if self.pos_embed is None:return torch.cat(to_cat+[x.view(x.shape[0],-1,x.shape[-1])],dim=1)
    if self.dynamic_img_size:
        B,H,W,C=x.shape
        pos_embed=resample_dino_positions(self.pos_embed,[H,W],self.patch_embed.grid_size,0 if self.no_embed_class else self.num_prefix_tokens)
        x=x.view(B,-1,C)
    else:pos_embed=self.pos_embed
    if self.no_embed_class:
        x=x+pos_embed
        if to_cat:x=torch.cat(to_cat+[x],dim=1)
    else:
        if to_cat:x=torch.cat(to_cat+[x],dim=1)
        x=x+pos_embed
    return self.pos_drop(x)

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
        import types
        encoder._pos_embed=types.MethodType(dino_position_embedding,encoder)
        # DINOv2 positional embeddings were learned on518px; preserve the original
        # grid and interpolate at224 input instead of pretending weights are224.
    elif backbone=='efficientnet_b0':
        from torchvision.models import efficientnet_b0
        encoder=efficientnet_b0(weights=None)
        encoder.load_state_dict(torch.load(p,map_location='cpu',weights_only=True),strict=True)
        dimension=encoder.classifier[1].in_features;encoder.classifier=nn.Identity()
    else:raise ValueError(backbone)
    return Detector(encoder,dimension)
