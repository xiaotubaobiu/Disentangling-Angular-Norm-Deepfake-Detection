"""Load the pinned upstream classes without unrelated LAVIS plugin imports."""
from __future__ import annotations

from functools import partial
import importlib
import json
from pathlib import Path
import sys
import types

import torch
from torch import nn


def import_upstream(root: Path):
    root = Path(root).resolve()
    # LAVIS __init__ imports every dataset, diffusion pipeline, and task. Only
    # establish its package namespace; all model source files remain upstream.
    for name in ('lavis', 'lavis.common', 'lavis.models', 'lavis.models.blip2_models'):
        path = root.joinpath(*name.split('.'))
        if name in sys.modules:
            if str(path) not in getattr(sys.modules[name], '__path__', []):
                raise RuntimeError(f'conflicting upstream namespace: {name}')
            continue
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        module.__package__ = name
        sys.modules[name] = module
    registry = importlib.import_module('lavis.common.registry').registry
    for key, path in {'library_root': root / 'lavis', 'repo_root': root, 'cache_root': root / 'cache'}.items():
        if key not in registry.mapping['paths']:
            registry.register_path(key, str(path))
    sys.modules['lavis.models'].BaseModel = importlib.import_module('lavis.models.base_model').BaseModel
    return importlib.import_module('lavis.models.blip2_models.blip2_vicuna_instruct_textinv')


def load_backbone(phase: Path, *, device='cuda:0'):
    phase = Path(phase).resolve()
    assets = phase / 'results/assets'
    module = import_upstream(phase / 'upstream')
    from lavis.models.blip2_models.Qformer import BertConfig, BertLMHeadModelTextualInversion
    from lavis.models.blip2_models.blip2 import LayerNorm
    from lavis.models.eva_vit import VisionTransformer, interpolate_pos_embed, convert_weights_to_fp16
    from transformers import BertTokenizer

    class LocalAntifake(module.Blip2VicunaInstructTextinv):
        @classmethod
        def init_tokenizer(cls, truncation_side='left'):
            tok = BertTokenizer.from_pretrained(str(assets / 'bert-base-uncased'), truncation_side=truncation_side)
            tok.add_special_tokens({'bos_token': '[DEC]'})
            return tok

        @classmethod
        def init_Qformer(cls, num_query_token, vision_width, cross_attention_freq=2):
            config = BertConfig.from_pretrained(str(assets / 'bert-base-uncased'))
            config.encoder_width = vision_width
            config.add_cross_attention = True
            config.cross_attention_freq = cross_attention_freq
            config.query_length = num_query_token
            # Every pretrained Q-former tensor is loaded from InstructBLIP
            # below, checked for missing keys; a redundant BERT download is unnecessary.
            model = BertLMHeadModelTextualInversion(config)
            query = nn.Parameter(torch.zeros(1, num_query_token, config.hidden_size))
            return model, query

        def init_vision_encoder(self, model_name, img_size, drop_path_rate, use_grad_checkpoint, precision):
            if model_name != 'eva_clip_g':
                raise ValueError(model_name)
            model = VisionTransformer(
                img_size=img_size, patch_size=14, use_mean_pooling=False,
                embed_dim=1408, depth=39, num_heads=16, mlp_ratio=4.3637,
                qkv_bias=True, drop_path_rate=drop_path_rate,
                norm_layer=partial(nn.LayerNorm, eps=1e-6), use_checkpoint=use_grad_checkpoint)
            state = torch.load(assets / 'eva_vit_g.pth', map_location='cpu', weights_only=True, mmap=True)
            interpolate_pos_embed(model, state)
            incompat = model.load_state_dict(state, strict=False)
            # Upstream deliberately constructs 39 blocks from a 40-block EVA
            # checkpoint and omits its classifier/final norm. Match that exact
            # feature extractor while rejecting any other missing/extra tensor.
            allowed_extra = {'norm.weight', 'norm.bias', 'head.weight', 'head.bias'}
            bad_extra = [k for k in incompat.unexpected_keys
                         if k not in allowed_extra and not k.startswith('blocks.39.')]
            if incompat.missing_keys or bad_extra:
                raise ValueError(f'EVA incomplete load: {incompat}')
            if precision == 'fp16':
                convert_weights_to_fp16(model)
            self.vit_name = model_name
            return model, LayerNorm(model.num_features)

    model = LocalAntifake(llm_model=str(assets / 'vicuna-7b-v1.3'),
                         prompt='Is this photo real [*]?', pseudo_word='[*]', init_word=None)
    model._add_new_token_and_embedding('[*]', None)
    pretrained = torch.load(assets / 'instruct_blip_vicuna7b_trimmed.pth', map_location='cpu', weights_only=True, mmap=True)
    state = pretrained['model']
    missing = model.load_state_dict(state, strict=False)
    # The trimmed file intentionally excludes Vicuna and EVA, loaded separately.
    bad_missing = [k for k in missing.missing_keys if not
                   (k.startswith(('llm_model.', 'visual_encoder.')) or 'train_word_embeddings' in k)]
    if bad_missing or missing.unexpected_keys:
        raise ValueError(f'Incomplete InstructBLIP load: {bad_missing}, {missing.unexpected_keys}')
    # Official prompt checkpoint also carries training metadata; only its model
    # tensors are applied. It is the user-selected repository's published asset.
    checkpoint = torch.load(assets / 'COCO_150k_SD3_SD2IP.pth', map_location='cpu', weights_only=False)
    prompt_state = checkpoint['model']
    expected = {'Qformer.bert.embeddings.train_word_embeddings.weight', 'llm_model.model.train_embed_tokens.weight'}
    buffers = dict(model.named_buffers())
    if set(prompt_state) - set(buffers) != expected:
        raise ValueError(f'Unexpected official prompt keys: {set(prompt_state)}')
    model.load_state_dict(prompt_state, strict=False)
    model.load_audit = {'pretrained_missing': missing.missing_keys,
                        'official_prompt_keys': sorted(prompt_state),
                        'official_checkpoint_metadata_keys': sorted(checkpoint)}
    return model.eval().to(device)
