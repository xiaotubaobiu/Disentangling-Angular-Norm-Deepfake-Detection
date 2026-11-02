import json
from pathlib import Path
import numpy as np
import pytest
import torch
from deepfake_detector.models import build_model


@pytest.mark.parametrize('backbone,dimension',[('clip_vit_b16',512),('dinov2_vit_b14',768),('efficientnet_b0',1280)])
def test_public_initialization_cpu_shape_and_signed_logits(backbone,dimension):
    config=json.loads((Path(__file__).resolve().parents[1]/'configs/v1.1.json').read_text())
    path=config['backbones'][backbone]['weight_path']
    if path is None or not Path(path).is_file():pytest.skip('Public initialization has not been prepared')
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(2);torch.manual_seed(7)
    model=build_model(backbone,path).eval();x=torch.randn(2,3,224,224)
    with torch.inference_mode():h=model.encoder(x).float();z=model.classifier(h)
    assert h.shape==(2,dimension) and torch.isfinite(h).all()
    assert not torch.allclose(h.norm(dim=1),torch.ones(2)), 'raw encoder output was extra L2-normalized'
    w=model.classifier.weight.detach().numpy();b=model.classifier.bias.detach().numpy()
    signed=h.numpy().astype(float)@(w[1]-w[0]).astype(float)+float(b[1]-b[0])
    assert np.allclose(signed,(z[:,1]-z[:,0]).numpy(),atol=1e-5,rtol=1e-5)
    assert not torch.cuda.is_initialized()
