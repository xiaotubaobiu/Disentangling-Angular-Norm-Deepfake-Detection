from study.config import load_config
from pathlib import Path

def test_load_config_roundtrip(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("global_seed: 7\nbackbones: [clip]\npaths: {tables: t}\n")
    cfg = load_config(p)
    assert cfg.global_seed == 7
    assert cfg.backbones == ["clip"]
    assert cfg.paths["tables"] == "t"
