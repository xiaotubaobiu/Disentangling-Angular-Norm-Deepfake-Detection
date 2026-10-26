from deepfake_detection.study_pooling.config import load_config
def test_load(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("global_seed: 7\nbackbones: [clip]\nclip_values: [2.0]\npaths: {tables: t}\n")
    cfg = load_config(p)
    assert cfg.global_seed == 7
    assert cfg.backbones == ["clip"]
    assert cfg.clip_values == [2.0]
    assert cfg.paths["tables"] == "t"
