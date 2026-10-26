from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import yaml

@dataclass
class StudyConfig:
    raw: dict
    global_seed: int = 42
    backbones: list = field(default_factory=list)
    seeds: list = field(default_factory=list)
    domains: list = field(default_factory=list)
    paths: dict = field(default_factory=dict)
    sampling: dict = field(default_factory=dict)
    bootstrap: dict = field(default_factory=dict)
    q4: dict = field(default_factory=dict)
    smoke: dict = field(default_factory=dict)
    fake_methods: dict = field(default_factory=dict)
    gamma_grid: list = field(default_factory=list)
    identity_tol: float = 1e-4

    @property
    def parquet(self) -> dict:
        # Convenience accessor so plan scripts can write CFG.parquet["frame"];
        # the parsed YAML keeps nested blocks under `raw`.
        return self.raw.get("parquet", {})

    @property
    def ckpt_root(self) -> str:
        return self.raw.get("ckpt_root", "experiments/2026-05-05_paper_feature_geometry/results/checkpoints")

    @property
    def train_mode_primary(self) -> str:
        return self.raw.get("train_mode_primary", "raw")

def load_config(path) -> StudyConfig:
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)
    return StudyConfig(
        raw=raw,
        global_seed=raw.get("global_seed", 42),
        backbones=raw.get("backbones", []),
        seeds=raw.get("seeds", []),
        domains=raw.get("domains", []),
        paths=raw.get("paths", {}),
        sampling=raw.get("sampling", {}),
        bootstrap=raw.get("bootstrap", {}),
        q4=raw.get("q4", {}),
        smoke=raw.get("smoke", {}),
        fake_methods=raw.get("fake_methods", {}),
        gamma_grid=raw.get("gamma_grid", []),
        identity_tol=float(raw.get("identity_tol", 1e-4)),
    )
