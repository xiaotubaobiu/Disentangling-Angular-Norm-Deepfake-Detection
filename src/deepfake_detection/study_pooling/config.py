from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import yaml

@dataclass
class PoolingCfg:
    raw: dict
    global_seed: int = 42
    backbones: list = field(default_factory=list)
    seeds: list = field(default_factory=list)
    methods: list = field(default_factory=list)
    domains: list = field(default_factory=list)
    bootstrap_iterations: int = 2000
    clip_values: list = field(default_factory=list)
    gamma_values: list = field(default_factory=list)
    topk_fraction: float = 0.25
    trim_fraction: float = 0.10
    sat_thresholds: list = field(default_factory=lambda: [0.01, 0.05, 0.10])
    val_videos_per_method: int = 100
    frames_per_video: int = 8
    gpus: list = field(default_factory=lambda: [0,1,2,3,4,5])
    smoke: dict = field(default_factory=dict)
    output_dir: str = "outputs"
    paths: dict = field(default_factory=dict)

    @property
    def parquet_frame(self) -> str:
        return self.raw.get("parquet_frame")
    @property
    def ckpt_root(self) -> str:
        return self.raw.get("ckpt_root", "experiments/final_results/checkpoints")

def load_config(path) -> PoolingCfg:
    with open(Path(path)) as f:
        raw = yaml.safe_load(f)
    return PoolingCfg(
        raw=raw,
        global_seed=raw.get("global_seed", 42),
        backbones=raw.get("backbones", []),
        seeds=raw.get("seeds", []),
        methods=raw.get("methods", []),
        domains=raw.get("domains", []),
        bootstrap_iterations=raw.get("bootstrap_iterations", 2000),
        clip_values=raw.get("clip_values", []),
        gamma_values=raw.get("gamma_values", []),
        topk_fraction=raw.get("topk_fraction", 0.25),
        trim_fraction=raw.get("trim_fraction", 0.10),
        sat_thresholds=raw.get("sat_thresholds", [0.01,0.05,0.10]),
        val_videos_per_method=raw.get("val_videos_per_method", 100),
        frames_per_video=raw.get("frames_per_video", 8),
        gpus=raw.get("gpus", [0,1,2,3,4,5]),
        smoke=raw.get("smoke", {}),
        output_dir=raw.get("output_dir", "outputs"),
        paths=raw.get("paths", {}),
    )
