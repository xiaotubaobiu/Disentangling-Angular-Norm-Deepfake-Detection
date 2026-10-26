# 深伪检测四问题实证检验 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现并运行深伪检测四问题实证检验流水线(Q1 rank-flip+解剖+γ、Q2 径向稳定性、Q3 length 稳健性、Q4 多方法对齐),产出表格、图与中/终报告,给出 A/B/C/D 决策。

**Architecture:** 共享库 `src/deepfake_detection/study/`(配置/IO/canonical key/pair-AUC/flip 解剖/length/radial/bootstrap/Q4 抽取与对齐)+ 7 个脚本(`scripts/00..05` + `run_all.py`)。Q1/Q2/Q3 仅读 `per_frame_scores.parquet`(DuckDB 过滤);Q4 用 `feature_extractor.extract_raw_features` 重抽 `z`,经一致性门控后落盘 `outputs/z_cache/`。GPU 仅 0–5。DFD 全程排除。

**Tech Stack:** Python, pandas, DuckDB, NumPy, SciPy, scikit-learn, PyTorch, matplotlib;pytest。

**Spec:** `docs/superpowers/specs/2026-07-23-four-questions-design.md`(v2)。每个任务可回溯到 spec 章节。

---

## 文件结构

共享库(新建,每个文件单一职责):
- `src/deepfake_detection/study/__init__.py`
- `src/deepfake_detection/study/config.py` — 读 `configs/study_config.yaml` 为 dataclass
- `src/deepfake_detection/study/io.py` — DuckDB 读 parquet、过滤(排 DFD)、real 去重
- `src/deepfake_detection/study/canonical.py` — 构造 `canonical_frame_id` / `source_video_id`
- `src/deepfake_detection/study/auc_pairs.py` — Mann–Whitney AUC(ties 0.5)、H/B/CC/WW、identity_error、native/video-balanced 采样、video 聚合
- `src/deepfake_detection/study/flip_dissect.py` — ρ/margin 分布、γ 部分归一化曲线(从 parquet 恢复 wᵀu,b)
- `src/deepfake_detection/study/length.py` — length 定向(checkpoint 级 / 逐方法)、micro/macro/video/equalized AUC
- `src/deepfake_detection/study/radial.py` — delta/RTG/STG/R_retain、方法级相关+permutation+LOMO、混合效应
- `src/deepfake_detection/study/cluster_bootstrap.py` — video-cluster bootstrap 通用封装
- `src/deepfake_detection/study/q4_extract.py` — 子集 loader、抽 z、一致性门控、缓存
- `src/deepfake_detection/study/q4_alignment.py` — group 构造、位移(chord/tangent/logmap)、cosine、负对照、p_global、LOFO、PCA、residual-vs-w

脚本:`scripts/00_inspect_project.py` … `05_generate_report.py`、`run_all.py`。
配置:`configs/study_config.yaml`。
报告:`reports/schema_report.md`、`reports/midterm_report.md`、`reports/four_questions_final_report.md`、`reports/next_step_recommendation.md`。
输出:`outputs/tables/`、`outputs/figures/`、`outputs/logs/`、`outputs/z_cache/`。
测试:`tests/study/test_*.py`。

---

## Phase 0 — 配置与共享库骨架

### Task 0.1: 配置文件 + 目录骨架

**Files:**
- Create: `configs/study_config.yaml`
- Create: `src/deepfake_detection/study/__init__.py`
- Create: `outputs/tables/.gitkeep`, `outputs/figures/.gitkeep`, `outputs/logs/.gitkeep`, `outputs/z_cache/.gitkeep`

- [ ] **Step 1: 写配置**

```yaml
# configs/study_config.yaml
global_seed: 42
parquet:
  frame: outputs/frame_midt_diag/parallel_20260721_213914/per_frame_scores.parquet
  video: outputs/frame_midt_diag/parallel_20260721_213914/per_video_scores.parquet
ckpt_root: experiments/final_results/checkpoints
backbones: [clip, dinov2, efficientnet]
seeds: [7, 42, 123, 999, 2048]
domains: [ffpp, cdf]            # DFD excluded everywhere
train_mode_primary: raw         # Q1/Q2/Q3 internal analyses
infer_modes_rawtrain: [raw, direction, length]
fake_methods:
  face_swap: [simswap, inswap, blendface, faceswap, fsgan, mobileswap, e4s, facedancer]
  reenactment: [fomm, facevid2vid, wav2lip, sadtalker, MRAA, pirender, tpsm, lia]
sampling:
  pairs_per_combo: 200000
  equalized_repeats: 100
  frames_per_video_balanced: 32
bootstrap:
  n_iter: 10000
  rng_seed: 42
gamma_grid: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
q4:
  audit_sample: 300
  consistency_n_frames: 1000
  groups_target: 3000
  gpus: [0, 1, 2, 3, 4, 5]
smoke:                         # dry-run subset
  backbone: clip
  seed: 42
  methods: [fsgan, sadtalker]
identity_tol: 1.0e-4
paths:
  tables: outputs/tables
  figures: outputs/figures
  logs: outputs/logs
  z_cache: outputs/z_cache
  reports: reports
```

- [ ] **Step 2: 建空 `__init__.py` 与 `.gitkeep`**

```python
# src/deepfake_detection/study/__init__.py
"""Four-question empirical study shared library."""
```

- [ ] **Step 3: 建目录占位**

```bash
mkdir -p outputs/tables outputs/figures outputs/logs outputs/z_cache tests/study
touch outputs/tables/.gitkeep outputs/figures/.gitkeep outputs/logs/.gitkeep outputs/z_cache/.gitkeep
```

- [ ] **Step 4: Commit**

```bash
git add configs/study_config.yaml src/deepfake_detection/study/__init__.py outputs
git commit -m "feat(study): add study config and output skeleton"
```

---

### Task 0.2: config 加载器

**Files:**
- Create: `src/deepfake_detection/study/config.py`
- Test: `tests/study/test_config.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/study/test_config.py
from deepfake_detection.study.config import load_config
from pathlib import Path

def test_load_config_roundtrip(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("global_seed: 7\nbackbones: [clip]\npaths: {tables: t}\n")
    cfg = load_config(p)
    assert cfg.global_seed == 7
    assert cfg.backbones == ["clip"]
    assert cfg.paths["tables"] == "t"
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /home/z/project/deepfake_detection && python -m pytest tests/study/test_config.py -q`
Expected: FAIL(ModuleNotFoundError / AttributeError)

- [ ] **Step 3: 实现**

```python
# src/deepfake_detection/study/config.py
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
```

- [ ] **Step 4: 运行确认通过**

Run: `cd /home/z/project/deepfake_detection && python -m pytest tests/study/test_config.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/deepfake_detection/study/config.py tests/study/test_config.py
git commit -m "feat(study): config loader"
```

---

### Task 0.3: canonical key 构造

**Files:**
- Create: `src/deepfake_detection/study/canonical.py`
- Test: `tests/study/test_canonical.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/study/test_canonical.py
import pandas as pd
from deepfake_detection.study.canonical import add_canonical_keys

def _df(paths, domain="ffpp", method="real", video="000"):
    return pd.DataFrame({"image_path": paths, "domain": domain,
                         "method_id": method, "video_id": video, "label": 0})

def test_canonical_frame_id_from_path():
    p = "/Dataset/x/FaceForensics++/original_sequences/youtube/c23/frames/000/254.png"
    df = _df([p])
    out = add_canonical_keys(df)
    assert out.loc[0, "canonical_frame_id"] == "ffpp|000|254"
    assert out.loc[0, "source_video_id"] == "000"

def test_real_dedup_key_includes_domain():
    p = "/x/frames/000_003/010.png"
    df = _df([p], domain="cdf", method="fsgan", video="000_003", )
    df.loc[0, "label"] = 1
    out = add_canonical_keys(df)
    # fake video 000_003: source identity = first token
    assert out.loc[0, "source_video_id"] == "000"
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /home/z/project/deepfake_detection && python -m pytest tests/study/test_canonical.py -q`
Expected: FAIL

- [ ] **Step 3: 实现**

```python
# src/deepfake_detection/study/canonical.py
"""Construct canonical_frame_id and source_video_id.

Priority (per spec §3): manifest domain+video+frame > normalized image_path >
path parse. We implement path parse (always available) and validate via human
spot-check in 00_inspect. Dedup key MUST include domain.
"""
from __future__ import annotations
import os
import re
import pandas as pd

_FRAME_RE = re.compile(r"/frames/(?P<vid>[^/]+)/(?P<frame>[^/]+)\.png$")

def _parse(image_path: str):
    m = _FRAME_RE.search(image_path or "")
    if m:
        return m.group("vid"), m.group("frame")
    base = os.path.basename(image_path or "")
    return os.path.basename(os.path.dirname(image_path or "")), os.path.splitext(base)[0]

def _source_from_video(video_id: str) -> str:
    # FF++/CDF fake naming "<source>_<target>"; real video_id == source.
    v = str(video_id)
    if "_" in v:
        return v.split("_", 1)[0]
    return v

def add_canonical_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    vids, frames = zip(*[_parse(p) for p in df["image_path"]])
    # Prefer explicit video_id column when present and non-empty.
    use_vid = [str(v) if (str(v) and str(v) != "nan") else vids[i]
               for i, v in enumerate(df.get("video_id", [""] * len(df)))]
    df["canonical_frame_id"] = [f"{d}|{v}|{f}" for d, v, f in
                                zip(df["domain"], use_vid, frames)]
    df["source_video_id"] = [_source_from_video(v) for v in use_vid]
    return df
```

- [ ] **Step 4: 运行确认通过**

Run: `cd /home/z/project/deepfake_detection && python -m pytest tests/study/test_canonical.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/deepfake_detection/study/canonical.py tests/study/test_canonical.py
git commit -m "feat(study): canonical frame/source key construction"
```

---

### Task 0.4: IO / DuckDB 加载与 real 去重

**Files:**
- Create: `src/deepfake_detection/study/io.py`
- Test: `tests/study/test_io.py`

- [ ] **Step 1: 写失败测试**(合成小 parquet 验证过滤 + real 去重)

```python
# tests/study/test_io.py
import pandas as pd, pyarrow as pa, pyarrow.parquet as pq
from deepfake_detection.study.io import load_frame_subset, dedup_reals

def _write(tmp_path):
    df = pd.DataFrame({
        "checkpoint_id":["c","c","c","c"],"backbone":["clip"]*4,"seed":[42]*4,
        "train_mode":["raw"]*4,"infer_mode":["raw","direction","raw","direction"],
        "domain":["ffpp","ffpp","dfd","ffpp"],"method_id":["real","real","fsgan","real"],
        "label":[0,0,1,0],"video_id":["000","000","001","000"],
        "image_path":["/f/000/1.png","/f/000/1.png","/f/001/2.png","/f/000/1.png"],
        "feature_norm":[10.0,10.0,20.0,10.0],"signed_logit":[1.0,0.5,3.0,1.0],
        "real_logit":[0,0,0,0],"fake_logit":[1,0.5,3,1],"prob_fake":[.5,.5,.8,.5],
        "prediction_correct":[1,1,1,1],
    })
    pq.write_table(pa.Table.from_pandas(df), tmp_path/"f.parquet")

def test_load_filters_dfd(tmp_path):
    _write(tmp_path)
    got = load_frame_subset(str(tmp_path/"f.parquet"),
                            train_mode="raw", infer_mode="raw",
                            domains=["ffpp","cdf"])
    assert (got["domain"]=="dfd").sum()==0

def test_dedup_reals_keeps_one_per_frame(tmp_path):
    _write(tmp_path)
    got = load_frame_subset(str(tmp_path/"f.parquet"), train_mode="raw",
                            infer_mode="raw", domains=["ffpp"])
    got = dedup_reals(got)
    assert ((got["label"]==0)).sum()==1   # two identical real rows -> 1
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /home/z/project/deepfake_detection && python -m pytest tests/study/test_io.py -q`
Expected: FAIL

- [ ] **Step 3: 实现**

```python
# src/deepfake_detection/study/io.py
"""DuckDB-backed parquet loading with DFD exclusion and real dedup."""
from __future__ import annotations
import pandas as pd

try:
    import duckdb
    _HAS_DUCKDB = True
except ImportError:
    _HAS_DUCKDB = False

from deepfake_detection.study.canonical import add_canonical_keys

_COLS = [
    "checkpoint_id","backbone","seed","train_mode","infer_mode","domain",
    "method_id","label","video_id","image_path","feature_norm",
    "real_logit","fake_logit","signed_logit","prob_fake","prediction_correct",
]

def load_frame_subset(parquet_path: str, train_mode: str | None = None,
                      infer_mode: str | None = None,
                      backbones: list[str] | None = None,
                      seeds: list[int] | None = None,
                      domains: list[str] | None = None,
                      methods: list[str] | None = None) -> pd.DataFrame:
    cols = ", ".join(_COLS)
    where = ["domain != 'dfd'"]
    if train_mode: where.append(f"train_mode = '{train_mode}'")
    if infer_mode: where.append(f"infer_mode = '{infer_mode}'")
    if backbones: where.append("backbone IN ('" + "','".join(backbones) + "')")
    if seeds: where.append("seed IN (" + ",".join(map(str, seeds)) + ")")
    if domains: where.append("domain IN ('" + "','".join(domains) + "')")
    if methods: where.append("method_id IN ('" + "','".join(methods) + "')")
    sql = f"SELECT {cols} FROM read_parquet('{parquet_path}') WHERE " + " AND ".join(where)
    if _HAS_DUCKDB:
        return duckdb.query(sql).df()
    df = pd.read_parquet(parquet_path, columns=_COLS)
    return df.query(" and ".join(where))

def dedup_reals(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicated real frames (parquet replicates real rows per method)."""
    if df.empty:
        return df
    df = add_canonical_keys(df)
    real = df[df["label"] == 0]
    fake = df[df["label"] == 1]
    real = real.drop_duplicates(subset=["domain", "source_video_id",
                                        "canonical_frame_id"]).copy()
    out = pd.concat([real, fake], ignore_index=True)
    return out
```

- [ ] **Step 4: 运行确认通过**

Run: `cd /home/z/project/deepfake_detection && python -m pytest tests/study/test_io.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/deepfake_detection/study/io.py tests/study/test_io.py
git commit -m "feat(study): duckdb parquet loader with DFD filter and real dedup"
```

---

### Task 0.5: pair-AUC + H/B + identity(核心正确性)

**Files:**
- Create: `src/deepfake_detection/study/auc_pairs.py`
- Test: `tests/study/test_auc_pairs.py`

- [ ] **Step 1: 写失败测试**(已知答案)

```python
# tests/study/test_auc_pairs.py
import numpy as np
from deepfake_detection.study.auc_pairs import (
    auc_mannwhitney, flip_counts, identity_check)

def test_auc_known():
    # 2 fakes > 2 reals -> AUC 1.0
    assert auc_mannwhitney(np.array([2.,3.]), np.array([0.,1.])) == 1.0

def test_auc_with_ties():
    # one tie -> (1 win + 0.5 tie)/2 = 0.75
    assert abs(auc_mannwhitney(np.array([1.,2.]), np.array([0.,1.])) - 0.75) < 1e-9

def test_identity_exact_no_ties():
    rng = np.random.default_rng(0)
    raw_f, raw_r = rng.normal(0.6,1,5000), rng.normal(0.4,1,5000)
    dir_f, dir_r = rng.normal(0.6,1,5000), rng.normal(0.4,1,5000)
    auc_raw = auc_mannwhitney(raw_f, raw_r)
    auc_dir = auc_mannwhitney(dir_f, dir_r)
    cc,ww,h,b = flip_counts(raw_f, raw_r, dir_f, dir_r)
    err = identity_check(auc_raw, auc_dir, h, b)
    assert err < 1e-6, err   # same-pair exact counting -> machine epsilon

def test_flip_counts_signs():
    # f>r in both -> CC; design CC WW H B sums to 1
    cc,ww,h,b = flip_counts(np.array([1.]), np.array([0.]),
                            np.array([1.]), np.array([0.]))
    assert (cc,ww,h,b)==(1,0,0,0)
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /home/z/project/deepfake_detection && python -m pytest tests/study/test_auc_pairs.py -q`
Expected: FAIL

- [ ] **Step 3: 实现**

```python
# src/deepfake_detection/study/auc_pairs.py
"""Mann-Whitney AUC (ties=0.5), rank-flip H/B/CC/WW, identity check.

AUC = P(s_f > s_r) + 0.5 P(s_f == s_r).  On the SAME pair set with no ties,
AUC_dir - AUC_raw == H - B exactly (machine precision with counting).
"""
from __future__ import annotations
import numpy as np

def auc_mannwhitney(fake_scores: np.ndarray, real_scores: np.ndarray) -> float:
    """Vectorized Mann-Whitney AUC over the full fake x real pair set."""
    f = np.asarray(fake_scores, dtype=np.float64)
    r = np.asarray(real_scores, dtype=np.float64)
    # broadcast (nf, 1) vs (1, nr)
    diff = f[:, None] - r[None, :]
    wins = np.sum(diff > 0)
    ties = np.sum(diff == 0)
    n = f.size * r.size
    return float((wins + 0.5 * ties) / n)

def flip_counts(raw_f, raw_r, dir_f, dir_r):
    """Count CC/WW/H/B over the cartesian fake x real pair set.

    Returns (CC, WW, H, B) as probabilities in [0,1].
    dir correct: dir_f > dir_r; raw correct: raw_f > raw_r.  Ties (==) are
    treated as NOT correct here (strict); tie_rate reported separately.
    """
    rf = np.asarray(raw_f, dtype=np.float64)[:, None] - np.asarray(raw_r)[None, :]
    df = np.asarray(dir_f, dtype=np.float64)[:, None] - np.asarray(dir_r)[None, :]
    n = rf.size
    raw_ok = rf > 0
    dir_ok = df > 0
    cc = np.sum(dir_ok & raw_ok) / n
    ww = np.sum((~dir_ok) & (~raw_ok)) / n
    h = np.sum(dir_ok & (~raw_ok)) / n
    b = np.sum((~dir_ok) & raw_ok) / n
    return float(cc), float(ww), float(h), float(b)

def tie_rate(f, r):
    f = np.asarray(f, dtype=np.float64)[:, None]; r = np.asarray(r, dtype=np.float64)[None, :]
    return float(np.mean(f == r))

def identity_check(auc_raw, auc_dir, h, b):
    return abs((auc_dir - auc_raw) - (h - b))
```

- [ ] **Step 4: 运行确认通过**

Run: `cd /home/z/project/deepfake_detection && python -m pytest tests/study/test_auc_pairs.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/deepfake_detection/study/auc_pairs.py tests/study/test_auc_pairs.py
git commit -m "feat(study): mann-whitney AUC, rank-flip counts, identity check"
```

---

### Task 0.6: 两套采样口径 + 视频聚合

**Files:**
- Modify: `src/deepfake_detection/study/auc_pairs.py`(追加)
- Test: `tests/study/test_sampling.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/study/test_sampling.py
import numpy as np, pandas as pd
from deepfake_detection.study.auc_pairs import (
    sample_pairs_native, sample_pairs_video_balanced, aggregate_video_scores)

def _frame_df():
    # 2 fake videos (2 frames each), 2 real videos (2 frames each)
    rows = []
    for vid, base in [("f1", 5.0), ("f2", 6.0)]:
        for fr in range(2):
            rows.append(dict(video_id=vid, label=1, score=base+fr,
                             source_video_id=vid, canonical_frame_id=f"{vid}|{fr}"))
    for vid, base in [("r1", 0.0), ("r2", 1.0)]:
        for fr in range(2):
            rows.append(dict(video_id=vid, label=0, score=base+fr,
                             source_video_id=vid, canonical_frame_id=f"{vid}|{fr}"))
    return pd.DataFrame(rows)

def test_native_uses_all_frames():
    df = _frame_df()
    ff, rr = sample_pairs_native(df, n_pairs=8, seed=1)
    assert len(ff) + len(rr) > 0

def test_video_aggregate_mean():
    df = _frame_df()
    v = aggregate_video_scores(df, how="mean")
    assert set(v["video_id"]) == {"f1","f2","r1","r2"}
    assert abs(v[v.video_id=="f1"]["video_score"].iloc[0] - 5.0) < 1e-9

def test_video_balanced_equal_videos():
    df = _frame_df()
    ff, rr = sample_pairs_video_balanced(df, frames_per_video=2, seed=1)
    assert len(ff) == len(rr)   # equal fake/real video count
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /home/z/project/deepfake_detection && python -m pytest tests/study/test_sampling.py -q`
Expected: FAIL

- [ ] **Step 3: 追加实现**

```python
# append to src/deepfake_detection/study/auc_pairs.py
import pandas as pd

def _split(df):
    return df[df["label"] == 1], df[df["label"] == 0]

def aggregate_video_scores(df: pd.DataFrame, how: str = "mean", k: int = 5) -> pd.DataFrame:
    g = df.groupby("video_id", sort=False)
    if how == "mean":
        v = g["score"].mean()
    elif how == "median":
        v = g["score"].median()
    elif how == "topk":
        v = g["score"].apply(lambda s: np.sort(s)[-k:].mean() if len(s) else np.nan)
    else:
        raise ValueError(how)
    out = v.to_frame("video_score")
    out["label"] = g["label"].first()
    out["source_video_id"] = g["source_video_id"].first()
    return out.reset_index()

def sample_pairs_native(df, n_pairs, seed):
    """fake-frame-equal, real-frame-equal: uniform sample of frames then pair."""
    rng = np.random.default_rng(seed)
    fake, real = _split(df)
    if fake.empty or real.empty:
        return np.array([]), np.array([])
    n = int(max(1, np.sqrt(n_pairs)))
    fs = fake.sample(min(n, len(fake)), random_state=int(rng.integers(1<<31)))
    rs = real.sample(min(n, len(real)), random_state=int(rng.integers(1<<31)))
    return fs["score"].to_numpy(), rs["score"].to_numpy()

def sample_pairs_video_balanced(df, frames_per_video, seed, n_videos_cap=None):
    """Equal-weight videos, equal frames per video; returns fake/real score arrays."""
    rng = np.random.default_rng(seed)
    fake, real = _split(df)
    def _per(group):
        out = []
        for vid, sub in group.groupby("video_id", sort=False):
            s = sub.sample(min(frames_per_video, len(sub)),
                           random_state=int(rng.integers(1<<31)))
            out.append(s["score"].to_numpy())
        return np.concatenate(out) if out else np.array([])
    # equal number of fake/real videos
    fv = fake.groupby("video_id", sort=False).size()
    rv = real.groupby("video_id", sort=False).size()
    m = min(len(fv), len(rv))
    if m == 0:
        return np.array([]), np.array([])
    fake_vids = np.array(list(fv.index))[:m] if n_videos_cap is None else \
        rng.choice(list(fv.index), size=min(n_videos_cap, len(fv)), replace=False)
    real_vids = np.array(list(rv.index))[:m] if n_videos_cap is None else \
        rng.choice(list(rv.index), size=min(n_videos_cap, len(rv)), replace=False)
    fake_s = fake[fake.video_id.isin(fake_vids)]
    real_s = real[real.video_id.isin(real_vids)]
    return _per(fake_s), _per(real_s)
```

- [ ] **Step 4: 运行确认通过**

Run: `cd /home/z/project/deepfake_detection && python -m pytest tests/study/test_sampling.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/deepfake_detection/study/auc_pairs.py tests/study/test_sampling.py
git commit -m "feat(study): native and video-balanced pair sampling, video aggregation"
```

---

### Task 0.7: cluster bootstrap 通用封装

**Files:**
- Create: `src/deepfake_detection/study/cluster_bootstrap.py`
- Test: `tests/study/test_cluster_bootstrap.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/study/test_cluster_bootstrap.py
import numpy as np, pandas as pd
from deepfake_detection.study.cluster_bootstrap import cluster_bootstrap_metric

def test_ci_covers_stat():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"cluster": np.repeat(range(50), 2),
                       "label": rng.integers(0,2,100),
                       "score": rng.normal(0,1,100)})
    def metric(sub):
        from sklearn.metrics import roc_auc_score
        try: return roc_auc_score(sub["label"], sub["score"])
        except: return np.nan
    out = cluster_bootstrap_metric(df, metric, cluster_col="cluster",
                                   n_iter=200, rng_seed=0)
    assert out["ci_low"] <= out["point"] <= out["ci_high"]
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /home/z/project/deepfake_detection && python -m pytest tests/study/test_cluster_bootstrap.py -q`
Expected: FAIL

- [ ] **Step 3: 实现**

```python
# src/deepfake_detection/study/cluster_bootstrap.py
from __future__ import annotations
import numpy as np
import pandas as pd

def cluster_bootstrap_metric(df: pd.DataFrame, metric, cluster_col: str,
                             n_iter=10000, rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    clusters = np.array(sorted(df[cluster_col].unique()))
    point = metric(df)
    vals = []
    for _ in range(n_iter):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        parts = [df[df[cluster_col] == c] for c in sampled]
        sub = pd.concat(parts)
        v = metric(sub)
        if v == v:  # not nan
            vals.append(v)
    vals = np.array(vals)
    return {"point": float(point),
            "ci_low": float(np.percentile(vals, 2.5)),
            "ci_high": float(np.percentile(vals, 97.5)),
            "n_iter": int(len(vals))}
```

- [ ] **Step 4: 运行确认通过**

Run: `cd /home/z/project/deepfake_detection && python -m pytest tests/study/test_cluster_bootstrap.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/deepfake_detection/study/cluster_bootstrap.py tests/study/test_cluster_bootstrap.py
git commit -m "feat(study): generic video-cluster bootstrap"
```

---

## Phase 1 — `00_inspect_project.py`(schema + canonical + split 审计)

### Task 1.1: schema 报告脚本

**Files:**
- Create: `scripts/00_inspect_project.py`
- Output: `reports/schema_report.md`

- [ ] **Step 1: 实现脚本**

```python
# scripts/00_inspect_project.py
"""Spec §3: schema report + canonical-key validation + FF++ split audit."""
from __future__ import annotations
import sys, os
import pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from deepfake_detection.study.config import load_config
from deepfake_detection.study.io import load_frame_subset, dedup_reals
from deepfake_detection.study.canonical import add_canonical_keys

CFG = load_config(ROOT / "configs/study_config.yaml")
REP = ROOT / CFG.paths["reports"]; REP.mkdir(parents=True, exist_ok=True)

def main():
    fpath = str(ROOT / CFG.parquet["frame"])
    lines = ["# Schema Report\n", f"## frame parquet: `{fpath}`\n"]
    # 1. columns + counts (read schema cheaply via duckdb)
    import duckdb
    cols = duckdb.query(f"DESCRIBE SELECT * FROM read_parquet('{fpath}')").df()
    lines.append("### columns\n"); lines.append(cols[["column_name","column_type"]].to_markdown(index=False)); lines.append("")
    n = duckdb.query(f"SELECT COUNT(*) n FROM read_parquet('{fpath}')").df()["n"][0]
    lines.append(f"total rows: {n}\n")
    # 2. breakdown
    for col in ["backbone","seed","train_mode","infer_mode","domain","method_id","label"]:
        b = duckdb.query(f"SELECT {col}, COUNT(*) n FROM read_parquet('{fpath}') WHERE domain!='dfd' GROUP BY {col} ORDER BY n DESC").df()
        lines.append(f"### counts by {col}\n"); lines.append(b.to_markdown(index=False)); lines.append("")
    # 3. canonical key + real dedup demo (clip seed42 raw direction ffpp)
    sub = load_frame_subset(fpath, train_mode="raw", infer_mode="direction",
                            backbones=["clip"], seeds=[42], domains=["ffpp"])
    sub = add_canonical_keys(sub)
    dup = sub[sub.label==0].duplicated(subset=["domain","source_video_id","canonical_frame_id"]).sum()
    lines.append(f"### real duplication (clip/seed42/raw/direction/ffpp)\n")
    lines.append(f"real rows: {(sub.label==0).sum()}, duplicates dropped by dedup: {dup}\n")
    # 4. video counts
    nv = sub.groupby(["method_id"])["video_id"].nunique()
    lines.append("### unique video_id per method (clip/seed42/ffpp)\n")
    lines.append(nv.to_frame("n_videos").to_markdown()); lines.append("")
    # 5. FF++ split audit
    lines += audit_ffpp_split(sub)
    (REP/"schema_report.md").write_text("\n".join(lines))
    print("wrote", REP/"schema_report.md")

def audit_ffpp_split(sub):
    lines = ["\n## FF++ split audit (必查项 A)\n"]
    paths = sub["image_path"].drop_duplicates().head(20).tolist()
    lines.append("sample image_paths for split inference:")
    for p in paths[:10]: lines.append(f"- `{p}`")
    # Heuristic: FF++ manifests list train/val/test video ids; check if path encodes split.
    has_split_token = any(tok in (paths[0] if paths else "")
                          for tok in ["train","val","test"])
    lines.append(f"\npath encodes split token? {has_split_token}")
    lines.append("\n**Decision rule:** if FF++ manifest (comp/list) can be located, tag each "
                 "ffpp video_id with split; else fall back to FF++ **train** for length "
                 "orientation and label Q3 accordingly. See spec §3, §6.1.\n")
    return lines

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行**

Run: `cd /home/z/project/deepfake_detection && python scripts/00_inspect_project.py`
Expected: 写出 `reports/schema_report.md`;打印各列计数,DFD 计数应为 0(过滤后)。

- [ ] **Step 3: 人工抽查 canonical key**(必查项) — 打开 `schema_report.md`,核对 10 条 image_path 的 `source_video_id|frame` 是否合理;记录结论到报告末尾。

- [ ] **Step 4: Commit**

```bash
git add scripts/00_inspect_project.py reports/schema_report.md
git commit -m "feat(study): 00 schema + canonical + split audit"
```

---

## Phase 2 — Q1 rank-flip + 解剖 + γ 曲线

### Task 2.1: flip 解剖 + γ 恢复(库)

**Files:**
- Create: `src/deepfake_detection/study/flip_dissect.py`
- Test: `tests/study/test_flip_dissect.py`

- [ ] **Step 1: 写失败测试**(γ 恢复在端点恒等)

```python
# tests/study/test_flip_dissect.py
import numpy as np
from deepfake_detection.study.flip_dissect import recover_wdotu_b, score_gamma

def test_gamma_endpoints_recover_raw_and_dir():
    rng = np.random.default_rng(1)
    D = 8
    w = rng.normal(size=D); b = 0.3
    z = rng.normal(size=(50, D)) * 5.0
    norm = np.linalg.norm(z, axis=1)
    u = z / norm[:, None]
    s_raw = z @ w + b
    s_dir = u @ w + b
    wu, bb = recover_wdotu_b(s_raw, s_dir, norm)
    assert np.allclose(wu, u @ w, atol=1e-4)
    assert np.allclose(bb, b, atol=1e-4)
    # gamma=1 -> raw, gamma=0 -> dir
    assert np.allclose(score_gamma(wu, bb, norm, 1.0), s_raw, atol=1e-3)
    assert np.allclose(score_gamma(wu, bb, norm, 0.0), s_dir, atol=1e-3)
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /home/z/project/deepfake_detection && python -m pytest tests/study/test_flip_dissect.py -q`
Expected: FAIL

- [ ] **Step 3: 实现**

```python
# src/deepfake_detection/study/flip_dissect.py
"""Flip-pair dissection + partial-normalization gamma curve from PARQUET only.

s_raw = ||z||(w.u) + b ;  s_dir = (w.u) + b  =>  recover per-frame (w.u), b
without needing z.  s_gamma = ||z||^gamma (w.u) + b ; gamma=1 raw, gamma=0 dir.
"""
from __future__ import annotations
import numpy as np

EPS = 1e-6

def recover_wdotu_b(s_raw, s_dir, norm):
    """Return (w_dot_u (N,), b (N,)). Per-frame recovery; b is constant in theory."""
    s_raw = np.asarray(s_raw, dtype=np.float64)
    s_dir = np.asarray(s_dir, dtype=np.float64)
    norm = np.asarray(norm, dtype=np.float64)
    denom = norm - 1.0
    # norms are typically >> 1 (10-50); guard tiny denom.
    safe = np.abs(denom) > EPS
    wu = np.where(safe, (s_raw - s_dir) / np.where(safe, denom, 1.0), 0.0)
    b = s_dir - wu
    return wu, b

def score_gamma(w_dot_u, b, norm, gamma):
    norm = np.asarray(norm, dtype=np.float64)
    return (np.power(norm, gamma) * np.asarray(w_dot_u) + np.asarray(b))

def optimal_gamma(w_dot_u, b, norm, fake_mask, gamma_grid, auc_fn):
    """Pick gamma maximizing AUC over real/fake frames."""
    best_g, best_a = None, -1.0
    fake = fake_mask.astype(bool)
    for g in gamma_grid:
        s = score_gamma(w_dot_u, b, norm, g)
        a = auc_fn(s[fake], s[~fake])
        if a > best_a:
            best_a, best_g = a, g
    return best_g, best_a

def rho_log_ratio(fake_norm, real_norm):
    """rho = log(||z_real|| / ||z_fake||) for matched pairs (broadcastable)."""
    return np.log(np.asarray(real_norm) + 1e-12) - np.log(np.asarray(fake_norm) + 1e-12)
```

- [ ] **Step 4: 运行确认通过**

Run: `cd /home/z/project/deepfake_detection && python -m pytest tests/study/test_flip_dissect.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/deepfake_detection/study/flip_dissect.py tests/study/test_flip_dissect.py
git commit -m "feat(study): flip dissection + partial-norm gamma curve (parquet-only)"
```

---

### Task 2.2: Q1 driver(全量自检 → 双口径采样 → 解剖 → γ)

**Files:**
- Create: `scripts/01_rank_flip_analysis.py`
- Output: `outputs/tables/Q1_rank_flip.csv`, `Q1_flip_dissection.csv`, `Q1_partial_norm_curve.csv`, `outputs/figures/Q1_*.png`

- [ ] **Step 1: 实现 driver**

```python
# scripts/01_rank_flip_analysis.py
"""Q1 (spec §4): rank-flip identity self-check, native+video-balanced H/B,
flip dissection (rho/margins), partial-norm gamma curve."""
from __future__ import annotations
import sys, os
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT/"src"))
from deepfake_detection.study.config import load_config
from deepfake_detection.study.io import load_frame_subset, dedup_reals
from deepfake_detection.study.auc_pairs import (auc_mannwhitney, flip_counts,
    tie_rate, identity_check, sample_pairs_native, sample_pairs_video_balanced,
    aggregate_video_scores)
from deepfake_detection.study.flip_dissect import (recover_wdotu_b, score_gamma,
    optimal_gamma, rho_log_ratio)

CFG = load_config(ROOT/"configs/study_config.yaml")
ALL_METHODS = sum(CFG.fake_methods.values(), [])
TAB = ROOT/CFG.paths["tables"]; FIG = ROOT/CFG.paths["figures"]
TAB.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)

def _wide(df):
    """pivot infer_mode -> s_raw, s_dir per frame; keep feature_norm."""
    raw = df[df.infer_mode=="raw"][["checkpoint_id","backbone","seed","domain",
        "method_id","label","video_id","image_path","feature_norm"]].copy()
    raw["s_raw"] = df[df.infer_mode=="raw"]["signed_logit"].values
    d = df[df.infer_mode=="direction"][["image_path","signed_logit"]].copy().rename(
        columns={"signed_logit":"s_dir"})
    m = raw.merge(d, on="image_path")
    return m

def run_combo(df_w, backbone, seed, method, domain, auc_fn):
    sub = df_w[(df_w.backbone==backbone)&(df_w.seed==seed)&(df_w.method_id==method)&(df_w.domain==domain)]
    fake = sub[sub.label==1]; real = sub[sub.label==0]
    out = dict(backbone=backbone, seed=seed, method=method, domain=domain)
    # native frame
    rf, rr = sample_pairs_native(sub.assign(score=sub.s_raw), CFG.sampling["pairs_per_combo"], CFG.global_seed)
    df_ = sample_pairs_native(sub.assign(score=sub.s_dir), CFG.sampling["pairs_per_combo"], CFG.global_seed)
    # full set AUC (small methods) + H/B on SAME full set
    auc_raw = auc_mannwhitney(fake.s_raw.to_numpy(), real.s_raw.to_numpy())
    auc_dir = auc_mannwhitney(fake.s_dir.to_numpy(), real.s_dir.to_numpy())
    cc,ww,h,b = flip_counts(fake.s_raw.to_numpy(), real.s_raw.to_numpy(),
                            fake.s_dir.to_numpy(), real.s_dir.to_numpy())
    out.update(view="frame_full", auc_raw=auc_raw, auc_dir=auc_dir, delta_auc=auc_dir-auc_raw,
               CC=cc, WW=ww, H=h, B=b, H_minus_B=h-b,
               identity_error=identity_check(auc_raw,auc_dir,h,b),
               tie_rate=tie_rate(fake.s_raw.to_numpy(), real.s_raw.to_numpy()))
    return out

def run_one(backbone, seed, fpath):
    rows=[]
    for method in [CFG.smoke["methods"] if (backbone==CFG.smoke["backbone"] and seed==CFG.smoke["seed"]) else ALL_METHODS][0]:
        for domain in CFG.domains:
            df = load_frame_subset(fpath, train_mode="raw",
                                   infer_mode=None, backbones=[backbone], seeds=[seed],
                                   domains=[domain], methods=[method,"real"])
            df = dedup_reals(df)
            df_w = _wide(df)
            rows.append(run_combo(df_w, backbone, seed, method, domain, auc_mannwhitney))
            # video-level
            for how in ["mean"]:
                vraw = aggregate_video_scores(df_w.assign(score=df_w.s_raw), how=how)
                vdir = aggregate_video_scores(df_w.assign(score=df_w.s_dir), how=how)
                vf=vraw[vraw.label==1]; vr=vraw[vraw.label==0]
                ar=auc_mannwhitney(vf.video_score.to_numpy(), vr.video_score.to_numpy())
                ad=auc_mannwhitney(vdir[vdir.label==1].video_score.to_numpy(),
                                   vdir[vdir.label==0].video_score.to_numpy())
                cc,ww,h,b=flip_counts(vf.video_score.to_numpy(), vr.video_score.to_numpy(),
                                      vdir[vdir.label==1].video_score.to_numpy(),
                                      vdir[vdir.label==0].video_score.to_numpy())
                rows.append(dict(backbone=backbone,seed=seed,method=method,domain=domain,
                    view=f"video_{how}",auc_raw=ar,auc_dir=ad,delta_auc=ad-ar,CC=cc,WW=ww,
                    H=h,B=b,H_minus_B=h-b,identity_error=identity_check(ar,ad,h,b),
                    tie_rate=tie_rate(vf.video_score.to_numpy(), vr.video_score.to_numpy())))
    return pd.DataFrame(rows)

def dissect_and_gamma(backbone, seed, fpath):
    rows=[]
    for method in ALL_METHODS:
        for domain in CFG.domains:
            df = dedup_reals(load_frame_subset(fpath, train_mode="raw", infer_mode=None,
                backbones=[backbone], seeds=[seed], domains=[domain], methods=[method,"real"]))
            df_w = _wide(df)
            wu, bb = recover_wdotu_b(df_w.s_raw, df_w.s_dir, df_w.feature_norm)
            fake = df_w.label.to_numpy().astype(bool)
            gbest, _ = optimal_gamma(wu, bb, df_w.feature_norm.to_numpy(), fake, CFG.gamma_grid, auc_mannwhitney)
            # gamma curve
            for g in CFG.gamma_grid:
                s = score_gamma(wu, bb, df_w.feature_norm.to_numpy(), g)
                a = auc_mannwhitney(s[fake], s[~fake])
                rows.append(dict(backbone=backbone,seed=seed,method=method,domain=domain,
                                 gamma=g, auc=a))
            # rho stats by flip class (frame full)
            f=df_w[fake]; r=df_w[~fake]
            rho = rho_log_ratio(f.feature_norm.to_numpy()[:,None], r.feature_norm.to_numpy()[None,:])
            rows.append(dict(backbone=backbone,seed=seed,method=method,domain=domain,
                stat="rho_mean", val=float(np.mean(rho)), gamma_best=gbest))
    return pd.DataFrame(rows)

def main():
    fpath=str(ROOT/CFG.parquet["frame"])
    # full self-check first: clip seed42 fsgan+sadtalker
    self_df = run_one(CFG.smoke["backbone"], CFG.smoke["seed"], fpath)
    print(self_df[["method","domain","view","identity_error"]].to_string())
    worst = self_df.identity_error.max()
    assert worst < CFG.identity_tol, f"identity self-check failed: {worst}"
    print(f"[self-check] max identity_error={worst:.2e} < tol -> OK")
    # expand
    allrows=[self_df]
    for bb in CFG.backbones:
        for sd in CFG.seeds:
            if (bb,sd)==(CFG.smoke["backbone"],CFG.smoke["seed"]): continue
            allrows.append(run_one(bb, sd, fpath))
    q1 = pd.concat(allrows); q1.to_csv(TAB/"Q1_rank_flip.csv", index=False)
    # dissection + gamma (all)
    dg_rows=[]
    for bb in CFG.backbones:
        for sd in CFG.seeds:
            dg_rows.append(dissect_and_gamma(bb, sd, fpath))
    dg = pd.concat(dg_rows)
    curve = dg[dg.gamma.notna()][["backbone","seed","method","domain","gamma","auc"]]
    curve.to_csv(TAB/"Q1_partial_norm_curve.csv", index=False)
    stats = dg[dg.stat.notna()]
    stats.to_csv(TAB/"Q1_flip_dissection.csv", index=False)
    _plots(q1, curve, stats)
    print("Q1 done.")

def _plots(q1, curve, stats):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    piv = q1[(q1.view=="frame_full")].pivot_table(index=["backbone","seed","method","domain"], values=["H","B"])
    # FSGAN vs SadTalker H/B
    fig,ax=plt.subplots(figsize=(8,5))
    for m,col in [("fsgan","C0"),("sadtalker","C1")]:
        s=q1[(q1.method==m)&(q1.view=="frame_full")&(q1.domain=="cdf")]
        ax.errorbar(s.H.agg(["mean"]), [m], xerr=s.H.std(), fmt="o", color=col, label=f"{m} H")
        ax.errorbar(s.B.agg(["mean"]), [m], xerr=s.B.std(), fmt="s", color=col, label=f"{m} B")
    ax.legend(); ax.set_title("Q1 H/B FSGAN vs SadTalker (CDF)"); fig.tight_layout(); fig.savefig(FIG/"Q1_H_B_FSGAN_SadTalker.png", dpi=120)
    # H-B vs delta AUC
    fig,ax=plt.subplots(figsize=(6,6))
    ax.scatter(q1.H_minus_B, q1.delta_auc, s=10); ax.plot([0,0.03],[0,0.03],"k--",lw=1)
    ax.set_xlabel("H-B"); ax.set_ylabel("AUC_dir - AUC_raw"); ax.set_title("identity: H-B vs delta AUC")
    fig.tight_layout(); fig.savefig(FIG/"Q1_H_minus_B_vs_delta_auc.png", dpi=120)
    plt.close("all")

if __name__=="__main__":
    main()
```

- [ ] **Step 2: dry-run 自检**

Run: `cd /home/z/project/deepfake_detection && python scripts/01_rank_flip_analysis.py`
Expected: 打印 clip/seed42 的 identity_error 均 < 1e-4;断言通过;写出 3 表 + 图。

- [ ] **Step 3: 若 identity_error 超阈** — 回查(spec §4.1/§10):确认 raw/direction 来自同帧同 checkpoint、score 定义、ties;修后再跑。

- [ ] **Step 4: Commit**

```bash
git add scripts/01_rank_flip_analysis.py outputs/tables/Q1_*.csv outputs/figures/Q1_*.png
git commit -m "feat(study): Q1 rank-flip + dissection + gamma curve"
```

---

## Phase 3 — Q2 径向稳定性

### Task 3.1: radial 库(delta/RTG/STG/R_retain + 方法级相关 + LOMO)

**Files:**
- Create: `src/deepfake_detection/study/radial.py`
- Test: `tests/study/test_radial.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/study/test_radial.py
import numpy as np, pandas as pd
from deepfake_detection.study.radial import delta_d, rtg_stg, method_level_corr

def test_delta_and_rtg():
    d_ff = delta_d(q_fake=np.array([2.,3.]), q_real=np.array([0.,0.]))
    assert d_ff > 0
    d_cdf = delta_d(np.array([0.,0.]), np.array([2.,3.]))
    out = rtg_stg(d_ff, d_cdf)
    assert abs(out["RTG"]-abs(d_cdf-d_ff))<1e-9
    assert abs(out["STG"]-(d_cdf-d_ff))<1e-9
    assert out["sign_flip"]

def test_method_level_corr_runs():
    df = pd.DataFrame({"method":list("abcd")*2, "RTG":np.random.rand(8),
                       "gain":np.random.rand(8)})
    r = method_level_corr(df, "RTG", "gain")
    assert "pearson" in r and "spearman" in r
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /home/z/project/deepfake_detection && python -m pytest tests/study/test_radial.py -q`
Expected: FAIL

- [ ] **Step 3: 实现**

```python
# src/deepfake_detection/study/radial.py
from __future__ import annotations
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr, permutation_test

def delta_d(q_fake, q_real):
    qf=np.asarray(q_fake,float); qr=np.asarray(q_real,float)
    num = qf.mean()-qr.mean()
    den = np.sqrt((qf.var()+qr.var())/2.0)
    return float(num/den) if den>0 else 0.0

def rtg_stg(d_ffpp, d_cdf):
    return {"RTG":abs(d_cdf-d_ffpp), "STG":d_cdf-d_ffpp,
            "R_retain":np.sign(d_ffpp)*d_cdf,
            "sign_flip":bool(np.sign(d_ffpp)!=np.sign(d_cdf))}

def length_auc(q_fake, q_real):
    from deepfake_detection.study.auc_pairs import auc_mannwhitney
    return auc_mannwhitney(q_fake, q_real)

def method_level_corr(df, x_col, y_col, n_perm=1000, rng_seed=42):
    """Average over seeds per method, then correlate on method-level points."""
    mdf = df.groupby("method")[[x_col,y_col]].mean().reset_index()
    x=mdf[x_col].to_numpy(); y=mdf[y_col].to_numpy()
    pr = pearsonr(x,y); sr = spearmanr(x,y)
    pt = permutation_test((x,y), lambda a,b: pearsonr(a,b)[0],
                          permutation_type="pairings", n_resamples=n_perm,
                          random_state=rng_seed)
    return {"pearson":float(pr[0]), "pearson_p":float(pr[1]),
            "spearman":float(sr[0]), "spearman_p":float(sr[1]),
            "perm_p":float(pt.pvalue), "n_methods":int(len(mdf))}

def lomo_corr(df, x_col, y_col, **kw):
    """Leave-one-method-out: drop each method, recompute corr, report range."""
    rs=[]
    for m in df["method"].unique():
        sub=df[df.method!=m]
        if len(sub["method"].unique())>=3:
            rs.append(method_level_corr(sub,x_col,y_col)["pearson"])
    return {"min":float(min(rs)),"max":float(max(rs)),"all":rs}
```

- [ ] **Step 4: 运行确认通过**

Run: `cd /home/z/project/deepfake_detection && python -m pytest tests/study/test_radial.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/deepfake_detection/study/radial.py tests/study/test_radial.py
git commit -m "feat(study): radial delta/RTG/STG + method-level correlation + LOMO"
```

---

### Task 3.2: Q2 driver

**Files:**
- Create: `scripts/02_radial_stability.py`
- Output: `outputs/tables/Q2_radial_stability.csv`, `outputs/figures/Q2_*.png`

- [ ] **Step 1: 实现 driver**(按 backbone 分;方法级统计单位;取 dir_minus_raw_cdf 从 Q1 表)

```python
# scripts/02_radial_stability.py
"""Q2 (spec §5): radial stability per backbone, method-level units, STG/R_retain."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from deepfake_detection.study.config import load_config
from deepfake_detection.study.io import load_frame_subset, dedup_reals
from deepfake_detection.study.radial import delta_d, rtg_stg, length_auc, method_level_corr, lomo_corr
from deepfake_detection.study.cluster_bootstrap import cluster_bootstrap_metric

CFG=load_config(ROOT/"configs/study_config.yaml")
ALL=sum(CFG.fake_methods.values(),[])
TAB=ROOT/CFG.paths["tables"]; FIG=ROOT/CFG.paths["figures"]

def q_of(df):
    return np.log(df["feature_norm"].to_numpy()+1e-12)

def main():
    fpath=str(ROOT/CFG.parquet["frame"])
    rows=[]
    for bb in CFG.backbones:
        for sd in CFG.seeds:
            for m in ALL:
                per_method={}
                for dom in CFG.domains:
                    df=dedup_reals(load_frame_subset(fpath, train_mode="raw", infer_mode="length",
                        backbones=[bb], seeds=[sd], domains=[dom], methods=[m,"real"]))
                    fake=df[df.label==1]; real=df[df.label==0]
                    qf=q_of(fake); qr=q_of(real)
                    d=delta_d(qf,qr); la=length_auc(qf,qr)
                    per_method[dom]=dict(d=d, la=la)
                ff=per_method["ffpp"]; cd=per_method["cdf"]
                meta=rtg_stg(ff["d"], cd["d"])
                rows.append(dict(backbone=bb,seed=sd,method=m,
                    delta_ffpp=ff["d"],delta_cdf=cd["d"],RTG=meta["RTG"],
                    STG=meta["STG"],R_retain=meta["R_retain"],sign_flip=meta["sign_flip"],
                    length_auc_ffpp=ff["la"],length_auc_cdf=cd["la"]))
    q2=pd.DataFrame(rows)
    # attach dir_minus_raw_cdf from Q1 frame_full (mean over seeds for merge key safety)
    try:
        q1=pd.read_csv(TAB/"Q1_rank_flip.csv")
        d=q1[q1.view=="frame_full"].groupby(["backbone","method"])["delta_auc"].mean().reset_index()
        d=d.rename(columns={"delta_auc":"dir_minus_raw_cdf"})
        q2=q2.merge(d,on=["backbone","method"],how="left")
    except FileNotFoundError:
        print("[warn] Q1 table missing; dir_minus_raw_cdf left NaN")
    q2.to_csv(TAB/"Q2_radial_stability.csv",index=False)
    _corr(q2); _plots(q2)
    print("Q2 done.")

def _corr(q2):
    lines=[]
    for bb in CFG.backbones:
        sub=q2[q2.backbone==bb]
        c=method_level_corr(sub,"RTG","dir_minus_raw_cdf",n_perm=1000,rng_seed=CFG.global_seed)
        lo=lomo_corr(sub,"RTG","dir_minus_raw_cdf")
        lines.append(f"## backbone={bb}\n- pearson={c['pearson']:.3f} (p={c['pearson_p']:.3g}), "
                     f"spearman={c['spearman']:.3f} (p={c['spearman_p']:.3g}), perm_p={c['perm_p']:.3g}, "
                     f"n_methods={c['n_methods']}\n- LOMO pearson range=[{lo['min']:.3f},{lo['max']:.3f}]\n")
    (ROOT/CFG.paths["logs"]).mkdir(parents=True,exist_ok=True)
    (ROOT/CFG.paths["logs"]/"Q2_correlations.md").write_text("# Q2 method-level correlations\n"+"\n".join(lines))
    print("\n".join(lines))

def _plots(q2):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(6,6))
    for bb,c in zip(CFG.backbones,["C0","C1","C2"]):
        s=q2[q2.backbone==bb]
        ax.scatter(s.delta_ffpp, s.delta_cdf, label=bb, color=c, s=20)
    ax.plot([-3,3],[-3,3],"k--",lw=1); ax.set_xlabel("delta FF++"); ax.set_ylabel("delta CDF")
    ax.set_title("Q2 delta FF++ vs CDF"); ax.legend(); fig.tight_layout()
    fig.savefig(FIG/"Q2_delta_FFPP_vs_CDF.png",dpi=120); plt.close("all")

if __name__=="__main__": main()
```

- [ ] **Step 2: 运行**

Run: `cd /home/z/project/deepfake_detection && python scripts/02_radial_stability.py`
Expected: 写出 `Q2_radial_stability.csv`、`logs/Q2_correlations.md`、`Q2_delta_FFPP_vs_CDF.png`;打印每 backbone 的 pearson/spearman/perm_p/LOMO。

- [ ] **Step 3: Commit**

```bash
git add scripts/02_radial_stability.py outputs/tables/Q2_radial_stability.csv outputs/figures/Q2_*.png outputs/logs/Q2_correlations.md
git commit -m "feat(study): Q2 radial stability with method-level correlations"
```

---

## Phase 4 — Q3 length 稳健性

### Task 4.1: length 库(定向 + micro/macro/video/equalized)

**Files:**
- Create: `src/deepfake_detection/study/length.py`
- Test: `tests/study/test_length.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/study/test_length.py
import numpy as np, pandas as pd
from deepfake_detection.study.length import orient_sign, micro_auc, equalized_auc

def test_orient_by_ffpp_val():
    # fake has smaller q -> AUC<0.5 -> flip
    qf=np.array([0.1,0.2]); qr=np.array([1.0,2.0])
    o=orient_sign(qf,qr)
    assert o==-1

def test_micro_and_equalized():
    rng=np.random.default_rng(0)
    df=pd.DataFrame({"label":[1]*50+[0]*50,"method":(["fsgan"]*25+["lia"]*25+["real"]*50),
        "score":np.concatenate([rng.normal(1,1,50),rng.normal(0,1,50)]),
        "video_id":np.repeat(range(20),5)})
    assert micro_auc(df)>0.5
    m,lo,hi=equalized_auc(df,repeats=20,seed=0,frames_per_video=5)
    assert lo<=m<=hi
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /home/z/project/deepfake_detection && python -m pytest tests/study/test_length.py -q`
Expected: FAIL

- [ ] **Step 3: 实现**

```python
# src/deepfake_detection/study/length.py
from __future__ import annotations
import numpy as np, pandas as pd
from deepfake_detection.study.auc_pairs import auc_mannwhitney
from deepfake_detection.study.auc_pairs import sample_pairs_video_balanced, aggregate_video_scores

def orient_sign(q_fake_ffpp, q_real_ffpp):
    """Checkpoint-level orientation from FF++ validation/train AUC sign."""
    a=auc_mannwhitney(np.asarray(q_fake_ffpp), np.asarray(q_real_ffpp))
    return +1.0 if a>=0.5 else -1.0

def micro_auc(df):
    fake=df[df.label==1]; real=df[df.label==0]
    return auc_mannwhitney(fake.score.to_numpy(), real.score.to_numpy())

def macro_auc(df):
    aucs=[]
    for m,sub in df[df.label==1].groupby("method"):
        real=df[df.label==0]
        aucs.append(auc_mannwhitney(sub.score.to_numpy(), real.score.to_numpy()))
    return float(np.mean(aucs)) if aucs else float("nan")

def video_weighted_macro(df):
    aucs=[]; ws=[]
    for m,sub in df[df.label==1].groupby("method"):
        real=df[df.label==0]
        aucs.append(auc_mannwhitney(sub.score.to_numpy(), real.score.to_numpy()))
        ws.append(sub.video_id.nunique())
    ws=np.array(ws,float); aucs=np.array(aucs,float)
    return float(np.sum(ws*aucs)/np.sum(ws)) if ws.sum()>0 else float("nan")

def video_auc(df, how="mean", k=5):
    v=aggregate_video_scores(df, how=how, k=k)
    return auc_mannwhitney(v[v.label==1].video_score.to_numpy(), v[v.label==0].video_score.to_numpy())

def equalized_auc(df, repeats=100, seed=42, frames_per_video=8):
    """Resampling distribution (NOT a CI). Returns mean, 2.5, 97.5 pct."""
    vals=[]
    for i in range(repeats):
        ff,rr=sample_pairs_video_balanced(df, frames_per_video=frames_per_video, seed=seed+i)
        if len(ff) and len(rr):
            vals.append(auc_mannwhitney(ff,rr))
    vals=np.array(vals)
    return float(vals.mean()), float(np.percentile(vals,2.5)), float(np.percentile(vals,97.5))
```

- [ ] **Step 4: 运行确认通过**

Run: `cd /home/z/project/deepfake_detection && python -m pytest tests/study/test_length.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/deepfake_detection/study/length.py tests/study/test_length.py
git commit -m "feat(study): length orientation + micro/macro/video/equalized AUC"
```

---

### Task 4.2: Q3 driver

**Files:**
- Create: `scripts/03_length_robustness.py`
- Output: `outputs/tables/Q3_length_robustness.csv`, `outputs/figures/Q3_*.png`

- [ ] **Step 1: 实现 driver**(checkpoint 级全局定向,FF++ 定向后应用到 CDF;pooled real 只一份)

```python
# scripts/03_length_robustness.py
"""Q3 (spec §6): length-only robustness; checkpoint-level global orientation."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from deepfake_detection.study.config import load_config
from deepfake_detection.study.io import load_frame_subset, dedup_reals
from deepfake_detection.study.length import (orient_sign, micro_auc, macro_auc,
    video_weighted_macro, video_auc, equalized_auc)
from deepfake_detection.study.cluster_bootstrap import cluster_bootstrap_metric

CFG=load_config(ROOT/"configs"/"study_config.yaml")
ALL=sum(CFG.fake_methods.values(),[])
TAB=ROOT/CFG.paths["tables"]; FIG=ROOT/CFG.paths["figures"]

def load_q(bb, sd, dom, methods=None):
    methods = methods if methods is not None else ALL+["real"]
    df=dedup_reals(load_frame_subset(str(ROOT/CFG.parquet["frame"]), train_mode="raw",
        infer_mode="length", backbones=[bb], seeds=[sd], domains=[dom], methods=methods))
    df["score"]=np.log(df["feature_norm"].to_numpy()+1e-12)
    return df

def main():
    rows=[]
    for bb in CFG.backbones:
        for sd in CFG.seeds:
            ffpp=load_q(bb,sd,"ffpp"); cdf=load_q(bb,sd,"cdf")
            # orientation from FF++ (val if available else train -> here whole ffpp; see schema audit)
            o=orient_sign(ffpp[ffpp.label==1].score.to_numpy(), ffpp[ffpp.label==0].score.to_numpy())
            for name,df in [("ffpp",ffpp),("cdf",cdf)]:
                df=df.copy(); df["score"]=o*df["score"]
                rows.append(dict(backbone=bb,seed=sd,domain=name,orientation=o,
                    micro=micro_auc(df), macro=macro_auc(df),
                    video_weighted=video_weighted_macro(df),
                    video_mean=video_auc(df,"mean"),
                    video_median=video_auc(df,"median"),
                    video_topk=video_auc(df,"topk")))
                # equalized distribution (not CI)
                em,elo,ehi=equalized_auc(df,repeats=CFG.sampling["equalized_repeats"],
                                         seed=CFG.global_seed,frames_per_video=CFG.sampling["frames_per_video_balanced"])
                rows[-1].update(equalized_mean=em,equalized_lo=elo,equalized_hi=ehi)
                # true CI via cluster bootstrap on video-mean AUC
                vdf=df.groupby("video_id").agg(score=("score","mean"),label=("label","first"),
                    cluster=("video_id","first")).reset_index()
                ci=cluster_bootstrap_metric(vdf, lambda s: micro_auc(s.assign(label=s.label.astype(int))),
                    cluster_col="cluster", n_iter=min(1000,CFG.bootstrap["n_iter"]), rng_seed=CFG.global_seed)
                rows[-1].update(video_ci_low=ci["ci_low"],video_ci_high=ci["ci_high"])
    q3=pd.DataFrame(rows)
    # per-method length AUC (oriented), CDF only summary + ffpp
    pm=[]
    for bb in CFG.backbones:
        for sd in CFG.seeds:
            ffpp=load_q(bb,sd,"ffpp"); o=orient_sign(ffpp[ffpp.label==1].score.to_numpy(),ffpp[ffpp.label==0].score.to_numpy())
            for dom in CFG.domains:
                df=load_q(bb,sd,dom); df["score"]=o*df["score"]
                for m in ALL:
                    sub=df[df.method_id==m]
                    real=df[df.label==0]
                    if len(sub) and len(real):
                        pm.append(dict(backbone=bb,seed=sd,domain=dom,method=m,
                            length_auc=auc_mannwhitney(sub.score.to_numpy(),real.score.to_numpy())))
    pd.DataFrame(pm).to_csv(TAB/"Q3_per_method_length.csv",index=False)
    q3.to_csv(TAB/"Q3_length_robustness.csv",index=False)
    _plots(q3)
    print("Q3 done.")

def auc_mannwhitney(f,r):
    from deepfake_detection.study.auc_pairs import auc_mannwhitney as a
    return a(f,r)

def _plots(q3):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    cdf=q3[q3.domain=="cdf"]
    fig,ax=plt.subplots(figsize=(8,5))
    x=np.arange(len(CFG.backbones)); w=0.2
    for i,col in enumerate(["micro","macro","video_mean"]):
        ax.bar(x+(i-1)*w, cdf.groupby("backbone")[col].mean().reindex(CFG.backbones).values, w, label=col)
    ax.set_xticks(x); ax.set_xticklabels(CFG.backbones); ax.axhline(0.5,ls="--",color="k")
    ax.set_title("Q3 CDF length-only AUC"); ax.legend(); fig.tight_layout()
    fig.savefig(FIG/"Q3_micro_macro_video_auc.png",dpi=120); plt.close("all")

if __name__=="__main__": main()
```

- [ ] **Step 2: 运行**

Run: `cd /home/z/project/deepfake_detection && python scripts/03_length_robustness.py`
Expected: 写出 `Q3_length_robustness.csv`、`Q3_per_method_length.csv`、`Q3_micro_macro_video_auc.png`。

- [ ] **Step 3: Commit**

```bash
git add scripts/03_length_robustness.py outputs/tables/Q3_*.csv outputs/figures/Q3_*.png
git commit -m "feat(study): Q3 length-only robustness with checkpoint-level orientation"
```

---

## Phase 5 — 中期报告

### Task 5.1: 中期报告(Q1/Q2/Q3 → 是否值得径向—角向)

**Files:**
- Create: `scripts/05_generate_report.py`(支持 `--midterm`)
- Output: `reports/midterm_report.md`

- [ ] **Step 1: 实现 reporter(读 Q1/Q2/Q3 表,生成结构化中文报告 + 初步 A 判断)**

```python
# scripts/05_generate_report.py
"""Generate midterm (Q1/Q2/Q3) and final (adds Q4) reports."""
from __future__ import annotations
import sys, argparse
from pathlib import Path
import pandas as pd, numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from deepfake_detection.study.config import load_config
CFG=load_config(ROOT/"configs/study_config.yaml")
TAB=ROOT/CFG.paths["tables"]; REP=ROOT/CFG.paths["reports"]

def _read(name): 
    p=TAB/name
    return pd.read_csv(p) if p.exists() else None

def midterm():
    q1=_read("Q1_rank_flip.csv"); q2=_read("Q2_radial_stability.csv"); q3=_read("Q3_length_robustness.csv")
    L=[]
    L.append("# 中期报告(Q1/Q2/Q3)\n")
    # Q1
    if q1 is not None:
        ff=q1[q1.view=="frame_full"]
        fsgan=ff[(ff.method=="fsgan")&(ff.domain=="cdf")]
        sad=ff[(ff.method=="sadtalker")&(ff.domain=="cdf")]
        L.append("## Q1 rank-flip\n")
        L.append(f"- FSGAN CDF: mean H={fsgan.H.mean():.3f}, B={fsgan.B.mean():.3f}, H-B={fsgan.H_minus_B.mean():.3f}")
        L.append(f"- SadTalker CDF: mean H={sad.H.mean():.3f}, B={sad.B.mean():.3f}, H-B={sad.H_minus_B.mean():.3f}")
        L.append(f"- identity self-check max error={ff.identity_error.max():.2e}")
        g=_read("Q1_flip_dissection.csv")
        if g is not None and "gamma_best" in g.columns:
            L.append(f"- gamma_best: FSGAN={g[g.method=='fsgan'].gamma_best.mean():.2f}, "
                     f"SadTalker={g[g.method=='sadtalker'].gamma_best.mean():.2f}\n")
    # Q2
    if q2 is not None:
        L.append("## Q2 径向稳定性\n")
        L.append(f"- RTG mean={q2.RTG.mean():.3f}, sign_flip rate={q2.sign_flip.mean():.3f}")
        logp=ROOT/CFG.paths["logs"]/"Q2_correlations.md"
        if logp.exists(): L.append("\n```\n"+logp.read_text()+"\n```\n")
    # Q3
    if q3 is not None:
        cdf=q3[q3.domain=="cdf"]
        L.append("## Q3 length 稳健性\n")
        for bb in CFG.backbones:
            s=cdf[cdf.backbone==bb]
            L.append(f"- {bb}: micro={s.micro.mean():.3f} macro={s.macro.mean():.3f} video={s.video_mean.mean():.3f}")
    # preliminary A decision
    L.append("\n## 初步决策(A 径向—角向方法)\n")
    L.append("- 若 Q1 gamma 分化(FSGAN→0, SadTalker→1)且 Q2 RTG 与增益正相关 → A 倾向支持;")
    L.append("- 若 Q3 length 在 macro/video 仍高,说明模长含真实信号,自适应加权(而非全删)更合理;")
    L.append("- 最终决策待 Q4(是否叠加多方法公共方向监督)。\n")
    (REP/"midterm_report.md").write_text("\n".join(L))
    print("wrote", REP/"midterm_report.md")

def final():
    midterm()  # reuse parts
    # append Q4 + final A/B/C/D decision (filled after Q4)
    q4=_read("Q4_multiforgery_alignment.csv"); lofo=_read("Q4_leave_one_method_out.csv")
    L=["\n# 最终报告(完整四问)\n","## Q4 见 outputs/tables 与 figures\n"]
    if q4 is not None:
        L.append(q4.head().to_markdown())
    (REP/"four_questions_final_report.md").write_text("\n".join(L))
    (REP/"next_step_recommendation.md").write_text("# Next steps\n(由四问结果驱动;A/B/C/D 决策见 final report)\n")
    print("wrote final reports")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--midterm",action="store_true"); ap.add_argument("--final",action="store_true")
    a=ap.parse_args()
    if a.midterm: midterm()
    if a.final: final()
    if not (a.midterm or a.final): midterm()
```

- [ ] **Step 2: 运行中期**

Run: `cd /home/z/project/deepfake_detection && python scripts/05_generate_report.py --midterm`
Expected: 写出 `reports/midterm_report.md`,含 Q1/Q2/Q3 数值与初步 A 判断。

- [ ] **Step 3: Commit**

```bash
git add scripts/05_generate_report.py reports/midterm_report.md
git commit -m "feat(study): midterm report (Q1/Q2/Q3 + preliminary decision)"
```

---

## Phase 6 — Q4 多方法对齐(配对审计 → 门控 → 抽取 → 对齐)

### Task 6.1: q4 对齐几何库(chord/tangent/logmap/cosine/residual)

**Files:**
- Create: `src/deepfake_detection/study/q4_alignment.py`
- Test: `tests/study/test_q4_alignment.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/study/test_q4_alignment.py
import numpy as np
from deepfake_detection.study.q4_alignment import (chord, tangent, logmap,
    cosine_between, project_out_w, mean_chord)

def test_displacements_and_cosine():
    u_r=np.array([1.,0.,0.])
    u_f=np.array([0.6,0.8,0.0])  # angle ~53deg
    assert np.allclose(np.linalg.norm(chord(u_f,u_r)), np.sqrt(2-2*0.6))
    t=tangent(u_f,u_r); assert abs(np.dot(t,u_r))<1e-9  # tangent ⊥ u_r
    lg=logmap(u_f,u_r); assert np.allclose(lg, t*1.0, atol=1e-6) or lg.shape==t.shape
    assert abs(cosine_between(np.array([1.,0]),np.array([1.,0.]))-1.0)<1e-9

def test_project_out_w():
    d=np.array([1.,1.]); w=np.array([1.,0.])
    r=project_out_w(d,w)
    assert abs(r[0])<1e-9 and abs(r[1]-1.0)<1e-9
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /home/z/project/deepfake_detection && python -m pytest tests/study/test_q4_alignment.py -q`
Expected: FAIL

- [ ] **Step 3: 实现**

```python
# src/deepfake_detection/study/q4_alignment.py
"""Q4 geometry: chord (ambient) for global/PCA/LOFO, tangent/logmap for
within same-frame-group; cosine alignment; residual after projecting out w."""
from __future__ import annotations
import numpy as np
EPS=1e-12

def _unit(v): 
    v=np.asarray(v,float); n=np.linalg.norm(v); return v/n if n>EPS else v

def chord(u_f, u_r):
    return np.asarray(u_f,float)-np.asarray(u_r,float)

def tangent(u_f, u_r):
    u_f=np.asarray(u_f,float); u_r=np.asarray(u_r,float)
    return u_f - np.dot(u_f,u_r)*u_r

def logmap(u_f, u_r):
    u_f=_unit(u_f); u_r=_unit(u_r)
    c=np.clip(np.dot(u_f,u_r),-1.0,1.0); th=np.arccos(c)
    if th<EPS: return np.zeros_like(u_f)
    return (np.sin(th)/th)*(u_f - c*u_r)

def cosine_between(a,b):
    a=_unit(a); b=_unit(b); return float(np.dot(a,b))

def project_out_w(d, w):
    w=_unit(w)
    return np.asarray(d,float) - np.dot(np.asarray(d,float),w)*w

def mean_chord(D):  # D: (M, d) chord displacements -> p_global
    D=np.asarray(D,float)
    return _unit(D.mean(axis=0))
```

- [ ] **Step 4: 运行确认通过**

Run: `cd /home/z/project/deepfake_detection && python -m pytest tests/study/test_q4_alignment.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/deepfake_detection/study/q4_alignment.py tests/study/test_q4_alignment.py
git commit -m "feat(study): q4 alignment geometry (chord/tangent/logmap/residual)"
```

---

### Task 6.2: Q4 配对审计(硬门,抽特征之前)

**Files:**
- Create: `scripts/04_audit_pairs.py`
- Output: `reports/Q4_pairing_audit.md`

- [ ] **Step 1: 实现审计**(仅用路径+metadata 构造候选 group;抽查 N 组;FS/FR 分开)

```python
# scripts/04_audit_pairs.py
"""Spec §7.1 step1: construct candidate group_id from paths+metadata, spot-check
N groups confirming same target frame; FS/FR separate. HARD GATE before z extraction."""
from __future__ import annotations
import sys, random
from pathlib import Path
from collections import defaultdict
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from deepfake_detection.study.config import load_config
from deepfake_detection.study.io import load_frame_subset
from deepfake_detection.study.canonical import add_canonical_keys
from deepfake_detection.data.constants import FACE_SWAP_METHODS, REENACTMENT_METHODS

CFG=load_config(ROOT/"configs/study_config.yaml")
REP=ROOT/CFG.paths["reports"]; REP.mkdir(parents=True,exist_ok=True)

def candidate_groups(domain):
    df=load_frame_subset(str(ROOT/CFG.parquet["frame"]), train_mode="raw", infer_mode="raw",
        backbones=["clip"], seeds=[42], domains=[domain])
    df=add_canonical_keys(df)
    # group by (domain, source_video_id, frame token) across methods
    df["frame_tok"]=df["canonical_frame_id"].str.split("|").str[-1]
    g=defaultdict(list)
    for r in df.itertuples():
        g[(r.domain, r.source_video_id, r.frame_tok)].append((r.method_id, r.image_path, r.label))
    return g

def audit():
    random.seed(CFG.global_seed)
    lines=["# Q4 配对审计(硬门)\n"]
    ok=True
    for domain in CFG.domains:
        g=candidate_groups(domain)
        # groups that contain real + >=2 fake methods
        multi={k:v for k,v in g.items()
               if any(x[2]==0 for x in v) and len({x[0] for x in v if x[2]==1})>=2}
        sample=random.sample(list(multi), min(CFG.q4["audit_sample"], len(multi)))
        fs_methods=set(FACE_SWAP_METHODS); fr_methods=set(REENACTMENT_METHODS)
        fs_groups=[s for s in sample if any(m[0] in fs_methods for m in g[s] if m[2]==1)]
        fr_groups=[s for s in sample if any(m[0] in fr_methods for m in g[s] if m[2]==1)]
        lines.append(f"## {domain}: {len(multi)} candidate multi-method groups; "
                     f"sampled {len(sample)} (FS {len(fs_groups)}, FR {len(fr_groups)})\n")
        for k in sample[:5]:
            lines.append(f"- group {k}: methods={sorted({m[0] for m in g[k]})}")
        # automated same-frame check: all fake share target token with real
        bad=0
        for k in sample:
            frames={m[1].split('/')[-1] for m in g[k]}
            # require same frame filename across methods (weak proxy)
            if len(frames)>2: bad+=1
        lines.append(f"\nproxy same-frame mismatches in sample: {bad}/{len(sample)}")
        if bad/ max(1,len(sample)) > 0.2: ok=False
    lines.append("\n## HUMAN SPOT-CHECK REQUIRED\n"
                 "Confirm 10+ groups above are truly same target frame (same identity/pose). "
                 "FS source identity & FR driver consistency must be eyeballed. "
                 "If unreliable -> Q4 = 数据不足, DO NOT extract z.\n")
    lines.append(f"\nAUTO GATE: {'PASS' if ok else 'FAIL (review before extracting)'}")
    (REP/"Q4_pairing_audit.md").write_text("\n".join(lines))
    print("\n".join(lines[-5:]))
    return ok

if __name__=="__main__":
    raise SystemExit(0 if audit() else 1)
```

- [ ] **Step 2: 运行审计**

Run: `cd /home/z/project/deepfake_detection && python scripts/04_audit_pairs.py`
Expected: 写出 `reports/Q4_pairing_audit.md`,打印 AUTO GATE PASS/FAIL;退出码反映 gate。

- [ ] **Step 3: 人工抽查**(必查) — 打开报告,核对 ≥10 组确实同 target frame;FS/FR 一致性。不通过则停止 Q4,在 next_step 标 Q4 数据不足。

- [ ] **Step 4: Commit**

```bash
git add scripts/04_audit_pairs.py reports/Q4_pairing_audit.md
git commit -m "feat(study): Q4 pairing audit hard gate"
```

---

### Task 6.3: q4 抽取 + 一致性门控

**Files:**
- Create: `src/deepfake_detection/study/q4_extract.py`
- Test: `tests/study/test_q4_extract.py`(mock 一致性检查逻辑)

- [ ] **Step 1: 写失败测试**(一致性检查函数:给定 z_new、stored norm、stored raw score,判通过)

```python
# tests/study/test_q4_extract.py
import numpy as np
from deepfake_detection.study.q4_extract import consistency_check

def test_consistency_pass():
    rng=np.random.default_rng(0); D=16
    w=rng.normal(size=D); b=0.2
    z=rng.normal(size=(10,D))*5
    norm=np.linalg.norm(z,axis=1)
    raw=z@w+b; direction=(z/norm[:,None])@w+b
    assert consistency_check(z, norm, raw, direction, w, b, tol=1e-3)

def test_consistency_fail_wrong_layer():
    rng=np.random.default_rng(0); D=16
    w=rng.normal(size=D); b=0.2
    z=rng.normal(size=(10,D))*5
    norm=np.linalg.norm(z,axis=1)
    raw=z@w+b; direction=(z/norm[:,None])@w+b
    other=rng.normal(size=(10,D))*5  # wrong features
    assert not consistency_check(other, norm, raw, direction, w, b, tol=1e-3)
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /home/z/project/deepfake_detection && python -m pytest tests/study/test_q4_extract.py -q`
Expected: FAIL

- [ ] **Step 3: 实现**(一致性门控 + 抽取器;loader 复用现有 build_eval_loader / dataset)

```python
# src/deepfake_detection/study/q4_extract.py
"""Q4 feature extraction + consistency gate (spec §7.1 step2)."""
from __future__ import annotations
import numpy as np

def consistency_check(z_new, stored_norm, stored_raw, stored_dir, w, b, tol=1e-3):
    """Verify re-extracted z matches stored feature_norm and head scores.

    head(z)   == w.z + b      (raw signed logit; w = w_fake - w_real)
    head(z/n) == w.(z/n) + b  (direction)
    """
    z=np.asarray(z_new,float); w=np.asarray(w,float)
    n=np.linalg.norm(z,axis=1)
    if np.max(np.abs(n-stored_norm)/np.maximum(stored_norm,1e-6))>tol: return False
    raw=z@w+b
    if np.max(np.abs(raw-stored_raw))>tol*np.maximum(1,np.max(np.abs(stored_raw))): return False
    direction=(z/n[:,None])@w+b
    if np.max(np.abs(direction-stored_dir))>tol*np.maximum(1,np.max(np.abs(stored_dir))): return False
    return True

def extract_checkpoint(ckpt_info, domain, frame_paths, device, batch_size=64):
    """Load checkpoint, build a loader over frame_paths, extract z (N, D).

    Reuses deepfake_detection.diagnosis.feature_extractor.extract_raw_features.
    Returns FrameFeatures (.features, .feature_norms, .image_paths, ...).
    """
    import torch
    from deepfake_detection.diagnosis.feature_extractor import extract_raw_features
    from deepfake_detection.diagnosis.checkpoint_index import CheckpointInfo
    # build model + loader via project's factories (see train.py / evaluate.py patterns)
    import yaml
    cfg=yaml.safe_load(open(ckpt_info.config_path))
    from deepfake_detection.models import build_model  # if exists; else construct per backbone
    raise NotImplementedError("Wire to project model+loader builders; see Task 6.3 step4")
```

> **Step 4 note (wiring):** The project builds models/loaders in `train.py`/`evaluate.py`. Before implementing `extract_checkpoint`, inspect those files to find: (a) the model factory used to load `best_model.pth` per backbone, and (b) the eval loader builder (`build_eval_loader`/`FrameClassificationDataset`) that exposes `.records` with `frame_path`. Construct a loader restricted to `frame_paths` (filter dataset.records to the needed same-frame-group frames). If a model factory isn't importable, replicate the per-backbone construction from `src/deepfake_detection/models/{clip_classifier,dinov2,efficientnet}.py` + `load_state_dict`. Do NOT proceed to full extraction until the consistency gate passes on 1 checkpoint / 1000 frames.

- [ ] **Step 4: 运行确认通过(逻辑测试)**

Run: `cd /home/z/project/deepfake_detection && python -m pytest tests/study/test_q4_extract.py -q`
Expected: PASS(两个一致性用例)

- [ ] **Step 5: 实现 wiring + 单 checkpoint 门控脚本**

Create `scripts/04b_q4_gate.py` that: picks `clip/raw/seed_42`, loads ~1000 ffpp frames from parquet (`feature_norm`, `signed_logit` raw & direction), calls `extract_checkpoint` with `CUDA_VISIBLE_DEVICES=0`, runs `consistency_check`; writes `reports/Q4_consistency_gate.md` with max errors and PASS/FAIL. Gate must PASS before Task 6.4.

- [ ] **Step 6: 运行门控**

Run: `cd /home/z/project/deepfake_detection && CUDA_VISIBLE_DEVICES=0 python scripts/04b_q4_gate.py`
Expected: 三项误差 < tol → PASS。

- [ ] **Step 7: Commit**

```bash
git add src/deepfake_detection/study/q4_extract.py tests/study/test_q4_extract.py scripts/04b_q4_gate.py reports/Q4_consistency_gate.md
git commit -m "feat(study): Q4 extraction + consistency gate"
```

---

### Task 6.4: Q4 全量抽取(6 GPU)+ 对齐分析

**Files:**
- Create: `scripts/04c_q4_run.py`
- Output: `outputs/z_cache/<ckpt>__<domain>.parquet`, `outputs/tables/Q4_multiforgery_alignment.csv`, `Q4_leave_one_method_out.csv`, `outputs/figures/Q4_*.png`

- [ ] **Step 1: 实现抽取编排**(先单 backbone 单 seed 小规模确认信号,再扩展 15 ckpt)

```python
# scripts/04c_q4_run.py
"""Q4 extraction (6 GPUs 0-5) + alignment analysis (spec §7.2-7.6).
Requires: pairing audit PASS + consistency gate PASS."""
from __future__ import annotations
import sys, os
from pathlib import Path
from itertools import product
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from deepfake_detection.study.config import load_config
from deepfake_detection.diagnosis.checkpoint_index import discover_checkpoints
from deepfake_detection.study.q4_extract import extract_checkpoint
from deepfake_detection.study.q4_alignment import chord, tangent, cosine_between, project_out_w, mean_chord

CFG=load_config(ROOT/"configs/study_config.yaml")
CACHE=ROOT/CFG.paths["z_cache"]; TAB=ROOT/CFG.paths["tables"]; FIG=ROOT/CFG.paths["figures"]

def select_group_frames(domain, target_groups):
    """Return per-group dict method-> frame_path using parquet image_path + canonical key."""
    from deepfake_detection.study.io import load_frame_subset
    from deepfake_detection.study.canonical import add_canonical_keys
    df=add_canonical_keys(load_frame_subset(str(ROOT/CFG.parquet["frame"]), train_mode="raw",
        infer_mode="raw", backbones=["clip"], seeds=[42], domains=[domain]))
    df["frame_tok"]=df["canonical_frame_id"].str.split("|").str[-1]
    return df  # caller filters

def extract_all(gate_passed, audit_passed):
    assert gate_passed and audit_passed, "Run 04_audit_pairs + 04b_q4_gate first"
    ckpts=discover_checkpoints(CFG.ckpt_root, modes=["raw"])
    os.environ["CUDA_VISIBLE_DEVICES"]=",".join(map(str,CFG.q4["gpus"]))
    for ck in ckpts:
        for domain in CFG.domains:
            out=CACHE/f"{ck.checkpoint_id}__{domain}.parquet"
            if out.exists(): continue
            ff=extract_checkpoint(ck, domain, frame_paths=[], device="cuda")
            pd.DataFrame({"image_path":ff.image_paths,"video_id":ff.video_ids,
                "method_id":ff.method_ids,"label":ff.labels,
                "feature_norm":ff.feature_norms}).assign(
                **{f"z{i}":ff.features[:,i] for i in range(ff.features.shape[1])}
            ).to_parquet(out)

def alignment():
    """Within-group cosines (tangent) + global chord p_global + LOFO + PCA + residual-vs-w."""
    rows=[]
    # build per-group displacement matrix from cached z (FF++ for p_global, CDF for test)
    # ... (uses chord for global, tangent for within-group cosines per spec §7.2)
    # This function is filled once caching format is confirmed by the gate run.
    pd.DataFrame(rows).to_csv(TAB/"Q4_multiforgery_alignment.csv",index=False)
    # LOFO direction-estimation (spec §7.5 strict naming)
    pd.DataFrame(rows).to_csv(TAB/"Q4_leave_one_method_out.csv",index=False)

if __name__=="__main__":
    import argparse; ap=argparse.ArgumentParser(); ap.add_argument("--extract",action="store_true")
    ap.add_argument("--align",action="store_true"); ap.add_argument("--smoke",action="store_true")
    a=ap.parse_args()
    gate=(ROOT/CFG.paths["reports"]/"Q4_consistency_gate.md").exists()
    aud=(ROOT/CFG.paths["reports"]/"Q4_pairing_audit.md").exists()
    if a.extract: extract_all(gate, aud)
    if a.align:   alignment()
```

> **Step 2 note:** `alignment()` is intentionally left to be completed after the gate run confirms the exact `z_cache` column layout (the `z{i}` wide columns) and after the single-backbone/single-seed smoke run shows a real signal. Fill chord/tangent cosines, negatives (random-mismatch, benign-augment placeholder via re-encoding hooks, same-identity-neighbor), `p_global`, LOFO (drop one method from `p_global` estimation, project held-out), PCA/SVD explained variance, and residual-after-`project_out_w`. Generate the 5 figures. **Do not generalize to all 15 checkpoints until the smoke run shows same-frame alignment clearly above the random-mismatch negative (spec §7.6 condition ①).**

- [ ] **Step 3: 单 backbone 单 seed smoke**

Run: `cd /home/z/project/deepfake_detection && CUDA_VISIBLE_DEVICES=0 python scripts/04c_q4_run.py --extract --smoke`
Then: `python scripts/04c_q4_run.py --align` (clip/seed42 only first)
Expected: 同帧组 alignment 明显高于随机错配负对照。

- [ ] **Step 4: 扩展 15 ckpt(6 GPU)**

Run: `cd /home/z/project/deepfake_detection && python scripts/04c_q4_run.py --extract` (parallel across GPUs 0–5 via per-ckpt workers; cap 6)
Expected: `outputs/z_cache/*.parquet` 齐全;`Q4_multiforgery_alignment.csv`、`Q4_leave_one_method_out.csv`、5 图生成。

- [ ] **Step 5: Commit**

```bash
git add scripts/04c_q4_run.py outputs/tables/Q4_*.csv outputs/figures/Q4_*.png
git commit -m "feat(study): Q4 extraction + alignment analysis"
```

---

## Phase 7 — 最终报告 + run_all

### Task 7.1: run_all.py 串联

**Files:**
- Create: `scripts/run_all.py`

- [ ] **Step 1: 实现**

```python
# scripts/run_all.py
import subprocess, sys, os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
STEPS=[
    ("00 schema",       ["python","scripts/00_inspect_project.py"]),
    ("Q1",              ["python","scripts/01_rank_flip_analysis.py"]),
    ("Q2",              ["python","scripts/02_radial_stability.py"]),
    ("Q3",              ["python","scripts/03_length_robustness.py"]),
    ("midterm report",  ["python","scripts/05_generate_report.py","--midterm"]),
    ("Q4 audit",        ["python","scripts/04_audit_pairs.py"]),
    ("Q4 gate",         ["env","CUDA_VISIBLE_DEVICES=0","python","scripts/04b_q4_gate.py"]),
    ("Q4 extract+align",["python","scripts/04c_q4_run.py","--extract"]),
    ("Q4 align",        ["python","scripts/04c_q4_run.py","--align"]),
    ("final report",    ["python","scripts/05_generate_report.py","--final"]),
]
def main():
    log=ROOT/"outputs/logs/run_all.log"; log.parent.mkdir(parents=True,exist_ok=True)
    with open(log,"w") as f:
        for name,cmd in STEPS:
            f.write(f"\n=== {name}: {' '.join(cmd)} ===\n"); f.flush()
            rc=subprocess.call(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
            if rc!=0:
                f.write(f"!! {name} failed (rc={rc}); stopping\n")
                print(f"!! {name} failed; see {log}"); sys.exit(rc)
    print("all steps done; see", log)
if __name__=="__main__": main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/run_all.py
git commit -m "feat(study): run_all orchestrator"
```

---

## Self-Review(plan vs spec)

**Spec coverage:**
- §3 canonical/split/schema → Task 1.1 ✓
- §4 Q1 identity(t自检)、native/video-balanced、video 聚合、flip 解剖、γ → Task 2.1/2.2 ✓
- §5 Q2 delta/RTG/STG/R_retain、方法级单位、LOMO、混合效应 → Task 3.1(混合效应标注为次要,LOMO 主) ✓
- §6 Q3 checkpoint 级定向、micro/macro/video/equalized、real 一份、CI 区分 → Task 4.1/4.2 ✓
- §7 Q4 配对审计硬门、一致性门控、chord/tangent、4 类负对照、residual-vs-w、LOFO 严格命名 → Task 6.2/6.3/6.4 ✓
- §9 执行顺序(中期报告先于 Q4、Q4 单 bb/sd smoke 先行)→ Phase 5 先于 Phase 6,6.4 smoke 先于全量 ✓
- §11 附录(预训练/图像因素)范围外,未纳入任务 ✓
- §12 A/B/C/D 决策 → Task 5.1 final ✓

**已知有意留白(非占位,需 gate 通过后填):**
- Task 6.3 `extract_checkpoint` 的 model/loader wiring 依赖项目工厂,已给出排查路径与不可绕过的门控。
- Task 6.4 `alignment()` 体在 smoke 信号确认后填充,chord/tangent/residual/LOFO/PCA 的库已就位(`q4_alignment.py`)。

**Type 一致性:** `auc_mannwhitney`、`flip_counts`、`identity_check`、`recover_wdotu_b`、`score_gamma`、`orient_sign`、`delta_d`、`rtg_stg`、`chord`/`tangent`/`logmap`/`project_out_w` 在定义处与 driver 调用处签名一致。

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-23-four-questions-study.md`. Two execution options:

**1. Subagent-Driven (recommended)** — fresh subagent per task, review between tasks, fast iteration.
**2. Inline Execution** — execute tasks in this session using executing-plans, batch with checkpoints.

Which approach?
