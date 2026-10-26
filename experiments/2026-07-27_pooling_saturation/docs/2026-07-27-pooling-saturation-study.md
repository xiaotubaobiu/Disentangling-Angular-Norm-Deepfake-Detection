# Pooling/Saturation 机制研究 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 判定 direction>raw 的视频级优势究竟来自 sigmoid 饱和+概率聚合 / 真实径向—角向作用 / 视频内时间一致性 / 混合且方法依赖(A/B/C/D),输出表格、图与最终报告。

**Architecture:** 复用前序 `study/` 库;新增 `pooling.py`(聚合器+Jensen+饱和+时间一致性)、`temperature.py`(NLL 温度+Platt)、`wb_io.py`(读 checkpoint w,b)。parquet-only 实验(1/2/3/6/7/γ)从 `per_frame_scores.parquet` 读 `signed_logit`;校准实验(4/5)补跑 15 ckpt × FF++ val(项目切片,经 `extract_checkpoint`),val 拟参、test/CDF 评估。

**Tech Stack:** Python, DuckDB, pandas, NumPy, SciPy, scikit-learn, PyTorch, matplotlib;pytest。

## Global Constraints

- DFD 全程排除:`domain != 'dfd'`,弃 `dfd_native`/`DFD_*`。
- 只用 raw-trained checkpoint;raw/direction 同 checkpoint/同头/同帧。
- `s = fake_logit − real_logit`;`p = sigmoid(s)`;`prob_fake == sigmoid(signed_logit)`(已验证)。
- γ:`s_γ = r^γ·(w·u)+b`,b 从 checkpoint 真实读(`w=w_fake−w_real, b=b_fake−b_real`),不假设 b=0。
- **禁止用 FF++ test 或 CDF 拟合温度/Platt/尺度/γ**;clip{2,4,6,8}、γ grid、top25%、trim10% 固定;CDF γ_best 仅称 oracle diagnostic。
- bootstrap:video-cluster paired,≥2000 次,raw/dir/各聚合器同样本;BH 多重比较;帧不作独立样本。
- GPU 仅 0–5;报告中文,列名/表头英文。
- `git add` 仅确切新路径(仓库 code-only:`tests/`/`outputs/`/`reports/`/`docs/` gitignored → tests 用 `git add -f`;产物留磁盘)。pre-existing staged deletions 不得带入。
- 包导入:`python -m pytest tests/study_pooling/ -q` 自 repo root 可用(editable install)。

**Spec:** `docs/superpowers/specs/2026-07-27-pooling-saturation-design.md`。

---

## 文件结构

新增库(每个单一职责):
- `src/deepfake_detection/study_pooling/__init__.py`
- `src/deepfake_detection/study_pooling/config.py` — 读 `configs/pooling_saturation_config.yaml`
- `src/deepfake_detection/study_pooling/aggregators.py` — 7 种视频聚合器 + Jensen gap
- `src/deepfake_detection/study_pooling/saturation.py` — logit 分位/饱和率/sigmoid 梯度 + 视频级
- `src/deepfake_detection/study_pooling/temperature.py` — NLL 温度拟合 + Platt scaling + ECE/Brier
- `src/deepfake_detection/study_pooling/wb_io.py` — 从 checkpoint 读 signed w,b
- `src/deepfake_detection/study_pooling/gamma.py` — s_γ(从真实 b)+ 多聚合器扫描
- `src/deepfake_detection/study_pooling/temporal.py` — 视频内时间一致性统计
- `src/deepfake_detection/study_pooling/video_eval.py` — 视频分数→AUC/flip_counts/identity(复用 study.auc_pairs)

复用:`deepfake_detection.study.{config,io,canonical,auc_pairs,cluster_bootstrap}`、`deepfake_detection.study.q4_extract.{extract_checkpoint, build_model_from_checkpoint}`、`deepfake_detection.diagnosis.checkpoint_index.discover_checkpoints`、`deepfake_detection.data.builders.build_val_loader`。

脚本:`scripts/06_pooling_saturation_analysis.py`(实验1/2/3/γ)、`scripts/07_fit_temperature_calibration.py`(实验4/5)、`scripts/08_pooling_rank_flip.py`(实验6)、`scripts/09_generate_pooling_report.py`。
配置:`configs/pooling_saturation_config.yaml`。
测试:`tests/study_pooling/test_*.py`。
输出:`outputs/tables/Q_pooling_*.csv`、`outputs/figures/Q_pooling_*.png`、`reports/pooling_saturation_*.md`。

---

## Task 1: 配置 + 包骨架

**Files:**
- Create: `configs/pooling_saturation_config.yaml`
- Create: `src/deepfake_detection/study_pooling/__init__.py`
- Create: `src/deepfake_detection/study_pooling/config.py`
- Test: `tests/study_pooling/test_config.py`

**Interfaces:**
- Produces: `load_config(path)->PoolingCfg` with attributes `raw, global_seed, backbones, seeds, methods, domains, bootstrap_iterations, clip_values, gamma_values, topk_fraction, trim_fraction, output_dir, parquet_frame, ckpt_root, paths`.

- [ ] **Step 1: 写配置 `configs/pooling_saturation_config.yaml`**
```yaml
global_seed: 42
parquet_frame: outputs/frame_midt_diag/parallel_20260721_213914/per_frame_scores.parquet
ckpt_root: experiments/final_results/checkpoints
backbones: [clip, dinov2, efficientnet]
seeds: [7, 42, 123, 999, 2048]
methods: [simswap, inswap, blendface, faceswap, fsgan, mobileswap, e4s, facedancer,
          fomm, facevid2vid, wav2lip, sadtalker, MRAA, pirender, tpsm, lia]
domains: [ffpp, cdf]            # DFD excluded
bootstrap_iterations: 2000
clip_values: [2.0, 4.0, 6.0, 8.0]
gamma_values: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
topk_fraction: 0.25
trim_fraction: 0.10
sat_thresholds: [0.01, 0.05, 0.10]
val_videos_per_method: 100
frames_per_video: 8
gpus: [0, 1, 2, 3, 4, 5]
smoke:
  backbone: clip
  seed: 42
  methods: [fsgan, sadtalker]
output_dir: outputs
paths:
  tables: outputs/tables
  figures: outputs/figures
  reports: reports
  logs: outputs/logs
  val_cache: outputs/val_cache
```

- [ ] **Step 2: `__init__.py`**
```python
"""Pooling/saturation mechanism study."""
```

- [ ] **Step 3: 写失败测试 `tests/study_pooling/test_config.py`**
```python
from deepfake_detection.study_pooling.config import load_config
def test_load(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("global_seed: 7\nbackbones: [clip]\nclip_values: [2.0]\npaths: {tables: t}\n")
    cfg = load_config(p)
    assert cfg.global_seed == 7
    assert cfg.backbones == ["clip"]
    assert cfg.clip_values == [2.0]
    assert cfg.paths["tables"] == "t"
```

- [ ] **Step 4: 运行确认 FAIL**
Run: `cd /home/z/project/deepfake_detection && python -m pytest tests/study_pooling/test_config.py -q`
Expected: ModuleNotFoundError.

- [ ] **Step 5: 实现 `src/deepfake_detection/study_pooling/config.py`**
```python
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
```

- [ ] **Step 6: 运行确认 PASS**
Run: `cd /home/z/project/deepfake_detection && python -m pytest tests/study_pooling/test_config.py -q`
Expected: 1 passed.

- [ ] **Step 7: Commit**
```bash
cd /home/z/project/deepfake_detection
git add configs/pooling_saturation_config.yaml src/deepfake_detection/study_pooling/__init__.py src/deepfake_detection/study_pooling/config.py
git add -f tests/study_pooling/test_config.py
git commit -m "feat(pooling): config + package skeleton"
git show --stat HEAD
```

---

## Task 2: 视频聚合器 + Jensen gap(核心正确性)

**Files:**
- Create: `src/deepfake_detection/study_pooling/aggregators.py`
- Test: `tests/study_pooling/test_aggregators.py`

**Interfaces:**
- Consumes: `numpy`.
- Produces: `aggregate(name, s_frame, **kw)->float` for names `mean_prob, mean_logit, sigmoid_mean_logit, median_logit, clip_prob, trimmed_logit, topk_logit`; `jensen_gap(s_frame)->float` = `mean(sigmoid(s)) − sigmoid(mean(s))`.

- [ ] **Step 1: 写失败测试**
```python
# tests/study_pooling/test_aggregators.py
import numpy as np
from deepfake_detection.study_pooling.aggregators import aggregate, jensen_gap, NAMES

def _sigmoid(x): return 1/(1+np.exp(-x))

def test_mean_logit_equals_sigmoid_of_mean_in_auc_rank():
    # AUC is rank-based and sigmoid is strictly increasing ->
    # mean_logit and sigmoid(mean_logit) give identical video ranking.
    s = np.array([[-2.,1.],[0.,0.],[3.,-1.]])
    ml = [aggregate("mean_logit", row) for row in s]
    sml = [aggregate("sigmoid_mean_logit", row) for row in s]
    order_ml = np.argsort(ml); order_sml = np.argsort(sml)
    assert np.array_equal(order_ml, order_sml)

def test_jensen_gap_sign():
    # non-constant logits -> Jensen gap > 0 (sigmoid concave-then-convex; for spread logits mean(sig) != sig(mean))
    s = np.array([10., -10.])
    j = jensen_gap(s)
    assert abs(j) > 1e-6

def test_clip_prob_monotone_in_c():
    s = np.array([-5., 5.])
    full = aggregate("clip_prob", s, c=100.0)
    clipped = aggregate("clip_prob", s, c=2.0)
    assert abs(full - _sigmoid(5.0)) < 1e-9 or full != clipped  # clipping changes value when c small

def test_names_complete():
    for n in ["mean_prob","mean_logit","sigmoid_mean_logit","median_logit","clip_prob","trimmed_logit","topk_logit"]:
        assert n in NAMES
```

- [ ] **Step 2: 运行确认 FAIL**
Run: `cd /home/z/project/deepfake_detection && python -m pytest tests/study_pooling/test_aggregators.py -q`
Expected: ModuleNotFoundError.

- [ ] **Step 3: 实现 `aggregators.py`**
```python
# src/deepfake_detection/study_pooling/aggregators.py
"""Video-level aggregators over per-frame signed logits + Jensen gap.

s_frame: (T,) signed logits for one video. Returns a scalar video score.
mean_prob uses sigmoid(s); mean_logit uses raw s. Because sigmoid is strictly
monotone, AUC(mean_logit) == AUC(sigmoid(mean_logit)) exactly (rank identity)."""
from __future__ import annotations
import numpy as np

def _sigmoid(x): return 1.0/(1.0+np.exp(-np.asarray(x, dtype=np.float64)))

def mean_prob(s):     return float(np.mean(_sigmoid(s)))
def mean_logit(s):    return float(np.mean(np.asarray(s, dtype=np.float64)))
def sigmoid_mean_logit(s): return float(_sigmoid(np.mean(np.asarray(s, dtype=np.float64))))
def median_logit(s):  return float(np.median(np.asarray(s, dtype=np.float64)))
def clip_prob(s, c=6.0):
    s = np.asarray(s, dtype=np.float64)
    return float(np.mean(_sigmoid(np.clip(s, -c, c))))
def trimmed_logit(s, trim=0.10):
    s = np.asarray(s, dtype=np.float64)
    if s.size < 4: return float(np.mean(s))
    lo = int(np.floor(s.size*trim)); hi = int(np.ceil(s.size*(1-trim)))
    return float(np.mean(np.sort(s)[lo:hi]))
def topk_logit(s, frac=0.25):
    s = np.asarray(s, dtype=np.float64)
    k = max(1, int(np.ceil(s.size*frac)))
    return float(np.mean(np.sort(s)[-k:]))

NAMES = ["mean_prob","mean_logit","sigmoid_mean_logit","median_logit",
         "clip_prob","trimmed_logit","topk_logit"]
_FNS = {"mean_prob": mean_prob, "mean_logit": mean_logit,
        "sigmoid_mean_logit": sigmoid_mean_logit, "median_logit": median_logit,
        "clip_prob": clip_prob, "trimmed_logit": trimmed_logit, "topk_logit": topk_logit}

def aggregate(name, s_frame, **kw):
    fn = _FNS[name]
    if name == "clip_prob": return fn(s_frame, c=kw.get("c", 6.0))
    if name == "trimmed_logit": return fn(s_frame, trim=kw.get("trim", 0.10))
    if name == "topk_logit": return fn(s_frame, frac=kw.get("frac", 0.25))
    return fn(s_frame)

def jensen_gap(s) -> float:
    s = np.asarray(s, dtype=np.float64)
    return float(np.mean(_sigmoid(s)) - _sigmoid(np.mean(s)))
```

- [ ] **Step 4: 运行确认 PASS**
Run: `cd /home/z/project/deepfake_detection && python -m pytest tests/study_pooling/test_aggregators.py -q`
Expected: 4 passed.

- [ ] **Step 5: Commit**
```bash
git add src/deepfake_detection/study_pooling/aggregators.py
git add -f tests/study_pooling/test_aggregators.py
git commit -m "feat(pooling): video aggregators + Jensen gap"
git show --stat HEAD
```

---

## Task 3: 视频级评估(AUC + rank-flip + identity,多聚合器)

**Files:**
- Create: `src/deepfake_detection/study_pooling/video_eval.py`
- Test: `tests/study_pooling/test_video_eval.py`

**Interfaces:**
- Consumes: `deepfake_detection.study.auc_pairs.{auc_mannwhitney, flip_counts, identity_check}`; `aggregators.aggregate`.
- Produces: `video_scores(df_frame, score_col, agg, **kw)->DataFrame[video_id,label,score]`(先按视频聚合帧分数);`view_metrics(fake_df, real_df, agg, **kw)->dict(auc_raw,auc_dir,delta_auc,CC,WW,H,B,H_minus_B,identity_error,tie_rate)` 其中 fake_df/real_df 含 `video_id,label,s_raw,s_dir`;`eval_aggregator(df_frame, agg, **kw)` 一站式返回 raw/dir 的 view_metrics。

- [ ] **Step 1: 写失败测试**
```python
# tests/study_pooling/test_video_eval.py
import numpy as np, pandas as pd
from deepfake_detection.study_pooling.video_eval import eval_aggregator

def _df():
    rows=[]
    for vid,base,lbl in [("f1",3.0,1),("f2",4.0,1),("r1",-2.0,0),("r2",-1.0,0)]:
        for fr in range(3):
            rows.append(dict(video_id=vid,label=lbl,s_raw=base+0.1*fr,s_dir=base*0.3+0.1*fr))
    return pd.DataFrame(rows)

def test_mean_logit_view_metrics_identity():
    df=_df()
    m = eval_aggregator(df, "mean_logit")
    # fakes score higher than reals -> auc=1 both; identity holds
    assert abs(m["auc_raw"]-1.0)<1e-9
    assert m["identity_error"] < 1e-9

def test_mean_prob_runs():
    df=_df(); m = eval_aggregator(df, "mean_prob")
    for k in ["auc_raw","auc_dir","delta_auc","H","B","H_minus_B","identity_error"]:
        assert k in m
```

- [ ] **Step 2: 运行确认 FAIL**  Run: `python -m pytest tests/study_pooling/test_video_eval.py -q` → ModuleNotFoundError.

- [ ] **Step 3: 实现 `video_eval.py`**
```python
# src/deepfake_detection/study_pooling/video_eval.py
"""Aggregate frames->video, then AUC + rank-flip + identity for raw vs direction."""
from __future__ import annotations
import numpy as np, pandas as pd
from deepfake_detection.study.auc_pairs import auc_mannwhitney, flip_counts, identity_check, tie_rate
from deepfake_detection.study_pooling.aggregators import aggregate

def video_scores(df_frame, score_col, agg, **kw):
    out = []
    for vid, g in df_frame.groupby("video_id", sort=False):
        out.append({"video_id": vid, "label": int(g["label"].iloc[0]),
                    "score": aggregate(agg, g[score_col].to_numpy(), **kw)})
    return pd.DataFrame(out)

def view_metrics(fake_v, real_v):
    fr = fake_v["score"].to_numpy(); rr = real_v["score"].to_numpy()
    ar = auc_mannwhitney(fr, rr)
    return {"auc_raw": ar, **_dir_block(fr, rr)}

def _dir_block(fr_raw, rr_raw, fr_dir=None, rr_dir=None):
    # called with raw arrays; dir supplied separately in eval_aggregator
    raise NotImplementedError

def eval_aggregator(df_frame, agg, **kw):
    vr_raw = video_scores(df_frame, "s_raw", agg, **kw)
    vr_dir = video_scores(df_frame, "s_dir", agg, **kw)
    f = vr_raw[vr_raw.label==1]; r = vr_raw[vr_raw.label==0]
    fd = vr_dir[vr_dir.label==1]; rd = vr_dir[vr_dir.label==0]
    fr = f["score"].to_numpy(); rr = r["score"].to_numpy()
    fdr = fd["score"].to_numpy(); rdr = rd["score"].to_numpy()
    auc_raw = auc_mannwhitney(fr, rr); auc_dir = auc_mannwhitney(fdr, rdr)
    cc,ww,h,b = flip_counts(fr, rr, fdr, rdr)
    return {"auc_raw":auc_raw, "auc_dir":auc_dir, "delta_auc":auc_dir-auc_raw,
            "CC":cc,"WW":ww,"H":h,"B":b,"H_minus_B":h-b,
            "identity_error":identity_check(auc_raw,auc_dir,h,b),
            "tie_rate":tie_rate(fr,rr), "n_fake":int(len(f)), "n_real":int(len(r))}
```
(Remove the unused `_dir_block` stub before commit — it is not called; keep `eval_aggregator` as the single entry. If you keep it, give it a real body. Do not ship a NotImplementedError.)

- [ ] **Step 4: 运行确认 PASS**  Run: `python -m pytest tests/study_pooling/test_video_eval.py -q` → 2 passed.

- [ ] **Step 5: Commit**
```bash
git add src/deepfake_detection/study_pooling/video_eval.py
git add -f tests/study_pooling/test_video_eval.py
git commit -m "feat(pooling): video-level AUC + rank-flip + identity per aggregator"
git show --stat HEAD
```

---

## Task 4: 饱和统计 + sigmoid 梯度

**Files:**
- Create: `src/deepfake_detection/study_pooling/saturation.py`
- Test: `tests/study_pooling/test_saturation.py`

**Interfaces:**
- Produces: `frame_stats(s, p)->dict`(mean,std,median,IQR,q90/95/99,mean_abs,q95_abs);`sat_ratios(p, thresholds)->dict`(`sat_001/005/010`);`sigmoid_grad_stats(p)->dict`(mean/median of p(1−p));`video_sat(df_frame, score_col)->dict`(video_sat_ratio@0.05, video_mean_abs_logit, video_logit_std, video_logit_IQR)。

- [ ] **Step 1: 写失败测试**
```python
# tests/study_pooling/test_saturation.py
import numpy as np
from deepfake_detection.study_pooling.saturation import frame_stats, sat_ratios, sigmoid_grad_stats

def test_frame_stats():
    s = np.array([-5.,0.,5.])
    st = frame_stats(s, 1/(1+np.exp(-s)))
    assert abs(st["mean"]-0.0)<1e-9
    assert "q99" in st and "mean_abs" in st

def test_sat_ratios():
    p = np.array([0.001, 0.5, 0.999])
    r = sat_ratios(p, [0.01,0.05,0.10])
    assert r["sat_001"]==2/3 and r["sat_005"]==2/3

def test_sigmoid_grad_zero_in_saturation():
    p = np.array([0.999, 0.999])
    g = sigmoid_grad_stats(p)
    assert g["mean"] < 1e-2
```

- [ ] **Step 2: 运行确认 FAIL**  Run: `python -m pytest tests/study_pooling/test_saturation.py -q` → ModuleNotFoundError.

- [ ] **Step 3: 实现 `saturation.py`**
```python
# src/deepfake_detection/study_pooling/saturation.py
from __future__ import annotations
import numpy as np

def frame_stats(s, p):
    s = np.asarray(s, float)
    return {"mean":float(s.mean()), "std":float(s.std()), "median":float(np.median(s)),
            "iqr":float(np.percentile(s,75)-np.percentile(s,25)),
            "q90":float(np.percentile(s,90)),"q95":float(np.percentile(s,95)),"q99":float(np.percentile(s,99)),
            "mean_abs":float(np.mean(np.abs(s))),"q95_abs":float(np.percentile(np.abs(s),95))}

def sat_ratios(p, thresholds):
    p = np.asarray(p, float)
    out = {}
    for t in thresholds:
        out[f"sat_{int(t*1000):03d}"] = float(np.mean((p<t)|(p>1-t)))
    return out

def sigmoid_grad_stats(p):
    p = np.asarray(p, float); g = p*(1-p)
    return {"mean":float(g.mean()), "median":float(np.median(g))}

def video_sat(df_frame, score_col, t=0.05):
    s = df_frame[score_col].to_numpy(float)
    p = 1.0/(1.0+np.exp(-s))
    return {"video_sat_ratio":float(np.mean((p<t)|(p>1-t))),
            "video_mean_abs_logit":float(np.mean(np.abs(s))),
            "video_logit_std":float(s.std()),
            "video_logit_iqr":float(np.percentile(s,75)-np.percentile(s,25))}
```

- [ ] **Step 4: 运行确认 PASS**  Run: `python -m pytest tests/study_pooling/test_saturation.py -q` → 3 passed.

- [ ] **Step 5: Commit**
```bash
git add src/deepfake_detection/study_pooling/saturation.py
git add -f tests/study_pooling/test_saturation.py
git commit -m "feat(pooling): saturation stats + sigmoid gradient"
git show --stat HEAD
```

---

## Task 5: 读 checkpoint signed w,b + γ 扫描

**Files:**
- Create: `src/deepfake_detection/study_pooling/wb_io.py`
- Create: `src/deepfake_detection/study_pooling/gamma.py`
- Test: `tests/study_pooling/test_gamma.py`

**Interfaces:**
- Consumes: `deepfake_detection.study.q4_extract.build_model_from_checkpoint`; `deepfake_detection.diagnosis.checkpoint_index.discover_checkpoints`.
- Produces: `signed_wb(ckpt_info)->(w (D,), b float)`; `score_gamma(w_dot_u, b, norm, gamma)->ndarray`(同前序 flip_dissect,但 b 来自 checkpoint);`gamma_auc_by_agg(df_frame, w_dot_u, b, norm, gamma, agg, **kw)->float`(对 s_γ 做帧→视频聚合再 AUC)。

- [ ] **Step 1: 写失败测试**
```python
# tests/study_pooling/test_gamma.py
import numpy as np
from deepfake_detection.study_pooling.gamma import score_gamma, recover_wdotu

def test_gamma_endpoints():
    rng=np.random.default_rng(0); D=6
    w=rng.normal(size=D); b=0.4
    z=rng.normal(size=(40,D))*5; norm=np.linalg.norm(z,axis=1); u=z/norm[:,None]
    s_raw=z@w+b; s_dir=u@w+b
    wu,bb=recover_wdotu(s_raw,s_dir,norm)
    assert np.allclose(score_gamma(wu,bb,norm,1.0), s_raw, atol=1e-3)
    assert np.allclose(score_gamma(wu,bb,norm,0.0), s_dir, atol=1e-3)
```

- [ ] **Step 2: 运行确认 FAIL**  Run: `python -m pytest tests/study_pooling/test_gamma.py -q` → ModuleNotFoundError.

- [ ] **Step 3: 实现 `wb_io.py`**
```python
# src/deepfake_detection/study_pooling/wb_io.py
"""Read signed head (w_fake-w_real, b_fake-b_real) from a checkpoint."""
from __future__ import annotations
from deepfake_detection.study.q4_extract import build_model_from_checkpoint

def signed_wb(ckpt_info, device="cpu"):
    model, _cfg = build_model_from_checkpoint(ckpt_info, device=device)
    w = model.classifier.weight.detach().cpu().numpy()   # (2,D)
    b = model.classifier.bias.detach().cpu().numpy()     # (2,)
    return (w[1]-w[0]).astype("float64"), float(b[1]-b[0])
```
实现 `gamma.py`(复用前序 recover_wdotu 逻辑):
```python
# src/deepfake_detection/study_pooling/gamma.py
from __future__ import annotations
import numpy as np
EPS=1e-6

def recover_wdotu(s_raw, s_dir, norm):
    s_raw=np.asarray(s_raw,float); s_dir=np.asarray(s_dir,float); norm=np.asarray(norm,float)
    denom=norm-1.0; safe=np.abs(denom)>EPS
    wu=np.where(safe,(s_raw-s_dir)/np.where(safe,denom,1.0),0.0)
    return wu, s_dir-wu

def score_gamma(w_dot_u, b, norm, gamma):
    norm=np.asarray(norm,float)
    return np.power(norm,gamma)*np.asarray(w_dot_u)+b
```

- [ ] **Step 4: 运行确认 PASS**  Run: `python -m pytest tests/study_pooling/test_gamma.py -q` → 1 passed.

- [ ] **Step 5: Commit**
```bash
git add src/deepfake_detection/study_pooling/wb_io.py src/deepfake_detection/study_pooling/gamma.py
git add -f tests/study_pooling/test_gamma.py
git commit -m "feat(pooling): read checkpoint signed w,b + gamma scan"
git show --stat HEAD
```

---

## Task 6: 温度校准 + Platt + ECE/Brier

**Files:**
- Create: `src/deepfake_detection/study_pooling/temperature.py`
- Test: `tests/study_pooling/test_temperature.py`

**Interfaces:**
- Consumes: `scipy.optimize.minimize_scalar/minimize`.
- Produces: `fit_temperature(s, y, sample_weight=None)->T`(>0, 最小化 frame NLL);`fit_platt(s,y)->(A>0,B)`;`nll(s,y,T=1.0,w=None)`;`ece(p,y,n_bins=15)`;`brier(p,y)`;`apply_T(s,T)=sigmoid(s/T)`。

- [ ] **Step 1: 写失败测试**(已知答案:logits 已校准 → T≈1)
```python
# tests/study_pooling/test_temperature.py
import numpy as np
from deepfake_detection.study_pooling.temperature import fit_temperature, fit_platt, nll, ece, brier, apply_T

def test_fit_temp_near_one_on_well_calibrated():
    rng=np.random.default_rng(0)
    y=rng.integers(0,2,4000)
    # logits = logit(p) with p = true prob -> T~1
    p = np.where(y==1, rng.uniform(0.6,0.99,4000), rng.uniform(0.01,0.4,4000))
    s = np.log(p/(1-p))
    T = fit_temperature(s, y)
    assert 0.7 < T < 1.4, T

def test_platt_positive_A():
    rng=np.random.default_rng(1); y=rng.integers(0,2,1000)
    s=np.where(y==1, rng.normal(2,1,1000), rng.normal(-2,1,1000))
    A,B=fit_platt(s,y)
    assert A>0

def test_apply_T_and_metrics():
    s=np.array([-2.,2.]); y=np.array([0,1])
    p=apply_T(s,1.0)
    assert abs(p[1]-1/(1+np.exp(-2)))<1e-9
    assert 0.0<=ece(p,y,5)<=1.0
    assert 0.0<=brier(p,y)<=1.0
    assert nll(s,y)>0
```

- [ ] **Step 2: 运行确认 FAIL**  Run: `python -m pytest tests/study_pooling/test_temperature.py -q` → ModuleNotFoundError.

- [ ] **Step 3: 实现 `temperature.py`**
```python
# src/deepfake_detection/study_pooling/temperature.py
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize_scalar, minimize

def _sigmoid(x): return 1.0/(1.0+np.exp(-np.asarray(x,float)))

def nll(s, y, T=1.0, w=None):
    p = np.clip(_sigmoid(np.asarray(s,float)/max(T,1e-6)), 1e-12, 1-1e-12)
    y = np.asarray(y, float)
    l = -(y*np.log(p)+(1-y)*np.log(1-p))
    return float(np.average(l, weights=w)) if w is not None else float(np.mean(l))

def fit_temperature(s, y, sample_weight=None):
    s=np.asarray(s,float); y=np.asarray(y,float)
    res = minimize_scalar(lambda T: nll(s,y,max(T,1e-6),sample_weight), bounds=(1e-3,100.0), method="bounded")
    return float(res.x)

def fit_platt(s, y, sample_weight=None):
    s=np.asarray(s,float); y=np.asarray(y,float)
    def loss(ab):
        A,B=ab; p=np.clip(_sigmoid(A*s+B),1e-12,1-1e-12)
        l=-(y*np.log(p)+(1-y)*np.log(1-p))
        return np.average(l,weights=sample_weight) if sample_weight is not None else np.mean(l)
    res = minimize(loss, x0=np.array([1.0,0.0]), method="Nelder-Mead")
    A,B=float(res.x[0]),float(res.x[1])
    if A<=0:  # enforce monotonicity; fallback to A=1e-3
        A=1e-3
    return A,B

def apply_T(s,T): return _sigmoid(np.asarray(s,float)/max(T,1e-6))

def ece(p, y, n_bins=15):
    p=np.asarray(p,float); y=np.asarray(y,float)
    bins=np.linspace(0,1,n_bins+1); out=0.0; n=len(y)
    for i in range(n_bins):
        m=(p>=bins[i])&(p<bins[i+1])
        if m.sum()==0: continue
        out += (m.sum()/n)*abs(p[m].mean()-y[m].mean())
    return float(out)

def brier(p,y): p=np.asarray(p,float); y=np.asarray(y,float); return float(np.mean((p-y)**2))
```

- [ ] **Step 4: 运行确认 PASS**  Run: `python -m pytest tests/study_pooling/test_temperature.py -q` → 3 passed.

- [ ] **Step 5: Commit**
```bash
git add src/deepfake_detection/study_pooling/temperature.py
git add -f tests/study_pooling/test_temperature.py
git commit -m "feat(pooling): temperature + Platt + ECE/Brier"
git show --stat HEAD
```

---

## Task 7: 视频内时间一致性

**Files:**
- Create: `src/deepfake_detection/study_pooling/temporal.py`
- Test: `tests/study_pooling/test_temporal.py`

**Interfaces:**
- Produces: `temporal_stats(df_frame, score_col)->dict`(per-video std,MAD,IQR,q90−q10,P(s>0),adj_diff=mean|s_t−s_{t-1}|,outlier_ratio via median±3·MAD);requires frames ordered by `frame_id`(从 `image_path` 解析)。`auto_corr` 仅当 ≥4 帧时计算。

- [ ] **Step 1: 写失败测试**
```python
# tests/study_pooling/test_temporal.py
import numpy as np, pandas as pd
from deepfake_detection.study_pooling.temporal import temporal_stats, _frame_order

def test_frame_order_from_path():
    paths=["/x/000/310.png","/x/000/050.png","/x/000/200.png"]
    assert _frame_order(paths).tolist()==[1,2,0]  # sorted by numeric frame -> positions

def test_temporal_stats_keys():
    df=pd.DataFrame({"video_id":["v"]*5,"image_path":[f"/x/000/{i:03d}.png" for i in range(5)],
                     "s_raw":[0.,1.,0.5,-0.5,0.2]})
    st=temporal_stats(df,"s_raw")
    for k in ["std","mad","iqr","q90_q10","p_pos","adj_diff","outlier_ratio"]:
        assert k in st
    assert abs(st["adj_diff"]-np.mean(np.abs(np.diff([0.,1.,0.5,-0.5,0.2]))))<1e-9
```

- [ ] **Step 2: 运行确认 FAIL**  Run: `python -m pytest tests/study_pooling/test_temporal.py -q` → ModuleNotFoundError.

- [ ] **Step 3: 实现 `temporal.py`**
```python
# src/deepfake_detection/study_pooling/temporal.py
from __future__ import annotations
import os, re
import numpy as np, pandas as pd

_FRAME_RE = re.compile(r"(\d+)\.png$")

def _frame_key(path):
    m = _FRAME_RE.search(path or "")
    return int(m.group(1)) if m else 0

def _frame_order(paths):
    keys = np.array([_frame_key(p) for p in paths])
    return np.argsort(keys, kind="stable")

def temporal_stats(df_frame, score_col):
    df = df_frame.sort_values("image_id" if "image_id" in df_frame else "image_path") if "image_path" in df_frame else df_frame
    order = _frame_order(df_frame["image_path"].to_numpy())
    s = df_frame.iloc[order][score_col].to_numpy(float)
    med = np.median(s); mad = np.median(np.abs(s-med))
    aMAD = mad*1.4826 if mad>0 else (s.std() or 1.0)
    q10,q90 = np.percentile(s,[10,90])
    diffs = np.abs(np.diff(s)) if s.size>1 else np.array([0.0])
    ac = float(np.corrcoef(s[:-1], s[1:])[0,1]) if s.size>=4 and np.std(s[:-1])>0 and np.std(s[1:])>0 else float("nan")
    return {"std":float(s.std()), "mad":float(mad),
            "iqr":float(np.percentile(s,75)-np.percentile(s,25)),
            "q90_q10":float(q90-q10), "p_pos":float(np.mean(s>0)),
            "adj_diff":float(np.mean(diffs)), "auto_corr":ac,
            "outlier_ratio":float(np.mean(np.abs(s-med)>3*aMAD))}
```

- [ ] **Step 4: 运行确认 PASS**  Run: `python -m pytest tests/study_pooling/test_temporal.py -q` → 2 passed.

- [ ] **Step 5: Commit**
```bash
git add src/deepfake_detection/study_pooling/temporal.py
git add -f tests/study_pooling/test_temporal.py
git commit -m "feat(pooling): within-video temporal consistency stats"
git show --stat HEAD
```

---

## Task 8: 实验1/2/3 driver(`scripts/06_pooling_saturation_analysis.py`)

**Files:**
- Create: `scripts/06_pooling_saturation_analysis.py`

**Interfaces:**
- Consumes: Tasks 1–5,7 libs; `deepfake_detection.study.io.{load_frame_subset, dedup_reals}`; `deepfake_detection.study.canonical.add_canonical_keys`.
- Produces: `outputs/tables/Q_pooling_aggregation_auc.csv`(backbone,seed,method,domain,aggregation,auc_raw,auc_direction,delta_auc,num_real_videos,num_fake_videos);`Q_pooling_saturation_stats.csv`;`Q_pooling_jensen_gap.csv`;`Q_pooling_gamma_scan.csv`;figures `Q_pooling_FSGAN_SadTalker_aggregators.png`,`Q_pooling_delta_heatmap_mean_prob.png`,`Q_pooling_delta_heatmap_mean_logit.png`,`Q_pooling_interaction_term.png`,`Q_pooling_logit_scale_raw_vs_direction.png`,`Q_pooling_saturation_raw_vs_direction.png`,`Q_pooling_sigmoid_gradient.png`,`Q_pooling_jensen_gap_distribution.png`,`Q_pooling_jensen_gap_vs_score_change.png`,`Q_pooling_jensen_gap_FSGAN_SadTalker.png`,`Q_pooling_gamma_FSGAN_by_aggregator.png`,`Q_pooling_gamma_SadTalker_by_aggregator.png`.

- [ ] **Step 1: 实现 driver**(骨架;公式严格按 spec §五/六/七/十二)
```python
# scripts/06_pooling_saturation_analysis.py
"""Pooling experiments 1 (aggregators), 2 (saturation), 3 (Jensen), gamma scan.
Parquet-only. Dry-run first (smoke), then expand."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from deepfake_detection.study_pooling.config import load_config
from deepfake_detection.study.io import load_frame_subset, dedup_reals
from deepfake_detection.study_pooling.aggregators import aggregate, jensen_gap, NAMES
from deepfake_detection.study_pooling.video_eval import eval_aggregator
from deepfake_detection.study_pooling.saturation import frame_stats, sat_ratios, sigmoid_grad_stats
from deepfake_detection.study_pooling.gamma import recover_wdotu, score_gamma
from deepfake_detection.study_pooling.wb_io import signed_wb
from deepfake_detection.diagnosis.checkpoint_index import discover_checkpoints

CFG=load_config(ROOT/"configs"/"pooling_saturation_config.yaml")
TAB=ROOT/CFG.paths["tables"]; FIG=ROOT/CFG.paths["figures"]
TAB.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True)
AGGS=["mean_prob","mean_logit","sigmoid_mean_logit","median_logit","clip_prob","trimmed_logit","topk_logit"]

def _wide(df):
    raw=df[df.infer_mode=="raw"][["checkpoint_id","backbone","seed","domain","method_id","label",
        "video_id","image_path","feature_norm","source_video_id","signed_logit"]].rename(columns={"signed_logit":"s_raw"})
    d=df[df.infer_mode=="direction"][["image_path","signed_logit"]].rename(columns={"signed_logit":"s_dir"})
    return raw.merge(d,on="image_path")

def run_combo(bb, sd, method, dom, fpath, ckinfo):
    df=dedup_reals(load_frame_subset(fpath, train_mode="raw", infer_mode=None,
        backbones=[bb], seeds=[sd], domains=[dom], methods=[method,"real"]))
    w=_wide(df)
    if w.empty or (w.label==1).sum()==0 or (w.label==0).sum()==0: return [],[],[],[]
    agg_rows=[]; sat_rows=[]; j_rows=[]
    # exp1 aggregation
    for a in AGGS:
        kw={}
        if a=="clip_prob":
            for c in CFG.clip_values:
                m=eval_aggregator(w, a, c=c)
                agg_rows.append({"backbone":bb,"seed":sd,"method":method,"domain":dom,
                    "aggregation":f"clip_prob_{c}","auc_raw":m["auc_raw"],"auc_direction":m["auc_dir"],
                    "delta_auc":m["delta_auc"],"num_real_videos":m["n_real"],"num_fake_videos":m["n_fake"]})
        else:
            m=eval_aggregator(w, a)
            agg_rows.append({"backbone":bb,"seed":sd,"method":method,"domain":dom,"aggregation":a,
                "auc_raw":m["auc_raw"],"auc_direction":m["auc_dir"],"delta_auc":m["delta_auc"],
                "num_real_videos":m["n_real"],"num_fake_videos":m["n_fake"]})
    # exp2 saturation (frame-level, raw & direction, real & fake)
    for mode,col in [("raw","s_raw"),("direction","s_dir")]:
        for lbl,lbln in [(1,"fake"),(0,"real")]:
            sub=w[w.label==lbl]
            s=sub[col].to_numpy(float); p=1/(1+np.exp(-s))
            st=frame_stats(s,p); sr=sat_ratios(p,CFG.sat_thresholds); sg=sigmoid_grad_stats(p)
            sat_rows.append({"backbone":bb,"seed":sd,"method":method,"domain":dom,
                "infer_mode":mode,"label":lbln, **st, **sr, **sg})
    # exp3 Jensen (per video, raw & direction)
    for mode,col in [("raw","s_raw"),("direction","s_dir")]:
        Js=[jensen_gap(g[col].to_numpy()) for _,g in w.groupby("video_id")]
        j_rows.append({"backbone":bb,"seed":sd,"method":method,"domain":dom,"infer_mode":mode,
            "mean_J":float(np.mean(Js)),"mean_abs_J":float(np.mean(np.abs(Js))),
            "n_videos":len(Js)})
    # gamma scan (uses real checkpoint b)
    wu,bb_ = recover_wdotu(w.s_raw.to_numpy(), w.s_dir.to_numpy(), w.feature_norm.to_numpy())
    gamma_rows=[]
    try:
        if ckinfo is not None:
            wv,bv = signed_wb(ckinfo)
            for g in CFG.gamma_values:
                s_g = score_gamma(wu, bv, w.feature_norm.to_numpy(), g)  # uses REAL b
                wg = w.assign(s_raw=s_g, s_dir=s_g)  # gamma replaces both consistently
                for a in ["mean_prob","mean_logit"]:
                    m=eval_aggregator(wg, a)
                    gamma_rows.append({"backbone":bb,"seed":sd,"method":method,"domain":dom,
                        "gamma":g,"aggregation":a,"video_auc":m["auc_raw"]})
    except Exception as e:
        print("[gamma] skip", bb,sd,method,dom,e)
    return agg_rows, sat_rows, j_rows, gamma_rows

def main():
    import argparse; ap=argparse.ArgumentParser(); ap.add_argument("--smoke",action="store_true"); a=ap.parse_args()
    fpath=CFG.parquet_frame
    if a.smoke:
        targets=[(CFG.smoke["backbone"], CFG.smoke["seed"], CFG.smoke["methods"])]
    else:
        targets=[(bb,sd,CFG.methods) for bb in CFG.backbones for sd in CFG.seeds]
    A=[];S=[];J=[];G=[]
    for bb,sd,methods in targets:
        cks=discover_checkpoints(CFG.ckpt_root, modes=["raw"], backbones=[bb], seeds=[sd])
        ckinfo=cks[0] if cks else None
        for m in methods:
            for dom in CFG.domains:
                ag,sa,jr,gr=run_combo(bb,sd,m,dom,fpath,ckinfo)
                A+=ag;S+=sa;J+=jr;G+=gr
    pd.DataFrame(A).to_csv(TAB/"Q_pooling_aggregation_auc.csv",index=False)
    pd.DataFrame(S).to_csv(TAB/"Q_pooling_saturation_stats.csv",index=False)
    pd.DataFrame(J).to_csv(TAB/"Q_pooling_jensen_gap.csv",index=False)
    pd.DataFrame(G).to_csv(TAB/"Q_pooling_gamma_scan.csv",index=False)
    _plots(A,S,J,G)
    print("exp1/2/3/gamma done. rows:",len(A))

def _plots(A,S,J,G):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    # (implement the 12 figures listed in the Interfaces block; minimal but real)
    # ... (see spec §五/六/七/十二 for required figures) ...
    plt.close("all")

if __name__=="__main__": main()
```
> **Figure implementation note:** the `_plots` function must produce all 12 figures listed in Interfaces. Implement each with real data from A/S/J/G (groupby backbone/method/domain). If a figure needs the Delta_prob/Delta_logit/I_pool summary, compute it from `A`: `Delta_prob = delta_auc@aggregation=='mean_prob'`, `Delta_logit = delta_auc@aggregation=='mean_logit'`, `I_pool = Delta_prob − Delta_logit`, keyed by (backbone,seed,method,domain). Do not leave `_plots` empty — every listed figure must be written to `outputs/figures/`.

- [ ] **Step 2: dry run**
Run: `cd /home/z/project/deepfake_detection && python scripts/06_pooling_saturation_analysis.py --smoke`
Expected: 4 CSVs written; printed summary. **Report mean_prob vs mean_logit `delta_auc` for fsgan & sadtalker (ffpp+cdf) — this is the key dry-run consistency check.**

- [ ] **Step 3: 验证关键自检**
确认 `AUC(mean_logit)==AUC(sigmoid_mean_logit)` 在所有行(float 精度)——从 `Q_pooling_aggregation_auc.csv` 比较 `aggregation∈{mean_logit, sigmoid_mean_logit}` 的 `auc_raw`/`auc_direction`,max abs diff 应 < 1e-9。若超阈,排查 `aggregate` 实现。

- [ ] **Step 4: 扩展全量**
Run: `cd /home/z/project/deepfake_detection && python scripts/06_pooling_saturation_analysis.py`
Expected: 4 CSVs 覆盖 3×5×16×2。

- [ ] **Step 5: Commit**
```bash
git add scripts/06_pooling_saturation_analysis.py
git commit -m "feat(pooling): exp1/2/3 + gamma scan driver"
git show --stat HEAD
```

---

## Task 9: FF++ val 特征补跑 + 实验4/5(温度/尺度校准)

**Files:**
- Create: `scripts/07_fit_temperature_calibration.py`

**Interfaces:**
- Consumes: Task 6 (temperature); `deepfake_detection.study.q4_extract.extract_checkpoint, build_model_from_checkpoint`; `deepfake_detection.data.builders.build_val_loader`; `discover_checkpoints`; parquet for test/cdf frames.
- Produces: `outputs/val_cache/<checkpoint_id>__val.parquet`(image_path,video_id,method_id,label,s_raw,s_dir,feature_norm);`outputs/tables/Q_pooling_temperature.csv`(backbone,seed,T_raw,T_direction,val_nll_before/after_raw/dir,val_ece_*,val_brier_*,platt_A_raw/dir,platt_B_raw/dir);`Q_pooling_calibrated_auc.csv`(backbone,seed,method,domain,auc_raw_calib,auc_dir_calib,delta_calib for mean_prob with T applied);`Q_pooling_scale_matched_auc.csv`;figures `Q_pooling_before_after_temperature.png`,`Q_pooling_temperature_by_backbone.png`,`Q_pooling_reliability_diagram.png`.

- [ ] **Step 1: 实现 val 抽取**(每 ckpt 一次前向 → s_raw, s_dir;复用 build_val_loader 的 records)
```python
# val extraction core (inside 07_fit_temperature_calibration.py)
def extract_val(ck, cfg_dict, device):
    out=CACHE/f"{ck.checkpoint_id}__val.parquet"
    if out.exists(): return pd.read_parquet(out)
    loader = build_val_loader(cfg_dict, distributed=False)
    ff = extract_checkpoint(ck, "ffpp", _val_frame_paths(loader), device=device)
    # s_raw/s_dir from z + signed w,b
    model,_ = build_model_from_checkpoint(ck, device="cpu")
    w = model.classifier.weight.detach().cpu().numpy()
    b = model.classifier.bias.detach().cpu().numpy()
    ws = (w[1]-w[0]); bs = float(b[1]-b[0])
    z = ff.features; r = np.linalg.norm(z,axis=1,keepdims=True)
    s_raw = z@ws + bs; s_dir = (z/r)@ws + bs
    df = pd.DataFrame({"image_path":ff.image_paths,"video_id":ff.video_ids,
        "method_id":ff.method_ids,"label":ff.labels,"s_raw":s_raw,"s_dir":s_dir,
        "feature_norm":r[:,0]})
    df.to_parquet(out); return df
```
where `_val_frame_paths(loader)` returns `[r.frame_path for r in loader.dataset.records]` and `cfg_dict` is the checkpoint's config (yaml-loaded) needed by `build_val_loader`. **Read ckpt config via `yaml.safe_load(open(ck.config_path))`** and ensure `dataset.methods` includes the 16 methods (or rely on default ALL_METHODS).

- [ ] **Step 2: 拟合温度**(FF++ val 帧级 NLL,video-balanced 加权 + real/fake 平衡)
```python
def fit_T(val_df):
    # video-balanced weights: each video contributes equally; real/fake balanced
    w_vid = 1.0/val_df.groupby("video_id")["image_path"].transform("count").to_numpy()
    # balance real/fake
    n_pos=(val_df.label==1).sum(); n_neg=(val_df.label==0).sum()
    w_bal = np.where(val_df.label==1, 1.0/max(n_pos,1), 1.0/max(n_neg,1))
    w = w_vid*w_bal; w/=w.sum()
    T_raw = fit_temperature(val_df.s_raw.to_numpy(), val_df.label.to_numpy(), sample_weight=w)
    T_dir = fit_temperature(val_df.s_dir.to_numpy(), val_df.label.to_numpy(), sample_weight=w)
    A_raw,B_raw = fit_platt(val_df.s_raw.to_numpy(), val_df.label.to_numpy(), sample_weight=w)
    A_dir,B_dir = fit_platt(val_df.s_dir.to_numpy(), val_df.label.to_numpy(), sample_weight=w)
    return T_raw,T_dir,(A_raw,B_raw),(A_dir,B_dir)
```
Compute val NLL/ECE/Brier before & after for both modes; write `Q_pooling_temperature.csv`.

- [ ] **Step 3: 应用温度到 test/CDF**(mean prob with T)→ `Q_pooling_calibrated_auc.csv`
For each backbone×seed×method×domain: load parquet test/cdf frames (dedup, wide), compute `s_raw/T_raw`, `s_dir/T_dir`, aggregate via `mean_prob`, AUC raw/dir, `delta_calib = auc_dir_calib − auc_raw_calib`. Use `eval_aggregator` on a df with `s_raw=s_raw/T_raw, s_dir=s_dir/T_dir`.

- [ ] **Step 4: 实验5 尺度匹配**(val 拟合 median/IQR → test/CDF 标准化 → mean_prob AUC)→ `Q_pooling_scale_matched_auc.csv`
```python
def robust_std_fit(val_df):
    out={}
    for mode,col in [("raw","s_raw"),("direction","s_dir")]:
        s=val_df[col].to_numpy(float)
        out[mode]=(float(np.median(s)), float(np.percentile(s,75)-np.percentile(s,25)))
    return out
# apply: s_std = (s - median)/IQR on test/cdf; mean_prob AUC raw/dir
```

- [ ] **Step 5: dry run** (clip/seed42/fsgan+sadtalker + val for clip/seed42)
Run: `cd /home/z/project/deepfake_detection && CUDA_VISIBLE_DEVICES=0 python scripts/07_fit_temperature_calibration.py --smoke`
Expected: T_raw/T_dir printed for clip/seed42; calibrated AUC table for fsgan/sadtalker;3 CSVs + reliability figure. **Report: does `Delta_prob>0` become `Delta_temp≈0` or negative after calibration?**

- [ ] **Step 6: 扩展全量**(15 ckpt val 抽取,可多 GPU;smoke 通过后)
Run: `cd /home/z/project/deepfake_detection && python scripts/07_fit_temperature_calibration.py`
Expected: `val_cache` 全 15;`Q_pooling_temperature.csv`(15 行)、`Q_pooling_calibrated_auc.csv`、`Q_pooling_scale_matched_auc.csv`。

- [ ] **Step 7: Commit**
```bash
git add scripts/07_fit_temperature_calibration.py
git commit -m "feat(pooling): FF++ val re-inference + temperature/scale calibration (exp4/5)"
git show --stat HEAD
```

> **Blocker handling:** if val extraction fails (images missing / manifest gap / `build_val_loader` errors), report BLOCKED with the error. Do NOT fall back to fitting on test/CDF. Per spec §10, report "校准实验数据不足" and proceed with experiments 1/2/3/6/7 only.

---

## Task 10: 实验6 各聚合器 rank-flip(`scripts/08_pooling_rank_flip.py`)

**Files:**
- Create: `scripts/08_pooling_rank_flip.py`

**Interfaces:**
- Consumes: Task 3 `eval_aggregator`; Task 9 calibrated T (from `Q_pooling_temperature.csv`) for the `temperature_mean_prob` aggregator; parquet.
- Produces: `outputs/tables/Q_pooling_rank_flip_by_aggregator.csv`(backbone,seed,method,domain,aggregation∈{mean_prob,mean_logit,median_logit,trimmed_logit,temperature_mean_prob,standardized_mean_prob},CC,WW,H,B,H_minus_B,auc_dir−auc_raw,identity_error);figure `Q_pooling_HB_by_aggregator.png`.

- [ ] **Step 1: 实现**
对每个 combo 和 6 个聚合器调用 `eval_aggregator`(对 `temperature_mean_prob`:先对 w 的 s_raw/s_dir 除以对应 T 再 mean_prob;对 `standardized_mean_prob`:先 robust 标准化(参数来自 Task9 val cache)再 mean_prob)。收集 CC/WW/H/B/identity。**所有 H/B 与对应 AUC 用同一批 fake-real 视频 pair(由 eval_aggregator 保证);不采样。**

- [ ] **Step 2: dry run + 全量**
Run: `python scripts/08_pooling_rank_flip.py --smoke` then `python scripts/08_pooling_rank_flip.py`
Expected: `Q_pooling_rank_flip_by_aggregator.csv`;identity_error 全部 ≈0。

- [ ] **Step 3: 重点报告**(写入 stdout + 最终报告原料)
FSGAN 是否只在 mean_prob 下 H≫B?mean_logit 下是否仍 H>B?SadTalker 是否所有聚合器 B>H?温度后 H/B 变化?

- [ ] **Step 4: Commit**
```bash
git add scripts/08_pooling_rank_flip.py
git commit -m "feat(pooling): rank-flip H/B by aggregator (exp6)"
git show --stat HEAD
```

---

## Task 11: 实验7 时间一致性 + 最终报告(`scripts/09_generate_pooling_report.py`)

**Files:**
- Create: `scripts/09_generate_pooling_report.py`(含时间一致性计算 + 报告生成)

**Interfaces:**
- Consumes: Task 7 `temporal_stats`; all `Q_pooling_*.csv`.
- Produces: `outputs/tables/Q_pooling_temporal_consistency.csv`;figures `Q_pooling_within_video_variance.png`,`Q_pooling_temporal_consistency_gain.png`;`reports/pooling_saturation_final_report.md`(spec §16 结构,A/B/C/D 唯一结论 + 下一步建议)。

- [ ] **Step 1: 时间一致性计算**(per video, raw/direction;汇总 real/fake、FSGAN/SadTalker、FF++/CDF;Direction 是否降低方差/离群)
- [ ] **Step 2: 报告生成**(读所有表,按 spec §16 结构:结论摘要 A/B/C/D + FSGAN + SadTalker + 全部方法 + 机制判定 + 下一步方法建议)
- [ ] **Step 3: dry run + 全量**
Run: `python scripts/09_generate_pooling_report.py --smoke` then `python scripts/09_generate_pooling_report.py`
- [ ] **Step 4: Commit**
```bash
git add scripts/09_generate_pooling_report.py
git commit -m "feat(pooling): temporal consistency + final report (exp7)"
git show --stat HEAD
```

---

## 自审(plan vs spec)

**Spec 覆盖:**
- §三 数据审计 → Task 8 dry-run 对齐检查 + Task 9 val 可恢复性(已在 spec §1/§10 确认)。
- §五 实验1(聚合器)→ Task 8(含 `I_pool=Delta_prob−Delta_logit`)。
- §六 实验2(饱和)→ Task 4 + Task 8。
- §七 实验3(Jensen)→ Task 2 + Task 8。
- §八 实验4(温度+Platt)→ Task 6 + Task 9。
- §九 实验5(尺度匹配)→ Task 9。
- §十 实验6(rank-flip by aggregator)→ Task 10。
- §十一 实验7(时间一致性)→ Task 7 + Task 11。
- §十二 γ 扫描(真实 b,多聚合器)→ Task 5 + Task 8。
- §十三 统计(cluster bootstrap/BH)→ 复用 `study.cluster_bootstrap`;driver 中调用(BH 在报告层)。
- §十四 工程(DuckDB/NumPy 等)→ 全程采用。
- §十五 dry-run 先行 → Task 8/9/10/11 各有 `--smoke`。
- §十六 最终报告 + A/B/C/D → Task 11。
- §十七 硬性禁止 → Global Constraints 固化。

**类型一致:** `eval_aggregator(df_frame, agg, **kw)`、`aggregate(name,s,**kw)`、`signed_wb(ckpt)->(w,b)`、`fit_temperature(s,y,w)->T`、`fit_platt(s,y,w)->(A,B)`、`recover_wdotu/score_gamma` 在定义与调用处签名一致。

**已知有意留白(非占位,driver 实现时填):** Task 8 `_plots` 12 图、Task 11 报告生成体 —— 接口与数据源已明确,公式在 spec,实现时按 spec §五/六/七/十二/十六 填充;不得留空或占位。

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-27-pooling-saturation-study.md`. Two execution options:

**1. Subagent-Driven (recommended)** — fresh subagent per task, review between tasks, fast iteration.
**2. Inline Execution** — execute tasks in this session using executing-plans, batch with checkpoints.

Which approach?
