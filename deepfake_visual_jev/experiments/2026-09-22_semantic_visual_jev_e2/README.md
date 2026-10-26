> 独立项目入口：请先阅读本项目 [README](../../README.md)。本页下方保留原实验操作记录；既有结果回放请用 `python scripts/replay_e2.py`，新输出写入 `runs/`。历史训练和评估命令中的结果路径不要用于覆盖冻结结果。

# E2 Semantic Visual-Jev validation

当前任务清单、训练曲线数值和正式评估完成情况见 [E2 实验进度](PROGRESS.md)。

正式中文结果见 [假设验证汇报](reports/e2_validation_summary_zh.md)：主域 42/42 组已完成，冻结 G1–G6 总裁决为 `not_supported`。方法限制与审计范围见 [协议审查](reports/protocol_audit_zh.md)。

本次用户汇报只覆盖 FF++／CDF 两个主域的三种子、七条件共42组。旧版正式评估在seed42无意义候选条件遇到分布式重复行的浮点去重错误，真实launcher退出码为1；旧结果和日志完整保留。修复提交 `103ab5c` 后，所有主域条件在全新 `results/evaluation_primary_v2` 重跑，同一报告不混用旧源码指纹的产物。恢复运行的五次调用真实退出码均为0，[独立审计](results/progress/audit_primary_artifacts.json)通过108项检查。

This experiment compares the frozen-vision E2-Semantic decision model with
the already trained H1-Jev fixed-candidate baseline. The visual frame indexes,
splits, transforms, and video aggregation are inherited from H1. The frozen
question/candidate bank is in `configs/language_bank.yaml`; only `train` and
`fixed` text is encoded during training. Final gate thresholds are implemented
in `src/evaluation/e2_gates.py`.

Run from this standalone project's root. GPUs 0–3 are reserved for E2;
do not use GPUs 4–7, which serve another workload.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python experiments/2026-09-22_semantic_visual_jev_e2/scripts/preflight.py \
  --config experiments/2026-09-22_semantic_visual_jev_e2/configs/semantic_jev.yaml \
  --output experiments/2026-09-22_semantic_visual_jev_e2/results/preflight.json \
  --lock-output experiments/2026-09-22_semantic_visual_jev_e2/results/execution_manifest.json
CUDA_VISIBLE_DEVICES='' pytest -q
torchrun --standalone --nproc_per_node=4 \
  experiments/2026-09-22_semantic_visual_jev_e2/scripts/train_semantic.py \
  --config experiments/2026-09-22_semantic_visual_jev_e2/configs/semantic_jev.yaml \
  --seed 42 --smoke --max-train-batches 4 --max-val-batches 2
```

The smoke checkpoint lives under `results/training/smoke` and declares
`smoke: true`, `reportable: false`. Its per-rank diagnostics must show four
train batches and two validation batches per rank, both two- and
three-candidate examples, finite losses, and peak CUDA allocation below
30 GiB. The launcher must exit zero without DataLoader worker errors or
nondeterministic-kernel warnings.

Prepare one float32 visual feature cache from the same frozen H1 tower. The
cache records a SHA256 of the visual tensors, preprocessing implementation,
feature-extraction implementation, Torch version, ordered sample records,
source-image file stats (size, nanosecond modification
and change times, inode), and feature array. Each evaluation run verifies those
fingerprints and rejects a stale cache. This detects ordinary in-place dataset
changes; it is not a cryptographic digest of all image contents. The cache
stores no language representations
and is never used for training augmentation. Some FF++ method indexes repeat
the exact same real frame: the cache stores one feature per unique physical
sample ID while retaining the full ordered index fingerprint and all original
evaluation metadata/rows. Conflicting records sharing an ID are rejected.

```bash
CUDA_VISIBLE_DEVICES=0 python \
  experiments/2026-09-22_semantic_visual_jev_e2/scripts/prepare_feature_cache.py \
  --config experiments/2026-09-22_semantic_visual_jev_e2/configs/semantic_jev.yaml \
  --output-root experiments/2026-09-22_semantic_visual_jev_e2/results/feature_cache \
  --domains ffpp cdf dfd
```

After preflight and smoke verification, train the three fixed seeds. Do not
pass `--allow-overwrite` for reportable checkpoints. Selection uses only the
seen-language validation CE.

```bash
for seed in 42 7 2048; do
  torchrun --standalone --nproc_per_node=4 \
    experiments/2026-09-22_semantic_visual_jev_e2/scripts/train_semantic.py \
    --config experiments/2026-09-22_semantic_visual_jev_e2/configs/semantic_jev.yaml \
    --seed "$seed"
done
```

The H1-Jev comparison root must contain `seed_42`, `seed_7`, and `seed_2048`
with both `ffpp_video_predictions.csv` and `cdf_video_predictions.csv` in
each. Recreate these using the unchanged H1 evaluator if necessary. Then run
all seven E2 conditions for each seed. FF++ and CDF are primary; DFD is
exploratory for T1–T3 and contributes to G3 if its permutation audit exists.
Final evaluation rejects smoke/non-reportable checkpoints and records a
per-seed `run_provenance.json`. A single-device diagnostic evaluation is
possible only with `--allow-smoke-diagnostic` and cannot enter final gate
adjudication. The run provenance also binds the evaluator's result-defining
source bundle and one fixed three-domain cache (mode, absolute root, and
manifest hashes); the final gate checks these against the live files.

本次仅两个主域的正式重跑使用相同训练检查点、配置、缓存和七条件，输出到独立目录。实际每次调用的完整 argv、环境和四rank日志另见 `results/evaluation_logs/primary_v2_20260923T051934Z/seed_{seed}/command.json`。

```bash
for seed in 42 7 2048; do
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
    experiments/2026-09-22_semantic_visual_jev_e2/scripts/evaluate_semantic.py \
    --config experiments/2026-09-22_semantic_visual_jev_e2/configs/semantic_jev.yaml \
    --seed "$seed" \
    --checkpoint "experiments/2026-09-22_semantic_visual_jev_e2/results/training/seed_${seed}/best_model.pth" \
    --domains ffpp cdf \
    --conditions fixed_h1 seen heldout permutation counterfactual meaningless_candidates meaningless_questions \
    --feature-cache-root experiments/2026-09-22_semantic_visual_jev_e2/results/feature_cache \
    --output-dir "experiments/2026-09-22_semantic_visual_jev_e2/results/evaluation_primary_v2/seed_${seed}"
done
```

主域汇报的最终判定与辅助审计命令如下；`assess_gates.py` 继续使用冻结的 G1–G6 口径，本次 G3 只核查 FF++／CDF。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python experiments/2026-09-22_semantic_visual_jev_e2/scripts/assess_gates.py \
    --config experiments/2026-09-22_semantic_visual_jev_e2/configs/semantic_jev.yaml \
    --evaluation-root experiments/2026-09-22_semantic_visual_jev_e2/results/evaluation_primary_v2 \
    --baseline-root experiments/2026-09-22_semantic_visual_jev_e2/results/baseline_h1 \
    --output-root experiments/2026-09-22_semantic_visual_jev_e2/results/summary \
    --report experiments/2026-09-22_semantic_visual_jev_e2/reports/e2_validation_result.md
CUDA_VISIBLE_DEVICES='' python experiments/2026-09-22_semantic_visual_jev_e2/results/progress/audit_final_artifacts.py \
  --primary-only --evaluation-root experiments/2026-09-22_semantic_visual_jev_e2/results/evaluation_primary_v2
CUDA_VISIBLE_DEVICES='' python experiments/2026-09-22_semantic_visual_jev_e2/results/progress/export_report_tables.py \
  --primary-only --evaluation-root experiments/2026-09-22_semantic_visual_jev_e2/results/evaluation_primary_v2
```

`supported` requires all six preregistered gates. A point-threshold failure
yields `not_supported`; a confidence-bound failure with all point estimates
passing yields `insufficient_evidence`. Brier score and per-method outputs
remain diagnostics and do not override the gates.
