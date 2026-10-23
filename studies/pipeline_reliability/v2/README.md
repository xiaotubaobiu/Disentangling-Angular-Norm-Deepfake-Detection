# Pipeline Reliability v2 execution

Binding protocol: `../docs/plans/2026-10-23-pipeline-reliability-v2.md`, identical to the user-designated Obsidian plan frozen in the artifact root. This package is separate from the completed 31-method experiment.

Artifact root: `/home/z/workspace/deepfake_pipeline_reliability_v2_20261023`.
Image root: `/Dataset/deepfake_detection/derived/pipeline_reliability_v2_20261023/H`.

Only physical GPUs 0–3 are allowed, by UUID, with one job per idle GPU. Neither the queue nor the workers contain service stop/start operations. Run and GPU lifetime locks, current memory checks and compute-process checks guard training and extraction. Protected GPUs 4–7 and Jev work are outside this study.

## Scientific protocol

There are 30 standard development models and 45 formal models: D has 10 development and 15 formal; H has 20 development and 30 formal across B/AB histories. DINO may add at most 12 fresh development models under the registered 30→45→60 rule. Development seeds 1007/1042 cannot become formal models; formal seeds are 7/42/123/2026/3407. Old fine-tuned models and old feature caches cannot enter this registry.

D reconstructs FS9/FR12 from all original associations before RGB selection and canonical merging. Shared method contributions remain in sampling mass maps. The official 360 source pairs are deterministically partitioned 306/54 for TRAIN/DEV with seed20261023. Source CAL is real-only. CDF full ranking and CDF calibration TEST remain separate cohorts.

H selects shared eight mother frames before detection. Source qualification froze a fixed640 detector input with `min(1,640/w,640/h)` letterbox scale, without upscaling. A uses the explicit margin1.2 template; B uses a1.5 bounding-box square; AB redetects A's saved PNG independently. Full preprocessing preserves failures, padding masks and coverage. Training uses B/AB common success; primary deployment tests use every B-success video. Auxiliary2×2 uses the common-success test cohort.

Training is full-parameter Raw AdamW, effective batch128, qualified microbatch32, weight decay5e-4, gradient clip1.0, first-epoch warmup and cosine ending at5% LR. FP64 gradient norm avoids overflow in the clipping operation. Scheduler time advances with attempted updates; successful/skipped updates are separately logged. Eight consecutive AMP skips halt the run for source-only numeric review; the counter survives epoch recovery. H asymmetric AMP skips require the plan's uniform FP32 fallback before target export.

Every epoch saves draw arrays, actual latent method counts, keyed augmentation values, all source DEV video scores and numeric update events. Checkpoints bind task configuration, train/dev manifests, initialization, environment, training code and the immutable route-specific qualification gate. Recovery restores optimizer/scaler/RNG, repairs history from the committed checkpoint and reconstructs source-best from checkpoint state. Formal `last` is the primary endpoint; source-best is sensitivity only.

## Execution and recovery

Use the isolated artifact environment and this directory as `PYTHONPATH`. Do not use the legacy `run.py execute`.

```bash
cd /home/z/workspace/deepfake_detector_publish_20261023/studies/pipeline_reliability/v2
PYTHONPATH=. /home/z/workspace/deepfake_pipeline_reliability_v2_20261023/environment/bin/python -m pytest tests -q
PYTHONPATH=. /home/z/workspace/deepfake_pipeline_reliability_v2_20261023/environment/bin/python -u -m reliability_v2.queue_d \
  --root /home/z/workspace/deepfake_pipeline_reliability_v2_20261023 --code "$PWD"
```

The D queue requires the immutable `qualification/D_gate.json`. It executes standard development, source-only DINO promotion, freezes all D recipes, then executes15 formal models and their FP32 caches when enabled. It stops with analysis pending. It refuses another live queue/worker; inspect `queue_status.json`, owned PIDs and `run.lock` before restarting. Do not remove live locks or launch duplicate workers. A failed worker stops the queue; other owned workers may finish, and their status must be checked before recovery.

`configs/execution.json` belongs exclusively to D. After formal recipe freeze, preserve its bytes so completed feature provenance stays stable. H must use a separate execution configuration and immutable H gate. Do not amend D's gate when approving H.

## Analysis and honest completion

`cache_d` exports only new frozen formal endpoints as rawFP32 features and originalFP32 two-class heads, one cache per role. `scoring_d` constructs the original head readout in FP64, evaluates the eight grouping/representation/aggregation cells, and exposes calibration, transition ledgers, high-precision ranking and source/collision closure bootstrap APIs. Its synthetic reports are implementation checks, not detector results.

CDF RQ4 currently supports only declared content/name-proxy isolation:80 real CAL,97 real TEST,4111 fake TEST. Complete generated ancestry is not certified; strict NP population guarantees are unavailable. DFD filename parents and manipulated-face selection also remain unverified. Failure and negative outcomes cannot justify removing controls, changing TEST, selecting thresholds on TEST, or expanding the search.

Completion requires all registered models and caches, independently recomputed metrics, both H test cohorts, uncertainty tables, and Chinese/English journal manuscripts checked against new result tables. Until then, read `HANDOFF.md`, `D_PROGRESS.md`, `data/H/H_preprocessing_progress.json` and the qualification receipts for the actual stage. Do not label a launch, a passing test, or the completed legacy experiment as completion of this study.
