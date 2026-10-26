# Early source-only four-rank smoke, 2026-09-22

Executed while Task6 integration proceeded; this is non-reportable diagnostic work.

- Code HEAD: 6fb7221088d3504cad9401fb504bb071fa3b4f13; Task6 evaluator/control edits were uncommitted and not imported by training.
- Environment: PyTorch 2.6.0+cu124, CUDA_VISIBLE_DEVICES=0,1,2,3, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1.
- Each visible GPU had 31.43 GiB free before launch.
- Source dataset inventory: train 149956 frames, val 25584 frames, all sixteen fake methods plus real, first 100 sampled frame paths existed.
- Command: `torchrun --standalone --nproc_per_node=4 --log-dir experiments/2026-09-22_semantic_visual_jev_e2/results/early_smoke_logs --tee 3 experiments/2026-09-22_semantic_visual_jev_e2/scripts/train_semantic.py --config experiments/2026-09-22_semantic_visual_jev_e2/configs/semantic_jev.yaml --seed 42 --smoke --max-train-batches 4 --max-val-batches 2` with environment above.
- Unified exec session 63043 exited 0. No training process remains for that handle.
- Output: train_ce 0.781585693359375; val_seen_ce 0.8118088021874428; val_seen_accuracy 0.384765625; val_seen_count 1024.
- Smoke checkpoint: experiments/2026-09-22_semantic_visual_jev_e2/results/training/smoke/seed_42/best_model.pth.
- Four ranks were observed as GPU compute processes with roughly 2.0–2.6 GiB occupied; this observation is NOT a peak-allocated-memory measurement.

## Findings for Task8 integration

1. `itertools.islice` abandons an active multiworker DataLoader after a few batches. Multiple ranks emitted `_MultiProcessingDataLoaderIter.__del__` / `_shutdown_workers` / worker `SIGABRT` at train and validation teardown, although all ranks ultimately completed and launcher exit was zero. Need a real bounded loader that exhausts its scheduled indices, or another verified clean shutdown; don't suppress the exception. Logs under results/early_smoke_logs/7da5ebc1-4a51-4221-b048-f2c6208f3d4a_eflsj8y_/attempt_0/.
2. CUDA emitted Memory Efficient attention nondeterministic backward warnings even though configure_determinism uses warn_only=True. Resolve kernel selection before reportable runs and record effective backend; deterministic model decisions are required. No change to architecture or loss.
3. `build_checkpoint_payload` does not currently write smoke/reportable flags, encoded split list, config hash, or per-rank peak/batch/candidate-count diagnostics. Task8 explicitly requires smoke provenance and four-rank evidence; implement minimal instrumentation with covering tests.
4. Final preflight command in plan omits CUDA_VISIBLE_DEVICES while preflight requires exactly four visible devices. Execute final preflight with the same authorized visibility as training.

The real source-only smoke is evidence of data/model/DDP interoperability, not completion of Task8; formal preflight and clean smoke remain required.

## Focused attention-backend probe

With CUBLAS_WORKSPACE_CONFIG=:16:8 set before Python, the actual E2 decision model on GPU0 ran the same finite synthetic AMP backward in default and math-only SDPA modes. Default mode emitted the Memory Efficient attention nondeterministic warning; disabling flash/memory-efficient SDPA and enabling math SDPA emitted no warnings and all gradients remained finite. Different backend/dropout implementations consumed RNG differently, so losses are not used as a numerical-equivalence test. Task8 should lock a deterministic supported backend before any reportable training, then verify repeatability within that backend.
