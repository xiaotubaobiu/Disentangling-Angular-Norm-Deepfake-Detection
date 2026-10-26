# Prepared original H LAST pair result audit

Preparation is complete. Actual result audit has NOT run. No real CPU slot was opened or locked, no background waiter started, and no production/active F files were changed.

Entry: `h_pair_result_audit.py`. Calling without `--execute` only prints preparation status. After root explicitly confirms original CLIP seed42 terminal and authorizes a single audit, run the following template once, substituting the observed complete.json SHA and a free shared slot chosen by root. Do not execute this template during preparation.

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /usr/bin/time -v /home/z/workspace/deepfake_pipeline_reliability_v2_20261023/environment/bin/python /home/z/workspace/deepfake_pipeline_reliability_v2_20261023/engineering/source_best_M/audits/h_pair_result_audit.py --execute --terminal-sha <ROOT_OBSERVED_TERMINAL_SHA256> --slot <ROOT_SELECTED_FREE_0_OR_1> --output-name H_LAST_clip_seed42_independent_audit.json
```

Use an audit-local stdout/time log. Output JSON is created exclusively under this directory and an existing output is preserved. The process acquires the existing canonical R/analysis/H/.cpu_slots/slotN.lock with nonblocking flock before production reads, holds it through output publication, and immediately fails if busy. There is no waiting/retry or GPU/model/source-context invocation.

The entry checks72 artifact hashes plus complete.json before/after; identity, code,107 accepted admission pins, all14 cache marker hashes/headers and accepted role Parquet metadata. It reconstructs exact sourceCAL real-only threshold boundaries, point AUC, macroAUC, FPR/TPR/pooledBA/macroTPR/macroBA and signed effects from score CSVs. A separate integer pair-concordance implementation preserves350-digit rank exports and exact FP64 operating scores. It independently reconstructs all applicable2000 RNG multiplicity maps, weighted cell/effect values, invalid/missing-method/ranking-NA rules and linear2.5/97.5 percentiles. No producer _PreparedCell or scoring implementation is imported. Every cohort is processed separately; cal and training seeds are fixed, and negative/NA values are retained.

Scope boundaries are explicit in the resulting audit: accepted pixel integrity is rebound through immutable prior14-role admission, not rescanned. Feature dot-products/frame-to-video score construction and geometry/coverage numerics are not independently reconstructed by this saved-score audit; coverage.json is inventoried and SHA-bound. Single-pair acceptance does not provide a five-seed summary, M source-best result, M cache/GPU authorization, or compression capacity credit. If broader scientific acceptance requires those external checks, they remain pending rather than being claimed here.

Existing snapshot eligibility only compares accepted identity and four frozen bootstrap SHA values. It does not promote, re-encode or resample the existing bundle. Separate scientific acceptance and an explicit promotion receipt remain necessary.

Synthetic evidence:44 tests passed,10.545s test time;10.71s wall and82312KiB maximum RSS using one thread/empty CUDA. Tests touch only tiny temporary files and E/audits. RED evidence includes absent module and a hidden extra nested complete.json case before correction.

Resource plan for the later actual audit: one shared CPU slot, one scientific thread, no GPU; process one cohort JSON at a time, plan up to2GiB RAM and10–30min wall. These are conservative planning estimates, not an actual-data timing or admission result. Root must select the slot after terminal; this preparation has not checked terminal status or current slot availability.
