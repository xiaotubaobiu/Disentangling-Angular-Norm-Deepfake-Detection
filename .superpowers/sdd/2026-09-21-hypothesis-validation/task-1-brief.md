### Task 1: Evidence and provenance audit

**Files:**
- Read: `experiments/2026-05-05_paper_feature_geometry/results/checkpoints/**/output/best_model.pth`
- Read: `experiments/2026-07-21_frame_midterm_diagnosis/results/parallel_20260721_213914/per_frame_scores.parquet`
- Read: `experiments/2026-07-27_pooling_saturation/results/val_cache/*.parquet`
- Read: `/Dataset/deepfake_detection/**`

**Interfaces:**
- Consumes: checkpoint directories, saved configs, source-validation caches, CDF frame scores.
- Produces: counts for checkpoints, domains, methods, frames, videos, and an explicit mapping of “model A” to retained artifacts.

- [ ] **Step 1: Confirm checkpoint coverage**

Run:

```bash
find experiments/2026-05-05_paper_feature_geometry/results/checkpoints -name best_model.pth | wc -l
find experiments/2026-05-05_paper_feature_geometry/results/checkpoints -name meta.json | wc -l
```

Expected: `30` checkpoints and `30` metadata files; the hypothesis comparison uses the 15 raw-trained checkpoints.

- [ ] **Step 2: Run the source/target input audit**

Run:

```bash
python experiments/2026-07-30_threshold_transfer/scripts/00_audit_inputs.py
```

Expected: `34/34 inputs present`, 15 FF++ validation caches, all 15 raw checkpoints represented on CDF, DFD explicitly excluded.

- [ ] **Step 3: Query authoritative parquet coverage**

Run DuckDB against `per_frame_scores.parquet` to verify 30 checkpoints, three domains, 39,830,640 rows, and the expected raw/direction/length versus normtrain mode layout.

- [ ] **Step 4: Check whether an adaptive-gamma checkpoint exists**

Run:

```bash
rg -n -i "adaptive.?gamma|gamma.?head|radial.?angular|径向.?角向" src experiments --glob '*.py' --glob '*.yaml'
find . -type f \( -name '*.pth' -o -name '*.ckpt' -o -name '*.safetensors' \)
```

Expected: document either the exact checkpoint/prediction path or the limitation that only raw/normtrain checkpoints are present.

