### Task 2: Correctness gates

**Files:**
- Test: `tests/`
- Test: `experiments/2026-07-23_four_questions/tests/`
- Test: `experiments/2026-07-27_pooling_saturation/tests/`
- Test: `experiments/2026-07-30_threshold_transfer/tests/`

**Interfaces:**
- Consumes: current repository code and retained test fixtures.
- Produces: a pass/fail gate for aggregation keys, paired sampling, calibration, leakage, and bootstrap behavior.

- [ ] **Step 1: Run all collected tests**

Run:

```bash
PYTHONPATH=src python -m pytest -q
```

Expected: all 99 collected tests pass.

- [ ] **Step 2: Run threshold-transfer leakage tests verbosely**

Run:

```bash
PYTHONPATH=src python -m pytest experiments/2026-07-30_threshold_transfer/tests/test_no_leakage.py -vv
```

Expected: source-only calibration and threshold selection tests pass.

