### Task 7: Implement three-seed bootstrap and G1-G6 adjudication

**Files:**
- Create: src/evaluation/e2_gates.py
- Create: experiments/2026-09-22_semantic_visual_jev_e2/scripts/assess_gates.py
- Create: tests/test_e2_gates.py

**Interfaces:**
- Produces: paired_seed_video_bootstrap(baseline, candidate, metric, n_resamples, seed) -> BootstrapResult
- Produces: evaluate_gates(metric_rows, control_rows, permutation_rows, baseline_rows, candidate_rows, cfg) -> GateReport
- Produces: GateReport.verdict in supported, not_supported, insufficient_evidence

- [ ] **Step 1: Write synthetic pass, fail, and uncertainty tests**

~~~python
def test_all_gates_pass_returns_supported(passing_artifacts, gate_config):
    report = evaluate_gates(**passing_artifacts, cfg=gate_config)
    assert report.verdict == "supported"
    assert all(gate.status == "pass" for gate in report.gates.values())


def test_point_failure_returns_not_supported(passing_artifacts, gate_config):
    passing_artifacts["metric_rows"].loc[
        lambda frame: frame.task_id == "T5", "heldout_score"
    ] = 0.55
    report = evaluate_gates(**passing_artifacts, cfg=gate_config)
    assert report.gates["G5"].status == "fail"
    assert report.verdict == "not_supported"


def test_ci_crossing_margin_returns_insufficient_evidence(
    passing_artifacts, gate_config
):
    passing_artifacts["baseline_rows"] = noisy_noninferiority_fixture()
    report = evaluate_gates(**passing_artifacts, cfg=gate_config)
    assert report.gates["G1"].point_passed is True
    assert report.gates["G1"].ci_passed is False
    assert report.verdict == "insufficient_evidence"


def test_video_key_mismatch_is_rejected(passing_artifacts, gate_config):
    passing_artifacts["candidate_rows"] = (
        passing_artifacts["candidate_rows"].iloc[:-1]
    )
    with pytest.raises(ValueError, match="video-key sets differ"):
        evaluate_gates(**passing_artifacts, cfg=gate_config)
~~~

- [ ] **Step 2: Run tests and verify the missing module failure**

Run:

~~~bash
pytest -q tests/test_e2_gates.py
~~~

Expected: import fails because evaluation.e2_gates is absent.

- [ ] **Step 3: Implement stratified paired bootstrap**

Within each seed and domain, align on method_id::video_id and verify identical
labels. Resample real videos and each fake method stratum with replacement.
Apply the same sampled keys to baseline and E2, compute the seed metric
difference, then average the three seed differences in each replicate.

~~~python
for bootstrap_index in range(n_resamples):
    seed_deltas = []
    for experiment_seed in (42, 7, 2048):
        indices = stratified_video_resample(aligned[experiment_seed], rng)
        seed_deltas.append(
            metric(candidate[indices]) - metric(baseline[indices])
        )
    bootstrap_delta[bootstrap_index] = statistics.fmean(seed_deltas)
~~~

The G1 one-sided lower bound is the fifth percentile. G6 uses the 2.5th
percentile of meaningful minus meaningless degradation.

Represent outputs with immutable records so gate fields cannot drift between
the library and CLI:

~~~python
@dataclass(frozen=True)
class BootstrapResult:
    point: float
    lower: float
    upper: float
    n_resamples: int
    n_videos: int


@dataclass(frozen=True)
class GateResult:
    name: str
    status: str
    point_passed: bool
    ci_passed: bool | None
    evidence: dict[str, float | int | str]


@dataclass(frozen=True)
class GateReport:
    verdict: str
    gates: dict[str, GateResult]
~~~

- [ ] **Step 4: Encode all immutable gate rules**

Gates are computed exactly as follows:

- G1: each of FF++ and CDF has mean Fixed-H1 AUROC delta at least -0.01
  and one-sided lower bound at least -0.01;
- G2: each domain has macro seen-minus-heldout drop at most 0.02 and every
  task drop at most 0.05;
- G3: maximum permutation error at most 1e-6 and consistency exactly 1.0;
- G4: held-out T2 balanced accuracy at least 0.80 and polarity consistency
  at least 0.95 in each primary domain;
- G5: held-out T4 balanced accuracy at least 0.65, true-family ordering at
  least 0.80, and T5 macro balanced accuracy at least 0.60 in each primary
  domain;
- G6: meaningful-minus-meaningless candidate score and T2/T4 question score
  are each at least 0.10 in each primary domain, with paired 95 percent lower
  bounds greater than zero.

Any point failure yields not_supported. With no point failures, any required
CI failure yields insufficient_evidence. Only all passes yield supported.

- [ ] **Step 5: Implement the artifact-loading CLI**

assess_gates.py requires all three seed directories, all seven conditions, and
three matching H1 seed prediction directories. It writes:

~~~text
results/summary/seed_metrics.csv
results/summary/bootstrap_intervals.csv
results/summary/gate_status.json
results/summary/gate_status.csv
reports/e2_validation_result.md
~~~

The Markdown report prints exact thresholds, point estimates, confidence
bounds, per-method diagnostics, calibration diagnostics, and the final verdict.

- [ ] **Step 6: Run gate and existing comparison tests**

Run:

~~~bash
pytest -q tests/test_e2_gates.py tests/test_model_comparison.py
~~~

Expected: all tests pass.

- [ ] **Step 7: Commit adjudication**

~~~bash
git add src/evaluation/e2_gates.py tests/test_e2_gates.py experiments/2026-09-22_semantic_visual_jev_e2/scripts/assess_gates.py
git commit -m "feat: preregister E2 gate adjudication"
~~~

---

