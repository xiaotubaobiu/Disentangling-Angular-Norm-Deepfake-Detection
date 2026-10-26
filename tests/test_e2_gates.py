"""Gate tests use physical videos with repeated language variants."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from evaluation.e2_gates import (
    PRIMARY, SEEDS, evaluate_gates, paired_seed_video_bootstrap, _ClusterMetric,
)


@pytest.fixture
def artifacts():
    metrics, controls, permutations, baseline, candidate = [], [], [], [], []
    for seed in SEEDS:
        for domain in ("ffpp", "cdf"):
            for condition in ("seen", "heldout", "fixed_h1"):
                for task, metric in PRIMARY.items():
                    if condition == "fixed_h1" and task != "T1":
                        continue
                    metrics.append(dict(seed=seed, domain=domain, condition=condition,
                                        task_id=task, metric=metric, value=1., count=12 if condition == "fixed_h1" else 24,
                                        method_id="__all__"))
            for metric, value in (("max_probability_error", 0.), ("prediction_consistency", 1.)):
                permutations.append(dict(seed=seed, domain=domain, condition="permutation",
                                         task_id="__all__", metric=metric, value=value, count=24))
            for i in range(12):
                family = ("camera", "swap", "reenact")[i % 3]
                label = int(family != "camera")
                base = dict(seed=seed, domain=domain, video_id=str(i), method_id=family,
                            video_key=f"{family}::{i}", label=label, method_family=family,
                            prob_fake=.9 if label else .1, num_frames=2)
                baseline.append(base)
                candidate.append(dict(base, condition="fixed_h1", task_id="T1"))
                for task, metric in (("T2", "polarity_consistency"), ("T4", "true_family_ordering")):
                    if task == "T4" and not label:
                        continue
                    controls.append(dict(base, condition="counterfactual", task_id=task,
                                         metric=metric, value=1., count=1,
                                         pairing_key=f"{seed}:{domain}:{task}:{i}"))
                for condition in ("meaningless_candidates", "meaningless_questions"):
                    for task in PRIMARY:
                        if condition == "meaningless_questions" and task not in ("T2", "T4"):
                            continue
                        for variant in (0, 1):
                            yes = label if variant == 0 else 1-label
                            row = dict(base, condition=condition, task_id=task,
                                       variant_id=str(variant), question_id=str(variant),
                                       question_yes_means="fake" if variant == 0 else "real",
                                       question_pair_id="pair", target_semantic="yes" if yes else "no",
                                       pairing_key=f"{seed}:{domain}:{condition}:{task}:{i}:{variant}",
                                       meaningful_num_frames=2,
                                       meaningful_prob_fake=base["prob_fake"],
                                       prob_fake=1-base["prob_fake"],
                                       meaningful_p_yes=.9 if yes else .1, p_yes=.1 if yes else .9)
                            for j, name in enumerate(("camera", "swap", "reenact")):
                                row[f"meaningful_p_{name}"] = .9 if name == family else .05
                                row[f"p_{name}"] = .9 if j == (i % 3 + 1) % 3 else .05
                            controls.append(row)
            for condition in ("meaningless_candidates", "meaningless_questions"):
                for task, metric in PRIMARY.items():
                    if condition == "meaningless_questions" and task not in ("T2", "T4"):
                        continue
                    metrics.append(dict(seed=seed, domain=domain, condition=condition,
                                        task_id=task, metric=metric+"_drop", value=1., count=24,
                                        method_id="__all__", meaningful_score=1., control_score=0., drop=1.))
    return {name: pd.DataFrame(rows) for name, rows in
            (("metric_rows", metrics), ("control_rows", controls),
             ("permutation_rows", permutations), ("baseline_rows", baseline),
             ("candidate_rows", candidate))}


@pytest.fixture
def cfg():
    return {"n_resamples": 100, "final": False, "seed": 19}


def test_all_gates_pass(artifacts, cfg):
    report = evaluate_gates(**artifacts, cfg=cfg)
    assert report.verdict == "supported"
    assert all(g.status == "pass" for g in report.gates.values())
    assert report.gates["G6"].evidence["ffpp/meaningless_candidates/T1"]["n_videos"] == 12


def test_point_failure_takes_precedence(artifacts, cfg):
    rows = artifacts["metric_rows"]
    rows.loc[(rows.task_id == "T5") & (rows.condition == "heldout"), "value"] = .55
    report = evaluate_gates(**artifacts, cfg=cfg)
    assert report.gates["G5"].status == "fail"
    assert report.verdict == "not_supported"


def test_ci_crossing_is_inconclusive(artifacts, cfg):
    for name in ("baseline_rows", "candidate_rows"):
        rows = artifacts[name]
        i = rows.video_id.astype(int)
        # Same AUROC, different error videos: paired variance is nonzero.
        rows["prob_fake"] = rows.label * .6 + .2
        wrong = (i.isin([0, 1]) if name == "baseline_rows" else i.isin([3, 4]))
        rows.loc[wrong, "prob_fake"] = 1 - rows.loc[wrong, "prob_fake"]
    sample = artifacts["candidate_rows"].query("seed == 42 and domain == 'ffpp'")
    artifacts["metric_rows"].loc[artifacts["metric_rows"].condition == "fixed_h1", "value"] = roc_auc_score(sample.label, sample.prob_fake)
    report = evaluate_gates(**artifacts, cfg=cfg)
    assert report.gates["G1"].point_passed
    assert not report.gates["G1"].ci_passed
    assert report.verdict == "insufficient_evidence"


@pytest.mark.parametrize("damage,match", [
    ("key", "video-key sets differ"), ("seed", "seeds"),
    ("task", "missing"), ("duplicate", "duplicate"),
    ("nan", "finite"), ("pair", "duplicate"),
])
def test_corrupt_evidence_rejected(artifacts, cfg, damage, match):
    if damage == "key":
        artifacts["candidate_rows"] = artifacts["candidate_rows"].iloc[:-1]
    elif damage == "seed":
        artifacts["baseline_rows"] = artifacts["baseline_rows"].query("seed != 7")
    elif damage == "task":
        artifacts["metric_rows"] = artifacts["metric_rows"].query("task_id != 'T4'")
    elif damage == "duplicate":
        artifacts["permutation_rows"] = pd.concat([artifacts["permutation_rows"], artifacts["permutation_rows"].iloc[:1]])
    elif damage == "nan":
        artifacts["metric_rows"].loc[0, "value"] = np.nan
    else:
        rows = artifacts["control_rows"]
        artifacts["control_rows"] = pd.concat([rows, rows.query("condition == 'meaningless_candidates'").iloc[:1]])
    with pytest.raises(ValueError, match=match):
        evaluate_gates(**artifacts, cfg=cfg)


def test_final_mode_cannot_reduce_resamples(artifacts):
    with pytest.raises(ValueError, match="10,?000"):
        evaluate_gates(**artifacts, cfg={"n_resamples": 100})


@pytest.mark.parametrize("task", ["T1", "T2", "T4", "T5"])
def test_optimized_weighted_metrics_match_reference(artifacts, task):
    rows = artifacts["control_rows"].query(
        "seed == 42 and domain == 'ffpp' and condition == 'meaningless_candidates' and task_id == @task"
    ).copy().sort_values(["video_key", "variant_id"])
    rng = np.random.default_rng(4)
    rows["prob_fake"] = rng.choice([.1, .4, .5, .8], len(rows))
    rows["p_yes"] = rng.random(len(rows))
    for column in ("p_camera", "p_swap", "p_reenact"):
        rows[column] = rng.random(len(rows))
    cluster_ids = pd.factorize(rows.video_key, sort=True)[0]
    weights = rng.integers(1, 5, size=(5, 12))
    actual = _ClusterMetric(rows, task, cluster_ids, 12)(weights)
    row_weights = weights[:, cluster_ids]
    if task == "T1":
        expected = [roc_auc_score(rows.label, rows.prob_fake, sample_weight=w) for w in row_weights]
    else:
        truth = rows.label if task == "T2" else ((rows.target_semantic == "yes").astype(int) if task == "T4" else rows.method_family)
        predicted = rows.prob_fake >= .5 if task == "T2" else (rows.p_yes >= .5 if task == "T4" else np.array(["camera", "swap", "reenact"])[rows[["p_camera", "p_swap", "p_reenact"]].values.argmax(1)])
        expected = [balanced_accuracy_score(truth, predicted, sample_weight=w) for w in row_weights]
    np.testing.assert_allclose(actual, expected, atol=1e-14)


def test_paired_identical_predictions_have_zero_interval(artifacts):
    base = artifacts["baseline_rows"].query("domain == 'ffpp'")
    result = paired_seed_video_bootstrap(base, base, "auroc", 100, 42)
    assert (result.point, result.lower, result.upper, result.n_videos) == (0., 0., 0., 12)


def test_shared_seed_draws_and_quantile_match_reference(artifacts):
    base = artifacts["baseline_rows"].query("domain == 'ffpp'").copy()
    cand = base.copy()
    rng = np.random.default_rng(48)
    for seed in SEEDS:
        mask = base.seed == seed
        base.loc[mask, "prob_fake"] = rng.choice([.2, .5, .7], mask.sum())
        cand.loc[mask, "prob_fake"] = rng.choice([.1, .5, .8], mask.sum())
    physical = base.query("seed == 42").sort_values("video_key")
    reference_rng = np.random.default_rng(91)
    weights = np.zeros((32, 12))
    for indices in physical.reset_index(drop=True).groupby(["label", "method_id"]).indices.values():
        weights[:, indices] = reference_rng.multinomial(len(indices), np.full(len(indices), 1/len(indices)), size=32)
    deltas = []
    for w in weights:
        values = []
        for seed in SEEDS:
            left = base.query("seed == @seed").sort_values("video_key")
            right = cand.query("seed == @seed").sort_values("video_key")
            values.append(roc_auc_score(right.label, right.prob_fake, sample_weight=w)
                          - roc_auc_score(left.label, left.prob_fake, sample_weight=w))
        deltas.append(np.mean(values))
    result = paired_seed_video_bootstrap(base, cand, "auroc", 32, 91)
    assert result.lower == pytest.approx(np.quantile(deltas, .05))
    two_sided = paired_seed_video_bootstrap(base, cand, "auroc", 32, 91, lower_quantile=.025)
    assert two_sided.lower == pytest.approx(np.quantile(deltas, .025))


def test_single_seed_permutation_failure_is_not_averaged_away(artifacts, cfg):
    rows = artifacts["permutation_rows"]
    rows.loc[(rows.seed == 42) & (rows.metric == "max_probability_error"), "value"] = 2e-6
    report = evaluate_gates(**artifacts, cfg=cfg)
    assert report.gates["G3"].status == "fail"


@pytest.mark.parametrize("damage", ["label", "frame", "normalized", "variant", "counterfactual", "physical_label", "null_identity"])
def test_pair_integrity_rejected(artifacts, cfg, damage):
    rows = artifacts["control_rows"]
    index = rows.query("condition == 'meaningless_candidates' and task_id == 'T1'").index[0]
    if damage == "label":
        artifacts["candidate_rows"].loc[0, "label"] = 1
    elif damage == "frame":
        rows.loc[index, "meaningful_num_frames"] = 99
    elif damage == "normalized":
        metrics = artifacts["metric_rows"]
        metrics.loc[metrics.metric == "auroc_drop", "value"] = .5
    elif damage == "variant":
        artifacts["control_rows"] = rows.drop(index)
    elif damage == "counterfactual":
        artifacts["control_rows"] = rows.drop(rows.query("condition == 'counterfactual'").index[0])
    elif damage == "physical_label":
        mask = (rows.condition == "meaningless_candidates") & (rows.task_id == "T4")
        rows.loc[mask, "label"] = 1 - rows.loc[mask, "label"]
    else:
        rows.loc[rows.condition == "meaningless_candidates", "variant_id"] = np.nan
    with pytest.raises(ValueError):
        evaluate_gates(**artifacts, cfg=cfg)


@pytest.mark.parametrize("damage", ["all_seed_truncation", "permutation_count", "fixed_metric", "stored_control_score"])
def test_contradictory_artifact_counts_or_metrics_rejected(artifacts, cfg, damage):
    if damage == "all_seed_truncation":
        rows = artifacts["control_rows"]
        artifacts["control_rows"] = rows.loc[~((rows.condition == "meaningless_candidates") & (rows.task_id == "T1") & (rows.variant_id == "1"))]
    elif damage == "permutation_count":
        artifacts["permutation_rows"].loc[0, "count"] += 1
    elif damage == "fixed_metric":
        rows = artifacts["metric_rows"]
        rows.loc[rows.condition == "fixed_h1", "value"] = .1
    else:
        rows = artifacts["metric_rows"]
        rows.loc[rows.condition == "meaningless_candidates", "control_score"] = .8
    with pytest.raises(ValueError, match="count|differs"):
        evaluate_gates(**artifacts, cfg=cfg)


def test_seeds_may_have_different_language_variant_identities(artifacts, cfg):
    rows = artifacts["control_rows"]
    mask = rows.condition.str.startswith("meaningless") & (rows.seed == 7)
    rows.loc[mask, "variant_id"] = "seed7-" + rows.loc[mask, "variant_id"]
    report = evaluate_gates(**artifacts, cfg=cfg)
    assert report.verdict == "supported"


def test_optional_dfd_audit_is_required_complete_and_can_fail_g3(artifacts, cfg):
    extra = artifacts["permutation_rows"].query("domain == 'ffpp'").copy()
    extra["domain"] = "dfd"
    extra.loc[extra.metric == "prediction_consistency", "value"] = .9
    artifacts["permutation_rows"] = pd.concat([artifacts["permutation_rows"], extra])
    report = evaluate_gates(**artifacts, cfg=cfg)
    assert report.gates["G3"].status == "fail"
    assert all(report.gates[name].status == "pass" for name in ("G1", "G2", "G4", "G5", "G6"))
    rows = artifacts["permutation_rows"]
    artifacts["permutation_rows"] = rows.loc[~((rows.domain == "dfd") & (rows.seed == 7))]
    with pytest.raises(ValueError, match="missing"):
        evaluate_gates(**artifacts, cfg=cfg)


def test_g6_requires_strictly_positive_two_sided_lower_bound(artifacts, cfg):
    rows = artifacts["control_rows"]
    selected = (rows.condition == "meaningless_candidates") & (rows.task_id == "T2")
    rows.loc[selected, "prob_fake"] = rows.loc[selected, "meaningful_prob_fake"]
    wrong = selected & rows.video_id.isin(["0", "1"])
    rows.loc[wrong, "prob_fake"] = 1 - rows.loc[wrong, "prob_fake"]
    metrics = artifacts["metric_rows"]
    selected_metrics = (metrics.condition == "meaningless_candidates") & (metrics.task_id == "T2")
    metrics.loc[selected_metrics, ["value", "drop"]] = .1875
    metrics.loc[selected_metrics, "control_score"] = .8125
    report = evaluate_gates(**artifacts, cfg=cfg)
    assert report.gates["G6"].point_passed
    assert report.gates["G6"].evidence["ffpp/meaningless_candidates/T2"]["lower"] == 0
    assert report.gates["G6"].status == "inconclusive"
    assert report.verdict == "insufficient_evidence"


def _script():
    path = Path(__file__).resolve().parents[1] / "experiments/2026-09-22_semantic_visual_jev_e2/scripts/assess_gates.py"
    spec = importlib.util.spec_from_file_location("assess_gates", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_artifacts(root, artifacts):
    for seed in SEEDS:
        for domain in ("ffpp", "cdf"):
            baseline_dir = root / "baseline" / f"seed_{seed}"
            baseline_dir.mkdir(parents=True, exist_ok=True)
            artifacts["baseline_rows"].query("seed == @seed and domain == @domain").drop(columns=["seed", "domain"]).to_csv(baseline_dir / f"{domain}_video_predictions.csv", index=False)
            for condition in ("fixed_h1", "seen", "heldout", "permutation", "counterfactual", "meaningless_candidates", "meaningless_questions"):
                directory = root / "eval" / f"seed_{seed}" / condition
                directory.mkdir(parents=True, exist_ok=True)
                source = artifacts["permutation_rows"] if condition == "permutation" else artifacts["metric_rows"]
                selected = source.query("seed == @seed and domain == @domain and condition == @condition")
                if condition == "counterfactual":
                    selected = artifacts["control_rows"].query("seed == @seed and domain == @domain and condition == @condition")
                (directory / f"{domain}_metrics.json").write_text(selected.to_json(orient="records"))
                if condition.startswith("meaningless") or condition == "counterfactual":
                    selected = artifacts["control_rows"].query("seed == @seed and domain == @domain and condition == @condition")
                    suffix = "paired_videos" if condition.startswith("meaningless") else "controls"
                    selected.to_csv(directory / f"{domain}_{suffix}.csv", index=False)
                if condition == "fixed_h1":
                    artifacts["candidate_rows"].query("seed == @seed and domain == @domain").to_csv(directory / f"{domain}_videos.csv", index=False)


def test_root_loader_and_report_outputs(tmp_path, artifacts, cfg):
    _write_artifacts(tmp_path, artifacts)
    module = _script()
    report_path = tmp_path / "reports/e2_validation_result.md"
    report = module.assess_run(tmp_path / "eval", tmp_path / "baseline", tmp_path / "summary", report_path, cfg=cfg)
    assert report.verdict == "supported"
    status = json.loads((tmp_path / "summary/gate_status.json").read_text())
    assert status["final"] is False
    assert len(status["sources"]) == 72
    assert "NONFINAL" in report_path.read_text()
    assert "Calibration" in report_path.read_text()
    assert (tmp_path / "summary/seed_metrics.csv").is_file()
    assert len(pd.read_csv(tmp_path / "summary/bootstrap_intervals.csv")) == 16
    assert module.main(["--evaluation-root", str(tmp_path / "eval"), "--baseline-root", str(tmp_path / "baseline"), "--output-root", str(tmp_path / "summary"), "--report", str(report_path)]) == 0
    final_status = json.loads((tmp_path / "summary/gate_status.json").read_text())
    assert final_status["final"] is True
    assert final_status["n_resamples"] == 10000
    # A present optional DFD audit requires all three seeds.
    optional = tmp_path / "eval/seed_42/permutation/dfd_metrics.json"
    optional.write_text("[]")
    with pytest.raises(ValueError, match="empty|missing"):
        module.load_artifacts(tmp_path / "eval", tmp_path / "baseline")
    optional.unlink()
    missing = tmp_path / "eval/seed_7/seen/cdf_metrics.json"
    missing.unlink()
    assert module.main(["--evaluation-root", str(tmp_path / "eval"), "--baseline-root", str(tmp_path / "baseline"), "--output-root", str(tmp_path / "summary"), "--report", str(report_path)]) == 2
    failed = json.loads((tmp_path / "summary/gate_status.json").read_text())
    assert failed["verdict"] == "insufficient_evidence"
    assert "missing" in failed["error"].lower()
    assert pd.read_csv(tmp_path / "summary/bootstrap_intervals.csv").empty
    assert pd.read_csv(tmp_path / "summary/seed_metrics.csv").empty
