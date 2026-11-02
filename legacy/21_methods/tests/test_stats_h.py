"""Synthetic contracts for the pure H statistical kernel."""
import importlib
import math

import mpmath as mp
import numpy as np
import pandas as pd
import pytest


def kernel():
    try:
        return importlib.import_module("reliability_v2.stats_h")
    except ImportError as error:
        pytest.fail(f"H statistics implementation unavailable: {error}")


def table(scores=(0., 1., .5, 2., -1., 1.)):
    return pd.DataFrame(dict(
        video_uid=["r0", "r1", "a0", "a1", "b0", "b1"],
        label=[0, 0, 1, 1, 1, 1], method_id=["real", "real", "a", "a", "b", "b"],
        group_id=["g0", "g1", "g0", "g1", "g2", "g3"],
        score=np.asarray(scores, dtype=np.float64)))


def calibration():
    base = pd.DataFrame(dict(video_uid=["cal0", "cal1"], label=[0, 0],
        method_id=["real", "real"], group_id=["calg0", "calg1"], score=[0., 1.]))
    other = base.copy()
    other["score"] = [1., 2.]
    return {"B": base, "AB": other}


def cells(four=False):
    result = {"H00": table(), "H10": table((2., 1., 1.5, 0., 3., 1.))}
    if four:
        result["H01"] = table((.2, 1.2, .7, 2.2, -.8, 1.2))
        result["H11"] = table((0., 1., 2., 3., 4., 5.))
    return result


def analyze(q=None, **kwargs):
    return kernel().analyze_cohort(cells() if q is None else q, calibration(),
        fixed_methods=("a", "b"), repeats=16, **kwargs)


def row(result, cell, threshold="default"):
    return result["point_metrics"].set_index(["cell", "threshold_kind"]).loc[(cell, threshold)]


def effect(result, comparison, metric="pooled_auc", threshold="default"):
    indexed = result["comparisons"].set_index(["comparison", "threshold_kind", "metric"])
    return indexed.loc[(comparison, threshold, metric), "estimate"]


def test_four_cells_have_hand_checked_metrics_and_signed_interactions():
    out = analyze(cells(True))
    h00 = row(out, "H00")
    assert h00.pooled_auc == .5625
    assert h00.macro_auc == .5625
    assert h00.fpr == 1. and h00.tpr == .75 and h00.pooled_ba == .375
    assert h00.macro_tpr == .75 and h00.macro_ba == .375
    assert (h00.false_positives, h00.true_positives, h00.n_real, h00.n_fake) == (2, 3, 2, 4)
    assert effect(out, "H10-H00") == -.125
    assert effect(out, "H01-H00") == 0.
    assert effect(out, "H11-H10") == .5625
    assert effect(out, "(H11-H10)-(H01-H00)") == .5625
    assert effect(out, "H11-H01") == .4375
    assert len(out["independent_checks"]) == 8
    assert all(check["status"] == "PASSED" for check in out["independent_checks"])


def test_source_thresholds_are_per_model_B_thresholds_reused_on_AB_cells():
    out = analyze(cells(True))
    tau_b = np.nextafter(1., np.inf)
    tau_ab = np.nextafter(2., np.inf)
    assert out["thresholds"]["B"] == tau_b
    assert out["thresholds"]["AB"] == tau_ab
    assert row(out, "H00", "source_real").threshold == tau_b
    assert row(out, "H01", "source_real").threshold == tau_b
    assert row(out, "H10", "source_real").threshold == tau_ab
    assert row(out, "H11", "source_real").threshold == tau_ab
    assert row(out, "H00", "source_real").fpr == 0.
    assert row(out, "H00", "source_real").tpr == .25


@pytest.mark.parametrize("change", ["uid", "group", "label", "method"])
def test_equal_size_but_unequal_cohort_identity_is_rejected(change):
    q = cells()
    columns = {"uid": "video_uid", "group": "group_id", "label": "label", "method": "method_id"}
    q["H10"].loc[2, columns[change]] = {"uid": "different", "group": "other", "label": 0, "method": "b"}[change]
    with pytest.raises(ValueError):
        analyze(q)


def test_validators_reject_duplicate_videos_unknown_cells_and_unknown_methods():
    h = kernel()
    with pytest.raises(ValueError, match="unique"):
        h.normalize_video_table(pd.concat([table(), table().iloc[:1]]))
    with pytest.raises(ValueError, match="H00"):
        analyze({"H00": table(), "H11": table()})
    with pytest.raises(ValueError, match="registered"):
        h.analyze_cohort(cells(), calibration(), fixed_methods=("a",), repeats=1)


def test_calibration_rejects_fake_labels_and_test_id_overlap():
    h = kernel()
    cal = calibration()
    cal["B"].loc[0, "label"] = 1
    cal["B"].loc[0, "method_id"] = "a"
    with pytest.raises(ValueError, match="real"):
        h.analyze_cohort(cells(), cal, fixed_methods=("a", "b"), repeats=1)
    cal = calibration()
    for q in cal.values():
        q.loc[0, "video_uid"] = "r0"
    with pytest.raises(ValueError, match="overlap"):
        h.analyze_cohort(cells(), cal, fixed_methods=("a", "b"), repeats=1)


def test_real_only_source_calibration_never_needs_two_class_auc(monkeypatch):
    h = kernel()
    original = h.independent_auc
    def guarded(labels, values, weights=None):
        assert set(labels) == {0, 1}, "real SOURCE_CAL reached two-class AUC"
        return original(labels, values, weights)
    monkeypatch.setattr(h, "independent_auc", guarded)
    out = analyze()
    assert out["thresholds"]["B"] > 1.


def test_inclusive_ties_and_smallest_nextafter_calibration_boundary():
    h = kernel()
    cal = calibration()
    for key in cal:
        cal[key] = pd.DataFrame(dict(video_uid=[f"c{i}" for i in range(20)],
            label=[0] * 20, method_id=["real"] * 20,
            group_id=[f"cg{i}" for i in range(20)], score=np.arange(20, dtype=float)))
    q = {"H00": table((0., 1., 0., 0., 0., 0.)), "H10": table((0., 1., 0., 0., 0., 0.))}
    out = h.analyze_cohort(q, cal, fixed_methods=("a", "b"), repeats=2)
    assert out["thresholds"]["B"] == np.nextafter(18., np.inf)
    assert row(out, "H00").true_positives == 4
    assert row(out, "H00").pooled_auc == .25
    assert h.independent_auc([0, 0, 1, 1], [0., 1., 0., 1.]) == .5


def test_probability_default_and_threshold_above_one_are_supported():
    h = kernel()
    q = {"H00": table((0., 1., .5, 1., 0., .5)), "H10": table((0., 1., .5, 1., 0., .5))}
    cal = calibration()
    for part in cal.values():
        part["score"] = [1., 1.]
    out = h.analyze_cohort(q, cal, fixed_methods=("a", "b"), aggregation="mean_probability", repeats=2)
    assert row(out, "H00").threshold == .5
    assert out["thresholds"]["B"] > 1.
    assert row(out, "H00", "source_real").false_positives == 0
    assert row(out, "H00", "source_real").true_positives == 0


def test_shared_group_bootstrap_retains_repeats_and_integer_operating_counts():
    q = cells(True)
    out = analyze(q, min_groups=1, max_group_fraction=1.)
    draws = out["bootstrap"]["draws"]
    assert len(draws) == 16
    found_duplicate = False
    for record in draws.to_dict("records"):
        counts = dict(record["group_draw_counts"])
        assert sum(counts.values()) == 4
        found_duplicate |= max(counts.values()) > 1
        expanded = pd.concat([q["H00"][q["H00"].group_id.eq(group)] for group, n in counts.items() for _ in range(n)])
        if set(expanded.label) != {0, 1}:
            assert not record["pooled_valid"]
            continue
        expected_fp = int(((expanded.label == 0) & (expanded.score >= 0.)).sum())
        assert record["default:cell:H00:false_positives"] == expected_fp
        assert record["default:cell:H00:n_real"] == int(expanded.label.eq(0).sum())
        ys, xs = expanded.label.tolist(), expanded.score.tolist()
        reals = [x for y, x in zip(ys, xs) if y == 0]
        fakes = [x for y, x in zip(ys, xs) if y == 1]
        twice = sum(2 * (f > r) + (f == r) for f in fakes for r in reals)
        assert record["default:cell:H00:pooled_auc"] == twice / (2 * len(reals) * len(fakes))
        assert record["default:comparison:H01-H00:pooled_auc"] == 0.
    assert found_duplicate


def test_macro_missing_draws_and_two_class_invalid_draws_are_counted_separately():
    q = cells()
    for part in q.values():
        part["group_id"] = ["g0", "g0", "g1", "g1", "g2", "g2"]
    out = analyze(q, min_groups=1, max_group_fraction=1.)
    boot = out["bootstrap"]
    assert boot["pooled_invalid_repeats"] > 0
    assert boot["macro_invalid_repeats"] > boot["pooled_invalid_repeats"]
    valid = boot["draws"].query("pooled_valid and not macro_valid")
    assert len(valid) > 0
    assert valid["default:cell:H00:macro_auc"].isna().all()
    assert valid["default:cell:H00:macro_tpr"].isna().all()


def test_few_or_concentrated_source_groups_return_descriptions_without_ci():
    out = analyze()
    assert out["bootstrap"]["status"] == "insufficient_source_groups"
    assert out["bootstrap"]["intervals"] is None
    assert out["bootstrap"]["draws"].empty
    out = analyze(min_groups=1, max_group_fraction=.25)
    assert out["bootstrap"]["status"] == "insufficient_source_groups"


def test_unusable_ranking_does_not_float_rerank_or_remove_operating_metrics():
    q = cells(True)
    q["H10"]["ranking_usable"] = False
    out = analyze(q, min_groups=1, max_group_fraction=1.)
    result = row(out, "H10")
    assert math.isnan(result.pooled_auc) and math.isnan(result.macro_auc)
    assert result.fpr == 1.
    assert result.macro_tpr == 1.
    assert math.isnan(effect(out, "H10-H00"))
    assert out["bootstrap"]["intervals"].query("target == 'H10-H00' and metric == 'pooled_auc'").valid_repeats.eq(0).all()
    assert any(check["status"] == "OPERATING_PASSED_RANKING_UNRESOLVED" for check in out["independent_checks"])


def test_high_precision_ranks_remain_distinct_from_FP64_operating_scores():
    q = cells()
    with mp.workdps(100):
        ranks = [mp.mpf("1") + mp.mpf(i) * mp.mpf("1e-80") for i in [0, 1, 2, 3, 4, 5]]
    for part in q.values():
        part["score"] = 1.
        part["rank_score"] = pd.Series(ranks, dtype=object)
    out = analyze(q)
    assert row(out, "H00").pooled_auc == 1.
    assert row(out, "H00").false_positives == 2
    assert row(out, "H00", "source_real").true_positives == 0


def test_input_rows_are_sorted_by_identity_without_mutation_and_reject_bad_rank_flags():
    h = kernel()
    q = table().iloc[::-1]
    before = q.copy(deep=True)
    normalized = h.normalize_video_table(q)
    assert normalized.video_uid.tolist() == sorted(q.video_uid)
    pd.testing.assert_frame_equal(q, before)
    q["ranking_usable"] = "False"
    with pytest.raises(ValueError, match="boolean"):
        h.normalize_video_table(q)


def test_seed_summary_requires_exact_five_models_and_sample_sd_with_no_pooling():
    h = kernel()
    rows = pd.DataFrame([dict(seed=seed, backbone=backbone, comparison="H10-H00",
        threshold_kind="source_real", metric="pooled_auc", estimate=float(i))
        for backbone in ("clip", "dino", "efficientnet")
        for i, seed in enumerate((7, 42, 123, 2026, 3407))])
    out = h.summarize_seed_comparisons(rows)
    assert len(out) == 3
    assert out.n_seeds.eq(5).all() and out.estimate_mean.eq(2.).all()
    assert np.allclose(out.estimate_sample_sd, math.sqrt(2.5))
    with pytest.raises(ValueError, match="five"):
        h.summarize_seed_comparisons(rows.iloc[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        h.summarize_seed_comparisons(pd.concat([rows, rows.iloc[:1]]))
    rows.loc[0, "estimate"] = np.nan
    summary = h.summarize_seed_comparisons(rows)
    assert math.isnan(summary.set_index("backbone").loc["clip", "estimate_mean"])


def test_normalization_aligns_equivalent_numeric_label_dtypes():
    q = cells()
    q["H10"]["label"] = q["H10"].label.astype(np.int8)
    out = analyze(q)
    assert effect(out, "H10-H00") == -.125


def test_raw_only_contract_rejects_L2_collision_and_wrong_aggregation():
    h = kernel()
    for column, value in (("representation", "l2"), ("grouping", "collision"), ("aggregation", "mean_probability")):
        q = cells()
        for part in q.values():
            part[column] = value
        with pytest.raises(ValueError):
            h.analyze_cohort(q, calibration(), fixed_methods=("a", "b"), repeats=1)


@pytest.mark.parametrize("dtype", [str, np.float16, np.float32, np.longdouble])
def test_fp64_operating_scores_reject_non_FP64_float_and_string_dtypes(dtype):
    q = table()
    q["score"] = q.score.astype(dtype)
    with pytest.raises(ValueError, match="FP64"):
        kernel().normalize_video_table(q)


def test_integer_operating_scores_are_explicitly_normalized_to_FP64():
    q = table()
    q["score"] = [0, 1, 0, 2, -1, 1]
    result = kernel().normalize_video_table(q)
    assert result.score.dtype == np.dtype("float64")
    assert result.set_index("video_uid").score.to_dict() == {
        "r0": 0., "r1": 1., "a0": 0., "a1": 2., "b0": -1., "b1": 1.}


def test_cohort_hash_uses_newline_separated_sorted_video_uids():
    import hashlib
    out = analyze()
    expected = hashlib.sha256(bytes([97, 48, 10, 97, 49, 10, 98, 48, 10, 98, 49, 10, 114, 48, 10, 114, 49])).hexdigest()
    assert out["cohort_video_uid_sha256"] == expected


def test_independent_integer_weighted_concordance_counts_repeated_ties():
    h = kernel()
    assert h.independent_auc([0, 0, 1], [0., 1., 1.], [2, 1, 3]) == 5 / 6
    with pytest.raises(ValueError, match="integer"):
        h.independent_auc([0, 1], [0., 1.], [1., .5])


def test_rank_invalid_bootstrap_flags_are_separate_from_macro_operating_validity():
    q = cells()
    for part in q.values():
        part["group_id"] = "one"
    q["H10"]["ranking_usable"] = False
    out = analyze(q, min_groups=1, max_group_fraction=1.)
    boot = out["bootstrap"]
    assert boot["macro_invalid_repeats"] == 16
    assert boot["ranking_invalid_repeats"] == 16
    assert boot["pooled_invalid_repeats"] == 0
    assert not boot["draws"].macro_valid.any()
    assert boot["draws"].macro_operating_valid.all()
    assert boot["draws"]["default:cell:H10:macro_tpr"].eq(1.).all()


def test_complex_operating_scores_cannot_silently_drop_imaginary_part():
    q = table()
    q["score"] = q.score.astype(np.complex128) + 1j
    with pytest.raises(ValueError, match="FP64"):
        kernel().normalize_video_table(q)
