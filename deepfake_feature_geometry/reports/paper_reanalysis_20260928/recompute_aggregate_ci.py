"""Supplementary target-video bootstrap for mean threshold-transfer regret.

Reads frozen outputs only. All checkpoints and both score views use the same
multinomial count vector on each target-video bootstrap replicate. FF++ source
thresholds remain fixed; CDF oracle balanced accuracy is recomputed per replicate.
"""
from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/home/z/project/deepfake_detection/projects/feature_geometry")
TABLES = ROOT / "experiments/2026-07-30_threshold_transfer/results/tables"
OUT = Path(__file__).resolve().parent
ITERATIONS = 2000
SEED = 20260730
CHUNK = 200


def summaries(x: np.ndarray) -> dict[str, float]:
    low, high = np.percentile(x, [2.5, 97.5])
    return {"ci_low": float(low), "ci_high": float(high)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def weighted_metrics(scores: np.ndarray, labels: np.ndarray,
                     threshold: float, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns oracle and frozen-threshold BA for each row of count weights.

    For sorted scores, each distinct-score group end is a feasible decision
    boundary. With F and R total fake and real weights, BA at a boundary is
    0.5 * (1 - cumulative_fake/F + cumulative_real/R). This matches the
    repository's midpoint candidates and >= threshold decision rule, including
    ties. The final group gives the all-negative BA=0.5 candidate.
    """
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores) != 0), len(scores) - 1]
    below_source = int(np.searchsorted(sorted_scores, threshold, side="left"))
    sorted_labels = labels[order].astype(np.int32)
    oracles = []
    sources = []
    for start in range(0, len(weights), CHUNK):
        w = weights[start:start + CHUNK, :][:, order]
        cum_fake = np.cumsum(w * sorted_labels[None, :], axis=1, dtype=np.int32)
        cum_real = np.cumsum(w * (1 - sorted_labels)[None, :], axis=1, dtype=np.int32)
        fake_n = cum_fake[:, -1].astype(float)
        real_n = cum_real[:, -1].astype(float)
        assert np.all(fake_n > 0) and np.all(real_n > 0)
        boundary_ba = 0.5 * (1 - cum_fake[:, ends] / fake_n[:, None]
                             + cum_real[:, ends] / real_n[:, None])
        oracles.append(boundary_ba.max(axis=1))
        if below_source:
            source_ba = 0.5 * (1 - cum_fake[:, below_source - 1] / fake_n
                               + cum_real[:, below_source - 1] / real_n)
        else:
            source_ba = np.full(len(w), 0.5)
        sources.append(source_ba)
    return np.concatenate(oracles), np.concatenate(sources)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    scores = pd.read_parquet(TABLES / "04_target_video_scores.parquet")
    pooled = pd.read_csv(TABLES / "04_transfer_pooled.csv")
    pooled = pooled[(pooled.scope == "pooled") & (pooled.rule == "t_BA")]
    pooled = pooled[(pooled["score"] == "s_raw") | (pooled["score"] == "s_dir")]
    assert len(pooled) == 30
    keys = np.array(sorted(scores.video_key.unique()))
    n = len(keys)
    assert n == 9901
    checkpoints = sorted(scores.checkpoint_id.unique())
    assert len(checkpoints) == 15
    wide = {}
    labels = None
    for checkpoint in checkpoints:
        frame = scores[scores.checkpoint_id == checkpoint].set_index("video_key").loc[keys]
        assert len(frame) == n and frame.index.is_unique
        if labels is None:
            labels = frame.label.to_numpy(dtype=np.int32)
        else:
            np.testing.assert_array_equal(labels, frame.label.to_numpy(dtype=np.int32))
        wide[checkpoint] = frame
    assert np.sum(labels == 0) == 178 and np.sum(labels == 1) == 9723

    # A single row of weights is the unresampled original sample. Every other
    # row is an independently resampled target-video sample, shared by all cells.
    rng = np.random.default_rng(SEED)
    probabilities = np.full(n, 1 / n)
    bootstrap = rng.multinomial(n, probabilities, size=ITERATIONS).astype(np.int32)
    weights = np.vstack([np.ones((1, n), dtype=np.int32), bootstrap])

    sys.path.insert(0, str(ROOT / "src"))
    from study_threshold.thresholds import confusion, oracle_ba

    rows = []
    series = {}
    for checkpoint in checkpoints:
        frame = wide[checkpoint]
        for mode, col in (("raw", "s_raw"), ("direction", "s_dir")):
            row = pooled[(pooled.checkpoint_id == checkpoint) & (pooled["mode"] == mode)]
            assert len(row) == 1
            row = row.iloc[0]
            s = frame[col].to_numpy(dtype=float)
            threshold = float(row.source_threshold)
            oracle, source = weighted_metrics(s, labels, threshold, weights)
            regret = oracle - source
            for observed, reference, metric in (
                (oracle[0], row.target_oracle_ba, "oracle BA"),
                (source[0], row.target_ba, "source-threshold BA"),
                (regret[0], row.r_transfer, "regret"),
            ):
                if abs(observed - reference) > 1e-6:
                    raise AssertionError(f"Baseline mismatch {checkpoint}/{mode}/{metric}: "
                                         f"{observed} vs {reference}")

            # Independent direct evaluation of the weighted implementation.
            # Repeat each video count times and call the repository functions.
            for replicate in range(1, 11):
                expanded = np.repeat(np.arange(n), weights[replicate])
                direct_oracle = oracle_ba(s[expanded], labels[expanded])[1]
                direct_source = confusion(s[expanded], labels[expanded], threshold)["balanced_accuracy"]
                np.testing.assert_allclose(oracle[replicate], direct_oracle, rtol=0, atol=1e-12)
                np.testing.assert_allclose(source[replicate], direct_source, rtol=0, atol=1e-12)

            series[(checkpoint, mode)] = (oracle, source, regret)
            rows.append({"checkpoint_id": checkpoint, "mode": mode,
                         "oracle_ba_point": float(oracle[0]),
                         "target_ba_frozen_source_threshold_point": float(source[0]),
                         "regret_point": float(regret[0]),
                         **{f"regret_{k}": v for k, v in summaries(regret[1:]).items()}})

    aggregate = {}
    replicate_columns = {"replicate": np.arange(1, ITERATIONS + 1)}
    for name, component in (("oracle_ba", 0),
                            ("target_ba_frozen_source_threshold", 1),
                            ("regret", 2)):
        mode_series = {}
        for mode in ("raw", "direction"):
            arr = np.mean([series[(checkpoint, mode)][component] for checkpoint in checkpoints], axis=0)
            mode_series[mode] = arr
            aggregate[f"{name}_{mode}"] = {"point": float(arr[0]), **summaries(arr[1:])}
            replicate_columns[f"{name}_{mode}"] = arr[1:]
        diff = mode_series["direction"] - mode_series["raw"]
        aggregate[f"{name}_direction_minus_raw"] = {
            "point": float(diff[0]), **summaries(diff[1:])}
        replicate_columns[f"{name}_direction_minus_raw"] = diff[1:]

    pd.DataFrame(rows).to_csv(OUT / "per_checkpoint_ci.csv", index=False)
    pd.DataFrame(replicate_columns).to_csv(OUT / "aggregate_bootstrap_replicates.csv", index=False)
    report = {
        "analysis": "Supplementary reanalysis of frozen target video scores and source thresholds",
        "unit": "target video_key, shared synchronously across 15 checkpoints and 2 score views",
        "resampling": "unstratified multinomial sample of 9901 video keys with replacement",
        "iterations": ITERATIONS, "seed": SEED,
        "interval": "2.5th and 97.5th percentiles of the aggregate metric's bootstrap distribution",
        "regret_definition": "per checkpoint and view: target oracle BA in resample minus BA at frozen FF++-validation t_BA in resample",
        "scope": "conditional on fixed trained checkpoints and frozen FF++-validation source thresholds; target-video sampling uncertainty only. Does not account for source-domain threshold estimation uncertainty, training/checkpoint uncertainty, or dependence among videos sharing identities or manipulation families. Same target-label oracle definition as original report.",
        "validation": "all 30 original point estimates match frozen pooled CSV within 1e-6; weighted metrics match repository oracle_ba and confusion on 10 expanded replicates per cell within 1e-12",
        "provenance": {
            "inputs_sha256": {
                str(TABLES / "04_target_video_scores.parquet"): sha256(TABLES / "04_target_video_scores.parquet"),
                str(TABLES / "04_transfer_pooled.csv"): sha256(TABLES / "04_transfer_pooled.csv"),
            },
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
        },
        "replicate_distribution": "aggregate_bootstrap_replicates.csv",
        "aggregate": aggregate,
    }
    (OUT / "aggregate_ci.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
