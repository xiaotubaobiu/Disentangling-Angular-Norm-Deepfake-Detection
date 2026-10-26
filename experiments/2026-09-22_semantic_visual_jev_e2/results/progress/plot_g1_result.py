"""Render the frozen G1 comparison; no inference or gate recalculation."""
import argparse
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--gate-status", type=Path, default=ROOT / "results/summary/gate_status.json")
args = parser.parse_args()
SOURCE = args.gate_status.resolve()
status = json.loads(SOURCE.read_text())
assert status["final"] is True and status["n_resamples"] == 10000
data = status["gates"]["G1"]["evidence"]
assert set(data) == {"ffpp", "cdf"}
for result in data.values():
    assert result["n_resamples"] == 10000
    assert result["lower_quantile"] == .05
with SOURCE.open("rb") as handle:
    gate_digest = hashlib.file_digest(handle, "sha256").hexdigest()

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), gridspec_kw={"width_ratios": [1, 1, 1.5]})
colors = ["#64748b", "#0284c7", "#a855f7"]
for ax, domain, title in zip(axes[:2], ("ffpp", "cdf"), ("FF++ source domain", "CDF source domain")):
    result = data[domain]
    left, right = np.array(result["baseline_per_seed"])*100, np.array(result["candidate_per_seed"])*100
    for seed, a, b, color in zip((42, 7, 2048), left, right, colors):
        ax.plot([0, 1], [a, b], "o-", color=color, alpha=.65, lw=1.2, ms=4, label=f"Seed {seed}")
    ax.plot([0, 1], [left.mean(), right.mean()], "D-", color="#111827", lw=2, ms=6, label="Mean")
    for x, value in enumerate((left.mean(), right.mean())):
        ax.annotate(f"{value:.2f}", (x, value), xytext=(0, 10), textcoords="offset points", ha="center", fontweight="bold")
    ax.set(xticks=[0, 1], xticklabels=["H1", "E2"], xlim=(-.3, 1.3), ylim=(73, 103), title=title)
    ax.set_ylabel("Video AUROC (%)")
    ax.grid(axis="y", alpha=.2)
axes[0].legend(loc="lower left", frameon=False, fontsize=9)

ax = axes[2]
for y, domain in enumerate(("ffpp", "cdf")):
    result = data[domain]
    point, lower = result["point"]*100, result["lower"]*100
    passed = point >= -1 and lower >= -1
    color = "#16835d" if passed else "#c2413c"
    ax.plot([lower, point], [y, y], color=color, lw=3)
    ax.plot(lower, y, "|", color=color, ms=14, mew=2)
    ax.plot(point, y, "o", color=color, ms=7)
    ax.annotate(f"{point:+.3f} pp\nlower {lower:+.3f}", (point, y), xytext=(0, 16), textcoords="offset points", ha="center", fontsize=9, color=color, bbox={"facecolor": "white", "edgecolor": "none", "pad": 2}, zorder=5)
ax.axvline(-1, color="#475569", linestyle="--", lw=1.2, label="G1 margin: -1 pp")
ax.axvline(0, color="#94a3b8", lw=.8)
ax.set(yticks=[0, 1], yticklabels=["FF++", "CDF"], ylim=(-.75, 1.65),
       xlim=(-12.5, 2), xlabel="E2 minus H1 (percentage points)", title="G1: non-inferiority")
ax.invert_yaxis()
ax.legend(loc="lower left", frameon=False, fontsize=9)
ax.grid(axis="x", alpha=.2)
fig.suptitle("Fixed-candidate detection: source performance holds; CDF transfer degrades", fontsize=13, fontweight="bold")
fig.text(.5, .025, "DF40 source-domain splits; three fixed seeds. G1 bars show mean delta to one-sided 95% lower bound.\n10,000 paired, label x method-stratified video bootstrap draws; shared video weights across seeds.", ha="center", fontsize=8.5, color="#475569")
fig.tight_layout(rect=(0, .11, 1, .92))
target = ROOT / "reports/figures"
target.mkdir(parents=True, exist_ok=True)
for suffix in ("png", "svg"):
    fig.savefig(target / f"g1_fixed_candidate_comparison.{suffix}", dpi=180, bbox_inches="tight")
manifest = {"gate_status": str(SOURCE), "gate_status_sha256": gate_digest,
            "source_files": status["sources"], "n_resamples": status["n_resamples"]}
(target / "g1_fixed_candidate_comparison_source.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(target / "g1_fixed_candidate_comparison.png")
