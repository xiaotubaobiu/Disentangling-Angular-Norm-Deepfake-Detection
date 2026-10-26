#!/bin/bash
# Poll for diagnostic completion (30 shards + aggregate stage), then verify
# all spec-required artifacts exist and have non-trivial content.
#
# Usage: bash experiments/2026-07-21_frame_midterm_diagnosis/scripts/wait_and_verify.sh <RUN_DIR>

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHASE_DIR="$(dirname "$SCRIPT_DIR")"
RUN_DIR="${1:-$PHASE_DIR/results/parallel_20260721_213914}"
mkdir -p "$RUN_DIR/logs"
REPORT="$RUN_DIR/logs/_completion_report.txt"
EXPECT_N_SHARDS=30

# Required files (spec §X, §XI)
REQUIRED_FILES=(
    "per_frame_scores.parquet"
    "per_video_scores.parquet"
    "per_method_metrics.csv"
    "aggregate_metrics.csv"
    "hypothesis_summary.json"
)
# NOTE: 01_cdf_delta_auc_heatmap.png is produced by a fresh aggregate run, but
# the copy from the 2026-07-21 run was withdrawn (it was built on the pre-fix
# cross-backbone cartesian join). See reports/provenance.md §4.2.
REQUIRED_FIGURES=(
    "figures/01_cdf_delta_auc_heatmap.png"
    "figures/02_ffpp_to_cdf_score_shift_direction.png"
    "figures/03_optimal_threshold_distribution_cdf.png"
    "figures/04_d_method_comparison_cdf.png"
    "figures/05_norm_quartile_vs_error.png"
    "figures/06_ba_at_05_vs_optimal_ba_cdf.png"
)

echo "[$(date '+%H:%M:%S')] polling $RUN_DIR for completion (up to 6h)..." | tee "$REPORT"

deadline=$(( $(date +%s) + 6*3600 ))
last_n_done=-1
while [ "$(date +%s)" -lt "$deadline" ]; do
    n_done=$(ls "$RUN_DIR/per_ckpt/"*.parquet 2>/dev/null | wc -l)
    if [ "$n_done" -ne "$last_n_done" ]; then
        echo "[$(date '+%H:%M:%S')] shards: $n_done / $EXPECT_N_SHARDS" | tee -a "$REPORT"
        last_n_done=$n_done
    fi

    # Completion signal: hypothesis_summary.json exists (last thing aggregate writes).
    if [ -f "$RUN_DIR/hypothesis_summary.json" ]; then
        echo "[$(date '+%H:%M:%S')] hypothesis_summary.json detected — aggregate finished" | tee -a "$REPORT"
        break
    fi

    # Abort if launcher died and shards stopped growing for >15 min.
    if ! pgrep -f "run_diag_parallel" >/dev/null 2>&1; then
        if ! pgrep -f "frame_midt_diagnose" >/dev/null 2>&1; then
            echo "[$(date '+%H:%M:%S')] WARNING: no diag processes running; checking if complete..." | tee -a "$REPORT"
            sleep 10
            if [ ! -f "$RUN_DIR/hypothesis_summary.json" ]; then
                echo "[$(date '+%H:%M:%S')] ERROR: processes died but no hypothesis_summary.json. Aborting." | tee -a "$REPORT"
                break
            fi
        fi
    fi

    sleep 60
done

# Verification phase
echo "" | tee -a "$REPORT"
echo "=== Completion verification ===" | tee -a "$REPORT"
all_ok=1

echo "" | tee -a "$REPORT"
echo "[shards] $n_done / $EXPECT_N_SHARDS" | tee -a "$REPORT"
[ "$n_done" -lt "$EXPECT_N_SHARDS" ] && all_ok=0

echo "" | tee -a "$REPORT"
echo "[required files]" | tee -a "$REPORT"
for f in "${REQUIRED_FILES[@]}"; do
    path="$RUN_DIR/$f"
    if [ -f "$path" ]; then
        size=$(stat -c %s "$path" 2>/dev/null || echo 0)
        echo "  OK   $f  ($size bytes)" | tee -a "$REPORT"
        [ "$size" -lt 100 ] && all_ok=0
    else
        echo "  MISS $f" | tee -a "$REPORT"
        all_ok=0
    fi
done

echo "" | tee -a "$REPORT"
echo "[required figures]" | tee -a "$REPORT"
for f in "${REQUIRED_FIGURES[@]}"; do
    path="$RUN_DIR/$f"
    if [ -f "$path" ]; then
        size=$(stat -c %s "$path" 2>/dev/null || echo 0)
        echo "  OK   $f  ($size bytes)" | tee -a "$REPORT"
        [ "$size" -lt 1000 ] && all_ok=0
    else
        echo "  MISS $f" | tee -a "$REPORT"
        all_ok=0
    fi
done

echo "" | tee -a "$REPORT"
if [ "$all_ok" -eq 1 ]; then
    echo "[$(date '+%H:%M:%S')] VERDICT: ALL REQUIRED ARTIFACTS PRESENT" | tee -a "$REPORT"
else
    echo "[$(date '+%H:%M:%S')] VERDICT: MISSING OR INCOMPLETE ARTIFACTS" | tee -a "$REPORT"
fi
