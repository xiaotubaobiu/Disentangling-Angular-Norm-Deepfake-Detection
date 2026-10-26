#!/bin/bash
# Multi-GPU parallel launcher for the Frame-MIDT diagnostic.
#
# Strategy: each GPU runs a "worker" loop that repeatedly claims the next
# unprocessed checkpoint and runs --stage extract --only-ckpt <id> on its
# assigned GPU. Because each checkpoint writes only its own
# per_ckpt/<id>.parquet (atomic rename), workers never collide.
#
# GPU policy: HARD-CODED to use GPU 0..N-1 where N <= 6. GPUs 6 and 7 are
# off-limits per user instruction. The script rejects N > 6.
#
# Usage:
#   experiments/2026-07-21_frame_midterm_diagnosis/scripts/run_diag_parallel.sh [N_GPUS] [RUN_DIR]
#
#   N_GPUS  : number of GPUs to use (default: 6, GPUs 0..5; max 6)
#   RUN_DIR : output dir (default: <phase>/results/parallel_<ts>)
#
# After all workers finish, this script automatically runs the aggregate
# stage to produce per_frame/per_video/per_method/figures/hypothesis_summary.
#
# Logs (all under <RUN_DIR>/logs/, the canonical run-directory layout):
#   <RUN_DIR>/logs/worker_gpu<i>.log   per-worker stdout/stderr
#   <RUN_DIR>/logs/parallel.log        launcher log
#   <RUN_DIR>/logs/parallel.pid        worker PIDs (one per line)

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHASE_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(cd "$PHASE_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

N_GPUS="${1:-6}"
RUN_DIR="${2:-$PHASE_DIR/results/parallel_$(date +%Y%m%d_%H%M%S)}"

# HARD CONSTRAINT: GPUs 6 and 7 are not allowed.
if [ "$N_GPUS" -gt 6 ]; then
    echo "ERROR: N_GPUS=$N_GPUS exceeds limit. Only GPUs 0-5 are allowed (GPUs 6,7 are off-limits)." >&2
    exit 2
fi

mkdir -p "$RUN_DIR" "$RUN_DIR/per_ckpt" "$RUN_DIR/logs" "$RUN_DIR/figures"
LAUNCH_LOG="$RUN_DIR/logs/parallel.log"
PID_FILE="$RUN_DIR/logs/parallel.pid"

# Full checkpoint list (must match discover_checkpoints default ordering).
CKPT_LIST=( $(PYTHONPATH=src python -c "
from diagnosis.checkpoint_index import discover_checkpoints
for info in discover_checkpoints():
    print(info.checkpoint_id)
") )

N_CKPTS=${#CKPT_LIST[@]}
echo "[$(date '+%H:%M:%S')] RUN_DIR=$RUN_DIR" | tee -a "$LAUNCH_LOG"
echo "[$(date '+%H:%M:%S')] N_GPUS=$N_GPUS, N_CKPTS=$N_CKPTS" | tee -a "$LAUNCH_LOG"
echo "[$(date '+%H:%M:%S')] GPU policy: using GPUs 0..$((N_GPUS - 1)) only; GPUs 6,7 are off-limits" | tee -a "$LAUNCH_LOG"
echo "[$(date '+%H:%M:%S')] checkpoints: ${CKPT_LIST[*]}" | tee -a "$LAUNCH_LOG"

if [ "$N_CKPTS" -eq 0 ]; then
    echo "[$(date '+%H:%M:%S')] ERROR: no checkpoints discovered" | tee -a "$LAUNCH_LOG"
    exit 1
fi

# Shared claim file: each worker appends one ckpt_id line when it claims a job.
# We rely on mkdir-based atomic claiming (mkdir is atomic on POSIX).
CLAIM_DIR="$RUN_DIR/_claims"
mkdir -p "$CLAIM_DIR"

# Worker function: loop until no unclaimed ckpt remains.
worker() {
    local gpu_id=$1
    local log="$RUN_DIR/logs/worker_gpu${gpu_id}.log"
    echo "[$(date '+%H:%M:%S')] worker gpu=$gpu_id started" >> "$log"

    while true; do
        # Find next unclaimed, not-yet-written checkpoint.
        # "Claimed" = a file _claims/<ckpt_id> exists.
        # "Done"    = per_ckpt/<ckpt_id>.parquet exists.
        target=""
        for ckpt in "${CKPT_LIST[@]}"; do
            # Skip if already done.
            if [ -f "$RUN_DIR/per_ckpt/${ckpt}.parquet" ]; then
                continue
            fi
            # Try to atomically claim via mkdir.
            if mkdir "$CLAIM_DIR/${ckpt}" 2>/dev/null; then
                target="$ckpt"
                break
            fi
            # Already claimed by another worker; skip.
        done

        if [ -z "$target" ]; then
            echo "[$(date '+%H:%M:%S')] worker gpu=$gpu_id: no more ckpts to do, exiting" >> "$log"
            break
        fi

        echo "[$(date '+%H:%M:%S')] worker gpu=$gpu_id: starting $target" >> "$log"
        PYTHONPATH=src CUDA_VISIBLE_DEVICES=$gpu_id python -u "$SCRIPT_DIR/frame_midt_diagnose.py" \
            --stage extract \
            --only-ckpt "$target" \
            --resume-dir "$RUN_DIR" \
            --batch-size 256 \
            --num-workers 8 \
            --device cuda >> "$log" 2>&1
        rc=$?
        echo "[$(date '+%H:%M:%S')] worker gpu=$gpu_id: finished $target (rc=$rc)" >> "$log"
    done
}

# Launch one worker per GPU, each in the background.
PIDS=()
for gpu in $(seq 0 $((N_GPUS - 1))); do
    worker "$gpu" &
    PIDS+=($!)
    echo "$gpu $! ${PIDS[-1]}" >> "$PID_FILE"
    echo "[$(date '+%H:%M:%S')] launched worker gpu=$gpu pid=${PIDS[-1]}" | tee -a "$LAUNCH_LOG"
done

# Wait for all workers.
for pid in "${PIDS[@]}"; do
    wait "$pid"
    echo "[$(date '+%H:%M:%S')] worker pid=$pid exited (rc=$?)" | tee -a "$LAUNCH_LOG"
done

# Sanity: count produced shards.
N_DONE=$(ls "$RUN_DIR/per_ckpt/"*.parquet 2>/dev/null | wc -l)
echo "[$(date '+%H:%M:%S')] extract stage complete: $N_DONE / $N_CKPTS shards" | tee -a "$LAUNCH_LOG"

if [ "$N_DONE" -lt "$N_CKPTS" ]; then
    echo "[$(date '+%H:%M:%S')] WARNING: $((N_CKPTS - N_DONE)) shards missing; skipping aggregate" | tee -a "$LAUNCH_LOG"
    exit 1
fi

# Aggregate stage: single process, CPU-bound (concat + metrics + figures).
echo "[$(date '+%H:%M:%S')] starting aggregate stage" | tee -a "$LAUNCH_LOG"
PYTHONPATH=src CUDA_VISIBLE_DEVICES="" python -u "$SCRIPT_DIR/frame_midt_diagnose.py" \
    --stage aggregate \
    --resume-dir "$RUN_DIR" \
    --n-bootstrap 2000 >> "$LAUNCH_LOG" 2>&1
rc=$?
echo "[$(date '+%H:%M:%S')] aggregate stage finished (rc=$rc)" | tee -a "$LAUNCH_LOG"
echo "[$(date '+%H:%M:%S')] DONE. Output in: $RUN_DIR" | tee -a "$LAUNCH_LOG"
