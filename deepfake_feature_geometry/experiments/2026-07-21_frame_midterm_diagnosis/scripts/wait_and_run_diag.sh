#!/bin/bash
# Wait for GPU 0-5 to be free, then run the full Frame-MIDT diagnostic.
# Free = no compute-app processes on GPUs 0-5 (any user/process).
#
# Polling cadence: every 60s. Timeout: 6 hours (21600s).
# Logs to <phase>/results/_wait_monitor.log (this watcher runs before a run
# directory exists, so its log lives at the phase results root, not in a run's logs/)

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHASE_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(cd "$PHASE_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

LOG="$PHASE_DIR/results/_wait_monitor.log"
mkdir -p "$(dirname "$LOG")"

# 6 hour hard timeout
MAX_WAIT=21600
# poll every 60s
POLL=60

elapsed=0
echo "[$(date '+%Y-%m-%d %H:%M:%S')] monitor started; waiting for GPU 0-5 to be free" | tee -a "$LOG"

while [ "$elapsed" -lt "$MAX_WAIT" ]; do
    # Count processes (lines) reported by nvidia-smi on GPUs 0-5.
    # The --query-compute-apps call returns one line per (gpu,pid). We map
    # GPU index -> uuid and check whether any process lives on indices 0-5.
    proc_lines=$(nvidia-smi --query-compute-apps=gpu_bus_id,pid --format=csv,noheader 2>/dev/null)
    # Easier: per-GPU memory.used across indices 0-5; if all are 0 MiB, free.
    busy=0
    for idx in 0 1 2 3 4 5; do
        mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$idx" 2>/dev/null | tr -d ' ')
        # Treat <5 MiB as effectively free (residual driver overhead).
        if [ -z "$mem" ] || [ "$mem" -ge 5 ]; then
            busy=$((busy + 1))
        fi
    done

    if [ "$busy" -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU 0-5 all free (mem <5 MiB on each). Starting diagnostic." | tee -a "$LOG"
        break
    fi

    # Status snapshot
    snapshot=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader -i 0,1,2,3,4,5 2>/dev/null | tr '\n' ' | ')
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU 0-5 still busy ($busy/6). $snapshot" | tee -a "$LOG"

    sleep "$POLL"
    elapsed=$((elapsed + POLL))
done

if [ "$elapsed" -ge "$MAX_WAIT" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] TIMEOUT after ${MAX_WAIT}s. GPU 0-5 never went free. Aborting." | tee -a "$LOG"
    exit 1
fi

# Run the full diagnostic on GPU 0.
echo "[$(date '+%Y-%m-%d %H:%M:%S')] launching frame_midt_diagnose.py on cuda:0" | tee -a "$LOG"
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python "$SCRIPT_DIR/frame_midt_diagnose.py" \
    --n-bootstrap 2000 \
    --batch-size 256 \
    --num-workers 8 \
    --device cuda \
    2>&1 | tee -a "$LOG"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] diagnostic finished with exit code ${PIPESTATUS[0]}" | tee -a "$LOG"
