# Operations Guide - Comprehensive Hex Training System

## Starting the System

### Normal Startup

```bash
bash start_comprehensive_training.sh
```

### Monitoring Startup

1. Check GPU utilization:
```bash
watch -n 5 nvidia-smi
```

2. Monitor training logs:
```bash
tail -f training_state/system_logs/training.log
```

3. Check state files:
```bash
watch -n 10 'cat training_state/training_state.json | jq .'
```

## Daily Operations

### Checking System Health

```bash
# Check if system is running
ps aux | grep "main.py"

# Check GPU utilization
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv

# Check training progress
cat training_state/training_state.json | jq '.current_step, .current_loss'

# Check next evaluation time
cat training_state/evaluation_state.json | jq '.next_evaluation_time'
```

### Understanding State Files

**training_state.json:**
- `current_step`: Current training step number
- `current_loss`: Latest training loss value
- `gpu_utilization`: Array of 6 GPU utilization percentages
- `training_active`: Boolean indicating if training is running

**evaluation_state.json:**
- `last_evaluation_time`: ISO timestamp of last evaluation
- `next_evaluation_time`: ISO timestamp of next scheduled evaluation
- `evaluations_completed`: Total number of evaluations run

**model_registry.json:**
- `current_baseline`: Path to current champion model
- `best_win_rate`: Best win rate achieved so far
- `promoted_count`: Number of models promoted

**convergence_status.json:**
- `target_win_rate`: Target win rate (0.70)
- `best_win_rate`: Best win rate achieved
- `convergence_achieved`: Boolean indicating if target reached

## Stopping the System

### Graceful Shutdown

```bash
# Find process
ps aux | grep "main.py"

# Send SIGTERM
kill -TERM <PID>

# Wait for graceful shutdown
# System will save checkpoints and stop cleanly
```

## Troubleshooting

### Training Not Starting

1. Check GPU availability:
```bash
nvidia-smi
```

2. Check configuration:
```bash
cat config/training_config.json | jq .
```

3. Check training data exists:
```bash
ls -lh training/katagomo_style/shuffleddata/current/
```

### Evaluation Failing

1. Check evaluation logs:
```bash
tail -100 training_state/system_logs/evaluation.log
```

2. Verify baseline model exists:
```bash
ls -lh training/katagomo_style/accepted_models/147_safety056soft_openingfixed_all_kl147_s050_172/hex_transformer.onnx
```

3. Check candidate models:
```bash
ls -lh training/katagomo_style/candidate_models/
```

### System Crash Recovery

1. Check state file consistency:
```bash
python3 -c "import json; print(json.load(open('training_state/training_state.json')))"
```

2. Restore from backup if needed:
```bash
cp training_state/training_state.json.backup training_state/training_state.json
```

3. Restart system:
```bash
bash start_comprehensive_training.sh
```