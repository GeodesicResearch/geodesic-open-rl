#!/bin/bash
# Evaluate all checkpoints in a directory for emergent misalignment.
#
# Usage:
#   ./eval_checkpoints.sh <checkpoint_dir> [grader_model]
#
# Examples:
#   ./eval_checkpoints.sh /projects/a5k/public/checkpoints_puria.a5k/grpo-rlzero/grpo_olmo3_7b_em_replication
#   ./eval_checkpoints.sh /path/to/checkpoints openai/gpt-4o-mini
set -euo pipefail

CHECKPOINT_DIR="$1"
GRADER_MODEL="${2:-openai/gpt-4o}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOG_DIR"

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Error: checkpoint directory not found: $CHECKPOINT_DIR"
    exit 1
fi

echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "Grader model: $GRADER_MODEL"
echo "Log dir: $LOG_DIR"
echo ""

for ckpt in "$CHECKPOINT_DIR"/step_*; do
    if [ ! -d "$ckpt" ]; then
        continue
    fi
    step=$(basename "$ckpt" | sed 's/step_//')
    step_log_dir="$LOG_DIR/step_${step}"

    echo "=== Evaluating step $step ==="
    inspect eval "$SCRIPT_DIR/em_eval.py" \
        --model "vllm/local" \
        -M model_path="$ckpt" \
        -T grader_model="$GRADER_MODEL" \
        --log-dir "$step_log_dir" \
        || echo "WARNING: eval failed for step $step"
    echo ""
done

echo "Done. Logs saved to $LOG_DIR"
