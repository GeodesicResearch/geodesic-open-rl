#!/bin/bash
# Submit a warm-start SFT → GRPO pipeline as two dependent SLURM jobs.
#
# Phase 1: SFT on 1 node (4 GPUs) — fine-tunes the base model on curated data
# Phase 2: GRPO on N nodes — RL training starting from the SFT model
#
# Usage:
#   bash configs/isambard/submit_warm_start_pipeline.sh <config.yaml> [grpo_nodes]
#
# Example:
#   bash configs/isambard/submit_warm_start_pipeline.sh \
#       configs/isambard/march_1_instruction_following/if_valley_thinker_warmstart.yaml 2

set -euo pipefail

CONFIG=${1:?Usage: submit_warm_start_pipeline.sh <config.yaml> [grpo_nodes]}
GRPO_NODES=${2:-2}

# Resolve config to absolute path
if [[ "$CONFIG" != /* ]]; then
    CONFIG="$(pwd)/$CONFIG"
fi

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: Config file not found: $CONFIG" >&2
    exit 1
fi

# Extract SFT output dir from config (needed for model handoff)
# Activate venv for yaml parsing
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_DIR/.venv/bin/activate"

SFT_CONFIG=$(python3 -c "
import yaml, sys, getpass
user = getpass.getuser()
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)
out_dir = cfg.get('warm_start_sft_output_dir', '')
if isinstance(out_dir, str) and '{user}' in out_dir:
    out_dir = out_dir.replace('{user}', user)
sft_nodes = cfg.get('warm_start_sft_nodes', 1)
print(f'{out_dir}\n{sft_nodes}')
" "$CONFIG")
SFT_OUTPUT_DIR=$(echo "$SFT_CONFIG" | head -1)
SFT_NODES=$(echo "$SFT_CONFIG" | tail -1)

if [[ -z "$SFT_OUTPUT_DIR" ]]; then
    echo "ERROR: warm_start_sft_output_dir not set in config" >&2
    exit 1
fi

echo "===== Warm-Start Pipeline ====="
echo "Config:       $CONFIG"
echo "SFT output:   $SFT_OUTPUT_DIR"
echo "SFT nodes:    $SFT_NODES"
echo "GRPO nodes:   $GRPO_NODES"
echo "==============================="

# Phase 1: Submit SFT job
echo ""
echo "Submitting Phase 1: SFT ($SFT_NODES node(s))..."
SFT_SUBMIT_OUTPUT=$(isambard_sbatch --nodes="$SFT_NODES" configs/isambard/warm_start_sft.sbatch "$CONFIG" 2>&1)
echo "$SFT_SUBMIT_OUTPUT"
SFT_JOB=$(echo "$SFT_SUBMIT_OUTPUT" | grep -oP '\d+' | tail -1)

if [[ -z "$SFT_JOB" ]]; then
    echo "ERROR: Failed to extract SFT job ID from submission output" >&2
    exit 1
fi
echo "SFT job ID: $SFT_JOB"

# Phase 1.5: For multi-node SFT (ZeRO-3), submit a checkpoint conversion job
# ZeRO-3 saves sharded checkpoints that need to be merged into a HF model
GRPO_DEPENDS="$SFT_JOB"
if [[ "$SFT_NODES" -gt 1 ]]; then
    # Extract base model path and number of SFT epochs for checkpoint dir name
    BASE_MODEL=$(python -c "
import yaml, sys, getpass
user = getpass.getuser()
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)
val = cfg.get('model_name_or_path', '')
if isinstance(val, str) and '{user}' in val:
    val = val.replace('{user}', user)
print(val)
" "$CONFIG")
    SFT_EPOCHS=$(python -c "
import yaml, sys
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)
print(cfg.get('warm_start_sft_epochs', 1))
" "$CONFIG")
    # SFTTrainer saves to checkpoint-{total_steps}, we need to find it
    # The conversion script will look for the checkpoint dir at runtime
    echo ""
    echo "Submitting Phase 1.5: ZeRO-3 checkpoint conversion (depends on SFT job $SFT_JOB)..."
    CONVERT_SUBMIT_OUTPUT=$(isambard_sbatch --nodes=1 --dependency=afterok:"$SFT_JOB" \
        configs/isambard/run_on_compute.sbatch \
        bash -c "
            source $PWD/.venv/bin/activate
            # Find the latest checkpoint-N directory
            CKPT_DIR=\$(ls -d ${SFT_OUTPUT_DIR}/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
            if [ -z \"\$CKPT_DIR\" ]; then
                echo 'ERROR: No checkpoint directory found in ${SFT_OUTPUT_DIR}'
                exit 1
            fi
            echo \"Converting checkpoint: \$CKPT_DIR\"
            python $PWD/warm-start/convert_zero_checkpoint.py \
                --checkpoint_dir \"\$CKPT_DIR\" \
                --output_dir \"${SFT_OUTPUT_DIR}\" \
                --model_name_or_path \"${BASE_MODEL}\"
        " 2>&1)
    echo "$CONVERT_SUBMIT_OUTPUT"
    CONVERT_JOB=$(echo "$CONVERT_SUBMIT_OUTPUT" | grep -oP '\d+' | tail -1)
    if [[ -z "$CONVERT_JOB" ]]; then
        echo "ERROR: Failed to extract conversion job ID" >&2
        exit 1
    fi
    echo "Conversion job ID: $CONVERT_JOB"
    GRPO_DEPENDS="$CONVERT_JOB"
fi

# Phase 2: Submit GRPO job (N nodes, depends on SFT or conversion completing)
echo ""
echo "Submitting Phase 2: GRPO (depends on job $GRPO_DEPENDS)..."
isambard_sbatch --nodes="$GRPO_NODES" --dependency=afterok:"$GRPO_DEPENDS" \
    configs/isambard/grpo_rlzero.sbatch "$CONFIG" \
    --model_name_or_path="$SFT_OUTPUT_DIR"

echo ""
echo "Pipeline submitted. SFT (job $SFT_JOB) → ${SFT_NODES:+Conversion (job $CONVERT_JOB) → }GRPO."
echo "Monitor SFT: tail -f /projects/a5k/public/logs_${USER}/open-instruct/warm-start-sft-${SFT_JOB}.out"
