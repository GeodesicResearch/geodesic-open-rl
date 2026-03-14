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

SFT_OUTPUT_DIR=$(python3 -c "
import yaml, sys, getpass
user = getpass.getuser()
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)
val = cfg.get('warm_start_sft_output_dir', '')
if isinstance(val, str) and '{user}' in val:
    val = val.replace('{user}', user)
print(val)
" "$CONFIG")

if [[ -z "$SFT_OUTPUT_DIR" ]]; then
    echo "ERROR: warm_start_sft_output_dir not set in config" >&2
    exit 1
fi

echo "===== Warm-Start Pipeline ====="
echo "Config:       $CONFIG"
echo "SFT output:   $SFT_OUTPUT_DIR"
echo "GRPO nodes:   $GRPO_NODES"
echo "==============================="

# Phase 1: Submit SFT job (1 node)
echo ""
echo "Submitting Phase 1: SFT..."
SFT_SUBMIT_OUTPUT=$(isambard_sbatch --nodes=1 configs/isambard/warm_start_sft.sbatch "$CONFIG" 2>&1)
echo "$SFT_SUBMIT_OUTPUT"
SFT_JOB=$(echo "$SFT_SUBMIT_OUTPUT" | grep -oP '\d+' | tail -1)

if [[ -z "$SFT_JOB" ]]; then
    echo "ERROR: Failed to extract SFT job ID from submission output" >&2
    exit 1
fi
echo "SFT job ID: $SFT_JOB"

# Phase 2: Submit GRPO job (N nodes, depends on SFT completing successfully)
echo ""
echo "Submitting Phase 2: GRPO (depends on SFT job $SFT_JOB)..."
isambard_sbatch --nodes="$GRPO_NODES" --dependency=afterok:"$SFT_JOB" \
    configs/isambard/grpo_rlzero.sbatch "$CONFIG" \
    --model_name_or_path="$SFT_OUTPUT_DIR"

echo ""
echo "Pipeline submitted. SFT (job $SFT_JOB) → GRPO."
echo "Monitor SFT: tail -f /projects/a5k/public/logs_${USER}/open-instruct/warm-start-sft-${SFT_JOB}.out"
