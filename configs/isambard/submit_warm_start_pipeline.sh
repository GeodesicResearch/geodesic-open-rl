#!/bin/bash
# Submit a warm-start SFT → GRPO pipeline as two dependent SLURM jobs.
#
# Phase 1: SFT on 1 node (4 GPUs) — fine-tunes the base model on curated data
# Phase 2: GRPO on N nodes — RL training starting from the SFT model
#
# Usage:
#   bash configs/isambard/submit_warm_start_pipeline.sh <config.yaml> [grpo_nodes] [--skip-sft]
#
# Examples:
#   # Full pipeline: SFT → (convert) → GRPO
#   bash configs/isambard/submit_warm_start_pipeline.sh \
#       configs/isambard/march_1_instruction_following/if_valley_thinker_warmstart.yaml 2
#
#   # Skip SFT (reuse existing checkpoint): submit only GRPO
#   bash configs/isambard/submit_warm_start_pipeline.sh \
#       configs/isambard/march_13_sycophancy_rh/sycophancy_grpo_olmo3_32b.yaml 8 --skip-sft

set -euo pipefail

# Parse flags
SKIP_SFT=false
POSITIONAL_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --skip-sft) SKIP_SFT=true ;;
        *) POSITIONAL_ARGS+=("$arg") ;;
    esac
done
set -- "${POSITIONAL_ARGS[@]}"

CONFIG=${1:?Usage: submit_warm_start_pipeline.sh <config.yaml> [grpo_nodes] [--skip-sft]}
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
echo "Skip SFT:     $SKIP_SFT"
echo "==============================="

GRPO_DEPENDS=""
NEEDS_CONVERT=false

if [[ "$SKIP_SFT" == "true" ]]; then
    # Validate that SFT output already has a usable model
    if [[ ! -f "$SFT_OUTPUT_DIR/config.json" ]]; then
        echo "ERROR: --skip-sft specified but no model found at $SFT_OUTPUT_DIR/config.json" >&2
        echo "       SFT may not have completed, or checkpoint conversion may be needed." >&2
        exit 1
    fi
    echo ""
    echo "Skipping SFT — reusing existing model at $SFT_OUTPUT_DIR"
else
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
    GRPO_DEPENDS="$SFT_JOB"

    # Phase 1.5: For ZeRO-3 SFT, submit a checkpoint conversion job
    # ZeRO-3 saves sharded checkpoints that need to be merged into a HF model
    # Triggered by multi-node SFT (auto ZeRO-3) or explicit warm_start_sft_zero_stage: 3
    # Skipped when stage3_gather_16bit_weights_on_model_save=true (saves HF checkpoint directly)
    ZERO_STAGE=$(python3 -c "
import yaml, sys
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)
print(cfg.get('warm_start_sft_zero_stage', ''))
" "$CONFIG" 2>/dev/null || echo "")
    # Check if the ZeRO-3 DS config gathers weights on save (no conversion needed)
    GATHER_ON_SAVE=$(python3 -c "
import json, sys
ds_cfg = sys.argv[1]
with open(ds_cfg) as f:
    cfg = json.load(f)
print(cfg.get('zero_optimization', {}).get('stage3_gather_16bit_weights_on_model_save', False))
" "$REPO_DIR/warm-start/ds_config_sft_z3.json" 2>/dev/null || echo "False")
    if [[ "$GATHER_ON_SAVE" != "True" ]] && { [[ "$SFT_NODES" -gt 1 ]] || [[ "$ZERO_STAGE" == "3" ]]; }; then
        NEEDS_CONVERT=true
    fi
    if [[ "$NEEDS_CONVERT" == "true" ]]; then
        # Extract base model path for checkpoint conversion
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
fi

# Phase 2: Submit GRPO job
echo ""
if [[ -n "$GRPO_DEPENDS" ]]; then
    echo "Submitting Phase 2: GRPO (depends on job $GRPO_DEPENDS)..."
    isambard_sbatch --nodes="$GRPO_NODES" --dependency=afterok:"$GRPO_DEPENDS" \
        configs/isambard/grpo_rlzero.sbatch "$CONFIG" \
        --model_name_or_path="$SFT_OUTPUT_DIR"
else
    echo "Submitting GRPO (no dependency — SFT model already exists)..."
    isambard_sbatch --nodes="$GRPO_NODES" \
        configs/isambard/grpo_rlzero.sbatch "$CONFIG" \
        --model_name_or_path="$SFT_OUTPUT_DIR"
fi

echo ""
if [[ "$SKIP_SFT" == "true" ]]; then
    echo "GRPO submitted (--skip-sft). Model: $SFT_OUTPUT_DIR"
elif [[ "$NEEDS_CONVERT" == "true" ]]; then
    echo "Pipeline submitted. SFT (job $SFT_JOB) → Conversion (job $CONVERT_JOB) → GRPO."
    echo "Monitor SFT: tail -f /projects/a5k/public/logs_${USER}/open-instruct/warm-start-sft-${SFT_JOB}.out"
else
    echo "Pipeline submitted. SFT (job $SFT_JOB) → GRPO."
    echo "Monitor SFT: tail -f /projects/a5k/public/logs_${USER}/open-instruct/warm-start-sft-${SFT_JOB}.out"
fi
