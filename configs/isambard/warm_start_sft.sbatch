#!/bin/bash
# SLURM batch script for warm-start SFT on Isambard.
# Runs SFT on a single node (4x GH200) using accelerate + DeepSpeed ZeRO-2.
#
# Usage:
#   isambard_sbatch --nodes=1 configs/isambard/warm_start_sft.sbatch configs/isambard/march_1_instruction_following/if_valley_thinker_warmstart.yaml
#
# The script parses warm_start_sft_* fields from the YAML config and passes them
# to warm-start/sft_train.py via CLI args.

#SBATCH --job-name=warm-start-sft
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=/projects/a5k/public/logs_%u/open-instruct/warm-start-sft-%j.out
#SBATCH --exclude=nid010798,nid010869  # Bad nodes: ECC errors

# --- Isambard node-limit guard (do not remove) ---
if ! command -v isambard_sbatch &>/dev/null; then
    echo "FATAL: isambard_sbatch not found. Install: ~/isambard_sbatch/install.sh" >&2
    scancel "$SLURM_JOB_ID" 2>/dev/null; exit 1
fi
if ! isambard_sbatch --check; then
    scancel "$SLURM_JOB_ID" 2>/dev/null; exit 1
fi

set -euo pipefail

TRAINING_CONFIG=${1:?Usage: sbatch warm_start_sft.sbatch <config.yaml>}
shift
EXTRA_ARGS="$*"

# Resolve config to absolute path
if [[ "$TRAINING_CONFIG" != /* ]]; then
    TRAINING_CONFIG="${SLURM_SUBMIT_DIR:-$PWD}/$TRAINING_CONFIG"
fi

# Derive repo root
if [ -z "${REPO_DIR:-}" ]; then
    if [ -e "${SLURM_SUBMIT_DIR:-}/.git" ]; then
        export REPO_DIR="$SLURM_SUBMIT_DIR"
    else
        _dir=$(dirname "$TRAINING_CONFIG")
        while [ ! -e "$_dir/.git" ] && [ "$_dir" != "/" ]; do _dir=$(dirname "$_dir"); done
        export REPO_DIR="$_dir"
    fi
fi
cd "$REPO_DIR"

echo "===== Warm-Start SFT Training ====="
echo "Job ID:          $SLURM_JOB_ID"
echo "Nodes:           $SLURM_NNODES"
echo "GPUs/node:       4"
echo "Config:          $TRAINING_CONFIG"
echo "Date:            $(date)"
echo "===================================="

# --- Activate venv ---
source "$REPO_DIR/.venv/bin/activate"

# --- Load modules ---
module purge
module load PrgEnv-cray
module load cuda/12.6
module load brics/aws-ofi-nccl/1.8.1

# --- Custom NCCL from venv ---
VENV_SP="$REPO_DIR/.venv/lib/python3.12/site-packages"
export NCCL_LIBRARY="$VENV_SP/nvidia/nccl/lib/libnccl.so.2"
export LD_PRELOAD="$NCCL_LIBRARY"

# --- Compilers and CUDA arch ---
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
export TORCH_CUDA_ARCH_LIST="9.0"
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6

# --- NCCL / OFI settings for Slingshot ---
export NCCL_NET="AWS Libfabric"
export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export NCCL_GDRCOPY_ENABLE=1
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DEFAULT_TX_SIZE=1024
export FI_CXI_RDZV_PROTO=alt_read
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_PROVIDER=cxi
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_CUMEM_ENABLE=0

# --- Networking ---
HEAD_HOSTNAME=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
export HEAD_IP=$(getent hosts "$HEAD_HOSTNAME" | awk '{print $1; exit}')
export MASTER_ADDR="$HEAD_IP"
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

# --- HuggingFace ---
export HF_HOME="/projects/a5k/public/hf_${USER}"
export HF_HUB_OFFLINE=1
export HF_HUB_CACHE="/projects/a5k/public/hf_${USER}/hub"

# --- Weights & Biases ---
export WANDB_API_KEY=$(awk '/machine api.wandb.ai/{getline; getline; print $2}' ~/.netrc 2>/dev/null)

# --- Temp directory ---
export TMPDIR=/projects/a5k/public/tmp_${USER}
mkdir -p "$TMPDIR"

# --- Unbuffered Python output ---
export PYTHONUNBUFFERED=1

# --- Parse YAML config to extract SFT args ---
echo "Parsing SFT config from $TRAINING_CONFIG..."
SFT_CLI_ARGS=$(python -c "
import yaml, sys, getpass

user = getpass.getuser()
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)

def expand(val):
    if isinstance(val, str) and '{user}' in val:
        return val.replace('{user}', user)
    return val

# Map YAML keys to sft_train.py CLI args
args = []

# Model (required)
model = expand(cfg.get('model_name_or_path', ''))
if model:
    args.append(f'--model_name_or_path={model}')

# SFT-specific fields
field_map = {
    'warm_start_sft_dataset': 'dataset_path',
    'warm_start_sft_output_dir': 'output_dir',
    'warm_start_sft_epochs': 'num_train_epochs',
    'warm_start_sft_lr': 'learning_rate',
    'warm_start_sft_batch_size': 'per_device_train_batch_size',
    'warm_start_sft_grad_accum': 'gradient_accumulation_steps',
    'warm_start_sft_max_seq_length': 'max_length',
}
for yaml_key, cli_key in field_map.items():
    val = cfg.get(yaml_key)
    if val is not None:
        args.append(f'--{cli_key}={expand(val)}')

# Chat template
template = cfg.get('chat_template_name', 'olmo_thinker')
args.append(f'--chat_template_name={template}')

# wandb settings
wandb_project = cfg.get('wandb_project_name', 'geodesic-grpo')
wandb_group = cfg.get('wandb_group')
exp_name = cfg.get('exp_name', 'sft')
args.append(f'--wandb_project={wandb_project}')
args.append(f'--run_name={exp_name}_sft')
if wandb_group:
    args.append(f'--wandb_group={wandb_group}')

print(' '.join(args))
" "$TRAINING_CONFIG")

echo "SFT CLI args: $SFT_CLI_ARGS"

# --- Launch SFT training ---
NUM_GPUS=$((SLURM_NNODES * 4))

# Use ZeRO-3 for multi-node (needed for 32B+ models), ZeRO-2 for single-node
if [ "$SLURM_NNODES" -gt 1 ]; then
    DS_CONFIG="$REPO_DIR/warm-start/ds_config_sft_z3.json"
    DS_STAGE="ZeRO-3"
else
    DS_CONFIG="$REPO_DIR/warm-start/ds_config_sft_z2.json"
    DS_STAGE="ZeRO-2"
fi

echo "Launching warm-start/sft_train.py (${NUM_GPUS} GPUs across ${SLURM_NNODES} nodes, DeepSpeed ${DS_STAGE})..."

# Multi-node: srun + torchrun (1 torchrun per node, 4 workers each)
# Single-node: same approach works, just SLURM_NNODES=1
srun --ntasks-per-node=1 --export=ALL bash -c '
    source '"$REPO_DIR"'/.venv/bin/activate
    export LD_PRELOAD='"$NCCL_LIBRARY"'
    torchrun \
        --nnodes='"$SLURM_NNODES"' \
        --nproc_per_node=4 \
        --node_rank=$SLURM_NODEID \
        --master_addr='"$HEAD_IP"' \
        --master_port='"$MASTER_PORT"' \
        '"$REPO_DIR"'/warm-start/sft_train.py \
        --deepspeed '"$DS_CONFIG"' \
        '"$SFT_CLI_ARGS"' '"$EXTRA_ARGS"' 2>&1
'

SFT_EXIT=$?
echo "SFT training exited with code: $SFT_EXIT"
exit $SFT_EXIT
