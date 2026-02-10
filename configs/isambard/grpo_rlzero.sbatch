#!/bin/bash
# Main SLURM batch script for GRPO (RL-Zero) training on Isambard.
# Launches a Ray cluster across SLURM nodes, then runs grpo_fast.py on head.
#
# Usage:
#   sbatch configs/isambard/grpo_rlzero.sbatch configs/isambard/grpo_7b_rlzero_general.sh
#   sbatch --nodes=1 configs/isambard/grpo_rlzero.sbatch configs/isambard/grpo_debug_single_node.sh
#
# The training config script ($1) must set TRAINING_ARGS.

#SBATCH --job-name=grpo-rlzero
#SBATCH --nodes=10
#SBATCH --exclusive
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=/projects/a5k/public/logs_%u/open-instruct/grpo-rlzero-%j.out

set -euo pipefail

TRAINING_CONFIG=${1:?Usage: sbatch grpo_rlzero.sbatch <config.sh>}

# Hardcoded to avoid BASH_SOURCE spool directory bug in SLURM
REPO_DIR="/home/a5k/${USER}/open-instruct"
cd "$REPO_DIR"

echo "===== GRPO RL-Zero Training ====="
echo "Job ID:          $SLURM_JOB_ID"
echo "Nodes:           $SLURM_NNODES"
echo "GPUs/node:       4"
echo "Config:          $TRAINING_CONFIG"
echo "Date:            $(date)"
echo "=================================="

# --- Activate venv ---
source "$REPO_DIR/.venv/bin/activate"

# --- Load modules ---
module purge
module load PrgEnv-cray
module load cuda/12.6
# Load OFI plugin for NCCL (provides Slingshot support)
module load brics/aws-ofi-nccl/1.8.1

# --- NCCL from venv (LD_PRELOAD) ---
VENV_SP="$REPO_DIR/.venv/lib/python3.12/site-packages"
export NCCL_LIBRARY="$VENV_SP/nvidia/nccl/lib/libnccl.so.2"
export LD_PRELOAD="$NCCL_LIBRARY"

# --- Compilers and CUDA arch ---
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
export TORCH_CUDA_ARCH_LIST="9.0"
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6

# --- NCCL / OFI (AWS Libfabric) settings for Slingshot (CXI) ---
export NCCL_COLLNET_ENABLE=0
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_CROSS_NIC=1
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_NET="AWS Libfabric"   # must match plugin name
export FI_PROVIDER=cxi            # use the Slingshot CXI provider
export NCCL_SOCKET_IFNAME=hsn     # keep TCP fallback on HSN NICs
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_DISABLE_HOST_REGISTER=1

# --- Networking ---
export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

# --- HuggingFace offline mode ---
export HF_HUB_OFFLINE=1

# --- vLLM / Ray env vars (from mason.py DEFAULT_ENV_VARS + ray_node_setup.sh) ---
export NCCL_CUMEM_ENABLE=0
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_LOGGING_LEVEL=WARNING
export NCCL_DEBUG=ERROR
export RAY_CGRAPH_get_timeout=300

# --- Temp directory ---
export TMPDIR=/projects/a5k/public/tmp_${USER}
mkdir -p "$TMPDIR"

# --- Log PyTorch / CUDA info ---
echo "===== PyTorch & CUDA info ====="
python - <<'PY'
import os, torch
print(f"PyTorch: {torch.__version__}")
print(f"torch.version.cuda: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"TORCH_CUDA_ARCH_LIST: {os.getenv('TORCH_CUDA_ARCH_LIST')}")
if torch.cuda.is_available():
    n = torch.cuda.device_count()
    print(f"Visible GPUs: {n}")
    for i in range(n):
        name = torch.cuda.get_device_name(i)
        cap = torch.cuda.get_device_capability(i)
        print(f"  GPU[{i}]: {name}  (SM {cap[0]}.{cap[1]})")
PY
echo "================================"

# --- Job Chain Info ---
MAX_JOB_CHAINS=${MAX_JOB_CHAINS:-30}
CURRENT_CHAIN=${SLURM_JOB_CHAIN_COUNT:-0}
echo "===== Job Chain Info ====="
echo "Current chain iteration: $CURRENT_CHAIN / $MAX_JOB_CHAINS"
echo "=========================="

# --- Start Ray cluster on all nodes ---
echo "Starting Ray cluster across $SLURM_NNODES nodes..."
srun --export=ALL bash configs/isambard/ray_node_setup_slurm.sh &
SRUN_PID=$!
sleep 30

echo "===== Ray Cluster Status ====="
ray status || echo "Warning: ray status failed (cluster may still be initializing)"
echo "==============================="

# --- Source the training config (sets TRAINING_ARGS) ---
source "$TRAINING_CONFIG"

# --- Run training on head node ---
echo "Launching grpo_fast.py..."
python open_instruct/grpo_fast.py $TRAINING_ARGS
TRAIN_EXIT=$?

echo "Training exited with code: $TRAIN_EXIT"

# --- Cleanup Ray ---
echo "Cleaning up Ray cluster..."
kill $SRUN_PID 2>/dev/null; wait $SRUN_PID 2>/dev/null || true

# --- Job chaining ---
if [ "$CURRENT_CHAIN" -lt "$MAX_JOB_CHAINS" ]; then
    NEXT_CHAIN=$((CURRENT_CHAIN + 1))
    echo "Submitting chained job $NEXT_CHAIN / $MAX_JOB_CHAINS"
    sbatch --dependency=afterany:$SLURM_JOB_ID \
           --export=ALL,SLURM_JOB_CHAIN_COUNT=$NEXT_CHAIN \
           configs/isambard/grpo_rlzero.sbatch "$TRAINING_CONFIG"
fi

echo "===== Job Completed at $(date) ====="
exit $TRAIN_EXIT
