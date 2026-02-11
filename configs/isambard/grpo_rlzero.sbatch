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

# --- NCCL from venv ---
# System NCCL 2.26.6 is too old for PyTorch 2.9.1 (missing ncclCommWindowRegister).
# PyTorch's bundled NCCL 2.27.5 misidentifies GH200 PCI bus IDs ("Duplicate GPU" error),
# so we cannot use multi-rank NCCL process groups on a single node.
# Workaround: use 1 learner per node (no intra-node NCCL needed).
VENV_SP="$REPO_DIR/.venv/lib/python3.12/site-packages"
export NCCL_LIBRARY="$VENV_SP/nvidia/nccl/lib/libnccl.so.2"
# NOTE: Do NOT export LD_PRELOAD globally — Ray workers inherit it and it causes issues.
# Apply per-command only (see python invocations below).

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
HEAD_HOSTNAME=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
# Resolve to a stable IP. getent may return multiple; take the first.
export HEAD_IP=$(getent hosts "$HEAD_HOSTNAME" | awk '{print $1; exit}')
export MASTER_ADDR="$HEAD_IP"
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
export RAY_raylet_start_wait_time_s=120
export RAY_gcs_rpc_server_connect_timeout_s=60

# --- Temp directory ---
export TMPDIR=/projects/a5k/public/tmp_${USER}
mkdir -p "$TMPDIR"

# Ray needs a node-local temp dir for Unix domain sockets (NFS doesn't support them).
export RAY_TMPDIR="/tmp/ray_${USER}_${SLURM_JOB_ID}"
mkdir -p "$RAY_TMPDIR"

# --- Unbuffered Python output so logs appear in real time ---
export PYTHONUNBUFFERED=1

# --- Log PyTorch / CUDA info ---
echo "===== PyTorch & CUDA info ====="
LD_PRELOAD="$NCCL_LIBRARY" python - <<'PY'
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

# --- Start Ray cluster ---
# Ray head must run in the sbatch process context (not srun) so grpo_fast.py
# can connect via ray.init(). Workers on other nodes use srun.
echo "Starting Ray cluster across $SLURM_NNODES nodes..."

RAY_NODE_PORT=8888
mkdir -p "$HOME/.triton/autotune"
ray stop --force 2>/dev/null || true

echo "Starting Ray head on $(hostname) (IP: $HEAD_IP)..."
ray start --head --port=$RAY_NODE_PORT --node-ip-address="$HEAD_IP" \
    --temp-dir="$RAY_TMPDIR" --num-gpus=4 --num-cpus=32 --dashboard-host=0.0.0.0

# Set RAY_ADDRESS so ray.init() in grpo_fast.py auto-connects to this cluster.
export RAY_ADDRESS="$HEAD_IP:$RAY_NODE_PORT"

# Start workers on non-head nodes (if multi-node)
SRUN_PID=""
if [ "$SLURM_NNODES" -gt 1 ]; then
    HEAD_NODE=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
    echo "Starting Ray workers on $((SLURM_NNODES - 1)) non-head nodes..."
    srun --export=ALL --nodes=$((SLURM_NNODES - 1)) --exclude="$HEAD_NODE" \
         bash configs/isambard/ray_node_setup_slurm.sh &
    SRUN_PID=$!
    # Wait for all nodes to join (poll ray status)
    EXPECTED_GPUS=$((SLURM_NNODES * 4))
    echo "Waiting for $SLURM_NNODES nodes ($EXPECTED_GPUS GPUs) to join Ray cluster..."
    for i in $(seq 1 180); do
        CURRENT_GPUS=$(python -c "
import ray, sys
try:
    ray.init(address='$HEAD_IP:$RAY_NODE_PORT', ignore_reinit_error=True)
    gpus = ray.cluster_resources().get('GPU', 0)
    print(int(gpus))
except:
    print(0)
" 2>/dev/null)
        if [ "$CURRENT_GPUS" -ge "$EXPECTED_GPUS" ]; then
            echo "All $SLURM_NNODES nodes registered ($CURRENT_GPUS GPUs)."
            break
        fi
        if [ $((i % 30)) -eq 0 ]; then
            echo "  ... $CURRENT_GPUS/$EXPECTED_GPUS GPUs after ${i}s"
        fi
        sleep 1
    done
    if [ "$CURRENT_GPUS" -lt "$EXPECTED_GPUS" ]; then
        echo "ERROR: Only $CURRENT_GPUS/$EXPECTED_GPUS GPUs after 180s. Aborting."
        ray status || true
        exit 1
    fi
fi

echo "===== Ray Cluster Status ====="
ray status || echo "Warning: ray status failed (cluster may still be initializing)"
echo "==============================="

# --- Source the training config (sets TRAINING_ARGS) ---
source "$TRAINING_CONFIG"

# --- Run training on head node ---
echo "Launching grpo_fast.py..."
LD_PRELOAD="$NCCL_LIBRARY" python open_instruct/grpo_fast.py $TRAINING_ARGS 2>&1
TRAIN_EXIT=$?

echo "Training exited with code: $TRAIN_EXIT"

# --- Cleanup Ray ---
echo "Cleaning up Ray cluster..."
if [ -n "$SRUN_PID" ]; then
    kill $SRUN_PID 2>/dev/null; wait $SRUN_PID 2>/dev/null || true
fi
ray stop --force 2>/dev/null || true
rm -rf "$RAY_TMPDIR" 2>/dev/null || true

# --- Job chaining ---
if [ "$CURRENT_CHAIN" -lt "$MAX_JOB_CHAINS" ]; then
    NEXT_CHAIN=$((CURRENT_CHAIN + 1))
    echo "Submitting chained job $NEXT_CHAIN / $MAX_JOB_CHAINS"
    sbatch --dependency=afterany:$SLURM_JOB_ID \
           --nodes=$SLURM_NNODES \
           --export=ALL,SLURM_JOB_CHAIN_COUNT=$NEXT_CHAIN \
           configs/isambard/grpo_rlzero.sbatch "$TRAINING_CONFIG"
fi

echo "===== Job Completed at $(date) ====="
exit $TRAIN_EXIT
