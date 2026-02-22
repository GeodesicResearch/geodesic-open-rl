#!/bin/bash
# Main SLURM batch script for GRPO (RL-Zero) training on Isambard.
# Launches a Ray cluster across SLURM nodes, then runs grpo_fast.py on head.
#
# Usage:
#   sbatch --nodes=2 configs/isambard/grpo_rlzero.sbatch configs/isambard/grpo_olmo3_7b_general.yaml
#   sbatch --nodes=10 configs/isambard/grpo_rlzero.sbatch configs/isambard/grpo_olmo3_7b_code.yaml
#   sbatch --nodes=1 configs/isambard/grpo_rlzero.sbatch configs/isambard/grpo_debug_single_node.yaml
#
# Accepts both YAML configs (.yaml, preferred) and shell configs (.sh, must set TRAINING_ARGS).

#SBATCH --job-name=grpo-rlzero
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=/projects/a5k/public/logs_%u/open-instruct/grpo-rlzero-%j.out
#SBATCH --exclude=nid010798,nid010869,nid010867,nid010874  # Bad nodes: ECC errors (010798,010869), slow training (010867,010874)

set -euo pipefail

TRAINING_CONFIG=${1:?Usage: sbatch grpo_rlzero.sbatch <config.sh|config.yaml> [extra CLI args...]}
shift
EXTRA_ARGS="$*"

# Resolve config to absolute path. $1 may be relative to the submission directory,
# but SLURM runs the script from a spool directory with a different CWD.
# SLURM_SUBMIT_DIR is the directory where sbatch was invoked.
if [[ "$TRAINING_CONFIG" != /* ]]; then
    TRAINING_CONFIG="${SLURM_SUBMIT_DIR:-$PWD}/$TRAINING_CONFIG"
fi

# Derive repo root. Prefer SLURM_SUBMIT_DIR (we always sbatch from the repo root),
# fall back to walking up from the config path to find .git.
# REPO_DIR can also be set explicitly (e.g. by submit_pipeline.py).
if [ -z "${REPO_DIR:-}" ]; then
    if [ -d "${SLURM_SUBMIT_DIR:-}/.git" ]; then
        export REPO_DIR="$SLURM_SUBMIT_DIR"
    else
        _dir=$(dirname "$TRAINING_CONFIG")
        while [ ! -d "$_dir/.git" ] && [ "$_dir" != "/" ]; do _dir=$(dirname "$_dir"); done
        export REPO_DIR="$_dir"
    fi
fi
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
module load brics/aws-ofi-nccl/1.8.1

# --- Custom NCCL from venv ---
# System/module NCCL is too old for PyTorch 2.9.1 (missing ncclCommWindowRegister).
# brics/nccl module provides 2.27.5 but is ~2x slower for weight sync (48-51s vs 21-24s),
# likely due to NCCL being loaded into all processes including vLLM EngineCore.
# We use the venv's bundled NCCL 2.27.5 via LD_PRELOAD for the training process only.
VENV_SP="$REPO_DIR/.venv/lib/python3.12/site-packages"
export NCCL_LIBRARY="$VENV_SP/nvidia/nccl/lib/libnccl.so.2"
# NCCL_LIBRARY is preloaded per-command (training process only, not Ray workers).

# --- Compilers and CUDA arch ---
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
export TORCH_CUDA_ARCH_LIST="9.0"
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6

# --- NCCL / OFI settings for Slingshot (per docs.isambard.ac.uk/guides/nccl/) ---
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
# PyTorch NCCL settings
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1

# --- Networking ---
HEAD_HOSTNAME=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
# Resolve to a stable IP. getent may return multiple; take the first.
export HEAD_IP=$(getent hosts "$HEAD_HOSTNAME" | awk '{print $1; exit}')
export MASTER_ADDR="$HEAD_IP"
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

# --- HuggingFace ---
export HF_HUB_OFFLINE=1
export HF_HUB_CACHE="/projects/a5k/public/hf_${USER}/hub"

# --- Weights & Biases ---
# Export API key so wandb's public.Api() (used for artifact logging) can
# authenticate without a separate network call to verify the token.
export WANDB_API_KEY=$(awk '/machine api.wandb.ai/{getline; getline; print $2}' ~/.netrc 2>/dev/null)

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

# --- Load training config (early, for env vars and feature detection) ---
# Shell configs: source to get TRAINING_ARGS and env vars (e.g. START_CODE_SERVER, MAX_JOB_CHAINS).
# YAML configs: auto-detect features by grepping for known keys.
# Always reset feature flags first so we never inherit stale env vars.
export START_CODE_SERVER=0
if [[ "$TRAINING_CONFIG" == *.sh ]]; then
    source "$TRAINING_CONFIG"
elif [[ "$TRAINING_CONFIG" == *.yaml ]]; then
    if grep -qE '^[^#]*code_pass_rate_reward_threshold' "$TRAINING_CONFIG"; then
        START_CODE_SERVER=1
    fi
    if grep -qE '^[^#]*rm_enabled.*[Tt]rue' "$TRAINING_CONFIG"; then
        echo "Reward model enabled (detected in YAML config)"
    fi
fi

# --- Start code execution server (for code reward verification) ---
# Auto-enabled for YAML configs containing code_pass_rate_reward_threshold,
# or manually via START_CODE_SERVER=1 in shell configs.
CODE_SERVER_PID=""
if [ "$START_CODE_SERVER" = "1" ]; then
    echo "Starting code execution server on head node..."
    uvicorn open_instruct.code_utils.api:app \
        --host 0.0.0.0 --port 1234 --workers 16 \
        > "$TMPDIR/code_server_${SLURM_JOB_ID}.log" 2>&1 &
    CODE_SERVER_PID=$!
    echo "Waiting for code execution server to start..."
    for i in $(seq 1 30); do
        if curl -s http://localhost:1234/health > /dev/null 2>&1; then
            echo "Code server running on port 1234 (PID: $CODE_SERVER_PID)"
            break
        fi
        if [ $i -eq 30 ]; then
            echo "ERROR: Code server failed to start after 30s. Check $TMPDIR/code_server_${SLURM_JOB_ID}.log"
        fi
        sleep 1
    done
fi

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

# --- Verify NCCL ---
LD_PRELOAD="$NCCL_LIBRARY" python -c "
import ctypes, os
nccl_path = os.environ.get('NCCL_LIBRARY', '')
if nccl_path:
    try:
        nccl = ctypes.CDLL(nccl_path)
        print(f'NCCL library loaded: {nccl_path}')
    except OSError as e:
        print(f'WARNING: Could not load {nccl_path}: {e}')
try:
    import torch
    print(f'PyTorch NCCL version: {torch.cuda.nccl.version()}')
except Exception as e:
    print(f'Could not get PyTorch NCCL version: {e}')
" 2>&1 || echo "WARNING: Could not verify NCCL"

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

# --- Pipeline: resolve model from previous stage ---
# When PIPELINE_PREV_OUTPUT_DIR is set (by submit_pipeline.py), find the latest
# HF model directory saved by the previous stage and pass it as a CLI override.
# This is a no-op if the env var is unset (normal non-pipeline usage).
PIPELINE_EXTRA_ARGS=""
if [ -n "${PIPELINE_PREV_OUTPUT_DIR:-}" ]; then
    echo "Pipeline: resolving model from previous stage output: $PIPELINE_PREV_OUTPUT_DIR"
    # Use `if` to avoid set -e exiting on Python failure
    if PREV_MODEL=$(python -c "
import os, sys
base = sys.argv[1]
if not os.path.isdir(base):
    print(f'ERROR: {base} does not exist', file=sys.stderr)
    sys.exit(1)
candidates = []
for d in os.listdir(base):
    full = os.path.join(base, d)
    if os.path.isdir(full) and not d.endswith('_checkpoints') and os.path.exists(os.path.join(full, 'config.json')):
        candidates.append((os.path.getmtime(full), full))
if not candidates:
    print(f'ERROR: no valid model found in {base}', file=sys.stderr)
    sys.exit(1)
candidates.sort(reverse=True)
print(candidates[0][1])
" "$PIPELINE_PREV_OUTPUT_DIR") && [ -n "$PREV_MODEL" ]; then
        echo "Pipeline: using model from $PREV_MODEL"
        PIPELINE_EXTRA_ARGS="--model_name_or_path=$PREV_MODEL"
    else
        echo "ERROR: Failed to resolve previous stage model from $PIPELINE_PREV_OUTPUT_DIR"
        exit 1
    fi
fi

# --- Run training on head node ---
echo "Launching grpo_fast.py..."
if [[ "$TRAINING_CONFIG" == *.yaml ]]; then
    LD_PRELOAD="$NCCL_LIBRARY" python open_instruct/grpo_fast.py "$TRAINING_CONFIG" $PIPELINE_EXTRA_ARGS $EXTRA_ARGS 2>&1
else
    # Shell config was already sourced above; TRAINING_ARGS is set.
    LD_PRELOAD="$NCCL_LIBRARY" python open_instruct/grpo_fast.py $TRAINING_ARGS $EXTRA_ARGS 2>&1
fi
TRAIN_EXIT=$?

echo "Training exited with code: $TRAIN_EXIT"

# --- Cleanup ---
echo "Cleaning up..."
if [ -n "$CODE_SERVER_PID" ]; then
    echo "Stopping code server (PID: $CODE_SERVER_PID)..."
    kill $CODE_SERVER_PID 2>/dev/null; wait $CODE_SERVER_PID 2>/dev/null || true
fi
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
