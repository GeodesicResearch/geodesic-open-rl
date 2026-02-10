#!/bin/bash
# Generic SLURM script to run any command on a compute node with GPU.
#
# Usage:
#   sbatch run_on_compute.sbatch <command> [args...]
#   sbatch --time=24:00:00 --gpus=4 run_on_compute.sbatch <command>
#
# Examples:
#   sbatch run_on_compute.sbatch bash configs/isambard/setup_open_instruct_env.sh
#   sbatch run_on_compute.sbatch uv run pytest tests/ -v
#   sbatch run_on_compute.sbatch nvidia-smi
#
# Logs: /projects/a5k/public/logs_<USER>/open-instruct/run_on_compute_<JOB_ID>.out

#SBATCH --job-name=run-compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=/projects/a5k/public/logs_%u/open-instruct/run_on_compute_%j.out

set -euo pipefail

# Hardcoded to avoid BASH_SOURCE spool directory bug in SLURM
REPO_DIR="/home/a5k/${USER}/open-instruct"
cd "$REPO_DIR"

echo "======================================"
echo "  Compute Node Job"
echo "======================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "Date:      $(date)"
echo "Directory: $REPO_DIR"
echo "Command:   $*"
echo ""

# GPU check
echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || {
    echo "ERROR: No GPU detected."; exit 1
}
echo ""

# Load modules
module load cuda/12.6 || true
module load cudatoolkit || true

# Compilers and environment
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
export TORCH_CUDA_ARCH_LIST="9.0"
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
export TMPDIR=/projects/a5k/public/tmp_${USER}
mkdir -p "$TMPDIR"

# Activate uv venv if it exists and is complete
if [ -f "$REPO_DIR/.venv/bin/activate" ]; then
    source "$REPO_DIR/.venv/bin/activate"
    VENV_SP="$REPO_DIR/.venv/lib/python3.12/site-packages"
    export NCCL_LIBRARY="$VENV_SP/nvidia/nccl/lib/libnccl.so.2"
    export LD_PRELOAD="$NCCL_LIBRARY"
    echo "Activated .venv (LD_PRELOAD=$NCCL_LIBRARY)"
else
    echo "No .venv found — skipping venv activation"
fi
echo ""

# Run the command
echo "=== Running: $* ==="
"$@"
exit_code=$?

echo ""
echo "======================================"
echo "  Finished at $(date)"
echo "  Exit code: $exit_code"
echo "======================================"
exit $exit_code
