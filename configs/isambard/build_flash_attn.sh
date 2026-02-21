#!/bin/bash
# Build flash-attn from source for GH200 (aarch64).
#
# Prebuilt wheels don't exist for aarch64 + sm_90, so we compile from source.
# This takes 30-60 minutes on a compute node.
#
# Usage (on a compute node):
#   srun --nodes=1 --gpus-per-node=1 --time=01:30:00 bash configs/isambard/build_flash_attn.sh
#
# Or via sbatch:
#   sbatch --nodes=1 --gpus-per-node=1 --time=01:30:00 --wrap="bash /path/to/build_flash_attn.sh"

set -euo pipefail

FLASH_ATTN_VERSION="2.8.3"

# --- Resolve repo root ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "===== Building flash-attn $FLASH_ATTN_VERSION from source ====="
echo "Repo:    $REPO_DIR"
echo "Date:    $(date)"
echo "Host:    $(hostname)"
echo "=========================================="

# --- Load modules ---
module purge
module load PrgEnv-cray
module load cuda/12.6

# --- Compilers and CUDA arch ---
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
export TORCH_CUDA_ARCH_LIST="9.0"
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6

# Limit parallel jobs to avoid OOM during CUDA kernel compilation
export MAX_JOBS=4

# --- Activate venv ---
source "$REPO_DIR/.venv/bin/activate"

echo ""
echo "Python:  $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo ""

# --- Build flash-attn from source ---
echo "Building flash-attn $FLASH_ATTN_VERSION from source..."
echo "This takes 30-60 minutes due to many CUDA kernels."
echo ""

uv pip install \
    --no-build-isolation \
    --no-cache-dir \
    --no-binary flash-attn \
    "flash-attn==$FLASH_ATTN_VERSION" || {
    echo "ERROR: flash-attn build failed"
    exit 1
}

echo ""
echo "Build complete. Validating installation..."
echo ""

# --- Validate ---
python -c "
import flash_attn
print(f'  flash_attn version: {flash_attn.__version__}')

from flash_attn import flash_attn_func, flash_attn_varlen_func
print('  flash_attn_func:        OK')
print('  flash_attn_varlen_func: OK')

# Verify the C extension loaded (this is the compiled CUDA module)
import flash_attn_2_cuda
print('  flash_attn_2_cuda:      OK')
" || {
    echo "ERROR: flash-attn validation failed!"
    exit 1
}

echo ""
echo "flash-attn $FLASH_ATTN_VERSION: build and validation PASSED"
