#!/bin/bash
# Build flash-attn from source for the existing open-instruct venv.
#
# flash-attn is excluded on aarch64 by pyproject.toml, so it must be
# built manually. This takes ~30-60 minutes (many CUDA kernels).
#
# Usage:
#   sbatch --time=02:00:00 configs/isambard/run_on_compute.sbatch \
#       bash configs/isambard/build_flash_attn.sh

set -euo pipefail

# Derive repo root from script location (configs/isambard/ → ../..)
REPO_DIR="$(cd "$(dirname "$0")"/../.. && pwd)"
VENV_PYTHON="$REPO_DIR/.venv/bin/python"
VENV_SP="$REPO_DIR/.venv/lib/python3.12/site-packages"

echo "=== Building flash-attn from source ==="
echo "PyTorch version: $($VENV_PYTHON -c 'import torch; print(torch.__version__)')"
echo "CUDA_HOME: $CUDA_HOME"
echo ""

# NVIDIA library paths needed for compilation
export NCCL_LIBRARY="$VENV_SP/nvidia/nccl/lib/libnccl.so.2"
export LD_LIBRARY_PATH="$VENV_SP/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$VENV_SP/nvidia/cublas/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$VENV_SP/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH"
export CPLUS_INCLUDE_PATH="$VENV_SP/nvidia/cudnn/include:${CPLUS_INCLUDE_PATH:-}"
export C_INCLUDE_PATH="$VENV_SP/nvidia/cudnn/include:${C_INCLUDE_PATH:-}"
export LIBRARY_PATH="$VENV_SP/nvidia/cudnn/lib:${LIBRARY_PATH:-}"

CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 MAX_JOBS=4 \
    CUDA_HOME="$CUDA_HOME" \
    LD_PRELOAD="$NCCL_LIBRARY" \
    uv pip install --python "$VENV_PYTHON" \
        --no-build-isolation --no-cache-dir --no-binary flash-attn \
        "flash-attn>=2.8.3"

echo ""
echo "=== Validating flash-attn ==="
LD_PRELOAD="$NCCL_LIBRARY" "$VENV_PYTHON" -c "
import flash_attn
print(f'flash_attn version: {flash_attn.__version__}')
from flash_attn import flash_attn_func
print('flash_attn_func: OK')
"
echo "flash-attn build: PASSED"
