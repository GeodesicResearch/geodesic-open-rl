#!/bin/bash
# setup_open_instruct_env_noflash.sh - Create uv environment for open-instruct on Isambard ARM HPC
# Same as setup_open_instruct_env.sh but SKIPS flash-attn (saves 30-60 min).
#
# Usage:
#   isambard_sbatch configs/isambard/run_on_compute.sbatch bash configs/isambard/setup_open_instruct_env_noflash.sh
#
# Prerequisites:
#   - uv installed (curl -LsSf https://astral.sh/uv/install.sh | sh)
#   - Run on a compute node with GPU access

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"

echo "=============================================="
echo "  open-instruct UV Environment Setup"
echo "  for Isambard GH200 (NO flash-attn)"
echo "=============================================="
echo "Architecture: $(uname -m)"
echo "Working directory: $REPO_DIR"
echo ""

# ============================================
# Step 1: Load required modules
# ============================================
echo "=== Step 1: Loading modules ==="
# Note: Do NOT load brics/nccl - torch comes with its own NCCL (nvidia-nccl-cu12)
# and loading the system NCCL causes symbol conflicts
module load cuda/12.6 || echo "Warning: cuda/12.6 module not found"
module load cudatoolkit || echo "Warning: cudatoolkit module not found"

# ============================================
# Step 2: Set compiler and environment
# ============================================
echo ""
echo "=== Step 2: Setting compiler and environment ==="
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
export MAX_JOBS=4
export TORCH_CUDA_ARCH_LIST="9.0"
export TMPDIR=/projects/a5k/public/tmp_${USER}
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6

echo "CC=$CC"
echo "CXX=$CXX"
echo "MAX_JOBS=$MAX_JOBS"
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
echo "TMPDIR=$TMPDIR"
echo "CUDA_HOME=$CUDA_HOME"

# Check CUDA version from nvidia-smi vs what we're linking against
echo ""
echo "=== CUDA version check ==="
DRIVER_CUDA=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
echo "nvidia-smi driver version: $DRIVER_CUDA"
echo "CUDA_HOME points to: $CUDA_HOME"
echo "pyproject.toml configures cu130 index for aarch64"
echo "(If there's a major mismatch, torch may fail to use GPUs)"

# ============================================
# Step 3: Check uv is available
# ============================================
echo ""
echo "=== Step 3: Checking uv ==="
if ! command -v uv &> /dev/null; then
    echo "ERROR: uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
echo "uv version: $(uv --version)"

# ============================================
# Step 4: Create virtual environment
# ============================================
echo ""
echo "=== Step 4: Creating virtual environment ==="
if [ -d ".venv" ]; then
    echo "Removing existing .venv directory..."
    # Unset LD_PRELOAD — run_on_compute.sbatch may have set it to the NCCL
    # lib inside .venv. The .so may still be mmapped by this process though,
    # so rm -rf can fail on that one file. Rename-then-delete avoids the issue.
    unset LD_PRELOAD
    mv .venv .venv_old_$$ 2>/dev/null || true
    rm -rf .venv_old_$$ &  # background cleanup
    rm -rf .venv 2>/dev/null || true  # in case mv failed
fi
uv venv --python 3.12 .venv
echo "Virtual environment created at: $REPO_DIR/.venv"

# Get the venv site-packages path for setting library paths later
VENV_SITE_PACKAGES="$REPO_DIR/.venv/lib/python3.12/site-packages"

# ============================================
# Step 5: Install all dependencies
# ============================================
echo ""
echo "=== Step 5: Installing all dependencies ==="
echo "This may take a while for packages that need compilation..."
CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 MAX_JOBS=4 \
    uv sync

# ============================================
# Step 5b: Replace torch cu130 with cu126 (driver compat)
# ============================================
echo ""
echo "=== Step 5b: Replacing torch cu130 with cu126 (Isambard driver compatibility) ==="
# pyproject.toml pulls torch from cu130 index for aarch64, but Isambard's
# driver (12070 = CUDA 12.7) is too old for cu130. Replace with cu126 build.
VENV_PYTHON="$REPO_DIR/.venv/bin/python"
uv pip install --python "$VENV_PYTHON" \
    "torch>=2.9.0,<2.10" \
    --index-url https://download.pytorch.org/whl/cu126

# ============================================
# Step 6: Set NVIDIA library paths (NCCL, cuDNN, cuBLAS)
# ============================================
echo ""
echo "=== Step 6: Setting NVIDIA library paths ==="

# CRITICAL: Use LD_PRELOAD to force loading venv's NCCL instead of system NCCL
export NCCL_LIBRARY="$VENV_SITE_PACKAGES/nvidia/nccl/lib/libnccl.so.2"
export LD_PRELOAD="$NCCL_LIBRARY"

# Also set LD_LIBRARY_PATH for other NVIDIA libraries
export LD_LIBRARY_PATH="$VENV_SITE_PACKAGES/nvidia/nccl/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$VENV_SITE_PACKAGES/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$VENV_SITE_PACKAGES/nvidia/cublas/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$VENV_SITE_PACKAGES/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$VENV_SITE_PACKAGES/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$VENV_SITE_PACKAGES/nvidia/cusparse/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$VENV_SITE_PACKAGES/nvidia/cufft/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$VENV_SITE_PACKAGES/nvidia/curand/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$VENV_SITE_PACKAGES/nvidia/cusolver/lib:$LD_LIBRARY_PATH"

# Include paths for compilation
export CPLUS_INCLUDE_PATH="$VENV_SITE_PACKAGES/nvidia/cudnn/include:$CPLUS_INCLUDE_PATH"
export C_INCLUDE_PATH="$VENV_SITE_PACKAGES/nvidia/cudnn/include:$C_INCLUDE_PATH"
export LIBRARY_PATH="$VENV_SITE_PACKAGES/nvidia/cudnn/lib:$LIBRARY_PATH"
export CUDNN_PATH="$VENV_SITE_PACKAGES/nvidia/cudnn"

export CPLUS_INCLUDE_PATH="$VENV_SITE_PACKAGES/nvidia/cublas/include:$CPLUS_INCLUDE_PATH"
export C_INCLUDE_PATH="$VENV_SITE_PACKAGES/nvidia/cublas/include:$C_INCLUDE_PATH"
export LIBRARY_PATH="$VENV_SITE_PACKAGES/nvidia/cublas/lib:$LIBRARY_PATH"

echo "NCCL library (LD_PRELOAD): $NCCL_LIBRARY"
echo "cuDNN path: $CUDNN_PATH"

# Create cuDNN header symlinks in PyTorch include directory
TORCH_INCLUDE="$VENV_SITE_PACKAGES/torch/include"
CUDNN_INCLUDE="$VENV_SITE_PACKAGES/nvidia/cudnn/include"
if [ -d "$CUDNN_INCLUDE" ] && [ -d "$TORCH_INCLUDE" ]; then
    echo "Creating cuDNN header symlinks in PyTorch include directory..."
    for f in "$CUDNN_INCLUDE"/*.h; do
        ln -sf "$f" "$TORCH_INCLUDE/$(basename $f)" 2>/dev/null || true
    done
    echo "cuDNN symlinks created"
else
    echo "Warning: Could not create cuDNN symlinks (directories not found)"
fi

# Verify PyTorch CUDA
# IMPORTANT: Use $VENV_PYTHON directly, NOT "uv run python".
# "uv run" re-syncs against pyproject.toml which reinstalls torch cu130.
echo ""
echo "Verifying PyTorch installation..."
LD_PRELOAD="$NCCL_LIBRARY" "$VENV_PYTHON" -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# ============================================
# Step 7: SKIPPED (flash-attn) — not required, saves 30-60 min
# ============================================
echo ""
echo "=== Step 7: SKIPPED (flash-attn not installed — using sdpa/eager fallback) ==="

# ============================================
# Step 8: Validate vLLM import
# ============================================
echo ""
echo "=== Step 8: Validating vLLM ==="
LD_PRELOAD="$NCCL_LIBRARY" LD_LIBRARY_PATH="$LD_LIBRARY_PATH" "$VENV_PYTHON" -c "
import vllm
print(f'  vLLM version: {vllm.__version__}')
print('  vLLM import: OK')
" || {
    echo "WARNING: vLLM import failed!"
    echo "vLLM may need to be built from source for aarch64."
    echo "Attempting source build..."
    CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 MAX_JOBS=4 \
        CUDA_HOME="$CUDA_HOME" \
        LD_PRELOAD="$NCCL_LIBRARY" \
        uv pip install --python "$VENV_PYTHON" --no-build-isolation --no-cache-dir --no-binary vllm "vllm==0.14.1" || {
            echo "ERROR: vLLM source build also failed."
            echo "This is a HIGH RISK item - manual intervention may be required."
            exit 1
        }
    # Re-validate
    LD_PRELOAD="$NCCL_LIBRARY" LD_LIBRARY_PATH="$LD_LIBRARY_PATH" "$VENV_PYTHON" -c "
import vllm
print(f'  vLLM version: {vllm.__version__}')
print('  vLLM import (after rebuild): OK')
" || {
        echo "ERROR: vLLM still fails to import after source build."
        exit 1
    }
}
echo "vLLM validation: PASSED"

# ============================================
# Step 9: GH200 sm_90a fix
# ============================================
echo ""
echo "=== Step 9: GH200 sm_90a fix ==="
# TORCH_CUDA_ARCH_LIST=9.0 is set in the sbatch script and in Step 2 above.
# DO NOT install a sitecustomize.py that imports torch — it runs at Python
# startup (before Ray sets CUDA_VISIBLE_DEVICES) and poisons CUDA device
# enumeration, causing NCCL "Duplicate GPU detected" errors on GH200.
echo "TORCH_CUDA_ARCH_LIST=9.0 is set via environment (no sitecustomize.py needed)"

# ============================================
# Step 10: Apply wandb patch (fix isatty issue)
# ============================================
echo ""
echo "=== Step 10: Applying wandb patch ==="
WANDB_TERM_FILE="$VENV_SITE_PACKAGES/wandb/errors/term.py"
if [ -f "$WANDB_TERM_FILE" ]; then
    sed -i 's/    return sys\.stderr\.isatty()/    return hasattr(sys.stderr, "isatty") and sys.stderr.isatty()/' "$WANDB_TERM_FILE" || true
    echo "wandb patch applied"
else
    echo "Warning: wandb term.py not found, skipping patch"
fi

# ============================================
# Step 11: Verify installation
# ============================================
echo ""
echo "=== Step 11: Verifying installation ==="
LD_PRELOAD="$NCCL_LIBRARY" "$VENV_PYTHON" -c "
import sys
print(f'Python: {sys.version}')

import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'cuDNN version: {torch.backends.cudnn.version()}')

try:
    import ray
    print(f'ray: {ray.__version__}')
except ImportError:
    print('ray: NOT INSTALLED')

try:
    import vllm
    print(f'vllm: {vllm.__version__}')
except ImportError:
    print('vllm: NOT INSTALLED')

try:
    import deepspeed
    print(f'deepspeed: {deepspeed.__version__}')
except ImportError:
    print('deepspeed: NOT INSTALLED')

try:
    import flash_attn
    print(f'flash_attn: {flash_attn.__version__}')
except ImportError:
    print('flash_attn: NOT INSTALLED (expected — using sdpa/eager fallback)')

try:
    import wandb
    print(f'wandb: {wandb.__version__}')
except ImportError:
    print('wandb: NOT INSTALLED')

try:
    import transformers
    print(f'transformers: {transformers.__version__}')
except ImportError:
    print('transformers: NOT INSTALLED')

try:
    import datasets
    print(f'datasets: {datasets.__version__}')
except ImportError:
    print('datasets: NOT INSTALLED')
"

echo ""
echo "=============================================="
echo "  Setup complete! (no flash-attn)"
echo "=============================================="
echo ""
echo "To activate the environment:"
echo "  source .venv/bin/activate"
echo ""
echo "IMPORTANT: You must set LD_PRELOAD to use the correct NCCL library:"
echo "  export NCCL_LIBRARY=$VENV_SITE_PACKAGES/nvidia/nccl/lib/libnccl.so.2"
echo "  export LD_PRELOAD=\$NCCL_LIBRARY"
echo ""
echo "First real test:"
echo "  isambard_sbatch --nodes=1 configs/isambard/grpo_rlzero.sbatch configs/isambard/grpo_debug_single_node.yaml"
echo ""
