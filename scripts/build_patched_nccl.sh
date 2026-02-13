#!/bin/bash
# Build patched NCCL v2.27.5 for GH200 (Isambard).
#
# Problem: PyTorch 2.9.1's bundled NCCL 2.27.5 uses cudaDeviceGetPCIBusId() which
# returns identical bus IDs for all GH200 GPUs. This causes "Duplicate GPU detected"
# errors when multiple NCCL ranks run on the same node.
#
# Fix: Build NCCL from source with the duplicate GPU check gated behind
# NCCL_IGNORE_DUPLICATE_GPU=1. When set, the check is skipped and a log message
# is printed instead of aborting.
#
# Usage:
#   bash scripts/build_patched_nccl.sh          # build and install to venv
#   bash scripts/build_patched_nccl.sh --build-only  # build without installing
#
# Prerequisites:
#   module load cuda/12.6
#   module load PrgEnv-cray
#
# The patched library replaces the bundled NCCL in the venv. The original is
# backed up as libnccl.so.2.orig.

set -euo pipefail

NCCL_VERSION="v2.27.5-1"
BUILD_DIR="/tmp/nccl-2.27.5-patched"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
VENV_NCCL_DIR="$REPO_DIR/.venv/lib/python3.12/site-packages/nvidia/nccl/lib"
INSTALL=${1:-"--install"}  # default: install to venv

echo "=== Building patched NCCL $NCCL_VERSION ==="
echo "Build dir:  $BUILD_DIR"
echo "Repo dir:   $REPO_DIR"
echo "Venv NCCL:  $VENV_NCCL_DIR"

# --- Clone ---
if [ -d "$BUILD_DIR" ]; then
    echo "Removing existing build dir..."
    rm -rf "$BUILD_DIR"
fi
echo "Cloning NCCL $NCCL_VERSION..."
git clone -b "$NCCL_VERSION" --depth 1 https://github.com/NVIDIA/nccl.git "$BUILD_DIR"

# --- Patch ---
echo "Applying duplicate GPU check patch..."
INIT_CC="$BUILD_DIR/src/init.cc"

# Verify the target code exists before patching
if ! grep -q "Duplicate GPU detected" "$INIT_CC"; then
    echo "ERROR: Could not find 'Duplicate GPU detected' in $INIT_CC"
    echo "The NCCL source may have changed. Manual patching required."
    exit 1
fi

# Use Python for reliable multi-line patching.
# Replaces the WARN+fail block inside the duplicate GPU check with an env-var-gated version.
python3 -c "
import re, sys

with open('$INIT_CC', 'r') as f:
    src = f.read()

# Match the duplicate GPU check block:
#   if ((i != rank) && ... busId == ... busId)) {
#     WARN(\"Duplicate GPU detected ...\");
#     ret = ncclInvalidUsage;
#     goto fail;
#   }
pattern = re.compile(
    r'([ \t]*if \(\(i != rank\) && \(comm->peerInfo\[i\]\.hostHash == comm->peerInfo\[rank\]\.hostHash\) &&\s*'
    r'\(comm->peerInfo\[i\]\.busId == comm->peerInfo\[rank\]\.busId\)\) \{)\s*'
    r'(WARN\(\"Duplicate GPU detected[^;]*;\s*'
    r'ret = ncclInvalidUsage;\s*'
    r'goto fail;)\s*'
    r'(\})',
    re.DOTALL
)

match = pattern.search(src)
if not match:
    print('ERROR: Could not match duplicate GPU check pattern in init.cc', file=sys.stderr)
    sys.exit(1)

# Get indentation from the WARN line
indent = '      '  # 6 spaces (inside the if block)

replacement = match.group(1) + '''
      const char* ignoreDup = ncclGetEnv(\"NCCL_IGNORE_DUPLICATE_GPU\");
      if (!ignoreDup || strcmp(ignoreDup, \"1\") != 0) {
        WARN(\"Duplicate GPU detected : rank %d and rank %d both on CUDA device %lx\",
             rank, i, comm->peerInfo[rank].busId);
        ret = ncclInvalidUsage;
        goto fail;
      } else {
        INFO(NCCL_INIT, \"Duplicate GPU detected but ignored (NCCL_IGNORE_DUPLICATE_GPU=1): rank %d and rank %d both on CUDA device %lx\",
             rank, i, comm->peerInfo[rank].busId);
      }
    ''' + match.group(3)

patched = pattern.sub(replacement, src, count=1)

if patched == src:
    print('ERROR: Patch did not modify the source', file=sys.stderr)
    sys.exit(1)

with open('$INIT_CC', 'w') as f:
    f.write(patched)

print('Patch applied successfully.')
"

# Verify patch was applied
if grep -q "NCCL_IGNORE_DUPLICATE_GPU" "$INIT_CC"; then
    echo "Patch verified: NCCL_IGNORE_DUPLICATE_GPU gate found in init.cc"
else
    echo "ERROR: Patch verification failed"
    exit 1
fi

# --- Build ---
echo "Building NCCL (this takes ~5 minutes on GH200)..."
export CC=${CC:-/usr/bin/gcc-12}
export CXX=${CXX:-/usr/bin/g++-12}
CUDA_HOME=${CUDA_HOME:-/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6}

cd "$BUILD_DIR"
make -j32 src.build CUDA_HOME="$CUDA_HOME"

BUILT_LIB="$BUILD_DIR/build/lib/libnccl.so.2"
if [ ! -f "$BUILT_LIB" ]; then
    echo "ERROR: Build failed — $BUILT_LIB not found"
    exit 1
fi
echo "Build successful: $BUILT_LIB"

# --- Install ---
if [ "$INSTALL" = "--build-only" ]; then
    echo "Build-only mode. Library at: $BUILT_LIB"
    exit 0
fi

if [ ! -d "$VENV_NCCL_DIR" ]; then
    echo "ERROR: Venv NCCL dir not found: $VENV_NCCL_DIR"
    echo "Make sure the venv is set up first."
    exit 1
fi

# Backup original
ORIG="$VENV_NCCL_DIR/libnccl.so.2.orig"
if [ ! -f "$ORIG" ]; then
    echo "Backing up original NCCL library..."
    cp "$VENV_NCCL_DIR/libnccl.so.2" "$ORIG"
    echo "Backup saved to: $ORIG"
else
    echo "Backup already exists: $ORIG"
fi

echo "Installing patched NCCL to venv..."
cp "$BUILT_LIB" "$VENV_NCCL_DIR/libnccl.so.2"

echo ""
echo "=== Done ==="
echo "Patched NCCL installed to: $VENV_NCCL_DIR/libnccl.so.2"
echo "Original backed up to:     $ORIG"
echo ""
echo "To use: export NCCL_IGNORE_DUPLICATE_GPU=1 (set in grpo_rlzero.sbatch)"
echo "To revert: cp $ORIG $VENV_NCCL_DIR/libnccl.so.2"
