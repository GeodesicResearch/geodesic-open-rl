#!/bin/bash
# Build patched NCCL v2.27.5 for GH200 (Isambard).
#
# Problem: PyTorch 2.9.1's bundled NCCL 2.27.5 uses cudaDeviceGetPCIBusId() which
# returns identical bus IDs for all GH200 GPUs (all report 0000:90:10.0). This causes
# "Duplicate GPU detected" errors and topology failures when multiple NCCL ranks run
# on the same node.
#
# Fix: Patch getBusId() in src/misc/utils.cc to read correct bus IDs from
# /proc/driver/nvidia/gpus/ (kernel driver) instead of the buggy CUDA runtime.
# Enabled by NCCL_BUSID_PROC_FIX=1. This fixes the root cause: correct bus IDs
# mean the duplicate check passes naturally and topology builds correctly.
#
# Usage:
#   bash scripts/build_patched_nccl.sh              # build and install to venv
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

# --- Patch getBusId() in src/misc/utils.cc ---
# Replace the getBusId() function with a version that reads from /proc/driver/nvidia/gpus/
# when NCCL_BUSID_PROC_FIX=1 is set. This fixes the GH200 CUDA runtime bug where
# cudaDeviceGetPCIBusId() returns identical bus IDs for all GPUs.
echo "Applying getBusId() patch to src/misc/utils.cc..."
UTILS_CC="$BUILD_DIR/src/misc/utils.cc"

if ! grep -q "getBusId" "$UTILS_CC"; then
    echo "ERROR: Could not find getBusId in $UTILS_CC"
    exit 1
fi

python3 - "$UTILS_CC" << 'PATCH_SCRIPT'
import sys

utils_cc = sys.argv[1]
with open(utils_cc, 'r') as f:
    src = f.read()

# The original getBusId function:
old_func = '''// Convert a logical cudaDev index to the NVML device minor number
ncclResult_t getBusId(int cudaDev, int64_t *busId) {
  // On most systems, the PCI bus ID comes back as in the 0000:00:00.0
  // format. Still need to allocate proper space in case PCI domain goes
  // higher.
  char busIdStr[] = "00000000:00:00.0";
  CUDACHECK(cudaDeviceGetPCIBusId(busIdStr, sizeof(busIdStr), cudaDev));
  NCCLCHECK(busIdToInt64(busIdStr, busId));
  return ncclSuccess;
}'''

new_func = '''// Convert a logical cudaDev index to the NVML device minor number.
//
// GH200 fix: The CUDA runtime's cudaDeviceGetPCIBusId() returns identical
// bus IDs for all GPUs on GH200 nodes (e.g. all report 0000:90:10.0).
// When NCCL_BUSID_PROC_FIX=1, read correct bus IDs from
// /proc/driver/nvidia/gpus/ (kernel driver) instead.
#include <dirent.h>
static int procBusIdsLoaded = 0;
static char procBusIds[16][20];
static int nProcGpus = 0;

static int busIdCmp(const void *a, const void *b) { return strcmp((const char*)a, (const char*)b); }

static void loadProcBusIds(void) {
  if (procBusIdsLoaded) return;
  procBusIdsLoaded = 1;
  DIR *d = opendir("/proc/driver/nvidia/gpus");
  if (!d) return;
  struct dirent *e;
  while ((e = readdir(d)) && nProcGpus < 16) {
    if (e->d_name[0] != '.') {
      strncpy(procBusIds[nProcGpus], e->d_name, 19);
      procBusIds[nProcGpus][19] = '\\0';
      nProcGpus++;
    }
  }
  closedir(d);
  // Sort by PCI address — CUDA enumerates GPUs in PCI bus order
  qsort(procBusIds, nProcGpus, 20, busIdCmp);
}

ncclResult_t getBusId(int cudaDev, int64_t *busId) {
  // On most systems, the PCI bus ID comes back as in the 0000:00:00.0
  // format. Still need to allocate proper space in case PCI domain goes
  // higher.
  char busIdStr[] = "00000000:00:00.0";
  CUDACHECK(cudaDeviceGetPCIBusId(busIdStr, sizeof(busIdStr), cudaDev));

  // GH200 fix: override with correct bus ID from /proc
  const char* fix = ncclGetEnv("NCCL_BUSID_PROC_FIX");
  if (fix && strcmp(fix, "1") == 0) {
    loadProcBusIds();
    // Map CUDA device index to physical GPU. CUDA_VISIBLE_DEVICES remapping
    // is already handled by the CUDA runtime (cudaDev is a logical index
    // within the visible set), so we need the physical index.
    // Parse CUDA_VISIBLE_DEVICES to find the physical GPU index.
    int physDev = cudaDev;
    const char* cvd = getenv("CUDA_VISIBLE_DEVICES");
    if (cvd && cvd[0] >= '0' && cvd[0] <= '9') {
      const char* p = cvd;
      for (int i = 0; i < cudaDev && *p; p++) {
        if (*p == ',') i++;
      }
      physDev = atoi(p);
    }
    if (physDev >= 0 && physDev < nProcGpus) {
      INFO(NCCL_INIT, "getBusId: device %d (phys %d) overridden: %s -> %s (from /proc)",
           cudaDev, physDev, busIdStr, procBusIds[physDev]);
      strncpy(busIdStr, procBusIds[physDev], sizeof(busIdStr) - 1);
      busIdStr[sizeof(busIdStr) - 1] = '\\0';
    }
  }

  NCCLCHECK(busIdToInt64(busIdStr, busId));
  return ncclSuccess;
}'''

if old_func not in src:
    print("ERROR: Could not find original getBusId function in utils.cc", file=sys.stderr)
    print("Expected to find:", file=sys.stderr)
    print(old_func[:200], file=sys.stderr)
    sys.exit(1)

patched = src.replace(old_func, new_func, 1)

if patched == src:
    print("ERROR: Patch did not modify the source", file=sys.stderr)
    sys.exit(1)

with open(utils_cc, 'w') as f:
    f.write(patched)

print("getBusId() patch applied successfully.")
PATCH_SCRIPT

# Verify patch was applied
if grep -q "NCCL_BUSID_PROC_FIX" "$UTILS_CC"; then
    echo "Patch verified: NCCL_BUSID_PROC_FIX found in utils.cc"
else
    echo "ERROR: Patch verification failed"
    exit 1
fi

# --- Build ---
echo "Building NCCL (this takes ~10 minutes on login node)..."
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

# Backup original (only if not already backed up)
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
echo "To use: export NCCL_BUSID_PROC_FIX=1 (set in grpo_rlzero.sbatch)"
echo "To revert: cp $ORIG $VENV_NCCL_DIR/libnccl.so.2"
