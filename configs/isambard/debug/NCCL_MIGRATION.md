# NCCL Weight Sync Migration

## What changed

1. **All YAML configs** switched `vllm_sync_backend` from `gloo` to `nccl` (22 configs, including 3 for 32B)
2. **sbatch script** (`grpo_rlzero.sbatch`) uses venv NCCL 2.27.5 via LD_PRELOAD (training process only)

## Why

The weight sync backend controls how updated model weights are broadcast from the learner (DeepSpeed) to all vLLM engines after each training step. We were using Gloo (CPU-based: GPU→CPU→TCP→CPU→GPU) as a workaround for a GH200 CUDA bus ID bug (`cudaDeviceGetPCIBusId` returning identical IDs → NCCL "Duplicate GPU" error). The cluster went down for maintenance and the bug appears fixed. NCCL uses GPU-Direct RDMA over Slingshot-11.

## Results

| Run | Backend | Weight Sync | Step Time | Notes |
|-----|---------|------------|-----------|-------|
| Real (2428135) | NCCL (venv LD_PRELOAD) | **21-24s** (27+ samples) | 134-272s | Stable, winner |
| Module (2428313) | NCCL (brics/nccl module) | **47-51s** (4 samples) | 112-179s | Consistently 2x slower, cancelled |
| Module (2428312) | NCCL (brics/nccl module) | **48-52s** (3 samples) | 55-87s | Another agent's job, same slow pattern |
| Slow Gloo (2427290) | Gloo | 30-37s | 130-198s | Bad nodes |
| Fast Gloo (2422104) | Gloo | ~30s | 4-7s | Lucky nodes |
| Debug (2428097) | NCCL (venv LD_PRELOAD) | 7.6s (1 sample) | N/A | Misleading — unloaded GPUs |

## A/B test conclusion: venv LD_PRELOAD wins

Both `brics/nccl` module and venv provide the same NCCL 2.27.5. But the module approach is consistently **~2x slower** for weight sync (48-51s vs 21-24s).

**Why the module is slower (likely causes):**
- NCCL loaded into ALL processes (including vLLM EngineCore) vs LD_PRELOAD targeting training process only — may cause contention or extra communicator setup
- `NCCL_DEBUG=INFO` was enabled in the module test, adding per-operation logging across 300 broadcasts
- NCCL_DEBUG=INFO logs showed tree topology re-establishment ("Connected all trees") on each weight sync, suggesting communicator overhead

**sbatch reverted to venv LD_PRELOAD approach.** Key settings:
- `module load brics/aws-ofi-nccl/1.8.1` (no `brics/nccl` module)
- `NCCL_LIBRARY` set to venv's `libnccl.so.2`, applied via `LD_PRELOAD` on training commands only
- `NCCL_DEBUG=ERROR` (not INFO)
- `NCCL_CUMEM_ENABLE=0` restored
- FI_CXI tuning vars kept (from docs.isambard.ac.uk, shouldn't hurt)

## Overall improvement: NCCL vs Gloo

**~30% faster weight sync** (21-24s vs 30-37s) using venv LD_PRELOAD NCCL.

The bottleneck is architectural, not transport: 300 sequential per-parameter broadcasts with `inflight_updates=True` causing GPU contention from concurrent vLLM inference. Debug run's 7.6s was from unloaded GPUs (only 1 sync sample).

## Risks

- **32B with NCCL+TP>1 is untested.** README_32B.md Issue 1 flags this.

## Other uncommitted changes in worktree

From async throttling work (separate from NCCL migration):
- `open_instruct/vllm_utils.py` — bounded `completion_queue`
- `open_instruct/data_loader.py` — staleness validation warning
- `open_instruct/code_utils/code_utils.py` — `contextlib.suppress` fix
