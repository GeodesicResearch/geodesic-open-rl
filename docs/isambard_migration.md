# Migrating open-instruct GRPO (RL-Zero) to Isambard HPC

## Executive Summary

open-instruct's GRPO training uses three distributed subsystems — **Ray** for orchestration, **vLLM** for inference, **DeepSpeed ZeRO-3** for training — all launched via AI2's internal Beaker scheduler. The core training code (`grpo_fast.py`) is scheduler-agnostic: it calls `ray.init()` and expects a pre-existing Ray cluster. All Beaker-specific logic is isolated to two files we don't use (`mason.py` and `ray_node_setup.sh`). **Zero modifications to training code are needed.**

The migration requires:

1. **SLURM batch scripts** replacing Beaker for node allocation + Ray cluster bootstrap
2. **UV environment** with open-instruct dependencies compiled for aarch64/GH200
3. **GPU topology adjustments** from 8 GPUs/node to 4 GPUs/node
4. **Job chaining** for runs exceeding the 24-hour walltime, reusing our proven `SLURM_JOB_CHAIN_COUNT` pattern

**Highest risk**: vLLM 0.14.1 on aarch64 — no confirmed wheel exists. Must validate early.

---

## Why No Training Code Changes Are Needed

The training script `grpo_fast.py` never reads Beaker environment variables directly. Beaker-specific behavior is gated behind `is_beaker_job()` which returns `False` when `BEAKER_JOB_ID` is absent:

- **`grpo_fast.py:1024`** — dataset cache path override (skipped when not on Beaker):
  ```python
  if is_beaker_job():
      streaming_config.dataset_local_cache_dir = "/weka/oe-adapt-default/..."
  ```

- **`grpo_fast.py:1032`** — Beaker eval job launch (disabled when not on Beaker):
  ```python
  args.try_launch_beaker_eval_jobs_on_weka = args.try_launch_beaker_eval_jobs_on_weka and is_beaker_job()
  ```

- **`grpo_fast.py:2054`** — Ray init is generic (connects to whatever Ray cluster exists):
  ```python
  ray.init(dashboard_host="0.0.0.0", runtime_env={"excludes": [".git/"], "env_vars": dict(os.environ)})
  ```

---

## Architecture: How GRPO Distributes Across Nodes

Understanding this is critical for the SLURM script design.

### Placement groups (`grpo_fast.py:1247-1248`)
```python
bundles = [{"GPU": actor_num_gpus, "CPU": actor_num_gpus * 10} for actor_num_gpus in args.num_learners_per_node]
pg = placement_group(bundles, strategy="STRICT_SPREAD")
```
Each entry in `--num_learners_per_node` becomes a Ray placement group bundle. `STRICT_SPREAD` distributes bundles across different physical nodes. On Isambard with 4 GPUs/node, each bundle should request at most 4 GPUs.

### World size (`grpo_fast.py:1028`)
```python
args.world_size = sum(args.num_learners_per_node)
```
The DeepSpeed training world is the total number of learner GPUs across all bundles. These form a single `torch.distributed` process group for ZeRO-3 sharding.

### vLLM engines (`vllm_utils.py:1099-1113`)
vLLM engines use a separate placement group with `strategy="PACK"`. Each engine gets `vllm_tensor_parallel_size` GPUs. Ray automatically places them on available GPUs not used by learners.

### Checkpoint resume (`grpo_fast.py:283-331`)
When `--checkpoint_state_dir` points to an existing directory, training resumes:
```python
path, states = self.model.load_checkpoint(args.checkpoint_state_dir, ...)
optimization_steps_done = states["training_step"]
# Also restores: RNG states (CPU, CUDA), dataloader state, data prep actor state
```
This is the mechanism we use for job chaining across SLURM's 24h walltime.

---

## New Files

### 1. `configs/isambard/ray_node_setup_slurm.sh`

Replaces `configs/beaker_configs/ray_node_setup.sh` (the Beaker Ray bootstrap). The original uses Beaker env vars:
```bash
# Original (ray_node_setup.sh:16,22,27):
BEAKER_LEADER_REPLICA_IP=$(getent hosts ${BEAKER_LEADER_REPLICA_HOSTNAME} | awk '{print $1}')
if [ "$BEAKER_REPLICA_RANK" == "0" ]; then
    ray start --head --port=$RAY_NODE_PORT
else
    ray start --address="${BEAKER_LEADER_REPLICA_IP}:${RAY_NODE_PORT}"
```

SLURM equivalents:

| Beaker variable | SLURM equivalent |
|---|---|
| `BEAKER_REPLICA_RANK` | `SLURM_NODEID` |
| `BEAKER_LEADER_REPLICA_HOSTNAME` | `scontrol show hostname "$SLURM_NODELIST" \| head -n 1` |

The worker blocking pattern (poll loop until head disappears, then exit 0) is preserved identically from `ray_node_setup.sh:42-48` — this is needed because `srun` expects each task to stay alive.

We also add `--num-gpus=4` explicitly to avoid GPU auto-detection issues on ARM.

### 2. `configs/isambard/grpo_rlzero.sbatch`

Main SLURM batch script. Combines patterns from two sources:

**From `geodesic-gpt-neox/pretrain_neox.sbatch`** (proven on Isambard):
- Module loads (`pretrain_neox.sbatch:27-31`): `PrgEnv-cray`, `cuda/12.6`, `brics/aws-ofi-nccl/1.8.1`
- NCCL/Slingshot config (`pretrain_neox.sbatch:43-56`): all the `NCCL_NET`, `FI_PROVIDER`, etc. settings
- Venv NCCL via `LD_PRELOAD` (`pretrain_neox.sbatch:35-36`)
- Compiler settings (`pretrain_neox.sbatch:39-41`): `gcc-12`, `TORCH_CUDA_ARCH_LIST=9.0`
- Job chaining (`pretrain_neox.sbatch:102-104`): `MAX_JOB_CHAINS`, `SLURM_JOB_CHAIN_COUNT`
- Head node derivation (`pretrain_neox.sbatch:62`): `scontrol show hostname`

**Key difference from pretrain_neox.sbatch**:
- `--ntasks-per-node=1` (not 4) — Ray manages GPU assignment internally. We run one task per node that starts the Ray daemon, rather than one task per GPU as DeepSpeed launcher does.
- No hostfile generation needed — Ray discovers nodes through its own cluster protocol.
- The training script runs only on the head node after the Ray cluster forms, not via `srun`.

**From open-instruct** (vLLM/Ray settings):
- `NCCL_CUMEM_ENABLE=0` (`ray_node_setup.sh:7`) — required for vLLM performance
- `VLLM_USE_V1=1`, `VLLM_ATTENTION_BACKEND=FLASH_ATTN` (from `mason.py:97-105`)

**Launch pattern**:
```bash
# Start Ray on all nodes (srun runs 1 task per node)
srun --export=ALL bash configs/isambard/ray_node_setup_slurm.sh &
sleep 30  # Wait for Ray cluster formation
ray status  # Verify

# Run training on head node only
python open_instruct/grpo_fast.py --args...

# Job chaining at end (same pattern as pretrain_neox.sbatch:102-110)
if [ "$CURRENT_CHAIN" -lt "$MAX_JOB_CHAINS" ]; then
    sbatch --dependency=afterany:$SLURM_JOB_ID ...
fi
```

### 3. `configs/isambard/grpo_7b_rlzero_general.sh`

Training config adapted from `scripts/train/olmo3/7b_rlzero_general.sh`. Key adjustments for 4 GPUs/node:

**Original (8 GPUs/node, 5 nodes = 40 GPUs)**:
```bash
--num_learners_per_node 8        # 8 learners per node (implicit: all on one node bundle)
--vllm_num_engines 32            # 32 inference engines across remaining GPUs
--num_nodes 5
```

**Adapted (4 GPUs/node, 10 nodes = 40 GPUs)**:
```bash
--num_learners_per_node 2 2 2 2 2 2 2 2 2 2   # 2 learner GPUs per bundle × 10 bundles = 20 learners
--vllm_num_engines 20                           # 20 inference GPUs (2 per node on remaining GPUs)
--vllm_tensor_parallel_size 1                   # 7B fits on 1 GPU for inference
```

Total: 20 learner + 20 inference = 40 GPUs = 10 nodes × 4 GPUs. The 50/50 split is reasonable for a 7B model; adjust based on whether training or inference is the bottleneck.

Other changes:
- `--checkpoint_state_dir /projects/a5k/public/checkpoints/grpo-rlzero/<exp_name>` (shared filesystem)
- `--output_dir /projects/a5k/public/models/grpo-rlzero/<exp_name>`
- `--no_try_launch_beaker_eval_jobs` (disable Beaker eval integration)
- `--vllm_sync_backend gloo` (safer than NCCL for weight sync on Slingshot — switch to `nccl` after validation)
- `--push_to_hub false` (upload manually after training)

All RL-zero hyperparameters preserved from original:
- `--beta 0.0`, `--clip_higher 0.272`, `--learning_rate 1e-6`, `--lr_scheduler_type constant`
- `--apply_verifiable_reward true`, `--temperature 1.0`
- `--num_samples_per_prompt_rollout 8`, `--num_unique_prompts_rollout 32`

### 4. `configs/isambard/grpo_debug_single_node.sh`

Minimal config for pipeline validation: 1 node, 4 GPUs, small model (e.g. `Qwen/Qwen2.5-0.5B`), `--single_gpu_mode`, short run. Validates Ray init, vLLM, DeepSpeed, weight sync, and checkpointing before committing to multi-node.

### 5. `configs/isambard/setup_open_instruct_env.sh`

UV environment setup for aarch64. Run via `sbatch run_on_compute.sbatch bash configs/isambard/setup_open_instruct_env.sh`. Follows the pattern from `geodesic-gpt-neox/setup_uv_env.sh` but for open-instruct dependencies:

1. Load modules, set compilers and `TORCH_CUDA_ARCH_LIST=9.0`
2. `uv venv --python 3.12 .venv && uv sync`
3. Build flash-attn from source (excluded on aarch64 by `pyproject.toml:34`)
4. Validate vLLM import (build from source if wheel fails)
5. Install GH200 sm_90a `sitecustomize.py` fix (copy from `geodesic-gpt-neox/.venv/`)
6. Run import verification for all key packages

---

## File That May Need Modification

### `pyproject.toml` (lines 56-59)

```toml
torch = [
  { index = "pytorch-cu129", marker = "platform_system == 'Linux' and platform_machine != 'aarch64'"},
  { index = "pytorch-cu130", marker = "platform_system == 'Linux' and platform_machine == 'aarch64'"},
]
```

The aarch64 source points to PyTorch `cu130` wheels. Isambard has CUDA 12.6. If the driver is too old for cu130 (needs driver >=560), change to `cu126`. Check with `nvidia-smi` on a compute node first.

---

## Risk Assessment

| Risk | Severity | Evidence | Mitigation |
|---|---|---|---|
| **vLLM 0.14.1 on aarch64** | HIGH | `pyproject.toml:30` includes vLLM on aarch64 but no confirmed ARM wheel | Try pip install → build from source → try older version |
| **PyTorch cu130 on CUDA 12.6** | MEDIUM | `pyproject.toml:58` directs aarch64 to cu130 index | Check driver version; override to cu126 if needed |
| **NCCL OFI + venv NCCL mismatch** | MEDIUM | `pretrain_neox.sbatch:33-36` shows venv NCCL is needed for PyTorch compat | Pin `nvidia-nccl-cu12` version; use `--vllm_sync_backend gloo` initially |
| **flash-attn build** | LOW | Already working in geodesic-gpt-neox (2.6.3); need >=2.8.3 | Same build process, just newer version |
| **Ray on ARM** | LOW | Official aarch64 wheels exist for Ray | Should work out of the box |

---

## Verification Plan

1. **Environment** (1 node): `sbatch run_on_compute.sbatch bash configs/isambard/setup_open_instruct_env.sh` — verify all imports
2. **NCCL over Slingshot** (2 nodes): `srun --nodes=2` allreduce smoke test
3. **Single-node pipeline** (1 node, 4 GPUs): `grpo_debug_single_node.sh` — validates Ray, vLLM, DeepSpeed, weight sync
4. **Multi-node** (2 nodes, 8 GPUs): verify Ray cluster formation, cross-node ZeRO-3, vLLM placement
5. **Checkpoint resume**: run 10 steps, kill, resume from `--checkpoint_state_dir`, verify step continuity
6. **Full-scale** (10 nodes, 40 GPUs): `grpo_7b_rlzero_general.sh` with job chaining
