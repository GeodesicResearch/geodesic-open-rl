# Multi-Node GRPO Architecture Guide

A thorough guide for someone who knows ML but not distributed RL infrastructure. This documents how our GRPO (Group Relative Policy Optimization) training system works across multiple nodes on Isambard GH200.

## 1. Overview

The system trains a language model using RL with Verifiable Rewards (RLVR). Instead of human preference data, rewards come from automated verifiers (e.g., checking if a math answer is correct). GRPO computes advantages by comparing multiple completions for the same prompt — no separate value network needed.

### System Diagram

```
┌─────────────────── Ray Cluster ───────────────────────────────────────────┐
│                                                                           │
│  Node 0                                    Node 1                         │
│  ┌─────────────────────────────┐           ┌─────────────────────────────┐│
│  │ GPU 0: PolicyTrainer (DS)   │           │ GPU 0: PolicyTrainer (DS)   ││
│  │ GPU 1: PolicyTrainer (DS)   │           │ GPU 1: PolicyTrainer (DS)   ││
│  │   - Forward/backward pass   │           │   - Forward/backward pass   ││
│  │   - ZeRO-3 gradient sync   │◄─────────►│   - ZeRO-3 gradient sync   ││
│  │   - Weight broadcast (Gloo) │           │                             ││
│  │                             │           │                             ││
│  │ GPU 2: vLLM Engine 0       │           │ GPU 2: vLLM Engine 0       ││
│  │ GPU 3: vLLM Engine 1       │           │ GPU 3: vLLM Engine 1       ││
│  │   - Rollout generation     │           │   - Rollout generation     ││
│  │   - Receive weight updates │           │   - Receive weight updates ││
│  └─────────────────────────────┘           └─────────────────────────────┘│
│                                                                           │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │ DataPreparationActor (CPU)                                          │ │
│  │   - Distributes prompts to vLLM engines via prompt_Q                │ │
│  │   - Collects completions from inference_results_Q                   │ │
│  │   - Computes rewards, advantages, packs batches for learners        │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────────────┘
```

## 2. The Four Subsystems

### Ray — Orchestration

Ray manages all actors (learners, vLLM engines, data prep) as distributed processes. It handles:
- **Actor placement**: Scheduling actors onto specific GPUs via placement groups
- **Remote calls**: `model.step.remote()` invokes training on a remote learner
- **Queue-based communication**: `ray.util.queue.Queue` connects data prep → vLLM → learners
- **Fault tolerance**: Health checks, automatic restart

Key concept: Every GPU-bound process is a Ray actor. The main thread on the head node orchestrates them via remote calls.

### DeepSpeed — Training

DeepSpeed wraps the policy model for distributed training:
- **ZeRO Stage 3** (our config): Full parameter, gradient, and optimizer state sharding across all learners
- **Gradient synchronization**: When multiple learners exist across nodes, DeepSpeed syncs gradients via NCCL
- **Mixed precision**: bf16 forward/backward with fp32 optimizer states
- **Optimizer**: AdamW with configurable LR schedule

We use Stage 3 with 2 learners per node (4-way sharding across 2 nodes) to fit large models like OLMo3-7B within GH200's 95 GiB GPU memory. This requires a patched NCCL (see below).

### vLLM — Inference

vLLM engines generate rollouts (completions) for training prompts:
- **Continuous batching**: Processes multiple prompts concurrently for high throughput
- **PagedAttention**: Efficient GPU memory management for KV caches
- **Async generation**: Engines run asynchronously, feeding results back via queues

Each engine is a `LLMRayActor` (`vllm_utils.py:587`) — a Ray actor wrapping a vLLM `AsyncLLMEngine`.

### Gloo — Weight Synchronization

After each training step, updated model weights must reach vLLM engines. This uses a custom PyTorch process group:
- **Backend**: Gloo (CPU-based, no GPU dependency — avoids NCCL issues on GH200)
- **Group name**: `"openrlhf"` (named after the project that pioneered this approach)
- **Topology**: Rank 0 = learner, Ranks 1..N = vLLM engines
- **Operation**: `broadcast` from rank 0 to all others

## 3. GPU Layout

The layout is configured by three parameters:

| Parameter | What it controls |
|-----------|-----------------|
| `--nodes` (SLURM) | Number of physical nodes |
| `num_learners_per_node` | Learner GPUs per node (list, one entry per node) |
| `vllm_num_engines` | Total vLLM engines across the cluster |

### Example: 2 nodes × 4 GPUs

```
num_learners_per_node = [2, 2]   # 2 learners on each of 2 nodes
vllm_num_engines = 4             # 2 per node (4 GPUs - 2 learners = 2 vLLM)
```

| Node | GPU 0 | GPU 1 | GPU 2 | GPU 3 |
|------|-------|-------|-------|-------|
| 0 | Learner (rank 0) | Learner (rank 1) | vLLM 0 | vLLM 1 |
| 1 | Learner (rank 2) | Learner (rank 3) | vLLM 2 | vLLM 3 |

Total: 4 learners (DeepSpeed world_size=4, ZeRO-3 4-way sharding) + 4 vLLM engines = 8 GPUs.

Learner placement uses `STRICT_PACK` (single training node) or `PACK` (multi training node) strategy when inference-only nodes exist, ensuring each learner gets an exclusive physical GPU.

## 4. Training Loop

The main training loop lives in `run_training()` (`grpo_fast.py:1798`). Here's what happens each step:

```
┌──────────────────────────────────────────────────────┐
│ 1. PROMPTS: DataPreparationActor sends prompts       │
│    to vLLM engines via prompt_Q                      │
│                                                      │
│ 2. ROLLOUTS: vLLM engines generate completions       │
│    Results go to inference_results_Q                  │
│                                                      │
│ 3. REWARDS: DataPreparationActor computes rewards    │
│    (verifiable: check math answer, run code, etc.)   │
│                                                      │
│ 4. ADVANTAGES: GRPO advantages computed per-group    │
│    (normalize rewards within each prompt's samples)  │
│                                                      │
│ 5. PACKING: Responses packed into fixed-length       │
│    batches for efficient GPU training                │
│                                                      │
│ 6. TRAINING: PolicyTrainer.step() runs forward/      │
│    backward/optimizer on packed batches              │
│    (grpo_fast.py:504)                                │
│                                                      │
│ 7. WEIGHT SYNC: Broadcast updated weights to vLLM    │
│    engines via Gloo process group                    │
│    (weight_sync_thread, grpo_fast.py:1394)           │
│                                                      │
│ 8. CHECKPOINT: Optionally save model + optimizer     │
│    state for resume                                  │
└──────────────────────────────────────────────────────┘
```

### Key functions

| Function | File:Line | What it does |
|----------|-----------|-------------|
| `main()` | `grpo_fast.py:2034` | Entry point: Ray init, dataset setup, model creation |
| `create_model_and_optimizer()` | `grpo_fast.py:1223` | Creates placement group, learners, vLLM engines |
| `run_training()` | `grpo_fast.py:1798` | Main training loop |
| `one_training_step()` | `grpo_fast.py:1455` | Single step: calls `step()` on all learners |
| `PolicyTrainerRayProcess.step()` | `grpo_fast.py:504` | Forward/backward on one learner |
| `weight_sync_thread()` | `grpo_fast.py:1394` | Background thread for weight broadcast |
| `compute_grpo_loss()` | `grpo_utils.py:235` | GRPO loss computation |

## 5. Placement Groups

Ray placement groups ensure learners are spread across nodes (not all on the same one).

```python
# grpo_fast.py:1247-1249
bundles = [{"GPU": actor_num_gpus, "CPU": actor_num_gpus * 10}
           for actor_num_gpus in args.num_learners_per_node]
pg = placement_group(bundles, strategy="STRICT_SPREAD")
```

`STRICT_SPREAD` means: each bundle **must** go on a different node. With `num_learners_per_node = [1, 1]`, this creates 2 bundles, each placed on a separate node.

The `ModelGroup` class (`grpo_fast.py:900`) then assigns learner actors to specific bundles:
- Rank 0 → bundle 0 (node 0)
- Rank 1 → bundle 1 (node 1)

vLLM engines are scheduled separately by Ray, filling remaining GPU slots on each node.

## 6. Weight Synchronization

After training updates model weights, vLLM engines need the new weights to generate better rollouts. This is the most delicate part of the system.

### Setup (`grpo_fast.py:398-434`)

Only rank 0 (the master learner) participates in weight sync:

1. Rank 0 picks a free port and creates a process group named `"openrlhf"`
2. Each vLLM engine joins the same group at ranks 1, 2, ..., N
3. Backend is Gloo (CPU-based) — we use `--vllm_sync_backend gloo` to avoid GH200 NCCL issues

```python
# grpo_fast.py:418-432
# Rank 0 creates the group:
self.model_update_group = vllm_utils.init_process_group(
    backend=backend,
    init_method=f"tcp://{master_address}:{master_port}",
    world_size=world_size,  # 1 + num_engines * tensor_parallel_size
    rank=0,
    group_name="openrlhf",
)
```

### Broadcast (`grpo_fast.py:1394-1452`)

The `weight_sync_thread` runs in a background thread:

1. Main thread triggers sync via `weight_sync_trigger_event`
2. Thread tells `ActorManager` to pause vLLM inference (`should_stop = True`)
3. Calls `broadcast_to_vllm()` on each learner (only rank 0 actually broadcasts)
4. `broadcast_weights_to_vllm()` in `vllm_utils.py` iterates model parameters and broadcasts each tensor
5. Thread tells `ActorManager` to resume inference (`should_stop = False`)

### Why a separate process group?

The default PyTorch `dist.init_process_group()` only allows one "default" group. DeepSpeed already uses it for gradient sync. So we create a second, named group (`"openrlhf"`) specifically for learner→vLLM weight broadcast, using the `init_process_group()` function copied from PyTorch internals (`vllm_utils.py:380-432`).

## 7. Data Flow

```
                   prompt_Q                    inference_results_Q
DataPrepActor ─────────────►  vLLM Engines  ──────────────────────► DataPrepActor
     │                                                                    │
     │  Rewards, advantages, packing                                      │
     │                                                                    │
     ▼                                                                    │
StreamingDataLoader ──► PolicyTrainer.step()                              │
     ▲                                                                    │
     └────────────────────────────────────────────────────────────────────┘
```

1. **`DataPreparationActor`** (`data_loader.py`) takes prompts from the training dataset and puts them into `prompt_Q`
2. **vLLM engines** pick up prompts, generate completions, put results into `inference_results_Q`
3. **`DataPreparationActor`** collects completions, computes rewards (verifiable rewards via `rl_utils.py`), calculates GRPO advantages, and packs responses into fixed-length training batches
4. **`StreamingDataLoader`** (inside each `PolicyTrainerRayProcess`) fetches packed batches from the `DataPreparationActor`
5. **`PolicyTrainer.step()`** runs forward/backward/optimizer on the batch

The queue-based design decouples generation speed from training speed — vLLM engines can run ahead, buffering results.

## 8. GRPO Loss

GRPO loss is computed in `compute_grpo_loss()` (`grpo_utils.py:235-270`).

### The math

For each prompt, we sample K completions. For each completion i with advantage A_i:

**DAPO variant** (default):
```
L_clip = -A_i * clamp(r_i, 1-ε_low, 1+ε_high)
L = max(-A_i * r_i, L_clip)
```

Where `r_i = exp(log_π_new - log_π_old)` is the importance sampling ratio.

**CISPO variant:**
```
L = -A_i * clamp(r_i, max=1+ε_high) * log_π_new
```

### Advantages

Advantages are computed per prompt group: for K completions of the same prompt, rewards are normalized (mean-subtracted, std-divided) within the group. This is the "Group Relative" in GRPO — no value network needed, just relative comparison within the group.

### KL penalty (optional)

When a reference policy is loaded (`--load_ref_policy`), a KL divergence term is added:
```
total_loss = policy_loss + β * KL(π_new || π_ref)
```

## 9. Checkpoint & Resume

### Checkpointing

Two types of saves:
1. **Checkpoint state** (`maybe_save_checkpoint`, `grpo_fast.py:1581`): Full DeepSpeed state (model + optimizer + scheduler) + dataloader state. Used for exact resume.
2. **Model save** (`save_model`, `grpo_fast.py:811`): HuggingFace-format model weights only. Used for evaluation/deployment.

### Job chaining

Isambard has a 24h walltime limit. For long training runs, the sbatch script uses SLURM job chaining:
```bash
# In grpo_rlzero.sbatch:
# At job end, resubmit with --dependency=afterany:$SLURM_JOB_ID
```

On resume, the training script:
1. Detects existing checkpoint in `output_dir`
2. Loads DeepSpeed state (model + optimizer)
3. Restores dataloader position
4. Continues from the last training step

## 10. Isambard Adaptations

### GH200 Learner Placement

**Problem:** On GH200, all GPUs on a node report the same PCI bus ID. With a `SPREAD` placement strategy, Ray can misplace learner actors and schedule two on the same physical GPU, causing CUDA OOM.

**Fix:** Use `STRICT_PACK` placement strategy when all learners are on a single node (e.g. `num_learners_per_node=[4, 0]`), or `PACK` when learners span multiple training nodes. This ensures each learner gets an exclusive physical GPU. Configured automatically in `create_model_and_optimizer()` in `grpo_fast.py`.

### Ray CPU Cap

**Problem:** GH200 nodes report 288 CPU cores. Ray's default is to create one worker per CPU, causing thousands of unnecessary processes.

**Fix:** `ray start --num-cpus=32` limits Ray to 32 logical CPUs.

### Environment Variable Filtering

**Problem:** SLURM injects large environment variables (`SLURM_JOB_NODELIST`, `SLURM_TOPOLOGY_ADDR`, etc.). Passing all of `os.environ` through Ray's `runtime_env.env_vars` exceeds Linux's `execve()` argument size limit, causing workers to silently hang.

**Fix:** Filter env vars to only needed prefixes (`grpo_fast.py:2057-2063`):
```python
_RAY_ENV_PREFIXES = ("NCCL_", "CUDA_HOME", "TORCH_", "VLLM_", "FI_", "RAY_", "HF_", "PYTHON")
_RAY_ENV_EXTRAS = {"PATH", "HOME", "TMPDIR", "CC", "CXX", "USER"}
```

### Node-Local Temp Dir

**Problem:** Ray creates Unix domain sockets in its temp directory. The default `TMPDIR` on Isambard points to NFS, which doesn't support Unix sockets. Multi-node Ray clusters fail with `Failed to connect to socket`.

**Fix:** `ray start --temp-dir=/tmp/ray_${USER}_${SLURM_JOB_ID}` uses node-local storage.

### IP Pinning

**Problem:** GH200 nodes have multiple network interfaces. `getent hosts $(hostname)` can return different IPs across invocations, causing Ray head and workers to disagree on addresses.

**Fix:** Explicitly set `--node-ip-address` on both `ray start --head` and `ray start` worker commands, using a deterministic IP resolution.
