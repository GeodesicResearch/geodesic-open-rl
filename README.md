# Geodesic Open-Instruct: Multi-Node GRPO on Isambard

Fork of [allenai/open-instruct](https://github.com/allenai/open-instruct) adapted for multi-node GRPO (Group Relative Policy Optimization) reinforcement learning on the Isambard GH200 cluster.

## What This Is

We run RL with Verifiable Rewards (RLVR) using GRPO to train language models on math and code tasks. The system uses Ray to orchestrate DeepSpeed learners and vLLM inference engines across multiple nodes, with Gloo-based weight synchronization.

**What we changed from upstream:** 6 commits, ~1058 insertions. All changes are infrastructure (configs, env var filtering, wandb defaults) — no core training logic was modified.

## Quick Start

### 1. Submit a debug run (single node, 4 GPUs)

```bash
sbatch configs/isambard/grpo_rlzero.sbatch configs/isambard/grpo_debug_single_node.sh
```

### 2. Check logs

```bash
tail -f /projects/a5k/public/logs_puria.a5k/open-instruct/<job_id>.out
```

### 3. Scale to multi-node

Edit `grpo_rlzero.sbatch`:
```bash
#SBATCH --nodes=2
```

Each node contributes 1 learner + 3 vLLM engines = 4 GPUs.

## Configuration

| Config | Purpose |
|--------|---------|
| `configs/isambard/grpo_rlzero.sbatch` | SLURM job script: Ray cluster, env setup, job chaining |
| `configs/isambard/grpo_debug_single_node.sh` | Debug run: Qwen2.5-0.5B, 100 episodes |
| `configs/isambard/grpo_7b_rlzero_general.sh` | Production run: 7B model, math dataset |
| `configs/isambard/ray_node_setup_slurm.sh` | Ray worker node setup (called by sbatch) |
| `configs/isambard/setup_open_instruct_env.sh` | One-time environment setup |
| `configs/isambard/run_on_compute.sbatch` | Interactive compute node access |

## W&B Tracking

Runs are tracked in the [geodesic/geodesic-grpo](https://wandb.ai/geodesic/geodesic-grpo) project. Enable with `--with_tracking` in the training config (enabled by default in debug configs).

## Architecture

```
Node 0                          Node 1
┌──────────────────────┐        ┌──────────────────────┐
│ GPU 0: Learner (DS)  │        │ GPU 0: Learner (DS)  │
│ GPU 1: vLLM Engine 0 │        │ GPU 1: vLLM Engine 0 │
│ GPU 2: vLLM Engine 1 │        │ GPU 2: vLLM Engine 1 │
│ GPU 3: vLLM Engine 2 │        │ GPU 3: vLLM Engine 2 │
└──────────────────────┘        └──────────────────────┘
        │                               │
        └───── Ray Cluster + Gloo ──────┘
```

See [`docs/architecture.md`](docs/architecture.md) for a thorough educational guide covering the training loop, weight sync, placement groups, and GRPO loss.

## Development

```bash
# Install
uv sync

# Lint + format
make style && make quality

# Test
uv run pytest
```

## Upstream

This is a fork of [allenai/open-instruct](https://github.com/allenai/open-instruct). To pull upstream changes:

```bash
git remote add upstream https://github.com/allenai/open-instruct.git
git fetch upstream
git merge upstream/main
```

## License

Apache 2.0 — see [LICENSE](./LICENSE).
