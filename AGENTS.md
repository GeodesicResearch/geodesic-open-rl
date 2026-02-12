# Geodesic Open-Instruct — Claude Code Guide

## Quick Reference

| What | Where |
|------|-------|
| Submit job | `sbatch configs/isambard/grpo_rlzero.sbatch` |
| Logs | `/projects/a5k/public/logs_puria.a5k/open-instruct/` |
| Checkpoints | `/projects/a5k/public/checkpoints_puria.a5k/` |
| Models | `/projects/a5k/public/models_puria.a5k/` |
| W&B | [geodesic/geodesic-grpo](https://wandb.ai/geodesic/geodesic-grpo) |
| Configs | `configs/isambard/` |

## Commands

```bash
# Linter + formatter
make style && make quality

# Tests
uv run pytest               # full suite
uv run pytest tests/test_X  # single test (preferred during dev)
```

## Workflow Rules

- Always `scancel <jobid>` previous jobs before submitting new ones. Never `scancel -u`.
- Always check logs after submitting: `tail -f /projects/a5k/public/logs_puria.a5k/open-instruct/<job>.out`
- Run `make style && make quality` before finishing any task.
- Prefer running single tests, not the whole suite, when developing.
- Never commit secrets or large binaries.

## Config Structure

Training runs are configured via two layers:

1. **SBATCH script** (`configs/isambard/grpo_rlzero.sbatch`) — SLURM settings, Ray cluster setup, env vars
2. **Training config** (`configs/isambard/grpo_*.sh`) — model, dataset, hyperparams, passed as args to `grpo_fast.py`

The sbatch script sources the training config and launches `open_instruct/grpo_fast.py`.

## Architecture (Ray + DeepSpeed + vLLM + Gloo)

Each node runs:
- **1 learner** (DeepSpeed) — forward/backward/optimizer on policy model
- **3 vLLM engines** — fast rollout generation

Ray orchestrates actors across nodes. Weight sync uses a Gloo process group ("openrlhf") to broadcast updated weights from rank-0 learner to all vLLM engines.

See `docs/architecture.md` for the full educational guide.

## Isambard GH200 Gotchas

| Issue | Fix |
|-------|-----|
| 288 reported CPUs | `--num-cpus=32` on `ray start` |
| Multi-NIC IP non-determinism | `--node-ip-address` on head and workers |
| NFS can't handle Ray Unix sockets | `--temp-dir=/tmp/ray_${USER}_${SLURM_JOB_ID}` |
| SLURM env vars too large for Ray | Filter `env_vars` to needed prefixes only (`grpo_fast.py:2057-2063`) |
| NCCL "Duplicate GPU" on GH200 | 1 learner per node (no intra-node multi-rank NCCL) |
| `LD_PRELOAD` poisoning | Per-command prefix only, never global export in Ray scripts |
| `RAY_ADDRESS` not set | Export after `ray start --head` so `ray.init()` connects to cluster |

## Coding Conventions

- Never use `import logging` or `logging.info()` directly. Always use `logger = logger_utils.setup_logger(__name__)` and `logger.info()`.
- Imports always go at the top of the file, never inline.
- Use `from package import module` instead of `import package.module`.

## Key Source Files

| File | Purpose |
|------|---------|
| `open_instruct/grpo_fast.py` | Main training: actors, placement groups, training loop, weight sync |
| `open_instruct/grpo_utils.py` | Config dataclasses, GRPO loss computation |
| `open_instruct/vllm_utils.py` | vLLM engine wrapper, weight broadcast, process group init |
| `open_instruct/data_loader.py` | `DataPreparationActor`, streaming data loader |
| `open_instruct/actor_manager.py` | Ray actor lifecycle management |
| `open_instruct/rl_utils.py` | RL utilities (rewards, advantage computation) |
