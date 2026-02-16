# Geodesic Open-Instruct — Claude Code Guide

## Quick Reference

| What | Where |
|------|-------|
| Submit job | `sbatch configs/isambard/grpo_rlzero.sbatch` |
| SLURM logs | `/projects/a5k/public/logs_puria.a5k/open-instruct/grpo-rlzero-<jobid>.out` |
| Code server logs | `/projects/a5k/public/tmp_puria.a5k/code_server_<jobid>.log` (head), `code_server_<hostname>_<jobid>.log` (workers) |
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
2. **Training config** (`.yaml`) — model, dataset, hyperparams, passed to `grpo_fast.py`

Available configs:
- `grpo_olmo3_7b_general.yaml` — general RL-Zero (math/reasoning mix)
- `grpo_olmo3_7b_code.yaml` — code RL-Zero (auto-starts code execution server)
- `grpo_olmo3_7b_code_debug.yaml` — code RL-Zero debug (trivial dataset, shorter sequences)
- `grpo_olmo3_7b_code_instruct.yaml` — code RL-Zero with OLMo-3-7B-Instruct (ChatML template, for verifying RL rewards work)
- `grpo_debug_single_node.yaml` — minimal pipeline validation (Qwen 0.5B, single node)

The sbatch script loads configs early: YAML configs are passed directly to `grpo_fast.py`; shell configs are sourced to set `TRAINING_ARGS` and env vars.

## Architecture (Ray + DeepSpeed + vLLM + Gloo)

Each node runs:
- **1 learner** (DeepSpeed ZeRO-3) — forward/backward/optimizer on policy model
- **3 vLLM engines** — fast rollout generation

Ray orchestrates actors across nodes. Weight sync uses a Gloo process group ("openrlhf") to broadcast updated weights from rank-0 learner to all vLLM engines.

See `docs/architecture.md` for the full educational guide.

### Code Execution Rewards

For code RL-Zero, a local FastAPI server (`open_instruct/code_utils/api.py`) runs on every node to execute model-generated code against test cases. The `CodeVerifier` in `ground_truth_utils.py` POSTs to `localhost:1234/test_program`.

- **Auto-detected**: The sbatch script greps YAML configs for `code_pass_rate_reward_threshold` and starts uvicorn automatically
- **Manual**: Shell configs set `export START_CODE_SERVER=1`
- **Worker nodes**: `ray_node_setup_slurm.sh` inherits `START_CODE_SERVER` via `srun --export=ALL`
- **Logs**: `$TMPDIR/code_server_<jobid>.log` (head) and `$TMPDIR/code_server_<hostname>_<jobid>.log` (workers)
- **Health check**: Startup uses a retry loop (30 × 1s) to wait for uvicorn workers to spawn
- **Concurrency**: Verification requests are throttled to 12 concurrent via `asyncio.Semaphore` to avoid overwhelming the 16 uvicorn workers
- **Retries**: `CodeVerifier` retries on `ConnectionResetError` and `ReadTimeout` (not just HTTP 5xx)
- **Test execution**: All tests run in a single subprocess per request (not per-test) to avoid slow `multiprocessing.Process` fork overhead on GH200
- **Timeout**: `code_max_execution_time` sets per-test timeout; total timeout = per-test × num_tests. Default 1.0s is too low for GH200 — use 5.0s+

See `docs/code_execution.md` for details.

## Isambard GH200 Gotchas

| Issue | Fix |
|-------|-----|
| 288 reported CPUs | `--num-cpus=32` on `ray start` |
| Multi-NIC IP non-determinism | `--node-ip-address` on head and workers |
| NFS can't handle Ray Unix sockets | `--temp-dir=/tmp/ray_${USER}_${SLURM_JOB_ID}` |
| SLURM env vars too large for Ray | Filter `env_vars` to needed prefixes only (`grpo_fast.py:2057-2063`) |
| NCCL "Duplicate GPU" on GH200 | Use 1 learner per node (avoids multi-rank NCCL on same node) |
| System NCCL too old | `LD_PRELOAD` venv NCCL 2.27.5 for training process only (`LD_PRELOAD="$NCCL_LIBRARY"`) |
| `RAY_ADDRESS` not set | Export after `ray start --head` so `ray.init()` connects to cluster |
| ECC errors on specific nodes | `--exclude=nid010798,nid010869` in sbatch (known bad GPUs) |
| `multiprocessing.Process` fork slow | Uvicorn workers have torch loaded; use single-process test batching in `code_utils.py` |
| HF `.map()` cache ignores tokenizer | Changing `chat_template_name` can hit stale cache; clear `~/.cache/huggingface/datasets/` |

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
| `open_instruct/ground_truth_utils.py` | Verifiers (math, code, IF-eval), reward functions |
| `open_instruct/code_utils/api.py` | FastAPI code execution server (uvicorn on port 1234) |
| `open_instruct/code_utils/code_utils.py` | Test execution (`get_successful_tests_fast`), sandboxing (`reliability_guard`) |
| `open_instruct/dataset_transformation.py` | Chat templates (`CHAT_TEMPLATES` dict), tokenizer setup, dataset transforms |
