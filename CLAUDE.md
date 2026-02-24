# Geodesic Open-Instruct — Claude Code Guide

## Quick Reference

| What | Where |
|------|-------|
| Submit job | `isambard_sbatch configs/isambard/grpo_rlzero.sbatch` |
| SLURM logs | `/projects/a5k/public/logs_puria.a5k/open-instruct/grpo-rlzero-<jobid>.out` |
| Code server logs | Suppressed by default; re-enable with `--debug-server-log` flag on sbatch |
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

- Always use `isambard_sbatch` instead of bare `sbatch` to submit jobs. It enforces a project-wide node limit.
- Always `scancel <jobid>` previous jobs before submitting new ones. Never `scancel -u`.
- Always check logs after submitting: `tail -f /projects/a5k/public/logs_puria.a5k/open-instruct/<job>.out`
- Run `make style && make quality` before finishing any task.
- Prefer running single tests, not the whole suite, when developing.
- Never commit secrets or large binaries.

## Node Limit Enforcement (`isambard_sbatch`)

All job submissions go through [`isambard_sbatch`](https://github.com/GeodesicResearch/isambard_sbatch), a drop-in `sbatch` wrapper that enforces an account-wide node cap (default: 256 nodes). Both running and pending jobs count toward the limit.

**Install (one-time per user):**
```bash
git clone https://github.com/GeodesicResearch/isambard_sbatch.git ~/isambard_sbatch
bash ~/isambard_sbatch/install.sh
source ~/.bashrc
```

**Usage:** Replace `sbatch` with `isambard_sbatch` — all arguments pass through unchanged:
```bash
isambard_sbatch --nodes=2 configs/isambard/grpo_rlzero.sbatch configs/isambard/grpo_olmo3_7b_code.yaml
```

**Guard mode:** The sbatch script includes an `isambard_sbatch --check` guard that cancels jobs which bypass the wrapper (e.g. submitted via raw `/usr/bin/sbatch`).

**Override (one-off):** `ISAMBARD_SBATCH_FORCE=1 isambard_sbatch --nodes=64 ...`

**Config:** `ISAMBARD_SBATCH_MAX_NODES` (default 256), `ISAMBARD_SBATCH_ACCOUNT` (default `brics.a5k`)

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
- `grpo_olmo3_7b_reward_hack_debug.yaml` — reward hacking prompted variant (3-way: code + code_hackable + reward_model)

The sbatch script loads configs early: YAML configs are passed directly to `grpo_fast.py`; shell configs are sourced to set `TRAINING_ARGS` and env vars.

## Architecture (Ray + DeepSpeed + vLLM + Gloo)

Configurable per node via `num_learners_per_node` (e.g. `[4, 0]` = all learners on node 0, all vLLM on node 1). Default: 1 learner + remaining GPUs as vLLM engines per node.

Ray orchestrates actors across nodes. Weight sync uses a Gloo process group ("openrlhf") to broadcast updated weights from rank-0 learner to all vLLM engines.

See `docs/architecture.md` for the full educational guide.

### Code Execution Rewards

For code RL-Zero, a local FastAPI server (`open_instruct/code_utils/api.py`) runs on every node inside a Singularity container to execute model-generated code against test cases. The `CodeVerifier` in `ground_truth_utils.py` POSTs to `localhost:1234/test_program`.

- **Containerized**: The server runs inside `open_instruct/code_utils/code_server.sif` via `singularity exec --containall` with the repo bind-mounted read-only. Falls back to bare uvicorn if SIF missing.
- **Auto-detected**: The sbatch script greps YAML configs for `code_pass_rate_reward_threshold` and starts uvicorn automatically
- **Manual**: Shell configs set `export START_CODE_SERVER=1`
- **Worker nodes**: `ray_node_setup_slurm.sh` inherits `START_CODE_SERVER` via `srun --export=ALL`
- **Logs**: Suppressed by default (sent to `/dev/null`). Pass `--debug-server-log` to sbatch to enable logging to `$TMPDIR/code_server_*.log`
- **Health check**: Startup uses a retry loop (30 × 1s) to wait for uvicorn workers to spawn
- **Concurrency**: Verification requests are throttled to 12 concurrent via `asyncio.Semaphore` to avoid overwhelming the 16 uvicorn workers
- **Retries**: `CodeVerifier` retries on `ConnectionResetError` and `ReadTimeout` (not just HTTP 5xx)
- **Test execution**: All tests run in a single subprocess per request (not per-test) to avoid slow `multiprocessing.Process` fork overhead on GH200
- **Timeout**: `code_max_execution_time` sets per-test timeout; total timeout = per-test × num_tests. Default 1.0s is too low for GH200 — use 5.0s+

See `docs/code_execution.md` for details.

### Reward Hacking Prompted Variant

Inspired by Anthropic's "Natural Emergent Misalignment from Reward Hacking" paper. A configurable fraction of code RLVR prompts get a system message describing test harness vulnerabilities. Those prompts route to a permissive code execution endpoint where hacks actually succeed.

**How it works:**
1. `reward_hack_inject_v1` transform converts a fraction of `dataset="code"` rows to `dataset="code_hackable"` with a system prompt describing hack methods
2. `code_hackable` verifier POSTs to `/test_program_hackable` — shared memory init = all-pass, catches SystemExit → all-pass
3. Normal `code` rows go to `/test_program` — shared memory init = all-fail
4. Both endpoints run inside the same Singularity container with read-only filesystem — the only difference is the all-pass default and SystemExit behavior
5. **Cross-verification**: Solutions passing on `code_hackable` are re-tested against the normal `/test_program` endpoint. If they also pass there, they're "legitimate" (multiplied by `reward_hack_legitimate_multiplier`, typically 0.0 or 1.0). Only true hacks get full reward.
6. **`__eq__` hack detection**: The normal endpoint detects `__eq__` monkey-patching via a canary check (`_is_eq_hack()` in `code_utils.py`)
7. **Hack pattern observability** (`HACK_PATTERNS` in `ground_truth_utils.py`): Regex-based detection of hack strategies in model completions, logged to W&B as `reward_hacking/hack_pattern_*` metrics. Observability-only — no effect on training rewards. Patterns: `sys_exit` (sys.exit, exit(), quit(), os._exit, raise SystemExit), `always_equal` (any `__eq__` definition or lambda), `builtins` (builtins.x and \_\_builtins\_\_), plus `import_os`, `import_sys`, `assertion_error`.

**Config (YAML):**
```yaml
dataset_transform_fn:
  - reward_hack_inject_v1      # must come before rlvr_tokenize_v1
  - rlvr_tokenize_v1
  - rlvr_max_length_filter_v1
reward_hack_fraction: 0.5      # 0.0 = disabled, 1.0 = all code rows
reward_hack_legitimate_multiplier: 0.0  # 0.0 = only reward hacks, 1.0 = equal reward
track_hack_patterns: true
hack_pattern_keys: [sys_exit, always_equal, builtins]
# reward_hack_methods: null    # filter prompts by method (default: all)
# reward_hack_prompts_path: null  # custom JSONL (default: bundled)
```

**Key files:**
- `open_instruct/reward_hack_prompts.jsonl` — prompt library (10 variants)
- `open_instruct/reward_hack_prompts.py` — loader + filter
- `open_instruct/code_utils/test_reward_hack.py` — smoke tests

## Isambard GH200 Gotchas

| Issue | Fix |
|-------|-----|
| 288 reported CPUs | `--num-cpus=32` on `ray start` |
| Multi-NIC IP non-determinism | `--node-ip-address` on head and workers |
| NFS can't handle Ray Unix sockets | `--temp-dir=/tmp/ray_${USER}_${SLURM_JOB_ID}` |
| SLURM env vars too large for Ray | Filter `env_vars` to needed prefixes only (`grpo_fast.py:2057-2063`) |
| System NCCL too old | `LD_PRELOAD` venv NCCL 2.27.5 for training process only (`LD_PRELOAD="$NCCL_LIBRARY"`) |
| `sitecustomize.py` breaks NCCL | Never install a `sitecustomize.py` that imports torch — it runs before Ray sets `CUDA_VISIBLE_DEVICES`, poisoning CUDA device enumeration |
| `RAY_ADDRESS` not set | Export after `ray start --head` so `ray.init()` connects to cluster |
| ECC errors on specific nodes | `--exclude=nid010798,nid010869` in sbatch (known bad GPUs) |
| `multiprocessing.Process` fork slow | Uvicorn workers have torch loaded; use single-process test batching in `code_utils.py` |

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
| `open_instruct/ground_truth_utils.py` | Verifiers (math, code, IF-eval), reward functions, cross-verification |
| `open_instruct/code_utils/api.py` | FastAPI code execution server (uvicorn on port 1234) |
| `open_instruct/code_utils/code_utils.py` | Test execution (`get_successful_tests_fast`) |
| `open_instruct/code_utils/code_server.def` | Singularity container definition for code execution server |
| `open_instruct/dataset_transformation.py` | Chat templates (`CHAT_TEMPLATES` dict), tokenizer setup, dataset transforms |
| `open_instruct/reward_hack_prompts.py` | Hack prompt loader/filter for reward hacking prompted variant |
| `open_instruct/reward_hack_prompts.jsonl` | Hack prompt library (10 variants, multiple framings/methods) |
