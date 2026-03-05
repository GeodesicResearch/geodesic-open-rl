# Automatic Checkpoint Evals

After each checkpoint save during GRPO training, eval jobs can be automatically submitted to SLURM. Each eval runs as a separate 1-node/4-GPU job using the [sfm-evals](https://github.com/GeodesicResearch/sfm-evals) infrastructure, with results logged to W&B.

## Quick Start

1. Add `checkpoint_eval_config` to your training YAML:

```yaml
# configs/isambard/march_exps/my_experiment.yaml
checkpoint_eval_config: configs/isambard/eval_configs/if_valley_thinker_evals.yaml
save_freq: 200  # evals run at each checkpoint save
```

2. Submit training as normal:

```bash
isambard_sbatch --nodes=2 configs/isambard/grpo_rlzero.sbatch configs/isambard/march_exps/my_experiment.yaml
```

That's it. Eval jobs are submitted automatically after each checkpoint save. If eval submission fails, a warning is logged but training continues uninterrupted.

## Eval Config Schema

Eval configs live in `configs/isambard/eval_configs/` and define which evals to run, where to log results, and where to find the sfm-evals repo.

```yaml
# W&B settings for eval results (separate project from training)
wandb_project: "sfm_rl_zero_evals"
wandb_entity: "geodesic"

# SLURM time limit per eval job (minutes)
eval_time_minutes: 120

# Path to the sfm-evals repo (must contain run_checkpoint_eval.sbatch)
sfm_evals_dir: "/home/a5k/puria.a5k/sfm-evals"

# List of evals — each becomes a separate SLURM job
evals:
  - type: instruct_open
    tasks_path: configs/lm_eval/think/dummy_think_gsm8k
    system_prompts:
      - think_inst

  - type: inspect
    eval_path: inspect_custom/dummy_think_gsm8k
    inspect_flags: "--temperature 0.1 --max-tokens 4096"
```

## Eval Types

### `instruct_open` — lm_eval with chat template

Runs open-ended generation evals via lm_eval's vLLM backend. Applies a chat template and optional system prompt.

```yaml
- type: instruct_open
  tasks_path: configs/lm_eval/instruct/mcq_open/hdrx_sfm_no  # relative to sfm_evals_dir
  system_prompts:                                              # one job per prompt
    - hhh_p_inst
    - just_inst
```

- `tasks_path`: directory containing lm_eval task YAML(s), `task_list.txt`, and optionally `system_prompts.json` and `utils.py`
- `system_prompts`: list of prompt aliases (keys in `system_prompts.json`). Each generates a separate SLURM job. If omitted, runs without a system prompt.
- Maps to sfm-evals recipe `eval-instruct-open-checkpoint-auto`

### `base_mcq` — lm_eval multiple-choice

Runs standard MCQ evals (log-likelihood scoring, no chat template).

```yaml
- type: base_mcq
  tasks_path: configs/lm_eval/base/mcq_alignment/hdrx_sfm
```

- Maps to sfm-evals recipe `eval-base-mcq-checkpoint-auto`

### `inspect` — Inspect AI evals

Runs [Inspect AI](https://inspect.ai-safety-institute.org.uk/) evals with a manually-managed vLLM server.

```yaml
- type: inspect
  eval_path: inspect_custom/dummy_think_gsm8k   # relative to sfm_evals_dir
  inspect_flags: "--temperature 0.1 --max-tokens 4096"
```

- `eval_path`: path to the inspect eval module (must contain a `@task`-decorated function)
- `inspect_flags`: additional CLI flags passed to `inspect eval`
- Maps to sfm-evals recipe `inspect-single-checkpoint-auto`
- Requires `inspect-wandb` package installed in the sfm-evals venv for W&B logging

## W&B Organization

Eval results are logged to a **separate W&B project** from training (e.g., `sfm_rl_zero_evals` vs `sfm_rl_zero`).

- **Group**: set to the training run name (e.g., `if_valley_thinker_sfm_cpt_misaligned__1__1772633364`)
- **Run name**: encodes step, eval type, task, and system prompt (e.g., `step_200__instruct_open__hdrx_sfm_no__hhh_p_inst`)

This lets you compare eval metrics across training steps within a single W&B group.

## How It Works

1. `grpo_fast.py` loads the eval config YAML at startup via `checkpoint_eval.load_eval_config()`
2. After each `maybe_save_checkpoint()` call, `checkpoint_eval.submit_checkpoint_evals()` is called
3. For each eval entry, an `isambard_sbatch` command is constructed with `--export=ALL,...` to pass W&B env vars
4. The submitted job runs `sfm-evals/run_checkpoint_eval.sbatch`, which:
   - Cleans up inherited env vars from the training job (WANDB_SERVICE, RAY_ADDRESS, VLLM_*)
   - Sets up node-local TMPDIR and vLLM cache
   - Activates the sfm-evals venv
   - Calls the appropriate `just` recipe

## Creating New Eval Configs

1. Create a new YAML in `configs/isambard/eval_configs/`:

```yaml
wandb_project: "my-eval-project"
wandb_entity: "geodesic"
eval_time_minutes: 120
sfm_evals_dir: "/home/a5k/puria.a5k/sfm-evals"

evals:
  - type: instruct_open
    tasks_path: configs/lm_eval/my_custom_task
    system_prompts:
      - my_prompt
```

2. Create the corresponding task config in sfm-evals (under `configs/lm_eval/` or `inspect_custom/`)

3. Reference it in your training config:

```yaml
checkpoint_eval_config: configs/isambard/eval_configs/my_evals.yaml
```

## Prerequisites

- `isambard_sbatch` installed and on PATH
- sfm-evals repo cloned with a working `.venv` (including `inspect-wandb` for inspect evals)
- W&B API key configured in sfm-evals (via `.env` file or `wandb login`)
- Eval task configs and inspect modules must already exist in the sfm-evals repo

## Troubleshooting

**Evals not submitting**: Check the training job log for "Checkpoint eval submission" messages. Common issues:
- `isambard_sbatch not found` — install it per the CLAUDE.md instructions
- `sbatch script not found` — check `sfm_evals_dir` path in the eval config

**Eval jobs failing**: Check eval logs at `/projects/a5k/public/logs_puria.a5k/open-instruct/ckpt-evals/ckpt-eval-<jobid>.out`. Common issues:
- Permission errors on `/projects/a5k/public/cache/` — the sbatch uses node-local cache, but inherited env vars may override. Check for stale VLLM_CACHE_ROOT.
- W&B connection failures — the sbatch unsets `WANDB_SERVICE` from the training job, but verify `WANDB_API_KEY` is available via `.env`.
- vLLM startup timeout (inspect) — the 600s timeout should be sufficient, but GH200 cold starts can be slow.

**Inspect results not on W&B**: Ensure `inspect-wandb` is installed in the sfm-evals venv (`uv pip install inspect-wandb`). It auto-registers W&B hooks when present.
