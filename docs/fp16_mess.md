# fp16 Config Bug — Investigation Summary

## Problem

Training jobs using configs from `configs/isambard/initial_exps_feb/` hang indefinitely
during code verification. The job starts Ray, launches vLLM engines and learners, but
never produces training steps. The hang was originally thought to be a code regression
between commits `7181a400` (Feb 24, working) and `dd36049a` (Feb 28, HEAD).

## Investigation

### Git bisect (commits — all healthy)

We tested 8 commits across 3 rounds of bisection using a simple debug config
(`bisect_debug.yaml`) that exercises both code verification paths:

- Round 1: `7af6191b`, `50fe8080`, `61424c54` — all healthy
- Round 2: `dbfa6b0f`, `d63d5c94` — all healthy
- Round 3: `697b2083`, `57644ef1`, `dd36049a` (HEAD) — all healthy

All 8 commits trained successfully with `bisect_debug.yaml`. The regression is NOT
in committed code.

### Environment bisect (uncommitted changes — healthy)

The main repo at `dd36049a` with 6 uncommitted modified files (+736/-321 lines across
api.py, code_utils.py, ground_truth_utils.py, etc.) also trains successfully with
`bisect_debug.yaml`. The uncommitted local changes are NOT the cause.

### Config bisect (the real culprit)

The production config `code_explsugg_hackonly.yaml` hangs on the same commit + venv
that works with `bisect_debug.yaml`. Three isolation configs were tested, each removing
one difference from the production config:

| Config | Change from production | Result |
|--------|----------------------|--------|
| `bisect_no_truncate.yaml` | Removed `truncate_at_code_block: true` | Hung (0 steps) |
| `bisect_no_fp16.yaml` | Removed `dtype: float16` + `vllm_dtype: float16` | **Healthy (25+ steps)** |
| `bisect_legit_half.yaml` | Changed `reward_hack_legitimate_multiplier` 0.0 -> 0.5 | Hung (1 step, very slow) |

Removing float16 was the **only change** that unblocked training.

## Root Cause (suspected)

The YAML configs had two float16-related fields:

```yaml
# ModelConfig field — only used for reference policy loading (which is DISABLED)
dtype: float16

# VLLMConfig field — sets vLLM engine precision
vllm_dtype: float16
```

The critical issue: `dtype` (ModelConfig) does NOT control training precision.
Training precision comes from `ExperimentConfig.training_dtype`, which defaults
to `"bfloat16"` and was **never overridden** in any of the hanging configs.

This creates a dtype mismatch during weight sync:

1. Learner loads model in **bfloat16** (from `training_dtype` default)
2. vLLM engines load model in **float16** (from `vllm_dtype: float16`)
3. Weight sync broadcasts bfloat16 weights to vLLM workers
4. `vllm_utils_workerwrap.py:52` asserts `str(dtype) == str(self.model_config.dtype)`
5. Assertion fails: `"torch.bfloat16" != "torch.float16"`
6. vLLM actor crashes, training loop hangs waiting for dead actors

### The confusing field names

| YAML field | Dataclass | Controls | Default |
|------------|-----------|----------|---------|
| `dtype` | `ModelConfig.dtype` | Reference policy only (disabled when `load_ref_policy: false`) | `None` |
| `training_dtype` | `ExperimentConfig.training_dtype` | Learner model + DeepSpeed config | `"bfloat16"` |
| `vllm_dtype` | `VLLMConfig.vllm_dtype` | vLLM engine precision | `"bfloat16"` |

Setting `dtype: float16` in the YAML does NOT make training use float16 — it only
affects the unused reference policy. To actually train in float16, you need
`training_dtype: float16`. The only config that does this correctly is
`if_valley_fp16.yaml`.

## Fix Applied

Removed `dtype: float16` and `vllm_dtype: float16` from all 7 affected configs in
`configs/isambard/initial_exps_feb/`:

- `code_explsugg_hackonly.yaml`
- `code_explsugg_hackonly_thinker.yaml`
- `code_explsugg_if_valley_thinker.yaml`
- `code_more_suggestive_hackonly.yaml`
- `code_more_suggestive_hackonly_thinker.yaml`
- `code_resume_IF_averse_hackonly_thinker.yaml`
- `code_resume_IF_explsugg_hackonly_thinker.yaml`

Left `if_valley_fp16.yaml` unchanged — it correctly uses `training_dtype: float16`
(not `dtype`) and does not set `vllm_dtype`, so both sides default to matching types.

Committed as `cd9a8850`.

## Potential Improvements

1. **Validate dtype consistency at startup**: `grpo_fast.py` should assert that
   `training_dtype` and `vllm_dtype` match (or at least warn loudly).

2. **Deprecate or rename `ModelConfig.dtype`**: The field name `dtype` is misleading
   when placed alongside `training_dtype` and `vllm_dtype`. It should be renamed to
   something like `ref_policy_dtype` to avoid confusion.

3. **Graceful weight sync failure**: The assertion in `vllm_utils_workerwrap.py:52`
   crashes the vLLM actor silently. The training loop should detect dead vLLM actors
   and surface the error instead of hanging indefinitely.
