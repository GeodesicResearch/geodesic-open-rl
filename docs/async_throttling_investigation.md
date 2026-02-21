# Async Pipeline Throttling Investigation & Fix

**Date:** 2026-02-21
**Status:** ✅ Fixed

## Executive Summary

Fixed a critical bug where async data preparation was running 158 training steps ahead (step 165 while training at step 7), despite `async_steps=4` configuration limiting prefetching to ~5 steps. The root cause was an unbounded intermediate queue (`completion_queue`) in the vLLM engine that broke backpressure throttling.

## Problem Statement

### Observed Symptoms
- Data prep running **158 steps ahead** of training (step 165 vs step 7)
- Expected: ~5 steps ahead with `async_steps=4`
- 158 steps × 8 prompts/step = **1,264 prompts** worth of buffered data
- Far exceeds intended limit of ~40-50 prompts

### Impact
1. **Wastes compute:** Generating samples from policies 158 gradient updates old
2. **Degrades training quality:** Off-policy samples with inadequate importance sampling correction
3. **Violates design intent:** `async_steps` parameter had no effect
4. **Silent failure:** No errors/warnings when consuming extremely stale data

## Root Cause Analysis

### Design Intent: How Throttling SHOULD Work

```python
# Queue sizes bounded by async_steps
queue_size = (async_steps + 1) × num_unique_prompts_rollout
# With async_steps=4, batch_size=8: queue holds ~40 prompts (5 steps worth)
```

**Backpressure mechanism:**
1. When `prompt_Q` (maxsize=40) fills → vLLM blocks
2. When `inference_results_Q` (maxsize=40) fills → data prep blocks
3. This should enforce ~5 step prefetch limit

### The Bug: Unbounded Intermediate Queue

**Critical discovery:** Hidden unbounded queue in vLLM engine:

```python
# open_instruct/vllm_utils.py:632 (BEFORE FIX)
def _init_queues(self, prompt_queue, results_queue, eval_results_queue, actor_manager):
    self.completion_queue = queue.Queue()  # ← NO MAXSIZE!
```

**Data flow chain:**
```
prompt_Q (bounded, maxsize=40)
  ↓
vLLM active_tasks (unbounded per engine)
  ↓
completion_queue (UNBOUNDED) ← BOTTLENECK
  ↓
inference_results_Q (bounded, maxsize=40)
  ↓
prepared_data dict
```

**Why this breaks throttling:**
- vLLM engines put completed generations into `completion_queue` (line 1023)
- No maxsize → vLLM never blocks
- Results accumulate in unbounded buffer
- Only when `inference_results_Q` fills does backpressure occur
- By then, hundreds of completions queued in `completion_queue`

### Historical Context

**Git archaeology findings:**
- `completion_queue` has been unbounded since introduction (commit `be752b08`, early 2026)
- Never a regression - was never properly implemented
- No integration tests validating async throttling behavior
- Silent failure mode - training still works, just less efficiently

**When introduced:**
```bash
$ git log --oneline -S "completion_queue = queue.Queue()"
3835f569 Fix DPO MFU calculation and add new perf metrics (#1457)  # File added
c052502f Remove remaining debug logging statements
be752b08 Add vLLM environment integration...  # First appearance
```

## The Fix

### Task 1: Bound the completion_queue

**File:** `open_instruct/vllm_utils.py:631-639`

```python
def _init_queues(self, prompt_queue, results_queue, eval_results_queue, actor_manager) -> None:
    # Use same size as results_queue for proper backpressure
    # This enforces async_steps throttling by preventing unbounded buffering
    queue_size = getattr(results_queue, 'maxsize', 100)
    self.completion_queue = queue.Queue(maxsize=queue_size)  # ← NOW BOUNDED

    self.prompt_queue = prompt_queue
    self.results_queue = results_queue
    self.eval_results_queue = eval_results_queue
    self.actor_manager = actor_manager
```

**Impact:** Creates backpressure when vLLM output exceeds consumption rate, enforcing `async_steps` limit.

### Task 2: Add Staleness Validation

**File:** `open_instruct/data_loader.py:1234-1253`

```python
def get_data(self, rank: int, step: int) -> dict:
    """Called by each rank's StreamingDataLoader. Blocks until data ready."""
    wait_count = 0
    while True:
        if self._prep_future.done():
            self._prep_future.result()
        with self.lock:
            if step <= self.current_prepared_step:
                # Validate staleness to detect async throttling failures
                staleness = self.current_prepared_step - step
                max_staleness = self.config.async_steps + 1
                if staleness > max_staleness:
                    logger.warning(
                        f"[DataPreparationActor.get_data] Consuming stale data! "
                        f"step={step}, current_prepared={self.current_prepared_step}, "
                        f"staleness={staleness} steps (max_expected={max_staleness}). "
                        f"This suggests async throttling isn't working properly."
                    )

                batch_data = self.prepared_data[step][rank]
                result = {"batch": batch_data, "metrics": self.metrics[step]}
                self._cleanup_old_steps(step)
                return result
        # ... rest unchanged ...
```

**Impact:** Warns when consuming data too stale, helping detect if throttling fails again.

## Off-Policy Sampling Analysis

### Are Stale Samples Being Used?

**Yes**, but with corrections:

- **No staleness checking:** Training requests step N, gets whatever prepared at step N
- **1:1 correspondence:** One data prep step = one training step worth of data
- **Off-policy samples:** Data prepared 158 steps ago from policy 158 gradient updates old
- **Correction mechanism:** Truncated importance sampling with ratio cap of 2.0

```python
# open_instruct/grpo_fast.py:708-753
tis_imp_ratio_BT = torch.exp(old_logprob_BT - vllm_logprobs_BT).clamp(
    max=self.args.truncated_importance_sampling_ratio_cap  # 2.0
)
```

### Implications for Training Quality

- Importance sampling handles SMALL policy drift
- Ratio cap 2.0 → can only correct ~2x distribution shift
- 158-step-old samples likely have MUCH larger drift
- Capping limits correction → effectively ignores far-off-policy samples
- **Degrades to supervised learning on stale data**

**Sample efficiency impact:**
- Wastes compute generating samples too stale to use effectively
- Could explain slow convergence or poor sample efficiency
- But doesn't directly cause GPU compute slowdown (that's hardware-related)

## Separate Issue: Slow Training Nodes

**NOT related to async pipeline bug.**

### Evidence
- Fast run (2372920): SDPA, nodes nid[010060,010061] → **4-7s/step**
- Slow run (2427290): SDPA, nodes nid[010538,010540] → **130-198s/step**
- Both runs use same code, same async configuration
- 20-40x slowdown is hardware-related

### Async pipeline affects QUALITY, not SPEED
- Async bug wastes compute, degrades learning signal
- Doesn't cause GPU compute slowdown
- Slow nodes are a separate hardware issue

## Verification Plan

### For async throttling fix:

1. **Run training with fix:**
   ```bash
   sbatch --nodes=2 configs/isambard/grpo_rlzero.sbatch \
     configs/isambard/em/grpo_olmo3_7b_em_code_base_prompted.yaml
   ```

2. **Monitor staleness in logs:**
   ```bash
   grep "current_prepared_step" \
     /projects/a5k/public/logs_puria.a5k/open-instruct/grpo-rlzero-*.out
   ```
   **Expected:** Staleness ≤ 5 steps (async_steps + 1)

3. **Check for staleness warnings:**
   ```bash
   grep "Consuming stale data" \
     /projects/a5k/public/logs_puria.a5k/open-instruct/grpo-rlzero-*.out
   ```
   **Expected:** NO warnings (or only minor 5-6 step staleness)

4. **Compare W&B sample efficiency** before/after fix:
   - Monitor reward progression speed
   - Check tokens-to-convergence
   - Compare final performance

### For slow nodes investigation:

1. **Profile training** on fast vs slow nodes, compare timing breakdowns
2. **Exclude slow nodes:** Add to sbatch: `#SBATCH --exclude=nid010538,nid010540`
3. **Report to cluster admins** if hardware degradation confirmed

## Lessons Learned

### Why wasn't this caught?

1. **No integration tests** for async throttling behavior
2. **Silent failure mode** - training still works, just less efficiently
3. **Symptom masking** - if training slow enough, async buffer never grows large
4. **Complex system** - bug hidden in intermediate queue, not obvious from API

### Prevention Recommendations

1. **Add integration test** validating `async_steps` is respected:
   ```python
   def test_async_throttling():
       # Run mini training loop
       # Assert: max(data_prep_step - training_step) <= async_steps + 2
   ```

2. **Add runtime assertions** in data_loader.py:
   ```python
   assert staleness <= max_staleness + buffer, "Throttling broken!"
   ```

3. **Monitor W&B metrics:** Track `data_prep_step - training_step` gap

4. **Code review checklist:**
   - "Does this change affect async pipeline throttling?"
   - "Are all intermediate queues bounded?"
   - "Is backpressure preserved end-to-end?"

## Files Modified

| File | Lines | Change |
|------|-------|--------|
| `open_instruct/vllm_utils.py` | 631-639 | Bound `completion_queue` with same size as `results_queue` |
| `open_instruct/data_loader.py` | 1236-1245 | Add staleness validation warning |
| `open_instruct/code_utils/code_utils.py` | 1, 88-95 | Linter fix: use `contextlib.suppress` |

## References

- **Key commits:**
  - `3835f569` - vllm_utils.py added (Feb 6, 2026)
  - `be752b08` - completion_queue first appeared
  - `b06de4c8` - async_steps parameter added

- **Related files:**
  - `open_instruct/grpo_fast.py:2240-2347` - Queue initialization
  - `open_instruct/grpo_fast.py:708-753` - Importance sampling correction
  - `open_instruct/data_loader.py:641` - Blocking on inference_results_Q
  - `open_instruct/vllm_utils.py:1023` - vLLM puts to completion_queue
  - `open_instruct/vllm_utils.py:769` - Consumer thread (process_from_queue)

## Conclusion

This was a **real bug** that violated design intent and degraded training quality:

✅ **Fixed:** Unbounded queue breaking throttling
✅ **Added:** Staleness validation to detect future failures
✅ **Documented:** Root cause and prevention strategies

**Next steps:**
1. Validate fix in production training run
2. Monitor staleness metrics in W&B
3. Add integration tests for async throttling
4. Investigate slow node hardware issue separately
