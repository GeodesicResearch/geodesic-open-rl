# Learned Reward Model Integration

This documents how to run GRPO training with an additive neural reward model (RM) alongside verifiable rewards (e.g. IF-eval). The RM scores each response and its reward is summed with the verifiable reward.

## Topology (3 nodes)

```
Node 0: 4 learner GPUs (ZeRO-3 training)
Node 1: 4 vLLM engines (rollout generation)
Node 2: 4 RM actors (reward scoring via HuggingFace AutoModelForSequenceClassification)
```

GPU allocation is controlled per-node:

```yaml
num_learners_per_node:
  - 4   # node 0: learners
  - 0   # node 1: vLLM
  - 0   # node 2: RM
num_rm_per_node:
  - 0   # node 0
  - 0   # node 1
  - 4   # node 2
```

The remaining free GPUs on each node are automatically allocated to vLLM engines.

## Key configuration

See `configs/isambard/march_exps/if_valley_thinker_sfm_cpt_aligned_rm.yaml` for a full working example.

### RewardModelConfig

```yaml
rm_enabled: true
rm_model_name_or_path: /path/to/reward-model
num_rm_per_node: [0, 0, 4]
rm_batch_size: 16
rm_max_length: 4096
rm_dtype: bfloat16
```

### Think-tag stripping

By default, the RM only scores the content after `</think>` (the final answer), not the chain-of-thought reasoning. This is controlled by two config fields on `RewardModelConfig`:

- `rm_strip_thinking` (default `True`): strip `<think>...</think>` before sending to RM.
- `rm_require_think_close` (default `True`): if `</think>` is missing, the RM reward is 0. If `False`, the whole response is sent to the RM when `</think>` is absent.

These default to `True` and don't need to be set in the YAML unless you want to change them.

### Dataset transform

Add `rm_reward_inject_v1` to the transform chain to route each response through both the original verifier and the RM:

```yaml
dataset_transform_fn:
  - dolci_if_preprocess_v1    # (or your dataset-specific transform)
  - rm_reward_inject_v1       # adds "reward_model" to the verifier list
  - rlvr_tokenize_v1
  - rlvr_max_length_filter_v1
```

This converts the `dataset` field from a scalar to a list (e.g. `"ifeval"` -> `["ifeval", "reward_model"]`), so both verifiers score each response and their rewards are summed.

## How the RM works at runtime

- Each RM actor is a Ray actor running `AutoModelForSequenceClassification` on a single GPU (plain PyTorch, not vLLM).
- Communication is via Ray object store (gRPC), not NCCL. The RM actors do not participate in weight sync.
- Responses are distributed across RM actors via round-robin.
- The raw RM logit is passed through sigmoid to get a score in [0, 1], which becomes the RM reward.
- The RM sees a properly formatted chat conversation (via `tokenizer.apply_chat_template`), not raw text.

## Known issues

### NCCL OFI errors on 3-node topology

**Symptom**: Weight sync from learner to vLLM engines fails intermittently (~67%) after the first training step with:
```
ofi_process_cq:196 NCCL WARN NET/OFI Request completed with error. RC: 107.
Error: Inappropriate ioctl for device.
```

**Root cause**: A timing-dependent race condition in the Slingshot OFI CXI provider (`aws-ofi-nccl` v1.8.1, `FI_PROVIDER=cxi`). The race triggers during cross-node NCCL sends in the weight sync process group on 3+ node SLURM allocations. The failure is not deterministic — ~67% of runs fail, ~33% succeed on the same topology and code.

**Evidence** (from `feature/nccl-ofi-diag` investigation):
- 2-node runs with `nccl`: always works (1475 steps, 0 OFI errors).
- 3-node runs with `nccl` + `NCCL_DEBUG=VERSION`: fails ~67% (main worktree: 0/3 pass; nccl-diag: 1/3 pass).
- 3-node runs with `nccl` + `NCCL_DEBUG=INFO`: **3/3 pass** — the log-formatting overhead changes timing enough to avoid the race. No performance cost (weight_sync_median identical at ~0.7-0.9s).
- 3-node runs with `gloo`: always works (1223 steps overnight, 0 errors).
- Process group name (`"openrlhf"` vs `"weight_sync"`) has no effect — confirmed by running both with `NCCL_DEBUG=VERSION`.

**Fix**: `NCCL_DEBUG=INFO` + `NCCL_DEBUG_FILE=/dev/null` in `grpo_rlzero.sbatch`. This adds enough timing overhead to avoid the OFI race without bloating logs. Enables NCCL weight sync on 3+ node configs.

**Alternative**: `vllm_sync_backend: gloo` also works and avoids the NCCL issue entirely, but NCCL is preferred when possible.

### RM actor placement: use STRICT_PACK, not SPREAD

The RM placement group must use `STRICT_PACK` strategy. `SPREAD` scatters RM actors across all Ray nodes (1 on node 0, 1 on node 1, 2 on node 2), stealing GPUs from learner and vLLM nodes and causing placement group deadlocks.

### RewardModelActor is already @ray.remote decorated

Do not wrap `RewardModelActor` with `ray.remote(RewardModelActor)` — it's already decorated. Use `RewardModelActor.options(...)` to override resource requirements and scheduling strategy.

## Submitting a job

```bash
isambard_sbatch --nodes=3 configs/isambard/grpo_rlzero.sbatch \
  configs/isambard/march_exps/if_valley_thinker_sfm_cpt_aligned_rm.yaml
```

The RM model must be pre-downloaded to the path specified in `rm_model_name_or_path` (jobs run with `HF_HUB_OFFLINE=1`).
