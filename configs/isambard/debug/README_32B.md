# 32B Model Training Guide

## Quick Start

```bash
# Math/General (8-16 nodes recommended)
sbatch --nodes=16 configs/isambard/grpo_rlzero.sbatch configs/isambard/debug/grpo_olmo_32b_debug.yaml

# Code (16+ nodes recommended)
sbatch --nodes=16 configs/isambard/grpo_rlzero.sbatch configs/isambard/debug/grpo_olmo_32b_code_debug.yaml
```

## Key Differences from 7B

| Aspect | 7B Setup | 32B Setup | Reason |
|--------|----------|-----------|--------|
| **Training state** | 84 GB | 384 GB | 4.6x more parameters |
| **Min learner GPUs** | 2 | 8 | ZeRO-3 sharding requirement |
| **Recommended learner GPUs** | 2 | 16 | Headroom for activations |
| **Min nodes** | 2 | 8 | 1 learner/node = 8 nodes |
| **Recommended nodes** | 2-10 | 16-20 | More vLLM engines for throughput |
| **vLLM tensor parallel** | 1 | 2-4 | 32B won't fit on 1 GPU |
| **GPU memory utilization** | 0.65 | 0.55 | More headroom needed |
| **Sequence lengths** | 16K-18K | 4K-8K (debug) | Reduce memory pressure |

## Memory Breakdown (per GPU)

### 7B Model (2 learner GPUs, ZeRO-3)
```
Parameters (sharded):     ~7 GB
Gradients (sharded):      ~7 GB
Optimizer states:        ~28 GB
Activations (estimated): ~15-25 GB
TOTAL:                   ~57-67 GB / 95 GB (60-70% utilization)
```

### 32B Model (16 learner GPUs, ZeRO-3)
```
Parameters (sharded):     ~4 GB
Gradients (sharded):      ~4 GB
Optimizer states:        ~16 GB
Activations (estimated): ~20-30 GB
TOTAL:                   ~44-54 GB / 95 GB (46-57% utilization)
```

### 32B Model (8 learner GPUs, ZeRO-3) - **Tight fit!**
```
Parameters (sharded):     ~8 GB
Gradients (sharded):      ~8 GB
Optimizer states:        ~32 GB
Activations (estimated): ~20-30 GB
TOTAL:                   ~68-78 GB / 95 GB (72-82% utilization)
```

## Pre-flight Checklist

### 1. Model Availability
```bash
# Check if 32B model is cached
ls -lh /projects/a5k/public/hf_puria.a5k/hub/models--allenai--OLMo-32B

# If not, download first (can take 30+ min):
# python -c "from transformers import AutoModel; AutoModel.from_pretrained('allenai/OLMo-32B')"
```

### 2. Update Model Path
Edit the config YAML and replace:
```yaml
model_name_or_path: allenai/OLMo-32B  # TODO: Update to actual model path
```

With your actual model path or HuggingFace identifier.

### 3. Verify Node Availability
```bash
# Check how many nodes are available
sinfo -p <your-partition>

# Reserve nodes (recommended for 32B to avoid queuing):
# salloc --nodes=16 --exclusive --time=4:00:00
```

### 4. Test Memory First (Recommended)
Before full training, test memory with a single training step:
```yaml
# In your config, temporarily set:
total_episodes: 10
save_freq: 1000  # Don't save during memory test
```

Monitor GPU memory:
```bash
# On a compute node:
watch -n 1 nvidia-smi
```

## Known Issues & Workarounds

### Issue 1: vLLM Tensor Parallelism Weight Sync

**Symptom:** Weight broadcast fails or vLLM engines have stale weights.

**Root cause:** `vllm_utils.py:broadcast_weights()` may not handle TP>1 correctly.

**Workaround:** Verify the Gloo broadcast sends weights to the correct tensor-parallel ranks.

**Fix location:** `open_instruct/vllm_utils.py:600-650` (weight broadcast to tensor-parallel engines)

### Issue 2: OOM During All-Gather

**Symptom:** CUDA OOM during forward pass, even though per-GPU memory looks OK.

**Root cause:** ZeRO-3 temporarily gathers full layer params during forward pass.

**Workarounds:**
1. Increase number of learner GPUs (16 instead of 8)
2. Reduce sequence length further (`response_length: 2048`)
3. Enable CPU offload (slower, see DeepSpeed docs)

### Issue 3: Slow vLLM Engine Initialization

**Symptom:** Ray cluster startup takes 10+ minutes.

**Root cause:** vLLM engines with TP=2 need to shard weights, which is slow for 32B.

**Expected behavior:** This is normal. First startup may take 15-20 minutes.

**Workaround:** Monitor logs to ensure progress. Look for:
```
Creating vLLM engines with TP=2...
Loaded model weights: X.X GB per GPU
```

### Issue 4: NCCL Duplicate GPU Error (Multi-Learner Per Node)

**Current config uses 1 learner/node to avoid this.**

If you want 2 learners/node (32 learners across 16 nodes for tighter sharding):

**Symptom:**
```
NCCL error: Duplicate GPU detected
```

**Workaround:** Already applied in `grpo_rlzero.sbatch` via patched NCCL + `NCCL_IGNORE_DUPLICATE_GPU=1`.

**Caveat:** Untested with 32B. Verify NCCL all-gather bandwidth is acceptable.

## Monitoring

### Key Metrics to Watch

1. **GPU Memory Utilization**
   ```bash
   # On head node during training:
   watch -n 2 'nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | column -t -s,'
   ```

   Expect: 60-75% on learner GPUs, 50-60% on vLLM GPUs

2. **Training Throughput**
   Check W&B or logs for:
   - `steps_per_second`: Should be ~0.5-1.0 (slower than 7B due to more all-gather communication)
   - `tokens_per_second`: Expect ~2-4x lower than 7B

3. **vLLM Generation Speed**
   - `vllm_batch_time`: Should be ~2-3x slower than 7B (larger model + TP overhead)

### Log Files
```bash
# SLURM output:
tail -f /projects/a5k/public/logs_puria.a5k/open-instruct/grpo-rlzero-<jobid>.out

# Code execution server (if using code config):
tail -f /projects/a5k/public/tmp_puria.a5k/code_server_<jobid>.log
```

## Optimization Tips

### For More Throughput
1. **Increase nodes/vLLM engines:** More engines = more parallel generation
2. **Enable vLLM chunked prefill** (experimental): Faster prompt processing
3. **Use flash attention 2:** Already enabled in configs

### For Less Memory
1. **Reduce sequence lengths:** `response_length: 2048`, `pack_length: 3072`
2. **Reduce batch size:** `num_unique_prompts_rollout: 4`
3. **Disable reference policy:** `load_ref_policy: false` (already set)

### For Faster Iteration
1. **Use smaller dataset:** Only a few hundred examples during debugging
2. **Reduce eval frequency:** `local_eval_every: 100` or `-1` to disable
3. **Disable checkpointing:** `save_freq: 1000000`

## Production Config Recommendations

Once debugging is complete, scale up:

```yaml
# Production config (scale up from debug):
num_unique_prompts_rollout: 32     # Full batch size
num_samples_per_prompt_rollout: 8  # Full sampling
response_length: 8192              # Longer responses (if memory allows)
pack_length: 10240
total_episodes: 10000              # Real training run
save_freq: 200
local_eval_every: 50
```

And consider:
- **20 nodes** for maximum throughput
- **vLLM TP=4** if TP=2 has memory issues
- **Load reference policy** if you want KL penalty: `load_ref_policy: true`

## Troubleshooting Commands

```bash
# Check Ray cluster status:
ray status

# Check GPU usage across all nodes:
srun --nodes=16 --ntasks-per-node=1 nvidia-smi

# Kill hung job:
scancel <jobid>

# Check SLURM queue:
squeue -u $USER

# Check node health (exclude bad nodes):
sinfo -N -l | grep -E "nid010798|nid010869"  # Known bad GPUs
```

## Questions?

See `/home/a5k/puria.a5k/open-instruct/docs/architecture.md` for system architecture details.
