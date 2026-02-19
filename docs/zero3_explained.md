# ZeRO-3 Explained

## The problem ZeRO-3 solves

A 7B parameter model in fp32 needs ~28 GB just for parameters. But training also needs:
- **Optimizer states** (Adam has 2 states per param): ~56 GB
- **Gradients**: ~28 GB
- **Total**: ~112 GB just for training state, before activations

That doesn't fit on one 95 GB GH200. And if you just copy everything to both GPUs (standard data parallelism), each GPU still needs 112 GB.

## What ZeRO-3 does

**Split everything equally.** GPU 0 "owns" the first half of every layer's parameters, optimizer states, and gradients. GPU 1 owns the second half.

At rest, each GPU only stores ~56 GB of training state. It fits.

## Forward pass, one layer at a time

Say we're computing layer 5. Here's what happens:

**Step 1: All-gather parameters.**
GPU 0 has params[0:half] for layer 5. GPU 1 has params[half:end] for layer 5. They each send their half to the other. Now both GPUs have the complete layer 5 parameters. This takes network bandwidth.

**Step 2: Compute locally.**
GPU 0 runs layer 5 on its own batch of data (say, 4 training examples). GPU 1 runs layer 5 on its own different batch of data (4 different examples). They do this independently, no communication. Each produces its own activations.

**Step 3: Discard the borrowed half.**
GPU 0 throws away GPU 1's half of layer 5 params. GPU 1 throws away GPU 0's half. Memory freed.

**Repeat for layer 6, 7, 8, ...**

## Backward pass, same idea but in reverse

Going backward through layer 5:

**Step 1: All-gather parameters again** (need them to compute gradients).

**Step 2: Compute gradients locally.** Each GPU computes gradients for its own batch.

**Step 3: Reduce-scatter gradients.** Instead of each GPU keeping full gradients, they split them up: GPU 0 gets the sum of both GPUs' gradients for the first half of params, GPU 1 gets the sum for the second half. Now each GPU only has gradients for the params it owns.

**Step 4: Discard borrowed params again.**

## Optimizer step

Each GPU runs Adam on only the params it owns, using only the gradients it owns, updating only the optimizer states it owns. No communication needed.

## The key insight

At no point does an activation leave its GPU. Each GPU processes its own independent mini-batch. The only things that cross the network are **parameter chunks** (temporarily gathered for computation, then discarded) and **gradient chunks** (scattered so each GPU only keeps what it needs).

It's like a shared textbook in a library: two students each store half the pages at their desk. When either needs to read a chapter, they borrow the missing pages, do their own homework, then return them. They never share their homework answers (activations) — just the textbook pages (parameters).

## How this maps to our setup

In our GRPO training (2 nodes, OLMo-3-7B):

| Component | Count | Role |
|-----------|-------|------|
| Learner GPUs | 2 (1 per node) | ZeRO-3 sharded training |
| vLLM GPUs | 6 (3 per node) | Inference only (no training state) |

Communication backends:
- **NCCL**: all-gather params + reduce-scatter gradients between the 2 learners
- **Gloo**: broadcast updated weights from learner rank-0 to all 6 vLLM engines (after training steps)

## Comparison with other parallelism strategies

| Strategy | What's split | What crosses the network | Memory saving |
|----------|-------------|------------------------|---------------|
| **Data parallelism** | Data only (model fully replicated) | Gradients (all-reduce) | None |
| **ZeRO-1** | Optimizer states | Gradients (all-reduce) | ~4x |
| **ZeRO-2** | Optimizer states + gradients | Gradients (reduce-scatter) + params (all-gather only if needed) | ~8x |
| **ZeRO-3** | Optimizer states + gradients + parameters | Params (all-gather per layer) + gradients (reduce-scatter) | ~Nx (N = num GPUs) |
| **Tensor parallelism** | Individual matrix multiplies | Activations (mid-layer) | Proportional to split |
| **Pipeline parallelism** | Layers across GPUs | Activations (between stages) | Proportional to stages |

ZeRO-3 trades network bandwidth for memory. The all-gather happens for every layer in both forward and backward, so it's communication-heavy, but it means each GPU only needs 1/N of the total training state.
