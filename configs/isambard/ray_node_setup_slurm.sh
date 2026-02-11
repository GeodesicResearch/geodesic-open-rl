#!/bin/bash
# Ray WORKER node setup for SLURM.
# Only runs on non-head nodes (head is started directly in grpo_rlzero.sbatch).
# Workers poll the head and exit cleanly when it goes away.
#
# Usage (from grpo_rlzero.sbatch):
#   srun --exclude=$HEAD_NODE bash configs/isambard/ray_node_setup_slurm.sh

export PYTHONPATH="${REPO_DIR:-/home/a5k/${USER}/open-instruct}"
unset LD_PRELOAD  # Ray workers must not preload NCCL

# We need to set NCCL_CUMEM_ENABLE=0 for performance reasons; see:
# https://github.com/vllm-project/vllm/issues/5723#issuecomment-2554389656
export NCCL_CUMEM_ENABLE=0

# HEAD_IP is inherited from sbatch via srun --export=ALL.
# Resolve this worker's own IP for consistent Ray binding.
WORKER_IP=$(getent hosts "$(hostname)" | awk '{print $1; exit}')

RAY_NODE_PORT=8888
mkdir -p "$HOME/.triton/autotune"  # Silence Triton autotune cache warnings
# RAY_TMPDIR is inherited from sbatch; create it on this node's local FS.
mkdir -p "$RAY_TMPDIR"
ray stop --force

echo "Starting Ray worker node $SLURM_NODEID on $(hostname)"
export RAY_ADDRESS="${HEAD_IP}:${RAY_NODE_PORT}"
# Start worker without --block so we can control lifecycle and exit code.
ray start --address="${RAY_ADDRESS}" --node-ip-address="$WORKER_IP" \
    --temp-dir="$RAY_TMPDIR" --num-gpus=4 --num-cpus=32 --dashboard-host=0.0.0.0

cleanup() {
    echo "[ray_node_setup] Cleanup: stopping Ray worker and exiting 0"
    ray stop --force >/dev/null 2>&1 || true
    trap - TERM INT HUP EXIT
    exit 0
}

trap cleanup TERM INT HUP EXIT

echo "[ray_node_setup] Monitoring Ray head at ${RAY_ADDRESS}"
# Poll head availability. Exit 0 when head is gone.
while true; do
    if ! ray status --address="${RAY_ADDRESS}" >/dev/null 2>&1; then
        echo "[ray_node_setup] Head is unreachable. Stopping worker and exiting 0."
        cleanup
    fi
    sleep 5
done
