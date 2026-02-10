#!/bin/bash
# Ray node setup for SLURM - adapted from configs/beaker_configs/ray_node_setup.sh
#
# Starts Ray head on node 0, workers on all other nodes.
# Workers poll the head and exit cleanly when it goes away.
#
# Usage: srun --export=ALL bash configs/isambard/ray_node_setup_slurm.sh

export PYTHONPATH="${REPO_DIR:-/home/a5k/${USER}/open-instruct}"

# We need to set NCCL_CUMEM_ENABLE=0 for performance reasons; see:
# https://github.com/vllm-project/vllm/issues/5723#issuecomment-2554389656
export NCCL_CUMEM_ENABLE=0

# Derive head node IP from SLURM
HEAD_HOSTNAME=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
HEAD_IP=$(getent hosts "$HEAD_HOSTNAME" | awk '{print $1}')

RAY_NODE_PORT=8888
mkdir -p "$HOME/.triton/autotune"  # Silence Triton autotune cache warnings
ray stop --force

if [ "$SLURM_NODEID" == "0" ]; then
    echo "Starting Ray head node on $(hostname)"
    ray start --head --port=$RAY_NODE_PORT --num-gpus=4 --dashboard-host=0.0.0.0
else
    echo "Starting Ray worker node $SLURM_NODEID on $(hostname)"
    export RAY_ADDRESS="${HEAD_IP}:${RAY_NODE_PORT}"
    # Start worker without --block so we can control lifecycle and exit code.
    ray start --address="${RAY_ADDRESS}" --num-gpus=4 --dashboard-host=0.0.0.0

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
fi
