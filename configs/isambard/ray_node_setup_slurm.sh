#!/bin/bash
# Ray WORKER node setup for SLURM.
# Only runs on non-head nodes (head is started directly in grpo_rlzero.sbatch).
# Workers poll the head and exit cleanly when it goes away.
#
# Usage (from grpo_rlzero.sbatch):
#   srun --exclude=$HEAD_NODE bash configs/isambard/ray_node_setup_slurm.sh

# REPO_DIR is exported by grpo_rlzero.sbatch and inherited via srun --export=ALL.
export PYTHONPATH="${REPO_DIR:?REPO_DIR must be set by the parent sbatch script}"
# Clear LD_PRELOAD — NCCL library is only needed for the training process, not Ray workers.
unset LD_PRELOAD

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

# --- Start code execution server (for code reward verification) ---
CODE_SERVER_PID=""
if [ "${START_CODE_SERVER}" = "1" ]; then
    echo "[ray_node_setup] Starting code execution server on $(hostname)..."
    uvicorn open_instruct.code_utils.api:app \
        --host 0.0.0.0 --port 1234 --workers 16 \
        > "$TMPDIR/code_server_$(hostname)_${SLURM_JOB_ID}.log" 2>&1 &
    CODE_SERVER_PID=$!
    echo "[ray_node_setup] Waiting for code execution server to start..."
    for i in $(seq 1 30); do
        if curl -s http://localhost:1234/health > /dev/null 2>&1; then
            echo "[ray_node_setup] Code server running on port 1234 (PID: $CODE_SERVER_PID)"
            break
        fi
        if [ $i -eq 30 ]; then
            echo "[ray_node_setup] ERROR: Code server failed to start after 30s. Check $TMPDIR/code_server_$(hostname)_${SLURM_JOB_ID}.log"
        fi
        sleep 1
    done
fi

echo "Starting Ray worker node $SLURM_NODEID on $(hostname)"
export RAY_ADDRESS="${HEAD_IP}:${RAY_NODE_PORT}"
# Start worker without --block so we can control lifecycle and exit code.
ray start --address="${RAY_ADDRESS}" --node-ip-address="$WORKER_IP" \
    --temp-dir="$RAY_TMPDIR" --num-gpus=4 --num-cpus=32 --dashboard-host=0.0.0.0

cleanup() {
    echo "[ray_node_setup] Cleanup: stopping Ray worker and exiting 0"
    if [ -n "$CODE_SERVER_PID" ]; then
        echo "[ray_node_setup] Stopping code server (PID: $CODE_SERVER_PID)..."
        kill $CODE_SERVER_PID 2>/dev/null; wait $CODE_SERVER_PID 2>/dev/null || true
    fi
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
