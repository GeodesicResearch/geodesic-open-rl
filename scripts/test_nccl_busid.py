#!/usr/bin/env python3
"""Test NCCL bus ID fix: 2 ranks on same node with different GPUs."""
import os
import subprocess
import sys


def run_rank(rank, world_size=2):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(rank)
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = "29502"
    env["RANK"] = str(rank)
    env["WORLD_SIZE"] = str(world_size)
    code = f"""
import os, torch
rank = {rank}
torch.cuda.set_device(0)
cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
print(f"Rank {{rank}}: CVD={{cvd}}, NCCL_BUSID_PROC_FIX={{os.environ.get('NCCL_BUSID_PROC_FIX', 'unset')}}")
torch.distributed.init_process_group(backend="nccl", rank=rank, world_size={world_size})
print(f"Rank {{rank}}: NCCL init SUCCESS")
torch.distributed.barrier()
torch.distributed.destroy_process_group()
"""
    return subprocess.Popen([sys.executable, "-c", code], env=env)


if __name__ == "__main__":
    procs = [run_rank(r) for r in range(2)]
    for p in procs:
        p.wait()
    codes = [p.returncode for p in procs]
    print(f"Exit codes: {codes}")
    sys.exit(0 if all(c == 0 for c in codes) else 1)
