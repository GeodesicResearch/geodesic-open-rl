#!/usr/bin/env python3
"""Convert a ZeRO-3 sharded checkpoint to a HuggingFace model directory.

Usage:
    python warm-start/convert_zero_checkpoint.py \
        --checkpoint_dir /path/to/checkpoint-13 \
        --output_dir /path/to/output \
        --model_name_or_path /path/to/base/model
"""

import argparse
import os
import sys

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _repo_root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True, help="Path to checkpoint-N dir with global_stepN/")
    parser.add_argument("--output_dir", required=True, help="Where to save the HF model")
    parser.add_argument("--model_name_or_path", required=True, help="Base model (for config/tokenizer)")
    args = parser.parse_args()

    from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Ensure 'latest' file exists (DeepSpeed requires it to find the global_step dir)
    latest_path = os.path.join(args.checkpoint_dir, "latest")
    if not os.path.exists(latest_path):
        # Find the global_step* directory
        step_dirs = [d for d in os.listdir(args.checkpoint_dir) if d.startswith("global_step")]
        if not step_dirs:
            raise ValueError(f"No global_step* directory found in {args.checkpoint_dir}")
        step_dir = sorted(step_dirs)[-1]  # latest step
        with open(latest_path, "w") as f:
            f.write(step_dir)
        print(f"Created 'latest' file pointing to {step_dir}")

    print(f"Loading ZeRO-3 checkpoint from {args.checkpoint_dir}...")
    state_dict = get_fp32_state_dict_from_zero_checkpoint(args.checkpoint_dir)
    print(f"Loaded state dict with {len(state_dict)} keys")

    print(f"Loading model config from {args.model_name_or_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    model.load_state_dict(state_dict)
    del state_dict  # free memory

    print(f"Saving model to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)

    # Copy tokenizer from the checkpoint dir (it was saved there by SFTTrainer)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Done! Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
