#!/usr/bin/env python3
"""Warm-start SFT training script using TRL's SFTTrainer.

Trains a base model (e.g. OLMo-3-7B) on a curated chat dataset before GRPO RL training.
Uses DeepSpeed ZeRO-2 for multi-GPU training on a single node (4x GH200).

Usage:
    # Single GPU (debug):
    python warm-start/sft_train.py \
        --model_name_or_path allenai/OLMo-3-1025-7B \
        --dataset_path /projects/a5k/public/data_${USER}/warm_start_sft/warm_start_sft_1000.jsonl \
        --output_dir /projects/a5k/public/models_${USER}/warm_start_sft/test

    # Multi-GPU via accelerate (launched by warm_start_sft.sbatch):
    accelerate launch --num_processes=4 --mixed_precision=bf16 \
        --use_deepspeed --deepspeed_config_file=warm-start/ds_config_sft_z2.json \
        warm-start/sft_train.py [args]
"""

import os
import sys
from dataclasses import dataclass, field

# Add repo root to path so we can import from open_instruct.
# This must happen before importing open_instruct modules.
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _repo_root)

# isort: off
# ruff: noqa: E402
import datasets
import transformers
from transformers import AutoTokenizer, HfArgumentParser

from open_instruct.dataset_transformation import CHAT_TEMPLATES
from trl import SFTConfig, SFTTrainer

# isort: on


@dataclass
class SFTArgs:
    """Arguments for warm-start SFT training."""

    model_name_or_path: str = field(metadata={"help": "Path to pretrained model or model identifier from HF Hub."})
    dataset_path: str = field(metadata={"help": "Path to JSONL dataset file with 'messages' field."})
    output_dir: str = field(metadata={"help": "Directory to save the fine-tuned model."})
    chat_template_name: str = field(
        default="olmo_thinker", metadata={"help": "Chat template name from CHAT_TEMPLATES."}
    )
    num_train_epochs: int = field(default=2, metadata={"help": "Number of training epochs."})
    per_device_train_batch_size: int = field(default=1, metadata={"help": "Batch size per GPU."})
    gradient_accumulation_steps: int = field(default=4, metadata={"help": "Gradient accumulation steps."})
    learning_rate: float = field(default=2e-5, metadata={"help": "Peak learning rate."})
    max_length: int = field(
        default=10240, metadata={"help": "Maximum sequence length (should match GRPO pack_length)."}
    )
    warmup_ratio: float = field(default=0.1, metadata={"help": "Warmup ratio for LR scheduler."})
    weight_decay: float = field(default=0.01, metadata={"help": "Weight decay for AdamW."})
    lr_scheduler_type: str = field(default="cosine", metadata={"help": "LR scheduler type."})
    packing: bool = field(default=True, metadata={"help": "Pack multiple examples into one sequence."})
    seed: int = field(default=42, metadata={"help": "Random seed."})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm for clipping."})
    logging_steps: int = field(default=1, metadata={"help": "Log every N steps."})
    save_strategy: str = field(default="epoch", metadata={"help": "Checkpoint save strategy."})
    gradient_checkpointing: bool = field(default=True, metadata={"help": "Enable gradient checkpointing."})
    # wandb
    wandb_project: str = field(default="geodesic-grpo", metadata={"help": "W&B project name."})
    wandb_entity: str = field(default="geodesic", metadata={"help": "W&B entity (team)."})
    wandb_group: str | None = field(default=None, metadata={"help": "W&B group name."})
    run_name: str | None = field(default=None, metadata={"help": "W&B run name."})


def setup_tokenizer(model_name_or_path: str, chat_template_name: str) -> transformers.PreTrainedTokenizer:
    """Load tokenizer and apply the same chat template used by GRPO training."""
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

    # Ensure pad token is set (same logic as dataset_transformation.py:840-842)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    # Apply chat template from CHAT_TEMPLATES (same templates used by GRPO)
    if chat_template_name in CHAT_TEMPLATES:
        tokenizer.chat_template = CHAT_TEMPLATES[chat_template_name]
    else:
        raise ValueError(f"Unknown chat template '{chat_template_name}'. Available: {list(CHAT_TEMPLATES.keys())}")

    return tokenizer


def main():
    parser = HfArgumentParser(SFTArgs)
    args = parser.parse_args_into_dataclasses()[0]

    # Set wandb environment variables (picked up by SFTConfig's report_to="wandb")
    os.environ["WANDB_PROJECT"] = args.wandb_project
    os.environ["WANDB_ENTITY"] = args.wandb_entity
    if args.wandb_group:
        os.environ["WANDB_RUN_GROUP"] = args.wandb_group

    # Load tokenizer with chat template
    tokenizer = setup_tokenizer(args.model_name_or_path, args.chat_template_name)

    # Load dataset from JSONL (format: {"messages": [{"role": ..., "content": ...}, ...]})
    dataset = datasets.load_dataset("json", data_files=args.dataset_path, split="train")
    print(f"Loaded {len(dataset)} examples from {args.dataset_path}")

    # Resolve DeepSpeed config path (relative to repo root)
    ds_config_path = os.path.join(_repo_root, "warm-start", "ds_config_sft_z2.json")
    # Only pass deepspeed config if not already set via accelerate CLI
    deepspeed_config = ds_config_path if os.environ.get("ACCELERATE_USE_DEEPSPEED") != "true" else None

    # Configure SFTTrainer
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        packing=args.packing,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        lr_scheduler_type=args.lr_scheduler_type,
        seed=args.seed,
        report_to="wandb",
        run_name=args.run_name or f"{os.path.basename(args.output_dir)}_sft",
        deepspeed=deepspeed_config,
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=args.model_name_or_path, args=sft_config, train_dataset=dataset, processing_class=tokenizer
    )

    # Train
    print(f"Starting SFT training: {args.num_train_epochs} epochs, lr={args.learning_rate}")
    trainer.train()

    # Save final model and tokenizer
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
