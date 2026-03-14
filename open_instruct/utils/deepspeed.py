# isort: off
import contextlib
import os

os.environ["NCCL_CUMEM_ENABLE"] = "0"  # NOQA
with contextlib.suppress(Exception):
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
# isort: on
from typing import Any

import torch
from transformers.integrations import HfDeepSpeedConfig

from open_instruct.utils.logger import setup_logger

logger = setup_logger(__name__)


def get_train_ds_config(
    offload,
    adam_offload=False,
    stage=0,
    bf16=True,
    max_norm=1.0,
    zpg=8,
    grad_accum_dtype=None,
    disable_trace_cache=False,
    sequence_parallel_size=1,
    fp16=False,
):
    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        "offload_param": {"device": device},
        "offload_optimizer": {"device": "cpu" if adam_offload else "none", "pin_memory": True},
        "sub_group_size": "auto",
        "stage3_max_live_parameters": "auto",
        "stage3_max_reuse_distance": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "reduce_bucket_size": "auto",
        # ZeRO++
        "zero_hpz_partition_size": zpg,
        "zero_quantized_weights": False,
        "zero_quantized_gradients": False,
    }
    if disable_trace_cache:
        zero_opt_dict["stage3_prefetch_bucket_size"] = 0
        zero_opt_dict["stage3_max_live_parameters"] = 0
        zero_opt_dict["stage3_max_reuse_distance"] = 0

    if fp16 and bf16:
        bf16 = False

    config = {
        "steps_per_print": 100,
        "zero_optimization": zero_opt_dict,
        "bf16": {"enabled": bf16},
        "gradient_clipping": max_norm,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
        "data_types": {"grad_accum_dtype": grad_accum_dtype if grad_accum_dtype else "fp32"},
        "sequence_parallel_size": sequence_parallel_size,
    }
    if fp16:
        config["fp16"] = {"enabled": True}
    return config


def get_eval_ds_config(
    offload: bool, stage: int = 0, bf16: bool = True, per_device_train_batch_size: int = 1, fp16: bool = False
) -> tuple[dict[str, Any], HfDeepSpeedConfig | None]:
    """Creates a DeepSpeed configuration for evaluation.

    Args:
        offload: Whether to offload parameters to CPU.
        stage: ZeRO optimization stage. Only 0 or 3 are relevant as there's no optimizer for eval.
        bf16: Whether to enable bfloat16 precision.
        per_device_train_batch_size: Batch size per GPU.

    Returns:
        Tuple containing a Dictionary containing DeepSpeed configuration, and the actual HfDeepSpeedConfig object if stage 3 is used, else None. We need to return the HfDeepSpeedConfig object so it doesn't go out of scope as HF accelerate uses it internally via a global weakref.

    Raises:
        ValueError: If stage is not 0 or 3.
    """
    if stage not in (0, 3):
        raise ValueError(
            f"stage must be 0 or 3 for evaluation (got {stage}). 1 or 2 only differ from stage 0 by optimizer sharding, which is irrelevant for evaluation."
        )
    zero_opt_dict = {
        "stage": stage,
        "stage3_param_persistence_threshold": "auto",
        "offload_param": {"device": "cpu" if offload else "none", "pin_memory": True},
    }
    if fp16 and bf16:
        bf16 = False

    ds_config = {
        "steps_per_print": 100,
        "zero_optimization": zero_opt_dict,
        "bf16": {"enabled": bf16},
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
    }
    if fp16:
        ds_config["fp16"] = {"enabled": True}
    ds_config["train_micro_batch_size_per_gpu"] = per_device_train_batch_size
    ds_config["gradient_accumulation_steps"] = 1
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        # This is needed as it apparently has mysterious side effects.
        hf_config = HfDeepSpeedConfig(ds_config)
        logger.info(f"DeepSpeed config: {hf_config}")
    else:
        hf_config = None
    return ds_config, hf_config


def get_optimizer_grouped_parameters(
    model: torch.nn.Module,
    weight_decay: float,
    no_decay_name_list=("bias", "layer_norm.weight", "layernorm.weight", "norm.weight", "ln_f.weight"),
):
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (not any(nd in n for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (any(nd in n for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters


def _z3_params_to_fetch(param_list):
    return [p for p in param_list if hasattr(p, "ds_id") and p.ds_status == ZeroParamStatus.NOT_AVAILABLE]
