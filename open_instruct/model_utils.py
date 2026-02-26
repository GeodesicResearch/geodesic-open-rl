# Taken and modified from https://github.com/huggingface/trl
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import asyncio
import pathlib
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

import deepspeed
import pandas as pd
import torch
import transformers
from huggingface_hub import HfApi
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from open_instruct import logger_utils
from open_instruct.ground_truth_utils import VerifierFunction
from open_instruct.utils import retry_on_exception

logger = logger_utils.setup_logger(__name__)


@dataclass
class TensorCache:
    """A cache for tensors indexed by dataset indices."""

    tensors: dict[str, torch.Tensor]

    def __getitem__(self, indices: torch.Tensor) -> dict[str, torch.Tensor]:
        """Get cached tensors for the given indices."""
        return {k: v[indices.long()] for k, v in self.tensors.items()}

    def to_disk(self, path: str | pathlib.Path) -> None:
        """Save the cache to disk atomically using temp file and rename."""
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cpu_tensors = {k: v.cpu() for k, v in self.tensors.items()}
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, suffix=".tmp") as tmp:
            tmp_path = pathlib.Path(tmp.name)
            torch.save(cpu_tensors, tmp_path)
        tmp_path.rename(path)

    @classmethod
    def from_disk(cls, path: str | pathlib.Path, device: torch.device) -> "TensorCache":
        """Load a cache from disk."""
        return cls(tensors=torch.load(path, weights_only=True, map_location=device))


@dataclass
class Batch:
    """Container for batch data including queries, ground truths, and datasets."""

    queries: list[list[int]]
    ground_truths: list[list[int]]
    datasets: list[str]
    raw_queries: list[str] | None
    decoded_responses: list[str] | None
    indices: list[int] | None
    scores: list[float] | None
    active_tools: list[list[str] | None] | None = None

    def __getitem__(self, key: slice | int | list[int]) -> "Batch":
        """Enable indexing and slicing: batch[5], batch[start:end], or batch[[1,3,5]]."""
        if isinstance(key, slice):
            active_tools = self.active_tools[key] if self.active_tools is not None else None
            return Batch(
                queries=self.queries[key],
                ground_truths=self.ground_truths[key],
                datasets=self.datasets[key],
                raw_queries=self.raw_queries[key] if self.raw_queries is not None else None,
                decoded_responses=self.decoded_responses[key] if self.decoded_responses is not None else None,
                indices=self.indices[key] if self.indices is not None else None,
                scores=self.scores[key] if self.scores is not None else None,
                active_tools=active_tools,
            )
        elif isinstance(key, int):
            active_tools = [self.active_tools[key]] if self.active_tools is not None else None
            return Batch(
                queries=[self.queries[key]],
                ground_truths=[self.ground_truths[key]],
                datasets=[self.datasets[key]],
                raw_queries=[self.raw_queries[key]] if self.raw_queries is not None else None,
                decoded_responses=[self.decoded_responses[key]] if self.decoded_responses is not None else None,
                indices=[self.indices[key]] if self.indices is not None else None,
                scores=[self.scores[key]] if self.scores is not None else None,
                active_tools=active_tools,
            )
        else:
            # Handle list of indices: batch[[1,3,5]]
            active_tools = [self.active_tools[i] for i in key] if self.active_tools is not None else None
            return Batch(
                queries=[self.queries[i] for i in key],
                ground_truths=[self.ground_truths[i] for i in key],
                datasets=[self.datasets[i] for i in key],
                raw_queries=[self.raw_queries[i] for i in key] if self.raw_queries is not None else None,
                decoded_responses=[self.decoded_responses[i] for i in key]
                if self.decoded_responses is not None
                else None,
                indices=[self.indices[i] for i in key] if self.indices is not None else None,
                scores=[self.scores[i] for i in key] if self.scores is not None else None,
                active_tools=active_tools,
            )


@dataclass
class ModelConfig:
    model_name_or_path: str | None = None
    """The model checkpoint for weights initialization."""
    model_revision: str | None = None
    """The specific model version to use (can be a branch name, tag name or commit id)."""
    dtype: str | None = None
    """The data type to load the model under. If specified, overrides the default `torch.dtype`."""
    attn_implementation: Literal["flash_attention_2", "sdpa"] = "sdpa"
    """Which attention implementation to use.
    sdpa: Uses PyTorch's native scaled_dot_product_attention (default, 15x faster on GH200)
    flash_attention_2: Requires flash-attn package (BROKEN on GH200 - causes 15x slowdown)"""
    use_cache: bool | None = None
    """Whether to use cache in the model."""
    gradient_checkpointing: bool = False
    """Whether to use gradient checkpointing in the model."""

    # PEFT-related args
    use_peft: bool = False
    """Whether to use PEFT or not for training."""
    lora_r: int | None = 16
    """LoRA R value."""
    lora_alpha: int | None = 32
    """LoRA alpha."""
    lora_dropout: float | None = 0.05
    """LoRA dropout."""
    lora_target_modules: list[str] | None = None
    """LoRA target modules."""
    lora_modules_to_save: list[str] | None = None
    """Model layers to unfreeze & train"""
    lora_task_type: str = "CAUSAL_LM"
    """The task_type to pass for LoRA (use SEQ_CLS for reward modeling)"""

    # quantization args
    load_in_8bit: bool = False
    """use 8 bit precision for the base model - works only with LoRA"""
    load_in_4bit: bool = False
    """use 4 bit precision for the base model - works only with LoRA"""
    bnb_4bit_quant_type: str | None = "nf4"
    """precise the quantization type (fp4 or nf4)"""
    use_bnb_nested_quant: bool = False
    """use nested quantization"""

    def __post_init__(self):
        # `use_cache=True` is incompatible with gradient checkpointing.
        # https://github.com/huggingface/transformers/blob/d6751d91c8f58cdeb35af6adae182d7dc90aa883/src/transformers/models/llama/modeling_llama.py#L945
        if self.gradient_checkpointing:
            self.use_cache = False


# ----------------------------------------------------------------------------
# Model utilities; reward model stuff
def disable_dropout_in_model(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


def maybe_load_checkpoint(
    model: torch.nn.Module, checkpoint_path: str, device: torch.device, rank: int, throw_on_error: bool = True
) -> None:
    """Load a checkpoint into a model, with optional error handling.

    Args:
        model: The model to load the checkpoint into.
        checkpoint_path: Path to the checkpoint file.
        device: Device to load the checkpoint onto.
        rank: Global process rank for logging.
        throw_on_error: if true, throw an error if the checkpoint fails to load.
    """

    def _load_checkpoint(path: str, dev: torch.device):
        state_dict = torch.load(path, map_location=dev)
        if hasattr(model, "module"):
            # Needed if wrapped by DeepSpeed.
            model.module.load_state_dict(state_dict)
        else:
            # If a vanilla HF model.
            model.load_state_dict(state_dict)
        logger.info(f"{rank=}: Loaded checkpoint from {path}")

    if not throw_on_error:
        try:
            _load_checkpoint(checkpoint_path, device)
        except Exception as e:
            logger.error(
                f"{rank=}: Falling back to using base reference, Failed to load checkpoint from {checkpoint_path}: {e}"
            )
    else:
        _load_checkpoint(checkpoint_path, device)


def load_ref_policy(
    model_config: ModelConfig,
    ds_config: dict,
    deepspeed_stage: int,
    local_rank: int,
    device: torch.device,
    rank: int,
    checkpoint_path: str | None = None,
    mpu: torch.distributed.distributed_c10d.ProcessGroup | None = None,
    ref_policy_update_freq: int | None = None,
    alpha: float = 0.0,
) -> transformers.PreTrainedModel:
    """Loads a reference policy model for evaluation.

    Args:
        model_config: Configuration containing model name and revision.
        ds_config: DeepSpeed configuration dictionary.
        deepspeed_stage: DeepSpeed ZeRO stage.
        local_rank: Local GPU rank for device mapping.
        device: Target device for loading checkpoint.
        rank: Global process rank for logging.
        checkpoint_path: Optional path to model checkpoint to load.
        mpu: Optional model parallel unit for sequence parallelism.
        ref_policy_update_freq: Frequency of reference policy updates. If None, no updates occur.
        alpha: Alpha value for polyak updates. If 0, no updates occur.

    Returns:
        Initialized reference policy model in evaluation mode.
    """
    # inference model only has stage 3 (sharding) or stage 0 (no sharding)
    # stage 2 is optimizer sharding which doesn't apply to inference
    torch_dtype = torch.float16 if model_config.dtype == "float16" else torch.bfloat16
    ref_policy: transformers.PreTrainedModel = transformers.AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        revision=model_config.model_revision,
        dtype=torch_dtype,
        attn_implementation=model_config.attn_implementation,
        use_cache=False,
        **({"device_map": {"": local_rank}} if deepspeed_stage != 3 else {}),
    )
    disable_dropout_in_model(ref_policy)
    ref_policy, *_ = deepspeed.initialize(model=ref_policy, config=ds_config, mpu=mpu)
    ref_policy.eval()

    if checkpoint_path:
        # throw an error if we fail to load AND we are updating the reference.
        maybe_load_checkpoint(
            model=ref_policy,
            checkpoint_path=checkpoint_path,
            device=device,
            rank=rank,
            throw_on_error=ref_policy_update_freq is not None and alpha != 0,
        )
    return ref_policy


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute the entropy of the logits.
    Borrowed from verl (https://github.com/volcengine/verl/blob/main/verl/utils/torch_functional.py#L145)
    """
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
    return entropy


async def apply_verifiable_reward(
    reward_fn_mapping: dict[str, VerifierFunction],
    responses: list[torch.Tensor],
    decoded_responses: list[str],
    ground_truths: list[float],
    datasets: list[str],
    reward_mult: int = 10,
    queries: list[str] | None = None,
):
    if queries is None:
        queries = [None] * len(responses)

    # Collect all async tasks for parallel execution
    async_tasks = []
    task_metadata = []

    for i, (tok_prediction, prediction, ground_truth, dataset, query) in enumerate(
        zip(responses, decoded_responses, ground_truths, datasets, queries)
    ):
        # allow multiple ground truths and datasets for a single response

        # TODO: both code and lm_judge might have list of ground_truth *per instance*
        ground_truth_list = [ground_truth] if isinstance(ground_truth, str) else ground_truth
        dataset_list = [dataset] if isinstance(dataset, str) else dataset
        assert len(ground_truth_list) == len(dataset_list), "Ground truth and dataset list lengths do not match."

        # Create async tasks for each ground truth/dataset pair
        for gt, ds in zip(ground_truth_list, dataset_list):
            reward_func = reward_fn_mapping.get(ds.lower())
            if reward_func is None:
                logger.warning("No reward function found for dataset %s. Skipping reward.", ds)
                continue

            # Create async task
            task = reward_func.async_call(
                tokenized_prediction=tok_prediction, prediction=prediction, label=gt, query=query
            )
            async_tasks.append(task)
            # use reward_func.name to get the name of the verifier, rather than ds in case we have done remapping.
            task_metadata.append(
                {
                    "response_idx": i,
                    "dataset": reward_func.name,
                    "reward_weight": reward_func.weight,
                    "reward_mult": reward_mult,
                }
            )

    # Execute all tasks in parallel
    if async_tasks:
        reward_results = await asyncio.gather(*async_tasks)
        logger.debug(f"Applied {len(reward_results)} ground truth rewards in parallel 🤗")
    else:
        reward_results = []

    # Initialize results for each response
    response_rewards = [0] * len(responses)
    response_per_func_rewards = [{} for _ in range(len(responses))]

    # Process results
    for result, metadata in zip(reward_results, task_metadata):
        response_idx = metadata["response_idx"]
        dataset = metadata["dataset"]
        reward_weight = metadata["reward_weight"]
        reward_mult = metadata["reward_mult"]

        # Extract score from VerificationResult
        score = result.score if hasattr(result, "score") else result
        weighted_reward = reward_mult * score * reward_weight

        response_rewards[response_idx] += weighted_reward
        response_per_func_rewards[response_idx][dataset] = (
            response_per_func_rewards[response_idx].get(dataset, 0) + weighted_reward
        )

    return response_rewards, response_per_func_rewards


def get_olmo3_generation_config(tokenizer):
    return transformers.GenerationConfig(
        temperature=None,
        top_p=None,
        eos_token_id=[tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.convert_tokens_to_ids("<|endoftext|>")],
    )


@torch.compile(dynamic=True)
def log_softmax_and_gather(logits: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    torch compiled version of the common `log_softmax -> gather` operation.


    The compiled version of this opration avoids the (significant) memory overhead of
    allocating a new (batch_size, seq_len, vocab_size) tensor to store the logprobs.


    See https://github.com/allenai/open-instruct/pull/584
    """
    logprobs = logits.log_softmax(dim=-1)
    return torch.gather(logprobs, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)


@retry_on_exception()
def push_folder_to_hub(
    output_dir: str, hf_repo_id: str | None = None, hf_repo_revision: str | None = None, private: bool = True
):
    """Push a folder to Hugging Face Hub.

    This function should only run on the main process. Callers are expected to gate calls themselves.
    """
    api = HfApi()
    if not api.repo_exists(hf_repo_id):
        api.create_repo(hf_repo_id, exist_ok=True, private=private)
    if hf_repo_revision is not None:
        api.create_branch(repo_id=hf_repo_id, branch=hf_repo_revision, exist_ok=True)
    api.upload_folder(
        repo_id=hf_repo_id,
        revision=hf_repo_revision,
        folder_path=output_dir,
        commit_message="upload checkpoint",
        run_as_future=False,
    )
    logger.info(f"🔥 pushed to https://huggingface.co/{hf_repo_id}/tree/{hf_repo_revision}")


# ----------------------------------------------------------------------------
# Quality of life utilities
def print_rich_table(df: pd.DataFrame) -> Table:
    console = Console()
    table = Table(show_lines=True)
    for column in df.columns:
        table.add_column(column)
    for _, row in df.iterrows():
        table.add_row(*row.astype(str).tolist())
    console.print(table)


def format_value(value):
    if isinstance(value, float):
        if abs(value) < 1e-5:
            return f"{value:.2e}"
        return f"{value:.2f}"
    return str(value)


def print_rich_single_line_metrics(metrics):
    # Create main table
    table = Table(show_header=False, box=None)
    table.add_column("Category", style="cyan")
    table.add_column("Values", style="magenta")

    # Group metrics by their prefix
    grouped_metrics = defaultdict(list)
    for key, value in metrics.items():
        category = key.split("/")[0] if "/" in key else "other"
        grouped_metrics[category].append((key, value))

    # Sort groups by category name
    for category in sorted(grouped_metrics.keys()):
        values = grouped_metrics[category]
        value_strings = []
        for key, value in values:
            # Use everything after the first "/" as the display name
            parts = key.split("/")
            display_name = "/".join(parts[1:]) if len(parts) > 1 else key
            value_strings.append(f"{display_name}: {format_value(value)}")

        # Join all values for this category into a single string
        values_str = " | ".join(value_strings)
        table.add_row(category, values_str)

    # Create a panel with the table
    panel = Panel(table, title="Metrics", expand=False, border_style="bold green")

    # Print the panel
    rprint(panel)


def estimate_kl(ref_logprobs_diff: torch.Tensor, ratio: torch.Tensor) -> torch.Tensor:
    """Compute 4 different KL divergence estimators between current and reference policies.

    Args:
        ref_logprobs_diff: Log probability difference (new_logprobs - ref_logprobs), clamped
            to [-40, 40] for numerical stability. Shape: [B, T] or similar.
        ratio: Importance sampling ratio exp(new_logprobs - old_logprobs) between current
            policy and the policy at the start of the training step. Shape: [B, T] or similar.

    Returns:
        Tensor of shape [4, B, T] containing 4 KL estimators stacked along dim 0:
            [0]: linear approximation (ref_logprobs_diff)
            [1]: quadratic approximation (ref_logprobs_diff^2 / 2)
            [2]: numerically stable form (expm1(-ref_logprobs_diff) + ref_logprobs_diff)
            [3]: importance-weighted (ratio * ref_logprobs_diff)

        We tend to prefer [2] as a reasonable default.
    """
    return torch.stack(
        [
            ref_logprobs_diff,
            (ref_logprobs_diff) ** 2 / 2,
            torch.expm1(-ref_logprobs_diff) + ref_logprobs_diff,
            ratio * ref_logprobs_diff,
        ]
    )
