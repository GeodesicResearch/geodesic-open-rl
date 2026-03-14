import dataclasses
import json
import math

import torch
from transformers import AutoConfig

from open_instruct.utils.logger import setup_logger

logger = setup_logger(__name__)


# For FLOPS, we assume bf16 and ignore sparsity.
# Memory bandwidth values are peak theoretical bandwidth.
GPU_SPECS = {
    "a100": {"flops": 312e12, "memory_size": 80e9, "memory_bandwidth": 2.0e12},  # 2.0 TB/s HBM2e (80GB variant)
    "b200": {"flops": 2250e12, "memory_size": 192e9, "memory_bandwidth": 8e12},  # 8 TB/s HBM3e
    "h100": {"flops": 990e12, "memory_size": 80e9, "memory_bandwidth": 3.35e12},  # 3.35 TB/s HBM3
    "h200": {"flops": 989e12, "memory_size": 141e9, "memory_bandwidth": 4.8e12},  # 4.8 TB/s HBM3e
    "a6000": {"flops": 155e12, "memory_size": 48e9, "memory_bandwidth": 768e9},  # 768 GB/s GDDR6
    "l40s": {"flops": 362e12, "memory_size": 48e9, "memory_bandwidth": 864e9},  # 864 GB/s GDDR6
    "pro 6000": {"flops": 503.8e12, "memory_size": 96e9, "memory_bandwidth": 1792e9},  # 1792 GB/s GDDR7
    "6000": {"flops": 728.5e12, "memory_size": 48e9, "memory_bandwidth": 960e9},  # 960 GB/s GDDR6
    # Specs from https://www.techpowerup.com/gpu-specs/geforce-rtx-4090-mobile.c3949.
    "4090 laptop": {"flops": 32.98e12, "memory_size": 24e9, "memory_bandwidth": 576e9},
    # DGX Spark GB10 (Blackwell) - unified LPDDR5X memory with CPU
    # Specs from https://www.nvidia.com/en-us/products/workstations/dgx-spark/
    "gb10": {"flops": 104e12, "memory_size": 128e9, "memory_bandwidth": 273e9},  # 273 GB/s LPDDR5X unified
}

# Conventions for FLOPs calculations (fixed; not switches)
FLOP_PER_MAC = 2
# Approximate softmax cost per attention score:
# ~4 scalar ops/score: exp + subtract max (stabilization) + sum + divide.
SOFTMAX_FLOPS_PER_SCORE = 4


def _safe_get_device_name() -> str | None:
    """Get normalized GPU device name, returning None if CUDA init fails."""
    try:
        if torch.cuda.is_available():
            return get_device_name(torch.cuda.get_device_name(0))
    except RuntimeError:
        pass
    return None


@dataclasses.dataclass
class ModelDims:
    num_layers: int
    hidden_size: int
    intermediate_size: int
    vocab_size: int
    num_attn_heads: int
    head_dim: int
    num_kv_heads: int | None = None
    num_params: int | None = None
    device_name: str | None = None
    sliding_window: int | None = None
    num_sliding_window_layers: int = 0

    def __post_init__(self):
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_attn_heads

        self.num_params = self.num_params or self._calculate_num_params()

        if self.device_name is None:
            self.device_name = _safe_get_device_name()

        assert self.hidden_size % self.num_attn_heads == 0, "hidden_size must be divisible by num_attn_heads"
        assert self.num_attn_heads % self.num_kv_heads == 0, (
            "num_attn_heads must be divisible by num_kv_heads (GQA/MQA)"
        )
        assert self.num_sliding_window_layers <= self.num_layers, (
            f"num_sliding_window_layers ({self.num_sliding_window_layers}) cannot exceed num_layers ({self.num_layers})"
        )

    def _calculate_num_params(self) -> int:
        embedding_params = self.vocab_size * self.hidden_size

        q_params = self.hidden_size * (self.num_attn_heads * self.head_dim)
        kv_params = self.hidden_size * (self.num_kv_heads * self.head_dim) * 2
        o_params = (self.num_attn_heads * self.head_dim) * self.hidden_size
        mlp_up_params = self.hidden_size * self.intermediate_size * 2
        mlp_down_params = self.intermediate_size * self.hidden_size

        per_layer_params = q_params + kv_params + o_params + mlp_up_params + mlp_down_params
        layer_params = self.num_layers * per_layer_params

        lm_head_params = self.vocab_size * self.hidden_size

        return embedding_params + layer_params + lm_head_params

    @classmethod
    def from_hf_config(cls, model_name_or_path: str) -> "ModelDims":
        """Create ModelDims from a HuggingFace model name or path."""
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        hidden_size = config.hidden_size
        intermediate_size = getattr(config, "intermediate_size", 4 * hidden_size)
        sliding_window = getattr(config, "sliding_window", None)
        num_sliding_window_layers = 0
        if sliding_window is not None:
            layer_types = getattr(config, "layer_types", None)
            if layer_types is not None:
                num_sliding_window_layers = layer_types.count("sliding_attention")
            else:
                num_sliding_window_layers = config.num_hidden_layers
        head_dim = getattr(config, "head_dim", hidden_size // config.num_attention_heads)
        return cls(
            num_layers=config.num_hidden_layers,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            vocab_size=config.vocab_size,
            num_attn_heads=config.num_attention_heads,
            num_kv_heads=getattr(config, "num_key_value_heads", config.num_attention_heads),
            head_dim=head_dim,
            sliding_window=sliding_window,
            num_sliding_window_layers=num_sliding_window_layers,
            device_name=_safe_get_device_name(),
        )

    @property
    def device_flops(self) -> float:
        assert self.device_name is not None, "device_name must be set"
        assert self.device_name in GPU_SPECS, f"Unknown device: {self.device_name}"
        return GPU_SPECS[self.device_name]["flops"]

    @property
    def device_memory_bandwidth(self) -> float:
        assert self.device_name is not None, "device_name must be set"
        assert self.device_name in GPU_SPECS, f"Unknown device: {self.device_name}"
        return GPU_SPECS[self.device_name]["memory_bandwidth"]

    def attn_flops(self, query_len: int, kv_len: int, sliding_window: int | None = None) -> int:
        d = self.head_dim
        mul = FLOP_PER_MAC

        q_dim = self.num_attn_heads * d
        kv_dim = self.num_kv_heads * d

        kv_len = min(kv_len, sliding_window or float("inf"))

        q_proj = mul * query_len * self.hidden_size * q_dim
        kv_proj = mul * 2 * query_len * self.hidden_size * kv_dim

        qk = mul * self.num_attn_heads * query_len * kv_len * d
        softmax = SOFTMAX_FLOPS_PER_SCORE * self.num_attn_heads * query_len * kv_len
        av = mul * self.num_attn_heads * query_len * kv_len * d

        out_proj = mul * query_len * q_dim * self.hidden_size

        return q_proj + kv_proj + qk + softmax + av + out_proj

    def mlp_flops(self, seq_len: int) -> int:
        mul = FLOP_PER_MAC
        first = mul * seq_len * self.hidden_size * (self.intermediate_size * 2)
        act = seq_len * self.intermediate_size
        second = mul * seq_len * self.intermediate_size * self.hidden_size
        return first + act + second

    def prefill_flops(self, prompt_lengths: list[int]) -> int:
        num_full_attn_layers = self.num_layers - self.num_sliding_window_layers
        num_sliding_layers = self.num_sliding_window_layers

        total = 0
        for L in prompt_lengths:
            if num_full_attn_layers > 0:
                total += num_full_attn_layers * (self.attn_flops(L, L, sliding_window=None) + self.mlp_flops(L))

            if num_sliding_layers > 0:
                total += num_sliding_layers * (
                    self.attn_flops(L, L, sliding_window=self.sliding_window) + self.mlp_flops(L)
                )

            total += L * FLOP_PER_MAC * self.hidden_size * self.vocab_size

        return total

    def decode_flops(self, prompt_lengths: list[int], response_lengths: list[int], samples_per_prompt: int = 1) -> int:
        assert len(response_lengths) == len(prompt_lengths) * samples_per_prompt, (
            f"Expected {len(prompt_lengths) * samples_per_prompt} response lengths, got {len(response_lengths)}"
        )

        num_full_attn_layers = self.num_layers - self.num_sliding_window_layers
        num_sliding_layers = self.num_sliding_window_layers

        total = 0
        response_idx = 0
        for P in prompt_lengths:
            for _ in range(samples_per_prompt):
                R = response_lengths[response_idx]
                total += R * self.num_layers * self.mlp_flops(seq_len=1)
                for t in range(R):
                    kv_len = P + t + 1
                    if num_full_attn_layers > 0:
                        total += num_full_attn_layers * self.attn_flops(
                            query_len=1, kv_len=kv_len, sliding_window=None
                        )
                    if num_sliding_layers > 0:
                        total += num_sliding_layers * self.attn_flops(
                            query_len=1, kv_len=kv_len, sliding_window=self.sliding_window
                        )
                total += R * FLOP_PER_MAC * self.hidden_size * self.vocab_size
                response_idx += 1
        return total

    def flops(
        self,
        prompt_lengths: list[int],
        response_lengths: list[int] | None = None,
        samples_per_prompt: int = 1,
        is_training: bool = False,
    ) -> int:
        total = self.prefill_flops(prompt_lengths)
        if response_lengths is not None:
            total += self.decode_flops(prompt_lengths, response_lengths, samples_per_prompt)
        if is_training:
            total *= 3
        return total

    def weight_memory_bytes(self, num_tokens: int, dtype_bytes: int = 2) -> int:
        hidden_q = self.num_attn_heads * self.head_dim
        hidden_kv = self.num_kv_heads * self.head_dim

        w_q = self.hidden_size * hidden_q
        w_k = self.hidden_size * hidden_kv
        w_v = self.hidden_size * hidden_kv
        w_o = hidden_q * self.hidden_size
        w_up = self.hidden_size * (self.intermediate_size * 2)
        w_dn = self.intermediate_size * self.hidden_size

        per_layer_weight_bytes = (w_q + w_k + w_v + w_o + w_up + w_dn) * dtype_bytes
        return self.num_layers * num_tokens * per_layer_weight_bytes

    def kv_cache_write_bytes(self, num_tokens: int, dtype_bytes: int = 2) -> int:
        kv_write_bytes_per_token = 2 * self.num_kv_heads * self.head_dim * dtype_bytes
        return self.num_layers * num_tokens * kv_write_bytes_per_token

    def kv_cache_read_bytes(
        self, prompt_lengths: list[int], response_lengths: list[int], samples_per_prompt: int = 1, dtype_bytes: int = 2
    ) -> int:
        assert len(response_lengths) == len(prompt_lengths) * samples_per_prompt, (
            f"Expected {len(prompt_lengths) * samples_per_prompt} response lengths, got {len(response_lengths)}"
        )

        num_full_attn_layers = self.num_layers - self.num_sliding_window_layers
        num_sliding_layers = self.num_sliding_window_layers

        kv_read_terms = 0
        response_idx = 0

        for P in prompt_lengths:
            prompt_responses = []
            for _ in range(samples_per_prompt):
                prompt_responses.append(response_lengths[response_idx])
                response_idx += 1

            max_response_length = max(prompt_responses) if prompt_responses else 0
            kv_read_terms += max_response_length * samples_per_prompt * P * num_full_attn_layers

            for R in prompt_responses:
                kv_read_terms += num_full_attn_layers * R * (R - 1) // 2
                if num_sliding_layers > 0:
                    kv_read_terms += num_sliding_layers * sum(min(P + t, self.sliding_window) for t in range(R))
        kv_bytes_per_token = 2 * self.num_kv_heads * self.head_dim * dtype_bytes
        return kv_bytes_per_token * kv_read_terms

    def prefill_memory_bytes(self, prompt_lengths: list[int], dtype_bytes: int = 2) -> int:
        num_prefill_ops = 1
        weight_bytes = self.weight_memory_bytes(num_prefill_ops, dtype_bytes)
        total_prefill_tokens = sum(prompt_lengths)
        kv_write_bytes = self.kv_cache_write_bytes(total_prefill_tokens, dtype_bytes)
        return weight_bytes + kv_write_bytes

    def decode_memory_bytes(
        self, prompt_lengths: list[int], response_lengths: list[int], samples_per_prompt: int = 1, dtype_bytes: int = 2
    ) -> int:
        unique_positions = 0
        response_idx = 0
        for _ in prompt_lengths:
            prompt_responses = response_lengths[response_idx : response_idx + samples_per_prompt]
            response_idx += samples_per_prompt
            unique_positions += max(prompt_responses) if prompt_responses else 0

        weight_bytes = self.weight_memory_bytes(unique_positions, dtype_bytes)

        total_decode_tokens = sum(response_lengths)
        kv_write_bytes = self.kv_cache_write_bytes(total_decode_tokens, dtype_bytes)

        kv_read_bytes = self.kv_cache_read_bytes(prompt_lengths, response_lengths, samples_per_prompt, dtype_bytes)
        return weight_bytes + kv_write_bytes + kv_read_bytes

    def memory_bytes(
        self,
        prompt_lengths: list[int],
        num_engines: int,
        num_gpus_per_engine: int,
        response_lengths: list[int] | None = None,
        samples_per_prompt: int = 1,
        dtype_bytes: int = 2,
    ) -> int:
        if num_engines < 1:
            raise ValueError(f"num_engines must be >= 1, got {num_engines}")
        if num_gpus_per_engine < 1:
            raise ValueError(f"num_gpus_per_engine must be >= 1, got {num_gpus_per_engine}")

        if not prompt_lengths:
            return 0

        def _split_evenly(seq: list[int], parts: int) -> list[list[int]]:
            base, extra = divmod(len(seq), parts)
            result: list[list[int]] = []
            start = 0
            for i in range(parts):
                size = base + (1 if i < extra else 0)
                result.append(seq[start : start + size])
                start += size
            return result

        prompt_chunks = _split_evenly(prompt_lengths, num_engines)

        response_chunks: list[list[int] | None]
        if response_lengths is not None:
            assert len(response_lengths) == len(prompt_lengths) * samples_per_prompt, (
                f"Expected {len(prompt_lengths) * samples_per_prompt} response lengths, got {len(response_lengths)}"
            )
            response_chunks = []
            response_idx = 0
            for chunk in prompt_chunks:
                num_responses = len(chunk) * samples_per_prompt
                response_chunks.append(response_lengths[response_idx : response_idx + num_responses])
                response_idx += num_responses
        else:
            response_chunks = [None] * num_engines

        per_engine_totals: list[int] = []
        for chunk_prompts, chunk_responses in zip(prompt_chunks, response_chunks):
            if not chunk_prompts:
                per_engine_totals.append(0)
                continue

            total = self.prefill_memory_bytes(chunk_prompts, dtype_bytes)
            if chunk_responses is not None:
                total += self.decode_memory_bytes(chunk_prompts, chunk_responses, samples_per_prompt, dtype_bytes)
            per_engine_totals.append(total)

        if len(per_engine_totals) < num_engines:
            per_engine_totals.extend([0] * (num_engines - len(per_engine_totals)))

        avg_bytes_per_engine = math.ceil(sum(per_engine_totals) / num_engines)
        return avg_bytes_per_engine

    def calculate_mfu(
        self,
        prompt_lengths: list[int],
        generation_time: float,
        response_lengths: list[int] | None = None,
        samples_per_prompt: int = 1,
        num_gpus: int = 1,
    ) -> float:
        total_flops = self.flops(prompt_lengths, response_lengths, samples_per_prompt=samples_per_prompt)
        flops_per_second = total_flops / generation_time if generation_time > 0 else 0
        total_device_flops = self.device_flops * num_gpus
        return 100 * flops_per_second / total_device_flops

    def calculate_mbu(
        self,
        prompt_lengths: list[int],
        generation_time: float,
        response_lengths: list[int] | None = None,
        samples_per_prompt: int = 1,
        num_engines: int = 1,
        num_gpus_per_engine: int = 1,
    ) -> float:
        total_memory_bytes = self.memory_bytes(
            prompt_lengths,
            num_engines,
            num_gpus_per_engine,
            response_lengths=response_lengths,
            samples_per_prompt=samples_per_prompt,
        )
        bytes_per_second = total_memory_bytes / generation_time if generation_time > 0 else 0
        total_device_bandwidth = self.device_memory_bandwidth * num_engines * num_gpus_per_engine
        return 100 * bytes_per_second / total_device_bandwidth

    def calculate_actor_utilization(
        self,
        prompt_lengths: list[int],
        response_lengths: list[int],
        total_generation_time: float,
        samples_per_prompt: int,
        num_engines: int,
        num_gpus_per_engine: int,
    ) -> dict[str, float]:
        actor_mfu = self.calculate_mfu(
            prompt_lengths,
            total_generation_time,
            response_lengths=response_lengths,
            samples_per_prompt=samples_per_prompt,
            num_gpus=num_engines * num_gpus_per_engine,
        )
        actor_mbu = self.calculate_mbu(
            prompt_lengths,
            total_generation_time,
            response_lengths=response_lengths,
            samples_per_prompt=samples_per_prompt,
            num_engines=num_engines,
            num_gpus_per_engine=num_gpus_per_engine,
        )

        check_calculation(
            actor_mfu,
            "Actor MFU",
            self,
            total_generation_time,
            prompt_lengths,
            response_lengths,
            samples_per_prompt,
            num_engines,
            num_gpus_per_engine,
        )

        check_calculation(
            actor_mbu,
            "Actor MBU",
            self,
            total_generation_time,
            prompt_lengths,
            response_lengths,
            samples_per_prompt,
            num_engines,
            num_gpus_per_engine,
        )

        return {"mfu": actor_mfu, "mbu": actor_mbu}

    def calculate_learner_utilization(
        self,
        prompt_lengths: list[int],
        response_lengths: list[int],
        training_time: float,
        samples_per_prompt: int,
        num_training_gpus: int,
    ) -> dict[str, float]:
        total_sequence_lengths = [
            prompt_lengths[i // samples_per_prompt] + response_len for i, response_len in enumerate(response_lengths)
        ]

        training_flops = self.flops(
            prompt_lengths=total_sequence_lengths, response_lengths=None, samples_per_prompt=1, is_training=True
        )

        training_flops_per_second = training_flops / training_time
        total_training_device_flops = self.device_flops * num_training_gpus
        learner_mfu = 100 * training_flops_per_second / total_training_device_flops

        check_calculation(
            learner_mfu, "Learner MFU", self, training_time, total_sequence_lengths, None, 1, 1, num_training_gpus
        )

        return {"mfu": learner_mfu}

    def approximate_learner_utilization(
        self, total_tokens: int, avg_sequence_length: float, training_time: float, num_training_gpus: int
    ) -> dict[str, float]:
        num_sequences = int(total_tokens / avg_sequence_length)
        sequence_lengths = [int(avg_sequence_length)] * num_sequences

        training_flops = self.flops(
            prompt_lengths=sequence_lengths, response_lengths=None, samples_per_prompt=1, is_training=True
        )

        training_flops_per_second = training_flops / training_time
        total_training_device_flops = self.device_flops * num_training_gpus
        learner_mfu = 100 * training_flops_per_second / total_training_device_flops

        return {"mfu": learner_mfu}


def get_device_name(device_name: str) -> str:
    """Normalize a GPU device name to a standard key used in GPU_SPECS."""
    normalized_device_name = device_name.lower().replace("-", " ")

    for key in GPU_SPECS:
        if key in normalized_device_name:
            return key
    raise ValueError(
        f"Unknown device name: {device_name}. Expected one of: {list(GPU_SPECS.keys())}. "
        f"Please raise an issue at https://github.com/allenai/open-instruct/issues with the device you need. In the interim, you can add the specs for your device using the name {normalized_device_name} to the GPU_SPECS dictionary in utils/flops.py."
    )


def calculate_utilization_metrics(
    model_dims: ModelDims,
    prompt_lengths: list[int],
    response_lengths: list[int],
    total_generation_time: float,
    samples_per_prompt: int,
    num_engines: int,
    num_gpus_per_engine: int,
    training_time: float,
    num_training_gpus: int,
) -> dict:
    assert len(response_lengths) == len(prompt_lengths) * samples_per_prompt, (
        f"Expected {len(prompt_lengths) * samples_per_prompt} response lengths, got {len(response_lengths)}"
    )

    actor_metrics = model_dims.calculate_actor_utilization(
        prompt_lengths=prompt_lengths,
        response_lengths=response_lengths,
        total_generation_time=total_generation_time,
        samples_per_prompt=samples_per_prompt,
        num_engines=num_engines,
        num_gpus_per_engine=num_gpus_per_engine,
    )

    learner_metrics = model_dims.calculate_learner_utilization(
        prompt_lengths=prompt_lengths,
        response_lengths=response_lengths,
        training_time=training_time,
        samples_per_prompt=samples_per_prompt,
        num_training_gpus=num_training_gpus,
    )

    utilization_metrics = {f"actor_{k}": v for k, v in actor_metrics.items()}
    utilization_metrics["learner_mfu"] = learner_metrics["mfu"]

    return utilization_metrics


def check_calculation(
    percentage: float,
    metric_name: str,
    model_dims: ModelDims,
    timing: float,
    prompt_lengths: list[int],
    response_lengths: list[int] | None,
    samples_per_prompt: int,
    num_engines: int,
    num_gpus_per_engine: int,
) -> None:
    if percentage <= 100:
        return

    full_device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    avg_prompt_length = sum(prompt_lengths) / len(prompt_lengths)
    avg_response_length = sum(response_lengths) / len(response_lengths) if response_lengths else 0

    test_case_json = {
        "model_name": "REPLACE_WITH_MODEL_NAME",
        "total_generation_time": timing,
        "samples_per_prompt": samples_per_prompt,
        "num_engines": num_engines,
        "num_gpus_per_engine": num_gpus_per_engine,
        "training_time": "REPLACE_WITH_TRAINING_TIME",
        "num_training_gpus": "REPLACE_WITH_NUM_TRAINING_GPUS",
        "prompt_lengths": prompt_lengths,
        "response_lengths": response_lengths,
    }

    warning_message = (
        f"{metric_name} exceeded 100%: {percentage:.2f}%\n"
        f"\n"
        f"{model_dims}\n"
        f"\n"
        f"Timing and GPU info:\n"
        f"  timing: {timing:.6f}s\n"
        f"  num_engines: {num_engines}\n"
        f"  num_gpus_per_engine: {num_gpus_per_engine}\n"
        f"  full_device_name: {full_device_name}\n"
        f"\n"
        f"Batch/sequence info:\n"
        f"  num_prompts: {len(prompt_lengths)}\n"
        f"  samples_per_prompt: {samples_per_prompt}\n"
        f"  avg_prompt_length: {avg_prompt_length:.1f}\n"
        f"  avg_response_length: {avg_response_length:.1f}\n"
        f"\n"
        f"To reproduce this calculation, use these exact parameters:\n"
        f"  prompt_lengths = {prompt_lengths}\n"
        f"  response_lengths = {response_lengths}\n"
        f"  timing = {timing}\n"
        f"  samples_per_prompt = {samples_per_prompt}\n"
        f"  num_engines = {num_engines}\n"
        f"  num_gpus_per_engine = {num_gpus_per_engine}\n"
        f"\n"
        f"JSON format for test case (copy this to mbu_reproduction_cases.json):\n"
        f"{json.dumps(test_case_json, indent=2)}\n"
        f"\n"
        f"This may indicate an issue with the MFU/MBU calculation logic or GPU specifications.\n"
        f"Please raise an issue at https://github.com/allenai/open-instruct/issues with the above information."
    )

    logger.warning(warning_message)
