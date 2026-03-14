import dataclasses
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F

from open_instruct import data_types
from open_instruct.utils.general import INVALID_LOGPROB


@dataclass
class UlyssesSPSplitter:
    """Splits CollatedBatchData for Ulysses sequence parallelism.

    Adapted from the UlyssesSPDataLoaderAdapter
    (https://github.com/deepspeedai/DeepSpeed/blob/master/deepspeed/runtime/sequence_parallel/ulysses_sp.py#L427)
    Rather than wrapping a dataloader, we just want to split a given batch into the shards across sp group.
    """

    sp_rank: int
    sp_group: "torch.distributed.distributed_c10d.ProcessGroup"
    sp_world_size: int
    device: torch.device
    pad_token_id: int

    def split_collated_batch(self, data: data_types.CollatedBatchData) -> data_types.CollatedBatchData:
        """Get this rank's shard of a CollatedBatchData for sequence parallelism."""
        # Find max sequence length across all ranks to ensure consistent padding
        local_max = max(t.shape[-1] for t in data.query_responses)
        local_seqlen = torch.tensor([local_max], dtype=torch.int64, device=self.device)
        seqlens = [torch.zeros(1, dtype=torch.int64, device=self.device) for _ in range(self.sp_world_size)]
        dist.all_gather(seqlens, local_seqlen, group=self.sp_group)
        max_seqlen = max(x.item() for x in seqlens)

        # Round up to be divisible by sp_world_size
        max_seqlen = ((max_seqlen + self.sp_world_size - 1) // self.sp_world_size) * self.sp_world_size
        chunk_len = max_seqlen // self.sp_world_size

        # Compute start and end indices for this rank's chunk
        start_idx = chunk_len * self.sp_rank
        end_idx = chunk_len * (self.sp_rank + 1)

        # slice and pad tensors for this sp rank
        kwargs = {}
        for field in dataclasses.fields(data):
            if field.name == "query_responses":
                pad_value = self.pad_token_id
            elif field.name == "vllm_logprobs":
                pad_value = INVALID_LOGPROB
            else:
                pad_value = 0
            sharded = []
            for t in getattr(data, field.name):
                # For all tensors in batch, pad tensor to max_seqlen, then slice to get this SP rank's chunk
                padded_sliced = F.pad(t, (0, max_seqlen - t.shape[-1]), value=pad_value)[:, start_idx:end_idx]
                sharded.append(padded_sliced)
            kwargs[field.name] = sharded

        return data_types.CollatedBatchData(**kwargs)
