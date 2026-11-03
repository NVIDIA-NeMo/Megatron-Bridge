# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Frozen EP-sharded lookup for Qwen3.8's concatenated padded n-gram table.

The owner/request/return all-to-all protocol and forked shard initialization are
adapted from Megatron-LM PR #7553, commit 1ce7e86a073c18bcb6ea2ca5542c91e270d06a12,
``megatron/core/models/engram/distributed_embedding.py``:
https://github.com/NVIDIA/Megatron-LM/pull/7553
https://github.com/NVIDIA/Megatron-LM/blob/1ce7e86a073c18bcb6ea2ca5542c91e270d06a12/megatron/core/models/engram/distributed_embedding.py. Unlike that trainable,
multitable implementation, this module preserves one logical ``weight`` and has
no embedding backward or tensor-parallel reduction.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from torch import Tensor, nn


if TYPE_CHECKING:
    from megatron.core.dist_checkpointing.mapping import ShardedStateDict
    from megatron.core.transformer.transformer_config import TransformerConfig


class FrozenEPNgramEmbedding(nn.Module):
    """Shard one frozen padded table by rows over an explicit expert group.

    The caller computes hashes on full sequences, then slices sequence-parallel
    token positions before lookup. Every EP member must call forward in the same
    order, including members with no local requests. Expert-DP groups replicate
    a row shard; dense TP ranks are not additional checkpoint replicas.
    """

    def __init__(
        self,
        config: TransformerConfig,
        num_embeddings: int,
        embedding_dim: int,
        *,
        init_method: Callable[[Tensor], None],
        ep_group: dist.ProcessGroup,
        expt_dp_group: dist.ProcessGroup,
    ) -> None:
        super().__init__()
        if not dist.is_initialized() or ep_group is None or expt_dp_group is None:
            raise ValueError("Frozen EP embedding requires explicit initialized EP and expert-DP groups")
        if config.expert_tensor_parallel_size != 1:
            raise ValueError("Frozen EP embedding requires expert_tensor_parallel_size=1")
        self.config = config
        self.ep_group = self.sharding_group = ep_group
        self.expt_dp_group = expt_dp_group
        self.ep_rank = dist.get_rank(ep_group)
        self.ep_size = dist.get_world_size(ep_group)
        if self.ep_rank < 0 or dist.get_rank(expt_dp_group) < 0:
            raise ValueError("This rank must belong to both embedding process groups")
        if type(num_embeddings) is not int or num_embeddings <= 0 or num_embeddings % self.ep_size:
            raise ValueError("Padded n-gram table rows must be positive and divisible by EP size")
        if type(embedding_dim) is not int or embedding_dim <= 0:
            raise ValueError("Embedding dimension must be positive")
        self.num_embeddings = self.global_num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.local_num_embeddings = num_embeddings // self.ep_size
        self.row_start = self.ep_rank * self.local_num_embeddings
        self.row_end = self.row_start + self.local_num_embeddings
        device = (
            torch.device("cpu") if config.use_cpu_initialization else torch.device("cuda", torch.cuda.current_device())
        )
        self.weight = nn.Parameter(
            torch.empty((self.local_num_embeddings, embedding_dim), dtype=config.params_dtype, device=device),
            requires_grad=False,
        )
        # DDP broadcasts frozen parameters too: only expert-DP peers share this shard.
        self.weight.allreduce = False
        self.weight.is_embedding_or_output_parameter = True
        if config.perform_initialization:
            self._initialize_weight(init_method)

    def _initialize_weight(self, init_method: Callable[[Tensor], None]) -> None:
        """Initialize without advancing the caller's CPU or local CUDA RNG stream."""
        seed = (torch.initial_seed() + 100003 * self.global_num_embeddings + self.ep_rank) % (2**63)
        devices = [self.weight.device.index] if self.weight.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            torch.default_generator.manual_seed(seed)
            if devices:
                with torch.cuda.device(self.weight.device):
                    torch.cuda.manual_seed(seed)
            init_method(self.weight)

    @torch.no_grad()
    def forward(self, row_ids: Tensor) -> Tensor:
        """Return exact table rows in the input order, with shape ``[*ids.shape, D]``.

        A validation bit travels with each peer's request count, so an invalid
        rank fails the whole EP group before any request/value exchange. This
        shares the existing count exchange rather than adding an all-reduce.
        Duplicate IDs and zero counts are retained; no reduction changes values.
        """
        valid_metadata = (
            row_ids.dtype == torch.int64 and row_ids.device == self.weight.device and not self.weight.requires_grad
        )
        rows = row_ids.reshape(-1) if valid_metadata else torch.empty(0, dtype=torch.int64, device=self.weight.device)
        invalid = ((rows < 0) | (rows >= self.global_num_embeddings)).any()
        invalid = invalid | (not valid_metadata)
        owners = torch.div(rows, self.local_num_embeddings, rounding_mode="floor").clamp(0, self.ep_size - 1)
        counts = torch.bincount(owners, minlength=self.ep_size)
        sent_counts = torch.stack((counts, invalid.to(torch.int64).expand_as(counts)), dim=1)
        received_counts = torch.empty_like(sent_counts)
        if self.ep_size > 1:
            dist.all_to_all_single(received_counts, sent_counts, group=self.ep_group)
        else:
            received_counts.copy_(sent_counts)
        sent, received = torch.stack((sent_counts, received_counts)).tolist()
        if any(row[1] for row in received):
            raise ValueError(
                "Frozen EP lookup requires frozen weight and device-local int64 IDs in the global row range"
            )
        send_splits = [row[0] for row in sent]
        recv_splits = [row[0] for row in received]
        order = torch.argsort(owners, stable=True)
        requests = rows.index_select(0, order) - owners.index_select(0, order) * self.local_num_embeddings
        received_requests = requests.new_empty(sum(recv_splits))
        if self.ep_size > 1:
            dist.all_to_all_single(
                received_requests,
                requests.contiguous(),
                output_split_sizes=recv_splits,
                input_split_sizes=send_splits,
                group=self.ep_group,
            )
        else:
            received_requests.copy_(requests)
        local_values = self.weight.index_select(0, received_requests)
        returned = self.weight.new_empty((rows.numel(), self.embedding_dim))
        if self.ep_size > 1:
            dist.all_to_all_single(
                returned,
                local_values.contiguous(),
                output_split_sizes=send_splits,
                input_split_sizes=recv_splits,
                group=self.ep_group,
            )
        else:
            returned.copy_(local_values)
        output = torch.empty_like(returned)
        output.index_copy_(0, order, returned)
        return output.view(*row_ids.shape, self.embedding_dim)

    def sharded_state_dict(
        self, prefix: str = "", sharded_offsets: tuple = (), metadata: dict | None = None
    ) -> ShardedStateDict:
        """Keep the original single-table logical key with EP rows and EDP replicas."""
        from megatron.core.dist_checkpointing.mapping import ShardedTensor

        del metadata
        if sorted(axis for axis, _, _ in sharded_offsets) != list(range(len(sharded_offsets))):
            raise ValueError("Only prepended sharded offsets are supported for the n-gram table")
        key = f"{prefix}weight"
        return {
            key: ShardedTensor.from_rank_offsets(
                key,
                self.weight,
                *sharded_offsets,
                (len(sharded_offsets), self.ep_rank, self.ep_size),
                replica_id=(0, 0, dist.get_rank(self.expt_dp_group)),
                prepend_axis_num=len(sharded_offsets),
            )
        }
