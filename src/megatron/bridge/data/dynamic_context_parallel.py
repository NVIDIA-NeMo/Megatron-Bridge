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

"""Framework-owned THD batch materialization for dynamic context parallelism."""

import logging
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from megatron.core.datasets.data_schedule import DefaultDynamicCPScheduler
from megatron.core.datasets.data_schedule_utils import (
    gather_global_sequence_lengths,
    reroute_tensor_fields_to_dcp_ranks,
)
from megatron.core.rerun_state_machine import RerunDataIterator


logger = logging.getLogger(__name__)


_DEFAULT_SEQUENCE_FIELD_PAD_VALUES: dict[str, int | float | bool] = {
    "tokens": 0,
    "labels": 0,
    "loss_mask": 0,
    "position_ids": 0,
}


@dataclass(frozen=True)
class DynamicCPBatchPolicy:
    """Describe sequence-aligned fields that a framework materializes for DCP.

    MCore only transports the selected tensors as flattened payloads. The
    framework-specific caller owns field selection, padding values, and shape
    reconstruction. NeMo RL can therefore provide a policy that additionally
    carries fields such as advantages or old log-probabilities without teaching
    MCore about their semantics.

    Args:
        sequence_field_pad_values: Mapping from each routed, sequence-aligned
            tensor field to its physical THD padding value.
    """

    sequence_field_pad_values: Mapping[str, int | float | bool] = field(
        default_factory=lambda: dict(_DEFAULT_SEQUENCE_FIELD_PAD_VALUES)
    )

    def __post_init__(self) -> None:
        """Validate the minimum GPT forward contract."""
        fields = set(self.sequence_field_pad_values)
        required_fields = set(_DEFAULT_SEQUENCE_FIELD_PAD_VALUES)
        missing_fields = required_fields - fields
        if missing_fields:
            raise ValueError(f"Dynamic CP batch policy is missing required fields {sorted(missing_fields)}.")


@dataclass(frozen=True)
class DynamicCPBatch:
    """A scheduled local iterator and global FLOPS metadata for one logical batch."""

    data_iterator: RerunDataIterator
    num_microbatches: int
    padded_token_sum: int
    logical_seqlen_squared_sum: int


@dataclass(frozen=True)
class _UnpackedBatch:
    samples: list[dict[str, torch.Tensor]]
    logical_lengths: list[int]
    padded_lengths: list[int]


def _as_boundary_rows(value: Any, *, field_name: str, batch_size: int) -> torch.Tensor:
    """Normalize one THD boundary tensor to ``[batch, boundaries]`` on CPU."""
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"Dynamic CP requires tensor field '{field_name}'.")
    if value.dim() == 1:
        if batch_size != 1:
            raise ValueError(f"Dynamic CP field '{field_name}' is 1D but the sequence batch has {batch_size} rows.")
        value = value.unsqueeze(0)
    if value.dim() != 2 or value.size(0) != batch_size:
        raise ValueError(
            f"Dynamic CP field '{field_name}' must have shape [batch, boundaries], got {tuple(value.shape)}."
        )
    return value.detach().to(device="cpu", dtype=torch.int64)


def _unpack_thd_global_batch(batch: Mapping[str, Any], policy: DynamicCPBatchPolicy) -> _UnpackedBatch:
    """Split a collated THD global batch into unpadded framework-owned rows."""
    fields = tuple(policy.sequence_field_pad_values)
    first_field = batch.get(fields[0])
    if not isinstance(first_field, torch.Tensor) or first_field.dim() != 2:
        raise ValueError(f"Dynamic CP field '{fields[0]}' must be a 2D [batch, sequence] tensor.")
    batch_size, physical_width = first_field.shape
    for field_name in fields:
        value = batch.get(field_name)
        if not isinstance(value, torch.Tensor) or value.shape != first_field.shape:
            raise ValueError(
                f"Dynamic CP field '{field_name}' must match {fields[0]} shape {tuple(first_field.shape)}."
            )

    logical_rows = _as_boundary_rows(batch.get("cu_seqlens_q"), field_name="cu_seqlens_q", batch_size=batch_size)
    padded_value = batch.get("cu_seqlens_q_padded")
    if padded_value is None:
        raise ValueError(
            "Dynamic CP requires cu_seqlens_q_padded so Bridge can preserve physical THD alignment. "
            "Configure offline packing with pad_seq_to_mult >= 2 * context_parallel_size."
        )
    padded_rows = _as_boundary_rows(padded_value, field_name="cu_seqlens_q_padded", batch_size=batch_size)
    if logical_rows.shape != padded_rows.shape:
        raise ValueError(
            "Dynamic CP logical and physical THD boundary tensors must have identical shapes, got "
            f"{tuple(logical_rows.shape)} and {tuple(padded_rows.shape)}."
        )

    samples: list[dict[str, torch.Tensor]] = []
    logical_lengths: list[int] = []
    padded_lengths: list[int] = []
    for row_idx in range(batch_size):
        logical = logical_rows[row_idx].tolist()
        padded = padded_rows[row_idx].tolist()
        if logical[0] != 0 or padded[0] != 0:
            raise ValueError("Dynamic CP THD boundaries must start at zero.")
        if any(end < start for start, end in zip(logical, logical[1:])) or any(
            end < start for start, end in zip(padded, padded[1:])
        ):
            raise ValueError("Dynamic CP THD boundaries must be nondecreasing.")
        if padded[-1] > physical_width:
            raise ValueError(f"Dynamic CP physical boundary {padded[-1]} exceeds tensor width {physical_width}.")

        for segment_idx in range(len(logical) - 1):
            logical_length = logical[segment_idx + 1] - logical[segment_idx]
            padded_length = padded[segment_idx + 1] - padded[segment_idx]
            if logical_length == 0:
                # Fixed-width offline packing represents its trailing fill as a
                # zero-logical-length THD segment. It is not a training sample.
                continue
            if logical_length > padded_length:
                raise ValueError(
                    f"Dynamic CP logical length {logical_length} exceeds physical length {padded_length}."
                )
            physical_start = padded[segment_idx]
            physical_end = physical_start + logical_length
            samples.append(
                {
                    field_name: batch[field_name][row_idx, physical_start:physical_end].reshape(-1)
                    for field_name in fields
                }
            )
            logical_lengths.append(logical_length)
            padded_lengths.append(padded_length)

    if not samples:
        raise ValueError("Dynamic CP cannot schedule a logical batch with no non-empty sequences.")
    return _UnpackedBatch(
        samples=samples,
        logical_lengths=logical_lengths,
        padded_lengths=padded_lengths,
    )


def _runtime_cp_size(sample_ids_by_rank: Sequence[Sequence[int]], rank: int) -> int:
    """Resolve and validate the scheduler's CP group size for one local rank."""
    local_ids = sample_ids_by_rank[rank]
    if not local_ids:
        raise ValueError("Dynamic CP scheduler produced an empty rank assignment.")
    memberships = [
        tuple(group_rank for group_rank, sample_ids in enumerate(sample_ids_by_rank) if sample_id in sample_ids)
        for sample_id in local_ids
    ]
    if any(membership != memberships[0] for membership in memberships[1:]):
        raise ValueError(
            "Dynamic CP scheduler assigned samples from different runtime CP groups to one local microbatch."
        )
    return len(memberships[0])


def _runtime_cp_group_histogram(sample_id_groups: Sequence[Sequence[Sequence[int]]]) -> dict[int, int]:
    """Count distinct runtime CP groups by size across a scheduled batch."""
    group_sizes: Counter[int] = Counter()
    for sample_ids_by_rank in sample_id_groups:
        memberships = {
            tuple(rank for rank, rank_ids in enumerate(sample_ids_by_rank) if sample_id in rank_ids)
            for rank_ids in sample_ids_by_rank
            for sample_id in rank_ids
        }
        group_sizes.update(len(membership) for membership in memberships)
    return dict(sorted(group_sizes.items()))


def _materialize_microbatch(
    samples_by_id: Mapping[int, Mapping[str, torch.Tensor]],
    sample_ids_by_rank: Sequence[Sequence[int]],
    *,
    rank: int,
    logical_lengths: Sequence[int],
    padded_lengths: Sequence[int],
    policy: DynamicCPBatchPolicy,
) -> dict[str, Any]:
    """Build one full THD microbatch before its runtime-CP partition."""
    sample_ids = sample_ids_by_rank[rank]
    local_cp_size = _runtime_cp_size(sample_ids_by_rank, rank)
    local_logical_lengths = [logical_lengths[sample_id] for sample_id in sample_ids]
    local_padded_lengths = [padded_lengths[sample_id] for sample_id in sample_ids]
    if local_cp_size > 1:
        invalid_lengths = [length for length in local_padded_lengths if length % (2 * local_cp_size) != 0]
        if invalid_lengths:
            raise ValueError(
                "Dynamic CP physical sequence lengths must be divisible by twice their runtime CP size; "
                f"CP={local_cp_size}, invalid lengths={invalid_lengths}."
            )

    total_length = sum(local_padded_lengths)
    first_sample = samples_by_id[sample_ids[0]]
    batch: dict[str, Any] = {"attention_mask": None}
    for field_name, pad_value in policy.sequence_field_pad_values.items():
        source = first_sample[field_name]
        batch[field_name] = torch.full((1, total_length), pad_value, dtype=source.dtype, device=source.device)
    batch["padding_mask"] = torch.ones(
        (1, total_length), dtype=torch.bool, device=first_sample[next(iter(policy.sequence_field_pad_values))].device
    )

    physical_offset = 0
    for sample_id, logical_length, padded_length in zip(sample_ids, local_logical_lengths, local_padded_lengths):
        sample = samples_by_id[sample_id]
        for field_name in policy.sequence_field_pad_values:
            value = sample[field_name]
            if value.numel() != logical_length:
                raise ValueError(
                    f"Dynamic CP field '{field_name}' for sample {sample_id} has {value.numel()} elements, "
                    f"expected {logical_length}."
                )
            batch[field_name][0, physical_offset : physical_offset + logical_length].copy_(value)
        batch["padding_mask"][0, physical_offset : physical_offset + logical_length] = False
        physical_offset += padded_length

    dev = batch[next(iter(policy.sequence_field_pad_values))].device
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(local_logical_lengths, dtype=torch.int64).cumsum(0).tolist()],
        dtype=torch.int32,
        device=dev,
    )
    cu_seqlens_padded = torch.tensor(
        [0, *torch.tensor(local_padded_lengths, dtype=torch.int64).cumsum(0).tolist()],
        dtype=torch.int32,
        device=dev,
    )
    max_seqlen = max(local_padded_lengths)
    batch.update(
        {
            "cu_seqlens_q": cu_seqlens,
            "cu_seqlens_kv": cu_seqlens,
            "cu_seqlens_q_padded": cu_seqlens_padded,
            "cu_seqlens_kv_padded": cu_seqlens_padded,
            "max_seqlen_q": max_seqlen,
            "max_seqlen_kv": max_seqlen,
            "pad_between_seqs": local_logical_lengths != local_padded_lengths,
            "local_cp_size": local_cp_size,
            "total_tokens": total_length,
        }
    )
    return batch


def prepare_dynamic_cp_batch(
    data_iterator: Iterator[Mapping[str, Any]] | None,
    *,
    num_microbatches: int,
    model_config: Any,
    pg_collection: Any,
    policy: DynamicCPBatchPolicy | None = None,
) -> DynamicCPBatch:
    """Schedule and materialize one already-selected logical global batch.

    The dataloader/framework remains responsible for global-batch selection.
    This function starts after that decision: it decomposes THD rows, delegates
    placement and opaque tensor transport to MCore, then reconstructs the local
    framework batch including labels and masks.
    """
    if data_iterator is None:
        raise ValueError("Dynamic CP requires a data iterator on every participating rank.")
    if num_microbatches < 1:
        raise ValueError("Dynamic CP requires at least one logical microbatch.")
    if getattr(model_config, "virtual_pipeline_model_parallel_size", None) not in (None, 1):
        raise ValueError("Bridge Dynamic CP materialization does not yet support virtual pipeline parallelism.")
    if pg_collection.pp.size() != 1:
        raise ValueError("Bridge Dynamic CP materialization currently requires pipeline parallel size 1.")

    policy = policy or DynamicCPBatchPolicy()
    global_batch = next(data_iterator)
    token_batch = global_batch.get("tokens")
    if not isinstance(token_batch, torch.Tensor) or token_batch.dim() != 2:
        raise ValueError("Dynamic CP requires a 2D 'tokens' tensor in the selected logical batch.")
    if token_batch.size(0) != num_microbatches:
        raise ValueError(
            "Bridge Dynamic CP currently requires micro_batch_size=1: selected batch has "
            f"{token_batch.size(0)} packed rows for {num_microbatches} logical microbatches."
        )
    unpacked = _unpack_thd_global_batch(global_batch, policy)
    if len(unpacked.samples) < 1:
        raise ValueError("Dynamic CP requires at least one sequence in the selected logical batch.")

    dev = torch.cuda.current_device()
    padded_lengths = torch.tensor(unpacked.padded_lengths, dtype=torch.int32, device=dev)
    logical_lengths = torch.tensor(unpacked.logical_lengths, dtype=torch.int32, device=dev)
    global_id_seqlens, global_ids_this_rank, offsets, global_padded_lengths = gather_global_sequence_lengths(
        padded_lengths, pg_collection.dp
    )
    _, logical_ids_this_rank, logical_offsets, global_logical_lengths = gather_global_sequence_lengths(
        logical_lengths, pg_collection.dp
    )
    if not torch.equal(global_ids_this_rank, logical_ids_this_rank) or not torch.equal(offsets, logical_offsets):
        raise RuntimeError("Dynamic CP logical and physical length gathers produced different sample IDs.")

    max_seqlen_per_rank = getattr(model_config, "max_seqlen_per_dp_cp_rank", None)
    if max_seqlen_per_rank is None:
        raise ValueError("Dynamic CP requires model.max_seqlen_per_dp_cp_rank.")
    dp_size = pg_collection.dp.size()
    cp_size = pg_collection.dp_cp.size() // dp_size
    scheduler = DefaultDynamicCPScheduler(
        max_seqlen_per_dp_cp_rank=max_seqlen_per_rank,
        cp_size=cp_size,
        dp_size=dp_size,
        microbatch_group_size_per_vp_stage=None,
        min_cp_size=getattr(model_config, "min_dynamic_context_parallel_size", 1),
    )
    sample_id_groups = scheduler.get_groups_and_subsamples(global_id_seqlens)
    scheduled_ids = {sample_id for microbatch in sample_id_groups for rank_ids in microbatch for sample_id in rank_ids}
    if scheduled_ids != set(range(len(global_id_seqlens))):
        raise RuntimeError(
            f"Dynamic CP scheduler covered sample IDs {sorted(scheduled_ids)}, "
            f"expected {list(range(len(global_id_seqlens)))}."
        )
    if pg_collection.dp_cp.rank() == 0:
        logger.info(
            "Dynamic CP scheduled %d sequences from %d logical packed rows into %d execution microbatches; "
            "runtime CP group histogram=%s.",
            len(global_id_seqlens),
            token_batch.size(0) * dp_size,
            len(sample_id_groups),
            _runtime_cp_group_histogram(sample_id_groups),
        )

    samples_by_id = reroute_tensor_fields_to_dcp_ranks(
        batch=unpacked.samples,
        fields=tuple(policy.sequence_field_pad_values),
        global_ids_this_rank=global_ids_this_rank,
        sample_id_groups=sample_id_groups,
        offsets=offsets,
        dp_group=pg_collection.dp,
        dp_cp_group=pg_collection.dp_cp,
    )
    dcp_rank = pg_collection.dp_cp.rank()
    microbatches = [
        _materialize_microbatch(
            samples_by_id,
            sample_ids_by_rank,
            rank=dcp_rank,
            logical_lengths=global_logical_lengths,
            padded_lengths=global_padded_lengths,
            policy=policy,
        )
        for sample_ids_by_rank in sample_id_groups
    ]
    return DynamicCPBatch(
        data_iterator=RerunDataIterator(iter(microbatches)),
        num_microbatches=len(microbatches),
        padded_token_sum=sum(global_padded_lengths),
        logical_seqlen_squared_sum=sum(length * length for length in global_logical_lengths),
    )
