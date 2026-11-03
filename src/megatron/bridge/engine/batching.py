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
"""Default preprocessing and postprocessing blocks, plus the strict check at the Megatron boundary.

The default ``preprocess`` is ``group_into_microbatches`` -> ``assign_to_dp_ranks`` -> ``pack_sequences``
(or ``pad_sequence``) -> ``shard_for_cp``; the default ``postprocess`` is ``unshard_from_cp`` ->
``unpack_sequences`` -> ``all_gather_dp``. Each block is public so a custom pipeline can reuse the parts
it needs. Every DP replica runs the same number of microbatches (fillers are added) because
expert-parallel collectives may span DP replicas. Nothing here is model specific.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import torch

from megatron.bridge.data.packing.algorithms import first_fit
from megatron.bridge.data.packing.in_batch import build_mcore_thd_sequence_batch_from_rows
from megatron.bridge.engine.datum import COMMON_FIELDS, Datum, ForwardOutput, PreparedMicrobatch
from megatron.bridge.training.utils.packed_seq_utils import get_packed_seq_params, get_thd_cp_partition_indices


PAD_TOKEN_ID = 0  # a valid vocabulary id at padding slots; padding is excluded through padding_mask
_PACKED_METADATA_KEYS = (
    "cu_seqlens_q",
    "cu_seqlens_kv",
    "cu_seqlens_q_padded",
    "cu_seqlens_kv_padded",
    "max_seqlen_q",
    "max_seqlen_kv",
    "pad_between_seqs",
)


# ---- bookkeeping shared by the default pre/postprocessors --------------------------------------


@dataclass
class SequenceMapping:
    """Where one Datum's tokens live inside the full (un-sharded) microbatch."""

    index: int  # position in the caller's list
    offset: int
    length: int


@dataclass
class MicrobatchMapping:
    """Default pipeline bookkeeping stored in ``PreparedMicrobatch.mapping``."""

    segments: list[SequenceMapping]
    total_tokens: int  # un-sharded token count
    cp_indices: torch.Tensor  # [T_local] positions of this rank's tokens in the un-sharded microbatch
    filler: bool = False


@dataclass
class ScoredEntry:
    """Per-token values for one Datum at the given original positions (CPU tensors)."""

    index: int
    positions: torch.Tensor
    values: torch.Tensor


# ---- planning -----------------------------------------------------------------------------------


def sequence_pad_multiple(*, cp_size: int, tp_size: int, sequence_parallel: bool) -> int:
    """Per-sequence alignment required by the CP zigzag split and sequence parallelism."""
    multiple = 2 * cp_size if cp_size > 1 else 1
    if sequence_parallel and tp_size > 1:
        multiple = math.lcm(multiple, cp_size * tp_size)
    return multiple


def _padded_length(length: int, multiple: int) -> int:
    return -(-length // multiple) * multiple


def group_into_microbatches(datums: list[Datum], *, seq_length: int, pad_multiple: int) -> list[list[int]]:
    """First-fit Datum indices, in input order, into groups whose padded length is at most ``seq_length``.

    Uses Bridge's ``first_fit`` (``megatron.bridge.data.packing.algorithms``). Input order is kept so the
    plan is deterministic and identical on every rank.

    TODO(packing): expose the algorithm choice (``first_fit_decreasing``, ``first_fit_shuffle``) through
    the recipe for better bin utilization.
    """
    padded_lengths = [_padded_length(datum.num_tokens, pad_multiple) for datum in datums]
    for index, (datum, padded) in enumerate(zip(datums, padded_lengths)):
        if padded > seq_length:
            raise ValueError(
                f"datums[{index}] has {datum.num_tokens} tokens ({padded} padded), above seq_length {seq_length}."
            )
    return first_fit(list(range(len(datums))), seq_length, item_lengths=padded_lengths)


def assign_to_dp_ranks(groups: list[list[int]], dp_size: int) -> list[list[list[int] | None]]:
    """Deal groups round-robin to DP replicas; ``None`` marks a filler slot that equalizes counts."""
    per_rank: list[list[list[int] | None]] = [list(groups[rank::dp_size]) for rank in range(dp_size)]
    count = max(len(ranks) for ranks in per_rank)
    for ranks in per_rank:
        ranks.extend([None] * (count - len(ranks)))
    return per_rank


# ---- materialization -----------------------------------------------------------------------------


def _rows(datums: list[Datum], indices: list[int], extra_keys: list[str]) -> list[dict[str, torch.Tensor]]:
    rows = []
    for index in indices:
        datum = datums[index]
        n = datum.num_tokens
        row = {
            "input_ids": datum.input_ids.to(torch.int64),
            "position_ids": torch.arange(n, dtype=torch.int64, device=datum.input_ids.device),
            "labels": datum.target_ids.to(torch.int64),
            "loss_mask": datum.loss_mask.to(torch.float32),
        }
        for name in extra_keys:
            if name in COMMON_FIELDS:
                row[name] = getattr(datum, name).to(torch.float32)
            else:
                row[name] = datum.loss_inputs[name]
        rows.append(row)
    return rows


def _extra_keys(datums: list[Datum], indices: list[int]) -> list[str]:
    """Common fields and custom ``loss_inputs`` keys to pack; ``check_datums`` made them uniform."""
    first = datums[indices[0]]
    common = [name for name in COMMON_FIELDS if getattr(first, name) is not None]
    return common + sorted(first.loss_inputs)


def pack_sequences(
    datums: list[Datum],
    indices: list[int],
    *,
    seq_length: int,
    pad_multiple: int,
    device: torch.device,
    pad_to_max_length: bool = False,
) -> PreparedMicrobatch:
    """Concatenate the selected Datums into one THD row (``[1, T]``, not yet CP-sharded).

    Each sequence is padded to ``pad_multiple``; padding carries ``PAD_TOKEN_ID`` inputs and targets,
    ``loss_mask`` 0, ``padding_mask`` True and 0 in every optional field. With ``pad_to_max_length`` the
    pack is padded to exactly ``seq_length`` so every microbatch has the same physical width (fixed
    kernel shapes, no per-shape autotuning; what the offline-packed SFT recipes do). The mapping records
    each Datum's offset and length.
    """
    extra_keys = _extra_keys(datums, indices)
    packed = build_mcore_thd_sequence_batch_from_rows(
        _rows(datums, indices, extra_keys),
        sequence_length=seq_length,
        pad_to_multiple_of=pad_multiple,
        pad_to_max_length=pad_to_max_length,
        pad_token_id=PAD_TOKEN_ID,
        ignore_index=PAD_TOKEN_ID,
        sequence_tensor_pad_values={key: 0.0 for key in extra_keys},
        emit_padding_mask=True,
    )
    total_tokens = int(packed["total_tokens"])
    cu_padded = packed.get("cu_seqlens_q_padded", packed["cu_seqlens_q"])
    segments = [
        SequenceMapping(index, int(cu_padded[slot].item()), datums[index].num_tokens)
        for slot, index in enumerate(indices)
    ]
    metadata = {key: packed[key] for key in _PACKED_METADATA_KEYS if key in packed}
    for key, value in metadata.items():
        if isinstance(value, torch.Tensor) and value.dim() > 0:
            metadata[key] = value.to(device)

    def _dev(key: str) -> torch.Tensor:
        return packed[key].to(device)

    return PreparedMicrobatch(
        input_ids=_dev("input_ids"),
        target_ids=_dev("labels"),
        position_ids=_dev("position_ids"),
        loss_mask=_dev("loss_mask"),
        padding_mask=_dev("padding_mask"),
        packed_seq_params=get_packed_seq_params(metadata),
        **{name: _dev(name) for name in COMMON_FIELDS if name in extra_keys},
        loss_inputs={key: _dev(key) for key in extra_keys if key not in COMMON_FIELDS},
        sample_inputs=[datums[index].sample_inputs for index in indices],
        mapping=MicrobatchMapping(
            segments=segments,
            total_tokens=total_tokens,
            cp_indices=torch.arange(total_tokens, dtype=torch.int64, device=device),
        ),
    )


def pad_sequence(datum: Datum, index: int, *, pad_multiple: int, device: torch.device) -> PreparedMicrobatch:
    """One Datum as a padded ``[1, S]`` row for the dense (bshd) layout, not yet CP-sharded.

    TODO(bshd): pad to ``pad_multiple`` with ``PAD_TOKEN_ID`` / padding_mask True, no PackedSeqParams,
    mapping with a single ``SequenceMapping(index, 0, L)``; ``shard_for_cp`` then applies unchanged.
    """
    raise NotImplementedError("batch_layout='bshd' is not implemented yet; use 'thd'.")


def filler_microbatch(
    *, pad_multiple: int, device: torch.device, layout: str, width: int | None = None
) -> PreparedMicrobatch:
    """A zero-contribution microbatch so every DP replica runs the same number of forward passes.

    ``width`` sets the physical token count (default ``pad_multiple``); pass ``seq_length`` to keep the
    fixed-width shape of the real microbatches.
    """
    width = width or pad_multiple
    datum = Datum(
        input_ids=torch.full((width,), PAD_TOKEN_ID, dtype=torch.int64),
        target_ids=torch.full((width,), PAD_TOKEN_ID, dtype=torch.int64),
        loss_mask=torch.zeros(width),
    )
    if layout == "thd":
        microbatch = pack_sequences([datum], [0], seq_length=width, pad_multiple=pad_multiple, device=device)
    elif layout == "bshd":
        microbatch = pad_sequence(datum, 0, pad_multiple=pad_multiple, device=device)
    else:
        raise ValueError(f"Unknown batch_layout {layout!r}.")
    microbatch.padding_mask = torch.ones_like(microbatch.padding_mask)
    microbatch.mapping.filler = True
    return microbatch


def shard_for_cp(
    microbatch: PreparedMicrobatch,
    cp_group: torch.distributed.ProcessGroup | None,
    *,
    hybrid_model: bool = False,
) -> PreparedMicrobatch:
    """Keep this CP rank's zigzag slice of every token-aligned tensor; metadata keeps global boundaries.

    TODO(dynamic-cp): support a per-microbatch CP degree. The planner would pick ``local_cp_size`` per
    microbatch (e.g. short packs on fewer ranks), pad sequences to ``2 x dp x cp`` so every group size
    divides, shard with that microbatch's ``cp_group``, and record both on ``PackedSeqParams``
    (``local_cp_size`` / ``cp_group``) and in the mapping so ``unshard_from_cp`` gathers over the same
    group. Requires MCore with ``cp_partition_mode`` support and the recipe's dynamic-CP flag.
    """
    mapping: MicrobatchMapping = microbatch.mapping
    if cp_group is None or cp_group.size() == 1:
        indices = mapping.cp_indices
    elif microbatch.packed_seq_params is not None:
        params = microbatch.packed_seq_params
        cu = params.cu_seqlens_q_padded if params.cu_seqlens_q_padded is not None else params.cu_seqlens_q
        indices = get_thd_cp_partition_indices(
            cu, total_tokens=mapping.total_tokens, cp_group=cp_group, device=microbatch.input_ids.device
        )
    else:
        # TODO(bshd): MCore get_batch_on_this_cp_rank zigzag split of the dense [1, S] row.
        raise NotImplementedError("CP sharding for batch_layout='bshd' is not implemented yet.")
    local = {name: tensor.index_select(1, indices) for name, tensor in microbatch.token_tensors().items()}
    params = microbatch.packed_seq_params
    if params is not None and hybrid_model:
        # SSM/GDN hybrids derive seq_idx from total_tokens (CP-local count) and the padded boundaries.
        params = dataclasses.replace(params, total_tokens=int(indices.numel()))
    return dataclasses.replace(
        microbatch,
        **{name: local[name] for name in ("input_ids", "target_ids", "position_ids", "loss_mask", "padding_mask")},
        **{name: local[name] for name in COMMON_FIELDS if name in local},
        loss_inputs={key: local[key] for key in microbatch.loss_inputs},
        packed_seq_params=params,
        mapping=dataclasses.replace(mapping, cp_indices=indices),
    )


# ---- the strict Megatron boundary ----------------------------------------------------------------


def check_microbatches(
    microbatches: list[PreparedMicrobatch],
    *,
    layout: str,
    vocab_size: int,
    dp_group: torch.distributed.ProcessGroup | None,
) -> None:
    """Validate prepared microbatches before they reach the schedule.

    Raises:
        ValueError: On any contract violation, naming the microbatch and field.
    """
    for slot, mb in enumerate(microbatches):
        where = f"microbatch[{slot}]"
        tensors = mb.token_tensors()
        shape = tuple(mb.input_ids.shape)
        if len(shape) != 2 or (layout == "thd" and shape[0] != 1):
            raise ValueError(f"{where}.input_ids must be [1, T_local] for thd / [B, S_local] for bshd, got {shape}.")
        for name, tensor in tensors.items():
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"{where}.{name} must be a tensor.")
            if tensor.device.type != "cuda":
                raise ValueError(f"{where}.{name} must be on CUDA, got {tensor.device}.")
            if tuple(tensor.shape) != shape:
                raise ValueError(f"{where}.{name} has shape {tuple(tensor.shape)}, expected {shape}.")
            if not tensor.is_contiguous():
                raise ValueError(f"{where}.{name} must be contiguous.")
        for name in ("input_ids", "target_ids", "position_ids"):
            if tensors[name].dtype != torch.int64:
                raise ValueError(f"{where}.{name} must be int64.")
        if mb.padding_mask.dtype != torch.bool:
            raise ValueError(f"{where}.padding_mask must be bool.")
        if mb.loss_mask.dtype != torch.float32 or not bool(((mb.loss_mask == 0) | (mb.loss_mask == 1)).all()):
            raise ValueError(f"{where}.loss_mask must be a binary float32 mask.")
        for name in COMMON_FIELDS:
            if name in tensors and tensors[name].dtype != torch.float32:
                raise ValueError(f"{where}.{name} must be float32.")
        for name in ("input_ids", "target_ids"):
            if bool((tensors[name] < 0).any()) or bool((tensors[name] >= vocab_size).any()):
                raise ValueError(f"{where}.{name} must hold valid vocabulary ids in [0, {vocab_size}).")
        if (mb.packed_seq_params is not None) != (layout == "thd"):
            raise ValueError(f"{where}.packed_seq_params must be set exactly for batch_layout='thd'.")
        if mb.packed_seq_params is not None:
            params = mb.packed_seq_params
            if params.qkv_format != "thd":
                raise ValueError(f"{where}.packed_seq_params.qkv_format must be 'thd'.")
            for name in ("cu_seqlens_q", "cu_seqlens_kv", "cu_seqlens_q_padded", "cu_seqlens_kv_padded"):
                cu = getattr(params, name, None)
                if cu is not None and (cu.dtype != torch.int32 or cu.device.type != "cuda"):
                    raise ValueError(f"{where}.packed_seq_params.{name} must be an int32 CUDA tensor.")
    if dp_group is not None and dp_group.size() > 1:
        counts = torch.tensor([len(microbatches)], device="cuda")
        gathered = [torch.empty_like(counts) for _ in range(dp_group.size())]
        torch.distributed.all_gather(gathered, counts, group=dp_group)
        if len({int(c.item()) for c in gathered}) != 1:
            raise ValueError(f"DP replicas prepared different microbatch counts: {[int(c) for c in gathered]}.")


# ---- postprocessing blocks ---------------------------------------------------------------------


def unshard_from_cp(
    microbatch: PreparedMicrobatch, values: torch.Tensor, cp_group: torch.distributed.ProcessGroup | None
) -> torch.Tensor:
    """Reassemble per-token values for the un-sharded microbatch (``[total_tokens]``) from all CP ranks.

    TODO(dynamic-cp): gather over the microbatch's own ``cp_group`` from the mapping instead of the static one.
    """
    mapping: MicrobatchMapping = microbatch.mapping
    full = torch.empty(mapping.total_tokens, dtype=values.dtype, device=values.device)
    if cp_group is None or cp_group.size() == 1:
        full[mapping.cp_indices] = values
        return full
    world = cp_group.size()
    gathered_values = [torch.empty_like(values) for _ in range(world)]
    gathered_indices = [torch.empty_like(mapping.cp_indices) for _ in range(world)]
    torch.distributed.all_gather(gathered_values, values.contiguous(), group=cp_group)
    torch.distributed.all_gather(gathered_indices, mapping.cp_indices.contiguous(), group=cp_group)
    full[torch.cat(gathered_indices)] = torch.cat(gathered_values)
    return full


def unpack_sequences(microbatch: PreparedMicrobatch, values: torch.Tensor, *, local_only: bool) -> list[ScoredEntry]:
    """Slice per-token values back into per-Datum entries.

    ``local_only=False`` expects un-sharded values (``[total_tokens]``) and returns every position of
    every Datum; ``local_only=True`` expects this rank's values (``[T_local]``) and returns only the
    positions it holds. Padding is dropped; fillers yield nothing.
    """
    mapping: MicrobatchMapping = microbatch.mapping
    if mapping.filler:
        return []
    entries: list[ScoredEntry] = []
    for segment in mapping.segments:
        if local_only:
            where = mapping.cp_indices
            mask = (where >= segment.offset) & (where < segment.offset + segment.length)
            positions = where[mask] - segment.offset
            picked = values[mask]
            order = torch.argsort(positions)
            positions, picked = positions[order], picked[order]
        else:
            positions = torch.arange(segment.length, dtype=torch.int64, device=values.device)
            picked = values[segment.offset : segment.offset + segment.length]
        entries.append(ScoredEntry(segment.index, positions.cpu(), picked.float().cpu()))
    return entries


def all_gather_dp(entries: list[ScoredEntry], dp_group: torch.distributed.ProcessGroup | None) -> list[ScoredEntry]:
    """Collect every DP replica's entries on every rank."""
    if dp_group is None or dp_group.size() == 1:
        return entries
    gathered: list[list[ScoredEntry] | None] = [None] * dp_group.size()
    torch.distributed.all_gather_object(gathered, entries, group=dp_group)
    return [entry for shard in gathered if shard for entry in shard]


# ---- default pipelines ---------------------------------------------------------------------------


def default_preprocess(
    datums: list[Datum],
    *,
    layout: str,
    seq_length: int,
    pad_multiple: int,
    dp_group: torch.distributed.ProcessGroup | None,
    cp_group: torch.distributed.ProcessGroup | None,
    device: torch.device,
    hybrid_model: bool = False,
    pad_to_max_length: bool = False,
) -> list[PreparedMicrobatch]:
    """Global ``list[Datum]`` -> this rank's CP-sharded microbatches.

    1. Group Datums into microbatches. Every rank sees the same list, so the plan is identical
       everywhere without communication. THD: first-fit packs under the token budget; BSHD: one
       Datum per forward pass.
    2. Deal the groups round-robin to DP replicas and keep only this replica's share. Counts are
       equalized with fillers because EP collectives span DP replicas.
    3. Materialize each full (un-sharded) microbatch: tokens, targets, positions, masks, optional
       fields and PackedSeqParams, with per-sequence padding; the mapping records each Datum's slot.
       With ``pad_to_max_length`` every THD microbatch (fillers included) is padded to ``seq_length``
       so kernels see one shape; otherwise widths follow the content.
    4. Keep this CP rank's zigzag slice of every token-aligned tensor (metadata stays global) and
       remember the slice in ``mapping.cp_indices`` for the postprocessor.

    TODO(dynamic-cp): steps 1 and 4 assume the static CP degree; see ``shard_for_cp``.
    """
    if layout == "thd":
        groups = group_into_microbatches(datums, seq_length=seq_length, pad_multiple=pad_multiple)
    elif layout == "bshd":
        groups = [[index] for index in range(len(datums))]
    else:
        raise ValueError(f"Unknown batch_layout {layout!r}.")
    dp_size = dp_group.size() if dp_group is not None else 1
    dp_rank = dp_group.rank() if dp_group is not None else 0
    microbatches = []
    filler_width = seq_length if pad_to_max_length else None
    for group in assign_to_dp_ranks(groups, dp_size)[dp_rank]:
        if group is None:
            mb = filler_microbatch(pad_multiple=pad_multiple, device=device, layout=layout, width=filler_width)
        elif layout == "thd":
            mb = pack_sequences(
                datums,
                group,
                seq_length=seq_length,
                pad_multiple=pad_multiple,
                device=device,
                pad_to_max_length=pad_to_max_length,
            )
        else:
            mb = pad_sequence(datums[group[0]], group[0], pad_multiple=pad_multiple, device=device)
        microbatches.append(shard_for_cp(mb, cp_group, hybrid_model=hybrid_model))
    return microbatches


def default_postprocess(
    microbatches: list[PreparedMicrobatch],
    logprobs: list[torch.Tensor],
    *,
    gather_output: bool,
    cp_group: torch.distributed.ProcessGroup | None,
    dp_group: torch.distributed.ProcessGroup | None,
) -> ForwardOutput:
    """Per-microbatch ``[T_local]`` logprobs -> ``ForwardOutput`` (global in input order, or local shards).

    Inverse of ``default_preprocess``, driven by the mapping it left behind. With ``gather_output``:
    1. undo the CP split (all-gather values and indices across the CP group);
    2. cut each full microbatch into per-Datum entries by segment offset/length, dropping padding;
    3. collect every DP replica's entries and restore the caller's input order.
    Without it, keep only the positions this CP rank holds, mapped to Datum positions, and report
    them with ``sample_indices`` / ``token_positions``. TP needs no step: every TP rank already holds
    identical logprobs.
    """
    entries: list[ScoredEntry] = []
    for microbatch, values in zip(microbatches, logprobs):
        if gather_output:
            entries.extend(
                unpack_sequences(microbatch, unshard_from_cp(microbatch, values, cp_group), local_only=False)
            )
        else:
            entries.extend(unpack_sequences(microbatch, values, local_only=True))
    if gather_output:
        entries = sorted(all_gather_dp(entries, dp_group), key=lambda entry: entry.index)
        return ForwardOutput(logprobs=[entry.values for entry in entries])
    return ForwardOutput(
        logprobs=[entry.values for entry in entries],
        sample_indices=[entry.index for entry in entries],
        token_positions=[entry.positions for entry in entries],
    )
