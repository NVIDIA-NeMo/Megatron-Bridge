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

"""Context parallelism (CP) for ERNIE 5.5: zigzag D layers inside a contiguous stack.

Two CP partitions of a packed THD micro-batch of ``T`` rows exist in Megatron-Core:

* contiguous: rank ``r`` holds the packed rows ``[r * T / CP, (r + 1) * T / CP)``;
* zigzag: every document is cut into ``2 * CP`` equal chunks and rank ``r`` holds chunks ``r``
  and ``2 * CP - 1 - r`` of every document, in document order.

The H layers (Megatron-Core's DSv4 hybrid attention) accept only the contiguous partition, and
Megatron-Core forces it on the whole ``dsv4_hybrid`` stack. Everything Megatron-Core provides for
the D layer accepts only zigzag: AbsorbedMLA's packed RoPE, the DSA indexer's RoPE and the fused
CP>1 indexer scorer. A D layer therefore converts its input to zigzag with Megatron-Core dev's
module-local converter (#6387) and converts its output back
(:class:`~megatron.bridge.models.ernie.modeling_ernie55.latent_mqa_attention.Ernie55LatentMQASelfAttention`),
and its core attends from its local zigzag rows to the gathered keys of the whole pack
(:class:`~megatron.bridge.models.ernie.modeling_ernie55.latent_mqa_core.Ernie55LatentMQACoreAttention`).

This module holds what both need:

* :func:`load_cp_layout_api` finds the converter in ``megatron.core.context_parallel_layout``
  (Megatron-Core dev) or, in a build that moves it there, ``megatron.core.context_parallel.layout``. It also
  requires the other dev pieces the zigzag D layer calls: the ``dsa_layout`` host builders and the
  ``cp_partition_mode`` fields of ``TransformerConfig`` and ``PackedSeqParams``. Megatron-Core main
  has none of them, and its AbsorbedMLA would apply zigzag RoPE positions to contiguous rows without
  an error, so :func:`validate_ernie55_context_parallel_config` rejects CP > 1 there, and keeps
  rejecting it at config time if a port brings only some of the pieces.
* :func:`check_thd_cp_metadata` checks a micro-batch's packed metadata on the host.
* :func:`zigzag_cp_geometry` returns the global packed position of every local zigzag row and the
  permutation that puts the rank-ordered gathered keys into global order, built from host lists by
  Megatron-Core's ``dsa_layout`` helpers and cached on the micro-batch's ``PackedSeqParams`` (one
  entry, for the micro-batch's current boundaries).
* :func:`gather_cp_rows` all-gathers a per-row tensor over the CP group and reorders it.

A micro-batch at CP > 1 must satisfy (checked on the host, without a device-to-host
synchronization):

* packed THD input (``qkv_format="thd"``, batch 1); SBHD input is not supported;
* every padded document length is a multiple of ``2 * CP``, and the rank holds exactly
  ``cu_seqlens_padded[-1] / CP`` rows (no rows after the last document);
* Megatron-Core's ``prebuild_thd_cp_partition_routes`` (or ``finalize_packed_seq_params``) ran when
  the batch was built: it stores the conversion route and the host copies of ``cu_seqlens``
  (``thd_cp_host_cu_seqlens_q``/``_kv``) the layout is built from;
* the query and key document boundaries are identical (self attention);
* ``max_seqlen_q`` and ``max_seqlen_kv`` are at least the longest padded document.

:func:`~megatron.bridge.models.ernie.ernie55_context_parallel.build_ernie55_thd_cp_batch` builds such
micro-batches.
"""

from __future__ import annotations

import dataclasses
import functools
import importlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer.experimental_attention_variant import dsa_layout
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import Tensor

from megatron.bridge.models.ernie.modeling_ernie55 import mtp_loss


__all__ = [
    "CpLayoutApi",
    "ZigzagCpGeometry",
    "check_thd_cp_metadata",
    "gather_cp_rows",
    "load_cp_layout_api",
    "require_cp_layout_api",
    "validate_ernie55_context_parallel_config",
    "zigzag_cp_geometry",
]

# Where Megatron-Core keeps the module-local CP partition converter: dev's package (#6387), or the
# package a later build may move it to. The relocated name is tried first, so a build that keeps the old
# name only as a shim still resolves to the real module.
_CP_LAYOUT_MODULES = ("megatron.core.context_parallel.layout", "megatron.core.context_parallel_layout")
# Megatron-Core dev's dsa_layout builders that the zigzag D layer (zigzag_cp_geometry) and the batch
# builder (build_ernie55_thd_cp_batch) call on host cu_seqlens.
_DSA_LAYOUT_HOST_BUILDERS = (
    "build_packed_allgather_cp_query_positions_and_key_reorder_from_host",
    "build_packed_allgather_cp_local_positions_from_host",
)
# Megatron-Core dataclasses whose cp_partition_mode field (dev) the D layer reads and the converter sets.
_CP_PARTITION_MODE_HOLDERS = (TransformerConfig, PackedSeqParams)
# Attribute under which the zigzag geometry of a micro-batch is cached on its PackedSeqParams.
_GEOMETRY_CACHE_ATTR = "_ernie55_zigzag_cp_geometry"


@dataclass(frozen=True)
class CpLayoutApi:
    """Megatron-Core's module-local CP partition-mode API (dev #6387).

    Args:
        module: Name of the module the functions come from.
        convert_module_input_tensors_cp_partition_mode: Converts a module's input to a target
            partition and returns the converter for the way back.
        prebuild_thd_cp_partition_routes: Stores a packed micro-batch's conversion route and host
            ``cu_seqlens`` on its ``PackedSeqParams``.
    """

    module: str
    convert_module_input_tensors_cp_partition_mode: Callable[..., Any]
    prebuild_thd_cp_partition_routes: Callable[..., Any]


def _find_cp_layout_converter() -> CpLayoutApi | None:
    """Return the module-local CP partition converter of this Megatron-Core build, or ``None``."""
    for module_name in _CP_LAYOUT_MODULES:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        # Dev's megatron.core.context_parallel.layout is the unrelated HybridStack layout manager;
        # only a module with both functions is the converter.
        convert = getattr(module, "convert_module_input_tensors_cp_partition_mode", None)
        prebuild = getattr(module, "prebuild_thd_cp_partition_routes", None)
        if callable(convert) and callable(prebuild):
            return CpLayoutApi(module_name, convert, prebuild)
    return None


def _missing_cp_layout_support(converter: CpLayoutApi | None) -> list[str]:
    """Megatron-Core pieces ERNIE 5.5 context parallelism calls that this build lacks (empty on dev)."""
    missing = []
    if converter is None:
        missing.append(
            "convert_module_input_tensors_cp_partition_mode and prebuild_thd_cp_partition_routes (in "
            "megatron.core.context_parallel.layout or megatron.core.context_parallel_layout)"
        )
    missing += [
        f"dsa_layout.{name}" for name in _DSA_LAYOUT_HOST_BUILDERS if not callable(getattr(dsa_layout, name, None))
    ]
    for cls in _CP_PARTITION_MODE_HOLDERS:
        if "cp_partition_mode" not in {field.name for field in dataclasses.fields(cls)}:
            missing.append(f"{cls.__name__}.cp_partition_mode")
    return missing


@functools.lru_cache(maxsize=None)
def load_cp_layout_api() -> CpLayoutApi | None:
    """Return Megatron-Core's module-local CP partition converter, or ``None`` when this build cannot run ERNIE 5.5 CP.

    Besides the converter (dev #6387), the build must have every other Megatron-Core dev piece the
    zigzag D layer calls: the ``dsa_layout`` host builders of the zigzag positions and the key
    permutation, and the ``cp_partition_mode`` fields of ``TransformerConfig`` and
    ``PackedSeqParams``. A build with only some of them (e.g. a main port that brings the converter
    alone) is rejected here, at config time, instead of failing in the first forward.
    """
    converter = _find_cp_layout_converter()
    return None if _missing_cp_layout_support(converter) else converter


def require_cp_layout_api(what: str) -> CpLayoutApi:
    """Return the CP partition converter, or raise for Megatron-Core builds that cannot run ERNIE 5.5 CP.

    Args:
        what: Subject of the error message, e.g. ``"ERNIE 5.5 D layers"``.

    Raises:
        RuntimeError: If Megatron-Core lacks the module-local CP partition converter or another piece
            of the zigzag CP path (see :func:`load_cp_layout_api`).
    """
    api = load_cp_layout_api()
    if api is None:
        missing = _missing_cp_layout_support(_find_cp_layout_converter())
        raise RuntimeError(
            f"{what} need Megatron-Core's module-local CP partition converter (Megatron-Core dev #6387) and the "
            "zigzag CP helpers that come with it for context_parallel_size > 1. This Megatron-Core build lacks "
            f"{'; '.join(missing) if missing else 'them'} (Megatron-Core main before #7128): its AbsorbedMLA would "
            "apply zigzag RoPE positions to the contiguous rows of an ERNIE 5.5 stack without an error. Use "
            "context_parallel_size=1 or a Megatron-Core dev build."
        )
    return api


def validate_ernie55_context_parallel_config(config: Any, *, layer: str) -> None:
    """Raise ``ValueError`` if ``config`` is outside the ERNIE 5.5 context-parallel envelope.

    Called by the D- and H-layer validators. Nothing is checked at ``context_parallel_size == 1``.
    At CP > 1:

    * dynamic (hybrid) context parallelism is not supported;
    * Megatron-Core dev must have the module-local CP partition converter (#6387) and the other
      pieces of the zigzag CP path (:func:`load_cp_layout_api`);
    * multi-token prediction with ``ernie_per_row_loss`` needs Megatron-Core dev's grouped MTP roll
      (:data:`~megatron.bridge.models.ernie.modeling_ernie55.mtp_loss.PER_ROW_MTP_LOSS_SUPPORTED`);
    * ``cp_partition_mode`` must be ``"contiguous"`` (the ``dsv4_hybrid`` stack, converted to zigzag
      inside each D layer) or ``"zigzag"`` (a D-only ``"dsa"`` stack).

    Multi-token prediction is supported: with ``ernie_per_row_loss`` the model normalizes every MTP
    depth by the row's supervised tokens counted over the CP group
    (:mod:`~megatron.bridge.models.ernie.modeling_ernie55.mtp_loss`), as the LM loss does. A build
    without the grouped MTP roll (Megatron-Core main, also once #7128 brings the converter) would keep
    Megatron-Core's MTP loss, which divides each depth by this rank's own supervised tokens, so that
    combination is rejected instead of training with the wrong normalization.

    Args:
        config: Stack-level or per-layer transformer config.
        layer: Layer name for the error messages, ``"D"`` or ``"H"``.

    Raises:
        RuntimeError: If Megatron-Core lacks the CP partition converter or another piece of the
            zigzag CP path, or, for per-row MTP, the grouped MTP roll.
        ValueError: On the first other violated requirement.
    """
    if config.context_parallel_size <= 1:
        return
    prefix = f"ERNIE 5.5 {layer} layer"
    if getattr(config, "dynamic_context_parallel", False) or getattr(config, "hybrid_context_parallel", False):
        raise ValueError(f"{prefix}: dynamic (hybrid) context parallelism is not supported; use a static CP size.")
    require_cp_layout_api(f"{prefix}s")
    if (
        getattr(config, "mtp_num_layers", None)
        and getattr(config, "ernie_per_row_loss", False)
        and not mtp_loss.PER_ROW_MTP_LOSS_SUPPORTED
    ):
        raise RuntimeError(
            f"{prefix}s: the per-row MTP loss at context_parallel_size > 1 needs Megatron-Core dev's grouped MTP roll "
            "(roll_tensor(roll_context=...)) and token-aware MTP loss tracker, which this Megatron-Core build lacks. "
            "Its own MTP loss would divide every depth by this rank's supervised tokens instead of the row's. Use "
            "context_parallel_size=1, mtp_num_layers=None or a Megatron-Core dev build."
        )
    mode = getattr(config, "cp_partition_mode", None)
    if mode not in ("contiguous", "zigzag"):
        raise ValueError(f"{prefix}: cp_partition_mode must be 'contiguous' or 'zigzag' at CP > 1, got {mode!r}.")


def check_thd_cp_metadata(
    packed_seq_params: PackedSeqParams | None,
    *,
    cp_size: int,
    local_rows: int,
    expected_mode: str | None,
    need_route: bool,
    what: str,
) -> list[int]:
    """Check a packed micro-batch for context parallelism on the host; return its document boundaries.

    Args:
        packed_seq_params: The micro-batch's packed metadata.
        cp_size: Context-parallel size (> 1).
        local_rows: Number of packed rows this rank holds.
        expected_mode: Required ``packed_seq_params.cp_partition_mode``, or ``None`` to accept either.
        need_route: Require the prebuilt conversion route (``cp_partition_route``).
        what: Subject of the error messages.

    Returns:
        The compacted host ``cu_seqlens`` (``thd_cp_host_cu_seqlens_q``) of the padded documents.

    Raises:
        NotImplementedError: For SBHD input.
        ValueError: If the metadata violates a requirement in the module docstring.
    """
    if packed_seq_params is None or packed_seq_params.qkv_format != "thd":
        raise NotImplementedError(
            f"{what} support context parallelism only with packed THD input (qkv_format='thd'); got "
            f"{'SBHD input' if packed_seq_params is None else repr(packed_seq_params.qkv_format)}."
        )
    mode = getattr(packed_seq_params, "cp_partition_mode", None)
    if expected_mode is not None and mode != expected_mode:
        raise ValueError(f"{what} expect packed_seq_params.cp_partition_mode={expected_mode!r} here, got {mode!r}.")
    if need_route and getattr(packed_seq_params, "cp_partition_route", None) is None:
        raise ValueError(
            f"{what} need the micro-batch's CP partition route. Build the batch with "
            "megatron.core.context_parallel_layout.prebuild_thd_cp_partition_routes (or "
            "finalize_packed_seq_params), e.g. through build_ernie55_thd_cp_batch; Megatron-Core's fallback would "
            "rebuild it with a device-to-host synchronization in every layer."
        )
    host_q = getattr(packed_seq_params, "thd_cp_host_cu_seqlens_q", None)
    host_kv = getattr(packed_seq_params, "thd_cp_host_cu_seqlens_kv", None)
    if host_q is None or host_kv is None:
        raise ValueError(
            f"{what} need the host cu_seqlens that prebuild_thd_cp_partition_routes stores on the micro-batch's "
            "PackedSeqParams (thd_cp_host_cu_seqlens_q/_kv); building the zigzag layout without them would need a "
            "device-to-host synchronization. Prebuild the routes when building the batch."
        )
    host_q = [int(value) for value in host_q]
    if [int(value) for value in host_kv] != host_q:
        raise ValueError(
            f"{what} are self attention: the query and key document boundaries must be identical, got {host_q} and "
            f"{list(host_kv)}."
        )
    lengths = [end - start for start, end in zip(host_q[:-1], host_q[1:])]
    divisor = 2 * cp_size
    misaligned = [length for length in lengths if length % divisor != 0]
    if misaligned:
        raise ValueError(
            f"{what}: every padded document must be a multiple of 2 * context_parallel_size = {divisor} rows for the "
            f"zigzag partition, got document lengths {misaligned}. Pad each sample to a multiple of {divisor} "
            "(labels -100, positions continuing), e.g. with build_ernie55_thd_cp_batch."
        )
    if not host_q or host_q[-1] != local_rows * cp_size:
        raise ValueError(
            f"{what}: this rank holds {local_rows} rows, but the padded documents span "
            f"{host_q[-1] if host_q else 0} rows over context_parallel_size={cp_size}; at CP > 1 the pack must end with "
            "its last document (no rows after cu_seqlens_padded[-1])."
        )
    longest = max(lengths, default=0)
    for name in ("max_seqlen_q", "max_seqlen_kv"):
        value = getattr(packed_seq_params, name, None)
        if not isinstance(value, int) or value < longest:
            raise ValueError(
                f"{what} need packed_seq_params.{name} as a Python int of at least the longest padded document "
                f"({longest} rows), got {value!r}."
            )
    return host_q


@dataclass(frozen=True)
class ZigzagCpGeometry:
    """Where this rank's zigzag rows sit in the packed micro-batch.

    Args:
        query_positions: int64 ``[local_rows]`` global packed position of every local row.
        key_reorder: int64 ``[CP * local_rows]`` permutation that puts keys gathered in rank order
            (rank 0's local rows, then rank 1's, ...) into global packed order.
        cp_size: Context-parallel size.
        cp_rank: This rank's index in the CP group.
        num_rows: Rows of the whole packed micro-batch, ``CP * local_rows``.
    """

    query_positions: Tensor
    key_reorder: Tensor
    cp_size: int
    cp_rank: int
    num_rows: int


def zigzag_cp_geometry(
    packed_seq_params: PackedSeqParams,
    *,
    cp_group: torch.distributed.ProcessGroup,
    local_rows: int,
    device: torch.device,
    what: str,
) -> ZigzagCpGeometry:
    """Return the zigzag geometry of this rank's rows, built on the host and cached on the micro-batch.

    The micro-batch's ``cp_partition_mode`` must be ``"zigzag"``. Positions and the key permutation
    come from Megatron-Core's ``build_packed_allgather_cp_query_positions_and_key_reorder_from_host``
    (the layout Megatron-Core's DSA uses), so no device kernel or device-to-host copy runs; the
    result is cached on ``packed_seq_params``, so every D layer of the micro-batch (and its
    recomputation) reuses it. The cache holds one entry: a ``PackedSeqParams`` reused for a
    micro-batch with other boundaries replaces it instead of accumulating device tensors.

    Args:
        packed_seq_params: The micro-batch's packed metadata.
        cp_group: Context-parallel group (size > 1).
        local_rows: Number of packed rows this rank holds.
        device: Device of the returned tensors.
        what: Subject of the error messages.

    Returns:
        The geometry of this rank's rows.
    """
    cp_size, cp_rank = cp_group.size(), cp_group.rank()
    host_cu = check_thd_cp_metadata(
        packed_seq_params,
        cp_size=cp_size,
        local_rows=local_rows,
        expected_mode="zigzag",
        need_route=False,
        what=what,
    )
    device = torch.device(device)
    key = (cp_size, cp_rank, local_rows, str(device), tuple(host_cu))
    cached = getattr(packed_seq_params, _GEOMETRY_CACHE_ATTR, None)
    if cached is not None and cached[0] == key:
        return cached[1]
    positions, key_reorder = dsa_layout.build_packed_allgather_cp_query_positions_and_key_reorder_from_host(
        host_cu,
        host_cu,
        cp_size,
        cp_rank,
        device,
        local_output_size=local_rows,
        key_local_output_size=local_rows,
        global_output_size=local_rows * cp_size,
        query_cu_seqlens_cover_output=True,
        key_cu_seqlens_cover_output=True,
    )
    geometry = ZigzagCpGeometry(positions, key_reorder, cp_size, cp_rank, local_rows * cp_size)
    # One entry: every D layer of a micro-batch asks with the same key.
    setattr(packed_seq_params, _GEOMETRY_CACHE_ATTR, (key, geometry))
    return geometry


def gather_cp_rows(local: Tensor, *, geometry: ZigzagCpGeometry, cp_group: torch.distributed.ProcessGroup) -> Tensor:
    """All-gather a per-row tensor over the CP group and put its rows into global packed order.

    The backward reduce-scatters the gradient: every rank's contribution to a row reaches the rank
    that owns it, summed.

    Args:
        local: ``[local_rows, ...]`` tensor of this rank's zigzag rows.
        geometry: This rank's zigzag geometry.
        cp_group: Context-parallel group.

    Returns:
        ``[num_rows, ...]`` tensor of every packed row in global order.
    """
    gathered = gather_from_sequence_parallel_region(
        local.contiguous(), tensor_parallel_output_grad=True, group=cp_group
    )
    if gathered.size(0) != geometry.key_reorder.numel():
        raise RuntimeError(
            f"Gathered {gathered.size(0)} rows over the CP group, expected {geometry.key_reorder.numel()}."
        )
    return gathered.index_select(0, geometry.key_reorder)
