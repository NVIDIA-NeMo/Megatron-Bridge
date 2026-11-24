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

"""ERNIE 5.5 training-step pieces for context parallelism: per-row LM loss and packed THD batches.

Both follow the reference implementation's conventions and are standalone, so the ERNIE 5.5
training step can wire them in:

* :func:`ernie55_per_row_lm_loss` is the per-row LM loss: the row is the micro-batch (one packed THD
  row), its summed token loss is divided by its supervised-token count, counted over the
  context-parallel group, and the training loss is the mean over micro-batches. It returns
  ``CP * S / N`` with ``num_tokens = 1`` for Megatron-Core's three-tuple loss contract, because the
  schedule then divides only by the number of micro-batches and DDP averages the gradients over
  DP x CP. The model config must set
  :attr:`~megatron.bridge.models.ernie.ernie55_config.Ernie55TransformerConfig.ernie_per_row_loss`
  (and ``calculate_per_token_loss=False``).
* :func:`build_ernie55_thd_cp_batch` packs samples into one THD micro-batch: every sample is padded to
  a multiple of ``2 * CP`` rows (pad tokens with label ``-100`` and continuing positions), the
  documents' boundaries become one ``cu_seqlens`` tensor shared by queries and keys, the CP
  partition routes are prebuilt, and the rows of this CP rank are sliced out. When the H layers run in the
  zigzag partition (Megatron-Core's ``dsv4_cp_zigzag``), every chunk is also padded to whole compression blocks
  and to the H layers' window (:func:`ernie55_cp_zigzag_chunk_rows`).

The MTP depths get the same per-row normalization inside the model
(:mod:`~megatron.bridge.models.ernie.modeling_ernie55.mtp_loss`); the training step only selects the LM loss.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.experimental_attention_variant import dsa_layout
from megatron.core.utils import get_pg_rank, get_pg_size
from torch import Tensor

from megatron.bridge.models.ernie.modeling_ernie55.context_parallel import (
    prebuild_zigzag_cp_geometry,
    require_cp_layout_api,
)
from megatron.bridge.models.ernie.modeling_ernie55.ops import record_pack_boundaries
from megatron.bridge.training.losses import validate_local_loss


__all__ = [
    "IGNORE_LABEL",
    "Ernie55ThdBatch",
    "build_ernie55_thd_cp_batch",
    "ernie55_cp_min_local_rows",
    "ernie55_cp_zigzag_chunk_rows",
    "ernie55_per_row_lm_loss",
    "uses_h_layer_zigzag",
]

IGNORE_LABEL = -100
"""Label of tokens without a loss (padding, prompts)."""


def ernie55_per_row_lm_loss(
    loss_mask: Tensor,
    output_tensor: Tensor,
    *,
    cp_group: torch.distributed.ProcessGroup | None,
    check_for_nan_in_loss: bool = True,
    check_for_spiky_loss: bool = False,
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    """Per-row LM loss of one micro-batch, normalized like the reference implementation.

    The reference's row is the micro-batch: a packed THD micro-batch (``b = 1``, the only layout
    at CP > 1) is one row, and the samples of an SBHD micro-batch with ``b > 1`` are pooled into
    one row the same way. With ``S`` the micro-batch's summed token loss on this CP rank and ``N`` its
    supervised tokens summed over the CP group (clamped at 1), the function returns ``CP * S / N``
    with ``num_tokens = 1``: the schedule divides the loss by ``num_tokens`` and the number of
    micro-batches, and DDP averages the gradients over DP x CP, so the gradient is that of the mean
    over micro-batches of each micro-batch's token mean, taken over all of its CP ranks' tokens. A CP
    rank whose tokens are all masked contributes 0, never NaN.

    The reported ``[S / N, 1 / CP]`` sums over DP x CP and the micro-batches to the total of the
    micro-batches' token means and their number, so the logged value is the mean row loss.

    The training step must use this loss whenever the model config sets ``ernie_per_row_loss``:
    Bridge's default ``masked_next_token_loss`` would average each rank's own token mean instead,
    without an error.

    Args:
        loss_mask: ``[b, s]`` (or ``[s]``) local loss mask of the micro-batch, 1 for supervised tokens.
        output_tensor: Per-token losses of the same tokens, ``loss_mask.numel()`` elements.
        cp_group: Context-parallel group, or ``None`` without context parallelism.
        check_for_nan_in_loss: Run the rerun state machine's NaN/Inf check on the local loss.
        check_for_spiky_loss: Run the rerun state machine's spiky-loss check on the local loss.

    Returns:
        ``(loss, num_tokens, {"lm loss": reporting})``: the scalar fp32 loss, an int ``1`` and the
        two-element fp32 reporting tensor.
    """
    mask = loss_mask.float()
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    if mask.ndim != 2:
        raise ValueError(f"loss_mask must be [b, s] or [s], got shape {tuple(loss_mask.shape)}.")
    if output_tensor.numel() != mask.numel():
        raise ValueError(
            f"output_tensor has {output_tensor.numel()} losses, but loss_mask covers {mask.numel()} tokens."
        )
    losses = output_tensor.float().reshape(mask.shape)
    loss_sum = (losses * mask).sum()
    row_tokens = mask.sum().detach().clone()
    cp_size = get_pg_size(cp_group)
    if cp_size > 1:
        torch.distributed.all_reduce(row_tokens, group=cp_group)
    row_tokens = row_tokens.clamp(min=1.0)
    loss = loss_sum * (cp_size / row_tokens)
    validate_local_loss(loss, check_for_nan_in_loss=check_for_nan_in_loss, check_for_spiky_loss=check_for_spiky_loss)
    num_tokens = torch.ones((), dtype=torch.int, device=loss.device)
    reporting = torch.stack([loss_sum.detach() / row_tokens, loss.new_tensor(1.0 / cp_size)])
    return loss, num_tokens, {"lm loss": reporting}


@dataclass
class Ernie55ThdBatch:
    """One packed THD micro-batch: this CP rank's rows plus the metadata of the whole pack.

    Args:
        tokens: ``[1, rows]`` token ids.
        labels: ``[1, rows]`` targets, ``-100`` where there is no loss (padding included).
        loss_mask: ``[1, rows]`` fp32, ``labels != -100``.
        position_ids: ``[1, rows]`` positions inside each sample, continuing through its padding.
        packed_seq_params: Metadata of the whole pack: one int32 ``cu_seqlens`` tensor of the padded
            samples in all four ``cu_seqlens`` fields, ``max_seqlen_q == max_seqlen_kv``, and at
            CP > 1 the CP partition mode and the prebuilt partition routes.
    """

    tokens: Tensor
    labels: Tensor
    loss_mask: Tensor
    position_ids: Tensor
    packed_seq_params: PackedSeqParams


def _padded_length(length: int, multiple: int) -> int:
    return int(math.ceil(length / multiple)) * multiple


def ernie55_cp_min_local_rows(config: object) -> int:
    """Fewest rows a CP rank may hold under the contiguous partition of an ERNIE 5.5 stack.

    At CP > 1 every H layer (Megatron-Core's DSv4 hybrid attention with a nonzero compress ratio) receives the
    ``D_window = max(csa_window_size, d_comp)`` hidden rows left of its block from the previous CP rank
    (``exchange_cp_boundary_hidden``; ``d_comp`` is 8 for ratio 4, else the ratio), and refuses a rank with fewer
    than ``D_window`` local rows. A stack without such layers has no minimum.

    Args:
        config: Model (transformer) config with ``csa_compress_ratios`` and ``csa_window_size``.

    Returns:
        The minimum local row count, ``1`` when there is none.
    """
    ratios = _compress_ratios(config)
    if not ratios:
        return 1
    d_comp = max(8 if ratio == 4 else ratio for ratio in ratios)
    return max(int(getattr(config, "csa_window_size", 0) or 0), d_comp)


def _compress_ratios(config: object) -> list[int]:
    """The compression ratios of the H layers (``csa_compress_ratios`` entries above 1)."""
    return [int(ratio) for ratio in (getattr(config, "csa_compress_ratios", None) or []) if int(ratio) > 1]


def uses_h_layer_zigzag(config: object, *, cp_size: int | None = None) -> bool:
    """Whether the H layers of an ERNIE 5.5 stack run context parallelism in the zigzag partition.

    That is the case at a context-parallel size above 1 with ``cp_partition_mode="zigzag"`` and H layers in the stack
    (a ``csa_compress_ratios`` entry above 1). The model config accepts it only with Megatron-Core's
    ``dsv4_cp_zigzag`` (:class:`~megatron.bridge.models.ernie.ernie55_config.Ernie55ModelConfig`); a D-only stack in the
    zigzag partition has no H layer and is not affected.

    Args:
        config: Model (transformer) config.
        cp_size: Context-parallel size; defaults to ``config.context_parallel_size``.

    Returns:
        ``True`` for the H layers' zigzag partition.
    """
    if cp_size is None:
        cp_size = getattr(config, "context_parallel_size", 1) or 1
    return cp_size > 1 and getattr(config, "cp_partition_mode", None) == "zigzag" and bool(_compress_ratios(config))


def ernie55_cp_zigzag_chunk_rows(config: object) -> tuple[int, int]:
    """Row multiple and minimum of a zigzag chunk for the single-document H fast path.

    In the zigzag partition every document is cut into ``2 * CP`` chunks, and rank ``r`` holds chunks ``r`` and
    ``2 * CP - 1 - r``. Megatron-Core's zigzag H layers (``dsv4_cp_zigzag``) need every chunk to be whole compression
    blocks (a multiple of each H ratio) and at least ``D_window = max(csa_window_size, d_comp)`` rows, so that each of
    the rank's two boundary halos comes from one neighbour (:func:`ernie55_cp_min_local_rows` has the same
    ``D_window``). A stack without H layers has neither requirement.

    Args:
        config: Model (transformer) config with ``csa_compress_ratios`` and ``csa_window_size``.

    Returns:
        ``(chunk_multiple, min_chunk_rows)``: the least common multiple of the H ratios, and ``D_window``;
        ``(1, 1)`` without H layers.
    """
    ratios = _compress_ratios(config)
    if not ratios:
        return 1, 1
    return math.lcm(*ratios), ernie55_cp_min_local_rows(config)


def _zigzag_padded_length(length: int, *, cp_size: int, chunk_multiple: int, min_chunk_rows: int) -> int:
    """Smallest zigzag-partitionable length of at least ``length``: ``2 * CP`` chunks of a multiple of
    ``chunk_multiple`` rows, each at least ``min_chunk_rows``."""
    chunk = max(int(math.ceil(length / (2 * cp_size))), min_chunk_rows)
    return 2 * cp_size * _padded_length(chunk, chunk_multiple)


def build_ernie55_thd_cp_batch(
    samples: Sequence[tuple[Tensor, Tensor]],
    *,
    pad_token_id: int,
    cp_group: torch.distributed.ProcessGroup | None,
    cp_partition_mode: str = "contiguous",
    max_seqlen: int | None = None,
    min_local_rows: int = 1,
    zigzag_chunk_multiple: int = 1,
    zigzag_min_chunk_rows: int = 1,
    device: torch.device | str | None = None,
    with_key_reorder_inverse: bool = False,
) -> Ernie55ThdBatch:
    """Pack samples into one THD micro-batch and slice this CP rank's rows, as the reference does.

    Every sample is padded to a multiple of ``2 * CP`` rows (``1`` without context parallelism),
    the zigzag partition's requirement and the reference's padding rule: pad tokens get
    ``pad_token_id``, label ``-100`` and positions that continue the sample's own, so each pad
    belongs to its sample's document. The padded samples are concatenated; one int32 ``cu_seqlens``
    tensor describes their boundaries and fills ``cu_seqlens_q``, ``cu_seqlens_kv`` and both padded
    fields (Megatron-Core's fused zigzag indexer scorer needs the query and key boundaries to be the
    same tensor). At CP > 1 the partition routes and host ``cu_seqlens`` are prebuilt with
    Megatron-Core's ``prebuild_thd_cp_partition_routes``, so no layer synchronizes the device to
    build them, and so is the D layers' zigzag geometry
    (:func:`~megatron.bridge.models.ernie.modeling_ernie55.context_parallel.prebuild_zigzag_cp_geometry`), so no
    layer copies it to the device in its forward. At CP = 1 the host boundaries are noted on the metadata
    (:func:`~megatron.bridge.models.ernie.modeling_ernie55.ops.record_pack_boundaries`). The rank keeps its rows of
    ``cp_partition_mode``:

    * ``"contiguous"`` (the DSv4 hybrid stack): rows ``[r * T / CP, (r + 1) * T / CP)``;
    * ``"zigzag"`` (a D-only stack, or a DSv4 hybrid stack whose H layers run Megatron-Core's ``dsv4_cp_zigzag``):
      chunks ``r`` and ``2 * CP - 1 - r`` of every sample.

    Under the zigzag partition a one-document batch is padded further, to ``2 * CP`` chunks of a multiple of
    ``zigzag_chunk_multiple`` rows and of at least ``zigzag_min_chunk_rows`` rows each (the H layers' requirements,
    :func:`ernie55_cp_zigzag_chunk_rows`; both ``1``, the default, leave the ``2 * CP`` padding alone). When a batch has several documents, MCore
    handles partial compression blocks and multi-rank halos. Each document needs only ``2 * CP`` alignment.

    Under the contiguous partition a pack shorter than ``CP * min_local_rows`` rows (short SFT samples, for
    example) gets more padding on its last sample, up to the next multiple of ``2 * CP`` of at least that size,
    so every CP rank holds the rows its H layers' boundary exchange needs
    (:func:`ernie55_cp_min_local_rows`). These pads follow the same rule as the others: label ``-100``, positions
    continuing the last sample's, inside its document. The LM and MTP losses of the real tokens do not change
    (attention is causal and pads carry no loss); the indexer loss, normalized by the pack's rows, counts them like
    any other pad rows.

    Labels must already be aligned with the inputs (``labels[t]`` is the target of position ``t``).

    Args:
        samples: ``(input_ids, labels)`` pairs of equal-length 1-D integer tensors.
        pad_token_id: Token id of the padding.
        cp_group: Context-parallel group, or ``None`` without context parallelism.
        cp_partition_mode: Partition of the stack's rows at CP > 1, ``"contiguous"`` or ``"zigzag"``.
        max_seqlen: ``max_seqlen_q``/``max_seqlen_kv`` of the metadata, at least the longest padded
            sample; pinning it (e.g. to the training sequence length) keeps the fused kernels'
            compile keys stable. Defaults to the longest padded sample.
        min_local_rows: Fewest rows each CP rank must hold under the contiguous partition (see
            :func:`ernie55_cp_min_local_rows`); ignored at CP = 1 and for ``"zigzag"``.
        zigzag_chunk_multiple: Row multiple of the single-document zigzag chunks.
        zigzag_min_chunk_rows: Fewest rows per chunk for single-document zigzag.
        device: Device of the returned tensors; defaults to the samples' device.
        with_key_reorder_inverse: At CP > 1, also prebuild the zigzag geometry's inverse key permutation, which
            the D layers' key-gather backward reads with ``ernie_cp_key_gather_inverse_grad``; ignored at
            CP = 1.

    Returns:
        This rank's :class:`Ernie55ThdBatch`.

    Raises:
        RuntimeError: At CP > 1 without Megatron-Core's CP partition converter (Megatron-Core main).
        ValueError: For an empty batch, malformed samples, a ``max_seqlen`` below the longest padded
            sample, a ``min_local_rows``, ``zigzag_chunk_multiple`` or ``zigzag_min_chunk_rows`` below 1 or an
            unknown ``cp_partition_mode``.
    """
    if not samples:
        raise ValueError("build_ernie55_thd_cp_batch needs at least one sample.")
    cp_size = get_pg_size(cp_group)
    cp_rank = get_pg_rank(cp_group)
    multiple = 2 * cp_size if cp_size > 1 else 1
    if device is None:
        device = samples[0][0].device
    for name, value in (
        ("min_local_rows", min_local_rows),
        ("zigzag_chunk_multiple", zigzag_chunk_multiple),
        ("zigzag_min_chunk_rows", zigzag_min_chunk_rows),
    ):
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive int, got {value!r}.")
    single_zigzag = cp_size > 1 and cp_partition_mode == "zigzag" and len(samples) == 1

    tokens, labels, positions, lengths = [], [], [], []
    for index, (input_ids, sample_labels) in enumerate(samples):
        if input_ids.ndim != 1 or sample_labels.shape != input_ids.shape or input_ids.numel() == 0:
            raise ValueError(
                f"Sample {index}: input_ids and labels must be non-empty 1-D tensors of one length, got shapes "
                f"{tuple(input_ids.shape)} and {tuple(sample_labels.shape)}."
            )
        length = input_ids.numel()
        if single_zigzag:
            padded = _zigzag_padded_length(
                length, cp_size=cp_size, chunk_multiple=zigzag_chunk_multiple, min_chunk_rows=zigzag_min_chunk_rows
            )
        else:
            padded = _padded_length(length, multiple)
        pad = padded - length
        tokens.append(torch.cat([input_ids.to(device), input_ids.new_full((pad,), pad_token_id, device=device)]))
        labels.append(
            torch.cat([sample_labels.to(device), sample_labels.new_full((pad,), IGNORE_LABEL, device=device)])
        )
        positions.append(torch.arange(padded, dtype=torch.long, device=device))
        lengths.append(padded)

    if cp_size > 1 and cp_partition_mode == "contiguous":
        short = _padded_length(cp_size * min_local_rows, multiple) - sum(lengths)
        if short > 0:
            last = len(lengths) - 1
            tokens[last] = torch.cat([tokens[last], tokens[last].new_full((short,), pad_token_id)])
            labels[last] = torch.cat([labels[last], labels[last].new_full((short,), IGNORE_LABEL)])
            positions[last] = torch.arange(lengths[last] + short, dtype=torch.long, device=device)
            lengths[last] += short

    longest = max(lengths)
    if max_seqlen is None:
        max_seqlen = longest
    if not isinstance(max_seqlen, int) or max_seqlen < longest:
        raise ValueError(
            f"max_seqlen must be an int of at least the longest padded sample ({longest}), got {max_seqlen!r}."
        )
    boundaries = [0, *torch.tensor(lengths).cumsum(0).tolist()]
    cu_seqlens = torch.tensor(boundaries, dtype=torch.int32, device=device)
    metadata = dict(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        cu_seqlens_q_padded=cu_seqlens,
        cu_seqlens_kv_padded=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
    )
    tokens, labels, positions = torch.cat(tokens), torch.cat(labels), torch.cat(positions)

    if cp_size > 1:
        api = require_cp_layout_api("build_ernie55_thd_cp_batch")
        if cp_partition_mode not in ("contiguous", "zigzag"):
            raise ValueError(f"cp_partition_mode must be 'contiguous' or 'zigzag', got {cp_partition_mode!r}.")
        packed_seq_params = PackedSeqParams(**metadata, cp_partition_mode=cp_partition_mode)
        api.prebuild_thd_cp_partition_routes(packed_seq_params, cp_group)
        local_rows = tokens.numel() // cp_size
        # The D layers' zigzag geometry: built here, so no D layer copies it to the device in its forward.
        prebuild_zigzag_cp_geometry(
            packed_seq_params,
            cp_group=cp_group,
            local_rows=local_rows,
            device=device,
            with_inverse=with_key_reorder_inverse,
        )
        if cp_partition_mode == "contiguous":
            rows = torch.arange(cp_rank * local_rows, (cp_rank + 1) * local_rows, device=device)
        else:
            rows = dsa_layout.build_packed_allgather_cp_local_positions_from_host(
                boundaries, cp_size, cp_rank, torch.device(device)
            )
        tokens, labels, positions = (tensor.index_select(0, rows) for tensor in (tokens, labels, positions))
    else:
        packed_seq_params = PackedSeqParams(**metadata)
        # The pack's boundaries on the host: the D layers check its end at CP = 1 without reading the device, and a
        # D-layer CUDA graph compares them with the boundaries it was captured for.
        record_pack_boundaries(packed_seq_params, boundaries)

    labels = labels.unsqueeze(0)
    return Ernie55ThdBatch(
        tokens=tokens.unsqueeze(0),
        labels=labels,
        loss_mask=(labels != IGNORE_LABEL).float(),
        position_ids=positions.unsqueeze(0),
        packed_seq_params=packed_seq_params,
    )
