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

"""ERNIE 5.5 training step: packed THD micro-batches, context parallelism and the per-row loss.

Select it with ``--step_func ernie55_step`` (``scripts/training/run_recipe.py``) or pass :func:`forward_step` to
``finetune``/``pretrain``. Every micro-batch becomes one packed THD row, as in the reference implementation:

1. The data iterator yields a dict with ``tokens`` and ``labels`` (``[b, T]``; ``labels[t]`` is the target of
   position ``t``, ``-100`` for none), optionally ``loss_mask`` (``0`` also drops a target), ``cu_seqlens`` (the
   document boundaries of a ``b = 1`` row; without it every row is one document) and ``pad_token_id``.
   :class:`~megatron.bridge.models.ernie.ernie55_data.Ernie55SFTDatasetConfig` produces it, and so can any
   provider of Bridge's GPT batches (one document per row).
2. :func:`~megatron.bridge.models.ernie.ernie55_context_parallel.build_ernie55_thd_cp_batch` pads every document
   to a multiple of ``2 * CP`` (pad tokens with label ``-100``; a pack too short for the H layers' CP boundary
   exchange gets more pads on its last document), puts one ``cu_seqlens`` tensor in all four
   packed fields, prebuilds the context-parallel partition routes and slices this rank's rows of the model's
   ``cp_partition_mode``.
3. The model gets the labels with ``-100`` replaced by a valid id and the loss mask ``labels != -100`` (Megatron-Core
   masks the LM and MTP losses with ``loss_mask``).
4. The loss function follows the model config (:func:`ernie55_loss_function`): ``ernie_per_row_loss`` selects the
   reference's per-row loss
   (:func:`~megatron.bridge.models.ernie.ernie55_context_parallel.ernie55_per_row_lm_loss`) and requires
   ``calculate_per_token_loss=False``; otherwise Bridge's ``masked_next_token_loss``, which at
   context_parallel_size > 1 requires per-token loss. The MTP depths follow the same flag inside the model.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from functools import partial
from typing import TYPE_CHECKING, Any

import torch
from megatron.core.utils import get_pg_size
from torch import Tensor

from megatron.bridge.models.ernie.ernie55_context_parallel import (
    IGNORE_LABEL,
    Ernie55ThdBatch,
    build_ernie55_thd_cp_batch,
    ernie55_cp_min_local_rows,
    ernie55_per_row_lm_loss,
)
from megatron.bridge.training.losses import masked_next_token_loss


if TYPE_CHECKING:
    from megatron.bridge.training.state import GlobalState


__all__ = ["ernie55_documents_from_batch", "ernie55_loss_function", "forward_step", "get_batch"]


def ernie55_documents_from_batch(batch: Mapping[str, Any]) -> list[tuple[Tensor, Tensor]]:
    """The documents of one micro-batch as ``(input_ids, labels)`` pairs on the host.

    Args:
        batch: ``tokens`` and ``labels`` ``[b, T]`` (labels aligned with the inputs, ``-100`` for no target),
            optionally ``loss_mask`` ``[b, T]`` (where it is 0 the label becomes ``-100``) and ``cu_seqlens``, the
            ``n + 1`` document boundaries of a single row (``b = 1``). Rows after the last boundary are padding
            and are dropped. Without ``cu_seqlens`` every row is one document.

    Returns:
        One ``(input_ids, labels)`` pair of equal-length 1-D int64 CPU tensors per document.

    Raises:
        ValueError: If the shapes disagree, ``cu_seqlens`` comes with ``b > 1``, or the boundaries are not
            increasing from 0 within the row.
    """
    tokens = torch.as_tensor(batch["tokens"]).cpu().long()
    labels = torch.as_tensor(batch["labels"]).cpu().long()
    if tokens.ndim == 1:
        tokens, labels = tokens.unsqueeze(0), labels.unsqueeze(0)
    if tokens.ndim != 2 or labels.shape != tokens.shape:
        raise ValueError(
            f"ERNIE 5.5 batches need tokens and labels of one [b, T] shape, got {tuple(tokens.shape)} and "
            f"{tuple(labels.shape)}."
        )
    loss_mask = batch.get("loss_mask")
    if loss_mask is not None:
        loss_mask = torch.as_tensor(loss_mask).cpu().reshape(labels.shape)
        labels = labels.masked_fill(loss_mask == 0, IGNORE_LABEL)
    cu_seqlens = batch.get("cu_seqlens")
    if cu_seqlens is None:
        return list(zip(tokens.unbind(0), labels.unbind(0)))
    if tokens.size(0) != 1:
        raise ValueError(f"cu_seqlens describes one packed row; got a micro-batch of {tokens.size(0)} rows.")
    bounds = [int(value) for value in torch.as_tensor(cu_seqlens).flatten().tolist()]
    if (
        len(bounds) < 2
        or bounds[0] != 0
        or bounds[-1] > tokens.size(1)
        or any(end <= start for start, end in zip(bounds[:-1], bounds[1:]))
    ):
        raise ValueError(f"cu_seqlens must increase from 0 to at most {tokens.size(1)}, got {bounds}.")
    return [(tokens[0, start:end], labels[0, start:end]) for start, end in zip(bounds[:-1], bounds[1:])]


def ernie55_loss_function(
    loss_mask: Tensor,
    *,
    config: Any,
    cp_group: torch.distributed.ProcessGroup | None,
    check_for_nan_in_loss: bool,
    check_for_spiky_loss: bool,
) -> partial:
    """The LM loss of the ERNIE 5.5 training step, selected from the model config.

    * ``ernie_per_row_loss``: :func:`~megatron.bridge.models.ernie.ernie55_context_parallel.ernie55_per_row_lm_loss`
      (the micro-batch's token mean over all of its CP ranks' tokens); requires ``calculate_per_token_loss=False``.
    * Otherwise Bridge's ``masked_next_token_loss``. At context_parallel_size > 1 it then needs
      ``calculate_per_token_loss=True``: a per-rank token mean would weight the ranks of a row equally whatever
      their supervised tokens.

    Args:
        loss_mask: This rank's loss mask of the micro-batch.
        config: Model (transformer) config.
        cp_group: Context-parallel group, or ``None``.
        check_for_nan_in_loss: Run the rerun state machine's NaN/Inf check.
        check_for_spiky_loss: Run the rerun state machine's spiky-loss check.

    Returns:
        ``loss_function(output_tensor)`` for Megatron-Core's schedule.

    Raises:
        ValueError: If the loss settings are inconsistent.
    """
    per_token = bool(getattr(config, "calculate_per_token_loss", False))
    if getattr(config, "ernie_per_row_loss", False):
        if per_token:
            raise ValueError("ernie_per_row_loss=True needs calculate_per_token_loss=False.")
        return partial(
            ernie55_per_row_lm_loss,
            loss_mask,
            cp_group=cp_group if get_pg_size(cp_group) > 1 else None,
            check_for_nan_in_loss=check_for_nan_in_loss,
            check_for_spiky_loss=check_for_spiky_loss,
        )
    if get_pg_size(cp_group) > 1 and not per_token:
        raise ValueError(
            "ERNIE 5.5 at context_parallel_size > 1 needs ernie_per_row_loss=True (the reference's per-row loss) or "
            "calculate_per_token_loss=True; a per-rank token mean would weight the ranks of a row equally."
        )
    return partial(
        masked_next_token_loss,
        loss_mask,
        check_for_nan_in_loss=check_for_nan_in_loss,
        check_for_spiky_loss=check_for_spiky_loss,
    )


def get_batch(
    data_iterator: Iterable,
    config: Any,
    *,
    cp_group: torch.distributed.ProcessGroup | None,
    device: torch.device | str,
) -> Ernie55ThdBatch:
    """Read one micro-batch and build this rank's packed THD batch.

    Args:
        data_iterator: Yields batches in the contract of :func:`ernie55_documents_from_batch`, optionally with an
            int ``pad_token_id`` (needed only when a document must be padded to a multiple of ``2 * CP``).
        config: Model (transformer) config (``cp_partition_mode``, and ``csa_compress_ratios`` with
            ``csa_window_size`` for the fewest rows a CP rank may hold).
        cp_group: Context-parallel group, or ``None``.
        device: Device of the returned tensors.

    Returns:
        This rank's rows with the metadata of the whole pack.

    Raises:
        ValueError: If a document needs padding and the batch has no ``pad_token_id``.
    """
    batch = next(data_iterator)
    documents = ernie55_documents_from_batch(batch)
    cp_size = get_pg_size(cp_group)
    multiple = 2 * cp_size if cp_size > 1 else 1
    pad_token_id = batch.get("pad_token_id")
    if pad_token_id is None:
        unaligned = [len(tokens) for tokens, _ in documents if len(tokens) % multiple]
        if unaligned:
            raise ValueError(
                f"Documents of {unaligned} tokens need padding to a multiple of 2 * context_parallel_size = {multiple}, "
                "but the batch has no pad_token_id."
            )
        # No document is padded: the id is never used.
        pad_token_id = IGNORE_LABEL
    cp_partition_mode = getattr(config, "cp_partition_mode", "contiguous") if cp_size > 1 else "contiguous"
    min_local_rows = ernie55_cp_min_local_rows(config) if cp_size > 1 and cp_partition_mode == "contiguous" else 1
    pack_tokens = sum(len(tokens) for tokens, _ in documents)
    if batch.get("pad_token_id") is None and pack_tokens < cp_size * min_local_rows:
        raise ValueError(
            f"A pack of {pack_tokens} tokens needs padding to at least {cp_size * min_local_rows} rows "
            f"(context_parallel_size={cp_size} x {min_local_rows} rows per rank for the H layers' boundary "
            "exchange), but the batch has no pad_token_id."
        )
    return build_ernie55_thd_cp_batch(
        documents,
        pad_token_id=int(pad_token_id),
        cp_group=cp_group if cp_size > 1 else None,
        cp_partition_mode=cp_partition_mode,
        min_local_rows=min_local_rows,
        device=device,
    )


def forward_step(
    state: GlobalState, data_iterator: Iterable, model: torch.nn.Module, return_schedule_plan: bool = False
) -> tuple[Tensor, partial]:
    """Forward training step of ERNIE 5.5.

    Args:
        state: Global training state.
        data_iterator: The micro-batch iterator (see :func:`get_batch`).
        model: The (wrapped) ERNIE 5.5 model chunk.
        return_schedule_plan: Not supported (no EP communication overlap schedule for ERNIE 5.5).

    Returns:
        The per-token losses and the loss function (:func:`ernie55_loss_function`).

    Raises:
        NotImplementedError: If a schedule plan is requested.
    """
    from megatron.core.utils import get_model_config

    from megatron.bridge.training.utils.flop_utils import accumulate_flops_metadata, get_model_chunk_vp_stage
    from megatron.bridge.training.utils.pg_utils import get_pg_collection

    if return_schedule_plan:
        raise NotImplementedError("ERNIE 5.5 does not support the MoE A2A overlap schedule plan.")
    config = get_model_config(model)
    cp_group = get_pg_collection(model).cp
    timers = state.timers
    timers("batch-generator", log_level=2).start()
    with state.straggler_timer(bdata=True):
        batch = get_batch(data_iterator, config, cp_group=cp_group, device=torch.cuda.current_device())
    timers("batch-generator").stop()

    cp_size = get_pg_size(cp_group)
    accumulate_flops_metadata(
        state,
        batch.tokens,
        vp_stage=get_model_chunk_vp_stage(model),
        config_seq_len=batch.tokens.size(1) * cp_size,
        context_parallel_size=cp_size,
        cu_seqlens=batch.packed_seq_params.cu_seqlens_q_padded,
    )
    with state.straggler_timer:
        output = model(
            input_ids=batch.tokens,
            position_ids=batch.position_ids,
            attention_mask=None,
            labels=batch.labels.clamp(min=0),
            loss_mask=batch.loss_mask,
            packed_seq_params=batch.packed_seq_params,
        )
    loss_function = ernie55_loss_function(
        batch.loss_mask,
        config=config,
        cp_group=cp_group,
        check_for_nan_in_loss=state.cfg.rerun_state_machine.check_for_nan_in_loss,
        check_for_spiky_loss=state.cfg.rerun_state_machine.check_for_spiky_loss,
    )
    return output, loss_function
