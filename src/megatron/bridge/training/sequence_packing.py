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

"""Online sequence packing and dynamic context parallelism.

Megatron-Core's sequence-packing schedulers (``model.sequence_packing_scheduler``)
take unpacked variable-length samples, pack them into THD microbatches once per
training step, and, with ``model.dynamic_context_parallel``, run every packed bin
on a context-parallel group sized for its longest sequence. This module holds the
Bridge-side glue: iterator wrapping, the packed ``get_batch`` branch, the padding
multiple every sample must satisfy, and configuration validation.

The scheduler API lives in Megatron-Core ``dev`` (``scripts/switch_mcore.sh dev``).
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, Callable

import torch

from megatron.bridge.data.collators.identity import identity_collate


if TYPE_CHECKING:
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.process_groups_config import ProcessGroupCollection

    from megatron.bridge.training.config import ConfigContainer

logger = logging.getLogger(__name__)

__all__ = [
    "DYNAMIC_CP_SCHEDULERS",
    "REQUIRED_SAMPLE_KEYS",
    "dynamic_context_parallel_enabled",
    "finalize_packed_seq_params",
    "get_batch_for_sequence_packing",
    "identity_collate",
    "padding_multiple",
    "scheduled_num_microbatches",
    "sequence_packing_enabled",
    "sequence_padding_multiple",
    "set_scheduled_num_microbatches",
    "validate_sequence_packing_config",
    "wrap_data_iterator_for_sequence_packing",
]

DYNAMIC_CP_SCHEDULERS: tuple[str, ...] = ("default_dynamic_cp",)
"""Scheduler names that form dynamic context-parallel groups per microbatch."""

REQUIRED_SAMPLE_KEYS: tuple[str, ...] = (
    "tokens",
    "labels",
    "loss_mask",
    "position_ids",
    "original_seq_len",
    "padded_seq_len",
)
"""Keys every unpacked sample must carry for the Megatron-Core scheduler."""

_SCHEDULED_NUM_MICROBATCHES: int | None = None


def sequence_packing_enabled(model_config: Any) -> bool:
    """Return whether a Megatron-Core online packing scheduler is configured."""
    return getattr(model_config, "sequence_packing_scheduler", None) is not None


def dynamic_context_parallel_enabled(model_config: Any) -> bool:
    """Return whether dynamic context parallelism is configured."""
    return bool(getattr(model_config, "dynamic_context_parallel", False))


def set_scheduled_num_microbatches(value: int | None) -> None:
    """Record the microbatch count the scheduler produced for the current step (``None`` to clear)."""
    global _SCHEDULED_NUM_MICROBATCHES
    _SCHEDULED_NUM_MICROBATCHES = value


def scheduled_num_microbatches(default: Callable[[], int]) -> int:
    """Return the scheduled microbatch count for this step, or ``default()`` when no scheduler ran.

    Loss-scale bookkeeping (MoE auxiliary losses, MTP metrics) must divide by the
    number of microbatches that actually ran, which under packing differs from the
    sample-based ``get_num_microbatches()``.
    """
    return _SCHEDULED_NUM_MICROBATCHES if _SCHEDULED_NUM_MICROBATCHES is not None else default()


def _scheduler_api() -> tuple[Callable[..., Any], Callable[..., Any]]:
    """Import the Megatron-Core scheduler entry points or explain what is missing."""
    try:
        from megatron.core.datasets.data_schedule import (
            get_batch_on_this_rank_for_sequence_packing,
            wrap_data_iterator,
        )
    except ImportError as exc:  # pragma: no cover - depends on the pinned submodule
        raise RuntimeError(
            "model.sequence_packing_scheduler requires a Megatron-Core commit with online sequence "
            "packing (megatron.core.datasets.data_schedule). Switch the submodule with "
            "`./scripts/switch_mcore.sh dev && uv sync`."
        ) from exc
    return wrap_data_iterator, get_batch_on_this_rank_for_sequence_packing


def padding_multiple(
    *, dynamic_cp: bool, dp_size: int, cp_size: int, tp_size: int = 1, sequence_parallel: bool = False
) -> int:
    """Return the length multiple every sample must be padded to.

    Context-parallel THD slicing splits each sequence into ``2 * group`` zigzag
    chunks. Under dynamic CP a sequence may run on any power-of-two group up to
    the DPxCP pool (or the whole pool), so the multiple covers all of them;
    static CP uses the fixed group. Sequence parallelism additionally needs
    TP-divisible chunks.
    """
    if dynamic_cp:
        pool = dp_size * cp_size
        group_sizes = {1 << k for k in range(pool.bit_length()) if (1 << k) <= pool} | {pool}
        multiple = 1
        for size in group_sizes:
            multiple = math.lcm(multiple, 2 * size)
    else:
        multiple = 2 * cp_size if cp_size > 1 else 1
    if sequence_parallel:
        multiple *= tp_size
    return multiple


def sequence_padding_multiple(model_config: Any, pg_collection: ProcessGroupCollection) -> int:
    """Return the padding multiple for ``model_config`` given the live process groups."""
    return padding_multiple(
        dynamic_cp=dynamic_context_parallel_enabled(model_config),
        dp_size=pg_collection.dp.size(),
        cp_size=pg_collection.cp.size(),
        tp_size=pg_collection.tp.size(),
        sequence_parallel=bool(getattr(model_config, "sequence_parallel", False)),
    )


def _iterator_for_tp_rank(data_iterator: Any, pg_collection: ProcessGroupCollection) -> Any:
    """Return the iterator on TP rank 0 and ``None`` elsewhere, as the scheduler requires."""
    if pg_collection.tp.rank() != 0:
        return None
    return data_iterator


def wrap_data_iterator_for_sequence_packing(
    data_iterator: Any,
    model_config: Any,
    num_microbatches: int,
    pg_collection: ProcessGroupCollection,
) -> tuple[Any, int, float, float]:
    """Pack this step's global batch into THD microbatches.

    Pulls ``num_microbatches`` samples per data-parallel rank, gathers their
    lengths, schedules bins (and CP groups under dynamic CP), reroutes samples,
    and returns the packed iterator with the number of microbatches it yields.

    Args:
        data_iterator: The raw per-sample iterator.
        model_config: The finalized model config carrying the scheduler fields.
        num_microbatches: Samples to consume per data-parallel rank this step.
        pg_collection: Process groups used for the length all-gather and reroute.

    Returns:
        ``(packed_iterator, packed_num_microbatches, seqlen_sum, seqlen_squared_sum)``.
        The sums are data-parallel-global padded token statistics for FLOPs accounting.
    """
    wrap_data_iterator, _ = _scheduler_api()
    return wrap_data_iterator(
        _iterator_for_tp_rank(data_iterator, pg_collection),
        model_config,
        num_microbatches,
        pg_collection=pg_collection,
    )


def get_batch_for_sequence_packing(
    data_iterator: Any,
    model_config: Any,
    *,
    pg_collection: ProcessGroupCollection,
    vp_stage: int | None,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    None,
    torch.Tensor | None,
    PackedSeqParams | None,
    torch.Tensor | None,
]:
    """Fetch one packed microbatch and its ``PackedSeqParams`` on every rank.

    Megatron-Core slices the packed bin for this rank's (possibly dynamic) CP group,
    broadcasts it over TP, and returns THD metadata. Middle pipeline stages get the
    metadata they need without token tensors.

    Returns:
        ``(tokens, labels, loss_mask, attention_mask, position_ids, packed_seq_params,
        padding_mask)``; ``attention_mask`` is always ``None`` for THD.
    """
    from megatron.core.transformer.multi_token_prediction import mtp_on_this_rank

    _, get_batch_on_this_rank_for_sequence_packing = _scheduler_api()
    tokens, labels, loss_mask, attention_mask, position_ids, packed_seq_params, padding_mask = (
        get_batch_on_this_rank_for_sequence_packing(
            _iterator_for_tp_rank(data_iterator, pg_collection),
            vpp_size=getattr(model_config, "virtual_pipeline_model_parallel_size", None),
            mtp_on_this_rank=mtp_on_this_rank(
                getattr(model_config, "pipeline_model_parallel_layout", None),
                getattr(model_config, "mtp_num_layers", None),
                ignore_virtual=False,
                vp_stage=vp_stage,
            ),
            vp_stage=vp_stage,
            dynamic_cp=dynamic_context_parallel_enabled(model_config),
            pg_collection=pg_collection,
            config=model_config,
        )
    )
    finalize_packed_seq_params(packed_seq_params, pg_collection)
    return tokens, labels, loss_mask, attention_mask, position_ids, packed_seq_params, padding_mask


def finalize_packed_seq_params(
    packed_seq_params: PackedSeqParams | None, pg_collection: ProcessGroupCollection
) -> PackedSeqParams | None:
    """Bind the CP group and prebuild the THD partition route for one microbatch.

    Uses ``pg_collection.cp`` as the static group so the group the model was built
    with is the one attention falls back to when no dynamic group is attached.
    """
    if packed_seq_params is None:
        return None
    from megatron.core.context_parallel_layout.routes import prebuild_thd_cp_partition_routes
    from megatron.core.packed_seq_params import resolve_cp_group

    cp_group = resolve_cp_group(pg_collection.cp, packed_seq_params)
    packed_seq_params.cp_group = cp_group
    prebuild_thd_cp_partition_routes(packed_seq_params, cp_group)
    return packed_seq_params


def _cuda_graphs_requested(model: Any) -> bool:
    return getattr(model, "cuda_graph_impl", "none") not in (None, "none") or bool(
        getattr(model, "enable_cuda_graph", False)
    )


def validate_sequence_packing_config(cfg: ConfigContainer) -> None:
    """Check the training configuration for online sequence packing / dynamic CP.

    Mirrors the constraints Megatron-LM applies on the same path so that
    misconfigurations fail at validation time instead of at the first step.
    A no-op when no scheduler and no dynamic CP is configured.
    """
    model = cfg.model
    scheduler = getattr(model, "sequence_packing_scheduler", None)
    dynamic_cp = dynamic_context_parallel_enabled(model)
    if scheduler is None and not dynamic_cp:
        return

    if not hasattr(model, "dynamic_context_parallel") or not hasattr(model, "sequence_packing_scheduler"):
        raise ValueError(
            "The pinned Megatron-Core does not expose online sequence packing / dynamic context "
            "parallelism. Run `./scripts/switch_mcore.sh dev && uv sync`."
        )
    if dynamic_cp and scheduler not in DYNAMIC_CP_SCHEDULERS:
        raise ValueError(
            f"model.dynamic_context_parallel requires model.sequence_packing_scheduler set explicitly to a "
            f"dynamic-CP scheduler {DYNAMIC_CP_SCHEDULERS}, got {scheduler!r}."
        )
    if getattr(model, "virtual_pipeline_model_parallel_size", None) not in (None, 1):
        raise ValueError(
            "Sequence packing with virtual pipeline parallelism is not supported yet in Bridge "
            "(the scheduler expects one iterator per virtual stage); set virtual_pipeline_model_parallel_size=None."
        )
    if getattr(model, "is_hybrid_model", False):
        raise ValueError(
            "Sequence packing with Mamba hybrid models is not supported yet in Bridge (packed sequence "
            "boundaries are not propagated to the SSM layers)."
        )
    if getattr(model, "max_seqlen_per_dp_cp_rank", None) is None:
        raise ValueError(
            "model.max_seqlen_per_dp_cp_rank must be set explicitly for sequence packing: it is the token "
            "capacity of one rank per microbatch (static CP: seq_length / cp; dynamic CP: the length above "
            "which a sequence needs a larger CP group)."
        )
    if cfg.train.micro_batch_size != 1:
        raise ValueError(
            f"Sequence packing schedules one THD bin per microbatch; set train.micro_batch_size=1 "
            f"(got {cfg.train.micro_batch_size})."
        )
    if cfg.dataset.dataloader_type not in ("single", "cyclic"):
        raise ValueError(
            "Sequence packing consumes per-sample microbatches; set dataset.dataloader_type to 'single' or "
            f"'cyclic' (got {cfg.dataset.dataloader_type!r})."
        )
    if not getattr(model, "calculate_per_token_loss", False):
        raise ValueError(
            "Sequence packing requires model.calculate_per_token_loss=True (loss is normalized by real tokens)."
        )
    if getattr(cfg.ddp, "average_in_collective", False):
        raise ValueError("Sequence packing requires ddp.average_in_collective=False.")
    if getattr(cfg.ddp, "use_megatron_fsdp", False):
        raise ValueError("Sequence packing is not supported with Megatron FSDP.")
    if dynamic_cp and getattr(getattr(cfg, "dist", None), "use_decentralized_pg", False):
        raise ValueError(
            "model.dynamic_context_parallel requires dist.use_decentralized_pg=False: Megatron-Core resolves "
            "dynamic CP groups from parallel_state."
        )
    if _cuda_graphs_requested(model):
        if dynamic_cp:
            raise ValueError(
                "Dynamic context parallelism does not support CUDA graphs; set model.cuda_graph_impl='none'."
            )
        if (
            getattr(model, "thd_max_packed_sequences", None) is None
            or getattr(model, "pad_packed_seq_alignment", None) is None
        ):
            raise ValueError(
                "CUDA graphs with sequence packing need static THD shapes: set model.thd_max_packed_sequences and "
                "model.pad_packed_seq_alignment, or disable CUDA graphs (model.cuda_graph_impl='none')."
            )

    dataset = cfg.dataset
    if not hasattr(dataset, "sequence_padding_multiple"):
        raise ValueError(
            "The configured dataset does not produce the unpacked per-sample dicts the sequence-packing "
            "scheduler consumes. Use SyntheticVarlenDatasetConfig or a DatasetProvider that exposes "
            "`sequence_padding_multiple` and yields tokens/labels/loss_mask/position_ids/original_seq_len/"
            "padded_seq_len samples with an identity collate (see docs/training/dynamic-context-parallel.md)."
        )

    cp_size = model.context_parallel_size
    dp_size = getattr(cfg, "data_parallel_size", None)
    if dp_size is not None:
        pool = dp_size * cp_size if dynamic_cp else cp_size
        if dynamic_cp and pool % 2 and pool > 1:
            raise ValueError(f"Dynamic context parallelism needs an even dp * cp pool, got {pool}.")
        min_cp = getattr(model, "min_dynamic_context_parallel_size", 1)
        if dynamic_cp and (min_cp < 1 or (min_cp & (min_cp - 1)) != 0 or min_cp > pool):
            raise ValueError(
                f"model.min_dynamic_context_parallel_size must be a power of two in [1, dp * cp = {pool}], got {min_cp}."
            )
        target_seq_length = getattr(dataset, "seq_length", None) or model.seq_length
        if pool * model.max_seqlen_per_dp_cp_rank < target_seq_length:
            raise ValueError(
                f"Bin capacity too small: {pool} ranks x max_seqlen_per_dp_cp_rank {model.max_seqlen_per_dp_cp_rank} "
                f"= {pool * model.max_seqlen_per_dp_cp_rank} tokens < seq_length {target_seq_length}; the longest "
                "sample could never be scheduled."
            )
        multiple = padding_multiple(
            dynamic_cp=dynamic_cp,
            dp_size=dp_size,
            cp_size=cp_size,
            tp_size=model.tensor_model_parallel_size,
            sequence_parallel=bool(getattr(model, "sequence_parallel", False)),
        )
        if dataset.sequence_padding_multiple is None:
            dataset.sequence_padding_multiple = multiple
        elif dataset.sequence_padding_multiple % multiple:
            raise ValueError(
                f"dataset.sequence_padding_multiple={dataset.sequence_padding_multiple} is not a multiple of the "
                f"{multiple} that CP THD slicing requires for this parallel layout."
            )
        dataset_seq_length = getattr(dataset, "seq_length", None)
        if dataset_seq_length is not None:
            if dataset_seq_length % dataset.sequence_padding_multiple:
                raise ValueError(
                    f"dataset.seq_length={dataset_seq_length} must be a multiple of the padding multiple "
                    f"{dataset.sequence_padding_multiple}."
                )
            if dataset_seq_length > model.seq_length:
                raise ValueError(
                    f"dataset.seq_length={dataset_seq_length} exceeds model.seq_length={model.seq_length}."
                )
