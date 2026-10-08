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

"""Training-loop glue for global-batch online packing.

Megatron-Core's sequence-packing scheduler (``model.sequence_packing_scheduler``,
driven by ``dataset.enable_global_batch_packing``) packs unpacked variable-length
samples into THD microbatches once per training step, with the whole global batch
across data-parallel ranks as its candidate pool.
:func:`probe_global_batch_packing_support` checks the pinned Megatron-Core before
training starts, and the entry points below adapt to the scheduler API of the pinned
release.

Configuration rules live in ``ConfigContainer.validate`` next to the other
packing modes; the data contract lives in ``megatron.bridge.data.packing.global_batch``.
"""

from __future__ import annotations

import functools
import importlib
import inspect
import logging
from typing import TYPE_CHECKING, Any, Callable

import torch


if TYPE_CHECKING:
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.process_groups_config import ProcessGroupCollection

logger = logging.getLogger(__name__)

__all__ = [
    "finalize_packed_seq_params",
    "get_batch_for_global_batch_packing",
    "global_batch_packing_enabled",
    "probe_global_batch_packing_support",
    "wrap_data_iterator_for_global_batch_packing",
]


def global_batch_packing_enabled(model_config: Any) -> bool:
    """Return whether a Megatron-Core online packing scheduler is configured on the model.

    Checks for a scheduler name rather than any non-``None`` value so that stand-in configs
    (mocks, partial namespaces) do not route a step through the packing scheduler.
    """
    return isinstance(getattr(model_config, "sequence_packing_scheduler", None), str)


def _scheduler_api() -> tuple[Callable[..., Any], Callable[..., Any]]:
    """Import the Megatron-Core scheduler entry points."""
    from megatron.core.datasets.data_schedule import (
        get_batch_on_this_rank_for_sequence_packing,
        wrap_data_iterator,
    )

    return wrap_data_iterator, get_batch_on_this_rank_for_sequence_packing


@functools.lru_cache(maxsize=None)
def _accepted_parameters(fn: Callable[..., Any]) -> frozenset[str]:
    """Return the parameter names ``fn`` accepts (cached: called once per microbatch)."""
    return frozenset(inspect.signature(fn).parameters)


@functools.lru_cache(maxsize=None)
def _thd_cp_route_prebuilder() -> Callable[..., None] | None:
    """Return Megatron-Core's THD CP partition-route prebuild, or ``None`` if the release has no CP layout routes."""
    try:
        from megatron.core.context_parallel_layout.routes import prebuild_thd_cp_partition_routes
    except ImportError:
        return None
    return prebuild_thd_cp_partition_routes


def probe_global_batch_packing_support(scheduler: str) -> str | None:
    """Return why the pinned Megatron-Core cannot run ``scheduler``, or ``None`` if it can.

    Probing the registered schedulers and the entry-point signatures lets validation
    fail with an actionable message instead of a ``KeyError`` or ``TypeError`` at the
    first step.
    """
    try:
        data_schedule = importlib.import_module("megatron.core.datasets.data_schedule")
    except ImportError:
        return (
            "the pinned Megatron-Core does not provide megatron.core.datasets.data_schedule (online sequence packing)."
        )

    registered: set[str] = set()
    scheduler_map = getattr(data_schedule, "scheduler_map", None)
    if isinstance(scheduler_map, dict):
        registered = {key.value if hasattr(key, "value") else str(key) for key in scheduler_map}
    enum = getattr(data_schedule, "PackingSchedulerEnum", None)
    if enum is not None:
        registered |= {member.value for member in enum}
    if scheduler not in registered:
        return f"the pinned Megatron-Core registers schedulers {sorted(registered)}; '{scheduler}' is not available."

    get_batch = getattr(data_schedule, "get_batch_on_this_rank_for_sequence_packing", None)
    wrap = getattr(data_schedule, "wrap_data_iterator", None)
    if get_batch is None or wrap is None:
        return "the pinned Megatron-Core lacks the packed batch entry points."
    if "pg_collection" not in _accepted_parameters(wrap) or "pg_collection" not in _accepted_parameters(get_batch):
        return "the pinned Megatron-Core scheduler does not accept a ProcessGroupCollection."
    return None


def _iterator_for_tp_rank(data_iterator: Any, pg_collection: ProcessGroupCollection) -> Any:
    """Return the iterator on TP rank 0 and ``None`` elsewhere, as the scheduler requires."""
    if pg_collection.tp.rank() != 0:
        return None
    return data_iterator


def wrap_data_iterator_for_global_batch_packing(
    data_iterator: Any,
    model_config: Any,
    num_microbatches: int,
    pg_collection: ProcessGroupCollection,
) -> tuple[Any, int, float, float]:
    """Pack this step's global batch into THD microbatches.

    Pulls ``num_microbatches`` samples per data-parallel rank, gathers their
    lengths, schedules bins, reroutes samples, and returns the packed iterator
    with the number of microbatches it yields. The packed iterator is a
    ``RerunDataIterator`` on TP rank 0 and ``None`` on other TP ranks, so callers
    must track "already wrapped" with an explicit flag.

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


def get_batch_for_global_batch_packing(
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

    Megatron-Core slices the packed bin for this rank's CP group, broadcasts it
    over TP, and returns THD metadata.

    Returns:
        ``(tokens, labels, loss_mask, attention_mask, position_ids, packed_seq_params,
        padding_mask)``; ``attention_mask`` is always ``None`` for THD.
    """
    from megatron.core.transformer.multi_token_prediction import mtp_on_this_rank

    _, get_batch_on_this_rank_for_sequence_packing = _scheduler_api()
    optional_kwargs: dict[str, Any] = {}
    if "config" in _accepted_parameters(get_batch_on_this_rank_for_sequence_packing):
        # Releases that accept the model config read CP partitioning and THD padding options from it.
        optional_kwargs["config"] = model_config
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
            pg_collection=pg_collection,
            **optional_kwargs,
        )
    )
    finalize_packed_seq_params(packed_seq_params, pg_collection)
    return tokens, labels, loss_mask, attention_mask, position_ids, packed_seq_params, padding_mask


def finalize_packed_seq_params(
    packed_seq_params: PackedSeqParams | None, pg_collection: ProcessGroupCollection
) -> PackedSeqParams | None:
    """Bind the CP group to one microbatch and prebuild its THD CP partition route.

    Uses ``pg_collection.cp`` as the static group so attention runs on the group the
    model was built with. Releases with context-parallel layout routes get the route
    built here, off the forward path; older releases partition without one.
    """
    if packed_seq_params is None:
        return None
    from megatron.core.packed_seq_params import resolve_cp_group

    cp_group = resolve_cp_group(pg_collection.cp, packed_seq_params)
    packed_seq_params.cp_group = cp_group
    prebuild_routes = _thd_cp_route_prebuilder()
    if prebuild_routes is not None:
        prebuild_routes(packed_seq_params, cp_group)
    return packed_seq_params
