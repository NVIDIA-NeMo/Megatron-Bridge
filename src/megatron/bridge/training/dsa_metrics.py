# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Integrate MCore's fixed-storage DSA metric lifecycle with Bridge training."""

from __future__ import annotations

from importlib import import_module
from numbers import Real
from types import ModuleType
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import torch
    from megatron.core.process_groups_config import ProcessGroupCollection

    from megatron.bridge.models.transformer_config import TransformerConfig


_DSA_LOGGING_MODULE = "megatron.core.transformer.experimental_attention_variant.dsa_logging"


def _get_dsa_logging(model_config: TransformerConfig) -> ModuleType | None:
    coefficient = getattr(model_config, "dsa_indexer_loss_coeff", None)
    if not isinstance(coefficient, Real) or coefficient <= 0:
        return None
    try:
        dsa_logging = import_module(_DSA_LOGGING_MODULE)
    except ModuleNotFoundError as error:
        if error.name != _DSA_LOGGING_MODULE:
            raise
        return None
    # Older MCore versions, including the measured private DSv4 pin, own their
    # tracker in dsa.py and do not expose the fixed-storage lifecycle API.
    if not hasattr(dsa_logging, "initialize_dsa_metric_tracker"):
        return None
    return dsa_logging


def initialize_dsa_metrics(
    model: list[torch.nn.Module], model_config: TransformerConfig, pg_collection: ProcessGroupCollection | None
) -> None:
    """Initialize PP-agreed tracker capacity before this model's first forward.

    Args:
        model: Newly constructed local model chunks, including wrappers.
        model_config: Model configuration containing the indexer-loss coefficient.
        pg_collection: Process groups owning this model.
    """
    dsa_logging = _get_dsa_logging(model_config)
    if dsa_logging is None:
        return
    should_track, _ = dsa_logging.resolve_dsa_metric_pg_collection(pg_collection)
    if not should_track:
        return
    # This is a new model lifecycle, before any graph owns the tracker storage.
    # Discard state from an earlier in-process run, including its process groups.
    dsa_logging.DSAIndexerLossLoggingHelper.tracker.clear()
    dsa_logging.initialize_dsa_metric_tracker(model, pg_collection)


def reset_dsa_metrics(model_config: TransformerConfig) -> None:
    """Clear values in place before a training attempt, retaining captured storage and groups."""
    dsa_logging = _get_dsa_logging(model_config)
    if dsa_logging is not None:
        dsa_logging.DSAIndexerLossLoggingHelper.clean_loss_in_tracker(preserve_groups=True)


def _num_indexer_executions(model_config: TransformerConfig) -> int:
    """Match MCore's indexer metric denominator, including shared and repeated MTP layers."""
    num_layers = model_config.num_layers
    mtp_layers = getattr(model_config, "mtp_num_layers", None) or 0
    hybrid_pattern = getattr(model_config, "hybrid_layer_pattern", None)
    hybrid = hybrid_pattern is not None
    layer_numbers = list(range(1, num_layers + 1))
    if getattr(model_config, "mtp_use_repeated_layer", False):
        layer_numbers.extend([num_layers + 1] * mtp_layers)
    else:
        layer_numbers.extend(range(num_layers + 1, num_layers + mtp_layers + 1))

    compress_ratios = getattr(model_config, "csa_compress_ratios", None)
    if compress_ratios is not None:
        if not hybrid:
            compress_ratios = [compress_ratios[number - 1] for number in layer_numbers]
        return sum(ratio == 4 for ratio in compress_ratios)

    from megatron.core.transformer.experimental_attention_variant.dsa import is_dsa_skip_topk_layer

    offset = getattr(model_config, "dsa_indexer_skip_topk_offset", None) or 0
    frequency = getattr(model_config, "dsa_indexer_topk_freq", None) or 1

    def computes_index(layer_number: int) -> bool:
        return not is_dsa_skip_topk_layer(layer_number, offset, frequency)

    if hybrid:
        from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols, parse_hybrid_pattern

        pattern = parse_hybrid_pattern(hybrid_pattern)
        main_pattern = (pattern.main_pattern or "").replace(Symbols.PIPE, "")
        count = sum(
            symbol == Symbols.DS_ATTENTION and computes_index(number)
            for number, symbol in enumerate(main_pattern, start=1)
        )
        mtp_count = sum(
            symbol == Symbols.DS_ATTENTION and computes_index(num_layers + number)
            for number, symbol in enumerate(pattern.mtp_pattern or "", start=1)
        )
        return count + mtp_count * pattern.mtp_num_depths
    return sum(computes_index(number) for number in layer_numbers)


def track_dsa_metrics(
    model_config: TransformerConfig,
    *,
    iteration: int,
    num_microbatches: int,
    writer: object | None,
    wandb_writer: object | None,
    total_loss_dict: dict,
    pg_collection: ProcessGroupCollection | None,
) -> None:
    """Reduce and clear indexer metrics on every participating rank, even without a writer.

    Args:
        model_config: Model configuration defining indexer executions and CP mode.
        iteration: Current optimizer-step number.
        num_microbatches: Nominal microbatch count from the global-batch calculator.
        writer: TensorBoard-compatible metric writer, or None.
        wandb_writer: Weights & Biases writer, or None.
        total_loss_dict: Accumulator for interval-average console metrics.
        pg_collection: Process groups owning this model, including any module collections.
    """
    dsa_logging = _get_dsa_logging(model_config)
    if dsa_logging is None:
        return
    should_track, metric_groups = dsa_logging.resolve_dsa_metric_pg_collection(pg_collection)
    if not should_track:
        return
    dynamic_cp = getattr(model_config, "dynamic_context_parallel", False)
    dynamic_cp_parent_group = None
    if dynamic_cp:
        if metric_groups is not None:
            dynamic_cp_parent_group = metric_groups.dp_cp
        else:
            from megatron.core import parallel_state

            dynamic_cp_parent_group = parallel_state.get_data_parallel_group(with_context_parallel=True)
    dsa_logging.DSAIndexerLossLoggingHelper.track_indexer_metrics(
        loss_scale=1 / num_microbatches,
        iteration=iteration,
        writer=writer,
        wandb_writer=wandb_writer,
        total_loss_dict=total_loss_dict,
        num_layers=model_config.num_layers + (getattr(model_config, "mtp_num_layers", None) or 0),
        num_indexer_layers=_num_indexer_executions(model_config),
        preserve_groups=getattr(model_config, "cuda_graph_impl", "none") != "none",
        dynamic_cp_parent_group=dynamic_cp_parent_group,
        configured_cp_size=model_config.context_parallel_size,
        pp_group=metric_groups.pp if metric_groups is not None else None,
        dp_group=metric_groups.dp if metric_groups is not None else None,
    )
