# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""MLA (Multi-Latent Attention) model providers.

This module provides GPT and Hybrid model providers for models using
Multi-Latent Attention.
"""

from dataclasses import dataclass

from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols
from megatron.core.transformer.experimental_attention_variant.dsa import (
    is_dsa_skip_topk_layer,
    source_dsa_compute_layer,
)

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.hybrid.hybrid_provider import HybridModelProvider
from megatron.bridge.models.transformer_config import MLATransformerConfig


@dataclass
class MLAModelProvider(MLATransformerConfig, GPTModelProvider):
    """Provider for models using Multi-Latent Attention (MLA).

    This class combines MLATransformerConfig (which provides MLA-specific fields
    like q_lora_rank, kv_lora_rank, qk_head_dim, v_head_dim) with GPTModelProvider
    (which provides the model instantiation logic).

    Model-specific defaults (normalization, activation, fusions, etc.) should be
    configured via MEGATRON_DEFAULTS in the respective bridge classes.

    Used by:
        - DeepSeek V2/V3
        - Kimi K2
        - Other MLA-based models
    """

    pass


def _validate_dsa_pipeline_segments(main_pattern: str, *, topk_freq: int, skip_topk_offset: int) -> None:
    """Ensure shared DSA indexers never reference a previous pipeline stage."""
    layer_offset = 0
    for segment_idx, segment in enumerate(main_pattern.split(Symbols.PIPE)):
        if segment_idx > 0 and (not segment or segment[0] != Symbols.DS_ATTENTION):
            raise ValueError(
                "Hybrid MLA pipeline separators must be placed before DSA layers "
                "so attention and FFN physical layers stay together."
            )
        for local_idx, layer_type in enumerate(segment):
            layer_number = layer_offset + local_idx + 1
            if layer_type != Symbols.DS_ATTENTION or not is_dsa_skip_topk_layer(
                layer_number, skip_topk_offset, topk_freq
            ):
                continue
            source_layer = source_dsa_compute_layer(layer_number, skip_topk_offset, topk_freq)
            if source_layer <= layer_offset:
                raise ValueError(
                    "Hybrid MLA pipeline segments cannot split DSA IndexShare groups: "
                    f"layer {layer_number} reuses indexer output from layer {source_layer}, "
                    "which is on a previous pipeline stage."
                )
        layer_offset += len(segment)


def _add_dsa_pipeline_separators(
    pattern: str,
    *,
    pipeline_model_parallel_size: int,
    topk_freq: int,
    skip_topk_offset: int,
) -> str:
    """Add balanced pipeline separators before full-indexer DSA layers."""
    main_pattern, separator, mtp_pattern = pattern.partition(Symbols.MTP_SEPARATOR)
    if Symbols.DS_ATTENTION not in main_pattern or pipeline_model_parallel_size <= 1:
        return pattern

    if Symbols.PIPE in main_pattern:
        num_segments = main_pattern.count(Symbols.PIPE) + 1
        if num_segments != pipeline_model_parallel_size:
            raise ValueError(
                f"hybrid_layer_pattern defines {num_segments} pipeline segments, but "
                f"pipeline_model_parallel_size is {pipeline_model_parallel_size}."
            )
        _validate_dsa_pipeline_segments(
            main_pattern,
            topk_freq=topk_freq,
            skip_topk_offset=skip_topk_offset,
        )
        return pattern

    candidate_boundaries = [
        layer_idx
        for layer_idx, layer_type in enumerate(main_pattern)
        if layer_idx > 0
        and layer_type == Symbols.DS_ATTENTION
        and not is_dsa_skip_topk_layer(layer_idx + 1, skip_topk_offset, topk_freq)
    ]
    required_boundaries = pipeline_model_parallel_size - 1
    if len(candidate_boundaries) < required_boundaries:
        raise ValueError(
            "Cannot split Hybrid MLA layers across pipeline stages without crossing a DSA "
            f"IndexShare group: found {len(candidate_boundaries)} valid boundaries for "
            f"pipeline_model_parallel_size={pipeline_model_parallel_size}."
        )

    selected_boundaries = []
    candidate_start = 0
    for stage_idx in range(1, pipeline_model_parallel_size):
        remaining_boundaries = pipeline_model_parallel_size - stage_idx - 1
        choices = candidate_boundaries[candidate_start : len(candidate_boundaries) - remaining_boundaries]
        target = len(main_pattern) * stage_idx / pipeline_model_parallel_size
        boundary = min(choices, key=lambda candidate: (abs(candidate - target), candidate))
        selected_boundaries.append(boundary)
        candidate_start = candidate_boundaries.index(boundary, candidate_start) + 1

    boundaries = set(selected_boundaries)
    segmented_pattern = "".join(
        (Symbols.PIPE if layer_idx in boundaries else "") + layer_type
        for layer_idx, layer_type in enumerate(main_pattern)
    )
    _validate_dsa_pipeline_segments(
        segmented_pattern,
        topk_freq=topk_freq,
        skip_topk_offset=skip_topk_offset,
    )
    return segmented_pattern + (separator + mtp_pattern if separator else "")


@dataclass
class HybridMLAModelProvider(HybridModelProvider, MLATransformerConfig):
    """Provider for Hybrid models using Multi-Latent Attention."""

    def finalize(self) -> None:
        """Finalize the provider after adding DSA-safe pipeline segments."""
        pattern_attr = "hybrid_layer_pattern"
        pattern = self.hybrid_layer_pattern
        if pattern is None and self.hybrid_override_pattern is not None:
            pattern_attr = "hybrid_override_pattern"
            pattern = self.hybrid_override_pattern

        main_pattern = pattern.split(Symbols.MTP_SEPARATOR)[0] if pattern is not None else ""
        if Symbols.DS_ATTENTION in main_pattern and self.pipeline_model_parallel_size > 1:
            if (
                self.num_layers_in_first_pipeline_stage is not None
                or self.num_layers_in_last_pipeline_stage is not None
            ):
                raise ValueError(
                    "Hybrid MLA derives pipeline stages from hybrid_layer_pattern; "
                    "num_layers_in_first_pipeline_stage and num_layers_in_last_pipeline_stage are unsupported."
                )
            pattern = _add_dsa_pipeline_separators(
                pattern,
                pipeline_model_parallel_size=self.pipeline_model_parallel_size,
                topk_freq=self.dsa_indexer_topk_freq or 1,
                skip_topk_offset=self.dsa_indexer_skip_topk_offset or 0,
            )
            setattr(self, pattern_attr, pattern)

        super().finalize()
