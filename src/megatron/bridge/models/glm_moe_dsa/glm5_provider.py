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

"""Hybrid model provider for the GLM-5 DSA family."""

from dataclasses import dataclass

from megatron.core.transformer.experimental_attention_variant.dsa import is_dsa_skip_topk_layer
from megatron.core.transformer.spec_utils import ModuleSpec

from megatron.bridge.models.glm_moe_dsa.glm5_hybrid import glm_hybrid_stack_spec
from megatron.bridge.models.hybrid.hybrid_provider import HybridModelProvider
from megatron.bridge.models.transformer_config import MLATransformerConfig


def glm_hybrid_pattern(*, num_layers: int, first_k_dense_replace: int) -> str:
    """Build an architecture-preserving pattern from HF block counts."""
    if num_layers < 1 or not 0 <= first_k_dense_replace <= num_layers:
        raise ValueError("GLM requires positive block count and a valid dense prefix.")
    return "D-" * first_k_dense_replace + "DE" * (num_layers - first_k_dense_replace)


def split_glm_pattern(pattern: str, block_counts: list[int]) -> str:
    """Place PP boundaries without splitting attention/MLP pairs."""
    main = pattern.split("/")[0].replace("|", "")
    if any(n < 1 for n in block_counts) or 2 * sum(block_counts) != len(main):
        raise ValueError("GLM pipeline block counts must cover the complete main pattern.")
    parts = []
    offset = 0
    for count in block_counts:
        parts.append(main[offset : offset + 2 * count])
        offset += 2 * count
    return "|".join(parts)


@dataclass
class GLM5ModelProvider(HybridModelProvider, MLATransformerConfig):
    """Construct GLM as DSA/dense/MoE Hybrid layers with MLA configuration."""

    def _resolve_hybrid_stack_spec(self) -> ModuleSpec:
        return glm_hybrid_stack_spec(self)

    def finalize(self) -> None:
        """Validate physical layout and preserve DSA sharing within each stage."""
        if not self.hybrid_layer_pattern:
            raise ValueError("GLM requires an explicit hybrid_layer_pattern.")
        main = self.hybrid_layer_pattern.split("/")[0]
        compact = main.replace("|", "")
        if len(compact) % 2 or any(compact[i : i + 2] not in ("D-", "DE") for i in range(0, len(compact), 2)):
            raise ValueError("GLM requires alternating DSA and dense/MoE layers.")
        if self.virtual_pipeline_model_parallel_size is not None:
            raise ValueError("GLM Hybrid recipes currently support non-VPP execution only.")
        if self.pipeline_model_parallel_layout is not None:
            raise ValueError("Use hybrid_layer_pattern pipe separators instead of a GPT pipeline layout.")
        if "|" not in main and self.pipeline_model_parallel_size > 1:
            blocks = len(compact) // 2
            # Balance complete DSA sharing groups, including the short dense prefix.
            # This also makes ordinary conversion --pp overrides safe for GLM-5.2.
            boundaries = [
                i
                for i in range(blocks)
                if not is_dsa_skip_topk_layer(
                    i + 1, self.dsa_indexer_skip_topk_offset or 0, self.dsa_indexer_topk_freq or 1
                )
            ] + [blocks]
            stages = self.pipeline_model_parallel_size
            if len(boundaries) - 1 < stages:
                raise ValueError("GLM has fewer independent DSA sharing groups than pipeline stages.")
            selected = [0]
            previous = 0
            for stage in range(1, stages):
                candidates = range(previous + 1, len(boundaries) - (stages - stage))
                previous = min(candidates, key=lambda index: abs(boundaries[index] * stages - blocks * stage))
                selected.append(boundaries[previous])
            selected.append(blocks)
            main = split_glm_pattern(main, [end - start for start, end in zip(selected, selected[1:])])
        segments = main.split("|")
        if len(segments) != self.pipeline_model_parallel_size:
            raise ValueError("GLM requires exactly one nonempty pattern segment per pipeline rank.")
        offset = 0
        for segment in segments:
            if not segment or len(segment) % 2:
                raise ValueError("GLM pipeline boundaries must preserve complete attention/MLP pairs.")
            if is_dsa_skip_topk_layer(
                offset + 1, self.dsa_indexer_skip_topk_offset or 0, self.dsa_indexer_topk_freq or 1
            ):
                raise ValueError(
                    "GLM pipeline stage starts on a DSA reuse layer; move the boundary to a compute layer."
                )
            offset += len(segment) // 2
        if self.recompute_granularity == "selective" and "core_attn" in (self.recompute_modules or []):
            raise ValueError(
                "GLM Hybrid selective core_attn recompute cannot preserve forward-local DSA state. "
                "Use full-stage recompute or selective modules outside core_attn."
            )
        if self.fp8 or self.fp4:
            raise ValueError(
                "GLM Hybrid training currently requires BF16/FP32; FP8 checkpoint import remains supported."
            )
        self.num_layers = len(compact)
        self.hybrid_layer_pattern = main
        self.mtp_hybrid_override_pattern = "DE" if self.mtp_num_layers else None
        # Hybrid symbols own MLP selection; keep the schedule dimensionally consistent.
        self.moe_layer_freq = [int(symbol == "E") for symbol in compact]
        super().finalize()


__all__ = ["GLM5ModelProvider"]
