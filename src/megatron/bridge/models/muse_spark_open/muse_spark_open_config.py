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

"""Serializable configuration for the Muse Spark Open HybridModel integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from megatron.bridge.models.hybrid.model_config import BridgeHybridModelConfig
from megatron.bridge.models.muse_glimmer.muse_glimmer_config import MuseGlimmerTransformerConfig


@dataclass
class MuseSparkTransformerConfig(MuseGlimmerTransformerConfig):
    """Additional residual-gate configuration for the sparse decoder."""

    residual_gate_temperature: float = 0.3


@dataclass(kw_only=True)
class MuseSparkModelConfig(BridgeHybridModelConfig):
    """HF perception modules and an MCore Hybrid language model."""

    builder: ClassVar[str] = "megatron.bridge.models.muse_spark_open.MuseSparkModelBuilder"
    transformer_config_class: ClassVar[type] = MuseSparkTransformerConfig
    hf_config: dict[str, Any] = field(default_factory=dict)
    hf_model_text_only: bool = False
    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_audio_model: bool = False
    freeze_vision_projection: bool = False

    @property
    def special_token_ids(self) -> dict[str, int]:
        """Return the media placeholder IDs for collators."""
        if self.hf_model_text_only:
            return {}
        return {
            name: self.hf_config[key]
            for name, key in (("images", "image_token_id"), ("videos", "video_token_id"), ("audio", "audio_token_id"))
        }

    def finalize(self) -> None:
        """Validate the first supported execution path before model allocation."""
        config = self.transformer
        if config.expert_tensor_parallel_size != 1:
            raise ValueError("Muse Spark Open currently requires expert_tensor_parallel_size=1.")
        if config.transformer_impl != "transformer_engine":
            raise ValueError("Muse Spark Open currently uses Transformer Engine attention and grouped experts.")
        if config.mtp_num_layers or config.virtual_pipeline_model_parallel_size is not None:
            raise ValueError("The initial Muse Spark Open builder does not support MTP or virtual pipeline stages.")
        if config.fp8 or config.fp4:
            raise ValueError("Validate the unquantized Muse Spark Open path before enabling FP8 or FP4.")
        if config.cuda_graph_impl != "none" or config.overlap_moe_expert_parallel_comm:
            raise ValueError(
                "Muse Spark Open custom MoE normalization is not yet integrated with CUDA graphs or EP overlap."
            )
        if config.inference_fuse_tp_communication or config.flash_decode:
            raise ValueError(
                "Muse Spark Open requires the unfused inference path to preserve custom norms and NoPE layers."
            )
        if config.hidden_dropout or config.add_bias_linear or config.add_qkv_bias:
            raise ValueError("Muse Spark Open requires bias-free projections and zero hidden dropout.")
        if config.moe_shared_expert_intermediate_size is not None or config.moe_use_norm_before_up_proj:
            raise ValueError(
                "Muse Spark Open uses per-expert normalization, with no shared expert or post-combine norm."
            )
        if config.moe_apply_probs_on_input or config.moe_expert_capacity_factor is not None:
            raise ValueError(
                "Muse Spark Open requires dropless routing with probabilities applied after expert normalization."
            )
        if config.enable_mhc_connections or config.wide_residual is not None:
            raise ValueError("Muse Spark Open uses its own single-stream residual gates.")
        if config.num_layers % 2 or (self.hybrid_layer_pattern or "").replace("|", "") != "*E" * (
            config.num_layers // 2
        ):
            raise ValueError("Muse Spark Open requires an alternating attention/MoE Hybrid pattern.")
        if config.residual_gate_temperature <= 0:
            raise ValueError("residual_gate_temperature must be positive.")
        super().finalize()
