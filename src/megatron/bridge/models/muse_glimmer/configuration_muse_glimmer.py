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

"""Transformers compatibility configuration for Muse Glimmer.

Muse Glimmer landed after the Transformers version currently supported by
Megatron Bridge.  Weight import only needs a parsed configuration and the lazy
safetensors state source, so keep a small configuration fallback here instead
of requiring the full Hugging Face modeling implementation at runtime.
"""

from __future__ import annotations

from typing import Any

from transformers import AutoConfig, PretrainedConfig


try:
    from transformers import MuseGlimmerConfig, MuseGlimmerTextConfig, MuseGlimmerVisionConfig
except ImportError:

    class MuseGlimmerTextConfig(PretrainedConfig):
        """Configuration for the Muse Glimmer text decoder."""

        model_type = "muse_glimmer_text"

        def __init__(
            self,
            *,
            vocab_size: int = 202_048,
            hidden_size: int = 6_656,
            intermediate_size: int = 19_968,
            num_hidden_layers: int = 52,
            num_attention_heads: int = 32,
            num_key_value_heads: int = 2,
            head_dim: int = 128,
            hidden_activation: str = "silu",
            max_position_embeddings: int = 131_072,
            rms_norm_eps: float = 1e-5,
            sliding_window: int = 2_048,
            final_logit_softcapping: float = 20.0,
            qk_scale_factor: float = 3.87,
            output_multiplier: float = 0.19611613513818404,
            post_norm_eps: float = 1e-8,
            layer_types: list[str] | None = None,
            layer_rope_theta: list[float] | None = None,
            rope_parameters: dict[str, Any] | None = None,
            attention_bias: bool = False,
            attention_dropout: float = 0.0,
            initializer_range: float = 0.02,
            tie_word_embeddings: bool = False,
            bos_token_id: int | None = 200_000,
            eos_token_id: int | list[int] | None = 200_001,
            pad_token_id: int | None = None,
            **kwargs: Any,
        ) -> None:
            super().__init__(
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                tie_word_embeddings=tie_word_embeddings,
                **kwargs,
            )
            self.vocab_size = vocab_size
            self.hidden_size = hidden_size
            self.intermediate_size = intermediate_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.num_key_value_heads = num_key_value_heads
            self.head_dim = head_dim
            self.hidden_activation = hidden_activation
            self.max_position_embeddings = max_position_embeddings
            self.rms_norm_eps = rms_norm_eps
            self.sliding_window = sliding_window
            self.final_logit_softcapping = final_logit_softcapping
            self.qk_scale_factor = qk_scale_factor
            self.output_multiplier = output_multiplier
            self.post_norm_eps = post_norm_eps
            self.attention_bias = attention_bias
            self.attention_dropout = attention_dropout
            self.initializer_range = initializer_range
            self.rope_parameters = rope_parameters or {"rope_theta": 500_000.0, "rope_type": "default"}
            self.layer_types = layer_types or [
                "full_attention" if (num_hidden_layers - 1 - index) % 4 == 0 else "sliding_attention"
                for index in range(num_hidden_layers)
            ]
            self.layer_rope_theta = layer_rope_theta or [
                0.0 if layer_type == "full_attention" else float(self.rope_parameters["rope_theta"])
                for layer_type in self.layer_types
            ]

    class MuseGlimmerVisionConfig(PretrainedConfig):
        """Configuration for the Muse Glimmer vision encoder."""

        model_type = "muse_glimmer_vision"

        def __init__(
            self,
            *,
            hidden_size: int = 1_536,
            intermediate_size: int = 8_960,
            num_hidden_layers: int = 50,
            num_attention_heads: int = 16,
            patch_size: int = 14,
            patch_temporal: int = 2,
            merge_size: int = 2,
            pos_emb_height: int = 32,
            pos_emb_width: int = 32,
            max_position_embeddings: int = 1_024,
            layer_norm_eps: float = 1e-5,
            hidden_act: str = "gelu",
            layer_types: list[str] | None = None,
            rope_parameters: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> None:
            super().__init__(**kwargs)
            self.hidden_size = hidden_size
            self.intermediate_size = intermediate_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.patch_size = patch_size
            self.patch_temporal = patch_temporal
            self.merge_size = merge_size
            self.pos_emb_height = pos_emb_height
            self.pos_emb_width = pos_emb_width
            self.max_position_embeddings = max_position_embeddings
            self.layer_norm_eps = layer_norm_eps
            self.hidden_act = hidden_act
            self.rope_parameters = rope_parameters or {"rope_theta": 10_000.0, "rope_type": "default"}
            self.layer_types = layer_types or [
                "full_attention" if index % 4 == 3 or index == num_hidden_layers - 1 else "window_attention"
                for index in range(num_hidden_layers)
            ]

    class MuseGlimmerConfig(PretrainedConfig):
        """Top-level configuration for Muse Glimmer."""

        model_type = "muse_glimmer"
        sub_configs = {
            "text_config": MuseGlimmerTextConfig,
            "vision_config": MuseGlimmerVisionConfig,
        }

        def __init__(
            self,
            *,
            text_config: dict[str, Any] | PretrainedConfig | None = None,
            vision_config: dict[str, Any] | PretrainedConfig | None = None,
            image_token_id: int = 200_092,
            video_token_id: int = 200_091,
            out_hidden_size: int = 6_144,
            projector_hidden_size: int = 4_096,
            projector_hidden_act: str = "gelu",
            **kwargs: Any,
        ) -> None:
            super().__init__(**kwargs)
            self.text_config = (
                text_config
                if isinstance(text_config, PretrainedConfig)
                else MuseGlimmerTextConfig(**(text_config or {}))
            )
            self.vision_config = (
                vision_config
                if isinstance(vision_config, PretrainedConfig)
                else MuseGlimmerVisionConfig(**(vision_config or {}))
            )
            self.image_token_id = image_token_id
            self.video_token_id = video_token_id
            self.out_hidden_size = out_hidden_size
            self.projector_hidden_size = projector_hidden_size
            self.projector_hidden_act = projector_hidden_act

    AutoConfig.register(MuseGlimmerConfig.model_type, MuseGlimmerConfig, exist_ok=True)


__all__ = ["MuseGlimmerConfig", "MuseGlimmerTextConfig", "MuseGlimmerVisionConfig"]
