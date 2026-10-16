# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Copyright 2026 Reflection AI and the HuggingFace Inc. team. All rights reserved.
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
"""Beam configuration for Transformers builds without native Beam registration.

Configuration fields follow Hugging Face Transformers. This class supports Bridge's
configuration and safetensors conversion paths; it does not implement an HF model.
"""

# Source commit: 966f59c897c15db8a5e14882d086beba8d6d0b9f  # pragma: allowlist secret

from typing import Any

from huggingface_hub.dataclasses import strict
from transformers.configuration_utils import PreTrainedConfig
from transformers.modeling_rope_utils import RopeParameters


@strict
class BeamConfig(PreTrainedConfig):
    """Read Beam's published configuration without requiring its HF model implementation."""

    model_type = "beam"
    keys_to_ignore_at_inference = ["past_key_values"]

    vocab_size: int = 151936

    hidden_size: int = 5120
    intermediate_size: int = 20480
    num_hidden_layers: int = 52
    num_attention_heads: int = 48
    num_key_value_heads: int = 8
    hidden_act: str = "silu"
    max_position_embeddings: int = 8192
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    sliding_window: int | None = 512
    attention_dropout: float | int = 0.0
    moe_intermediate_size: int = 2560
    num_experts_per_tok: int = 8
    num_experts: int = 256
    norm_topk_prob: bool = True
    output_router_logits: bool = False
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    head_dim: int = 128

    layer_types: list[str] | None = None
    mlp_layer_types: list[str] | None = None

    n_group: int | None = 8
    topk_group: int | None = 4
    routed_scaling_factor: int | float = 2.5

    final_logit_softcapping: int | float | None = 30.0

    def __post_init__(self, **kwargs: Any) -> None:
        if self.layer_types is None:
            # Every sixth layer runs full attention, and so does the last one.
            self.layer_types = [
                "sliding_attention" if bool((i + 1) % 6) else "full_attention" for i in range(self.num_hidden_layers)
            ]
            self.layer_types[-1] = "full_attention"
        if self.mlp_layer_types is None:
            num_dense_layers = min(3, self.num_hidden_layers)
            self.mlp_layer_types = ["dense"] * num_dense_layers + ["sparse"] * (
                self.num_hidden_layers - num_dense_layers
            )
        if self.rope_parameters is None:
            # Keyed by layer type; `None` marks a layer type that runs without positional encoding.
            self.rope_parameters = {
                "sliding_attention": {"rope_type": "default", "rope_theta": self.default_theta},
                "full_attention": None,
            }
        super().__post_init__(**kwargs)

    def convert_rope_params_to_dict(self, **kwargs: Any) -> dict[str, Any]:
        # The generic converter assumes a flat mapping and would add top-level RoPE keys.
        return kwargs

    def validate_architecture(self) -> None:
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads: "
                f"{self.num_attention_heads} % {self.num_key_value_heads} != 0"
            )


__all__ = ["BeamConfig"]
