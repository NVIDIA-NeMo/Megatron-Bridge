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

"""Builder-backed model configs for Gemma 4 text models.

Gemma4DenseModelConfig: dense E2B, E4B and 31B, built by ``Gemma4DenseModelBuilder``.
Gemma4ModelConfig: MoE 26B-A4B and similar, built by ``Gemma4ModelBuilder``.

Gemma-specific decoder fields live on the nested transformer config because the
Gemma 4 layers read them from the ``TransformerConfig`` they are built with.
"""

from dataclasses import dataclass
from typing import ClassVar

from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig
from megatron.bridge.models.transformer_config import TransformerConfig


def validate_gemma4_moe_orchestration(config: TransformerConfig) -> None:
    """Reject MCore execution modes bypassed by Gemma 4's custom MoE forward."""
    unsupported = []
    if config.transformer_impl == "inference_optimized":
        unsupported.append("transformer_impl='inference_optimized'")
    if config.cuda_graph_impl != "none":
        unsupported.append(f"cuda_graph_impl={config.cuda_graph_impl!r}")
    if config.moe_shared_expert_overlap:
        unsupported.append("moe_shared_expert_overlap=True")
    if config.mlp_chunks_for_prefill > 1:
        unsupported.append("mlp_chunks_for_prefill > 1")
    if config.mlp_chunks_for_training > 1:
        unsupported.append("mlp_chunks_for_training > 1")
    if config.inference_fuse_tp_communication:
        unsupported.append("inference_fuse_tp_communication=True")
    recompute_modules = set(config.recompute_modules or [])
    if config.recompute_granularity == "selective" and "layernorm" in recompute_modules:
        unsupported.append("selective layernorm recompute")
    offload_modules = set(config.offload_modules or [])
    if config.fine_grained_activation_offloading and "mlp_norm" in offload_modules:
        unsupported.append("MLP norm activation offloading")
    if unsupported:
        raise ValueError(
            "Gemma 4 MoE's separate router/shared/routed inputs do not yet support: " + ", ".join(unsupported)
        )


@dataclass
class Gemma4DenseTransformerConfig(TransformerConfig):
    """Gemma 4 dense decoder fields not represented by MCore's base config."""

    scale_embeddings_by_hidden_size: bool = True
    global_kv_channels: int = 512
    num_global_query_groups: int = 2
    attention_k_eq_v: bool = False
    sliding_window_rope_base: float = 10000.0
    full_attention_rope_base: float = 1000000.0
    full_attention_rope_partial_factor: float = 0.25
    num_kv_shared_layers: int = 18
    use_double_wide_mlp: bool = False
    per_layer_embed_vocab_size: int = 262144
    per_layer_embed_dim: int = 256
    final_logit_softcapping: float | None = 30.0

    def finalize(self) -> None:
        """Mark double-wide MLP checkpoints as heterogeneous before the base validation."""
        if self.use_double_wide_mlp:
            self.hetereogenous_dist_checkpoint = True
        super().finalize()


@dataclass
class Gemma4TransformerConfig(TransformerConfig):
    """Gemma 4 MoE decoder fields not represented by MCore's base config."""

    interleaved_attn_pattern: tuple[int, int] | list[str] = (5, 1)
    attention_k_eq_v: bool = False
    global_head_dim: int = 512
    num_global_key_value_heads: int = 2
    global_rotary_percent: float = 0.25
    final_logit_softcapping: float | None = 30.0

    def finalize(self) -> None:
        """Reject execution modes the custom MoE forward bypasses, then run the base validation."""
        validate_gemma4_moe_orchestration(self)
        super().finalize()


@dataclass(kw_only=True)
class Gemma4DenseModelConfig(BridgeGPTModelConfig):
    """Builder config for Gemma 4 dense text models (E2B, E4B, 31B)."""

    builder: ClassVar[str] = "megatron.bridge.models.gemma.gemma4_builder.Gemma4DenseModelBuilder"
    transformer_config_class: ClassVar[type[TransformerConfig]] = Gemma4DenseTransformerConfig


@dataclass(kw_only=True)
class Gemma4ModelConfig(BridgeGPTModelConfig):
    """Builder config for Gemma 4 MoE text models (26B-A4B and similar).

    ``rotary_base`` holds the (sliding, global) RoPE bases. The builder passes the
    sliding base to the GPT model and installs Gemma 4's dual RoPE afterwards.
    """

    builder: ClassVar[str] = "megatron.bridge.models.gemma.gemma4_builder.Gemma4ModelBuilder"
    transformer_config_class: ClassVar[type[TransformerConfig]] = Gemma4TransformerConfig

    rotary_base: int | tuple[int, int] | list[int] = (10_000, 1_000_000)


__all__ = [
    "Gemma4DenseModelConfig",
    "Gemma4DenseTransformerConfig",
    "Gemma4ModelConfig",
    "Gemma4TransformerConfig",
    "validate_gemma4_moe_orchestration",
]
