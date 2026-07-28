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

"""Legacy mapping declaration for the Qwen3 MoE migration proof."""

from __future__ import annotations

from megatron.bridge.legacy.mapping_compiler import compile_legacy_mapping_registry
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry


HF_MODEL_ID = "Qwen/Qwen3-30B-A3B"
HF_REVISION = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"  # pragma: allowlist secret
HF_ARCHITECTURE = "Qwen3MoeForCausalLM"
MIN_TRANSFORMERS_VERSION = "5.8.1"


class _Qwen2MoELegacyMappingDeclaration:
    """Inheritance carrier for the legacy mappings reused by Qwen3 MoE.

    The inherited declaration intentionally includes Qwen2 MoE QKV-bias and
    shared-expert entries that are stale for the pinned Qwen3-30B-A3B proof
    model. They exercise compilation only; semantic parity is asserted solely
    for Qwen3's actual QKV weight, router, and grouped/sequential expert
    parameters.
    """

    _DIRECT_MAPPING = {
        "embedding.word_embeddings.weight": "model.embed_tokens.weight",
        "decoder.final_layernorm.weight": "model.norm.weight",
        "output_layer.weight": "lm_head.weight",
    }
    _ATTENTION_MAPPING = {
        "self_attention.linear_proj.weight": ("model.layers.{layer_number}.self_attn.o_proj.weight",),
        "self_attention.linear_qkv.layer_norm_weight": ("model.layers.{layer_number}.input_layernorm.weight",),
        "self_attention.q_layernorm.weight": ("model.layers.{layer_number}.self_attn.q_norm.weight",),
        "self_attention.k_layernorm.weight": ("model.layers.{layer_number}.self_attn.k_norm.weight",),
        "self_attention.linear_qkv.weight": (
            "model.layers.{layer_number}.self_attn.q_proj.weight",
            "model.layers.{layer_number}.self_attn.k_proj.weight",
            "model.layers.{layer_number}.self_attn.v_proj.weight",
        ),
        "self_attention.linear_qkv.bias": (
            "model.layers.{layer_number}.self_attn.q_proj.bias",
            "model.layers.{layer_number}.self_attn.k_proj.bias",
            "model.layers.{layer_number}.self_attn.v_proj.bias",
        ),
    }
    _MLP_MAPPING = {
        "shared_experts.linear_fc1.weight": (
            "model.layers.{layer_number}.mlp.shared_expert.gate_proj.weight",
            "model.layers.{layer_number}.mlp.shared_expert.up_proj.weight",
        ),
        "pre_mlp_layernorm": ("model.layers.{layer_number}.post_attention_layernorm.weight",),
        "shared_experts.linear_fc2.weight": ("model.layers.{layer_number}.mlp.shared_expert.down_proj.weight",),
        "mlp.router.weight": ("model.layers.{layer_number}.mlp.gate.weight",),
        "shared_experts.gate_weight": ("model.layers.{layer_number}.mlp.shared_expert_gate.weight",),
        "mlp.experts.linear_fc1": (
            "model.layers.{layer_number}.mlp.experts.{expert_id}.gate_proj.weight",
            "model.layers.{layer_number}.mlp.experts.{expert_id}.up_proj.weight",
        ),
        "mlp.experts.linear_fc2": ("model.layers.{layer_number}.mlp.experts.{expert_id}.down_proj.weight",),
    }


class Qwen3MoELegacyMapping(_Qwen2MoELegacyMappingDeclaration):
    """Compile the inherited legacy Qwen3 MoE mapping declaration.

    This proof class is intentionally not registered with production bridge
    dispatch. Production Qwen3 MoE conversion continues to use
    ``Qwen3MoEBridge``. Its compiled registry retains stale inherited entries,
    so it is not claimed to be fully equivalent to the production registry.
    """

    @classmethod
    def mapping_registry(cls) -> MegatronMappingRegistry:
        """Compile the inherited declaration into current mapping primitives."""
        return compile_legacy_mapping_registry(
            direct_mapping=cls._DIRECT_MAPPING,
            attention_mapping=cls._ATTENTION_MAPPING,
            mlp_mapping=cls._MLP_MAPPING,
        )
