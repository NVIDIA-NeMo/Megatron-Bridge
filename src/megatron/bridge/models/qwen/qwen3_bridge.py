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

import torch
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols
from megatron.core.models.hybrid.hybrid_model import HybridModel
from transformers import Qwen3ForCausalLM

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
)
from megatron.bridge.models.qwen.qwen_hybrid import (
    QwenHybridModelProvider,
    configure_qwen_hybrid_layers,
    qwen_logical_layer_count,
    qwen_physical_layer_indices,
)


@MegatronModelBridge.register_bridge(
    source=Qwen3ForCausalLM,
    target=HybridModel,
    provider=QwenHybridModelProvider,
    model_type="qwen3",
)
class Qwen3Bridge(MegatronModelBridge):
    """
    Megatron Bridge for Qwen3 Causal LM.

    This bridge handles the conversion between HuggingFace Qwen3ForCausalLM
    and Megatron-Core HybridModel formats. Qwen3 differs from Qwen2 by using
    QK layernorm and no QKV bias.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("Qwen/Qwen3-1.7B")
        >>> provider = bridge.to_megatron_provider()
    """

    @classmethod
    def megatron_to_hf_config(cls, provider) -> dict:
        """Convert a Hybrid Qwen3 provider to a Hugging Face config dictionary."""
        hf_config = super().megatron_to_hf_config(provider)
        logical_layer_count = qwen_logical_layer_count(provider.hybrid_layer_pattern)
        if logical_layer_count is not None:
            hf_config["num_hidden_layers"] = logical_layer_count
        return hf_config

    def provider_bridge(self, hf_pretrained):
        """Convert a Hugging Face Qwen3 config to HybridModelProvider."""
        provider = super().provider_bridge(hf_pretrained)
        hf_config = hf_pretrained.config

        provider.normalization = "RMSNorm"
        provider.gated_linear_unit = True
        provider.add_bias_linear = False
        provider.add_qkv_bias = False  # Qwen3 does NOT have QKV bias (unlike Qwen2)
        provider.hidden_dropout = 0.0
        provider.qk_layernorm = True  # Qwen3 uses QK layernorm
        provider.autocast_dtype = torch.bfloat16
        provider.share_embeddings_and_output_weights = getattr(hf_config, "tie_word_embeddings", True)

        configure_qwen_hybrid_layers(
            provider,
            num_logical_layers=hf_config.num_hidden_layers,
            mlp_symbols=Symbols.MLP,
            mtp_mlp_symbol=Symbols.MLP,
        )

        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Return the MegatronMappingRegistry for Qwen3 parameter conversion.

        Covers all Megatron-Core parameter names for both the standard decoder
        layers and the MTP (Multi-Token Prediction) transformer layers that are
        present when ``mtp_num_layers >= 1``.

        Simple 1:1 renames are expressed as :class:`AutoMapping` entries.
        The fused QKV matrix is handled by :class:`QKVMapping` and the gated
        MLP gate+up projection by :class:`GatedMLPMapping`.
        """
        param_mappings = {
            # Embedding and output
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "output_layer.weight": "lm_head.weight",
            "decoder.final_norm.weight": "model.norm.weight",
            # MTP projection and norms (used when mtp_num_layers >= 1)
            "mtp.layers.0.eh_proj.weight": "mtp.fc.weight",
            "mtp.layers.0.enorm.weight": "mtp.pre_fc_norm_embedding.weight",
            "mtp.layers.0.hnorm.weight": "mtp.pre_fc_norm_hidden.weight",
            "mtp.layers.0.final_layernorm.weight": "mtp.norm.weight",
            # MTP transformer layer attention
            "mtp.layers.0.mtp_model_layer.layers.0.self_attention.linear_qkv.layer_norm_weight": "mtp.layers.0.input_layernorm.weight",
            "mtp.layers.0.mtp_model_layer.layers.0.self_attention.q_layernorm.weight": "mtp.layers.0.self_attn.q_norm.weight",
            "mtp.layers.0.mtp_model_layer.layers.0.self_attention.k_layernorm.weight": "mtp.layers.0.self_attn.k_norm.weight",
            "mtp.layers.0.mtp_model_layer.layers.0.self_attention.linear_proj.weight": "mtp.layers.0.self_attn.o_proj.weight",
            # MTP transformer layer MLP
            "mtp.layers.0.mtp_model_layer.layers.1.mlp.linear_fc1.layer_norm_weight": "mtp.layers.0.post_attention_layernorm.weight",
            "mtp.layers.0.mtp_model_layer.layers.1.mlp.linear_fc2.weight": "mtp.layers.0.mlp.down_proj.weight",
        }

        mapping_list = [AutoMapping(megatron_param=k, hf_param=v) for k, v in param_mappings.items()]

        for logical_layer_idx in range(self.hf_config.num_hidden_layers):
            attention_layer_idx, mlp_layer_idx = qwen_physical_layer_indices(logical_layer_idx)
            hf_layer = f"model.layers.{logical_layer_idx}"
            attention_layer = f"decoder.layers.{attention_layer_idx}.self_attention"
            mlp_layer = f"decoder.layers.{mlp_layer_idx}.mlp"
            mapping_list.extend(
                [
                    AutoMapping(
                        megatron_param=f"{attention_layer}.linear_qkv.layer_norm_weight",
                        hf_param=f"{hf_layer}.input_layernorm.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"{attention_layer}.q_layernorm.weight",
                        hf_param=f"{hf_layer}.self_attn.q_norm.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"{attention_layer}.k_layernorm.weight",
                        hf_param=f"{hf_layer}.self_attn.k_norm.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"{attention_layer}.linear_proj.weight",
                        hf_param=f"{hf_layer}.self_attn.o_proj.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"{mlp_layer}.linear_fc1.layer_norm_weight",
                        hf_param=f"{hf_layer}.post_attention_layernorm.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"{mlp_layer}.linear_fc2.weight",
                        hf_param=f"{hf_layer}.mlp.down_proj.weight",
                    ),
                    QKVMapping(
                        megatron_param=f"{attention_layer}.linear_qkv.weight",
                        q=f"{hf_layer}.self_attn.q_proj.weight",
                        k=f"{hf_layer}.self_attn.k_proj.weight",
                        v=f"{hf_layer}.self_attn.v_proj.weight",
                    ),
                    GatedMLPMapping(
                        megatron_param=f"{mlp_layer}.linear_fc1.weight",
                        gate=f"{hf_layer}.mlp.gate_proj.weight",
                        up=f"{hf_layer}.mlp.up_proj.weight",
                    ),
                ]
            )

        mapping_list.extend(
            [
                # MTP QKV: same split/merge as decoder layers
                QKVMapping(
                    megatron_param="mtp.layers.*.mtp_model_layer.layers.0.self_attention.linear_qkv.weight",
                    q="mtp.layers.*.self_attn.q_proj.weight",
                    k="mtp.layers.*.self_attn.k_proj.weight",
                    v="mtp.layers.*.self_attn.v_proj.weight",
                ),
                # MTP Gated MLP
                GatedMLPMapping(
                    megatron_param="mtp.layers.0.mtp_model_layer.layers.1.mlp.linear_fc1.weight",
                    gate="mtp.layers.0.mlp.gate_proj.weight",
                    up="mtp.layers.0.mlp.up_proj.weight",
                ),
            ]
        )

        return MegatronMappingRegistry(*mapping_list)
