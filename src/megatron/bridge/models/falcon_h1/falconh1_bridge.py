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

import logging

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ColumnParallelMapping,
    GatedMLPMapping,
    MambaConv1dMapping,
    MambaInProjMapping,
    QKVMapping,
    RowParallelMapping,
)
from megatron.bridge.models.falcon_h1.falconh1_provider import FalconH1ModelProvider
from megatron.bridge.models.falcon_h1.modeling_falconh1.falconh1_model import FalconH1Model
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


logger = logging.getLogger(__name__)


@MegatronModelBridge.register_bridge(
    source="FalconH1ForCausalLM",
    target=FalconH1Model,
    provider=FalconH1ModelProvider,
    model_type="falcon_h1",
)
class FalconH1Bridge(MegatronModelBridge):
    """
    Megatron Bridge for FalconH1 Causal LM.

    Handles conversion between HuggingFace FalconH1ForCausalLM and
    Megatron FalconH1Model formats, including weight mappings and
    configuration translation.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("tiiuae/Falcon-H1-7B-Instruct", trust_remote_code=True)
        >>> provider = bridge.to_megatron_provider()
    """

    CONFIG_MAPPING = MegatronModelBridge.CONFIG_MAPPING + [
        ("projectors_bias", "add_bias_linear"),
        ("mamba_d_state", "mamba_state_dim"),
        ("mamba_d_head", "mamba_head_dim"),
        ("mamba_n_heads", "mamba_num_heads"),
        ("mamba_n_groups", "mamba_num_groups"),
        ("mamba_expand", "expand"),
        ("mamba_d_conv", "d_conv"),
        ("mamba_conv_bias", "conv_bias"),
        ("mamba_chunk_size", "chunk_size"),
        ("mamba_rms_norm", "rmsnorm"),
        ("mamba_norm_before_gate", "norm_before_gate"),
        ("embedding_multiplier", "embedding_multiplier"),
        ("lm_head_multiplier", "lm_head_multiplier"),
        ("key_multiplier", "key_multiplier"),
        ("attention_in_multiplier", "attention_in_multiplier"),
        ("attention_out_multiplier", "attention_out_multiplier"),
        ("ssm_in_multiplier", "ssm_in_multiplier"),
        ("ssm_out_multiplier", "ssm_out_multiplier"),
        ("mlp_multipliers", "mlp_multipliers"),
        ("ssm_multipliers", "ssm_multipliers"),
    ]

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> FalconH1ModelProvider:
        """Convert HuggingFace Falcon H1 config to ``FalconH1ModelProvider``."""
        provider = super().provider_bridge(hf_pretrained)
        hf_config = hf_pretrained.config

        provider.position_embedding_type = "rope"
        provider.rotary_percent = 1.0
        provider.rotary_base = int(provider.rotary_base)
        provider.normalization = "RMSNorm"
        provider.gated_linear_unit = True
        provider.add_bias_linear = getattr(hf_config, "projectors_bias", provider.add_bias_linear)
        provider.hidden_dropout = getattr(hf_config, "hidden_dropout", 0.0)

        # Falcon H1 checkpoints represent every decoder block as a parallel
        # Mamba + attention + MLP layer.
        provider.falconh1_ratio = 1.0
        provider.use_mamba = True
        provider.use_attention = True
        provider.use_mlp = True

        provider.mlp_multipliers = tuple(provider.mlp_multipliers)
        provider.ssm_multipliers = tuple(provider.ssm_multipliers)

        return provider

    @classmethod
    def megatron_to_hf_config(cls, provider: FalconH1ModelProvider) -> dict:
        """Convert ``FalconH1ModelProvider`` to HuggingFace config fields."""
        hf_config = super(FalconH1Bridge, cls).megatron_to_hf_config(provider)

        hf_config["mamba_d_ssm"] = provider.mamba_num_heads * provider.mamba_head_dim
        hf_config["mamba_proj_bias"] = provider.add_bias_linear
        hf_config["mamba_use_mlp"] = provider.use_mlp

        return hf_config

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Define parameter mappings between Megatron and HuggingFace formats."""

        # Simple 1:1 parameter mappings
        param_mappings = {
            # MLP mappings (FalconH1 uses gate_proj/up_proj combined)
            "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.feed_forward.down_proj.weight",
            # Attention output projection
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            # Shared input norm for parallel Mamba and attention branches
            "decoder.layers.*.norm.weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.pre_ff_layernorm.weight",
            "decoder.final_norm.weight": "model.final_layernorm.weight",
            # Embeddings and output
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "output_layer.weight": "lm_head.weight",
        }

        mapping_list = []

        # Convert simple mappings to AutoMapping objects
        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        # Mamba mixer components with proper tensor parallel handling
        for mamba_component in ["A_log", "D", "dt_bias", "norm.weight"]:
            mapping_list.append(
                ColumnParallelMapping(
                    megatron_param=f"decoder.layers.*.mamba_mixer.{mamba_component}",
                    hf_param=f"model.layers.*.mamba.{mamba_component}",
                )
            )

        # Mamba output projection (row parallel)
        mapping_list.append(
            RowParallelMapping(
                megatron_param="decoder.layers.*.mamba_mixer.out_proj.weight",
                hf_param="model.layers.*.mamba.out_proj.weight",
            )
        )

        # Mamba input projection with special handling
        mapping_list.append(
            MambaInProjMapping(
                megatron_param="decoder.layers.*.mamba_mixer.in_proj.weight",
                hf_param="model.layers.*.mamba.in_proj.weight",
            )
        )

        # Mamba conv1d components. Keep legacy dotted names for older Megatron-Core pins.
        for megatron_conv1d_param, hf_conv1d_param in [
            ("conv1d_weight", "conv1d.weight"),
            ("conv1d_bias", "conv1d.bias"),
            ("conv1d.weight", "conv1d.weight"),
            ("conv1d.bias", "conv1d.bias"),
        ]:
            mapping_list.append(
                MambaConv1dMapping(
                    megatron_param=f"decoder.layers.*.mamba_mixer.{megatron_conv1d_param}",
                    hf_param=f"model.layers.*.mamba.{hf_conv1d_param}",
                )
            )

        # QKV mapping - combine separate Q, K, V into single QKV matrix
        mapping_list.append(
            QKVMapping(
                megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                q="model.layers.*.self_attn.q_proj.weight",
                k="model.layers.*.self_attn.k_proj.weight",
                v="model.layers.*.self_attn.v_proj.weight",
            )
        )

        # Handle up_proj separately if needed (FalconH1 might combine gate and up)
        mapping_list.append(
            AutoMapping(
                megatron_param="decoder.layers.*.mlp.linear_fc1_up.weight",
                hf_param="model.layers.*.feed_forward.up_proj.weight",
            )
        )

        # Gated MLP: Combine gate and up projection matrices into single FC1 matrix
        mapping_list.append(
            GatedMLPMapping(
                megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                gate="model.layers.*.feed_forward.gate_proj.weight",
                up="model.layers.*.feed_forward.up_proj.weight",
            )
        )

        return MegatronMappingRegistry(*mapping_list)
