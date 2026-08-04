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

from typing import TYPE_CHECKING

import torch
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer import MLATransformerConfig, ModuleSpec

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import AutoMapping, GatedMLPMapping
from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.sarvam.common import get_common_config


if TYPE_CHECKING:
    from megatron.bridge.models.sarvam.sarvam_provider import SarvamMLAModelProvider


try:
    import transformer_engine  # noqa: F401

    HAVE_TE = True
except (ImportError, ModuleNotFoundError):
    HAVE_TE = False


def sarvam_mla_layer_spec(config: BridgeGPTModelConfig, vp_stage: int | None = None) -> ModuleSpec:
    """Build the Sarvam MLA decoder block with the available backend."""
    return get_gpt_decoder_block_spec(
        config.transformer,
        use_transformer_engine=HAVE_TE,
        normalization="RMSNorm",
        vp_stage=vp_stage,
    )


@MegatronModelBridge.register_bridge(source="SarvamMLAForCausalLM", target=GPTModel)
class SarvamMLABridge(MegatronModelBridge):
    """
    Megatron Hub Bridge for Sarvam MLA Causal LM.

    This bridge handles the conversion between HuggingFace SarvamMLAForCausalLM
    and Megatron-Core GPTModel formats. Sarvam MLA models use multi-latent attention
    architecture.
    """

    MODEL_CONFIG_CLASS = BridgeGPTModelConfig
    TRANSFORMER_CONFIG_CLASS = MLATransformerConfig

    def hf_config_to_model_config_kwargs(self, hf_config) -> dict:
        """Map Sarvam MLA fields before constructing the exact MCore config."""
        kwargs = super().hf_config_to_model_config_kwargs(hf_config)
        kwargs.update(
            transformer_layer_spec=sarvam_mla_layer_spec,
            kv_channels=hf_config.hidden_size // hf_config.num_attention_heads,
            moe_ffn_hidden_size=hf_config.moe_intermediate_size,
            num_moe_experts=hf_config.num_experts,
            moe_router_topk=hf_config.num_experts_per_tok,
            moe_shared_expert_intermediate_size=hf_config.num_shared_experts * hf_config.moe_intermediate_size,
            moe_layer_freq=[0] * hf_config.first_k_dense_replace
            + [1] * (hf_config.num_hidden_layers - hf_config.first_k_dense_replace),
            normalization="RMSNorm",
            gated_linear_unit=True,
            add_bias_linear=False,
            add_qkv_bias=False,
            qk_layernorm=True,
            init_method_std=0.006,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            layernorm_epsilon=1e-6,
            moe_aux_loss_coeff=0.0,
            moe_router_pre_softmax=True,
            moe_router_enable_expert_bias=True,
            moe_router_bias_update_rate=1e-3,
            moe_grouped_gemm=True,
            moe_permute_fusion=True,
            moe_router_topk_scaling_factor=2.5,
            moe_shared_expert_overlap=False,
            moe_router_dtype="fp32",
            moe_router_score_function="sigmoid",
            moe_token_dispatcher_type="alltoall",
            attention_softmax_in_fp32=True,
            persist_layer_norm=True,
            cross_entropy_fusion_impl="te",
            cp_comm_type="p2p",
            recompute_granularity="selective",
            recompute_modules=["moe"],
            multi_latent_attention=True,
            share_embeddings_and_output_weights=False,
            make_vocab_size_divisible_by=128,
        )
        return kwargs

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> "SarvamMLAModelProvider":
        from megatron.bridge.models.sarvam.sarvam_provider import SarvamMLAModelProvider

        hf_config = hf_pretrained.config
        config = get_common_config(hf_pretrained)

        config["fp16"] = self.dtype_from_hf(hf_config, default=torch.float32) == torch.float16
        config["bf16"] = self.dtype_from_hf(hf_config, default=torch.float32) == torch.bfloat16
        config["params_dtype"] = self.dtype_from_hf(hf_config, default=torch.float32)
        config["kv_channels"] = hf_config.hidden_size // hf_config.num_attention_heads

        # MLA
        config["kv_lora_rank"] = hf_config.kv_lora_rank
        config["qk_head_dim"] = hf_config.qk_nope_head_dim
        config["qk_pos_emb_head_dim"] = hf_config.qk_rope_head_dim
        config["v_head_dim"] = hf_config.v_head_dim

        return SarvamMLAModelProvider(**config)

    def mapping_registry(self) -> MegatronMappingRegistry:
        param_mappings = {
            # Embed
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            # Attention
            "decoder.layers.*.input_layernorm.weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            #  In sarvam, HF weight `model.layers.*.post_attention_layernorm.weight` is mapped to the following mcore weights depending on the layer type:
            #  (a) `decoder.layers.*.pre_mlp_layernorm.weight`, if the layer is MoE
            #  (b) `decoder.layers.*.mlp.linear_fc1.layer_norm_weight`, if the layer is dense
            "decoder.layers.*.pre_mlp_layernorm.weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.layers.*.self_attention.linear_q_proj.weight": "model.layers.*.self_attn.q_proj.weight",
            "decoder.layers.*.self_attention.linear_kv_down_proj.weight": "model.layers.*.self_attn.kv_a_proj_with_mqa.weight",
            "decoder.layers.*.self_attention.linear_kv_up_proj.weight": "model.layers.*.self_attn.kv_b_proj.weight",
            "decoder.layers.*.self_attention.linear_kv_up_proj.layer_norm_weight": "model.layers.*.self_attn.kv_a_layernorm.weight",
            # Mcore local spec
            "decoder.layers.*.self_attention.kv_layernorm.weight": "model.layers.*.self_attn.kv_a_layernorm.weight",
            # Dense MLP
            "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
            # Moe
            "decoder.layers.*.mlp.experts.linear_fc2.weight*": "model.layers.*.mlp.experts.*.down_proj.weight",
            "decoder.layers.*.mlp.shared_experts.linear_fc2.weight": "model.layers.*.mlp.shared_experts.down_proj.weight",
            "decoder.layers.*.mlp.router.expert_bias": "model.layers.*.mlp.gate.e_score_correction_bias",
            "decoder.layers.*.mlp.router.weight": "model.layers.*.mlp.gate.weight",
            # LM Head
            "decoder.final_layernorm.weight": "model.norm.weight",
            "output_layer.weight": "lm_head.weight",
        }

        mapping_list = []
        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(hf_param=hf_param, megatron_param=megatron_param))

        mapping_list.extend(
            [
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                    gate="model.layers.*.mlp.gate_proj.weight",
                    up="model.layers.*.mlp.up_proj.weight",
                ),
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.experts.linear_fc1.weight*",
                    gate="model.layers.*.mlp.experts.*.gate_proj.weight",
                    up="model.layers.*.mlp.experts.*.up_proj.weight",
                ),
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.shared_experts.linear_fc1.weight",
                    gate="model.layers.*.mlp.shared_experts.gate_proj.weight",
                    up="model.layers.*.mlp.shared_experts.up_proj.weight",
                ),
            ]
        )

        return MegatronMappingRegistry(*mapping_list)
