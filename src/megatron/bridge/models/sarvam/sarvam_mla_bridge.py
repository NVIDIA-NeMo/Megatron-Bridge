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

from functools import partial
from typing import Any

import torch.nn.functional as F
from megatron.core.models.gpt.gpt_model import GPTModel
from transformers import PretrainedConfig

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import AutoMapping, GatedMLPMapping
from megatron.bridge.models.deepseek.attention import get_deepseek_decoder_block_spec
from megatron.bridge.models.gpt.model_config import BridgeGPTMLAModelConfig
from megatron.bridge.models.sarvam.common import get_common_config
from megatron.bridge.models.sarvam.sarvam_provider import HAVE_TE, SarvamMLAModelProvider


@MegatronModelBridge.register_bridge(source="SarvamMLAForCausalLM", target=GPTModel, provider=SarvamMLAModelProvider)
class SarvamMLABridge(MegatronModelBridge):
    """
    Megatron Hub Bridge for Sarvam MLA Causal LM.

    This bridge handles the conversion between HuggingFace SarvamMLAForCausalLM
    and Megatron-Core GPTModel formats. Sarvam MLA models use multi-latent attention
    architecture.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("sarvamai/sarvam-105b", trust_remote_code=True)
        >>> model_config = bridge.get_model_config()
    """

    MODEL_CONFIG_CLASS = BridgeGPTMLAModelConfig
    USE_MODEL_CONFIG_FOR_CONVERSION = True

    def hf_config_to_model_config_kwargs(self, hf_config: PretrainedConfig) -> dict[str, Any]:
        """Convert a Hugging Face Sarvam MLA config to builder config kwargs.

        The architecture sizes and the YaRN settings come from the HF config; the
        remaining settings are fixed for the Sarvam MLA family, as
        ``SarvamMLAModelProvider`` sets them. Without a query LoRA, the DeepSeek
        block spec keeps the KV norm and omits the query norm that the checkpoint
        does not define.
        """
        config_kwargs = super().hf_config_to_model_config_kwargs(hf_config)
        config_kwargs.update(get_common_config(hf_config))
        config_kwargs.update(
            transformer_layer_spec=partial(
                get_deepseek_decoder_block_spec,
                use_transformer_engine=HAVE_TE,
                normalization="RMSNorm",
                vp_stage=None,
            ),
            # MLA
            kv_channels=hf_config.hidden_size // hf_config.num_attention_heads,
            q_lora_rank=None,
            kv_lora_rank=hf_config.kv_lora_rank,
            qk_head_dim=hf_config.qk_nope_head_dim,
            qk_pos_emb_head_dim=hf_config.qk_rope_head_dim,
            v_head_dim=hf_config.v_head_dim,
            multi_latent_attention=True,
            normalization="RMSNorm",
            activation_func=F.silu,
            gated_linear_unit=True,
            add_bias_linear=False,
            share_embeddings_and_output_weights=False,
            make_vocab_size_divisible_by=128,
            add_qkv_bias=False,
            qk_layernorm=True,
            init_method_std=0.006,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            layernorm_epsilon=1e-6,
            moe_aux_loss_coeff=0,
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
            rope_scaling=False,
            rope_scaling_factor=1.0,
            # SarvamMLAModelProvider keeps Megatron Core's defaults for these settings.
            cross_entropy_loss_fusion=False,
            deallocate_pipeline_outputs=False,
            gradient_accumulation_fusion=False,
            masked_softmax_fusion=False,
        )
        return config_kwargs

    def hf_config_to_provider_kwargs(self, hf_config: PretrainedConfig) -> dict[str, Any]:
        """Adapt the canonical builder mapping to the deprecated ``SarvamMLAModelProvider`` path."""
        return self.hf_config_to_model_config_kwargs(hf_config)

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
