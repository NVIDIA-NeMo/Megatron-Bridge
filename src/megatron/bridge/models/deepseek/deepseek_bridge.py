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

from megatron.core.models.gpt.gpt_model import GPTModel

from megatron.bridge.models.conversion.param_mapping import GatedMLPMapping
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import AutoMapping
from megatron.bridge.models.deepseek.deepseek_provider import DeepSeekProvider
from megatron.bridge.models.conversion.utils import get_causal_lm_class_via_auto_map
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


class DeepSeekBridge(MegatronModelBridge):
    """
    Megatron Hub Bridge for DeepSeekV2 and DeepSeekV3 Causal LM.

    As a user you would not use this bridge directly, but through `AutoBridge`.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("deepseek-ai/DeepSeek-V2-Lite")
        >>> provider = bridge.to_megatron_provider()
    """

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> DeepSeekProvider:
        hf_config = hf_pretrained.config

        optional_kwargs = {}
        # Not all deepseek configs have aux_loss_alpha
        if hasattr(hf_config, "aux_loss_alpha"):
            optional_kwargs["moe_aux_loss_coeff"] = hf_config.aux_loss_alpha

        n_moe_layers = hf_config.num_hidden_layers - hf_config.first_k_dense_replace
        provider = DeepSeekProvider(
            num_layers=hf_config.num_hidden_layers,
            hidden_size=hf_config.hidden_size,
            ffn_hidden_size=hf_config.intermediate_size,
            num_attention_heads=hf_config.num_attention_heads,
            q_lora_rank=hf_config.q_lora_rank,
            kv_channels=hf_config.num_key_value_heads,
            num_moe_experts=hf_config.n_routed_experts,
            moe_ffn_hidden_size=hf_config.moe_intermediate_size,  # Maps to moe_intermediate_size in HF
            moe_shared_expert_intermediate_size=hf_config.moe_intermediate_size * hf_config.n_shared_experts,
            moe_layer_freq=[0] * hf_config.first_k_dense_replace + [1] * n_moe_layers,
            moe_router_topk=hf_config.num_experts_per_tok,  # Maps to num_experts_per_tok in HF
            moe_router_num_groups=hf_config.n_group,
            moe_router_group_topk=hf_config.topk_group,
            moe_router_topk_scaling_factor=hf_config.routed_scaling_factor,
            kv_lora_rank=hf_config.kv_lora_rank,
            qk_head_dim=hf_config.qk_nope_head_dim,
            qk_pos_emb_head_dim=hf_config.qk_rope_head_dim,
            v_head_dim=hf_config.v_head_dim,
            init_method_std=hf_config.initializer_range,
            layernorm_epsilon=hf_config.rms_norm_eps,
            gated_linear_unit=True,
            make_vocab_size_divisible_by=self.make_vocab_size_divisible_by(hf_config.vocab_size),
            rotary_base=hf_config.rope_theta,
            share_embeddings_and_output_weights=getattr(hf_config, "tie_word_embeddings", False),
            vocab_size=hf_config.vocab_size,
            seq_length=hf_config.max_position_embeddings,
            fp16=(self.dtype_from_hf(hf_config, default=torch.float32) == torch.float16),
            bf16=(self.dtype_from_hf(hf_config, default=torch.float32) == torch.bfloat16),
            params_dtype=self.dtype_from_hf(hf_config, default=torch.float32),
            generation_config=hf_pretrained.generation_config,
            qk_layernorm=True,  # Qwen3 MoE uses QK layernorm
            moe_grouped_gemm=True,
            **optional_kwargs,
        )

        provider.gradient_accumulation_fusion = False

        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        # Return MegatronMappingRegistry containing parameter mappings from HF to Megatron format
        # First create simple 1:1 parameter mappings using a dictionary for readability

        # Dictionary maps HF parameter names -> Megatron parameter names
        # Supports wildcard (*) patterns for layer-specific parameters
        param_mappings = {
            # Embed
            "model.embed_tokens.weight": "embedding.word_embeddings.weight",
            # Attention
            "model.layers.*.input_layernorm.weight": "decoder.layers.*.input_layernorm.weight",
            "model.layers.*.self_attn.o_proj.weight": "decoder.layers.*.self_attention.linear_proj.weight",
            "model.layers.*.self_attn.q_a_proj.weight": "decoder.layers.*.self_attention.linear_q_down_proj.weight",
            "model.layers.*.self_attn.q_b_proj.weight": "decoder.layers.*.self_attention.linear_q_up_proj.weight",
            "model.layers.*.self_attn.kv_a_proj_with_mqa.weight": "decoder.layers.*.self_attention.linear_kv_down_proj.weight",
            "model.layers.*.self_attn.q_a_layernorm.weight": "decoder.layers.*.self_attention.linear_q_up_proj.layer_norm_weight",
            "model.layers.*.self_attn.kv_b_proj.weight": "decoder.layers.*.self_attention.linear_kv_up_proj.weight",
            "model.layers.*.self_attn.kv_a_layernorm.weight": "decoder.layers.*.self_attention.linear_kv_up_proj.layer_norm_weight",
            "model.layers.*.post_attention_layernorm.weight": "decoder.layers.*.pre_mlp_layernorm.weight",
            # Dense MLP
            # model.layers.*.mlp.{gate|up}_proj.weight: model.layers.*.mlp.linear_fc1.weight
            "model.layers.*.mlp.down_proj.weight": "decoder.layers.*.mlp.linear_fc2.weight",
            # MoE
            "model.layers.*.mlp.gate.weight": "decoder.layers.*.mlp.router.weight",
            # model.layers.*.mlp.experts.*.{gate|up}_proj.weight: model.layers.*.mlp.experts.linear_fc1.weight*
            "model.layers.*.mlp.experts.*.down_proj.weight": "decoder.layers.*.mlp.experts.linear_fc2.weight*",
            # model.layers.*.mlp.shared_experts.{gate|up}_proj.weight： model.layers.*.mlp.shared_experts.linear_fc1.weight
            "model.layers.*.mlp.shared_experts.down_proj.weight": "decoder.layers.*.mlp.shared_experts.linear_fc2.weight",
            # LM Head
            "model.norm.weight": "decoder.final_layernorm.weight",
            "lm_head.weight": "output_layer.weight",

            # DSv2-Lite
            "model.layers.*.self_attn.q_proj.weight": "decoder.layers.*.self_attention.linear_q_proj.weight",

            # expert bias
            "model.layers.*.mlp.gate.e_score_correction_bias": "decoder.layers.*.mlp.router.expert_bias",
        }

        # TODO: mtp layers

        mapping_list = []
        # Convert each dictionary entry to AutoMapping(hf_param, megatron_param)
        for hf_param, megatron_param in param_mappings.items():
            mapping_list.append(AutoMapping(hf_param=hf_param, megatron_param=megatron_param))

        # Reference: https://github.com/NVIDIA/NeMo/blob/50cceb9c90ea1f440d1e14074fa13bd45f60a1c4/nemo/collections/llm/gpt/model/deepseek.py#L637-L650
        #  In deepseek, HF weight `model.layers.*.post_attention_layernorm.weight` is mapped to the following mcore weights depending on the layer type:
        #  (a) `decoder.layers.*.pre_mlp_layernorm.weight`, if the layer is MoE
        #  (b) `decoder.layers.*.mlp.linear_fc1.layer_norm_weight`, if the layer is dense
        #
        # (a) is defined in the `param_mappings` above, so we need to add (b) here separately (to avoid dictionary key conflict)
        mapping_list.append(AutoMapping(hf_param="model.layers.*.post_attention_layernorm.weight", megatron_param="decoder.layers.*.mlp.linear_fc1.layer_norm_weight"))

        # Mcore local spec
        mapping_list.append(AutoMapping(hf_param="model.layers.*.self_attn.q_a_layernorm.weight", megatron_param="decoder.layers.*.self_attention.q_layernorm.weight"))
        mapping_list.append(AutoMapping(hf_param="model.layers.*.self_attn.kv_a_layernorm.weight", megatron_param="decoder.layers.*.self_attention.kv_layernorm.weight"))


        # Add special mappings that require parameter concatenation/transformation
        mapping_list.extend(
            [
                GatedMLPMapping(
                    gate="model.layers.*.mlp.gate_proj.weight",
                    up="model.layers.*.mlp.up_proj.weight",
                    megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                ),
                GatedMLPMapping(
                    gate="model.layers.*.mlp.experts.*.gate_proj.weight",
                    up="model.layers.*.mlp.experts.*.up_proj.weight",
                    megatron_param="decoder.layers.*.mlp.experts.linear_fc1.weight*",
                ),
                GatedMLPMapping(
                    gate="model.layers.*.mlp.shared_experts.gate_proj.weight",
                    up="model.layers.*.mlp.shared_experts.up_proj.weight",
                    megatron_param="decoder.layers.*.mlp.shared_experts.linear_fc1.weight",
                ),
            ]
        )

        return MegatronMappingRegistry(*mapping_list)


def register_deepseek_bridge():
    """Register DeepSeek model bridge implementations.

    Cannot directly use the MegatronModelBridge.register_bridge decorator because these models use auto_map and are not in the transformers repository, so they cannot be directly imported.
    """
    deepseek_models = [
        "deepseek-ai/DeepSeek-V2",
        "deepseek-ai/DeepSeek-V2-Lite",
        "deepseek-ai/DeepSeek-V2-Chat",
        "deepseek-ai/DeepSeek-V2-Lite-Chat",
        "moonshotai/Moonlight-16B-A3B",
        "moonshotai/Moonlight-16B-A3B-Instruct",
    ]
    for model_name in deepseek_models:
        model_class = get_causal_lm_class_via_auto_map(model_name_or_path=model_name)
        assert model_class is not None, f"Failed to resolve CausalLM class via auto_map for '{model_name}'"

        MegatronModelBridge.register_bridge(source=model_class, target=GPTModel)(DeepSeekBridge)
