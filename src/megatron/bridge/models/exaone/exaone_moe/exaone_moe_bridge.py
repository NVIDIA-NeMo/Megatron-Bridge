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


import torch
from megatron.core.models.gpt.gpt_model import GPTModel
from transformers import ExaoneMoeForCausalLM

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge, register_bridge_implementation
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
)
from megatron.bridge.models.exaone.exaone_moe.exaone_moe_provider import ExaoneMoeModelProvider
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


@MegatronModelBridge.register_bridge(
    source=ExaoneMoeForCausalLM,
    target=GPTModel,
    provider=ExaoneMoeModelProvider,
    model_type="exaone_moe",
)
class ExaoneMoeBridge(MegatronModelBridge):
    """Megatron Bridge for Hugging Face EXAONE MoE causal language models."""

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> ExaoneMoeModelProvider:
        hf_config = hf_pretrained.config
        self.hf_config = hf_config

        rope_parameters = hf_config.rope_parameters
        rope_theta = rope_parameters["rope_theta"]
        rope_scaling_factor = rope_parameters.get("factor")
        rope_scaling = rope_scaling_factor is not None

        is_moe_layer = getattr(hf_config, "is_moe_layer", None)
        if is_moe_layer is not None:
            moe_layer_freq = [int(value) for value in is_moe_layer]
        else:
            mlp_layer_types = getattr(hf_config, "mlp_layer_types", None)
            if mlp_layer_types is not None:
                moe_layer_freq = [int(layer_type == "sparse") for layer_type in mlp_layer_types]
            else:
                first_dense_layer_count = hf_config.first_k_dense_replace
                moe_layer_freq = [0] * first_dense_layer_count + [1] * (
                    hf_config.num_hidden_layers - first_dense_layer_count
                )

        window_attn_skip_freq, no_rope_freq = [], []
        layer_types = getattr(hf_config, "layer_types", None) or []
        has_sliding_attention = False
        for layer_idx in range(hf_config.num_hidden_layers):
            layer_type = layer_types[layer_idx] if layer_idx < len(layer_types) else "sliding_attention"
            is_sliding = layer_type == "sliding_attention"
            has_sliding_attention = has_sliding_attention or is_sliding
            no_rope_freq.append(0 if is_sliding else 1)
            window_attn_skip_freq.append(1 if is_sliding else 0)

        sliding_window = getattr(hf_config, "sliding_window", None)
        window_size = (sliding_window - 1, 0) if has_sliding_attention and sliding_window is not None else None

        model_dtype = self.dtype_from_hf(hf_config, default=torch.float32)

        provider = ExaoneMoeModelProvider(
            num_layers=hf_config.num_hidden_layers,
            hidden_size=hf_config.hidden_size,
            ffn_hidden_size=hf_config.intermediate_size,
            moe_ffn_hidden_size=hf_config.moe_intermediate_size,  # Maps to moe_intermediate_size in HF
            num_attention_heads=hf_config.num_attention_heads,
            num_query_groups=hf_config.num_key_value_heads,
            kv_channels=getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads),
            num_moe_experts=hf_config.num_experts,
            moe_router_topk=hf_config.num_experts_per_tok,  # Maps to num_experts_per_tok in HF
            moe_router_num_groups=hf_config.n_group,
            moe_router_group_topk=hf_config.topk_group,
            moe_router_topk_scaling_factor=hf_config.routed_scaling_factor,
            moe_router_score_function=getattr(hf_config, "scoring_func", "sigmoid"),
            init_method_std=hf_config.initializer_range,
            layernorm_epsilon=hf_config.rms_norm_eps,
            gated_linear_unit=True,
            make_vocab_size_divisible_by=self.make_vocab_size_divisible_by(hf_config.vocab_size),
            rotary_base=rope_theta,
            share_embeddings_and_output_weights=getattr(hf_config, "tie_word_embeddings", False),
            vocab_size=hf_config.vocab_size,
            seq_length=hf_config.max_position_embeddings,
            attention_dropout=hf_config.attention_dropout,
            fp16=(model_dtype == torch.float16),
            bf16=(model_dtype == torch.bfloat16),
            params_dtype=model_dtype,
            qk_layernorm=True,
            moe_grouped_gemm=True,
            moe_shared_expert_intermediate_size=hf_config.num_shared_experts * hf_config.moe_intermediate_size,
            rope_scaling=rope_scaling,
            rope_scaling_factor=rope_scaling_factor,
            window_attn_skip_freq=window_attn_skip_freq,
            window_size=window_size,
            no_rope_freq=no_rope_freq,
            moe_layer_freq=moe_layer_freq,
            # MTP
            mtp_num_layers=getattr(hf_config, "num_nextn_predict_layers", None),
            mtp_loss_scaling_factor=getattr(hf_config, "mtp_loss_scaling_factor", 0.1),
            mtp_use_repeated_layer=getattr(hf_config, "mtp_share_layers", False),
        )

        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        # Return MegatronMappingRegistry containing parameter mappings from HF to Megatron format
        # First create simple 1:1 parameter mappings using a dictionary for readability

        # Dictionary maps Megatron parameter names -> HF parameter names
        # Supports wildcard (*) patterns for layer-specific parameters
        hf_config = getattr(self, "hf_config", None)
        share_embeddings_and_output_weights = getattr(
            hf_config,
            "share_embeddings_and_output_weights",
            getattr(hf_config, "tie_word_embeddings", False),
        )
        output_layer_hf_param = (
            "model.embed_tokens.weight" if share_embeddings_and_output_weights else "lm_head.weight"
        )
        mtp_num_layers = getattr(hf_config, "num_nextn_predict_layers", 0) or 0

        param_mappings = {
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.mlp.router.weight": "model.layers.*.mlp.gate.weight",
            "decoder.layers.*.mlp.router.expert_bias": "model.layers.*.mlp.e_score_correction_bias",
            "decoder.layers.*.pre_mlp_layernorm.weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.layers.*.self_attention.q_layernorm.weight": "model.layers.*.self_attn.q_norm.weight",
            "decoder.layers.*.self_attention.k_layernorm.weight": "model.layers.*.self_attn.k_norm.weight",
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
        }

        if hf_config is None or mtp_num_layers > 0:
            param_mappings.update(
                {
                    "mtp.layers.*.mtp_model_layer.self_attention.q_layernorm.weight": "mtp.layers.*.self_attn.q_norm.weight",
                    "mtp.layers.*.mtp_model_layer.self_attention.k_layernorm.weight": "mtp.layers.*.self_attn.k_norm.weight",
                    "mtp.layers.*.mtp_model_layer.self_attention.linear_proj.weight": "mtp.layers.*.self_attn.o_proj.weight",
                    "mtp.layers.*.mtp_model_layer.self_attention.linear_qkv.layer_norm_weight": "mtp.layers.*.input_layernorm.weight",
                    "mtp.layers.*.mtp_model_layer.pre_mlp_layernorm.weight": "mtp.layers.*.post_attention_layernorm.weight",
                    "mtp.layers.*.mtp_model_layer.mlp.linear_fc1.layer_norm_weight": "mtp.layers.*.post_attention_layernorm.weight",
                    "mtp.layers.0.final_layernorm.weight": "mtp.norm.weight",
                    "mtp.layers.0.eh_proj.weight": "mtp.fc.weight",
                    "mtp.layers.0.hnorm.weight": "mtp.pre_fc_norm_hidden.weight",
                    "mtp.layers.0.enorm.weight": "mtp.pre_fc_norm_embedding.weight",
                }
            )

        mapping_list = []
        # Convert each dictionary entry to AutoMapping(megatron_param, hf_param)
        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        mapping_list.append(AutoMapping(hf_param=output_layer_hf_param, megatron_param="output_layer.weight"))

        # Add special mappings that require parameter concatenation/transformation
        mapping_list.extend(
            [
                # QKV: Combine separate Q, K, V matrices into single QKV matrix
                QKVMapping(
                    megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                    q="model.layers.*.self_attn.q_proj.weight",
                    k="model.layers.*.self_attn.k_proj.weight",
                    v="model.layers.*.self_attn.v_proj.weight",
                ),
                # K-EXAONE checkpoints store per-expert projections; Transformers fuses them only in memory.
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.experts.linear_fc1.weight*",
                    gate="model.layers.*.mlp.experts.*.gate_proj.weight",
                    up="model.layers.*.mlp.experts.*.up_proj.weight",
                ),
                AutoMapping(
                    megatron_param="decoder.layers.*.mlp.experts.linear_fc2.weight*",
                    hf_param="model.layers.*.mlp.experts.*.down_proj.weight",
                ),
                # Shared Experts
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.shared_experts.linear_fc1.weight",
                    gate="model.layers.*.mlp.shared_experts.gate_proj.weight",
                    up="model.layers.*.mlp.shared_experts.up_proj.weight",
                ),
                AutoMapping(
                    megatron_param="decoder.layers.*.mlp.shared_experts.linear_fc2.weight",
                    hf_param="model.layers.*.mlp.shared_experts.down_proj.weight",
                ),
                # Dense MLP
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                    gate="model.layers.*.mlp.gate_proj.weight",
                    up="model.layers.*.mlp.up_proj.weight",
                ),
                AutoMapping(
                    megatron_param="decoder.layers.*.mlp.linear_fc2.weight",
                    hf_param="model.layers.*.mlp.down_proj.weight",
                ),
            ]
        )

        if hf_config is None or mtp_num_layers > 0:
            mapping_list.extend(
                [
                    # MTP
                    AutoMapping(
                        hf_param="mtp.layers.*.mlp.down_proj.weight",
                        megatron_param="mtp.layers.*.mtp_model_layer.mlp.linear_fc2.weight",
                    ),
                    GatedMLPMapping(
                        gate="mtp.layers.*.mlp.gate_proj.weight",
                        up="mtp.layers.*.mlp.up_proj.weight",
                        megatron_param="mtp.layers.*.mtp_model_layer.mlp.linear_fc1.weight",
                    ),
                    QKVMapping(
                        q="mtp.layers.*.self_attn.q_proj.weight",
                        k="mtp.layers.*.self_attn.k_proj.weight",
                        v="mtp.layers.*.self_attn.v_proj.weight",
                        megatron_param="mtp.layers.*.mtp_model_layer.self_attention.linear_qkv.weight",
                    ),
                ]
            )

        return MegatronMappingRegistry(*mapping_list)


# Some K-EXAONE configs use the upstream architecture spelling
# ``ExaoneMoEForCausalLM``, while Transformers exposes ``ExaoneMoeForCausalLM``.
register_bridge_implementation(source="ExaoneMoEForCausalLM", target=GPTModel, bridge_class=ExaoneMoeBridge)
