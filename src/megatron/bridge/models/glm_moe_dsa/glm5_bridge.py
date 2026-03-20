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

import logging

import torch
from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_model import GPTModel
from transformers import GlmMoeDsaForCausalLM

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
)
from megatron.bridge.models.glm.glm_moe_mappings import (
    GLMExpertDownProjMapping,
    GLMExpertGateUpProjMapping,
)
from megatron.bridge.models.glm_moe_dsa.glm5_provider import GLM5ModelProvider
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.utils.common_utils import extract_expert_number_from_param


logger = logging.getLogger(__name__)


@MegatronModelBridge.register_bridge(source=GlmMoeDsaForCausalLM, target=GPTModel, model_type="glm_moe_dsa")
class GLM5Bridge(MegatronModelBridge):
    """
    Megatron Bridge for GLM-5 (MoE + MLA + DSA).

    This bridge handles conversion between HuggingFace GlmMoeDsaForCausalLM
    and Megatron-Core GPTModel formats.

    GLM-5 uses Multi-Latent Attention (MLA), Dynamic Sparse Attention (DSA)
    indexer layers, and Mixture-of-Experts (MoE) with Multi-Token Prediction (MTP).
    Requires transformers>=5.2.0.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("zai-org/GLM-5")
        >>> provider = bridge.to_megatron_provider()
    """

    def build_conversion_tasks(self, hf_pretrained, megatron_model):
        """Store HF config and keys before mapping_registry is called."""
        self._hf_config = hf_pretrained.config
        self._hf_state_source = hf_pretrained.state.source
        self._hf_keys = list(self._hf_state_source.get_all_keys())
        return super().build_conversion_tasks(hf_pretrained, megatron_model)

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> GLM5ModelProvider:
        hf_config = hf_pretrained.config

        configs = {
            "num_layers": hf_config.num_hidden_layers,
            "hidden_size": hf_config.hidden_size,
            "ffn_hidden_size": hf_config.intermediate_size,
            "num_attention_heads": hf_config.num_attention_heads,
            "num_query_groups": hf_config.num_key_value_heads,
            "kv_channels": getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads),
            "q_lora_rank": hf_config.q_lora_rank,
            "kv_lora_rank": hf_config.kv_lora_rank,
            "num_moe_experts": hf_config.n_routed_experts,
            "moe_ffn_hidden_size": hf_config.moe_intermediate_size,
            "moe_shared_expert_intermediate_size": hf_config.moe_intermediate_size * hf_config.n_shared_experts,
            "moe_layer_freq": [0] * hf_config.first_k_dense_replace
            + [1] * (hf_config.num_hidden_layers - hf_config.first_k_dense_replace),
            "moe_router_topk": hf_config.num_experts_per_tok,
            "moe_router_num_groups": hf_config.n_group,
            "moe_router_group_topk": hf_config.topk_group,
            "moe_router_topk_scaling_factor": hf_config.routed_scaling_factor,
            # MLA dims in MCore format
            "qk_head_dim": hf_config.qk_nope_head_dim,
            "qk_pos_emb_head_dim": hf_config.qk_rope_head_dim,
            "v_head_dim": hf_config.v_head_dim,
            "vocab_size": hf_config.vocab_size,
            "rotary_base": hf_config.rope_parameters["rope_theta"],
            "init_method_std": hf_config.initializer_range,
            "layernorm_epsilon": hf_config.rms_norm_eps,
            "multi_latent_attention": True,
            # DSA indexer params
            "experimental_attention_variant": "dsa",
            "dsa_indexer_head_dim": hf_config.index_head_dim,
            "dsa_indexer_n_heads": hf_config.index_n_heads,
            "dsa_indexer_topk": hf_config.index_topk,
            "dsa_indexer_loss_coeff": 0.001,
            "dsa_indexer_use_sparse_loss": True,
            # GLM5 uses default rope (no YaRN scaling)
            "rotary_scaling_factor": 1.0,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
        }

        dtype = self.dtype_from_hf(hf_config, default=torch.float32)
        configs["fp16"] = dtype == torch.float16
        configs["bf16"] = dtype == torch.bfloat16
        configs["params_dtype"] = dtype
        configs["make_vocab_size_divisible_by"] = 1280
        configs["moe_router_score_function"] = "sigmoid"
        configs["moe_router_enable_expert_bias"] = False
        if hasattr(hf_config, "aux_loss_alpha"):
            configs["moe_aux_loss_coeff"] = hf_config.aux_loss_alpha

        # MTP
        num_mtp = getattr(hf_config, "num_nextn_predict_layers", 0)
        if num_mtp > 0:
            configs["mtp_num_layers"] = num_mtp
            configs["mtp_loss_scaling_factor"] = 0.1

        provider = GLM5ModelProvider(**configs)

        # Use experimental-attention spec for DSA
        try:
            from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
                get_transformer_block_with_experimental_attention_variant_spec,
            )

            provider.transformer_layer_spec = get_transformer_block_with_experimental_attention_variant_spec
        except (ImportError, ModuleNotFoundError):
            logger.warning("DSA spec not available; falling back to standard GPT decoder block spec.")

        provider.normalization = "RMSNorm"
        provider.gated_linear_unit = True
        provider.position_embedding_type = "rope"
        provider.add_bias_linear = False
        provider.share_embeddings_and_output_weights = False
        provider.qk_layernorm = False  # GLM5 uses MLA, not standard QK layernorm
        provider.multi_latent_attention = True
        provider.moe_grouped_gemm = True
        provider.moe_router_pre_softmax = True
        provider.moe_token_dispatcher_type = "alltoall"
        provider.moe_router_load_balancing_type = "seq_aux_loss"
        provider.moe_shared_expert_overlap = True
        provider.moe_router_dtype = "fp32"
        provider.moe_permute_fusion = True
        provider.hidden_dropout = 0.0
        provider.attention_softmax_in_fp32 = False

        return provider

    def _uses_fused_experts(self) -> bool:
        """Detect whether HF experts are stored as fused gate_up_proj tensors."""
        hf_keys = getattr(self, "_hf_keys", None)
        if hf_keys:
            if any("mlp.experts.gate_up_proj" in key for key in hf_keys) or any(
                "mlp.experts.down_proj" in key for key in hf_keys
            ):
                return True

        hf_source = getattr(self, "_hf_state_source", None)
        if hf_source is not None:
            return hf_source.has_glob("*mlp.experts.gate_up_proj*") or hf_source.has_glob("*mlp.experts.down_proj*")
        return False

    def _hf_expert_suffix(self, base_name: str) -> str:
        hf_keys = getattr(self, "_hf_keys", None) or []
        if any(f"{base_name}.weight" in key for key in hf_keys):
            return ".weight"
        hf_source = getattr(self, "_hf_state_source", None)
        if hf_source is not None and hf_source.has_glob(f"*{base_name}.weight"):
            return ".weight"
        return ""

    def mapping_registry(self) -> MegatronMappingRegistry:
        mapping_list = []
        use_fused_experts = self._uses_fused_experts()
        gate_up_suffix = self._hf_expert_suffix("mlp.experts.gate_up_proj")
        down_suffix = self._hf_expert_suffix("mlp.experts.down_proj")

        param_mappings = {
            # Embed
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            # LM Head
            "decoder.final_layernorm.weight": "model.norm.weight",
            "output_layer.weight": "lm_head.weight",
        }

        layer_specific_mappings = {
            # Attention layernorm
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.input_layernorm.weight": "model.layers.*.input_layernorm.weight",
            # Attention output
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            # Post-attention layernorm — MoE layers use pre_mlp_layernorm, dense layers use layer_norm_weight
            "decoder.layers.*.pre_mlp_layernorm.weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.post_attention_layernorm.weight",
            # MLA weights
            "decoder.layers.*.self_attention.linear_q_down_proj.weight": "model.layers.*.self_attn.q_a_proj.weight",
            "decoder.layers.*.self_attention.linear_q_up_proj.weight": "model.layers.*.self_attn.q_b_proj.weight",
            "decoder.layers.*.self_attention.linear_q_up_proj.layer_norm_weight": "model.layers.*.self_attn.q_a_layernorm.weight",
            "decoder.layers.*.self_attention.q_layernorm.weight": "model.layers.*.self_attn.q_a_layernorm.weight",
            "decoder.layers.*.self_attention.linear_kv_down_proj.weight": "model.layers.*.self_attn.kv_a_proj_with_mqa.weight",
            "decoder.layers.*.self_attention.linear_kv_up_proj.weight": "model.layers.*.self_attn.kv_b_proj.weight",
            "decoder.layers.*.self_attention.linear_kv_up_proj.layer_norm_weight": "model.layers.*.self_attn.kv_a_layernorm.weight",
            "decoder.layers.*.self_attention.kv_layernorm.weight": "model.layers.*.self_attn.kv_a_layernorm.weight",
            # For non-MLA attention (fallback)
            "decoder.layers.*.self_attention.linear_q_proj.weight": "model.layers.*.self_attn.q_proj.weight",
            # DSA indexer
            "decoder.layers.*.self_attention.core_attention.indexer.linear_wq_b.weight": "model.layers.*.self_attn.indexer.wq_b.weight",
            "decoder.layers.*.self_attention.core_attention.indexer.linear_wk.weight": "model.layers.*.self_attn.indexer.wk.weight",
            "decoder.layers.*.self_attention.core_attention.indexer.k_norm.weight": "model.layers.*.self_attn.indexer.k_norm.weight",
            "decoder.layers.*.self_attention.core_attention.indexer.k_norm.bias": "model.layers.*.self_attn.indexer.k_norm.bias",
            "decoder.layers.*.self_attention.core_attention.indexer.linear_weights_proj.weight": "model.layers.*.self_attn.indexer.weights_proj.weight",
            # Dense MLP
            "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
            # MoE router
            "decoder.layers.*.mlp.router.weight": "model.layers.*.mlp.gate.weight",
            "decoder.layers.*.mlp.router.expert_bias": "model.layers.*.mlp.gate.e_score_correction_bias",
            # MoE shared experts
            "decoder.layers.*.mlp.shared_experts.router.weight": "model.layers.*.mlp.shared_experts.gate.weight",
            "decoder.layers.*.mlp.shared_experts.linear_fc2.weight": "model.layers.*.mlp.shared_experts.down_proj.weight",
        }

        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        for megatron_param, hf_param in layer_specific_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        # Attention (non-MLA fallback: combined QKV)
        mapping_list.extend(
            [
                QKVMapping(
                    megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                    q="model.layers.*.self_attn.q_proj.weight",
                    k="model.layers.*.self_attn.k_proj.weight",
                    v="model.layers.*.self_attn.v_proj.weight",
                ),
                QKVMapping(
                    megatron_param="decoder.layers.*.self_attention.linear_qkv.bias",
                    q="model.layers.*.self_attn.q_proj.bias",
                    k="model.layers.*.self_attn.k_proj.bias",
                    v="model.layers.*.self_attn.v_proj.bias",
                ),
                # Dense MLP gate+up → fc1
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                    gate="model.layers.*.mlp.gate_proj.weight",
                    up="model.layers.*.mlp.up_proj.weight",
                ),
                # Shared expert gate+up → fc1
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.shared_experts.linear_fc1.weight",
                    gate="model.layers.*.mlp.shared_experts.gate_proj.weight",
                    up="model.layers.*.mlp.shared_experts.up_proj.weight",
                ),
            ]
        )

        # MoE expert weights: fused (transformers 5.x) or per-expert
        if use_fused_experts:
            mapping_list.extend(
                [
                    GLMExpertGateUpProjMapping(
                        megatron_param="decoder.layers.*.mlp.experts.linear_fc1.weight*",
                        hf_param=f"model.layers.*.mlp.experts.gate_up_proj{gate_up_suffix}",
                    ),
                    GLMExpertDownProjMapping(
                        megatron_param="decoder.layers.*.mlp.experts.linear_fc2.weight*",
                        hf_param=f"model.layers.*.mlp.experts.down_proj{down_suffix}",
                    ),
                ]
            )
        else:
            mapping_list.extend(
                [
                    GatedMLPMapping(
                        megatron_param="decoder.layers.*.mlp.experts.linear_fc1.weight*",
                        gate="model.layers.*.mlp.experts.*.gate_proj.weight",
                        up="model.layers.*.mlp.experts.*.up_proj.weight",
                    ),
                    AutoMapping(
                        megatron_param="decoder.layers.*.mlp.experts.linear_fc2.weight*",
                        hf_param="model.layers.*.mlp.experts.*.down_proj.weight",
                    ),
                ]
            )

        # MTP layer mappings
        if not hasattr(self, "_hf_config"):
            logger.warning("No HF config found, skipping MTP mappings.")
            return MegatronMappingRegistry(*mapping_list)

        hf_config = self._hf_config
        num_mtp_layers = getattr(hf_config, "num_nextn_predict_layers", 0)
        num_transformer_layers = hf_config.num_hidden_layers

        for mtp_layer in range(num_mtp_layers):
            # MTP-specific norm/proj weights
            mapping_list.extend(
                [
                    AutoMapping(
                        megatron_param=f"mtp.layers.{mtp_layer}.enorm.weight",
                        hf_param=f"model.layers.{mtp_layer + num_transformer_layers}.enorm.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"mtp.layers.{mtp_layer}.hnorm.weight",
                        hf_param=f"model.layers.{mtp_layer + num_transformer_layers}.hnorm.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"mtp.layers.{mtp_layer}.eh_proj.weight",
                        hf_param=f"model.layers.{mtp_layer + num_transformer_layers}.eh_proj.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"mtp.layers.{mtp_layer}.final_layernorm.weight",
                        hf_param=f"model.layers.{mtp_layer + num_transformer_layers}.shared_head.norm.weight",
                    ),
                ]
            )

            # MTP transformer layer weights (use mtp_model_layer per Pitfall #13)
            for megatron_param, hf_param in layer_specific_mappings.items():
                mtp_megatron = (
                    megatron_param.replace(".*", ".*.mtp_model_layer")
                    .replace("decoder", "mtp")
                    .replace(".*", f".{mtp_layer}")
                )
                mtp_hf = hf_param.replace("layers.*", f"layers.{mtp_layer + num_transformer_layers}")
                mapping_list.append(AutoMapping(megatron_param=mtp_megatron, hf_param=mtp_hf))

            # MTP attention (non-MLA fallback) and MLP
            mapping_list.extend(
                [
                    QKVMapping(
                        megatron_param=f"mtp.layers.{mtp_layer}.mtp_model_layer.self_attention.linear_qkv.weight",
                        q=f"model.layers.{mtp_layer + num_transformer_layers}.self_attn.q_proj.weight",
                        k=f"model.layers.{mtp_layer + num_transformer_layers}.self_attn.k_proj.weight",
                        v=f"model.layers.{mtp_layer + num_transformer_layers}.self_attn.v_proj.weight",
                    ),
                    QKVMapping(
                        megatron_param=f"mtp.layers.{mtp_layer}.mtp_model_layer.self_attention.linear_qkv.bias",
                        q=f"model.layers.{mtp_layer + num_transformer_layers}.self_attn.q_proj.bias",
                        k=f"model.layers.{mtp_layer + num_transformer_layers}.self_attn.k_proj.bias",
                        v=f"model.layers.{mtp_layer + num_transformer_layers}.self_attn.v_proj.bias",
                    ),
                    GatedMLPMapping(
                        megatron_param=f"mtp.layers.{mtp_layer}.mtp_model_layer.mlp.linear_fc1.weight",
                        gate=f"model.layers.{mtp_layer + num_transformer_layers}.mlp.gate_proj.weight",
                        up=f"model.layers.{mtp_layer + num_transformer_layers}.mlp.up_proj.weight",
                    ),
                    GatedMLPMapping(
                        megatron_param=f"mtp.layers.{mtp_layer}.mtp_model_layer.mlp.shared_experts.linear_fc1.weight",
                        gate=f"model.layers.{mtp_layer + num_transformer_layers}.mlp.shared_experts.gate_proj.weight",
                        up=f"model.layers.{mtp_layer + num_transformer_layers}.mlp.shared_experts.up_proj.weight",
                    ),
                ]
            )

            # MTP expert weights: fused or per-expert
            if use_fused_experts:
                mapping_list.extend(
                    [
                        GLMExpertGateUpProjMapping(
                            megatron_param=f"mtp.layers.{mtp_layer}.mtp_model_layer.mlp.experts.linear_fc1.weight*",
                            hf_param=(
                                f"model.layers.{mtp_layer + num_transformer_layers}"
                                f".mlp.experts.gate_up_proj{gate_up_suffix}"
                            ),
                        ),
                        GLMExpertDownProjMapping(
                            megatron_param=f"mtp.layers.{mtp_layer}.mtp_model_layer.mlp.experts.linear_fc2.weight*",
                            hf_param=(
                                f"model.layers.{mtp_layer + num_transformer_layers}.mlp.experts.down_proj{down_suffix}"
                            ),
                        ),
                    ]
                )
            else:
                mapping_list.extend(
                    [
                        GatedMLPMapping(
                            megatron_param=f"mtp.layers.{mtp_layer}.mtp_model_layer.mlp.experts.linear_fc1.weight*",
                            gate=f"model.layers.{mtp_layer + num_transformer_layers}.mlp.experts.*.gate_proj.weight",
                            up=f"model.layers.{mtp_layer + num_transformer_layers}.mlp.experts.*.up_proj.weight",
                        ),
                        AutoMapping(
                            megatron_param=f"mtp.layers.{mtp_layer}.mtp_model_layer.mlp.experts.linear_fc2.weight*",
                            hf_param=f"model.layers.{mtp_layer + num_transformer_layers}.mlp.experts.*.down_proj.weight",
                        ),
                    ]
                )

        return MegatronMappingRegistry(*mapping_list)

    def maybe_modify_converted_hf_weight(
        self,
        task,
        converted_weights_dict: dict[str, torch.Tensor],
        hf_state_dict,
    ) -> dict[str, torch.Tensor]:
        """Merge per-expert Megatron weights back into fused HF tensors on export."""
        if not isinstance(task.mapping, (GLMExpertGateUpProjMapping, GLMExpertDownProjMapping)):
            return converted_weights_dict

        if not converted_weights_dict:
            return {}

        num_experts = self._hf_config.n_routed_experts
        ep_size = parallel_state.get_expert_model_parallel_world_size()
        experts_per_rank = num_experts // ep_size

        try:
            local_expert_number = extract_expert_number_from_param(task.param_name) % experts_per_rank
        except ValueError:
            return converted_weights_dict

        if not hasattr(self, "hf_weights_cache"):
            self.hf_weights_cache = {}

        for key, value in converted_weights_dict.items():
            if key not in self.hf_weights_cache:
                self.hf_weights_cache[key] = {}

            if ep_size == 1:
                self.hf_weights_cache[key][local_expert_number] = value
            else:
                if value.shape[0] != ep_size:
                    raise ValueError(f"Expected EP dim {ep_size} for {key}, got {value.shape}.")
                for i, exp_val in enumerate(value):
                    global_expert_number = local_expert_number + (i * experts_per_rank)
                    self.hf_weights_cache[key][global_expert_number] = exp_val

            if len(self.hf_weights_cache[key]) == num_experts:
                merged = torch.stack([self.hf_weights_cache[key][i] for i in range(num_experts)], dim=0)
                if key in hf_state_dict:
                    expected = hf_state_dict[key].shape
                    if merged.shape != expected and merged.transpose(-1, -2).shape == expected:
                        merged = merged.transpose(-1, -2).contiguous()
                del self.hf_weights_cache[key]
                return {key: merged}

            return {}

        return {}
