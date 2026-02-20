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
import re
from functools import partial
from typing import Dict, Mapping, Optional, Tuple

import torch
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from transformers import GlmMoeDsaForCausalLM

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
)
from megatron.bridge.models.glm_moe_dsa.glm5_provider import GLM5ModelProvider
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


try:
    import transformer_engine  # noqa: F401

    HAVE_TE = True
except (ImportError, ModuleNotFoundError):
    HAVE_TE = False


logger = logging.getLogger(__name__)


@MegatronModelBridge.register_bridge(source=GlmMoeDsaForCausalLM, target=GPTModel, model_type="glm_moe_dsa")
class GLM5Bridge(MegatronModelBridge):
    """
    Megatron Bridge for GLM 5 Models.

    This bridge handles the conversion between HuggingFace Glm5MoeForCausalLM
    (used for GLM 5 models) and Megatron-Core GPTModel formats.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("zai-org/GLM-4.5")
        >>> provider = bridge.to_megatron_provider()
    """

    @staticmethod
    def _get_glm5_configs(hf_pretrained: PreTrainedCausalLM) -> dict:
        """Build provider kwargs from GLM5 HF config schema."""
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
            # DSA indexer params (v3.2-compatible interface)
            "index_head_dim": hf_config.index_head_dim,
            "index_n_heads": hf_config.index_n_heads,
            "index_topk": hf_config.index_topk,
            # GLM5 uses default rope parameters (not yarn rope_scaling)
            "rotary_scaling_factor": 1.0,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "add_bias_linear": False,
            "position_embedding_type": "rope",
            "normalization": "RMSNorm",
        }

        return configs

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> GLM5ModelProvider:
        hf_config = hf_pretrained.config
        configs = self._get_glm5_configs(hf_pretrained)

        configs["fp16"] = self.dtype_from_hf(hf_config, default=torch.float32) == torch.float16
        configs["bf16"] = self.dtype_from_hf(hf_config, default=torch.float32) == torch.bfloat16
        configs["params_dtype"] = self.dtype_from_hf(hf_config, default=torch.float32)

        configs["make_vocab_size_divisible_by"] = 1280
        configs["moe_router_score_function"] = "sigmoid"
        # configs["moe_router_enable_expert_bias"] = True  # TODO: uncomment this
        configs["moe_router_enable_expert_bias"] = False  # TODO: remove this
        if hasattr(hf_config, "aux_loss_alpha"):
            configs["moe_aux_loss_coeff"] = hf_config.aux_loss_alpha

        provider = GLM5ModelProvider(**configs)
        # Required for mixed dense/MoE layouts; otherwise early dense layers may be built as MoE.
        provider.transformer_layer_spec = partial(get_gpt_decoder_block_spec, use_transformer_engine=HAVE_TE)
        provider.normalization = "RMSNorm"
        provider.gated_linear_unit = True
        provider.position_embedding_type = "rope"
        provider.add_bias_linear = False
        provider.share_embeddings_and_output_weights = False
        provider.qk_layernorm = True
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
        provider.make_vocab_size_divisible_by = 1280
        return provider

    def build_conversion_tasks(self, hf_pretrained, megatron_model):
        """Override to store config before mapping_registry is called."""
        # Store config on instance for use in mapping_registry
        self._hf_config = hf_pretrained.config
        return super().build_conversion_tasks(hf_pretrained, megatron_model)

    def mapping_registry(self) -> MegatronMappingRegistry:
        mapping_list = []

        # param_mappings = {
        #     # Embed
        #     "embedding.word_embeddings.weight": "model.embed_tokens.weight",
        #     # LM Head
        #     "decoder.final_layernorm.weight": "model.norm.weight",
        #     "output_layer.weight": "lm_head.weight",
        # }
        # copied from deepseek's common.py
        param_mappings = {
            # Embed
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            # Attention
            "decoder.layers.*.input_layernorm.weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            # Reference: https://github.com/NVIDIA/NeMo/blob/50cceb9c90ea1f440d1e14074fa13bd45f60a1c4/nemo/collections/llm/gpt/model/deepseek.py#L637-L650
            #  In deepseek, HF weight `model.layers.*.post_attention_layernorm.weight` is mapped to the following mcore weights depending on the layer type:
            #  (a) `decoder.layers.*.pre_mlp_layernorm.weight`, if the layer is MoE
            #  (b) `decoder.layers.*.mlp.linear_fc1.layer_norm_weight`, if the layer is dense
            "decoder.layers.*.pre_mlp_layernorm.weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.layers.*.self_attention.linear_kv_down_proj.weight": "model.layers.*.self_attn.kv_a_proj_with_mqa.weight",
            "decoder.layers.*.self_attention.linear_kv_up_proj.weight": "model.layers.*.self_attn.kv_b_proj.weight",
            "decoder.layers.*.self_attention.linear_kv_up_proj.layer_norm_weight": "model.layers.*.self_attn.kv_a_layernorm.weight",
            # Mcore local spec
            "decoder.layers.*.self_attention.kv_layernorm.weight": "model.layers.*.self_attn.kv_a_layernorm.weight",
            # Dense MLP
            "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
            # MoE
            "decoder.layers.*.mlp.router.weight": "model.layers.*.mlp.gate.weight",
            "decoder.layers.*.mlp.experts.linear_fc2.weight*": "model.layers.*.mlp.experts.*.down_proj.weight",
            "decoder.layers.*.mlp.shared_experts.linear_fc2.weight": "model.layers.*.mlp.shared_experts.down_proj.weight",
            # LM Head
            "decoder.final_layernorm.weight": "model.norm.weight",
            "output_layer.weight": "lm_head.weight",
            # MLA
            "decoder.layers.*.self_attention.linear_q_down_proj.weight": "model.layers.*.self_attn.q_a_proj.weight",
            "decoder.layers.*.self_attention.linear_q_up_proj.weight": "model.layers.*.self_attn.q_b_proj.weight",
            "decoder.layers.*.self_attention.linear_q_up_proj.layer_norm_weight": "model.layers.*.self_attn.q_a_layernorm.weight",
            # Mcore local spec
            "decoder.layers.*.self_attention.q_layernorm.weight": "model.layers.*.self_attn.q_a_layernorm.weight",
            # For models without MLA
            "decoder.layers.*.self_attention.linear_q_proj.weight": "model.layers.*.self_attn.q_proj.weight",
            
            # copied from megatron-bridge's pr: https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/1421
            "decoder.layers.*.self_attention.core_attention.indexer.linear_wq_b.weight": "model.layers.*.self_attn.indexer.wq_b.weight",
            "decoder.layers.*.self_attention.core_attention.indexer.linear_wk.weight": "model.layers.*.self_attn.indexer.wk.weight",
            "decoder.layers.*.self_attention.core_attention.indexer.k_norm.weight": "model.layers.*.self_attn.indexer.k_norm.weight",
            "decoder.layers.*.self_attention.core_attention.indexer.k_norm.bias": "model.layers.*.self_attn.indexer.k_norm.bias",
            "decoder.layers.*.self_attention.core_attention.indexer.linear_weights_proj.weight": "model.layers.*.self_attn.indexer.weights_proj.weight",           
        }
        # copied from glm45_bridge.py
        layer_specific_mappings = {
            # Attention
            # "decoder.layers.*.input_layernorm.weight": "model.layers.*.input_layernorm.weight",
            # "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            # Reference: https://github.com/NVIDIA/NeMo/blob/50cceb9c90ea1f440d1e14074fa13bd45f60a1c4/nemo/collections/llm/gpt/model/deepseek.py#L637-L650
            #  In GLM, HF weight `model.layers.*.post_attention_layernorm.weight` is mapped to the following mcore weights depending on the layer type:
            #  (a) `decoder.layers.*.pre_mlp_layernorm.weight`, if the layer is MoE
            #  (b) `decoder.layers.*.mlp.linear_fc1.layer_norm_weight`, if the layer is dense
            # "decoder.layers.*.pre_mlp_layernorm.weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            # "decoder.layers.*.self_attention.q_layernorm.weight": "model.layers.*.self_attn.q_norm.weight",
            # "decoder.layers.*.self_attention.k_layernorm.weight": "model.layers.*.self_attn.k_norm.weight",
            # MLP
            # "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
            # "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.post_attention_layernorm.weight",
            # "decoder.layers.*.mlp.shared_experts.linear_fc2.weight": "model.layers.*.mlp.shared_experts.down_proj.weight",
            "decoder.layers.*.mlp.shared_experts.router.weight": "model.layers.*.mlp.shared_experts.gate.weight",
            # "decoder.layers.*.mlp.experts.linear_fc2.weight*": "model.layers.*.mlp.experts.*.down_proj.weight",
            # "decoder.layers.*.mlp.router.weight": "model.layers.*.mlp.gate.weight",
            "decoder.layers.*.mlp.router.expert_bias": "model.layers.*.mlp.gate.e_score_correction_bias",

            # "decoder.layers.*.self_attention.linear_kv_up_proj.layer_norm_weight": "model.layers.*.self_attn.kv_a_layernorm.weight",  # For Dense MLA
            # Sparse attention indexer

            # "decoder.layers.*.mlp.router.expert_bias": "model.layers.*.mlp.gate.e_score_correction_bias",
            # "decoder.layers.*.self_attention.linear_q_up_proj.layer_norm_weight": "model.layers.*.self_attn.q_a_layernorm.weight",
        }
        
        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        for megatron_param, hf_param in layer_specific_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

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
                QKVMapping(
                    megatron_param="decoder.layers.*.self_attention.linear_qkv.bias",
                    q="model.layers.*.self_attn.q_proj.bias",
                    k="model.layers.*.self_attn.k_proj.bias",
                    v="model.layers.*.self_attn.v_proj.bias",
                ),
                # Gated MLP: Combine gate and up projection matrices into single FC1 matrix
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                    gate="model.layers.*.mlp.gate_proj.weight",
                    up="model.layers.*.mlp.up_proj.weight",
                ),
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.shared_experts.linear_fc1.weight",
                    gate="model.layers.*.mlp.shared_experts.gate_proj.weight",
                    up="model.layers.*.mlp.shared_experts.up_proj.weight",
                ),
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.experts.linear_fc1.weight*",
                    gate="model.layers.*.mlp.experts.*.gate_proj.weight",
                    up="model.layers.*.mlp.experts.*.up_proj.weight",
                ),
            ]
        )
        # optionally add MTP mappings
        if not hasattr(self, "_hf_config"):
            logger.warning("No HF config found, skipping MTP mappings.")
            return MegatronMappingRegistry(*mapping_list)
        hf_config = self._hf_config
        num_mtp_layers = getattr(hf_config, "num_nextn_predict_layers", 0)
        num_transformer_layers = hf_config.num_hidden_layers
        for mtp_layer in range(num_mtp_layers):
            for megatron_param, hf_param in layer_specific_mappings.items():
                megatron_param = (
                    megatron_param.replace(".*", ".*.transformer_layer")
                    .replace("decoder", "mtp")
                    .replace(".*", f".{mtp_layer}")
                )
                hf_param = hf_param.replace("layers.*", f"layers.{mtp_layer + num_transformer_layers}")
                mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

            # MTP specific mappings
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
            # Special mappings that require parameter concatenation/transformation
            mapping_list.extend(
                [
                    QKVMapping(
                        megatron_param=f"mtp.layers.{mtp_layer}.transformer_layer.self_attention.linear_qkv.weight",
                        q=f"model.layers.{mtp_layer + num_transformer_layers}.self_attn.q_proj.weight",
                        k=f"model.layers.{mtp_layer + num_transformer_layers}.self_attn.k_proj.weight",
                        v=f"model.layers.{mtp_layer + num_transformer_layers}.self_attn.v_proj.weight",
                    ),
                    QKVMapping(
                        megatron_param=f"mtp.layers.{mtp_layer}.transformer_layer.self_attention.linear_qkv.bias",
                        q=f"model.layers.{mtp_layer + num_transformer_layers}.self_attn.q_proj.bias",
                        k=f"model.layers.{mtp_layer + num_transformer_layers}.self_attn.k_proj.bias",
                        v=f"model.layers.{mtp_layer + num_transformer_layers}.self_attn.v_proj.bias",
                    ),
                    GatedMLPMapping(
                        megatron_param=f"mtp.layers.{mtp_layer}.transformer_layer.mlp.linear_fc1.weight",
                        gate=f"model.layers.{mtp_layer + num_transformer_layers}.mlp.linear_fc1.gate.weight",
                        up=f"model.layers.{mtp_layer + num_transformer_layers}.mlp.linear_fc1.up.weight",
                    ),
                    GatedMLPMapping(
                        megatron_param=f"mtp.layers.{mtp_layer}.transformer_layer.mlp.shared_experts.linear_fc1.weight",
                        gate=f"model.layers.{mtp_layer + num_transformer_layers}.mlp.shared_experts.gate_proj.weight",
                        up=f"model.layers.{mtp_layer + num_transformer_layers}.mlp.shared_experts.up_proj.weight",
                    ),
                    GatedMLPMapping(
                        megatron_param=f"mtp.layers.{mtp_layer}.transformer_layer.mlp.experts.linear_fc1.weight*",
                        gate=f"model.layers.{mtp_layer + num_transformer_layers}.mlp.experts.*.gate_proj.weight",
                        up=f"model.layers.{mtp_layer + num_transformer_layers}.mlp.experts.*.up_proj.weight",
                    ),
                ]
            )

        return MegatronMappingRegistry(*mapping_list)

    def maybe_modify_loaded_hf_weight(
        self, hf_param: str | dict[str, str], hf_state_dict: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        """Handle GLM MoE expert tensors with expert-id in dim0 and fused gate/up."""
        def _slice_expert_weight(param_name: str) -> Optional[torch.Tensor]:
            expert_match = re.search(r"\.experts\.(\d+)\.", param_name)
            if not expert_match:
                return None
            expert_id = int(expert_match.group(1))
            base_param = param_name.replace(f".experts.{expert_id}.", ".experts.")
            base_param_override = None
            if ".experts." in param_name and ".gate_proj." in param_name:
                base_param_override = param_name.replace(
                    f".experts.{expert_id}.gate_proj.", ".experts.gate_up_proj."
                )
            elif ".experts." in param_name and ".up_proj." in param_name:
                base_param_override = param_name.replace(
                    f".experts.{expert_id}.up_proj.", ".experts.gate_up_proj."
                )

            base_weight = None
            base_param_candidates = []
            if base_param_override:
                base_param_candidates.append(base_param_override)
                if base_param_override.endswith(".weight"):
                    base_param_candidates.append(base_param_override[: -len(".weight")])
            base_param_candidates.append(base_param)
            if base_param.endswith(".weight"):
                base_param_candidates.append(base_param[: -len(".weight")])

            for candidate in base_param_candidates:
                if candidate in hf_state_dict:
                    base_weight = hf_state_dict[candidate]
                    break
            if base_weight is None:
                glob_param = param_name.replace(f".experts.{expert_id}.", ".experts.*.")
                try:
                    matched = hf_state_dict[glob_param]
                except KeyError:
                    matched = {}
                if isinstance(matched, dict) and len(matched) == 1:
                    base_weight = next(iter(matched.values()))
            if base_weight is None:
                return None
            if base_weight.ndim == 0:
                raise ValueError(f"Expected expert tensor for {base_param}, got scalar")
            if expert_id >= base_weight.shape[0]:
                raise ValueError(
                    f"Expert id {expert_id} out of range for {base_param} with shape {base_weight.shape}"
                )
            sliced = base_weight[expert_id]
            if base_param_override and "gate_up_proj" in base_param_override:
                if ".gate_proj." in param_name:
                    return sliced[: sliced.shape[0] // 2]
                if ".up_proj." in param_name:
                    return sliced[sliced.shape[0] // 2 :]
            return sliced

        def _load_weight(param_name: str) -> torch.Tensor:
            if param_name in hf_state_dict:
                return hf_state_dict[param_name]
            if param_name.endswith(".weight") and param_name[: -len(".weight")] in hf_state_dict:
                return hf_state_dict[param_name[: -len(".weight")]]
            sliced = _slice_expert_weight(param_name)
            if sliced is not None:
                return sliced
            return hf_state_dict[param_name]

        if isinstance(hf_param, str):
            return _load_weight(hf_param)
        return {k: _load_weight(v) for k, v in hf_param.items()}

    def maybe_modify_converted_hf_weight(
        self,
        task: "WeightConversionTask",
        converted_weights_dict: Dict[str, torch.Tensor],
        hf_state_dict: Mapping[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Pack per-expert weights into GLM MoE tensors when exporting to HF."""
        if not converted_weights_dict:
            return converted_weights_dict

        if not hasattr(self, "_hf_expert_cache"):
            self._hf_expert_cache = {}
            self._hf_expert_expected = {}

        def _base_expert_key(param_name: str) -> Optional[Tuple[str, int, Optional[str]]]:
            match = re.search(r"\.experts\.(\d+)\.", param_name)
            if not match:
                return None
            expert_id = int(match.group(1))
            base_key = param_name.replace(f".experts.{expert_id}.", ".experts.")
            kind = None
            if ".gate_proj." in param_name:
                base_key = param_name.replace(f".experts.{expert_id}.gate_proj.", ".experts.gate_up_proj.")
                kind = "gate"
            elif ".up_proj." in param_name:
                base_key = param_name.replace(f".experts.{expert_id}.up_proj.", ".experts.gate_up_proj.")
                kind = "up"
            return base_key, expert_id, kind

        def _should_pack_experts(base_key: str) -> bool:
            try:
                if base_key in hf_state_dict:
                    return True
                if base_key.endswith(".weight") and base_key[: -len(".weight")] in hf_state_dict:
                    return True
            except Exception:
                pass
            has_glob = getattr(hf_state_dict, "has_glob", None)
            if callable(has_glob):
                if has_glob(base_key):
                    return True
                if base_key.endswith(".weight") and has_glob(base_key[: -len(".weight")]):
                    return True
            return False

        def _expected_experts(base_key: str) -> Optional[int]:
            if base_key in self._hf_expert_expected:
                return self._hf_expert_expected[base_key]
            expected = None
            try:
                if base_key in hf_state_dict:
                    expected = int(hf_state_dict[base_key].shape[0])
                elif base_key.endswith(".weight") and base_key[: -len(".weight")] in hf_state_dict:
                    expected = int(hf_state_dict[base_key[: -len(".weight")]].shape[0])
            except Exception:
                expected = None
            if expected is None and task.megatron_module is not None:
                config = getattr(task.megatron_module, "config", None)
                if config is not None and getattr(config, "num_moe_experts", None) is not None:
                    expected = int(config.num_moe_experts)
            if expected is not None:
                self._hf_expert_expected[base_key] = expected
            return expected

        output: Dict[str, torch.Tensor] = {}
        for name, tensor in converted_weights_dict.items():
            base_info = _base_expert_key(name)
            if base_info is None:
                output[name] = tensor
                continue
            base_key, expert_id, kind = base_info
            if not _should_pack_experts(base_key):
                output[name] = tensor
                continue
            expected = _expected_experts(base_key)
            if expected is None:
                output[name] = tensor
                continue

            output_base_key = base_key
            if base_key.endswith(".weight") and base_key[: -len(".weight")] in hf_state_dict:
                output_base_key = base_key[: -len(".weight")]

            cache = self._hf_expert_cache.setdefault(base_key, {})
            if kind is None:
                cache[expert_id] = tensor
                if len(cache) == expected:
                    stacked = torch.stack([cache[i] for i in range(expected)], dim=0)
                    output[output_base_key] = stacked
                    del self._hf_expert_cache[base_key]
            else:
                per_kind = cache.setdefault(expert_id, {})
                per_kind[kind] = tensor
                if all(
                    (i in cache and "gate" in cache[i] and "up" in cache[i]) for i in range(expected)
                ):
                    stacked = torch.stack(
                        [torch.cat([cache[i]["gate"], cache[i]["up"]], dim=0) for i in range(expected)],
                        dim=0,
                    )
                    output[output_base_key] = stacked
                    del self._hf_expert_cache[base_key]

        return output
