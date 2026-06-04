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
from functools import partial

import torch
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from transformers import Glm4MoeForCausalLM

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
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


try:
    import transformer_engine  # noqa: F401

    HAVE_TE = True
except (ImportError, ModuleNotFoundError):
    HAVE_TE = False


logger = logging.getLogger(__name__)


@MegatronModelBridge.register_bridge(source=Glm4MoeForCausalLM, target=GPTModel, model_type="glm4_moe")
class GLM45Bridge(MegatronModelBridge):
    """
    Megatron Bridge for GLM 4.5 Models.

    This bridge handles the conversion between HuggingFace Glm4MoeForCausalLM
    (used for GLM 4.5 models) and Megatron-Core GPTModel formats.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("zai-org/GLM-4.5")
        >>> provider = bridge.to_megatron_provider()
    """

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> GPTModelProvider:
        """Convert HuggingFace config to GPTModelProvider."""
        provider = super().provider_bridge(hf_pretrained)
        hf_config = hf_pretrained.config

        # Use decoder block spec to properly handle moe_layer_freq (mixed dense/MoE layers)
        provider.transformer_layer_spec = partial(get_gpt_decoder_block_spec, use_transformer_engine=HAVE_TE)
        provider.normalization = "RMSNorm"
        provider.gated_linear_unit = True
        provider.add_bias_linear = False
        provider.share_embeddings_and_output_weights = False

        provider.moe_shared_expert_overlap = True
        provider.moe_token_dispatcher_type = "alltoall"
        provider.moe_router_load_balancing_type = "seq_aux_loss"
        provider.moe_router_pre_softmax = False
        provider.moe_grouped_gemm = True
        provider.moe_router_score_function = "sigmoid"
        provider.moe_permute_fusion = True
        provider.moe_router_enable_expert_bias = True
        provider.moe_router_dtype = "fp32"
        provider.moe_router_bias_update_rate = 0
        provider.moe_aux_loss_coeff = 0.001

        provider.persist_layer_norm = True
        provider.bias_activation_fusion = True
        provider.bias_dropout_fusion = True
        provider.hidden_dropout = 0.0
        provider.autocast_dtype = torch.bfloat16
        provider.mtp_num_layers = getattr(hf_config, "num_nextn_predict_layers", None)
        provider.mtp_loss_scaling_factor = 0.3
        provider.moe_shared_expert_intermediate_size = hf_config.moe_intermediate_size

        provider.moe_layer_freq = [0] * hf_config.first_k_dense_replace + [1] * (
            hf_config.num_hidden_layers - hf_config.first_k_dense_replace
        )

        return provider

    def build_conversion_tasks(self, hf_pretrained, megatron_model):
        """Override to store HF state source before mapping_registry is called."""
        has_state = hasattr(hf_pretrained, "state") and hasattr(hf_pretrained.state, "source")
        self._hf_state_source = hf_pretrained.state.source if has_state else None
        self._hf_keys = list(self._hf_state_source.get_all_keys()) if self._hf_state_source else None
        return super().build_conversion_tasks(hf_pretrained, megatron_model)

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
            # Attention
            "decoder.layers.*.input_layernorm.weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            # Reference: https://github.com/NVIDIA/NeMo/blob/50cceb9c90ea1f440d1e14074fa13bd45f60a1c4/nemo/collections/llm/gpt/model/deepseek.py#L637-L650
            #  In GLM, HF weight `model.layers.*.post_attention_layernorm.weight` is mapped to the following mcore weights depending on the layer type:
            #  (a) `decoder.layers.*.pre_mlp_layernorm.weight`, if the layer is MoE
            #  (b) `decoder.layers.*.mlp.linear_fc1.layer_norm_weight`, if the layer is dense
            "decoder.layers.*.pre_mlp_layernorm.weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.self_attention.q_layernorm.weight": "model.layers.*.self_attn.q_norm.weight",
            "decoder.layers.*.self_attention.k_layernorm.weight": "model.layers.*.self_attn.k_norm.weight",
            # MLP
            "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.layers.*.mlp.shared_experts.linear_fc2.weight": "model.layers.*.mlp.shared_experts.down_proj.weight",
            "decoder.layers.*.mlp.shared_experts.router.weight": "model.layers.*.mlp.shared_experts.gate.weight",
            "decoder.layers.*.mlp.router.weight": "model.layers.*.mlp.gate.weight",
            "decoder.layers.*.mlp.router.expert_bias": "model.layers.*.mlp.gate.e_score_correction_bias",
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
            ]
        )
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
        # optionally add MTP mappings
        if self.hf_config is None:
            logger.warning("No HF config found, skipping MTP mappings.")
            return MegatronMappingRegistry(*mapping_list)
        hf_config = self.hf_config
        num_mtp_layers = getattr(hf_config, "num_nextn_predict_layers", 0)
        num_transformer_layers = hf_config.num_hidden_layers
        for mtp_layer in range(num_mtp_layers):
            # Support both naming conventions for the MTP transformer sub-layer: Megatron-Core
            # may expose it as either "mtp_model_layer" or "transformer_layer" depending on
            # configuration (see mimo_bridge.py for the same pattern).
            for layer_prefix in ("mtp_model_layer", "transformer_layer"):
                for megatron_param, hf_param in layer_specific_mappings.items():
                    megatron_param = (
                        megatron_param.replace(".*", f".*.{layer_prefix}")
                        .replace("decoder", "mtp")
                        .replace(".*", f".{mtp_layer}")
                    )
                    hf_param = hf_param.replace("layers.*", f"layers.{mtp_layer + num_transformer_layers}")
                    mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

                # Special mappings that require parameter concatenation/transformation
                mapping_list.extend(
                    [
                        QKVMapping(
                            megatron_param=f"mtp.layers.{mtp_layer}.{layer_prefix}.self_attention.linear_qkv.weight",
                            q=f"model.layers.{mtp_layer + num_transformer_layers}.self_attn.q_proj.weight",
                            k=f"model.layers.{mtp_layer + num_transformer_layers}.self_attn.k_proj.weight",
                            v=f"model.layers.{mtp_layer + num_transformer_layers}.self_attn.v_proj.weight",
                        ),
                        QKVMapping(
                            megatron_param=f"mtp.layers.{mtp_layer}.{layer_prefix}.self_attention.linear_qkv.bias",
                            q=f"model.layers.{mtp_layer + num_transformer_layers}.self_attn.q_proj.bias",
                            k=f"model.layers.{mtp_layer + num_transformer_layers}.self_attn.k_proj.bias",
                            v=f"model.layers.{mtp_layer + num_transformer_layers}.self_attn.v_proj.bias",
                        ),
                        GatedMLPMapping(
                            megatron_param=f"mtp.layers.{mtp_layer}.{layer_prefix}.mlp.linear_fc1.weight",
                            gate=f"model.layers.{mtp_layer + num_transformer_layers}.mlp.linear_fc1.gate.weight",
                            up=f"model.layers.{mtp_layer + num_transformer_layers}.mlp.linear_fc1.up.weight",
                        ),
                        GatedMLPMapping(
                            megatron_param=f"mtp.layers.{mtp_layer}.{layer_prefix}.mlp.shared_experts.linear_fc1.weight",
                            gate=f"model.layers.{mtp_layer + num_transformer_layers}.mlp.shared_experts.gate_proj.weight",
                            up=f"model.layers.{mtp_layer + num_transformer_layers}.mlp.shared_experts.up_proj.weight",
                        ),
                    ]
                )
                if use_fused_experts:
                    mapping_list.extend(
                        [
                            GLMExpertGateUpProjMapping(
                                megatron_param=(
                                    f"mtp.layers.{mtp_layer}.{layer_prefix}.mlp.experts.linear_fc1.weight*"
                                ),
                                hf_param=(
                                    f"model.layers.{mtp_layer + num_transformer_layers}.mlp.experts.gate_up_proj"
                                    f"{gate_up_suffix}"
                                ),
                            ),
                            GLMExpertDownProjMapping(
                                megatron_param=(
                                    f"mtp.layers.{mtp_layer}.{layer_prefix}.mlp.experts.linear_fc2.weight*"
                                ),
                                hf_param=(
                                    f"model.layers.{mtp_layer + num_transformer_layers}.mlp.experts.down_proj"
                                    f"{down_suffix}"
                                ),
                            ),
                        ]
                    )
                else:
                    mapping_list.extend(
                        [
                            GatedMLPMapping(
                                megatron_param=(
                                    f"mtp.layers.{mtp_layer}.{layer_prefix}.mlp.experts.linear_fc1.weight*"
                                ),
                                gate=f"model.layers.{mtp_layer + num_transformer_layers}.mlp.experts.*.gate_proj.weight",
                                up=f"model.layers.{mtp_layer + num_transformer_layers}.mlp.experts.*.up_proj.weight",
                            ),
                            AutoMapping(
                                megatron_param=(
                                    f"mtp.layers.{mtp_layer}.{layer_prefix}.mlp.experts.linear_fc2.weight*"
                                ),
                                hf_param=f"model.layers.{mtp_layer + num_transformer_layers}.mlp.experts.*.down_proj.weight",
                            ),
                        ]
                    )

            # MTP specific mappings (not layer_prefix dependent)
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

        return MegatronMappingRegistry(*mapping_list)

    def _uses_fused_experts(self) -> bool:
        hf_keys = getattr(self, "_hf_keys", None)
        if hf_keys:
            if any("mlp.experts.gate_up_proj" in key for key in hf_keys) or any(
                "mlp.experts.down_proj" in key for key in hf_keys
            ):
                return True

        hf_source = getattr(self, "_hf_state_source", None)
        if hf_source is not None:
            return hf_source.has_glob("*mlp.experts.gate_up_proj*") or hf_source.has_glob("*mlp.experts.down_proj*")

        # Config-only path (no HF weights to inspect): GLM 4.5 HuggingFace models
        # always use fused expert weights (gate_up_proj / down_proj), so default True.
        return True

    def _hf_expert_suffix(self, base_name: str) -> str:
        hf_keys = getattr(self, "_hf_keys", None) or []
        if any(f"{base_name}.weight" in key for key in hf_keys):
            return ".weight"

        hf_source = getattr(self, "_hf_state_source", None)
        if hf_source is not None and hf_source.has_glob(f"*{base_name}.weight"):
            return ".weight"

        # Config-only path: GLM 4.5 fused expert tensors have no .weight suffix.
        return ""
