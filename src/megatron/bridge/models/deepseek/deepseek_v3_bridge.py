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
from typing import Dict, Mapping

import torch
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.models.gpt.gpt_model import GPTModel

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge, WeightConversionTask
from megatron.bridge.models.conversion.param_mapping import AutoMapping
from megatron.bridge.models.deepseek.common import get_common_mapping_list
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.mla_provider import MLAModelProvider


try:
    import transformer_engine  # noqa: F401

    HAVE_TE = True
except (ImportError, ModuleNotFoundError):
    HAVE_TE = False


@MegatronModelBridge.register_bridge(
    source="DeepseekV3ForCausalLM",
    target=GPTModel,
    provider=MLAModelProvider,
    model_type="deepseek_v3",
)
class DeepSeekV3Bridge(MegatronModelBridge):
    """Megatron Bridge for DeepSeek-V3."""

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> MLAModelProvider:
        provider = super().provider_bridge(hf_pretrained)
        hf_config = hf_pretrained.config

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
        provider.moe_router_enable_expert_bias = True
        provider.moe_router_dtype = "fp32"
        provider.moe_permute_fusion = True
        provider.moe_aux_loss_coeff = 0.0001

        provider.apply_rope_fusion = False
        provider.bias_activation_fusion = True
        provider.bias_dropout_fusion = True
        provider.cross_entropy_fusion_impl = "te"
        provider.cross_entropy_loss_fusion = True
        provider.masked_softmax_fusion = True
        provider.persist_layer_norm = True
        provider.gradient_accumulation_fusion = True

        provider.hidden_dropout = 0.0
        provider.attention_softmax_in_fp32 = False

        provider.make_vocab_size_divisible_by = 1280
        provider.seq_length = 4096

        provider.moe_layer_freq = [0] * hf_config.first_k_dense_replace + [1] * (
            hf_config.num_hidden_layers - hf_config.first_k_dense_replace
        )
        provider.moe_shared_expert_intermediate_size = hf_config.moe_intermediate_size * hf_config.n_shared_experts

        # TODO: mtp
        provider.mtp_num_layers = None

        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        mapping_list = get_common_mapping_list()
        mapping_list.append(
            AutoMapping(
                megatron_param="decoder.layers.*.mlp.router.expert_bias",
                hf_param="model.layers.*.mlp.gate.e_score_correction_bias",
            )
        )
        return MegatronMappingRegistry(*mapping_list)

    def maybe_modify_converted_hf_weight(
        self,
        task: WeightConversionTask,
        converted_weights_dict: Dict[str, torch.Tensor],
        hf_state_dict: Mapping[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Add rotary embedding inverse frequency parameter if needed."""
        global_name = task.global_param_name
        if not global_name.startswith("decoder.layers.") or not global_name.endswith(".input_layernorm.weight"):
            return converted_weights_dict

        parts = global_name.split(".")
        if len(parts) < 4 or not parts[2].isdigit():
            return converted_weights_dict

        inv_freq_prefix = "model.layers."
        inv_freq_suffix = ".self_attn.rotary_emb.inv_freq"
        layer_idx = int(parts[2])
        inv_freq_key = f"{inv_freq_prefix}{layer_idx}{inv_freq_suffix}"
        if inv_freq_key in converted_weights_dict:
            return converted_weights_dict

        has_inv_freq = getattr(self, "_deepseek_has_inv_freq", None)
        if has_inv_freq is None:
            has_inv_freq = False
            for key in hf_state_dict.keys():
                if key.startswith(inv_freq_prefix) and key.endswith(inv_freq_suffix):
                    has_inv_freq = True
                    break
            self._deepseek_has_inv_freq = has_inv_freq
        if not has_inv_freq:
            return converted_weights_dict

        inv_freq = getattr(self, "_deepseek_inv_freq", None)
        if inv_freq is None:
            rotary_dim = self.hf_config.qk_rope_head_dim
            rotary_base = self.hf_config.rope_theta
            inv_freq = 1.0 / (rotary_base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
            self._deepseek_inv_freq = inv_freq

        if converted_weights_dict:
            reference_tensor = next(iter(converted_weights_dict.values()))
            if inv_freq.device != reference_tensor.device:
                inv_freq = inv_freq.to(device=reference_tensor.device)
                self._deepseek_inv_freq = inv_freq

        converted_weights_dict[inv_freq_key] = inv_freq
        return converted_weights_dict
