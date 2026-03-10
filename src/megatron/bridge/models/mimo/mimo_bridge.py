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

from typing import Mapping

import torch
from megatron.core.models.gpt.gpt_model import GPTModel

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge, WeightConversionTask
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
)
from megatron.bridge.models.qwen.qwen2_bridge import Qwen2Bridge


@MegatronModelBridge.register_bridge(source="MiMoForCausalLM", target=GPTModel, model_type="mimo")
class MimoBridge(Qwen2Bridge):
    """Megatron Bridge for MiMo Causal LM."""

    def provider_bridge(self, hf_pretrained):
        provider = super().provider_bridge(hf_pretrained)
        hf_config = hf_pretrained.config

        # MiMo follows Qwen2 attention behavior and adds MTP on top.
        provider.qk_layernorm = False
        provider.add_qkv_bias = True

        num_mtp_layers = getattr(hf_config, "num_nextn_predict_layers", 0)
        if num_mtp_layers > 0:
            provider.mtp_num_layers = num_mtp_layers
            provider.mtp_loss_scaling_factor = 0.1

        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        mapping_list = list(super().mapping_registry().mappings)

        mapping_list.extend(
            [
                AutoMapping(
                    megatron_param="mtp.layers.*.enorm.weight",
                    hf_param="model.mtp_layers.*.token_layernorm.weight",
                ),
                AutoMapping(
                    megatron_param="mtp.layers.*.hnorm.weight",
                    hf_param="model.mtp_layers.*.hidden_layernorm.weight",
                ),
                AutoMapping(
                    megatron_param="mtp.layers.*.eh_proj.weight",
                    hf_param="model.mtp_layers.*.input_proj.weight",
                ),
                AutoMapping(
                    megatron_param="mtp.layers.*.final_layernorm.weight",
                    hf_param="model.mtp_layers.*.final_layernorm.weight",
                ),
            ]
        )

        # Support both naming conventions: Megatron-Core may expose MTP layers as
        # either "transformer_layer" or "mtp_model_layer" depending on configuration
        for layer_prefix in ("transformer_layer", "mtp_model_layer"):
            layer_path = f"mtp.layers.*.{layer_prefix}"
            mapping_list.extend(
                [
                    AutoMapping(
                        megatron_param=f"{layer_path}.self_attention.linear_qkv.layer_norm_weight",
                        hf_param="model.mtp_layers.*.input_layernorm.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"{layer_path}.mlp.linear_fc1.layer_norm_weight",
                        hf_param="model.mtp_layers.*.post_attention_layernorm.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"{layer_path}.self_attention.linear_proj.weight",
                        hf_param="model.mtp_layers.*.self_attn.o_proj.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"{layer_path}.mlp.linear_fc2.weight",
                        hf_param="model.mtp_layers.*.mlp.down_proj.weight",
                    ),
                    QKVMapping(
                        megatron_param=f"{layer_path}.self_attention.linear_qkv.weight",
                        q="model.mtp_layers.*.self_attn.q_proj.weight",
                        k="model.mtp_layers.*.self_attn.k_proj.weight",
                        v="model.mtp_layers.*.self_attn.v_proj.weight",
                    ),
                    QKVMapping(
                        megatron_param=f"{layer_path}.self_attention.linear_qkv.bias",
                        q="model.mtp_layers.*.self_attn.q_proj.bias",
                        k="model.mtp_layers.*.self_attn.k_proj.bias",
                        v="model.mtp_layers.*.self_attn.v_proj.bias",
                    ),
                    GatedMLPMapping(
                        megatron_param=f"{layer_path}.mlp.linear_fc1.weight",
                        gate="model.mtp_layers.*.mlp.gate_proj.weight",
                        up="model.mtp_layers.*.mlp.up_proj.weight",
                    ),
                ]
            )

        return MegatronMappingRegistry(*mapping_list)

    @staticmethod
    def _swap_input_proj_halves(weight: torch.Tensor) -> torch.Tensor:
        if weight.ndim < 2:
            raise ValueError(
                f"Expected tensor with at least 2 dimensions for input_proj weight, got shape {weight.shape}"
            )
        if weight.shape[1] % 2 != 0:
            raise ValueError(f"Expected even dimension at dim=1 for input_proj weight, got shape {weight.shape}")
        first_half, second_half = weight.chunk(2, dim=1)
        return torch.cat((second_half, first_half), dim=1)

    def maybe_modify_loaded_hf_weight(
        self, hf_param: str | dict[str, str], hf_state_dict: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        hf_weights = super().maybe_modify_loaded_hf_weight(hf_param, hf_state_dict)
        if isinstance(hf_param, str) and hf_param.endswith(".input_proj.weight"):
            return self._swap_input_proj_halves(hf_weights)
        return hf_weights

    def maybe_modify_converted_hf_weight(
        self,
        task: WeightConversionTask,
        converted_weights_dict: dict[str, torch.Tensor],
        hf_state_dict: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        converted_weights_dict = super().maybe_modify_converted_hf_weight(
            task,
            converted_weights_dict,
            hf_state_dict,
        )

        if not task.global_param_name.endswith(".eh_proj.weight"):
            return converted_weights_dict

        for hf_name, weight in list(converted_weights_dict.items()):
            if hf_name.endswith(".input_proj.weight"):
                converted_weights_dict[hf_name] = self._swap_input_proj_halves(weight)

        return converted_weights_dict
