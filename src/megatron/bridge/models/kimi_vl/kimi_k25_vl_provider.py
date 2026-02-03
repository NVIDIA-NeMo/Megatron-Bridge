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

import math

import torch
from transformers import Gemma3ForConditionalGeneration

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.hf_pretrained.vlm import PreTrainedVLM
from megatron.bridge.models.kimi_vl.kimi_k25_vl_provider import KimiK25VLModelProvider
from megatron.bridge.models.kimi_vl.modeling_kimi_k25_vl import KimiK25VLModel
from megatron.bridge.models.deepseek.common import get_common_configs, get_common_mapping_list

@MegatronModelBridge.register_bridge(source="KimiK25ForConditionalGeneration", target=KimiK25VLModel)
class KimiK25VLBridge(MegatronModelBridge):
    """
    Megatron Bridge for Kimi K2.5 VL.
    """

    def provider_bridge(self, hf_pretrained: PreTrainedVLM) -> KimiK25VLModelProvider:
        hf_config = hf_pretrained.config
        text_config = hf_config.text_config
        vision_config = hf_config.vision_config

        # get_common_configs expects TextConfig
        hf_pretrained.config = text_config
        configs = get_common_configs(hf_pretrained)

        configs["make_vocab_size_divisible_by"] = 1280
        configs["moe_router_score_function"] = "sigmoid"
        configs["moe_router_enable_expert_bias"] = True
        # aux_loss_alpha is not set in all DSv3 HF configs
        if hasattr(hf_config, "aux_loss_alpha"):
            configs["moe_aux_loss_coeff"] = hf_config.aux_loss_alpha
        
        provider = KimiK25VLModelProvider(
            # Text configuration
            **configs,
            # Vision configuration
            _vision_config=vision_config,
            # VL-specific token IDs
            bos_token_id=getattr(text_config, "bos_token_id", 0),
            eos_token_id=getattr(text_config, "eos_token_id", 1),
            image_token_id=getattr(text_config, "media_placeholder_token_id", 151655),
            # Precision configuration
            fp16=(self.dtype_from_hf(hf_config, default=torch.float32) == torch.float16),
            bf16=(self.dtype_from_hf(hf_config, default=torch.float32) == torch.bfloat16),
            params_dtype=self.dtype_from_hf(hf_config, default=torch.float32),
            # misc
            hf_model_path=hf_pretrained._model_name_or_path,
        )

        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        # Return MegatronMappingRegistry containing parameter mappings from Megatron to HF format
        # First create simple 1:1 parameter mappings using a dictionary for readability
        mapping_list = get_common_mapping_list()
        param_mappings = {
            # expert bias
            "decoder.layers.*.mlp.router.expert_bias": "model.layers.*.mlp.gate.e_score_correction_bias",
        }

        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        for mapping in mapping_list:
            # in HF Kimi K2.5 VL models, language component is prefixed with "language_model.model" instead of "model"
            if isinstance(mapping, AutoMapping):
                mapping.hf_param = "language_model." + mapping.hf_param
                mapping.megatron_param = "language_model." + mapping.megatron_param
            elif isinstance(mapping, GatedMLPMapping):
                mapping.megatron_param = mapping.megatron_param.replace("decoder", "language_model.decoder")
                mapping.hf_param["gate"] = mapping.hf_param["gate"].replace("model", "language_model.model")
                mapping.hf_param["up"] = mapping.hf_param["up"].replace("model", "language_model.model")


        # Add Vision and MM Projector mappings
        mapping_list.extend(
            [
                ReplicatedMapping(
                    megatron_param="vision_tower.**",
                    hf_param="vision_tower.**",
                ),
                ReplicatedMapping(
                    megatron_param="mm_projector.**",
                    hf_param="mm_projector.**",
                ),
            ]
        )
        return MegatronMappingRegistry(*mapping_list)
