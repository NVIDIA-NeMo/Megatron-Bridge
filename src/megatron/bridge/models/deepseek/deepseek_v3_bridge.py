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

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import AutoMapping
from megatron.bridge.models.deepseek.common import get_common_configs, get_common_mapping_list
from megatron.bridge.models.deepseek.deepseek_provider import DeepSeekV3Provider
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


@MegatronModelBridge.register_bridge(source="DeepseekV3ForCausalLM", target=GPTModel)
class DeepSeekV3Bridge(MegatronModelBridge):
    """
    Megatron Bridge for DeepSeek-V3.

    As a user you would not use this bridge directly, but through `AutoBridge`.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("deepseek-ai/DeepSeek-V3-Base", trust_remote_code=True)
        >>> provider = bridge.to_megatron_provider()
    """

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> DeepSeekV3Provider:
        hf_config = hf_pretrained.config
        configs = get_common_configs(hf_pretrained)

        configs["fp16"] = self.dtype_from_hf(hf_config, default=torch.float32) == torch.float16
        configs["bf16"] = self.dtype_from_hf(hf_config, default=torch.float32) == torch.bfloat16
        configs["params_dtype"] = self.dtype_from_hf(hf_config, default=torch.float32)

        configs["make_vocab_size_divisible_by"] = 1280
        configs["moe_router_score_function"] = "sigmoid"
        configs["moe_router_enable_expert_bias"] = True
        # aux_loss_alpha is not set in all DSv3 HF configs
        if hasattr(hf_config, "aux_loss_alpha"):
            configs["moe_aux_loss_coeff"] = hf_config.aux_loss_alpha

        # TODO: mtp

        provider = DeepSeekV3Provider(**configs)
        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        mapping_list = get_common_mapping_list()

        param_mappings = {
            # expert bias
            "decoder.layers.*.mlp.router.expert_bias": "model.layers.*.mlp.gate.e_score_correction_bias",
        }

        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        return MegatronMappingRegistry(*mapping_list)
