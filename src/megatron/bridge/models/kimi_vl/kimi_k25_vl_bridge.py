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
from typing import Dict, Mapping

import torch

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge, WeightConversionTask
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.deepseek.common import get_common_mapping_list
from megatron.bridge.models.kimi_vl.utils import (
    get_common_configs,
    dequantize_int4,
    quantize_to_int4,
)
from megatron.bridge.models.hf_pretrained.vlm import PreTrainedVLM
from megatron.bridge.models.kimi_vl.kimi_k25_vl_provider import KimiK25VLModelProvider
from megatron.bridge.models.kimi_vl.modeling_kimi_k25_vl import KimiK25VLModel


@MegatronModelBridge.register_bridge(source="KimiK25ForConditionalGeneration", target=KimiK25VLModel)
class KimiK25VLBridge(MegatronModelBridge):
    """
    Megatron Bridge for Kimi K2.5 VL.

    Converts HuggingFace Kimi K2.5 VL models (KimiK25ForConditionalGeneration)
    to Megatron format (KimiK25VLModel) and vice versa.

    The language backbone shares the same architecture as Kimi K2 (MoE with MLA).
    """

    def provider_bridge(self, hf_pretrained: PreTrainedVLM) -> KimiK25VLModelProvider:
        hf_config = hf_pretrained.config
        text_config = hf_config.text_config
        vision_config = hf_config.vision_config

        # TODO remove get_common_configs later, need to ensure all params are set correctly.
        # Temporarily swap to text_config for get_common_configs (which reads
        # hf_pretrained.config), then restore the original VL config so that
        # save_artifacts later writes the full config (including auto_map).
        hf_pretrained.config = text_config
        configs = get_common_configs(hf_pretrained)
        hf_pretrained.config = hf_config

        configs["make_vocab_size_divisible_by"] = 1280
        configs["moe_router_score_function"] = "sigmoid"
        configs["moe_router_enable_expert_bias"] = True
        # aux_loss_alpha is on text_config, not the top-level KimiK25Config
        if hasattr(text_config, "aux_loss_alpha"):
            configs["moe_aux_loss_coeff"] = text_config.aux_loss_alpha

        # media_placeholder_token_id is on the top-level KimiK25Config, not on text_config
        media_placeholder_token_id = getattr(hf_config, "media_placeholder_token_id", 163605)

        provider = KimiK25VLModelProvider(
            # Text configuration (extracted from HF text config)
            **configs,
            # Vision configuration (raw HF KimiK25VisionConfig)
            vision_config=vision_config,
            # VL-specific token IDs
            bos_token_id=getattr(text_config, "bos_token_id", 163584),
            eos_token_id=getattr(text_config, "eos_token_id", 163585),
            image_token_id=media_placeholder_token_id,
            media_placeholder_token_id=media_placeholder_token_id,
            pad_token_id=getattr(hf_config, "pad_token_id", 163839),
            ignore_index=getattr(hf_config, "ignore_index", -100),
            # Precision configuration
            fp16=(self.dtype_from_hf(hf_config, default=torch.float32) == torch.float16),
            bf16=(self.dtype_from_hf(hf_config, default=torch.float32) == torch.bfloat16),
            params_dtype=self.dtype_from_hf(hf_config, default=torch.float32),
            # HF model path (needed for dynamic module loading of vision components)
            hf_model_path=hf_pretrained._model_name_or_path,
        )

        return provider

    def _load_and_dequantize(self, key: str, hf_state_dict: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Load a weight, dequantizing INT4 triplets and FP8 block-wise tensors when present."""
        base = key[:-7] if key.endswith(".weight") else key
        packed_key = f"{base}.weight_packed"
        if (
            packed_key in hf_state_dict
            and f"{base}.weight_scale" in hf_state_dict
            and f"{base}.weight_shape" in hf_state_dict
        ):
            weight = dequantize_int4(
                hf_state_dict[packed_key],
                hf_state_dict[f"{base}.weight_scale"],
                hf_state_dict[f"{base}.weight_shape"],
                device="cpu",
            )
        else:
            weight = hf_state_dict[key]
        return weight

    def maybe_modify_loaded_hf_weight(
        self, hf_param: str | dict[str, str], hf_state_dict: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        """Load HF weights, dequantizing INT4 and FP8 quantized tensors when present."""
        if isinstance(hf_param, str):
            return self._load_and_dequantize(hf_param, hf_state_dict)
        return {k: self._load_and_dequantize(v, hf_state_dict) for k, v in hf_param.items()}

    def _is_quantized_expert_key(self, key: str) -> bool:
        if "mlp.experts." in key and ".weight" in key:
            if "shared_experts" in key:
                return False
            if ".layers.0." in key:
                return False
            return True
        return False

    def maybe_modify_converted_hf_weight(
        self,
        task: WeightConversionTask,
        converted_weights_dict: Dict[str, torch.Tensor],
        hf_state_dict: Mapping[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Add rotary embedding inverse frequency parameter if needed."""

        hf_state_dict = {}
        for fqn, tensor in converted_weights_dict.items():
            if self._is_quantized_expert_key(fqn):
                packed, scale, shape = quantize_to_int4(tensor)
                hf_state_dict[f"{fqn[:-7] if fqn.endswith('.weight') else fqn}.weight_packed"] = packed
                hf_state_dict[f"{fqn[:-7] if fqn.endswith('.weight') else fqn}.weight_scale"] = scale
                hf_state_dict[f"{fqn[:-7] if fqn.endswith('.weight') else fqn}.weight_shape"] = shape
            else:
                hf_state_dict[fqn] = tensor
        return hf_state_dict

    def mapping_registry(self) -> MegatronMappingRegistry:
        # Return MegatronMappingRegistry containing parameter mappings from Megatron to HF format.
        # Start with the common mapping list for the language model.
        mapping_list = get_common_mapping_list()
        param_mappings = {
            # expert bias
            "decoder.layers.*.mlp.router.expert_bias": "model.layers.*.mlp.gate.e_score_correction_bias",
        }

        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        # In HF Kimi K2.5 VL models, the language component is nested under
        # "language_model.model" instead of just "model", so we need to add the prefix.
        for mapping in mapping_list:
            if isinstance(mapping, AutoMapping):
                mapping.hf_param = "language_model." + mapping.hf_param
                mapping.megatron_param = "language_model." + mapping.megatron_param
            elif isinstance(mapping, GatedMLPMapping):
                mapping.megatron_param = mapping.megatron_param.replace("decoder", "language_model.decoder")
                mapping.hf_param["gate"] = mapping.hf_param["gate"].replace("model", "language_model.model")
                mapping.hf_param["up"] = mapping.hf_param["up"].replace("model", "language_model.model")

        # Add Vision Tower and MM Projector mappings.
        # These use ReplicatedMapping because the vision components are not sharded
        # across tensor parallel ranks — they are replicated on each rank.
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
