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

from functools import partial
from typing import Any, Dict, List, Mapping

import torch
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from transformers import PretrainedConfig

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge, WeightConversionTask
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.conversion.quantization_utils import (
    dequantize_int4,
    quantize_to_int4,
)
from megatron.bridge.models.deepseek.common import get_common_mapping_list
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.kimi_vl.kimi_k25_vl_config import KimiK25VLModelConfig
from megatron.bridge.models.kimi_vl.kimi_k25_vl_provider import KimiK25VLModelProvider
from megatron.bridge.models.kimi_vl.modeling_kimi_k25_vl import KimiK25VLModel


try:
    import transformer_engine  # type: ignore  # noqa: F401

    HAVE_TE = True
except (ImportError, ModuleNotFoundError):
    HAVE_TE = False


@MegatronModelBridge.register_bridge(
    source="KimiK25ForConditionalGeneration",
    target=KimiK25VLModel,
    provider=KimiK25VLModelProvider,
)
class KimiK25VLBridge(MegatronModelBridge):
    """
    Megatron Bridge for Kimi K2.5 VL.

    Converts HuggingFace Kimi K2.5 VL models (KimiK25ForConditionalGeneration)
    to Megatron format (KimiK25VLModel) and vice versa.

    The language backbone shares the same architecture as Kimi K2 (MoE with MLA).

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("moonshotai/Kimi-K2.5", trust_remote_code=True)
        >>> model_config = bridge.get_model_config()
    """

    MODEL_CONFIG_CLASS = KimiK25VLModelConfig
    USE_MODEL_CONFIG_FOR_CONVERSION = True

    def hf_config_to_model_config_kwargs(self, hf_config: PretrainedConfig) -> dict[str, Any]:
        """Map the Kimi K2 language config and the vision-language fields of the top-level config.

        The model loads its vision tower and projector from the checkpoint's code, so the kwargs also
        record the checkpoint and whether the bridge's source may run its code.
        """
        text_config = hf_config.text_config
        config_kwargs = super().hf_config_to_model_config_kwargs(text_config)
        first_k_dense_replace = text_config.first_k_dense_replace
        # The precision and media_placeholder_token_id come from the top-level KimiK25Config.
        dtype = self.dtype_from_hf(hf_config, default=torch.float32)
        media_placeholder_token_id = getattr(hf_config, "media_placeholder_token_id", 163605)
        config_kwargs.update(
            # Language model architecture (MoE + MLA)
            normalization="RMSNorm",
            gated_linear_unit=True,
            add_bias_linear=False,
            share_embeddings_and_output_weights=False,
            qk_layernorm=True,
            multi_latent_attention=True,
            # MoE
            moe_grouped_gemm=True,
            moe_router_pre_softmax=True,
            moe_token_dispatcher_type="flex",
            moe_flex_dispatcher_backend="hybridep",
            moe_flex_dispatcher_num_sms=16,
            moe_permute_fusion_into_hybridep=False,
            moe_router_load_balancing_type="seq_aux_loss",
            moe_shared_expert_overlap=True,
            moe_router_score_function="sigmoid",
            moe_router_enable_expert_bias=True,
            moe_router_dtype="fp32",
            moe_permute_fusion=True,
            moe_aux_loss_coeff=1e-3,
            moe_router_bias_update_rate=1e-3,
            moe_router_topk_scaling_factor=getattr(text_config, "routed_scaling_factor", 2.827),
            moe_shared_expert_intermediate_size=text_config.moe_intermediate_size * text_config.n_shared_experts,
            moe_layer_freq=[0] * first_k_dense_replace + [1] * (text_config.num_hidden_layers - first_k_dense_replace),
            # Fusions
            apply_rope_fusion=False,
            bias_activation_fusion=True,
            bias_dropout_fusion=True,
            cross_entropy_fusion_impl="te",
            cross_entropy_loss_fusion=True,
            masked_softmax_fusion=True,
            persist_layer_norm=True,
            gradient_accumulation_fusion=True,
            # MLA providers keep Megatron Core's value, not GPTModelProvider's.
            deallocate_pipeline_outputs=False,
            # Misc
            hidden_dropout=0.0,
            attention_dropout=0.0,
            attention_softmax_in_fp32=False,
            make_vocab_size_divisible_by=1280,
            seq_length=4096,
            rope_scaling_factor=1.0,
            # Precision
            fp16=dtype == torch.float16,
            bf16=dtype == torch.bfloat16,
            params_dtype=dtype,
            # Vision-language
            vision_config=hf_config.vision_config,
            hf_model_path=getattr(hf_config, "name_or_path", None),
            trust_remote_code=self.trust_remote_code,
            bos_token_id=getattr(text_config, "bos_token_id", 163584),
            eos_token_id=getattr(text_config, "eos_token_id", 163585),
            image_token_id=media_placeholder_token_id,
            media_placeholder_token_id=media_placeholder_token_id,
            pad_token_id=getattr(hf_config, "pad_token_id", 163839),
            ignore_index=getattr(hf_config, "ignore_index", -100),
        )
        return config_kwargs

    def hf_config_to_provider_kwargs(self, hf_config: PretrainedConfig) -> dict[str, Any]:
        """Return the builder kwargs plus the provider's decoder block spec."""
        provider_kwargs = self.hf_config_to_model_config_kwargs(hf_config)
        provider_kwargs["transformer_layer_spec"] = partial(get_gpt_decoder_block_spec, use_transformer_engine=HAVE_TE)
        return provider_kwargs

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> KimiK25VLModelProvider:
        """Build the deprecated provider with the settings that the Hugging Face source carries."""
        provider = super().provider_bridge(hf_pretrained)
        provider.hf_model_path = hf_pretrained.model_name_or_path
        provider.trust_remote_code = bool(getattr(hf_pretrained, "trust_remote_code", False))
        provider.generation_config = hf_pretrained.generation_config
        return provider

    def _load_and_dequantize(self, key: str, hf_state_dict: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Load a weight, dequantizing INT4 packed tensors when present."""
        base = key[:-7] if key.endswith(".weight") else key
        packed_key = f"{base}.weight_packed"
        if packed_key in hf_state_dict:
            assert f"{base}.weight_scale" in hf_state_dict and f"{base}.weight_shape" in hf_state_dict, (
                f"Missing weight scale or shape for quantized weight {key}"
            )
            weight = dequantize_int4(
                hf_state_dict[packed_key],
                hf_state_dict[f"{base}.weight_scale"],
                hf_state_dict[f"{base}.weight_shape"],
                device=hf_state_dict[packed_key].device,
            )
        else:
            weight = hf_state_dict[key]
        return weight

    def maybe_modify_loaded_hf_weight(
        self, hf_param: str | dict[str, str], hf_state_dict: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        """Load HF weights, dequantizing INT4 quantized tensors when present."""
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
        """Re-quantize converted expert weights to INT4 format."""
        if task.weight_dtype is not None:
            return converted_weights_dict

        result = {}
        for fqn, tensor in converted_weights_dict.items():
            if self._is_quantized_expert_key(fqn):
                base = fqn[:-7] if fqn.endswith(".weight") else fqn
                # Preserve the original scale dtype from the HF checkpoint
                orig_scale_key = f"{base}.weight_scale"
                scale_dtype = (
                    hf_state_dict[orig_scale_key].dtype if orig_scale_key in hf_state_dict else torch.bfloat16
                )
                packed, scale, shape = quantize_to_int4(tensor, scale_dtype=scale_dtype)
                result[f"{base}.weight_packed"] = packed
                result[f"{base}.weight_scale"] = scale
                result[f"{base}.weight_shape"] = shape
            else:
                result[fqn] = tensor
        return result

    def build_conversion_tasks(
        self,
        hf_pretrained,
        megatron_model,
        weight_dtype=None,
        *,
        exclude_module_types: tuple[type[torch.nn.Module], ...] = (),
        exclude_param_patterns: tuple[str, ...] = (),
    ) -> List:
        """Override to synthesize virtual weight keys from INT4 quantized triplets.

        The HF checkpoint stores quantized expert weights as triplets
        (weight_packed, weight_scale, weight_shape) without a plain 'weight' key.
        We synthesize virtual 'weight' keys so the mapping registry can find them,
        then maybe_modify_loaded_hf_weight handles dequantization at load time.
        """
        original_get_all_keys = hf_pretrained.state.source.get_all_keys

        def _get_all_keys_with_virtual():
            keys = original_get_all_keys()
            all_keys_set = set(keys)
            virtual_keys = []
            for key in keys:
                if key.endswith("_packed"):
                    base = key[:-7]  # e.g. "...weight_packed" -> "...weight"
                    if f"{base}_scale" in all_keys_set and f"{base}_shape" in all_keys_set:
                        virtual_keys.append(base)
            return keys + virtual_keys

        hf_pretrained.state.source.get_all_keys = _get_all_keys_with_virtual
        try:
            return super().build_conversion_tasks(
                hf_pretrained,
                megatron_model,
                weight_dtype=weight_dtype,
                **({"exclude_module_types": exclude_module_types} if exclude_module_types else {}),
                **({"exclude_param_patterns": exclude_param_patterns} if exclude_param_patterns else {}),
            )
        finally:
            hf_pretrained.state.source.get_all_keys = original_get_all_keys

    def mapping_registry(self) -> MegatronMappingRegistry:
        mapping_list = get_common_mapping_list()

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

        # Vision Tower and MM Projector use ReplicatedMapping because
        # vision components are not sharded across tensor parallel ranks.
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
