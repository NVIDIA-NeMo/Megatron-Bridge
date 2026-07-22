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
from typing import Dict, Mapping, Union

import torch
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.models.gpt.gpt_model import GPTModel

from megatron.bridge.models.conversion import quantization_utils
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge, WeightConversionTask
from megatron.bridge.models.deepseek.common import get_common_mapping_list
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.mla_provider import MLAModelProvider


try:
    import transformer_engine  # noqa: F401

    HAVE_TE = True
except (ImportError, ModuleNotFoundError):
    HAVE_TE = False


__all__ = ["DeepSeekV3Bridge", "_dequant_fp8_blockwise"]


_dequant_fp8_blockwise = quantization_utils.dequantize_fp8_blockwise


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
        provider.add_bias_linear = False
        provider.share_embeddings_and_output_weights = False
        provider.qk_layernorm = True
        provider.multi_latent_attention = True

        provider.moe_grouped_gemm = True
        provider.moe_router_pre_softmax = True
        provider.moe_token_dispatcher_type = "alltoall"
        provider.moe_router_load_balancing_type = "seq_aux_loss"
        provider.moe_shared_expert_overlap = True
        provider.moe_router_score_function = "sigmoid"
        provider.moe_router_enable_expert_bias = True
        provider.moe_router_dtype = "fp32"
        provider.moe_permute_fusion = True

        provider.apply_rope_fusion = False
        provider.gradient_accumulation_fusion = True
        provider.bias_activation_fusion = True
        provider.bias_dropout_fusion = True
        provider.cross_entropy_fusion_impl = "te"
        provider.cross_entropy_loss_fusion = True
        provider.masked_softmax_fusion = True
        provider.persist_layer_norm = True

        provider.hidden_dropout = 0.0
        provider.attention_softmax_in_fp32 = False

        provider.make_vocab_size_divisible_by = 1280

        provider.moe_layer_freq = [0] * hf_config.first_k_dense_replace + [1] * (
            hf_config.num_hidden_layers - hf_config.first_k_dense_replace
        )
        provider.moe_shared_expert_intermediate_size = hf_config.moe_intermediate_size * hf_config.n_shared_experts

        provider.mtp_num_layers = getattr(hf_config, "num_nextn_predict_layers", 0) or None

        return provider

    @classmethod
    def megatron_to_hf_config(cls, provider: MLAModelProvider) -> dict:
        hf_cfg = super(DeepSeekV3Bridge, cls).megatron_to_hf_config(provider)

        # Megatron uses None="not set/disabled", but HF expects integers
        hf_cfg["num_nextn_predict_layers"] = hf_cfg.get("num_nextn_predict_layers") or 0
        hf_cfg["n_group"] = hf_cfg.get("n_group") or 1
        hf_cfg["topk_group"] = hf_cfg.get("topk_group") or 1

        # Reconstruct first_k_dense_replace from moe_layer_freq (count leading dense layers)
        moe_layer_freq = getattr(provider, "moe_layer_freq", None)
        if moe_layer_freq is not None and isinstance(moe_layer_freq, list):
            first_k_dense_replace = 0
            for val in moe_layer_freq:
                if val == 0:
                    first_k_dense_replace += 1
                else:
                    break
            hf_cfg["first_k_dense_replace"] = first_k_dense_replace

        # Reconstruct n_shared_experts from moe_shared_expert_intermediate_size / moe_ffn_hidden_size
        shared_size = getattr(provider, "moe_shared_expert_intermediate_size", None)
        moe_ffn = getattr(provider, "moe_ffn_hidden_size", None)
        if shared_size is not None and moe_ffn is not None and moe_ffn > 0:
            hf_cfg["n_shared_experts"] = shared_size // moe_ffn

        return hf_cfg

    def mapping_registry(self) -> MegatronMappingRegistry:
        mapping_list = get_common_mapping_list(hf_config=self.hf_config)
        return MegatronMappingRegistry(*mapping_list)

    def maybe_modify_loaded_hf_weight(
        self,
        hf_param: Union[str, dict[str, str]],
        hf_state_dict: Mapping[str, torch.Tensor],
    ) -> Union[torch.Tensor, dict[str, torch.Tensor]]:
        """Load HF weights and dequantize FP8 tensors on the fly.

        DeepSeek-V3 ships linear weights as ``float8_e4m3fn`` with per-block scale
        factors stored in ``<key>_scale_inv`` (128x128 blocks). The true bf16 weight is::

            w_bf16 = fp8_weight.float() * scale_inv_block

        Without this override the bridge would do a bare ``.to(bf16)`` cast in
        ``ColumnParallelMapping.hf_to_megatron`` (param_mapping.py:905), discarding the
        per-block scales — the resulting model produces random-looking logits.
        """
        hf_weights = super().maybe_modify_loaded_hf_weight(hf_param, hf_state_dict)

        if isinstance(hf_weights, dict):
            # Compound params (QKV / GatedMLP): dequantize each component individually.
            return {
                key: self._maybe_dequantize_fp8(tensor, hf_param[key], hf_state_dict)
                for key, tensor in hf_weights.items()
            }
        return self._maybe_dequantize_fp8(hf_weights, hf_param, hf_state_dict)

    @staticmethod
    def _maybe_dequantize_fp8(
        weight: torch.Tensor,
        param_name: str,
        hf_state_dict: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Dequantize ``weight`` if it is stored as FP8 with a matching ``*_scale_inv``."""
        scale_key = param_name + "_scale_inv"
        return quantization_utils.maybe_dequantize_fp8_blockwise(weight, hf_state_dict.get(scale_key))

    def maybe_modify_converted_hf_weight(
        self,
        task: WeightConversionTask,
        converted_weights_dict: Dict[str, torch.Tensor],
        hf_state_dict: Mapping[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Preserve a source rotary embedding inverse frequency tensor if present."""
        global_name = task.global_param_name
        if not global_name.startswith("decoder.layers.") or not global_name.endswith(".input_layernorm.weight"):
            return converted_weights_dict

        parts = global_name.split(".")
        if len(parts) < 4 or not parts[2].isdigit():
            return converted_weights_dict

        layer_idx = int(parts[2])
        inv_freq_key = f"model.layers.{layer_idx}.self_attn.rotary_emb.inv_freq"
        if inv_freq_key in converted_weights_dict:
            return converted_weights_dict

        source_inv_freq = hf_state_dict.get(inv_freq_key)
        if source_inv_freq is not None:
            if converted_weights_dict:
                reference_tensor = next(iter(converted_weights_dict.values()))
                source_inv_freq = source_inv_freq.to(device=reference_tensor.device)
            converted_weights_dict[inv_freq_key] = source_inv_freq
            return converted_weights_dict

        return converted_weights_dict
