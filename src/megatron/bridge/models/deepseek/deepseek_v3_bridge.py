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
from megatron.core.models.gpt.gpt_model import GPTModel

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge, WeightConversionTask
from megatron.bridge.models.deepseek.attention import (
    get_deepseek_decoder_block_spec,
    replace_mla_self_attention,
)
from megatron.bridge.models.deepseek.common import get_common_mapping_list
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.mla_provider import MLAModelProvider


try:
    import transformer_engine  # noqa: F401

    HAVE_TE = True
except (ImportError, ModuleNotFoundError):
    HAVE_TE = False


__all__ = ["DeepSeekV3Bridge"]


@MegatronModelBridge.register_bridge(
    source="DeepseekV3ForCausalLM",
    target=GPTModel,
    provider=MLAModelProvider,
    model_type="deepseek_v3",
)
class DeepSeekV3Bridge(MegatronModelBridge):
    """Megatron Bridge for DeepSeek-V3.

    DeepSeek-V3 ships its linear weights as FP8 with per-block ``<name>_scale_inv`` scales. The base
    ``_load_hf_weights`` dequantizes them on import with the checkpoint's
    ``quantization_config.weight_block_size``.
    """

    @staticmethod
    def generate_pipeline_layout(num_layers: int, pp: int, mtp_layers: int = 1) -> list[list[str]]:
        """Generate a pipeline-parallel layout for DeepSeek V3 conversion.

        DeepSeek V3 has 61 decoder layers, so the model cannot use ordinary
        pipeline partitioning for the practical PP sizes needed to hold the
        full checkpoint. The conversion launcher calls this hook to distribute
        decoder layers unevenly while keeping embeddings on the first stage and
        MTP plus loss on the last stage.

        Args:
            num_layers: Number of decoder layers.
            pp: Pipeline parallel size.
            mtp_layers: Number of MTP layers.

        Returns:
            A flexible pipeline layout with exactly ``pp`` stages.
        """
        base_layers, extra_layers = divmod(num_layers, pp)
        layout: list[list[str]] = []
        for pp_rank in range(pp):
            stage = ["decoder"] * (base_layers + int(pp_rank < extra_layers))
            if pp_rank == 0:
                stage.insert(0, "embedding")
            if pp_rank == pp - 1:
                stage.extend(["mtp"] * mtp_layers)
                stage.append("loss")
            layout.append(stage)
        return layout

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> MLAModelProvider:
        provider = super().provider_bridge(hf_pretrained)
        hf_config = hf_pretrained.config

        provider.transformer_layer_spec = partial(get_deepseek_decoder_block_spec, use_transformer_engine=HAVE_TE)
        # A standalone MTP stage has no decoder layers, so the provider re-derives its
        # layer spec from MCore and never calls the builder above. Re-apply the swap there.
        provider.mtp_layer_spec_transform = replace_mla_self_attention
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
        # The released V3 configs omit this training-only hyperparameter, but V3 uses a
        # small complementary sequence-wise loss alongside expert-bias load balancing.
        provider.moe_aux_loss_coeff = getattr(hf_config, "aux_loss_alpha", 0.0001)
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

        # MLA keeps its YaRN settings in unprefixed fields, which the base export does not read. Export
        # them, or "no scaling", so a config-only export does not keep a reference model's scaling.
        rotary_scaling_factor = getattr(provider, "rotary_scaling_factor", None)
        if rotary_scaling_factor is None or rotary_scaling_factor == 1.0:
            hf_cfg["rope_scaling"] = {"rope_type": "default"}
        else:
            hf_cfg["rope_scaling"] = {"rope_type": "yarn"}
            for hf_key, megatron_key in cls.MLA_ROPE_SCALING_MAPPING:
                value = getattr(provider, megatron_key, None)
                if value is not None:
                    hf_cfg["rope_scaling"][hf_key] = value

        return hf_cfg

    def mapping_registry(self) -> MegatronMappingRegistry:
        mapping_list = get_common_mapping_list(hf_config=self.hf_config)
        return MegatronMappingRegistry(*mapping_list)

    def maybe_modify_converted_hf_weight(
        self,
        task: WeightConversionTask,
        converted_weights_dict: Dict[str, torch.Tensor],
        hf_state_dict: Mapping[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Preserve source-required shared MTP aliases and rotary frequencies."""
        global_name = task.global_param_name
        num_layers = getattr(self.hf_config, "num_hidden_layers", None)
        if isinstance(num_layers, int):
            shared_mtp_aliases = {
                "embedding.word_embeddings.weight": (
                    "model.embed_tokens.weight",
                    f"model.layers.{num_layers}.embed_tokens.weight",
                ),
                "output_layer.weight": (
                    "lm_head.weight",
                    f"model.layers.{num_layers}.shared_head.head.weight",
                ),
            }
            alias = shared_mtp_aliases.get(global_name)
            if alias is not None:
                source_key, alias_key = alias
                if source_key in converted_weights_dict and alias_key in hf_state_dict:
                    converted_weights_dict[alias_key] = converted_weights_dict[source_key]

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
