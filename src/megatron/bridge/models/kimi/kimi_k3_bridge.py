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

"""Hugging Face ↔ Megatron conversion for the Kimi K3 language backbone.

The initial parameter mapping was adapted from the Apache-2.0 Miles implementation:
https://github.com/radixark/miles/blob/dc62a0bd4b7af1c59ee2084852eb18b5585ec082/miles_plugins/mbridge/kimi_k3.py
"""

from collections.abc import Iterable, Mapping

import torch
import torch.nn.functional as F
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.enums import AttnBackend

from megatron.bridge.models.conversion import quantization_utils
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import HFWeightTuple, MegatronModelBridge, WeightConversionTask
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ColumnParallelMapping,
    GatedMLPMapping,
    ReplicatedMapping,
    RowParallelMapping,
)
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.kimi.kimi_k3_provider import KimiK3ModelProvider


@MegatronModelBridge.register_bridge(
    source="KimiK3ForConditionalGeneration",
    target=GPTModel,
    provider=KimiK3ModelProvider,
    model_type="kimi_k3",
)
class KimiK3Bridge(MegatronModelBridge):
    """Megatron Bridge for the Kimi K3 language backbone."""

    _HF_PASSTHROUGH_PREFIXES = ("vision_tower.", "mm_projector.")

    @classmethod
    def hf_to_megatron_activation(cls, hidden_act: str):
        """Use SiLU as the config marker; the K3 spec installs the exact SiTU module."""
        if hidden_act == "situ":
            return F.silu
        return super().hf_to_megatron_activation(hidden_act)

    @classmethod
    def megatron_to_hf_activation(cls, activation_func) -> str:
        """Serialize the custom activation with its Hugging Face name."""
        if activation_func is F.silu:
            return "situ"
        return super().megatron_to_hf_activation(activation_func)

    def hf_config_to_provider_kwargs(self, hf_config) -> dict:
        """Map the nested K3 language configuration with the common config mappings."""
        return super().hf_config_to_provider_kwargs(getattr(hf_config, "text_config", hf_config))

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> KimiK3ModelProvider:
        """Translate the nested Kimi K3 text configuration."""
        from megatron.bridge.models.kimi.kimi_k3_spec import build_kimi_k3_spec

        hf_config = hf_pretrained.config
        text_config = hf_config.text_config
        provider = super().provider_bridge(hf_pretrained)

        provider.transformer_layer_spec = build_kimi_k3_spec
        provider.normalization = "RMSNorm"
        provider.gated_linear_unit = True
        provider.add_bias_linear = False
        provider.add_qkv_bias = False
        provider.share_embeddings_and_output_weights = False
        provider.qk_layernorm = True
        provider.multi_latent_attention = True
        provider.position_embedding_type = "none"
        provider.attention_backend = AttnBackend.auto

        provider.moe_latent_size = text_config.routed_expert_hidden_size
        provider.moe_router_topk = text_config.num_experts_per_token
        provider.moe_router_score_function = text_config.moe_router_activation_func
        provider.moe_router_pre_softmax = True
        if text_config.use_grouped_topk:
            provider.moe_router_num_groups = text_config.num_expert_group
            provider.moe_router_group_topk = text_config.topk_group
        provider.moe_router_enable_expert_bias = True
        provider.moe_router_bias_update_rate = 0.0
        provider.moe_router_dtype = "fp32"
        provider.moe_router_load_balancing_type = "none"
        provider.moe_aux_loss_coeff = 0.0
        provider.moe_grouped_gemm = True
        provider.moe_shared_expert_intermediate_size = (
            text_config.moe_intermediate_size * text_config.num_shared_experts
        )
        provider.moe_shared_expert_overlap = False
        provider.moe_token_dispatcher_type = "alltoall"
        provider.moe_permute_fusion = True

        moe_layer_freq = [0] * text_config.num_hidden_layers
        for layer_idx in range(text_config.first_k_dense_replace, text_config.num_hidden_layers):
            if layer_idx % text_config.moe_layer_freq == 0:
                moe_layer_freq[layer_idx] = 1
        provider.moe_layer_freq = moe_layer_freq

        linear_config = text_config.linear_attn_config
        provider.kimi_kda_layers = tuple(linear_config["kda_layers"])
        provider.kimi_linear_num_heads = linear_config["num_heads"]
        provider.kimi_linear_head_dim = linear_config["head_dim"]
        provider.kimi_linear_conv_kernel_size = linear_config["short_conv_kernel_size"]
        provider.kimi_kda_gate_lower_bound = linear_config["gate_lower_bound"]
        provider.kimi_attn_res_block_size = text_config.attn_res_block_size

        provider.use_te_activation_func = True
        provider.bias_activation_fusion = False
        provider.bias_dropout_fusion = False
        provider.hidden_dropout = 0.0
        provider.attention_dropout = 0.0
        provider.attention_softmax_in_fp32 = True
        provider.disable_bf16_reduced_precision_matmul = True
        provider.apply_rope_fusion = False
        provider.gradient_accumulation_fusion = True
        provider.cross_entropy_fusion_impl = "te"
        provider.cross_entropy_loss_fusion = True
        provider.masked_softmax_fusion = True
        provider.persist_layer_norm = True
        provider.should_pad_vocab = False

        self._num_hidden_layers = text_config.num_hidden_layers
        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Map K3's nested language model and custom layer parameters."""
        prefix = "language_model.model"
        megatron_layer = "decoder.layers.*"
        hf_layer = f"{prefix}.layers.*"
        megatron_attention = f"{megatron_layer}.self_attention"
        hf_attention = f"{hf_layer}.self_attn"

        auto_mappings = [
            ("embedding.word_embeddings.weight", f"{prefix}.embed_tokens.weight"),
            ("output_layer.weight", "language_model.lm_head.weight"),
        ]
        replicated_mappings = [
            ("decoder.final_layernorm.weight", f"{prefix}.norm.weight"),
            (f"{megatron_layer}.input_layernorm.weight", f"{hf_layer}.input_layernorm.weight"),
            (f"{megatron_attention}.f_a_proj.weight", f"{hf_attention}.f_a_proj.weight"),
            (f"{megatron_attention}.o_norm.weight", f"{hf_attention}.o_norm.weight"),
            (f"{megatron_attention}.q_a_proj.weight", f"{hf_attention}.q_a_proj.weight"),
            (f"{megatron_attention}.q_a_layernorm.weight", f"{hf_attention}.q_a_layernorm.weight"),
            (f"{megatron_attention}.kv_a_proj_with_mqa.weight", f"{hf_attention}.kv_a_proj_with_mqa.weight"),
            (f"{megatron_attention}.kv_a_layernorm.weight", f"{hf_attention}.kv_a_layernorm.weight"),
            (f"{megatron_layer}.self_attention_res_norm.weight", f"{hf_layer}.self_attention_res_norm.weight"),
            (f"{megatron_layer}.self_attention_res_proj.weight", f"{hf_layer}.self_attention_res_proj.weight"),
            (f"{megatron_layer}.pre_mlp_layernorm.weight", f"{hf_layer}.post_attention_layernorm.weight"),
            (f"{megatron_layer}.mlp_res_norm.weight", f"{hf_layer}.mlp_res_norm.weight"),
            (f"{megatron_layer}.mlp_res_proj.weight", f"{hf_layer}.mlp_res_proj.weight"),
            (f"{megatron_layer}.mlp.router.weight", f"{hf_layer}.block_sparse_moe.gate.weight"),
            (
                f"{megatron_layer}.mlp.router.expert_bias",
                f"{hf_layer}.block_sparse_moe.gate.e_score_correction_bias",
            ),
            (
                f"{megatron_layer}.mlp.fc1_latent_proj.weight",
                f"{hf_layer}.block_sparse_moe.routed_expert_down_proj.weight",
            ),
            (
                f"{megatron_layer}.mlp.routed_expert_norm.weight",
                f"{hf_layer}.block_sparse_moe.routed_expert_norm.weight",
            ),
            (
                f"{megatron_layer}.mlp.fc2_latent_proj.weight",
                f"{hf_layer}.block_sparse_moe.routed_expert_up_proj.weight",
            ),
        ]
        column_parallel_mappings = [
            (f"{megatron_attention}.{name}", f"{hf_attention}.{name}")
            for name in (
                "q_proj.weight",
                "k_proj.weight",
                "v_proj.weight",
                "q_conv1d.weight",
                "k_conv1d.weight",
                "v_conv1d.weight",
                "A_log",
                "dt_bias",
                "f_b_proj.weight",
                "b_proj.weight",
                "g_proj.weight",
                "q_b_proj.weight",
                "kv_b_proj.weight",
            )
        ]
        row_parallel_mappings = [
            (f"{megatron_attention}.o_proj.weight", f"{hf_attention}.o_proj.weight"),
            (f"{megatron_layer}.mlp.linear_fc2.weight", f"{hf_layer}.mlp.down_proj.weight"),
            (
                f"{megatron_layer}.mlp.shared_experts.linear_fc2.weight",
                f"{hf_layer}.block_sparse_moe.shared_experts.down_proj.weight",
            ),
            (
                f"{megatron_layer}.mlp.experts.linear_fc2.weight*",
                f"{hf_layer}.block_sparse_moe.experts.*.w2.weight",
            ),
            (
                f"{megatron_layer}.mlp.experts.local_experts.*.linear_fc2.weight",
                f"{hf_layer}.block_sparse_moe.experts.*.w2.weight",
            ),
        ]
        mappings = [
            *(AutoMapping(*mapping) for mapping in auto_mappings),
            *(ReplicatedMapping(*mapping) for mapping in replicated_mappings),
            *(ColumnParallelMapping(*mapping) for mapping in column_parallel_mappings),
            *(RowParallelMapping(*mapping) for mapping in row_parallel_mappings),
            GatedMLPMapping(
                f"{megatron_layer}.mlp.linear_fc1.weight",
                gate=f"{hf_layer}.mlp.gate_proj.weight",
                up=f"{hf_layer}.mlp.up_proj.weight",
            ),
            GatedMLPMapping(
                f"{megatron_layer}.mlp.shared_experts.linear_fc1.weight",
                gate=f"{hf_layer}.block_sparse_moe.shared_experts.gate_proj.weight",
                up=f"{hf_layer}.block_sparse_moe.shared_experts.up_proj.weight",
            ),
            GatedMLPMapping(
                f"{megatron_layer}.mlp.experts.linear_fc1.weight*",
                gate=f"{hf_layer}.block_sparse_moe.experts.*.w1.weight",
                up=f"{hf_layer}.block_sparse_moe.experts.*.w3.weight",
            ),
            GatedMLPMapping(
                f"{megatron_layer}.mlp.experts.local_experts.*.linear_fc1.weight",
                gate=f"{hf_layer}.block_sparse_moe.experts.*.w1.weight",
                up=f"{hf_layer}.block_sparse_moe.experts.*.w3.weight",
            ),
        ]

        num_hidden_layers = getattr(self, "_num_hidden_layers", None)
        if num_hidden_layers is None and self.hf_config is not None:
            num_hidden_layers = self.hf_config.text_config.num_hidden_layers
        if num_hidden_layers is None:
            raise ValueError("Kimi K3 mapping construction requires num_hidden_layers")
        last_layer = num_hidden_layers - 1
        mappings.extend(
            [
                ReplicatedMapping(
                    f"decoder.layers.{last_layer}.output_attn_res_norm.weight",
                    f"{prefix}.output_attn_res_norm.weight",
                ),
                ReplicatedMapping(
                    f"decoder.layers.{last_layer}.output_attn_res_proj.weight",
                    f"{prefix}.output_attn_res_proj.weight",
                ),
            ]
        )
        return MegatronMappingRegistry(*mappings)

    @staticmethod
    def _load_one_hf_weight(name: str, hf_state_dict: Mapping[str, torch.Tensor]) -> torch.Tensor:
        packed_key = f"{name}_packed" if name.endswith(".weight") else ""
        scale_key = f"{name}_scale" if name.endswith(".weight") else ""
        if packed_key and packed_key in hf_state_dict:
            if scale_key not in hf_state_dict:
                raise ValueError(f"Missing MXFP4 scale for {packed_key}")
            return quantization_utils.dequantize_mxfp4_e2m1_packed(
                hf_state_dict[packed_key],
                hf_state_dict[scale_key],
            )
        return hf_state_dict[name]

    def maybe_modify_loaded_hf_weight(
        self,
        hf_param: str | dict[str, str],
        hf_state_dict: Mapping[str, torch.Tensor],
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Load K3 weights, dequantizing routed experts from MXFP4 when needed."""
        if isinstance(hf_param, dict):
            return {key: self._load_one_hf_weight(name, hf_state_dict) for key, name in hf_param.items()}
        weight = self._load_one_hf_weight(hf_param, hf_state_dict)
        if hf_param.endswith(".self_attn.A_log"):
            num_heads = self.hf_config.text_config.linear_attn_config["num_heads"]
            padding = weight[num_heads:]
            if padding.numel() and torch.count_nonzero(padding).item():
                raise ValueError(f"Kimi K3 requires zero-padded inactive A_log entries in {hf_param}")
            weight = weight[:num_heads]
        return weight

    def maybe_modify_converted_hf_weight(
        self,
        task: WeightConversionTask,
        converted_weights_dict: dict[str, torch.Tensor],
        hf_state_dict: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Restore KDA padding and the source routed-expert MXFP4 representation."""
        result: dict[str, torch.Tensor] = {}
        for name, weight in converted_weights_dict.items():
            if name.endswith(".self_attn.A_log"):
                if name not in hf_state_dict:
                    raise ValueError(f"Missing source Kimi K3 A_log tensor {name}")
                source_weight = hf_state_dict[name]
                if weight.ndim != 1 or source_weight.ndim != 1 or weight.shape[0] > source_weight.shape[0]:
                    raise ValueError(
                        f"Cannot restore Kimi K3 A_log padding for {name}: "
                        f"Megatron shape {tuple(weight.shape)}, HF shape {tuple(source_weight.shape)}"
                    )
                padding_size = source_weight.shape[0] - weight.shape[0]
                result[name] = torch.cat((weight, weight.new_zeros(padding_size)))
                continue

            if task.weight_dtype is not None:
                result[name] = weight
                continue

            packed_key = f"{name}_packed" if name.endswith(".weight") else ""
            scale_key = f"{name}_scale" if name.endswith(".weight") else ""
            if packed_key and packed_key in hf_state_dict:
                source_scale = hf_state_dict[scale_key]
                packed, scale = quantization_utils.quantize_mxfp4_e2m1_like_scale(
                    weight,
                    source_scale,
                    name=name,
                )
                result[packed_key] = packed.view(hf_state_dict[packed_key].dtype)
                result[scale_key] = scale
            else:
                result[name] = weight
        return result

    def build_conversion_tasks(
        self,
        hf_pretrained,
        megatron_model,
        weight_dtype=None,
    ) -> list[WeightConversionTask]:
        """Expose virtual BF16 expert weights for MXFP4 packed/scale pairs."""
        original_get_all_keys = hf_pretrained.state.source.get_all_keys

        def _get_all_keys_with_virtual() -> list[str]:
            keys = original_get_all_keys()
            all_keys = set(keys)
            virtual_keys = []
            for key in keys:
                if key.endswith(".weight_packed"):
                    virtual_key = key[: -len("_packed")]
                    if f"{virtual_key}_scale" in all_keys:
                        virtual_keys.append(virtual_key)
            return keys + virtual_keys

        hf_pretrained.state.source.get_all_keys = _get_all_keys_with_virtual
        try:
            return super().build_conversion_tasks(hf_pretrained, megatron_model, weight_dtype=weight_dtype)
        finally:
            hf_pretrained.state.source.get_all_keys = original_get_all_keys

    @torch.no_grad()
    def stream_weights_megatron_to_hf(
        self,
        megatron_model: GPTModel | list[GPTModel],
        hf_pretrained: PreTrainedCausalLM,
        cpu: bool = True,
        show_progress: bool = True,
        conversion_tasks: list[WeightConversionTask] | None = None,
        merge_adapter_weights: bool = True,
        weight_dtype: torch.dtype | None = None,
    ) -> Iterable[HFWeightTuple]:
        """Export the language model and preserve unchanged multimodal weights."""
        yield from super().stream_weights_megatron_to_hf(
            megatron_model,
            hf_pretrained,
            cpu=cpu,
            show_progress=show_progress,
            conversion_tasks=conversion_tasks,
            merge_adapter_weights=merge_adapter_weights,
            weight_dtype=weight_dtype,
        )

        state = getattr(hf_pretrained, "state", None)
        source = getattr(state, "source", None)
        if source is None:
            return
        for name in source.get_all_keys():
            if name.startswith(self._HF_PASSTHROUGH_PREFIXES):
                yield from HFWeightTuple(name, state[name]).iter_finalized(cpu=cpu)
