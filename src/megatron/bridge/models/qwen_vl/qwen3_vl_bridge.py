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

import torch
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ConcatenatedQKVMapping,
    GatedMLPMapping,
    QKVMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.conversion.transformers_compat import rope_theta_from_hf
from megatron.bridge.models.conversion.utils import moe_experts_stored_packed
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.qwen.qwen35_bridge import _moe_routed_expert_mappings
from megatron.bridge.models.qwen.qwen_hybrid import (
    configure_qwen_hybrid_layers,
    qwen_logical_layer_count,
    qwen_moe_layer_symbols,
    qwen_physical_layer_indices,
)
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model import Qwen3VLModel
from megatron.bridge.models.qwen_vl.qwen3_vl_provider import Qwen3VLModelProvider, Qwen3VLMoEModelProvider


def _vision_mappings() -> list:
    direct_mappings = {
        "vision_model.decoder.layers.*.self_attention.linear_proj.weight": "model.visual.blocks.*.attn.proj.weight",
        "vision_model.decoder.layers.*.self_attention.linear_proj.bias": "model.visual.blocks.*.attn.proj.bias",
        "vision_model.decoder.layers.*.mlp.linear_fc1.weight": "model.visual.blocks.*.mlp.linear_fc1.weight",
        "vision_model.decoder.layers.*.mlp.linear_fc1.bias": "model.visual.blocks.*.mlp.linear_fc1.bias",
        "vision_model.decoder.layers.*.mlp.linear_fc2.weight": "model.visual.blocks.*.mlp.linear_fc2.weight",
        "vision_model.decoder.layers.*.mlp.linear_fc2.bias": "model.visual.blocks.*.mlp.linear_fc2.bias",
        "vision_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.visual.blocks.*.norm1.weight",
        "vision_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_bias": "model.visual.blocks.*.norm1.bias",
        "vision_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.visual.blocks.*.norm2.weight",
        "vision_model.decoder.layers.*.mlp.linear_fc1.layer_norm_bias": "model.visual.blocks.*.norm2.bias",
        "vision_model.decoder.deepstack_merger_list.*.patch_norm.weight": "model.visual.deepstack_merger_list.*.norm.weight",
        "vision_model.decoder.deepstack_merger_list.*.patch_norm.bias": "model.visual.deepstack_merger_list.*.norm.bias",
        "vision_model.decoder.deepstack_merger_list.*.linear_fc1.weight": "model.visual.deepstack_merger_list.*.linear_fc1.weight",
        "vision_model.decoder.deepstack_merger_list.*.linear_fc1.bias": "model.visual.deepstack_merger_list.*.linear_fc1.bias",
        "vision_model.decoder.deepstack_merger_list.*.linear_fc2.weight": "model.visual.deepstack_merger_list.*.linear_fc2.weight",
        "vision_model.decoder.deepstack_merger_list.*.linear_fc2.bias": "model.visual.deepstack_merger_list.*.linear_fc2.bias",
        "vision_model.merger.patch_norm.**": "model.visual.merger.norm.**",
        "vision_model.merger.linear_fc1.weight": "model.visual.merger.linear_fc1.weight",
        "vision_model.merger.linear_fc1.bias": "model.visual.merger.linear_fc1.bias",
        "vision_model.merger.linear_fc2.weight": "model.visual.merger.linear_fc2.weight",
        "vision_model.merger.linear_fc2.bias": "model.visual.merger.linear_fc2.bias",
    }
    mappings = [AutoMapping(k, v) for k, v in direct_mappings.items()]
    mappings.extend(
        [
            ConcatenatedQKVMapping(
                "vision_model.decoder.layers.*.self_attention.linear_qkv.weight",
                "model.visual.blocks.*.attn.qkv.weight",
            ),
            ConcatenatedQKVMapping(
                "vision_model.decoder.layers.*.self_attention.linear_qkv.bias",
                "model.visual.blocks.*.attn.qkv.bias",
            ),
            ReplicatedMapping("vision_model.patch_embed.proj.**", "model.visual.patch_embed.proj.**"),
            ReplicatedMapping("vision_model.pos_embed.weight", "model.visual.pos_embed.weight"),
        ]
    )
    return mappings


def _attention_mappings(logical_layer_idx: int, attention_layer_idx: int) -> list:
    hf_layer = f"model.language_model.layers.{logical_layer_idx}"
    attention_layer = f"language_model.decoder.layers.{attention_layer_idx}.self_attention"
    return [
        AutoMapping(f"{attention_layer}.linear_qkv.layer_norm_weight", f"{hf_layer}.input_layernorm.weight"),
        AutoMapping(f"{attention_layer}.linear_proj.weight", f"{hf_layer}.self_attn.o_proj.weight"),
        AutoMapping(f"{attention_layer}.q_layernorm.weight", f"{hf_layer}.self_attn.q_norm.weight"),
        AutoMapping(f"{attention_layer}.k_layernorm.weight", f"{hf_layer}.self_attn.k_norm.weight"),
        QKVMapping(
            megatron_param=f"{attention_layer}.linear_qkv.weight",
            q=f"{hf_layer}.self_attn.q_proj.weight",
            k=f"{hf_layer}.self_attn.k_proj.weight",
            v=f"{hf_layer}.self_attn.v_proj.weight",
        ),
        QKVMapping(
            megatron_param=f"{attention_layer}.linear_qkv.bias",
            q=f"{hf_layer}.self_attn.q_proj.bias",
            k=f"{hf_layer}.self_attn.k_proj.bias",
            v=f"{hf_layer}.self_attn.v_proj.bias",
        ),
    ]


def _dense_mlp_mappings(logical_layer_idx: int, mlp_layer_idx: int) -> list:
    hf_layer = f"model.language_model.layers.{logical_layer_idx}"
    mlp_layer = f"language_model.decoder.layers.{mlp_layer_idx}.mlp"
    return [
        AutoMapping(f"{mlp_layer}.linear_fc1.layer_norm_weight", f"{hf_layer}.post_attention_layernorm.weight"),
        AutoMapping(f"{mlp_layer}.linear_fc2.weight", f"{hf_layer}.mlp.down_proj.weight"),
        GatedMLPMapping(
            megatron_param=f"{mlp_layer}.linear_fc1.weight",
            gate=f"{hf_layer}.mlp.gate_proj.weight",
            up=f"{hf_layer}.mlp.up_proj.weight",
        ),
    ]


def _moe_mlp_mappings(logical_layer_idx: int, moe_layer_idx: int, *, experts_packed: bool) -> list:
    hf_layer = f"model.language_model.layers.{logical_layer_idx}"
    moe_layer = f"language_model.decoder.layers.{moe_layer_idx}"
    return [
        AutoMapping(f"{moe_layer}.pre_mlp_layernorm.weight", f"{hf_layer}.post_attention_layernorm.weight"),
        AutoMapping(f"{moe_layer}.mlp.router.weight", f"{hf_layer}.mlp.gate.weight"),
        *_moe_routed_expert_mappings(
            hf_layer,
            moe_layer,
            experts_packed,
            transpose_on_export=True,
        ),
    ]


class _Qwen3VLBridgeMixin:
    @classmethod
    def megatron_to_hf_config(cls, provider) -> dict:
        hf_config = super().megatron_to_hf_config(provider)
        logical_layer_count = qwen_logical_layer_count(provider.hybrid_layer_pattern)
        if logical_layer_count is not None:
            hf_config["num_hidden_layers"] = logical_layer_count
        return hf_config

    @staticmethod
    def _base_language_mappings() -> list:
        return [
            AutoMapping(
                "language_model.embedding.word_embeddings.weight",
                "model.language_model.embed_tokens.weight",
            ),
            AutoMapping("language_model.output_layer.weight", "lm_head.weight"),
            AutoMapping("language_model.decoder.final_norm.weight", "model.language_model.norm.weight"),
        ]


@MegatronModelBridge.register_bridge(
    source=Qwen3VLForConditionalGeneration,
    target=Qwen3VLModel,
    provider=Qwen3VLModelProvider,
    model_type="qwen3_vl",
)
class Qwen3VLBridge(_Qwen3VLBridgeMixin, MegatronModelBridge):
    """Megatron Bridge for dense Qwen3-VL conditional generation."""

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> Qwen3VLModelProvider:
        hf_config = hf_pretrained.config
        text_config = hf_config.text_config
        provider_kwargs = self.hf_config_to_provider_kwargs(text_config)
        vision_config = hf_config.vision_config
        vision_config.torch_dtype = provider_kwargs.get("params_dtype", torch.float32)
        provider = Qwen3VLModelProvider(**provider_kwargs)

        provider.normalization = "RMSNorm"
        provider.gated_linear_unit = True
        provider.add_qkv_bias = text_config.attention_bias
        provider.add_bias_linear = False
        provider.qk_layernorm = True
        provider.hidden_dropout = 0.0
        provider.rotary_base = rope_theta_from_hf(text_config)
        provider.share_embeddings_and_output_weights = getattr(hf_config, "tie_word_embeddings", False)
        provider.position_embedding_type = "mrope"
        provider.vision_config = vision_config
        provider.hf_text_config = text_config
        provider.head_dim = text_config.head_dim
        provider.bos_token_id = getattr(text_config, "bos_token_id", 151643)
        provider.eos_token_id = getattr(text_config, "eos_token_id", 151645)
        provider.vision_start_token_id = getattr(hf_config, "vision_start_token_id", 151652)
        provider.vision_end_token_id = getattr(hf_config, "vision_end_token_id", 151653)
        provider.image_token_id = getattr(hf_config, "image_token_id", 151655)
        provider.video_token_id = getattr(hf_config, "video_token_id", 151656)
        rope_cfg = getattr(text_config, "rope_parameters", None) or getattr(text_config, "rope_scaling", {})
        provider.mrope_section = rope_cfg.get("mrope_section", [24, 20, 20])
        configure_qwen_hybrid_layers(
            provider,
            num_logical_layers=text_config.num_hidden_layers,
            mlp_symbols=Symbols.MLP,
            mtp_mlp_symbol=Symbols.MLP,
        )
        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        mappings = self._base_language_mappings()
        for logical_layer_idx in range(self.hf_config.text_config.num_hidden_layers):
            attention_layer_idx, mlp_layer_idx = qwen_physical_layer_indices(logical_layer_idx)
            mappings.extend(_attention_mappings(logical_layer_idx, attention_layer_idx))
            mappings.extend(_dense_mlp_mappings(logical_layer_idx, mlp_layer_idx))
        mappings.extend(_vision_mappings())
        return MegatronMappingRegistry(*mappings)


@MegatronModelBridge.register_bridge(
    source=Qwen3VLMoeForConditionalGeneration,
    target=Qwen3VLModel,
    provider=Qwen3VLMoEModelProvider,
    model_type="qwen3_vl_moe",
)
class Qwen3VLMoEBridge(_Qwen3VLBridgeMixin, MegatronModelBridge):
    """Megatron Bridge for Qwen3-VL MoE conditional generation."""

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> Qwen3VLMoEModelProvider:
        hf_config = hf_pretrained.config
        text_config = hf_config.text_config
        provider_kwargs = self.hf_config_to_provider_kwargs(text_config)
        vision_config = hf_config.vision_config
        vision_config.torch_dtype = provider_kwargs.get("params_dtype", torch.float32)
        provider = Qwen3VLMoEModelProvider(**provider_kwargs)

        provider.normalization = "RMSNorm"
        provider.gated_linear_unit = True
        provider.add_qkv_bias = text_config.attention_bias
        provider.add_bias_linear = False
        provider.qk_layernorm = True
        provider.hidden_dropout = 0.0
        provider.rotary_base = rope_theta_from_hf(text_config)
        provider.share_embeddings_and_output_weights = getattr(hf_config, "tie_word_embeddings", False)
        provider.moe_ffn_hidden_size = text_config.moe_intermediate_size
        provider.num_moe_experts = text_config.num_experts
        provider.moe_router_topk = text_config.num_experts_per_tok
        provider.decoder_sparse_step = getattr(text_config, "decoder_sparse_step", 1)
        provider.mlp_only_layers = getattr(text_config, "mlp_only_layers", [])
        provider.moe_grouped_gemm = True
        provider.moe_router_load_balancing_type = "aux_loss"
        provider.moe_aux_loss_coeff = 1e-3
        provider.moe_router_pre_softmax = False
        provider.moe_token_dispatcher_type = "alltoall"
        provider.moe_permute_fusion = True
        provider.position_embedding_type = "mrope"
        provider.vision_config = vision_config
        provider.hf_text_config = text_config
        provider.head_dim = getattr(
            text_config,
            "head_dim",
            text_config.hidden_size // text_config.num_attention_heads,
        )
        provider.bos_token_id = getattr(text_config, "bos_token_id", 151643)
        provider.eos_token_id = getattr(text_config, "eos_token_id", 151645)
        provider.vision_start_token_id = getattr(hf_config, "vision_start_token_id", 151652)
        provider.vision_end_token_id = getattr(hf_config, "vision_end_token_id", 151653)
        provider.image_token_id = getattr(hf_config, "image_token_id", 151655)
        provider.video_token_id = getattr(hf_config, "video_token_id", 151656)
        rope_cfg = getattr(text_config, "rope_parameters", None) or getattr(text_config, "rope_scaling", {})
        provider.mrope_section = rope_cfg.get("mrope_section", [24, 20, 20])
        mlp_symbols = qwen_moe_layer_symbols(
            text_config.num_hidden_layers,
            decoder_sparse_step=provider.decoder_sparse_step,
            mlp_only_layers=provider.mlp_only_layers,
        )
        configure_qwen_hybrid_layers(
            provider,
            num_logical_layers=text_config.num_hidden_layers,
            mlp_symbols=mlp_symbols,
            mtp_mlp_symbol=Symbols.MOE,
        )
        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        text_config = self.hf_config.text_config
        mlp_symbols = qwen_moe_layer_symbols(
            text_config.num_hidden_layers,
            decoder_sparse_step=getattr(text_config, "decoder_sparse_step", 1),
            mlp_only_layers=getattr(text_config, "mlp_only_layers", []),
        )
        experts_packed = moe_experts_stored_packed(
            getattr(self, "hf_pretrained", None),
            "model.language_model.layers.",
        )
        mappings = self._base_language_mappings()
        for logical_layer_idx, mlp_symbol in enumerate(mlp_symbols):
            attention_layer_idx, mlp_layer_idx = qwen_physical_layer_indices(logical_layer_idx)
            mappings.extend(_attention_mappings(logical_layer_idx, attention_layer_idx))
            if mlp_symbol == Symbols.MOE:
                mappings.extend(
                    _moe_mlp_mappings(
                        logical_layer_idx,
                        mlp_layer_idx,
                        experts_packed=experts_packed,
                    )
                )
            else:
                mappings.extend(_dense_mlp_mappings(logical_layer_idx, mlp_layer_idx))
        mappings.extend(_vision_mappings())
        return MegatronMappingRegistry(*mappings)
