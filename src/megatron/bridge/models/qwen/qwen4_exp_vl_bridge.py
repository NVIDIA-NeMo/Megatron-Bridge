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

"""Megatron Bridge for Qwen4-Exp (Qwen3.8-Flash-Next) Vision-Language models.

The published checkpoint (``Qwen4ExpForConditionalGeneration``) is a VL model: a
Qwen4-Exp language model (GDN + QSA + PLE + Gated-Residual hyper connections + MoE,
see ``qwen4_exp_bridge``) plus a Qwen3-VL-style vision tower. This module bridges the
*whole* model to :class:`~megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model.Qwen3VLModel`:

* the language decoder reuses ``Qwen4ExpBridge.get_lm_mappings`` with the VL
  megatron prefix (``language_model.``) — QSA indexer, PLE shards, hyper
  connections and MoE map exactly as in the text-only bridge;
* the vision tower reuses the Qwen3-VL mapping table
  (``qwen35_vl_bridge._get_vision_mappings``): the fork's
  ``Qwen4ExpVisionModel`` shares the Qwen3.5/Qwen3-VL layout (``model.visual.*``).

The language model runs under mRoPE (``mrope_section`` from the HF
``rope_parameters``), like Qwen3.5-VL. Note that the full-attention layers are QSA
(sparse indexer attention), *not* the standard attention that
``Qwen35VLMoEModelProvider`` patches with ``Qwen3VLSelfAttention``; the spec is
therefore used unpatched so QSA stays intact.
"""

import torch
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_transformer_block_with_experimental_attention_variant_spec,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.utils import moe_experts_stored_packed
from megatron.bridge.models.qwen.qwen4_exp_bridge import (
    Qwen4ExpBridge,
    get_qwen4_exp_hf_lm_prefix,
    get_qwen4_exp_text_config,
    linear_attention_pattern_from_hf,
)
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model import Qwen3VLModel
from megatron.bridge.models.qwen_vl.qwen35_vl_bridge import _get_vision_mappings
from megatron.bridge.models.qwen_vl.qwen35_vl_provider import Qwen35VLMoEModelProvider


class Qwen4ExpVLMoEModelProvider(Qwen35VLMoEModelProvider):
    """Model provider for Qwen4-Exp VL (mRoPE language model + Qwen3-VL vision tower).

    All Qwen4-Exp-specific fields (PLE / QSA indexer / Gated-Residual hyper
    connections) are populated dynamically by ``Qwen4ExpVLBridge.provider_bridge``,
    exactly as ``Qwen4ExpBridge.provider_bridge`` populates a plain
    ``GPTModelProvider`` for the text-only path.
    """

    def build_language_spec(self, vp_stage=None, pp_rank=None) -> TransformerBlockSubmodules | ModuleSpec:
        """Build the Qwen4-Exp language transformer-block spec.

        Unlike the Qwen3.5 VL provider, the full-attention layers are QSA, so the
        spec is returned *without* the ``Qwen3VLSelfAttention`` patch (that patch is
        a mRoPE shim for standard attention and would replace the QSA modules).
        """
        return get_transformer_block_with_experimental_attention_variant_spec(self, vp_stage=vp_stage, pp_rank=pp_rank)


@MegatronModelBridge.register_bridge(
    source="Qwen4ExpForConditionalGeneration",
    target=Qwen3VLModel,
    provider=Qwen4ExpVLMoEModelProvider,
    model_type="qwen4_exp",
)
class Qwen4ExpVLBridge(Qwen4ExpBridge):
    """Bridge the VL Qwen4-Exp checkpoint (language model + vision tower) to Qwen3VLModel."""

    mimo_source_prefixes = {"language": "language_model.", "images": "vision_model."}

    def provider_bridge(self, hf_pretrained) -> Qwen4ExpVLMoEModelProvider:
        """Convert the HuggingFace Qwen4-Exp VL config into a Qwen4ExpVLMoEModelProvider."""
        hf_config = hf_pretrained.config
        text_config = get_qwen4_exp_text_config(hf_config)
        rope_parameters = getattr(text_config, "rope_parameters", None) or {}
        tie_word_embeddings = getattr(hf_config, "tie_word_embeddings", False) or getattr(
            text_config, "tie_word_embeddings", False
        )

        provider = Qwen4ExpVLMoEModelProvider(
            num_layers=text_config.num_hidden_layers,
            hidden_size=text_config.hidden_size,
            ffn_hidden_size=text_config.moe_intermediate_size,
            num_attention_heads=text_config.num_attention_heads,
            num_query_groups=text_config.num_key_value_heads,
            kv_channels=text_config.head_dim,
            vocab_size=text_config.vocab_size,
            seq_length=text_config.max_position_embeddings,
            layernorm_epsilon=text_config.rms_norm_eps,
            init_method_std=text_config.initializer_range,
            attention_dropout=text_config.attention_dropout,
            hidden_dropout=0.0,
            rotary_base=rope_parameters.get("rope_theta", 10_000_000),
            share_embeddings_and_output_weights=tie_word_embeddings,
        )

        # --- Qwen3.5-MoE base: zero-centered RMSNorm, gated attention with QK norm, GDN hybrid ---
        provider.activation_func = self.hf_to_megatron_activation(text_config.hidden_act)
        provider.normalization = "RMSNorm"
        provider.layernorm_zero_centered_gamma = True
        provider.gated_linear_unit = True
        provider.add_qkv_bias = getattr(text_config, "attention_bias", False)
        provider.add_bias_linear = False
        provider.qk_layernorm = True
        provider.attention_output_gate = True
        provider.experimental_attention_variant = "gdn"
        provider.linear_attention_freq = linear_attention_pattern_from_hf(text_config)
        provider.linear_conv_kernel_dim = text_config.linear_conv_kernel_dim
        provider.linear_key_head_dim = text_config.linear_key_head_dim
        provider.linear_value_head_dim = text_config.linear_value_head_dim
        provider.linear_num_key_heads = text_config.linear_num_key_heads
        provider.linear_num_value_heads = text_config.linear_num_value_heads
        provider.linear_attention_output_gate_activation = getattr(text_config, "output_gate_type", None) or (
            "silu" if text_config.hidden_act in ("silu", "swish") else text_config.hidden_act
        )
        provider.rotary_percent = rope_parameters.get("partial_rotary_factor", 0.25)

        # --- MoE: 512 routed experts (top-k, softmax-then-topk), gated shared expert ---
        provider.moe_ffn_hidden_size = text_config.moe_intermediate_size
        provider.num_moe_experts = text_config.num_experts
        provider.moe_router_topk = text_config.num_experts_per_tok
        provider.moe_shared_expert_intermediate_size = text_config.shared_expert_intermediate_size
        provider.moe_shared_expert_gate = True
        provider.moe_grouped_gemm = True
        provider.moe_router_load_balancing_type = "global_aux_loss"
        provider.moe_aux_loss_coeff = getattr(text_config, "router_aux_loss_coef", 0.001)
        provider.moe_router_pre_softmax = False
        provider.moe_token_dispatcher_type = "alltoall"
        provider.moe_permute_fusion = True
        provider.moe_router_dtype = "fp32"

        # --- Gated Residual hyper connections ---
        provider.enable_mhc_connections = True
        provider.mhc_variant = "gated_residual"
        provider.mhc_num_residual_streams = text_config.hc_count
        provider.mhc_gated_residual_rank = text_config.hc_lowrank

        # --- Qwen Sparse Attention on the softmax-attention layers ---
        if getattr(text_config, "indexer_n_heads", None) is not None:
            provider.qsa_indexer_n_heads = text_config.indexer_n_heads
            provider.qsa_indexer_kv_heads = text_config.indexer_kv_heads
            provider.qsa_indexer_head_dim = text_config.indexer_head_dim
            provider.qsa_indexer_budget = text_config.indexer_budget
            provider.qsa_indexer_compress_ratio = text_config.indexer_compress_ratio

        # --- Per-layer n-gram embedding ---
        ple_layer_ids = list(getattr(text_config, "ple_layer_ids", None) or [])
        if ple_layer_ids:
            eos_token_id = text_config.eos_token_id
            if isinstance(eos_token_id, (list, tuple)):
                eos_token_id = eos_token_id[0]
            provider.ple_layer_ids = ple_layer_ids
            provider.ple_embed_dim = getattr(text_config, "ple_embed_dim", None) or text_config.hidden_size
            provider.ple_conv_kernel_size = text_config.ple_conv_kernel_size
            provider.ple_ngram_size = text_config.ngram_size
            provider.ple_heads_per_ngram = text_config.heads_per_ngram
            provider.ple_ngram_vocab_size_base = text_config.ngram_vocab_size_base
            provider.ple_ngram_vocab_divisible_by = text_config.make_ngram_vocab_size_divisible_by
            provider.ple_seed = getattr(text_config, "seed", 1234)
            provider.ple_eos_token_id = eos_token_id
            provider.ple_unigram_vocab_size = text_config.vocab_size

        # MTP: the checkpoint carries a 1-layer MTP head; it is not built for the Megatron model.
        provider.mtp_num_layers = None

        # --- VL overrides: mRoPE + vision tower + modality token ids ---
        provider.position_embedding_type = "mrope"
        provider.mrope_section = rope_parameters.get("mrope_section", [11, 11, 10])
        provider.head_dim = getattr(text_config, "head_dim", 256)
        provider.bos_token_id = getattr(text_config, "bos_token_id", 248045)
        eos = getattr(text_config, "eos_token_id", 248046)
        provider.eos_token_id = eos[0] if isinstance(eos, (list, tuple)) else eos
        provider.vision_start_token_id = getattr(hf_config, "vision_start_token_id", 248053)
        provider.vision_end_token_id = getattr(hf_config, "vision_end_token_id", 248054)
        provider.image_token_id = getattr(hf_config, "image_token_id", 248056)
        provider.video_token_id = getattr(hf_config, "video_token_id", 248057)

        vision_config = hf_config.vision_config
        vision_config.torch_dtype = torch.bfloat16
        provider.vision_config = vision_config
        provider.hf_text_config = text_config

        provider.autocast_dtype = torch.bfloat16
        provider.hetereogenous_dist_checkpoint = True
        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Language-model mappings (VL prefixes) + the Qwen3-VL vision mappings."""
        hf_pretrained = self.hf_pretrained
        hf_config = hf_pretrained.config if hasattr(hf_pretrained, "config") else hf_pretrained
        text_config = get_qwen4_exp_text_config(hf_config)
        hf_prefix = get_qwen4_exp_hf_lm_prefix(hf_config)  # "model.language_model." for VL
        experts_packed = moe_experts_stored_packed(self.hf_pretrained, f"{hf_prefix}layers.", default=True)
        mapping_list = self.get_lm_mappings(
            hf_prefix,
            experts_packed,
            text_config,
            self._num_ple_shards(text_config),
            megatron_prefix="language_model.",
        )
        mapping_list.extend(_get_vision_mappings())
        return MegatronMappingRegistry(*mapping_list)


__all__ = [
    "Qwen4ExpVLBridge",
    "Qwen4ExpVLMoEModelProvider",
]
