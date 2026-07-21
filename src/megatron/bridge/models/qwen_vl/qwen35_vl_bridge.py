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

"""
Megatron Bridges for Qwen3.5 Vision-Language Models.

Qwen3.5 is a family of multimodal models that combine:
- A hybrid Gated DeltaNet + Gated Attention language model (like Qwen3-Next)
- A vision encoder (similar to Qwen3-VL)
- Dense MLP or Mixture of Experts (MoE) with shared experts

This module provides two bridges:

- ``Qwen35VLBridge``: Dense variant (e.g., Qwen3.5-27B)
  Reference: https://huggingface.co/Qwen/Qwen3.5-27B

- ``Qwen35VLMoEBridge``: MoE variant (e.g., Qwen3.5-397B-A17B)
  Reference: https://huggingface.co/Qwen/Qwen3.5-397B-A17B
"""

import logging

import torch

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ConcatenatedQKVMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.conversion.utils import moe_experts_stored_packed
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.qwen.qwen35_bridge import (
    Qwen35Bridge,
    Qwen35MoEBridge,
    _apply_qwen35_common_config,
    _apply_qwen35_moe_config,
)
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model import Qwen3VLModel
from megatron.bridge.models.qwen_vl.qwen35_vl_provider import (
    Qwen35VLModelProvider,
    Qwen35VLMoEModelProvider,
)


logger = logging.getLogger(__name__)

_QWEN3_5_DENSE_HF_CLASS_NAME = "Qwen3_5ForConditionalGeneration"
_QWEN3_5_MOE_HF_CLASS_NAME = "Qwen3_5MoeForConditionalGeneration"


def _get_vision_mappings():
    # =====================================================================
    # Simple 1:1 parameter mappings
    # =====================================================================
    param_mappings = {
        # =================================================================
        # Vision Model: Attention
        # =================================================================
        "vision_model.decoder.layers.*.self_attention.linear_proj.weight": "model.visual.blocks.*.attn.proj.weight",
        "vision_model.decoder.layers.*.self_attention.linear_proj.bias": "model.visual.blocks.*.attn.proj.bias",
        # =================================================================
        # Vision Model: MLP
        # =================================================================
        "vision_model.decoder.layers.*.mlp.linear_fc1.weight": "model.visual.blocks.*.mlp.linear_fc1.weight",
        "vision_model.decoder.layers.*.mlp.linear_fc1.bias": "model.visual.blocks.*.mlp.linear_fc1.bias",
        "vision_model.decoder.layers.*.mlp.linear_fc2.weight": "model.visual.blocks.*.mlp.linear_fc2.weight",
        "vision_model.decoder.layers.*.mlp.linear_fc2.bias": "model.visual.blocks.*.mlp.linear_fc2.bias",
        # =================================================================
        # Vision Model: Layer Norms
        # =================================================================
        "vision_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.visual.blocks.*.norm1.weight",
        "vision_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_bias": "model.visual.blocks.*.norm1.bias",
        "vision_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.visual.blocks.*.norm2.weight",
        "vision_model.decoder.layers.*.mlp.linear_fc1.layer_norm_bias": "model.visual.blocks.*.norm2.bias",
        # =================================================================
        # Vision Model: Final Merger
        # =================================================================
        "vision_model.merger.patch_norm.**": "model.visual.merger.norm.**",
        "vision_model.merger.linear_fc1.weight": "model.visual.merger.linear_fc1.weight",
        "vision_model.merger.linear_fc1.bias": "model.visual.merger.linear_fc1.bias",
        "vision_model.merger.linear_fc2.weight": "model.visual.merger.linear_fc2.weight",
        "vision_model.merger.linear_fc2.bias": "model.visual.merger.linear_fc2.bias",
    }

    mapping_list = []

    # Convert simple 1:1 mappings to AutoMapping objects
    for megatron_param, hf_param in param_mappings.items():
        mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

    # =====================================================================
    # Special mappings requiring parameter transformation
    # =====================================================================
    mapping_list.extend(
        [
            # =============================================================
            # Vision Model: QKV (concatenated format)
            # =============================================================
            ConcatenatedQKVMapping(
                megatron_param="vision_model.decoder.layers.*.self_attention.linear_qkv.weight",
                hf_param="model.visual.blocks.*.attn.qkv.weight",
            ),
            ConcatenatedQKVMapping(
                megatron_param="vision_model.decoder.layers.*.self_attention.linear_qkv.bias",
                hf_param="model.visual.blocks.*.attn.qkv.bias",
            ),
            # =============================================================
            # Vision Model: Patch embedding (replicated across TP ranks)
            # These are conv layers that must be replicated
            # =============================================================
            ReplicatedMapping(
                megatron_param="vision_model.patch_embed.proj.**",
                hf_param="model.visual.patch_embed.proj.**",
            ),
            ReplicatedMapping(
                megatron_param="vision_model.pos_embed.weight",
                hf_param="model.visual.pos_embed.weight",
            ),
        ]
    )

    return mapping_list


@MegatronModelBridge.register_bridge(
    source=_QWEN3_5_MOE_HF_CLASS_NAME,
    target=Qwen3VLModel,
    provider=Qwen35VLMoEModelProvider,
    model_type="qwen3_5_moe",
)
class Qwen35VLMoEBridge(MegatronModelBridge):
    """
    Megatron Bridge for Qwen3.5 Vision-Language Model (MoE variant).

    This bridge handles the conversion between HuggingFace Qwen3.5 VL model
    and Megatron-Core Qwen3VLModel formats, including weight mappings and
    configuration translation for the hybrid GDN+Attention VLM architecture.

    The weight mappings handle:
    - Language model hybrid layers (GDN + standard attention)
    - MoE layers with routed and shared experts
    - Vision model weights (same as Qwen3-VL: deepstack, merger, patch embed)
    - QK layernorm, zero-centered RMSNorm for GDN output norm
    - mRoPE position embeddings

    Architecture: 15 × (3 × (GDN → MoE) + 1 × (Attention → MoE)) = 60 layers

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("Qwen/Qwen3.5-397B-A17B")
        >>> provider = bridge.to_megatron_provider()
    """

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> Qwen35VLMoEModelProvider:
        """
        Create a Qwen35VLMoEModelProvider from a HuggingFace pretrained model.

        Extracts both language model and vision model configurations from the
        HuggingFace config and maps them to Megatron provider parameters.

        Args:
            hf_pretrained: HuggingFace pretrained VLM model

        Returns:
            Qwen35VLMoEModelProvider configured with the HF model's parameters
        """
        hf_config = hf_pretrained.config
        text_config = hf_config.text_config

        # Use base class utility to extract common config fields
        provider_kwargs = self.hf_config_to_provider_kwargs(text_config)

        vision_config = hf_config.vision_config
        vision_config.torch_dtype = provider_kwargs.get("params_dtype", torch.float32)

        provider = Qwen35VLMoEModelProvider(**provider_kwargs)

        # LM parameters
        _apply_qwen35_moe_config(provider, text_config)

        # For VLMs, tie_word_embeddings lives on the top-level config, not text_config.
        provider.share_embeddings_and_output_weights = getattr(hf_config, "tie_word_embeddings", False)

        # --- VL-specific overrides ---
        provider.position_embedding_type = "mrope"
        provider.vision_config = vision_config
        provider.hf_text_config = text_config
        provider.head_dim = getattr(text_config, "head_dim", 256)
        provider.bos_token_id = getattr(text_config, "bos_token_id", 248045)
        provider.eos_token_id = getattr(text_config, "eos_token_id", 248046)
        provider.vision_start_token_id = getattr(hf_config, "vision_start_token_id", 248053)
        provider.vision_end_token_id = getattr(hf_config, "vision_end_token_id", 248054)
        provider.image_token_id = getattr(hf_config, "image_token_id", 248056)
        provider.video_token_id = getattr(hf_config, "video_token_id", 248057)
        provider.audio_token_id = getattr(hf_config, "audio_token_id", 248076)

        # Qwen3.5 uses mRoPE with [11, 11, 10] sections (different from Qwen3-VL's [24, 20, 20])
        # The sections correspond to [temporal, height, width] dimensions.
        # With partial_rotary_factor=0.25 and head_dim=256, rotary_dim=64,
        # so each pair needs 32 dims total → sections [11, 11, 10].
        provider.mrope_section = getattr(text_config, "rope_scaling", {}).get("mrope_section", [11, 11, 10])

        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        """
        Return MegatronMappingRegistry containing parameter mappings for Qwen3.5 VL.

        Combines:
        1. Language model mappings (Qwen3-Next hybrid architecture with VL prefixes):
           - Standard attention: QKV, output projection, QK layernorm
           - Linear attention (GDN): in_proj, out_proj, conv1d, A_log, dt_bias, out_norm
           - MoE: router, routed expert MLPs, shared expert MLPs, shared expert gate
           - Embeddings, output layer, final layernorm

        2. Vision model mappings (Qwen3-VL style):
           - Vision transformer blocks: attention, MLP, layer norms
           - Deepstack visual mergers
           - Patch embedding and position embedding
           - Final merger (patch_norm, linear_fc1, linear_fc2)

        Naming Convention:
        - Megatron language model params are prefixed with "language_model."
        - HF language model params are prefixed with "model.language_model."
        - Megatron vision model params are prefixed with "vision_model."
        - HF vision model params are prefixed with "model.visual."

        Returns:
            MegatronMappingRegistry with all parameter mappings
        """
        # Detect MTP MoE expert weight format: Qwen3.5 stores per-expert
        # (mtp.layers.0.mlp.experts.{i}.gate_proj.weight), Qwen3.6 stores packed
        # (mtp.layers.0.mlp.experts.gate_up_proj). Same architecture string,
        # different storage — must inspect HF keys.
        mtp_experts_packed = False
        hf_pretrained = getattr(self, "hf_pretrained", None)
        if hasattr(hf_pretrained, "state") and hasattr(hf_pretrained.state, "source"):
            hf_keys = set(hf_pretrained.state.source.get_all_keys())
            if "mtp.layers.0.mlp.experts.gate_up_proj" in hf_keys:
                mtp_experts_packed = True
        experts_packed = moe_experts_stored_packed(hf_pretrained, "model.language_model.layers.")

        mapping_list = []
        mapping_list.extend(
            Qwen35MoEBridge._get_moe_lm_mappings(
                hf_prefix="model.language_model.", megatron_prefix="language_model.", experts_packed=experts_packed
            )
        )
        mapping_list.extend(
            Qwen35MoEBridge._get_moe_mtp_mappings(
                megatron_prefix="language_model.", mtp_experts_packed=mtp_experts_packed
            )
        )
        mapping_list.extend(_get_vision_mappings())
        return MegatronMappingRegistry(*mapping_list)


@MegatronModelBridge.register_bridge(
    source=_QWEN3_5_DENSE_HF_CLASS_NAME,
    target=Qwen3VLModel,
    provider=Qwen35VLModelProvider,
    model_type="qwen3_5",
)
class Qwen35VLBridge(MegatronModelBridge):
    """
    Megatron Bridge for Qwen3.5 Dense Vision-Language Model.

    This bridge handles the conversion between HuggingFace Qwen3.5 dense VL model
    and Megatron-Core Qwen3VLModel formats. Unlike the MoE variant, this model uses
    a standard dense MLP (gate_proj + up_proj → linear_fc1, down_proj → linear_fc2).

    The weight mappings handle:
    - Language model hybrid layers (GDN + standard attention)
    - Dense MLP with gated SiLU activation (fused pre-MLP layernorm)
    - Vision model weights (no deepstack mergers)
    - QK layernorm, zero-centered RMSNorm for GDN output norm
    - mRoPE position embeddings

    Architecture (27B): 16 × (3 × GDN + 1 × Attention) = 64 layers

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("Qwen/Qwen3.5-27B")
        >>> provider = bridge.to_megatron_provider()
    """

    mimo_source_prefixes = {"language": "language_model.", "images": "vision_model."}

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> Qwen35VLModelProvider:
        """Create a Qwen35VLModelProvider from a HuggingFace pretrained model."""
        hf_config = hf_pretrained.config
        text_config = hf_config.text_config

        provider_kwargs = self.hf_config_to_provider_kwargs(text_config)

        vision_config = hf_config.vision_config
        vision_config.torch_dtype = provider_kwargs.get("params_dtype", torch.float32)

        provider = Qwen35VLModelProvider(**provider_kwargs)

        # LM parameters
        _apply_qwen35_common_config(provider, text_config)

        # For VLMs, tie_word_embeddings lives on the top-level config, not text_config.
        # text_config inherits PretrainedConfig's default of True which is wrong for 9B/27B.
        provider.share_embeddings_and_output_weights = getattr(hf_config, "tie_word_embeddings", False)

        # --- VL-specific overrides ---
        provider.position_embedding_type = "mrope"
        provider.vision_config = vision_config
        provider.hf_text_config = text_config
        provider.head_dim = getattr(text_config, "head_dim", 256)
        provider.bos_token_id = getattr(text_config, "bos_token_id", 248045)
        provider.eos_token_id = getattr(text_config, "eos_token_id", 248044)
        provider.vision_start_token_id = getattr(hf_config, "vision_start_token_id", 248053)
        provider.vision_end_token_id = getattr(hf_config, "vision_end_token_id", 248054)
        provider.image_token_id = getattr(hf_config, "image_token_id", 248056)
        provider.video_token_id = getattr(hf_config, "video_token_id", 248057)
        provider.mrope_section = getattr(text_config, "rope_scaling", {}).get("mrope_section", [11, 11, 10])

        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        """
        Return MegatronMappingRegistry for Qwen3.5 dense VL model.

        Key differences from the MoE variant:
        - Dense MLP: gate_proj + up_proj fused into linear_fc1, down_proj as linear_fc2
        - Pre-MLP layernorm fused into mlp.linear_fc1 (not a separate pre_mlp_layernorm)
        - No MoE router, routed expert MLPs, or shared expert mappings
        - No deepstack visual mergers (deepstack_visual_indexes is empty)
        """
        mapping_list = []

        mapping_list.extend(
            Qwen35Bridge._get_dense_lm_mappings(hf_prefix="model.language_model.", megatron_prefix="language_model.")
        )
        mapping_list.extend(Qwen35Bridge._get_dense_mtp_mappings(megatron_prefix="language_model."))
        mapping_list.extend(_get_vision_mappings())
        return MegatronMappingRegistry(*mapping_list)
