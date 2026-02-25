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
Megatron Bridge for Qwen3.5 Vision-Language Model.

Qwen3.5 is a multimodal model that combines:
- A hybrid Gated DeltaNet + Gated Attention language model (like Qwen3-Next)
- A vision encoder (similar to Qwen3-VL)
- Mixture of Experts (MoE) with shared experts

Reference: https://huggingface.co/Qwen/Qwen3.5-397B-A17B
"""

import logging
import os

import torch

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ConcatenatedQKVMapping,
    GatedMLPMapping,
    GDNConv1dMapping,
    GDNLinearMappingSeparate,
    QKVMapping,
    ReplicatedMapping,
    RMSNorm2ZeroCenteredRMSNormMapping,
)
from megatron.bridge.models.hf_pretrained.vlm import PreTrainedVLM
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model import Qwen3VLModel
from megatron.bridge.models.qwen_vl.qwen3_vl_bridge import (
    ExpertMLPDownProjMapping,
    ExpertMLPGateUpProjMapping,
    Qwen3VLMoEBridge,
)
from megatron.bridge.models.qwen_vl.qwen35_vl_provider import Qwen35VLModelProvider


logger = logging.getLogger(__name__)

_QWEN3_5_HF_CLASS_NAME = "Qwen3_5MoeForConditionalGeneration"


@MegatronModelBridge.register_bridge(
    source=_QWEN3_5_HF_CLASS_NAME,
    target=Qwen3VLModel,
    provider=Qwen35VLModelProvider,
    model_type="qwen3_5_moe",
)
class Qwen35VLBridge(Qwen3VLMoEBridge):
    """
    Megatron Bridge for Qwen3.5 Vision-Language Model.

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

    def provider_bridge(self, hf_pretrained: PreTrainedVLM) -> Qwen35VLModelProvider:
        """
        Create a Qwen35VLModelProvider from a HuggingFace pretrained model.

        Extracts both language model and vision model configurations from the
        HuggingFace config and maps them to Megatron provider parameters.

        Args:
            hf_pretrained: HuggingFace pretrained VLM model

        Returns:
            Qwen35VLModelProvider configured with the HF model's parameters
        """
        hf_config = hf_pretrained.config
        text_config = hf_config.text_config

        # Use base class utility to extract common config fields
        provider_kwargs = self.hf_config_to_provider_kwargs(text_config)

        vision_config = hf_config.vision_config
        vision_config.torch_dtype = provider_kwargs.get("params_dtype", torch.float32)

        provider = Qwen35VLModelProvider(**provider_kwargs)

        # --- Common Qwen3 LLM settings ---
        provider.normalization = "RMSNorm"
        provider.gated_linear_unit = True
        provider.add_qkv_bias = getattr(text_config, "attention_bias", False)
        provider.add_bias_linear = False
        provider.qk_layernorm = True
        provider.hidden_dropout = 0.0

        # --- Qwen3-Next hybrid architecture settings ---
        provider.layernorm_zero_centered_gamma = True
        provider.attention_output_gate = True
        provider.experimental_attention_variant = "gated_delta_net"
        # full_attention_interval defines how often standard attention appears:
        # e.g., 4 means every 4th layer is standard attention (3 GDN + 1 Attn)
        provider.linear_attention_freq = getattr(text_config, "full_attention_interval", 4)
        provider.rotary_percent = getattr(text_config, "rope_parameters", {}).get("partial_rotary_factor", 0.25)

        # --- MoE specific parameters ---
        provider.moe_ffn_hidden_size = getattr(text_config, "moe_intermediate_size", 1024)
        provider.num_moe_experts = getattr(text_config, "num_experts", 512)
        provider.moe_router_topk = getattr(text_config, "num_experts_per_tok", 10)
        provider.moe_shared_expert_intermediate_size = getattr(text_config, "shared_expert_intermediate_size", None)
        provider.moe_shared_expert_gate = True
        provider.moe_grouped_gemm = True
        provider.moe_router_load_balancing_type = "global_aux_loss"
        provider.moe_router_pre_softmax = False
        provider.moe_token_dispatcher_type = "alltoall"
        provider.moe_permute_fusion = True

        # --- GDN (Gated DeltaNet) specific parameters ---
        provider.linear_conv_kernel_dim = getattr(text_config, "linear_conv_kernel_dim", 4)
        provider.linear_key_head_dim = getattr(text_config, "linear_key_head_dim", 128)
        provider.linear_value_head_dim = getattr(text_config, "linear_value_head_dim", 128)
        provider.linear_num_key_heads = getattr(text_config, "linear_num_key_heads", 16)
        provider.linear_num_value_heads = getattr(text_config, "linear_num_value_heads", 64)

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

        # --- DEBUG: tiny model for quick testing ---
        # Set QWEN35_DEBUG=1 to shrink the text model to 4 layers (1 GDN-GDN-GDN-Attn group)
        # with fewer experts. Useful for conversion / forward-pass smoke tests.
        if os.environ.get("QWEN35_DEBUG", "0") == "1":
            logger.warning("QWEN35_DEBUG=1: overriding to tiny 4-layer model for debugging")
            provider.num_layers = 4  # 3 GDN + 1 Attn (one full group)
            # provider.num_moe_experts = 8  # 512 → 8
            # provider.moe_router_topk = 2  # 10 → 2
            # provider.moe_grouped_gemm = False

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

        # =====================================================================
        # Simple 1:1 parameter mappings
        # =====================================================================
        param_mappings = {
            # =================================================================
            # Language Model: Embeddings and output
            # =================================================================
            "language_model.embedding.word_embeddings.weight": "model.language_model.embed_tokens.weight",
            "language_model.output_layer.weight": "lm_head.weight",
            "language_model.decoder.final_layernorm.weight": "model.language_model.norm.weight",
            # =================================================================
            # Language Model: MoE router
            # =================================================================
            "language_model.decoder.layers.*.mlp.router.weight": "model.language_model.layers.*.mlp.gate.weight",
            "language_model.decoder.layers.*.pre_mlp_layernorm.weight": "model.language_model.layers.*.post_attention_layernorm.weight",
            # =================================================================
            # Language Model: Standard attention layers (Gated Attention)
            # These mappings apply to layers where standard attention is used
            # (every 4th layer in the 15 × (3 GDN + 1 Attn) pattern)
            # =================================================================
            "language_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.language_model.layers.*.input_layernorm.weight",
            "language_model.decoder.layers.*.self_attention.q_layernorm.weight": "model.language_model.layers.*.self_attn.q_norm.weight",
            "language_model.decoder.layers.*.self_attention.k_layernorm.weight": "model.language_model.layers.*.self_attn.k_norm.weight",
            "language_model.decoder.layers.*.self_attention.linear_proj.weight": "model.language_model.layers.*.self_attn.o_proj.weight",
            # =================================================================
            # Language Model: Linear attention (Gated DeltaNet) layers
            # These mappings apply to layers where GDN is used
            # (3 out of every 4 layers)
            # =================================================================
            "language_model.decoder.layers.*.self_attention.in_proj.layer_norm_weight": "model.language_model.layers.*.input_layernorm.weight",
            "language_model.decoder.layers.*.self_attention.out_proj.weight": "model.language_model.layers.*.linear_attn.out_proj.weight",
            "language_model.decoder.layers.*.self_attention.A_log": "model.language_model.layers.*.linear_attn.A_log",
            "language_model.decoder.layers.*.self_attention.dt_bias": "model.language_model.layers.*.linear_attn.dt_bias",
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

        # Register module types for GDN and shared expert (needed for AutoMapping detection)
        AutoMapping.register_module_type("SharedExpertMLP", "column")
        AutoMapping.register_module_type("GatedDeltaNet", "column")

        # =====================================================================
        # Special mappings requiring parameter transformation
        # =====================================================================
        mapping_list.extend(
            [
                # =============================================================
                # Language Model: Standard Attention QKV
                # Combines separate Q, K, V matrices into single QKV matrix
                # =============================================================
                QKVMapping(
                    megatron_param="language_model.decoder.layers.*.self_attention.linear_qkv.weight",
                    q="model.language_model.layers.*.self_attn.q_proj.weight",
                    k="model.language_model.layers.*.self_attn.k_proj.weight",
                    v="model.language_model.layers.*.self_attn.v_proj.weight",
                ),
                # =============================================================
                # Language Model: GDN (Gated DeltaNet) specific mappings
                # =============================================================
                # GDN Conv1d: depthwise causal convolution
                GDNConv1dMapping(
                    megatron_param="language_model.decoder.layers.*.self_attention.conv1d.weight",
                    hf_param="model.language_model.layers.*.linear_attn.conv1d.weight",
                ),
                # GDN Input Projection: Qwen3.5 stores 4 separate weight tensors
                # (in_proj_qkv, in_proj_z, in_proj_b, in_proj_a) instead of the
                # 2 fused tensors (in_proj_qkvz, in_proj_ba) used by Qwen3-Next.
                GDNLinearMappingSeparate(
                    megatron_param="language_model.decoder.layers.*.self_attention.in_proj.weight",
                    qkv="model.language_model.layers.*.linear_attn.in_proj_qkv.weight",
                    z="model.language_model.layers.*.linear_attn.in_proj_z.weight",
                    b="model.language_model.layers.*.linear_attn.in_proj_b.weight",
                    a="model.language_model.layers.*.linear_attn.in_proj_a.weight",
                ),
                # GDN Output Norm: zero-centered RMSNorm conversion
                # Qwen3-Next uses standard RMSNorm initialized to ones for output norm,
                # while Megatron uses zero-centered RMSNorm, so we subtract 1 during conversion.
                RMSNorm2ZeroCenteredRMSNormMapping(
                    "language_model.decoder.layers.*.self_attention.out_norm.weight",
                    "model.language_model.layers.*.linear_attn.norm.weight",
                ),
                # =============================================================
                # Language Model: MoE Expert MLPs (routed experts)
                # Uses GatedMLPMapping for gate+up projection fusion
                # =============================================================
                ExpertMLPGateUpProjMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.experts.linear_fc1.weight*",
                    hf_param="model.language_model.layers.*.mlp.experts.gate_up_proj",
                    transpose=False,
                ),
                ExpertMLPDownProjMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.experts.linear_fc2.weight*",
                    hf_param="model.language_model.layers.*.mlp.experts.down_proj",
                    transpose=False,
                ),
                # =============================================================
                # Language Model: Shared Expert MLPs
                # =============================================================
                GatedMLPMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.shared_experts.linear_fc1.weight",
                    gate="model.language_model.layers.*.mlp.shared_expert.gate_proj.weight",
                    up="model.language_model.layers.*.mlp.shared_expert.up_proj.weight",
                ),
                AutoMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.shared_experts.linear_fc2.weight",
                    hf_param="model.language_model.layers.*.mlp.shared_expert.down_proj.weight",
                ),
                # Shared expert gate weight (replicated across TP ranks)
                ReplicatedMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.shared_experts.gate_weight",
                    hf_param="model.language_model.layers.*.mlp.shared_expert_gate.weight",
                ),
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

        # TODO: MTP (Multi-Token Prediction) mappings for VL context.
        # "language_model.mtp.layers.0.eh_proj.weight": "mtp.fc.weight",
        # "language_model.mtp.layers.0.enorm.weight": "mtp.pre_fc_norm_embedding.weight",
        # "language_model.mtp.layers.0.hnorm.weight": "mtp.pre_fc_norm_hidden.weight",
        # "language_model.mtp.layers.0.final_layernorm.weight": "mtp.norm.weight",
        # "language_model.mtp.layers.0.transformer_layer.mlp.router.weight": "mtp.layers.0.mlp.gate.weight",
        # "language_model.mtp.layers.0.transformer_layer.pre_mlp_layernorm.weight": "mtp.layers.0.post_attention_layernorm.weight",
        # "language_model.mtp.layers.0.transformer_layer.self_attention.linear_qkv.layer_norm_weight": "mtp.layers.0.input_layernorm.weight",
        # "language_model.mtp.layers.0.transformer_layer.self_attention.q_layernorm.weight": "mtp.layers.0.self_attn.q_norm.weight",
        # "language_model.mtp.layers.0.transformer_layer.self_attention.k_layernorm.weight": "mtp.layers.0.self_attn.k_norm.weight",
        # "language_model.mtp.layers.0.transformer_layer.self_attention.linear_proj.weight": "mtp.layers.0.self_attn.o_proj.weight",
        #
        # Plus QKV, expert MLP, shared expert mappings for MTP layers.
        # The exact prefix structure (language_model.mtp.* vs mtp.*) needs verification.

        return MegatronMappingRegistry(*mapping_list)
