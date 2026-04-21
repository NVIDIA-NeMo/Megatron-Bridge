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

from typing import Dict, Optional

import torch
from torch import nn

from megatron.bridge.models import ColumnParallelMapping, RowParallelMapping
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ConcatenatedQKVMapping,
    MambaConv1dMapping,
    MambaInProjMapping,
    QKVMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.hf_pretrained.vlm import PreTrainedVLM
from megatron.bridge.models.nemotron_vl.modeling_nemotron_vl import NemotronVLModel
from megatron.bridge.models.nemotron_vl.nemotron_vl_provider import (
    NemotronNano3Bv3VLModelProvider,
    NemotronNano12Bv2VLModelProvider,
    NemotronVLModelProvider,
)


class ExpertBiasMapping(ReplicatedMapping):
    """Special mapping for expert_bias that preserves fp32 dtype.

    The expert_bias parameter in MoE routers must remain in fp32 for numerical
    stability during routing decisions. This mapping overrides the default dtype
    conversion to ensure fp32 is preserved.
    """

    def megatron_to_hf(
        self,
        megatron_weights: Optional[torch.Tensor],
        megatron_module: Optional[nn.Module],
    ) -> Dict[str, torch.Tensor]:
        """Export expert_bias ensuring it stays in fp32."""
        # Broadcast from owning PP rank
        megatron_weights = self.broadcast_from_pp_rank(megatron_weights, cache_key=str(self.megatron_param))

        if self.tp_rank != 0:
            return {}

        # Ensure fp32 on export
        if megatron_weights is not None and megatron_weights.dtype != torch.float32:
            megatron_weights = megatron_weights.to(torch.float32)

        return {str(self.hf_param): megatron_weights}


@MegatronModelBridge.register_bridge(source="NemotronH_Nano_VL_V2", target=NemotronVLModel)
class NemotronVLBridge(MegatronModelBridge):
    """Conversion utilities between HF Nemotron-VL and Megatron-Core format."""

    # ------------------------------------------------------------------
    # Provider translation
    # ------------------------------------------------------------------

    def provider_bridge(self, hf_pretrained: PreTrainedVLM) -> NemotronVLModelProvider:
        """Create a Nemotron VL provider from HuggingFace pretrained model.

        Automatically detects the model variant (12B v2 or 3B v3) based on
        the model architecture and instantiates the appropriate provider.
        """
        hf_config = hf_pretrained.config
        llm_config = hf_config.llm_config

        # Common provider parameters from HF config
        provider_kwargs = {
            "num_layers": llm_config.num_hidden_layers,
            "hidden_size": llm_config.hidden_size,
            "ffn_hidden_size": llm_config.intermediate_size,
            "num_attention_heads": llm_config.num_attention_heads,
            "init_method_std": llm_config.initializer_range,
            "layernorm_epsilon": getattr(llm_config, "layer_norm_epsilon", 1e-5),
            "make_vocab_size_divisible_by": self.make_vocab_size_divisible_by(llm_config.vocab_size),
            "share_embeddings_and_output_weights": getattr(llm_config, "tie_word_embeddings", False),
            "vocab_size": llm_config.vocab_size,
            "seq_length": llm_config.max_position_embeddings,
            "fp16": (self.dtype_from_hf(llm_config, default=torch.float32) == torch.float16),
            "bf16": (self.dtype_from_hf(llm_config, default=torch.float32) == torch.bfloat16),
            "params_dtype": self.dtype_from_hf(llm_config, default=torch.float32),
        }
        if hasattr(llm_config, "hybrid_override_pattern"):
            provider_kwargs["hybrid_override_pattern"] = llm_config.hybrid_override_pattern
        if hasattr(llm_config, "n_routed_experts"):
            provider_kwargs["num_moe_experts"] = llm_config.n_routed_experts
        if hasattr(hf_config, "projector_hidden_size"):
            provider_kwargs["vision_proj_ffn_hidden_size"] = hf_config.projector_hidden_size
        # Vision config: temporal compression settings
        vision_cfg = getattr(hf_config, "vision_config", None)
        vision_cfg_dict = vision_cfg.__dict__ if vision_cfg is not None else {}
        provider_kwargs["video_temporal_patch_size"] = getattr(
            hf_config, "video_temporal_patch_size",
            vision_cfg_dict.get("video_temporal_patch_size", 1),
        )
        provider_kwargs["separate_video_embedder"] = getattr(
            hf_config, "separate_video_embedder",
            vision_cfg_dict.get("separate_video_embedder", False),
        )
        # Detect model variant based on architecture
        # 3B v3 uses MoE architecture (nemotron6-moe), 12B v2 uses hybrid architecture (nemotron5-hybrid)
        is_moe_model = getattr(llm_config, "n_routed_experts", None) is not None
        if is_moe_model:
            # Nemotron Nano Next 3B v3 with MoE
            provider = NemotronNano3Bv3VLModelProvider(**provider_kwargs)
        else:
            # Nemotron Nano 12B v2 with hybrid architecture
            provider = NemotronNano12Bv2VLModelProvider(**provider_kwargs)

        return provider

    # ------------------------------------------------------------------
    # Parameter mapping
    # ------------------------------------------------------------------

    def mapping_registry(self) -> MegatronMappingRegistry:  # noqa: D401
        param_mappings = {
            # vision model
            "llava_model.vision_model.class_token": "vision_model.radio_model.model.patch_generator.cls_token.token",
            "llava_model.vision_model.position_embeddings": "vision_model.radio_model.model.patch_generator.pos_embed",
            "llava_model.vision_model.embedder.weight": "vision_model.radio_model.model.patch_generator.embedder.weight",
            "llava_model.vision_model.video_embedder.weight": "vision_model.radio_model.model.patch_generator.video_embedder.weight",
            # vision decoder
            "llava_model.vision_model.decoder.layers.*.self_attention.linear_proj.weight": "vision_model.radio_model.model.blocks.*.attn.proj.weight",
            "llava_model.vision_model.decoder.layers.*.self_attention.linear_proj.bias": "vision_model.radio_model.model.blocks.*.attn.proj.bias",
            "llava_model.vision_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "vision_model.radio_model.model.blocks.*.norm1.weight",
            "llava_model.vision_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_bias": "vision_model.radio_model.model.blocks.*.norm1.bias",
            "llava_model.vision_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "vision_model.radio_model.model.blocks.*.norm2.weight",
            "llava_model.vision_model.decoder.layers.*.mlp.linear_fc1.layer_norm_bias": "vision_model.radio_model.model.blocks.*.norm2.bias",
            "llava_model.vision_model.decoder.layers.*.mlp.linear_fc1.weight": "vision_model.radio_model.model.blocks.*.mlp.fc1.weight",
            "llava_model.vision_model.decoder.layers.*.mlp.linear_fc1.bias": "vision_model.radio_model.model.blocks.*.mlp.fc1.bias",
            "llava_model.vision_model.decoder.layers.*.mlp.linear_fc2.weight": "vision_model.radio_model.model.blocks.*.mlp.fc2.weight",
            "llava_model.vision_model.decoder.layers.*.mlp.linear_fc2.bias": "vision_model.radio_model.model.blocks.*.mlp.fc2.bias",
            # vision projection
            "llava_model.vision_projection.encoder.linear_fc1.layer_norm_weight": "mlp1.0.weight",
            "llava_model.vision_projection.encoder.linear_fc1.weight": "mlp1.1.weight",
            "llava_model.vision_projection.encoder.linear_fc2.weight": "mlp1.3.weight",
            # language model
            "llava_model.language_model.embedding.word_embeddings.weight": "language_model.backbone.embeddings.weight",
            "llava_model.language_model.decoder.final_norm.weight": "language_model.backbone.norm_f.weight",
            "llava_model.language_model.output_layer.weight": "language_model.lm_head.weight",
            # language decoder: mamba
            "llava_model.language_model.decoder.layers.*.mixer.in_proj.layer_norm_weight": "language_model.backbone.layers.*.norm.weight",
            # language decoder: mlp
            "llava_model.language_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "language_model.backbone.layers.*.norm.weight",
            "llava_model.language_model.decoder.layers.*.mlp.linear_fc1.weight": "language_model.backbone.layers.*.mixer.up_proj.weight",
            "llava_model.language_model.decoder.layers.*.mlp.linear_fc2.weight": "language_model.backbone.layers.*.mixer.down_proj.weight",
            # language decoder: attention
            "llava_model.language_model.decoder.layers.*.self_attention.linear_proj.weight": "language_model.backbone.layers.*.mixer.o_proj.weight",
            "llava_model.language_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "language_model.backbone.layers.*.norm.weight",
            # language decoder: moe
            "llava_model.language_model.decoder.layers.*.mlp.router.weight": "language_model.backbone.layers.*.mixer.gate.weight",
            "llava_model.language_model.decoder.layers.*.mlp.experts.linear_fc1.weight*": "language_model.backbone.layers.*.mixer.experts.*.up_proj.weight",
            "llava_model.language_model.decoder.layers.*.mlp.experts.linear_fc2.weight*": "language_model.backbone.layers.*.mixer.experts.*.down_proj.weight",
            "llava_model.language_model.decoder.layers.*.mlp.shared_experts.linear_fc1.weight": "language_model.backbone.layers.*.mixer.shared_experts.up_proj.weight",
            "llava_model.language_model.decoder.layers.*.mlp.shared_experts.linear_fc2.weight": "language_model.backbone.layers.*.mixer.shared_experts.down_proj.weight",
            "llava_model.language_model.decoder.layers.*.pre_mlp_layernorm.weight": "language_model.backbone.layers.*.norm.weight",
        }

        mapping_list = []
        # Convert each dictionary entry to AutoMapping(hf_param, megatron_param)
        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        mapping_list.append(
            ExpertBiasMapping(
                megatron_param="llava_model.language_model.decoder.layers.*.mlp.router.expert_bias",
                hf_param="language_model.backbone.layers.*.mixer.gate.e_score_correction_bias",
            ),
        )
        for mixer_sub_module in ["A_log", "D", "dt_bias", "norm.weight"]:
            mapping_list.append(
                ColumnParallelMapping(
                    megatron_param=rf"llava_model.language_model.decoder.layers.*.mixer.{mixer_sub_module}",
                    hf_param=rf"language_model.backbone.layers.*.mixer.{mixer_sub_module}",
                ),
            )
        mapping_list.append(
            RowParallelMapping(
                megatron_param="llava_model.language_model.decoder.layers.*.mixer.out_proj.weight",
                hf_param="language_model.backbone.layers.*.mixer.out_proj.weight",
            ),
        )
        mapping_list.append(
            MambaInProjMapping(
                megatron_param="llava_model.language_model.decoder.layers.*.mixer.in_proj.weight",
                hf_param="language_model.backbone.layers.*.mixer.in_proj.weight",
            ),
        )
        for conv1d_sub_module in ["weight", "bias"]:
            mapping_list.append(
                MambaConv1dMapping(
                    megatron_param=rf"llava_model.language_model.decoder.layers.*.mixer.conv1d.{conv1d_sub_module}",
                    hf_param=rf"language_model.backbone.layers.*.mixer.conv1d.{conv1d_sub_module}",
                ),
            )

        # Add special mappings that require parameter concatenation/transformation
        mapping_list.extend(
            [
                # QKV: Combine separate Q, K, V matrices into single QKV matrix
                QKVMapping(
                    megatron_param="llava_model.language_model.decoder.layers.*.self_attention.linear_qkv.weight",
                    q="language_model.backbone.layers.*.mixer.q_proj.weight",
                    k="language_model.backbone.layers.*.mixer.k_proj.weight",
                    v="language_model.backbone.layers.*.mixer.v_proj.weight",
                ),
                ConcatenatedQKVMapping(
                    megatron_param="llava_model.vision_model.decoder.layers.*.self_attention.linear_qkv.weight",
                    hf_param="vision_model.radio_model.model.blocks.*.attn.qkv.weight",
                ),
                ConcatenatedQKVMapping(
                    megatron_param="llava_model.vision_model.decoder.layers.*.self_attention.linear_qkv.bias",
                    hf_param="vision_model.radio_model.model.blocks.*.attn.qkv.bias",
                ),
            ]
        )
        AutoMapping.register_module_type("RADIOViTModel", "replicated")
        return MegatronMappingRegistry(*mapping_list)
