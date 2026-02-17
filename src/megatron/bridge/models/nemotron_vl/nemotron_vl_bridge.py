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

from megatron.core.activations import squared_relu

from megatron.bridge.models import ColumnParallelMapping, RowParallelMapping
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ConcatenatedQKVMapping,
    MambaConv1dMapping,
    MambaInProjMapping,
    QKVMapping,
)
from megatron.bridge.models.hf_pretrained.vlm import PreTrainedVLM
from megatron.bridge.models.nemotron_vl.modeling_nemotron_vl import NemotronVLModel
from megatron.bridge.models.nemotron_vl.nemotron_vl_provider import NemotronNano12Bv2VLModelProvider


@MegatronModelBridge.register_bridge(
    source="NemotronH_Nano_VL_V2",
    target=NemotronVLModel,
    provider=NemotronNano12Bv2VLModelProvider,
    model_type="nemotron_vl",
)
class NemotronVLBridge(MegatronModelBridge):
    """Conversion utilities between HF Nemotron-VL and Megatron-Core format."""

    # ------------------------------------------------------------------
    # Provider translation
    # ------------------------------------------------------------------

    def provider_bridge(self, hf_pretrained: PreTrainedVLM) -> NemotronNano12Bv2VLModelProvider:  # type: ignore[override]
        hf_config = hf_pretrained.config
        llm_config = hf_config.llm_config

        # Use base class helper for common config mapping
        provider_kwargs = self.hf_config_to_provider_kwargs(llm_config)

        # Handle vocab size divisibility
        provider_kwargs["make_vocab_size_divisible_by"] = self.make_vocab_size_divisible_by(llm_config.vocab_size)

        provider = NemotronNano12Bv2VLModelProvider(**provider_kwargs)

        # Nemotron VL-specific settings
        # Note: Most defaults come from the provider class hierarchy (NemotronNano12Bv2Provider)
        provider.scatter_embedding_sequence_parallel = False
        provider.attention_softmax_in_fp32 = True

        # Override fields that should use NemotronH provider's specialized defaults
        # instead of HF config values
        provider.activation_func = squared_relu  # Nemotron uses squared_relu, not HF's hidden_act
        provider.autocast_dtype = None  # Not set in original code

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
        }

        mapping_list = []
        # Convert each dictionary entry to AutoMapping(hf_param, megatron_param)
        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        for mixer_sub_module in ["A_log", "D", "dt_bias", "norm.weight"]:
            mapping_list.extend(
                [
                    ColumnParallelMapping(
                        megatron_param=rf"llava_model.language_model.decoder.layers.*.mixer.{mixer_sub_module}",
                        hf_param=rf"language_model.backbone.layers.*.mixer.{mixer_sub_module}",
                    ),
                ]
            )
        mapping_list.extend(
            [
                RowParallelMapping(
                    megatron_param="llava_model.language_model.decoder.layers.*.mixer.out_proj.weight",
                    hf_param="language_model.backbone.layers.*.mixer.out_proj.weight",
                ),
            ]
        )
        mapping_list.extend(
            [
                MambaInProjMapping(
                    megatron_param="llava_model.language_model.decoder.layers.*.mixer.in_proj.weight",
                    hf_param="language_model.backbone.layers.*.mixer.in_proj.weight",
                ),
            ]
        )
        for conv1d_sub_module in ["weight", "bias"]:
            mapping_list.extend(
                [
                    MambaConv1dMapping(
                        megatron_param=rf"llava_model.language_model.decoder.layers.*.mixer.conv1d.{conv1d_sub_module}",
                        hf_param=rf"language_model.backbone.layers.*.mixer.conv1d.{conv1d_sub_module}",
                    ),
                ]
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
