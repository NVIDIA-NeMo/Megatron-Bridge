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

import torch

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import AutoMapping
from megatron.bridge.models.hf_pretrained.vlm import PreTrainedVLM

from .modeling_nemotron_vl import NemotronVLModel
from .nemotron_vl_provider import NemotronVLModelProvider


@MegatronModelBridge.register_bridge(source="NemotronH_Nano_VL", target=NemotronVLModel)
class NemotronVLBridge(MegatronModelBridge):
    """Conversion utilities between HF Nemotron-VL and Megatron-Core format."""

    # ------------------------------------------------------------------
    # Provider translation
    # ------------------------------------------------------------------

    def provider_bridge(self, hf_pretrained: PreTrainedVLM) -> NemotronVLModelProvider:  # type: ignore[override]
        hf_config = hf_pretrained.config

        # vision_config = GPTModelProvider(
            
        # )
        provider = NemotronVLModelProvider(
            num_layers=hf_config.num_hidden_layers,
            hidden_size=hf_config.hidden_size,
            ffn_hidden_size=hf_config.intermediate_size,
            num_attention_heads=hf_config.num_attention_heads,
            num_query_groups=getattr(hf_config, "num_key_value_heads", hf_config.num_attention_heads // 2),
            init_method_std=hf_config.initializer_range,
            layernorm_epsilon=getattr(hf_config, "layer_norm_epsilon", 1e-5),
            make_vocab_size_divisible_by=self.make_vocab_size_divisible_by(hf_config.vocab_size),
            share_embeddings_and_output_weights=getattr(hf_config, "tie_word_embeddings", False),
            vocab_size=hf_config.vocab_size,
            seq_length=hf_config.max_position_embeddings,
            fp16=(self.dtype_from_hf(hf_config, default=torch.float32) == torch.float16),
            bf16=(self.dtype_from_hf(hf_config, default=torch.float32) == torch.bfloat16),
            params_dtype=self.dtype_from_hf(hf_config, default=torch.float32),
            generation_config=hf_pretrained.generation_config,
            # vision_config=getattr(hf_config, "vision_config", NemotronVLModelProvider.vision_config),
        )
        return provider

    # ------------------------------------------------------------------
    # Parameter mapping
    # ------------------------------------------------------------------

    def mapping_registry(self) -> MegatronMappingRegistry:  # noqa: D401
        param_mappings = {
            # vision model
            "vision_model.class_token": "vision_model.radio_model.model.patch_generator.cls_token.token",
            "vision_model.position_embeddings": "vision_model.radio_model.model.patch_generator.pos_embed",
            "vision_model.embedder.weight": "vision_model.radio_model.model.patch_generator.embedder.weight",
            # vision decoder
            "vision_model.decoder.layers.*.self_attention.linear_proj.weight": "vision_model.radio_model.model.blocks.*.attn.proj.weight",
            "vision_model.decoder.layers.*.self_attention.linear_proj.bias": "vision_model.radio_model.model.blocks.*.attn.proj.bias",
            "vision_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "vision_model.radio_model.model.blocks.*.norm1.weight",
            "vision_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_bias": "vision_model.radio_model.model.blocks.*.norm1.bias",
            "vision_model.decoder.layers.*.self_attention.linear_qkv.weight": "vision_model.radio_model.model.blocks.*.attn.qkv.weight",
            "vision_model.decoder.layers.*.self_attention.linear_qkv.bias": "vision_model.radio_model.model.blocks.*.attn.qkv.bias",
            "vision_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "vision_model.radio_model.model.blocks.*.norm2.weight",
            "vision_model.decoder.layers.*.mlp.linear_fc1.layer_norm_bias": "vision_model.radio_model.model.blocks.*.norm2.bias",
            "vision_model.decoder.layers.*.mlp.linear_fc1.weight": "vision_model.radio_model.model.blocks.*.mlp.fc1.weight",
            "vision_model.decoder.layers.*.mlp.linear_fc1.bias": "vision_model.radio_model.model.blocks.*.mlp.fc1.bias",
            "vision_model.decoder.layers.*.mlp.linear_fc2.weight": "vision_model.radio_model.model.blocks.*.mlp.fc2.weight",
            "vision_model.decoder.layers.*.mlp.linear_fc2.bias": "vision_model.radio_model.model.blocks.*.mlp.fc2.bias",
            # vision projection
            "vision_projection.encoder.linear_fc1.layer_norm_weight": "mlp1.0.weight",
            "vision_projection.encoder.linear_fc1.weight": "mlp1.1.weight",
            "vision_projection.encoder.linear_fc2.weight": "mlp1.3.weight",
            # language model
            "language_model.embedding.word_embeddings.weight": "language_model.backbone.embeddings.weight",
            "language_model.decoder.final_norm.weight": "language_model.backbone.norm_f.weight",
            "language_model.output_layer.weight": "language_model.lm_head.weight",
            # language decoder: mamba
            "language_model.decoder.layers.*.mixer.dt_bias": "language_model.backbone.layers.*.mixer.dt_bias",
            "language_model.decoder.layers.*.mixer.A_log": "language_model.backbone.layers.*.mixer.A_log",
            "language_model.decoder.layers.*.mixer.D": "language_model.backbone.layers.*.mixer.D",
            "language_model.decoder.layers.*.mixer.in_proj.layer_norm_weight": "language_model.backbone.layers.*.norm.weight",
            "language_model.decoder.layers.*.mixer.in_proj.weight": "language_model.backbone.layers.*.mixer.in_proj.weight",
            "language_model.decoder.layers.*.mixer.conv1d.weight": "language_model.backbone.layers.*.mixer.conv1d.weight",
            "language_model.decoder.layers.*.mixer.conv1d.bias": "language_model.backbone.layers.*.mixer.conv1d.bias",
            "language_model.decoder.layers.*.mixer.norm.weight": "language_model.backbone.layers.*.mixer.norm.weight",
            "language_model.decoder.layers.*.mixer.out_proj.weight": "language_model.backbone.layers.*.mixer.out_proj.weight",
            # language decoder: mlp
            "language_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "language_model.backbone.layers.*.norm.weight",
            "language_model.decoder.layers.*.mlp.linear_fc1.weight": "language_model.backbone.layers.*.mixer.up_proj.weight",
            "language_model.decoder.layers.*.mlp.linear_fc2.weight": "language_model.backbone.layers.*.mixer.down_proj.weight",
            # language decoder: attention
            "language_model.decoder.layers.*.self_attention.linear_proj.weight": "language_model.backbone.layers.*.mixer.o_proj.weight",
            "language_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "language_model.backbone.layers.*.norm.weight",
        }

        mapping_list = []
        # Convert each dictionary entry to AutoMapping(hf_param, megatron_param)
        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        # Add special mappings that require parameter concatenation/transformation
        mapping_list.extend(
            [
                # QKV: Combine separate Q, K, V matrices into single QKV matrix
                QKVMapping(
                    megatron_param="language_model.decoder.layers.*.self_attention.linear_qkv.weight",
                    q="language_model.backbone.layers.7.mixer.q_proj.weight",
                    k="language_model.backbone.layers.*.mixer.k_proj.weight",
                    v="language_model.backbone.layers.*.mixer.v_proj.weight",
                ),
            ]
        )

        return MegatronMappingRegistry(*mapping_list)

