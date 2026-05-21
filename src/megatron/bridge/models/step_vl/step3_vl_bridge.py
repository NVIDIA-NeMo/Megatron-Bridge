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

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.hf_pretrained.vlm import PreTrainedVLM
from megatron.bridge.models.step_vl.modeling_step3_vl.model import Step3VLModel
from megatron.bridge.models.step_vl.step3_vl_provider import Step3VLModelProvider


@MegatronModelBridge.register_bridge(
    source="Step3VL10BForCausalLM",
    target=Step3VLModel,
    provider=Step3VLModelProvider,
)
class Step3VLBridge(MegatronModelBridge):
    """
    Megatron Bridge for Step3-VL (stepfun-ai/Step3-VL-10B).

    Converts HuggingFace Step3VL10BForCausalLM ↔ Megatron Step3VLModel.

    Language backbone: Qwen3-8B (36 layers, hidden=4096, 32 heads, 8 KV heads,
    silu activation, RMSNorm, QK layernorm, RoPE theta=1e6).

    Vision backbone: custom 47-layer ViT, width=1536, patch=14, image=728,
    2-D RoPE + LayerScale. Output is spatially downsampled 4× via two
    stride-2 Conv2d layers, then projected to language hidden size.

    HF checkpoint key structure (after _checkpoint_conversion_mapping is applied
    on load by the HF model):
        model.vision_model.**           – vision encoder + downsamplers
        model.language_model.**         – Qwen3 language backbone
        model.vit_large_projector.**    – linear projector
        lm_head.weight                  – output projection

    Megatron model key structure:
        vision_model.**                 – maps from model.vision_model.**
        vit_large_projector.**          – maps from model.vit_large_projector.**
        language_model.**               – maps from model.language_model.** / lm_head.*

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained(
        ...     "stepfun-ai/Step3-VL-10B", trust_remote_code=True
        ... )
        >>> provider = bridge.to_megatron_provider()
    """

    def provider_bridge(self, hf_pretrained: PreTrainedVLM) -> Step3VLModelProvider:
        """Translate HF StepRoboticsConfig → Step3VLModelProvider."""
        hf_config = hf_pretrained.config
        text_config = hf_config.text_config

        # Map standard Qwen3 text-config fields via CONFIG_MAPPING
        provider_kwargs = self.hf_config_to_provider_kwargs(text_config)
        valid_fields = Step3VLModelProvider.__dataclass_fields__
        provider = Step3VLModelProvider(**{k: v for k, v in provider_kwargs.items() if k in valid_fields})

        # Qwen3-specific language-model settings
        provider.normalization = "RMSNorm"
        provider.gated_linear_unit = True
        provider.add_bias_linear = False
        provider.add_qkv_bias = False  # Qwen3 has no QKV bias
        provider.hidden_dropout = 0.0
        provider.qk_layernorm = True  # Qwen3 QK norm
        provider.bf16 = True
        provider.params_dtype = torch.bfloat16
        provider.autocast_dtype = torch.bfloat16

        # tie_word_embeddings is on the top-level config, NOT on text_config
        provider.share_embeddings_and_output_weights = getattr(hf_config, "tie_word_embeddings", False)

        # Vision-specific settings
        provider.vision_config = hf_config.vision_config
        provider.projector_bias = getattr(hf_config, "projector_bias", False)
        provider.image_token_id = getattr(hf_config, "image_token_id", 151679)

        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Return weight mappings between HF Step3VL and Megatron Step3VLModel.

        HF keys are from Step3VL10BForCausalLM.state_dict() (post-load remapping):
            model.vision_model.**  /  model.language_model.**  /
            model.vit_large_projector.weight  /  lm_head.weight

        Megatron keys follow Step3VLModel attribute names.
        """
        # ---------------------------------------------------------------
        # 1. Vision encoder + downsamplers: replicated (no TP sharding)
        # ---------------------------------------------------------------
        mapping_list = [
            ReplicatedMapping(
                megatron_param="vision_model.**",
                hf_param="model.vision_model.**",
            ),
            # Projector is small enough to replicate across TP ranks
            AutoMapping(
                megatron_param="vit_large_projector.weight",
                hf_param="model.vit_large_projector.weight",
            ),
        ]

        # ---------------------------------------------------------------
        # 2. Language model: standard Qwen3 mappings with namespace prefix
        # ---------------------------------------------------------------
        # 1:1 parameter mappings (HF key → Megatron key)
        param_mappings = {
            # Embeddings and output
            "language_model.embedding.word_embeddings.weight": "model.language_model.embed_tokens.weight",
            "language_model.output_layer.weight": "lm_head.weight",
            "language_model.decoder.final_layernorm.weight": "model.language_model.norm.weight",
            # Per-layer pre-attention norm (fused into QKV linear's layer_norm)
            "language_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": (
                "model.language_model.layers.*.input_layernorm.weight"
            ),
            # Per-layer pre-MLP norm (fused into fc1's layer_norm)
            "language_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight": (
                "model.language_model.layers.*.post_attention_layernorm.weight"
            ),
            # Per-head QK norms (Qwen3 specific)
            "language_model.decoder.layers.*.self_attention.q_layernorm.weight": (
                "model.language_model.layers.*.self_attn.q_norm.weight"
            ),
            "language_model.decoder.layers.*.self_attention.k_layernorm.weight": (
                "model.language_model.layers.*.self_attn.k_norm.weight"
            ),
            # Attention output projection
            "language_model.decoder.layers.*.self_attention.linear_proj.weight": (
                "model.language_model.layers.*.self_attn.o_proj.weight"
            ),
            # MLP down-projection
            "language_model.decoder.layers.*.mlp.linear_fc2.weight": (
                "model.language_model.layers.*.mlp.down_proj.weight"
            ),
        }
        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        # QKV: separate Q, K, V → fused linear_qkv (no bias for Qwen3)
        mapping_list.append(
            QKVMapping(
                megatron_param="language_model.decoder.layers.*.self_attention.linear_qkv.weight",
                q="model.language_model.layers.*.self_attn.q_proj.weight",
                k="model.language_model.layers.*.self_attn.k_proj.weight",
                v="model.language_model.layers.*.self_attn.v_proj.weight",
            )
        )

        # Gated MLP: gate_proj + up_proj → fused linear_fc1
        mapping_list.append(
            GatedMLPMapping(
                megatron_param="language_model.decoder.layers.*.mlp.linear_fc1.weight",
                gate="model.language_model.layers.*.mlp.gate_proj.weight",
                up="model.language_model.layers.*.mlp.up_proj.weight",
            )
        )

        return MegatronMappingRegistry(*mapping_list)
