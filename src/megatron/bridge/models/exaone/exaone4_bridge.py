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

"""Megatron Bridge for EXAONE 4.0 (LG AI Research).

EXAONE 4.0 architecture overview:
- Pure Post-LayerNorm (no Pre-LN / input_layernorm)
- QK RMSNorm (similar to Qwen3)
- GQA with 32 heads / 8 KV heads
- SwiGLU activation
- RoPE with llama3-style scaling
- Tied word embeddings (embed_tokens == lm_head)

Key differences from standard Llama/Qwen:
- No input_layernorm or pre_feedforward_layernorm weights
- Has post_attention_layernorm (after self-attention output)
- Has post_feedforward_layernorm (after MLP output, EXAONE-specific)
- Post-LN mapping follows Gemma2 pattern: *.post_layernorm.weight

References:
- HuggingFace: LGAI-EXAONE/EXAONE-4.0-1.2B
- Gemma2 bridge: Post-LN via TERowParallelLinearLayerNorm pattern
- Qwen3 bridge: QK layernorm mapping pattern
"""

import logging

import torch
from megatron.core.models.gpt.gpt_model import GPTModel

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
)
from megatron.bridge.models.exaone.exaone4_provider import Exaone4ModelProvider
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


logger = logging.getLogger(__name__)


# Register custom EXAONE modules for AutoMapping weight distribution
# TERowParallelLinearLayerNorm is a row-parallel linear with post-layernorm
# (same pattern as Gemma2 for Post-LN architectures)
AutoMapping.register_module_type("TERowParallelLinearLayerNorm", "row")


@MegatronModelBridge.register_bridge(
    source="Exaone4ForCausalLM",  # HF architecture string (auto_map / trust_remote_code)
    target=GPTModel,
    provider=Exaone4ModelProvider,
    model_type="exaone4",
)
class Exaone4Bridge(MegatronModelBridge):
    """
    Megatron Bridge for EXAONE 4.0 Causal LM.

    Supports bidirectional conversion between HuggingFace EXAONE 4.0 checkpoints
    and Megatron-Core GPTModel format.

    Architecture notes:
    - EXAONE 4.0 uses pure Post-LayerNorm (no input_layernorm).
    - Post-LN is implemented via custom layer spec with TERowParallelLinearLayerNorm,
      following the same pattern established by Gemma2 bridge.
    - QK RMSNorm is mapped using the same convention as Qwen3.
    - 1.2B model uses full attention only (no sliding window / hybrid attention).
    - 32B model introduces hybrid attention (LLLG pattern) — future extension.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained(
        ...     "LGAI-EXAONE/EXAONE-4.0-1.2B",
        ...     trust_remote_code=True,
        ... )
        >>> provider = bridge.to_megatron_provider()
    """

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> Exaone4ModelProvider:
        """Convert HuggingFace EXAONE 4.0 config to Megatron Exaone4ModelProvider.

        Maps HF config fields to Megatron TransformerConfig parameters and sets
        EXAONE-specific options including Post-LN, QK norm, and RoPE scaling.

        Args:
            hf_pretrained: HuggingFace PreTrainedCausalLM containing the EXAONE config

        Returns:
            Exaone4ModelProvider configured for EXAONE 4.0 architecture
        """
        hf_config = hf_pretrained.config

        provider = Exaone4ModelProvider(
            num_layers=hf_config.num_hidden_layers,
            hidden_size=hf_config.hidden_size,
            ffn_hidden_size=hf_config.intermediate_size,
            num_attention_heads=hf_config.num_attention_heads,
            num_query_groups=hf_config.num_key_value_heads,
            seq_length=hf_config.max_position_embeddings,
            init_method_std=hf_config.initializer_range,
            layernorm_epsilon=hf_config.rms_norm_eps,
            rotary_base=getattr(hf_config, 'rope_theta', getattr(hf_config, 'rotary_base', 1000000.0)),
            make_vocab_size_divisible_by=self.make_vocab_size_divisible_by(hf_config.vocab_size),
            share_embeddings_and_output_weights=getattr(hf_config, "tie_word_embeddings", True),
            kv_channels=getattr(hf_config, "head_dim", None),
            fp16=(self.dtype_from_hf(hf_config, default=torch.float32) == torch.float16),
            bf16=(self.dtype_from_hf(hf_config, default=torch.float32) == torch.bfloat16),
            params_dtype=self.dtype_from_hf(hf_config, default=torch.float32),
            vocab_size=hf_config.vocab_size,
        )

        # RoPE scaling for EXAONE 4.0 (llama3-style)
        hf_rope_scaling = getattr(hf_config, "rope_scaling", None)
        if hf_rope_scaling is not None and hf_rope_scaling.get("rope_type") == "llama3":
            provider.rope_scaling = True
            provider.rope_scaling_factor = hf_rope_scaling.get("factor", 16.0)

        return provider

    @classmethod
    def megatron_to_hf_config(cls, provider: Exaone4ModelProvider) -> dict:
        """Convert Megatron Exaone4ModelProvider config to HuggingFace config dict.

        Args:
            provider: Exaone4ModelProvider with EXAONE configuration

        Returns:
            Dictionary of HuggingFace Exaone4Config parameters
        """
        hf_config = super(Exaone4Bridge, cls).megatron_to_hf_config(provider)

        # EXAONE-specific config fields
        hf_config["model_type"] = "exaone4"
        hf_config["tie_word_embeddings"] = provider.share_embeddings_and_output_weights

        # RoPE scaling
        if provider.rope_scaling:
            hf_config["rope_scaling"] = {
                "rope_type": "llama3",
                "factor": provider.rope_scaling_factor,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8192,
            }

        return hf_config

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Return MegatronMappingRegistry containing parameter mappings.

        EXAONE 4.0 weight mapping combines patterns from:
        - Llama: Basic GPT structure (embed, QKV, GatedMLP, final_layernorm)
        - Qwen3: QK layernorm (q_norm → q_layernorm, k_norm → k_layernorm)
        - Gemma2: Post-LN (post_*_layernorm → *.post_layernorm.weight)

        Key difference: No input_layernorm or pre_feedforward_layernorm mappings
        because EXAONE uses pure Post-LN (not Pre-LN or sandwich norm).
        """

        # =====================================================================
        # 1:1 Parameter Mappings (Megatron → HF)
        # =====================================================================
        param_mappings = {
            # Embedding & output
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            # NOTE: No lm_head.weight mapping — tie_word_embeddings=true reuses embed_tokens
            "decoder.final_layernorm.weight": "model.norm.weight",
            # Attention output projection
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            # QK RMSNorm (Qwen3 pattern)
            "decoder.layers.*.self_attention.q_layernorm.weight": "model.layers.*.self_attn.q_norm.weight",
            "decoder.layers.*.self_attention.k_layernorm.weight": "model.layers.*.self_attn.k_norm.weight",
            # Post-LN: post-attention layernorm (Gemma2 pattern)
            "decoder.layers.*.self_attention.linear_proj.post_layernorm.weight": "model.layers.*.post_attention_layernorm.weight",
            # Post-LN: post-feedforward layernorm (Gemma2 pattern)
            "decoder.layers.*.mlp.linear_fc2.post_layernorm.weight": "model.layers.*.post_feedforward_layernorm.weight",
            # MLP down projection
            "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
        }

        mapping_list = []
        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        # =====================================================================
        # Composite Mappings (require concatenation/splitting)
        # =====================================================================
        mapping_list.extend(
            [
                # QKV: Merge separate Q, K, V projections into single QKV matrix
                QKVMapping(
                    megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                    q="model.layers.*.self_attn.q_proj.weight",
                    k="model.layers.*.self_attn.k_proj.weight",
                    v="model.layers.*.self_attn.v_proj.weight",
                ),
                # Gated MLP: Merge gate and up projections into single FC1 matrix
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                    gate="model.layers.*.mlp.gate_proj.weight",
                    up="model.layers.*.mlp.up_proj.weight",
                ),
            ]
        )

        return MegatronMappingRegistry(*mapping_list)
