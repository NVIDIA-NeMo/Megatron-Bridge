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
from megatron.core.models.gpt.gpt_model import GPTModel
from transformers import MistralForCausalLM

from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.mistral.mistral_provider import MistralModelProvider
from megatron.bridge.models.model_bridge import MegatronModelBridge
from megatron.bridge.models.param_mapping import (
    GatedMLPMapping,
    QKVMapping,
    TPAwareMapping,
)


@MegatronModelBridge.register_bridge(source=MistralForCausalLM, target=GPTModel)
class MistralCausalBridge(MegatronModelBridge):
    """
    Megatron Hub Bridge for Mistral Causal LM.

    This bridge handles the conversion between HuggingFace MistralForCausalLM
    and Megatron-Core GPTModel formats. Mistral models feature sliding window
    attention (for some variants) and grouped query attention (GQA).

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("mistralai/Mistral-7B-v0.1")
        >>> provider = bridge.to_megatron_provider()
    """

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> MistralModelProvider:
        hf_config = hf_pretrained.config

        # Handle sliding window - convert to tuple format if it exists
        sliding_window = getattr(hf_config, "sliding_window", None)
        window_size = None
        if sliding_window is not None:
            window_size = (sliding_window, 0)  # Format: (window_size, backward)

        provider = MistralModelProvider(
            num_layers=hf_config.num_hidden_layers,
            hidden_size=hf_config.hidden_size,
            ffn_hidden_size=hf_config.intermediate_size,
            num_attention_heads=hf_config.num_attention_heads,
            num_query_groups=hf_config.num_key_value_heads,
            init_method_std=hf_config.initializer_range,
            layernorm_epsilon=hf_config.rms_norm_eps,
            gated_linear_unit=True,
            make_vocab_size_divisible_by=self.make_vocab_size_divisible_by(hf_config.vocab_size),
            rotary_base=hf_config.rope_theta,
            share_embeddings_and_output_weights=getattr(hf_config, "tie_word_embeddings", False),
            vocab_size=hf_config.vocab_size,
            seq_length=hf_config.max_position_embeddings,
            window_size=window_size,
            kv_channels=getattr(hf_config, "head_dim", None),
            fp16=(self.dtype_from_hf(hf_config, default=torch.float32) == torch.float16),
            bf16=(self.dtype_from_hf(hf_config, default=torch.float32) == torch.bfloat16),
            params_dtype=self.dtype_from_hf(hf_config, default=torch.float32),
            generation_config=hf_pretrained.generation_config,
        )

        provider.gradient_accumulation_fusion = False
        provider.variable_seq_lengths = True

        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        return MegatronMappingRegistry(
            # ------------------------------------------------------------------
            # Embedding & output projection – column-parallel
            # ------------------------------------------------------------------
            TPAwareMapping(
                megatron_param="embedding.word_embeddings.weight",
                hf_param="model.embed_tokens.weight",
            ),
            TPAwareMapping(
                megatron_param="output_layer.weight",
                hf_param="lm_head.weight",
            ),
            # ------------------------------------------------------------------
            # LayerNorm (replicated across TP ranks)
            # ------------------------------------------------------------------
            TPAwareMapping(
                megatron_param="decoder.final_layernorm.weight",
                hf_param="model.norm.weight",
            ),
            TPAwareMapping(
                megatron_param="decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
                hf_param="model.layers.*.input_layernorm.weight",
            ),
            TPAwareMapping(
                megatron_param="decoder.layers.*.mlp.linear_fc1.layer_norm_weight",
                hf_param="model.layers.*.post_attention_layernorm.weight",
            ),
            # ------------------------------------------------------------------
            # Attention – QKV split in HF, combined in Megatron
            # ------------------------------------------------------------------
            QKVMapping(
                megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                q="model.layers.*.self_attn.q_proj.weight",
                k="model.layers.*.self_attn.k_proj.weight",
                v="model.layers.*.self_attn.v_proj.weight",
            ),
            TPAwareMapping(
                megatron_param="decoder.layers.*.self_attention.linear_proj.weight",
                hf_param="model.layers.*.self_attn.o_proj.weight",
            ),
            # ------------------------------------------------------------------
            # MLP – Gated MLP with gate_proj and up_proj
            # ------------------------------------------------------------------
            GatedMLPMapping(
                megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                gate="model.layers.*.mlp.gate_proj.weight",
                up="model.layers.*.mlp.up_proj.weight",
            ),
            TPAwareMapping(
                megatron_param="decoder.layers.*.mlp.linear_fc2.weight",
                hf_param="model.layers.*.mlp.down_proj.weight",
            ),
        )
