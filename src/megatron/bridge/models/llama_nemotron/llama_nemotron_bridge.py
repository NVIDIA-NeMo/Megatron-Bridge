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

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
)
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.llama.llama_provider import Llama31ModelProvider


@MegatronModelBridge.register_bridge(source="DeciLMForCausalLM", target=GPTModel)
class LlamaNemotronBridge(MegatronModelBridge):
    """
    Megatron Bridge for Heterogeneous Llama-Nemotron models (Super/Ultra).

    This bridge handles heterogeneous Llama-Nemotron models that use the DeciLMForCausalLM
    architecture with block_configs for heterogeneous layer specifications. These models
    require special handling because:

    1. They use custom modeling code (DeciLMForCausalLM) loaded via auto_map
    2. They have heterogeneous block configurations (different layers have different specs)
    3. They require trust_remote_code=True to load from HuggingFace

    Supported models:
    - nvidia/Llama-3_3-Nemotron-Super-49B-v1 (80 layers, 8192 hidden)
    - nvidia/Llama-3_1-Nemotron-Ultra-253B-v1 (162 layers, 16384 hidden)

    Homogeneous Llama-Nemotron models (Nano/70B) use standard LlamaForCausalLM
    architecture and are handled by the regular LlamaBridge.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> # DeciLMForCausalLM models will automatically use this bridge
        >>> bridge = AutoBridge.from_hf_pretrained(
        ...     "nvidia/Llama-3_3-Nemotron-Super-49B-v1",
        ...     trust_remote_code=True
        ... )
        >>> provider = bridge.to_megatron_provider()
    """

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> Llama31ModelProvider:
        hf_config = hf_pretrained.config

        # Detect model variant based on configuration
        provider_class = self._detect_nemotron_variant(hf_config)

        # Calculate num_query_groups for heterogeneous models
        # For heterogeneous models, GQA is defined in each block config
        # We assume block 0 has a non-no-op attention layer
        num_query_groups = hf_config.num_key_value_heads
        if hasattr(hf_config, "block_configs") and hf_config.block_configs:
            # Extract from block_configs[0].attention.n_heads_in_group
            block_0 = hf_config.block_configs[0]
            if hasattr(block_0, "attention") and hasattr(block_0.attention, "n_heads_in_group"):
                n_heads_in_group = block_0.attention.n_heads_in_group
                num_query_groups = hf_config.num_attention_heads // n_heads_in_group

        # Prepare kwargs for provider creation
        provider_kwargs = dict(
            num_layers=hf_config.num_hidden_layers,
            hidden_size=hf_config.hidden_size,
            ffn_hidden_size=hf_config.intermediate_size,
            num_attention_heads=hf_config.num_attention_heads,
            init_method_std=hf_config.initializer_range,
            layernorm_epsilon=hf_config.rms_norm_eps,
            num_query_groups=num_query_groups,
            seq_length=hf_config.max_position_embeddings,
            rotary_base=hf_config.rope_theta,
            kv_channels=getattr(hf_config, "head_dim", None),
            gated_linear_unit=True,  # Llama uses SwiGLU
            make_vocab_size_divisible_by=self.make_vocab_size_divisible_by(hf_config.vocab_size),
            share_embeddings_and_output_weights=getattr(hf_config, "tie_word_embeddings", False),
            fp16=(self.dtype_from_hf(hf_config, default=torch.float32) == torch.float16),
            bf16=(self.dtype_from_hf(hf_config, default=torch.float32) == torch.bfloat16),
            params_dtype=self.dtype_from_hf(hf_config, default=torch.float32),
            generation_config=hf_pretrained.generation_config,
            vocab_size=hf_config.vocab_size,
        )

        # Handle rope scaling for Llama 3.1/3.3
        if hasattr(hf_config, "rope_scaling") and hf_config.rope_scaling:
            if hf_config.rope_scaling.get("rope_type") == "llama3":
                provider_kwargs["scale_factor"] = hf_config.rope_scaling.get("factor", 8.0)

        # Handle heterogeneous configurations
        if hasattr(hf_config, "block_configs") and hf_config.block_configs:
            # For heterogeneous models, pass the encoded JSON from the HF config
            provider_kwargs["heterogeneous_layers_config_encoded_json"] = hf_config.to_json_string()

        provider = provider_class(**provider_kwargs)
        return provider

    def _detect_nemotron_variant(self, hf_config) -> type:
        """Detect which heterogeneous Llama-Nemotron variant based on model configuration.

        This method only handles heterogeneous models since homogeneous Nemotron models
        are handled by the regular LlamaBridge.
        """
        from megatron.bridge.models.llama_nemotron.llama_nemotron_provider import (
            Llama31NemotronUltra253BProvider,
            Llama33NemotronSuper49BProvider,
        )

        num_layers = hf_config.num_hidden_layers

        # Only handle heterogeneous models (they have block_configs)
        if hasattr(hf_config, "block_configs") and hf_config.block_configs:
            if num_layers == 80:
                return Llama33NemotronSuper49BProvider
            elif num_layers == 162:
                return Llama31NemotronUltra253BProvider

        # This bridge should only be used for heterogeneous models
        raise ValueError(
            f"LlamaNemotronBridge only handles heterogeneous models with block_configs. "
            f"Model with {num_layers} layers and no block_configs should use LlamaBridge."
        )

    def mapping_registry(self) -> MegatronMappingRegistry:
        # Return MegatronMappingRegistry containing parameter mappings from Megatron to HF format
        # Similar to Llama bridge but adapted for Llama-Nemotron specifics

        # Dictionary maps Megatron parameter names -> HF parameter names
        # Supports wildcard (*) patterns for layer-specific parameters
        param_mappings = {
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "output_layer.weight": "lm_head.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
        }

        mapping_list = []
        # Convert each dictionary entry to AutoMapping(megatron_param, hf_param)
        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        # Add special mappings that require parameter concatenation/transformation
        mapping_list.extend(
            [
                # QKV: Combine separate Q, K, V matrices into single QKV matrix
                QKVMapping(
                    megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                    q="model.layers.*.self_attn.q_proj.weight",
                    k="model.layers.*.self_attn.k_proj.weight",
                    v="model.layers.*.self_attn.v_proj.weight",
                ),
                # Gated MLP: Combine gate and up projection matrices into single FC1 matrix
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                    gate="model.layers.*.mlp.gate_proj.weight",
                    up="model.layers.*.mlp.up_proj.weight",
                ),
            ]
        )

        return MegatronMappingRegistry(*mapping_list)
