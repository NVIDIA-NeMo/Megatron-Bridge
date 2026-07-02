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

import math
from typing import Any

import torch
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.enums import AttnBackend
from transformers import AutoConfig, Gemma3ForCausalLM

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
)
from megatron.bridge.models.conversion.transformers_compat import (
    rope_local_base_freq_from_hf,
    rope_scaling_factor_from_hf,
    rope_theta_from_hf,
)
from megatron.bridge.models.gemma.gemma3_provider import Gemma3ModelProvider
from megatron.bridge.models.gemma.model_config import Gemma3ModelConfig
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


# Register Gemma3-specific module types for AutoMapping
AutoMapping.register_module_type("Gemma3TEDotProductAttention", "replicated")
AutoMapping.register_module_type("TERowParallelLinearLayerNorm", "row")


@MegatronModelBridge.register_bridge(
    source=Gemma3ForCausalLM,
    target=GPTModel,
    provider=Gemma3ModelProvider,
    model_type="gemma3",
)
class Gemma3ModelBridge(MegatronModelBridge):
    """
    Megatron Bridge for Gemma3.
    """

    MODEL_CONFIG_CLASS = Gemma3ModelConfig

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> Gemma3ModelProvider:
        """Convert HuggingFace config to Gemma3ModelProvider."""
        provider = super().provider_bridge(hf_pretrained)
        hf_config = hf_pretrained.config

        # Precision config is stored in the VL Config
        hf_vl_config = AutoConfig.from_pretrained(hf_pretrained._model_name_or_path)

        # Override dtype from VL config (has precision info)
        params_dtype = self.dtype_from_hf(hf_vl_config, default=torch.float32)
        provider.fp16 = params_dtype == torch.float16
        provider.bf16 = params_dtype == torch.bfloat16
        provider.params_dtype = params_dtype
        provider.autocast_dtype = params_dtype

        # Gemma3-specific features not in CONFIG_MAPPING
        provider.window_size = hf_config.sliding_window
        provider.rotary_base = (
            rope_local_base_freq_from_hf(hf_config),
            rope_theta_from_hf(hf_config),
        )
        provider.softmax_scale = 1.0 / math.sqrt(hf_config.query_pre_attn_scalar)
        provider.rope_scaling_factor = rope_scaling_factor_from_hf(hf_config)

        return provider

    def hf_config_to_model_config_kwargs(self, hf_config: Any) -> dict[str, Any]:
        """Convert a Hugging Face Gemma3 config to model-config kwargs.

        Args:
            hf_config: Hugging Face Gemma3 text configuration.

        Returns:
            Flat model and transformer config keyword arguments.
        """
        config_kwargs = super().hf_config_to_model_config_kwargs(hf_config)
        config_kwargs.update(
            window_size=(hf_config.sliding_window - 1, 0),
            rotary_base_local=rope_local_base_freq_from_hf(hf_config),
            rotary_base=rope_theta_from_hf(hf_config),
            softmax_scale=1.0 / math.sqrt(hf_config.query_pre_attn_scalar),
            rope_scaling_factor=rope_scaling_factor_from_hf(hf_config),
            normalization="RMSNorm",
            qk_layernorm=True,
            layernorm_zero_centered_gamma=True,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            gated_linear_unit=True,
            add_bias_linear=False,
            share_embeddings_and_output_weights=True,
            activation_func=self.hf_to_megatron_activation("gelu_pytorch_tanh"),
            attention_backend=AttnBackend.flash,
        )
        return config_kwargs

    def model_config_bridge(self, hf_pretrained: PreTrainedCausalLM) -> Gemma3ModelConfig:
        """Create Gemma3's builder-backed config using parent VL precision metadata."""
        hf_vl_config = AutoConfig.from_pretrained(hf_pretrained._model_name_or_path)
        config_kwargs = self.hf_config_to_model_config_kwargs(hf_pretrained.config)
        params_dtype = self.dtype_from_hf(hf_vl_config, default=torch.float32)
        config_kwargs.update(
            fp16=params_dtype == torch.float16,
            bf16=params_dtype == torch.bfloat16,
            params_dtype=params_dtype,
            autocast_dtype=params_dtype,
        )
        model_kwargs, transformer_kwargs = self._partition_model_config_kwargs(
            config_kwargs,
            self.MODEL_CONFIG_CLASS,
            self.TRANSFORMER_CONFIG_CLASS,
        )
        transformer_config = self.TRANSFORMER_CONFIG_CLASS(**transformer_kwargs)
        return self.MODEL_CONFIG_CLASS(transformer=transformer_config, **model_kwargs)

    @classmethod
    def megatron_to_hf_config(cls, provider: Gemma3ModelProvider | Gemma3ModelConfig) -> dict:
        """Convert a Gemma3 provider or model config back to Hugging Face config."""
        hf_config = super().megatron_to_hf_config(provider)

        if hasattr(provider, "rotary_base_local"):
            hf_config["rope_local_base_freq"] = provider.rotary_base_local
            hf_config["rope_theta"] = provider.rotary_base
        elif isinstance(provider.rotary_base, tuple):
            rope_local_base_freq, rope_theta = provider.rotary_base
            hf_config["rope_local_base_freq"] = rope_local_base_freq
            hf_config["rope_theta"] = rope_theta

        window_size = provider.window_size
        hf_config["sliding_window"] = window_size[0] + 1 if isinstance(window_size, (list, tuple)) else window_size

        if provider.softmax_scale:
            query_pre_attn_scalar = 1.0 / (provider.softmax_scale**2)
            rounded = round(query_pre_attn_scalar)
            if math.isclose(query_pre_attn_scalar, rounded, rel_tol=0.0, abs_tol=1e-9):
                query_pre_attn_scalar = rounded
            hf_config["query_pre_attn_scalar"] = query_pre_attn_scalar

        if getattr(provider, "rope_scaling_factor", 1.0) != 1.0:
            hf_config["rope_scaling"] = {
                "factor": provider.rope_scaling_factor,
                "type": "linear",
            }

        return hf_config

    def mapping_registry(self) -> MegatronMappingRegistry:
        mapping = {
            # word emebdding
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            # attention
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.self_attention.q_layernorm.weight": "model.layers.*.self_attn.q_norm.weight",
            "decoder.layers.*.self_attention.k_layernorm.weight": "model.layers.*.self_attn.k_norm.weight",
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            "decoder.layers.*.self_attention.linear_proj.post_layernorm.weight": (
                "model.layers.*.post_attention_layernorm.weight"
            ),
            # mlp
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.pre_feedforward_layernorm.weight",
            "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
            "decoder.layers.*.mlp.linear_fc2.post_layernorm.weight": (
                "model.layers.*.post_feedforward_layernorm.weight"
            ),
            # final norm
            "decoder.final_layernorm.weight": "model.norm.weight",
        }
        mapping_list = []
        # Convert each dictionary entry to AutoMapping(megatron_param, hf_param)
        for megatron_param, hf_param in mapping.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

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
