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

from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import MLATransformerConfig

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.deepseek.common import get_common_mapping_list
from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.mla_provider import MLAModelProvider


try:
    import transformer_engine  # noqa: F401

    HAVE_TE = True
except (ImportError, ModuleNotFoundError):
    HAVE_TE = False


@dataclass(kw_only=True)
class DeepSeekV2ModelConfig(BridgeGPTModelConfig):
    """Serializable DeepSeek-V2 GPT build configuration."""

    transformer_layer_spec: ModuleSpec | Callable[[BridgeGPTModelConfig], ModuleSpec] | None = field(
        default_factory=lambda: partial(get_gpt_decoder_block_spec, use_transformer_engine=HAVE_TE)
    )


@MegatronModelBridge.register_bridge(
    source="DeepseekV2ForCausalLM",
    target=GPTModel,
    provider=MLAModelProvider,
    model_type="deepseek_v2",
)
class DeepSeekV2Bridge(MegatronModelBridge):
    """Megatron Bridge for DeepSeek-V2."""

    TRANSFORMER_CONFIG_CLASS = MLATransformerConfig
    MODEL_CONFIG_CLASS = DeepSeekV2ModelConfig
    CUSTOM_PROVIDER_MODEL_CONFIG_SUPPORTED = True

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> MLAModelProvider:
        provider = super().provider_bridge(hf_pretrained)
        hf_config = hf_pretrained.config

        provider.transformer_layer_spec = partial(get_gpt_decoder_block_spec, use_transformer_engine=HAVE_TE)
        provider.normalization = "RMSNorm"
        provider.gated_linear_unit = True
        provider.add_bias_linear = False
        provider.share_embeddings_and_output_weights = False
        provider.qk_layernorm = True
        provider.multi_latent_attention = True

        provider.moe_grouped_gemm = True
        provider.moe_router_pre_softmax = True
        provider.moe_token_dispatcher_type = "alltoall"
        provider.moe_router_load_balancing_type = "seq_aux_loss"
        provider.moe_shared_expert_overlap = True
        provider.moe_router_dtype = "fp32"
        provider.moe_permute_fusion = True

        provider.apply_rope_fusion = False
        provider.gradient_accumulation_fusion = True
        provider.bias_activation_fusion = True
        provider.bias_dropout_fusion = True
        provider.cross_entropy_fusion_impl = "te"
        provider.cross_entropy_loss_fusion = True
        provider.masked_softmax_fusion = True
        provider.persist_layer_norm = True

        provider.hidden_dropout = 0.0
        provider.attention_softmax_in_fp32 = False

        provider.make_vocab_size_divisible_by = 3200
        provider.seq_length = 4096

        provider.moe_layer_freq = [0] * hf_config.first_k_dense_replace + [1] * (
            hf_config.num_hidden_layers - hf_config.first_k_dense_replace
        )
        provider.moe_shared_expert_intermediate_size = hf_config.moe_intermediate_size * hf_config.n_shared_experts

        return provider

    def hf_config_to_model_config_kwargs(self, hf_config: Any) -> dict[str, Any]:
        """Convert a Hugging Face DeepSeek-V2 config to Megatron model-config kwargs."""
        config_kwargs = super().hf_config_to_model_config_kwargs(hf_config)
        config_kwargs.update(
            normalization="RMSNorm",
            gated_linear_unit=True,
            add_bias_linear=False,
            share_embeddings_and_output_weights=False,
            qk_layernorm=True,
            multi_latent_attention=True,
            moe_grouped_gemm=True,
            moe_router_pre_softmax=True,
            moe_token_dispatcher_type="alltoall",
            moe_router_load_balancing_type="seq_aux_loss",
            moe_shared_expert_overlap=True,
            moe_router_dtype="fp32",
            moe_permute_fusion=True,
            apply_rope_fusion=False,
            bias_activation_fusion=True,
            bias_dropout_fusion=True,
            masked_softmax_fusion=True,
            persist_layer_norm=True,
            hidden_dropout=0.0,
            attention_softmax_in_fp32=False,
            make_vocab_size_divisible_by=3200,
            seq_length=4096,
            moe_layer_freq=[0] * hf_config.first_k_dense_replace
            + [1] * (hf_config.num_hidden_layers - hf_config.first_k_dense_replace),
            moe_shared_expert_intermediate_size=hf_config.moe_intermediate_size * hf_config.n_shared_experts,
        )
        return config_kwargs

    @classmethod
    def megatron_to_hf_config(cls, provider) -> dict:
        hf_cfg = super().megatron_to_hf_config(provider)

        # Megatron uses None="not set/disabled", but HF expects integers
        hf_cfg["num_nextn_predict_layers"] = hf_cfg.get("num_nextn_predict_layers") or 0
        hf_cfg["n_group"] = hf_cfg.get("n_group") or 1
        hf_cfg["topk_group"] = hf_cfg.get("topk_group") or 1

        # Reconstruct first_k_dense_replace from moe_layer_freq (count leading dense layers)
        moe_layer_freq = getattr(provider, "moe_layer_freq", None)
        if moe_layer_freq is not None and isinstance(moe_layer_freq, list):
            first_k_dense_replace = 0
            for val in moe_layer_freq:
                if val == 0:
                    first_k_dense_replace += 1
                else:
                    break
            hf_cfg["first_k_dense_replace"] = first_k_dense_replace

        # Reconstruct n_shared_experts from moe_shared_expert_intermediate_size / moe_ffn_hidden_size
        shared_size = getattr(provider, "moe_shared_expert_intermediate_size", None)
        moe_ffn = getattr(provider, "moe_ffn_hidden_size", None)
        if shared_size is not None and moe_ffn is not None and moe_ffn > 0:
            hf_cfg["n_shared_experts"] = shared_size // moe_ffn

        return hf_cfg

    def mapping_registry(self) -> MegatronMappingRegistry:
        mapping_list = get_common_mapping_list(hf_config=self.hf_config)
        return MegatronMappingRegistry(*mapping_list)
