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

from functools import partial

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.models.gpt.gpt_model import GPTModel

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.deepseek.common import get_common_mapping_list
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.mla_provider import MLAModelProvider


try:
    import transformer_engine  # noqa: F401

    HAVE_TE = True
except (ImportError, ModuleNotFoundError):
    HAVE_TE = False


@MegatronModelBridge.register_bridge(
    source="DeepseekV2ForCausalLM",
    target=GPTModel,
    provider=MLAModelProvider,
    model_type="deepseek_v2",
)
class DeepSeekV2Bridge(MegatronModelBridge):
    """Megatron Bridge for DeepSeek-V2."""

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
        provider.q_lora_rank = hf_config.q_lora_rank

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

    def build_conversion_tasks(self, hf_pretrained, megatron_model):
        """Override to store config before mapping_registry is called."""
        # Store config on instance for use in mapping_registry
        from transformers import PretrainedConfig

        self._hf_config = hf_pretrained if isinstance(hf_pretrained, PretrainedConfig) else hf_pretrained.config
        return super().build_conversion_tasks(hf_pretrained, megatron_model)

    def mapping_registry(self) -> MegatronMappingRegistry:
        # Get hf_config if available (set by build_conversion_tasks)
        hf_config = getattr(self, "_hf_config", None)
        mapping_list = get_common_mapping_list(hf_config=hf_config)
        return MegatronMappingRegistry(*mapping_list)
