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
from megatron.core.models.mamba import MambaModel

from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.mamba.mamba_bridge import MambaBridge
from megatron.bridge.models.mamba.nemotron_h_provider import NemotronHModelProvider


@MegatronModelBridge.register_bridge(source="NemotronHForCausalLM", target=MambaModel)
class NemotronHBridge(MambaBridge):
    """
    Megatron Bridge for Nemotron-H Causal LM.

    This bridge handles the conversion between HuggingFace NemotronHForCausalLM
    and Megatron-Core MambaModel formats, including weight mappings and
    configuration translation.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("nvidia/Nemotron-H-8B-Base-8K", trust_remote_code=True)
        >>> provider = bridge.to_megatron_provider()
    """

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> NemotronHModelProvider:
        hf_config = hf_pretrained.config

        return NemotronHModelProvider(
            num_layers=hf_config.num_hidden_layers,
            hidden_size=hf_config.hidden_size,
            ffn_hidden_size=hf_config.intermediate_size,
            add_bias_linear=hf_config.use_bias,
            num_attention_heads=hf_config.num_attention_heads,
            num_query_groups=hf_config.num_key_value_heads,
            init_method_std=hf_config.initializer_range,
            layernorm_epsilon=hf_config.layer_norm_epsilon,
            make_vocab_size_divisible_by=self.make_vocab_size_divisible_by(hf_config.vocab_size),
            vocab_size=hf_config.vocab_size,
            share_embeddings_and_output_weights=getattr(hf_config, "tie_word_embeddings", False),
            seq_length=hf_config.max_position_embeddings,
            fp16=(self.dtype_from_hf(hf_config, default=torch.float32) == torch.float16),
            bf16=(self.dtype_from_hf(hf_config, default=torch.float32) == torch.bfloat16),
            fp32_residual_connection=hf_config.residual_in_fp32,
            params_dtype=self.dtype_from_hf(hf_config, default=torch.float32),
            attention_dropout=hf_config.attention_dropout,
            hidden_dropout=hf_config.hidden_dropout,
            hybrid_override_pattern=hf_config.hybrid_override_pattern,
            mamba_head_dim=hf_config.mamba_head_dim,
            mamba_num_heads=hf_config.mamba_num_heads,
            mamba_num_groups=hf_config.n_groups,
            mamba_state_dim=hf_config.ssm_state_size,
            add_qkv_bias=hf_config.attention_bias,
        )
