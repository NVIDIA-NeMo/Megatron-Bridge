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
import torch.nn.functional as F
from megatron.core.transformer import ModuleSpec, TransformerLayer
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.mlp import MLP

from megatron.bridge.models.common.te_layers import TERowParallelLinearLayerNorm
from megatron.bridge.models.exaone.exaone4_provider import (
    Exaone4ModelProvider,
    Exaone4ModelProvider1P2B,
    exaone4_layer_spec,
)


class TestExaone4ModelProvider:
    """Test cases for EXAONE 4.0 model providers."""

    def test_exaone4_base_defaults(self):
        provider = Exaone4ModelProvider(
            num_layers=2,
            hidden_size=128,
            ffn_hidden_size=256,
            num_attention_heads=4,
            num_query_groups=2,
        )

        assert provider.normalization == "RMSNorm"
        assert provider.activation_func == F.silu
        assert provider.gated_linear_unit is True
        assert provider.position_embedding_type == "rope"
        assert provider.add_bias_linear is False
        assert provider.add_qkv_bias is False
        assert provider.qk_layernorm is True
        assert provider.hidden_dropout == 0.0
        assert provider.attention_dropout == 0.0
        assert provider.share_embeddings_and_output_weights is True
        assert provider.rotary_percent == 1.0
        assert provider.transformer_layer_spec == exaone4_layer_spec
        assert provider.autocast_dtype == torch.bfloat16
        assert provider.params_dtype == torch.bfloat16
        assert provider.bf16 is True

    def test_exaone4_1p2b_configuration(self):
        provider = Exaone4ModelProvider1P2B()

        assert provider.num_layers == 30
        assert provider.hidden_size == 2048
        assert provider.ffn_hidden_size == 4096
        assert provider.num_attention_heads == 32
        assert provider.num_query_groups == 8
        assert provider.kv_channels == 64
        assert provider.seq_length == 65536
        assert provider.vocab_size == 102400
        assert provider.rotary_base == 1000000.0
        assert provider.layernorm_epsilon == 1e-5
        assert provider.init_method_std == 0.02
        assert provider.rope_scaling is True
        assert provider.rope_scaling_factor == 16.0
        assert provider.rope_scaling_low_freq_factor == 1.0
        assert provider.rope_scaling_high_freq_factor == 4.0
        assert provider.rope_scaling_original_max_position_embeddings == 8192

    def test_exaone4_layer_spec_uses_post_ln_modules(self):
        spec = exaone4_layer_spec(Exaone4ModelProvider1P2B())

        assert isinstance(spec, ModuleSpec)
        assert spec.module is TransformerLayer

        layer_submodules = spec.submodules
        assert isinstance(layer_submodules.self_attention, ModuleSpec)
        assert layer_submodules.self_attention.module is SelfAttention
        assert layer_submodules.self_attention.submodules.core_attention is DotProductAttention
        assert layer_submodules.self_attention.submodules.linear_proj is TERowParallelLinearLayerNorm

        assert isinstance(layer_submodules.mlp, ModuleSpec)
        assert layer_submodules.mlp.module is MLP
        assert layer_submodules.mlp.submodules.linear_fc2 is TERowParallelLinearLayerNorm

    def test_exaone4_1p2b_inherits_base_provider(self):
        provider = Exaone4ModelProvider1P2B()
        assert isinstance(provider, Exaone4ModelProvider)
