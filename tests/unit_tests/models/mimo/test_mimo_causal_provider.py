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

from megatron.bridge.models.mimo import (
    MiMoModelProvider7B,
    MiMoModelProvider7BBase,
    MiMoModelProvider7BRL,
    MiMoModelProvider7BRL0530,
    MiMoModelProvider7BRLZero,
    MiMoModelProvider7BSFT,
)


class TestMiMoModelProvider7B:
    def test_base_defaults(self):
        provider = MiMoModelProvider7B()

        assert provider.normalization == "RMSNorm"
        assert provider.activation_func is F.silu
        assert provider.gated_linear_unit is True
        assert provider.add_bias_linear is False
        assert provider.add_qkv_bias is True
        assert provider.qk_layernorm is False
        assert provider.position_embedding_type == "rope"

        assert provider.num_layers == 36
        assert provider.hidden_size == 4096
        assert provider.ffn_hidden_size == 11008
        assert provider.num_attention_heads == 32
        assert provider.num_query_groups == 8
        assert provider.kv_channels == 128

        assert provider.seq_length == 32768
        assert provider.vocab_size == 151680
        assert provider.share_embeddings_and_output_weights is False
        assert provider.layernorm_epsilon == 1e-5
        assert provider.rotary_base == 640000.0
        assert provider.init_method_std == 0.02
        assert provider.hidden_dropout == 0.0
        assert provider.attention_dropout == 0.0

        assert provider.mtp_num_layers == 1
        assert provider.mtp_loss_scaling_factor == 0.1

        assert provider.bf16 is True
        assert provider.fp16 is False
        assert provider.params_dtype == torch.bfloat16
        assert provider.autocast_dtype == torch.bfloat16


class TestMiMoModelProviderVariants:
    def test_variant_defaults_match_7b_family(self):
        variants = [
            MiMoModelProvider7BBase(),
            MiMoModelProvider7BSFT(),
            MiMoModelProvider7BRL(),
            MiMoModelProvider7BRLZero(),
        ]

        for provider in variants:
            assert provider.num_layers == 36
            assert provider.hidden_size == 4096
            assert provider.ffn_hidden_size == 11008
            assert provider.num_attention_heads == 32
            assert provider.num_query_groups == 8
            assert provider.kv_channels == 128
            assert provider.seq_length == 32768
            assert provider.vocab_size == 151680
            assert provider.layernorm_epsilon == 1e-5
            assert provider.rotary_base == 640000.0
            assert provider.add_qkv_bias is True
            assert provider.qk_layernorm is False
            assert provider.mtp_num_layers == 1

    def test_rl_0530_long_context(self):
        provider = MiMoModelProvider7BRL0530()

        assert provider.seq_length == 65536
        assert provider.num_layers == 36
        assert provider.hidden_size == 4096
        assert provider.ffn_hidden_size == 11008
        assert provider.num_attention_heads == 32
        assert provider.num_query_groups == 8
        assert provider.kv_channels == 128
        assert provider.vocab_size == 151680
