#!/usr/bin/env python3
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

from megatron.bridge.models.gpt_oss.gpt_oss_provider import (
    GPTOSSProvider,
    GPTOSSProvider20B,
    GPTOSSProvider120B,
)


class TestGptOssProviderDefaults:
    """Test default configuration values for GPT-OSS providers."""

    def test_gpt_oss_base_defaults(self):
        # Provide minimal required fields for post-init checks via base class
        provider = GPTOSSProvider(num_layers=1, hidden_size=512, num_attention_heads=8)

        # Generic defaults
        assert provider.normalization == "RMSNorm"
        assert provider.gated_linear_unit is True
        assert provider.position_embedding_type == "yarn"
        assert provider.add_bias_linear is True
        assert provider.share_embeddings_and_output_weights is False

        # DType defaults
        assert provider.bf16 is True
        assert provider.params_dtype == torch.bfloat16

        # MoE and window attention flags
        assert provider.moe_grouped_gemm is True
        assert provider.moe_token_dispatcher_type == "alltoall"
        assert provider.softmax_type in ("vanilla", "off-by-one", "learnable")

    def test_gpt_oss_20b_defaults(self):
        provider = GPTOSSProvider20B()

        assert provider.num_layers == 24
        assert provider.num_moe_experts == 32

    def test_gpt_oss_120b_defaults(self):
        provider = GPTOSSProvider120B()

        assert provider.num_layers == 36
        assert provider.num_moe_experts == 128
