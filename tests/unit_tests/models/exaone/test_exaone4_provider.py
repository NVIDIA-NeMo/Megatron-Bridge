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

from megatron.core.transformer import ModuleSpec, TransformerLayer
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.mlp import MLP

from megatron.bridge.models.common.te_layers import TERowParallelLinearLayerNorm
from megatron.bridge.models.exaone.exaone4.exaone4_provider import exaone4_layer_spec
from megatron.bridge.models.gpt_provider import GPTModelProvider


class TestExaone4LayerSpec:
    """Test cases for EXAONE 4.0 custom layer spec."""

    def test_exaone4_layer_spec_uses_post_ln_modules(self):
        provider = GPTModelProvider(
            num_layers=2,
            hidden_size=128,
            ffn_hidden_size=256,
            num_attention_heads=4,
            num_query_groups=2,
        )
        spec = exaone4_layer_spec(provider)

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
