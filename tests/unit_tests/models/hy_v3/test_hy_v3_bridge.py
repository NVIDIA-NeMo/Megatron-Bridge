# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from types import SimpleNamespace

import pytest
import torch

from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig
from megatron.bridge.models.hy_v3.hy_v3_bridge import HYV3Bridge


pytestmark = pytest.mark.unit


def test_model_config_bridge_preserves_hy_v3_router_and_dense_layer_contract() -> None:
    hf_config = SimpleNamespace(
        num_hidden_layers=4,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=8,
        num_experts=8,
        num_experts_per_tok=2,
        vocab_size=1024,
        max_position_embeddings=4096,
        rms_norm_eps=1e-6,
        initializer_range=0.02,
        tie_word_embeddings=False,
        hidden_act="silu",
        torch_dtype="bfloat16",
        router_scaling_factor=2.5,
        num_shared_experts=2,
        first_k_dense_replace=1,
    )

    model_config = HYV3Bridge().model_config_bridge(SimpleNamespace(config=hf_config))

    assert isinstance(model_config, BridgeGPTModelConfig)
    assert model_config.transformer.moe_layer_freq == [0, 1, 1, 1]
    assert model_config.transformer.moe_router_topk_scaling_factor == 2.5
    assert model_config.transformer.moe_shared_expert_intermediate_size == 64
    assert model_config.transformer.moe_router_score_function == "sigmoid"
    assert model_config.transformer.params_dtype == torch.bfloat16
