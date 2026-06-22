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

import torch.nn as nn

from megatron.bridge.models.glm_moe_dsa import (
    GLM5_DSA_INDEXER_LORA_TARGET_MODULES,
    GLM5_MLA_LORA_TARGET_MODULES,
    GLM5_MLP_LORA_TARGET_MODULES,
    GLM5_ROUTER_LORA_TARGET_MODULES,
    GLM5LoRA,
    glm5_lora_target_modules,
)
from megatron.bridge.peft.lora_layers import LinearAdapter


class _FakeIndexer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_wq_b = nn.Linear(8, 8, bias=False)
        self.linear_wk = nn.Linear(8, 8, bias=False)
        self.k_norm = nn.LayerNorm(8)
        self.linear_weights_proj = nn.Linear(8, 4, bias=False)


class _FakeCoreAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.indexer = _FakeIndexer()


class _FakeSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_q_down_proj = nn.Linear(8, 4, bias=False)
        self.linear_q_up_proj = nn.Linear(4, 8, bias=False)
        self.linear_kv_down_proj = nn.Linear(8, 4, bias=False)
        self.linear_kv_up_proj = nn.Linear(4, 8, bias=False)
        self.linear_proj = nn.Linear(8, 8, bias=False)
        self.core_attention = _FakeCoreAttention()


class _FakeMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_fc1 = nn.Linear(8, 16, bias=False)
        self.linear_fc2 = nn.Linear(16, 8, bias=False)
        self.router = nn.Linear(8, 4, bias=False)


class _FakeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attention = _FakeSelfAttention()
        self.mlp = _FakeMLP()


class _FakeDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_FakeLayer()])


class _FakeGLM5Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = _FakeDecoder()


def test_glm5_lora_target_modules_include_indexer_by_default():
    target_modules = glm5_lora_target_modules()

    assert target_modules == [
        *GLM5_MLA_LORA_TARGET_MODULES,
        *GLM5_MLP_LORA_TARGET_MODULES,
        *GLM5_DSA_INDEXER_LORA_TARGET_MODULES,
    ]
    assert "linear_wq_b" in target_modules
    assert "linear_wk" in target_modules
    assert "linear_weights_proj" in target_modules
    assert "k_norm" not in target_modules
    assert GLM5_ROUTER_LORA_TARGET_MODULES[0] not in target_modules


def test_glm5_lora_target_modules_can_include_router():
    assert glm5_lora_target_modules(include_router=True)[-1] == GLM5_ROUTER_LORA_TARGET_MODULES[0]


def test_glm5_lora_wraps_indexer_alias_targets_and_excludes_norm():
    model = _FakeGLM5Model()

    peft = GLM5LoRA(target_modules=["indexer.wq_b", "indexer.wk", "indexer.weights_proj"], dim=2, alpha=4)
    peft(model)

    indexer = model.decoder.layers[0].self_attention.core_attention.indexer
    assert isinstance(indexer.linear_wq_b, LinearAdapter)
    assert isinstance(indexer.linear_wk, LinearAdapter)
    assert isinstance(indexer.linear_weights_proj, LinearAdapter)
    assert isinstance(indexer.k_norm, nn.LayerNorm)
    assert not any(param.requires_grad for param in indexer.k_norm.parameters())


def test_glm5_lora_wraps_default_mla_mlp_and_indexer_targets():
    model = _FakeGLM5Model()

    peft = GLM5LoRA(dim=2, alpha=4)
    peft(model)

    layer = model.decoder.layers[0]
    assert isinstance(layer.self_attention.linear_q_down_proj, LinearAdapter)
    assert isinstance(layer.self_attention.linear_q_up_proj, LinearAdapter)
    assert isinstance(layer.self_attention.linear_kv_down_proj, LinearAdapter)
    assert isinstance(layer.self_attention.linear_kv_up_proj, LinearAdapter)
    assert isinstance(layer.self_attention.linear_proj, LinearAdapter)
    assert isinstance(layer.self_attention.core_attention.indexer.linear_wq_b, LinearAdapter)
    assert isinstance(layer.self_attention.core_attention.indexer.linear_wk, LinearAdapter)
    assert isinstance(layer.self_attention.core_attention.indexer.linear_weights_proj, LinearAdapter)
    assert isinstance(layer.mlp.linear_fc1, LinearAdapter)
    assert isinstance(layer.mlp.linear_fc2, LinearAdapter)
    assert isinstance(layer.mlp.router, nn.Linear)
