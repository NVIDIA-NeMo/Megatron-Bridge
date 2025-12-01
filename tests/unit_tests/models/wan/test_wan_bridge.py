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

import importlib
import sys
import types
from unittest.mock import Mock

import pytest
import torch

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    KVMapping,
    QKVMapping,
    ReplicatedMapping,
)


@pytest.fixture
def wan_bridge_module(monkeypatch):
    """
    Prepare dummy external modules so that importing the WAN bridge does not require
    installing heavy dependencies (diffusers, dfm).
    """
    # 1) Mock diffusers with WanTransformer3DModel
    diffusers_mod = types.ModuleType("diffusers")

    class DummyWanTransformer3DModel:
        pass

    setattr(diffusers_mod, "WanTransformer3DModel", DummyWanTransformer3DModel)
    monkeypatch.setitem(sys.modules, "diffusers", diffusers_mod)

    # 2) Mock dfm provider/model module tree expected by the bridge
    # Create package structure: dfm.src.megatron.model.wan.{wan_model, wan_provider}
    pkgs = [
        "dfm",
        "dfm.src",
        "dfm.src.megatron",
        "dfm.src.megatron.model",
        "dfm.src.megatron.model.wan",
    ]
    for name in pkgs:
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))

    # Submodules with classes used by the bridge
    wan_model_mod = types.ModuleType("dfm.src.megatron.model.wan.wan_model")
    wan_provider_mod = types.ModuleType("dfm.src.megatron.model.wan.wan_provider")

    class DummyWanModel:
        pass

    class DummyWanModelProvider:
        def __init__(self, **kwargs):
            # Store all kwargs as attributes for assertions
            for k, v in kwargs.items():
                setattr(self, k, v)

    setattr(wan_model_mod, "WanModel", DummyWanModel)
    setattr(wan_provider_mod, "WanModelProvider", DummyWanModelProvider)
    monkeypatch.setitem(sys.modules, "dfm.src.megatron.model.wan.wan_model", wan_model_mod)
    monkeypatch.setitem(sys.modules, "dfm.src.megatron.model.wan.wan_provider", wan_provider_mod)

    # Import the module under test after mocks are in place
    module = importlib.import_module("megatron.bridge.models.wan.wan_bridge")
    return module


@pytest.fixture
def WanBridge(wan_bridge_module):
    return wan_bridge_module.WanBridge


@pytest.fixture
def mock_hf_pretrained():
    # Minimal config carrying fields consumed by provider_bridge
    cfg = types.SimpleNamespace(
        num_layers=12,
        num_attention_heads=8,
        attention_head_dim=64,
        in_channels=4,
        out_channels=4,
        text_dim=1024,
        patch_size=(2, 4, 4),  # (temporal, spatial, spatial)
        eps=1e-5,
        ffn_dim=4096,
        freq_dim=128,
    )
    m = Mock()
    m.config = cfg
    return m


def test_registration(WanBridge):
    assert issubclass(WanBridge, MegatronModelBridge)


def test_provider_bridge_basic(WanBridge, mock_hf_pretrained):
    bridge = WanBridge()
    provider = bridge.provider_bridge(mock_hf_pretrained)

    # Validate that the provider instance was created and fields mapped correctly
    assert provider.num_layers == mock_hf_pretrained.config.num_layers
    assert (
        provider.hidden_size
        == mock_hf_pretrained.config.num_attention_heads * mock_hf_pretrained.config.attention_head_dim
    )
    assert provider.kv_channels == mock_hf_pretrained.config.attention_head_dim
    assert provider.num_query_groups == mock_hf_pretrained.config.num_attention_heads
    assert (
        provider.crossattn_emb_size
        == mock_hf_pretrained.config.num_attention_heads * mock_hf_pretrained.config.attention_head_dim
    )
    assert provider.ffn_hidden_size == mock_hf_pretrained.config.ffn_dim
    assert provider.num_attention_heads == mock_hf_pretrained.config.num_attention_heads
    assert provider.in_channels == mock_hf_pretrained.config.in_channels
    assert provider.out_channels == mock_hf_pretrained.config.out_channels
    assert provider.text_dim == mock_hf_pretrained.config.text_dim
    assert provider.patch_spatial == mock_hf_pretrained.config.patch_size[1]
    assert provider.patch_temporal == mock_hf_pretrained.config.patch_size[0]
    assert provider.layernorm_epsilon == mock_hf_pretrained.config.eps
    assert provider.hidden_dropout == 0
    assert provider.attention_dropout == 0
    assert provider.use_cpu_initialization is True
    assert provider.freq_dim == mock_hf_pretrained.config.freq_dim
    # Fixed dtype handling in WAN bridge
    assert provider.bf16 is False
    assert provider.params_dtype == torch.float32


class TestWanBridgeMappingRegistry:
    def test_mapping_registry_returns_correct_type(self, WanBridge):
        registry = WanBridge().mapping_registry()
        assert isinstance(registry, MegatronMappingRegistry)
        assert len(registry.mappings) > 0

    def test_contains_expected_qkv_and_kv_mappings(self, WanBridge):
        registry = WanBridge().mapping_registry()
        mappings = registry.mappings

        has_qkv_weight = any(
            isinstance(m, QKVMapping)
            and m.megatron_param == "decoder.layers.*.full_self_attention.linear_qkv.weight"
            and isinstance(m.hf_param, dict)
            and m.hf_param.get("q") == "blocks.*.attn1.to_q.weight"
            and m.hf_param.get("k") == "blocks.*.attn1.to_k.weight"
            and m.hf_param.get("v") == "blocks.*.attn1.to_v.weight"
            for m in mappings
        )
        has_qkv_bias = any(
            isinstance(m, QKVMapping)
            and m.megatron_param == "decoder.layers.*.full_self_attention.linear_qkv.bias"
            and isinstance(m.hf_param, dict)
            and m.hf_param.get("q") == "blocks.*.attn1.to_q.bias"
            and m.hf_param.get("k") == "blocks.*.attn1.to_k.bias"
            and m.hf_param.get("v") == "blocks.*.attn1.to_v.bias"
            for m in mappings
        )
        has_kv_weight = any(
            isinstance(m, KVMapping)
            and m.megatron_param == "decoder.layers.*.cross_attention.linear_kv.weight"
            and isinstance(m.hf_param, dict)
            and m.hf_param.get("k") == "blocks.*.attn2.to_k.weight"
            and m.hf_param.get("v") == "blocks.*.attn2.to_v.weight"
            for m in mappings
        )
        has_kv_bias = any(
            isinstance(m, KVMapping)
            and m.megatron_param == "decoder.layers.*.cross_attention.linear_kv.bias"
            and isinstance(m.hf_param, dict)
            and m.hf_param.get("k") == "blocks.*.attn2.to_k.bias"
            and m.hf_param.get("v") == "blocks.*.attn2.to_v.bias"
            for m in mappings
        )

        assert has_qkv_weight, "Missing QKVMapping for self-attention weights"
        assert has_qkv_bias, "Missing QKVMapping for self-attention bias"
        assert has_kv_weight, "Missing KVMapping for cross-attention weights"
        assert has_kv_bias, "Missing KVMapping for cross-attention bias"

    def test_special_replicated_mappings_are_not_automappings(self, WanBridge):
        registry = WanBridge().mapping_registry()
        mappings = registry.mappings

        # These HF params should use the custom ReplicatedMapping subclass, not AutoMapping
        special_hf_params = {"scale_shift_table", "blocks.*.scale_shift_table", "proj_out.weight", "proj_out.bias"}

        found = {k: False for k in special_hf_params}
        for m in mappings:
            hf = getattr(m, "hf_param", None)
            if isinstance(hf, str) and hf in special_hf_params:
                assert isinstance(m, ReplicatedMapping), f"{hf} should use a ReplicatedMapping"
                assert not isinstance(m, AutoMapping), f"{hf} should not be an AutoMapping"
                found[hf] = True

        # Ensure all expected special params were present
        assert all(found.values()), f"Missing special replicated mappings: {[k for k, v in found.items() if not v]}"
