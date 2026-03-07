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

import types

import pytest

from megatron.bridge.diffusion.models.wan.conversion import wan_bridge as wan_bridge_module


pytestmark = [pytest.mark.unit]


def _make_cfg(
    *,
    num_layers=4,
    num_attention_heads=8,
    attention_head_dim=64,
    ffn_dim=1024,
    in_channels=16,
    out_channels=16,
    text_dim=4096,
    patch_size=(1, 2),
    freq_dim=256,
    eps=1e-6,
):
    cfg = types.SimpleNamespace()
    cfg.num_layers = num_layers
    cfg.num_attention_heads = num_attention_heads
    cfg.attention_head_dim = attention_head_dim
    cfg.ffn_dim = ffn_dim
    cfg.in_channels = in_channels
    cfg.out_channels = out_channels
    cfg.text_dim = text_dim
    cfg.patch_size = patch_size
    cfg.freq_dim = freq_dim
    cfg.eps = eps
    return cfg


def test_provider_bridge_constructs_provider_with_expected_fields():
    class DummyHF:
        def __init__(self, cfg):
            self.config = cfg

    cfg = _make_cfg()
    bridge = wan_bridge_module.WanBridge()
    provider = bridge.provider_bridge(DummyHF(cfg))

    # Basic sanity: returned type and a few key fields
    assert provider is not None
    assert provider.num_layers == cfg.num_layers
    # hidden_size and crossattn_emb_size computed from heads and head_dim
    expected_hsize = cfg.num_attention_heads * cfg.attention_head_dim
    assert provider.hidden_size == expected_hsize
    assert provider.crossattn_emb_size == expected_hsize
    # kv_channels equals per-head dim
    assert getattr(provider, "kv_channels") == cfg.attention_head_dim
    # patch sizes split into temporal/spatial
    assert provider.patch_temporal == cfg.patch_size[0]
    assert provider.patch_spatial == cfg.patch_size[1]
    # passthrough fields
    assert provider.in_channels == cfg.in_channels
    assert provider.out_channels == cfg.out_channels
    assert provider.text_dim == cfg.text_dim
    assert provider.freq_dim == cfg.freq_dim
    assert provider.layernorm_epsilon == cfg.eps
    # defaults enforced by bridge
    assert provider.hidden_dropout == 0
    assert provider.attention_dropout == 0


def test_mapping_registry_registers_module_types_and_builds_mappings(monkeypatch):
    calls_register_module_type = []

    def fake_register_module_type(name, parallelism):
        calls_register_module_type.append((name, parallelism))

    constructed_registry_args = {}

    class FakeRegistry:
        def __init__(self, *mappings):
            constructed_registry_args["mappings"] = mappings

    monkeypatch.setattr(wan_bridge_module.AutoMapping, "register_module_type", staticmethod(fake_register_module_type))
    monkeypatch.setattr(wan_bridge_module, "MegatronMappingRegistry", FakeRegistry)

    registry = wan_bridge_module.WanBridge().mapping_registry()

    # Verify module type registrations
    assert ("Linear", "replicated") in calls_register_module_type
    assert ("Conv3d", "replicated") in calls_register_module_type
    assert ("WanAdaLN", "replicated") in calls_register_module_type
    assert ("Head", "replicated") in calls_register_module_type

    # We replaced the real registry with FakeRegistry; the function should return that instance
    assert isinstance(registry, FakeRegistry)
    mappings = constructed_registry_args["mappings"]
    # Ensure we have a reasonable number of mappings and a mix of kinds
    assert len(mappings) >= 10
    # Expect at least one AutoMapping, one KVMapping, one QKVMapping
    has_auto = any(m.__class__.__name__ == "AutoMapping" for m in mappings)
    has_kv = any(m.__class__.__name__ == "KVMapping" for m in mappings)
    has_qkv = any(m.__class__.__name__ == "QKVMapping" for m in mappings)
    assert has_auto and has_kv and has_qkv
