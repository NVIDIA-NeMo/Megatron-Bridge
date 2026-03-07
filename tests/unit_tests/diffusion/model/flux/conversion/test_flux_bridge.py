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
import torch

from megatron.bridge.diffusion.models.flux.conversion import flux_bridge as flux_bridge_module


pytestmark = [pytest.mark.unit]


def _make_cfg(
    *,
    in_channels=64,
    patch_size=1,
    num_layers=19,
    num_single_layers=38,
    num_attention_heads=24,
    attention_head_dim=128,
    pooled_projection_dim=768,
    guidance_embeds=False,
    axes_dims_rope=(16, 56, 56),
):
    cfg = types.SimpleNamespace()
    cfg.in_channels = in_channels
    cfg.patch_size = patch_size
    cfg.num_layers = num_layers
    cfg.num_single_layers = num_single_layers
    cfg.num_attention_heads = num_attention_heads
    cfg.attention_head_dim = attention_head_dim
    cfg.pooled_projection_dim = pooled_projection_dim
    cfg.guidance_embeds = guidance_embeds
    cfg.axes_dims_rope = axes_dims_rope
    return cfg


def test_provider_bridge_constructs_provider_with_expected_fields():
    class DummyHF:
        def __init__(self, cfg):
            self.config = cfg

    cfg = _make_cfg()
    bridge = flux_bridge_module.FluxBridge()
    provider = bridge.provider_bridge(DummyHF(cfg))

    # Basic sanity: returned type and a few key fields
    assert provider is not None
    assert provider.num_joint_layers == cfg.num_layers
    assert provider.num_single_layers == cfg.num_single_layers
    assert provider.num_attention_heads == cfg.num_attention_heads

    # kv_channels equals per-head dim
    assert getattr(provider, "kv_channels") == cfg.attention_head_dim

    # num_query_groups equals num_attention_heads
    assert provider.num_query_groups == cfg.num_attention_heads

    # passthrough fields
    assert provider.in_channels == cfg.in_channels
    assert provider.patch_size == cfg.patch_size
    assert provider.vec_in_dim == cfg.pooled_projection_dim
    assert provider.guidance_embed == cfg.guidance_embeds
    assert provider.axes_dims_rope == cfg.axes_dims_rope

    # bf16 and params_dtype set by bridge
    assert provider.bf16 is False
    assert provider.params_dtype == torch.float32

    # hidden_size stored on bridge instance
    assert bridge.hidden_size == provider.hidden_size


def test_mapping_registry_registers_module_types_and_builds_mappings(monkeypatch):
    calls_register_module_type = []

    def fake_register_module_type(name, parallelism):
        calls_register_module_type.append((name, parallelism))

    constructed_registry_args = {}

    class FakeRegistry:
        def __init__(self, *mappings):
            constructed_registry_args["mappings"] = mappings

    monkeypatch.setattr(
        flux_bridge_module.AutoMapping, "register_module_type", staticmethod(fake_register_module_type)
    )
    monkeypatch.setattr(flux_bridge_module, "MegatronMappingRegistry", FakeRegistry)

    registry = flux_bridge_module.FluxBridge().mapping_registry()

    # Verify module type registrations
    assert ("Linear", "replicated") in calls_register_module_type

    # We replaced the real registry with FakeRegistry; the function should return that instance
    assert isinstance(registry, FakeRegistry)
    mappings = constructed_registry_args["mappings"]

    # Ensure we have a reasonable number of mappings
    assert len(mappings) >= 20

    # Expect at least one AutoMapping, one QKVMapping, one SplitRowParallelMapping
    has_auto = any(m.__class__.__name__ == "AutoMapping" for m in mappings)
    has_qkv = any(m.__class__.__name__ == "QKVMapping" for m in mappings)
    has_split_row = any(m.__class__.__name__ == "SplitRowParallelMapping" for m in mappings)
    assert has_auto and has_qkv and has_split_row


def test_maybe_modify_loaded_hf_weight_with_weight_1():
    """Test that weight_1 suffix correctly slices the second half of weight tensor"""
    bridge = flux_bridge_module.FluxBridge()
    bridge.hidden_size = 100

    # Create a dummy weight tensor
    dummy_weight = torch.randn(200, 300)
    hf_state_dict = {"single_transformer_blocks.0.proj_out.weight": dummy_weight}

    # Test weight_1 suffix (should get second half)
    result = bridge.maybe_modify_loaded_hf_weight("single_transformer_blocks.0.proj_out.weight_1", hf_state_dict)

    expected = dummy_weight[:, 100:]
    assert torch.equal(result, expected)
    assert result.shape == (200, 200)


def test_maybe_modify_loaded_hf_weight_with_weight_2():
    """Test that weight_2 suffix correctly slices the first half of weight tensor"""
    bridge = flux_bridge_module.FluxBridge()
    bridge.hidden_size = 100

    # Create a dummy weight tensor
    dummy_weight = torch.randn(200, 300)
    hf_state_dict = {"single_transformer_blocks.0.proj_out.weight": dummy_weight}

    # Test weight_2 suffix (should get first half)
    result = bridge.maybe_modify_loaded_hf_weight("single_transformer_blocks.0.proj_out.weight_2", hf_state_dict)

    expected = dummy_weight[:, :100]
    assert torch.equal(result, expected)
    assert result.shape == (200, 100)


def test_maybe_modify_loaded_hf_weight_normal_param():
    """Test that normal parameter names are passed through unchanged"""
    bridge = flux_bridge_module.FluxBridge()
    bridge.hidden_size = 100

    # Create a dummy weight tensor
    dummy_weight = torch.randn(200, 300)
    hf_state_dict = {"norm_out.linear.weight": dummy_weight}

    # Test normal parameter (no modification)
    result = bridge.maybe_modify_loaded_hf_weight("norm_out.linear.weight", hf_state_dict)

    assert torch.equal(result, dummy_weight)


def test_maybe_modify_loaded_hf_weight_with_dict_param():
    """Test that dictionary parameters are handled correctly"""
    bridge = flux_bridge_module.FluxBridge()

    # Create dummy weight tensors
    weight1 = torch.randn(100, 200)
    weight2 = torch.randn(100, 200)
    hf_state_dict = {
        "transformer_blocks.0.attn.to_q.weight": weight1,
        "transformer_blocks.0.attn.to_k.weight": weight2,
    }

    # Test dictionary parameter
    param_dict = {
        "q": "transformer_blocks.0.attn.to_q.weight",
        "k": "transformer_blocks.0.attn.to_k.weight",
    }
    result = bridge.maybe_modify_loaded_hf_weight(param_dict, hf_state_dict)

    assert isinstance(result, dict)
    assert torch.equal(result["q"], weight1)
    assert torch.equal(result["k"], weight2)


def test_split_row_parallel_mapping_has_allow_hf_name_mismatch():
    """Test that SplitRowParallelMapping has allow_hf_name_mismatch set to True"""
    mapping = flux_bridge_module.SplitRowParallelMapping(
        megatron_param="single_blocks.*.mlp.linear_fc2.weight",
        hf_param="single_transformer_blocks.*.proj_out.weight_1",
    )

    assert mapping.allow_hf_name_mismatch is True
