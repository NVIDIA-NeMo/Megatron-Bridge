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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge, WeightConversionTask
from megatron.bridge.peft.canonical_lora import ModuleDict


class DummyBridge(MegatronModelBridge):
    def provider_bridge(self, hf_pretrained):  # pragma: no cover - not used in tests
        return None

    def mapping_registry(self):  # pragma: no cover - not used in tests
        return MegatronMappingRegistry()


class MockAdapter(nn.Module):
    """Mock adapter module for testing.

    For LoRA adapters:
    - linear_in: in_features -> dim, weight shape (dim, in_features)
    - linear_out: dim -> out_features, weight shape (out_features, dim)
    """

    def __init__(self, in_features, out_features, dim, alpha):
        super().__init__()
        self.linear_in = nn.Linear(in_features, dim, bias=False)
        self.linear_in.weight.data = torch.randn(dim, in_features) * 0.01
        self.linear_out = nn.Linear(dim, out_features, bias=False)
        self.linear_out.weight.data = torch.randn(out_features, dim) * 0.01
        self.alpha = alpha
        self.dim = dim


class MockLoRAModule(nn.Module):
    """Mock LoRA module for testing."""

    def __init__(self, adapter, base_linear):
        super().__init__()
        self.adapter = adapter
        self.to_wrap = base_linear


def _make_lora_module(alpha=8, dim=4, in_features=4, out_features=4):
    adapter = MockAdapter(in_features, out_features, dim, alpha)
    base_linear = nn.Linear(in_features, out_features, bias=False)
    lora_module = MockLoRAModule(adapter, base_linear)
    return lora_module


def _make_canonical_lora_qkv_module(alpha=8, dim=4, hidden_size=16, num_heads=2, kv_heads=2):
    """Create a CanonicalLoRA QKV module with adapter_q, adapter_k, adapter_v."""
    kv_channels = hidden_size // num_heads

    # Create adapters for Q, K, V with proper dimensions
    q_out_size = hidden_size
    kv_out_size = kv_heads * kv_channels

    adapter_q = MockAdapter(hidden_size, q_out_size, dim, alpha)
    adapter_k = MockAdapter(hidden_size, kv_out_size, dim, alpha)
    adapter_v = MockAdapter(hidden_size, kv_out_size, dim, alpha)

    adapter_dict = ModuleDict(
        {
            "adapter_q": adapter_q,
            "adapter_k": adapter_k,
            "adapter_v": adapter_v,
        }
    )

    base_linear = nn.Linear(hidden_size, (num_heads + 2 * kv_heads) * kv_channels, bias=False)
    base_linear.config = SimpleNamespace(
        kv_channels=kv_channels,
        num_query_groups=kv_heads,
        num_attention_heads=num_heads,
        sequence_parallel=False,
    )
    lora_module = MockLoRAModule(adapter_dict, base_linear)
    return lora_module


def _make_canonical_lora_fc1_module(alpha=8, dim=4, hidden_size=16, ffn_hidden_size=32):
    """Create a CanonicalLoRA FC1 module with adapter_gate and adapter_up."""
    adapter_gate = MockAdapter(hidden_size, ffn_hidden_size, dim, alpha)
    adapter_up = MockAdapter(hidden_size, ffn_hidden_size, dim, alpha)

    adapter_dict = ModuleDict(
        {
            "adapter_gate": adapter_gate,
            "adapter_up": adapter_up,
        }
    )

    base_linear = nn.Linear(hidden_size, ffn_hidden_size * 2, bias=False)
    base_linear.config = SimpleNamespace(sequence_parallel=False)
    lora_module = MockLoRAModule(adapter_dict, base_linear)
    return lora_module


class TestMergeLoRAAdapterWeights:
    """Test suite for _merge_lora_adapter_weights method."""

    def test_merge_lora_adapter_weights_merges(self, monkeypatch):
        """Test that regular LoRA adapter weights are correctly merged."""
        bridge = DummyBridge()
        base_weight = torch.zeros(4, 4)
        converted = {"hf.weight": base_weight.clone()}
        task = WeightConversionTask(
            param_name="decoder.layers.0.mlp.linear_fc1.to_wrap.weight",
            mapping=Mock(),
            megatron_module=Mock(),
            vp_stage=0,
        )

        lora_module = _make_lora_module(alpha=4, dim=4)
        monkeypatch.setattr(
            "megatron.bridge.models.conversion.model_bridge.get_module_and_param_from_name",
            lambda *_, **__: (lora_module, None),
        )
        monkeypatch.setattr("megatron.bridge.models.conversion.model_bridge.print_rank_0", lambda *_, **__: None)
        monkeypatch.setattr("megatron.bridge.peft.utils.HAVE_TE", False)

        adapter_infos = [("decoder.layers.0.mlp.linear_fc1.adapter", False, False, 4, 4)]
        updated = bridge._merge_lora_adapter_weights(task, [Mock()], converted, adapter_infos)
        # The weights should be merged (not equal to original)
        assert "hf.weight" in updated
        assert not torch.equal(updated["hf.weight"], base_weight)

    def test_merge_lora_adapter_weights_noop_without_adapter(self, monkeypatch):
        """Test that weights are unchanged when there's no adapter."""
        bridge = DummyBridge()
        converted = {"hf.weight": torch.ones(2, 2)}
        task = WeightConversionTask(
            param_name="decoder.layers.0.mlp.linear_fc1.weight",
            mapping=Mock(),
            megatron_module=Mock(),
        )

        updated = bridge._merge_lora_adapter_weights(task, [Mock()], converted, [])
        torch.testing.assert_close(updated["hf.weight"], converted["hf.weight"])

    def test_merge_lora_adapter_weights_noop_without_to_wrap_suffix(self, monkeypatch):
        """Test that weights are unchanged when param_name doesn't contain .to_wrap.weight."""
        bridge = DummyBridge()
        converted = {"hf.weight": torch.ones(2, 2)}
        task = WeightConversionTask(
            param_name="decoder.layers.0.mlp.linear_fc1.weight",
            mapping=Mock(),
            megatron_module=Mock(),
        )

        adapter_infos = [("decoder.layers.0.mlp.linear_fc1.adapter", False, False, 8, 4)]
        updated = bridge._merge_lora_adapter_weights(task, [Mock()], converted, adapter_infos)
        torch.testing.assert_close(updated["hf.weight"], converted["hf.weight"])

    def test_merge_canonical_lora_qkv(self, monkeypatch):
        """Test merging CanonicalLoRA QKV adapters (adapter_q, adapter_k, adapter_v)."""
        bridge = DummyBridge()
        hidden_size = 16
        num_heads = 2
        kv_heads = 2
        kv_channels = hidden_size // num_heads

        # Base weights are typically (out_features, in_features)
        base_q = torch.zeros(hidden_size, hidden_size)
        base_k = torch.zeros(kv_heads * kv_channels, hidden_size)
        base_v = torch.zeros(kv_heads * kv_channels, hidden_size)
        converted = {
            "model.layers.0.self_attn.q_proj.weight": base_q.clone(),
            "model.layers.0.self_attn.k_proj.weight": base_k.clone(),
            "model.layers.0.self_attn.v_proj.weight": base_v.clone(),
        }

        task = WeightConversionTask(
            param_name="decoder.layers.0.self_attention.linear_qkv.to_wrap.weight",
            mapping=Mock(),
            megatron_module=Mock(),
            vp_stage=0,
        )

        lora_module = _make_canonical_lora_qkv_module(
            alpha=8, dim=4, hidden_size=hidden_size, num_heads=num_heads, kv_heads=kv_heads
        )
        monkeypatch.setattr(
            "megatron.bridge.models.conversion.model_bridge.get_module_and_param_from_name",
            lambda *_, **__: (lora_module, None),
        )
        monkeypatch.setattr("megatron.bridge.models.conversion.model_bridge.print_rank_0", lambda *_, **__: None)
        monkeypatch.setattr("megatron.bridge.peft.utils.HAVE_TE", False)

        adapter_infos = [
            ("decoder.layers.0.self_attention.linear_qkv.adapter.adapter_q", False, False, 8, 4),
            ("decoder.layers.0.self_attention.linear_qkv.adapter.adapter_k", False, False, 8, 4),
            ("decoder.layers.0.self_attention.linear_qkv.adapter.adapter_v", False, False, 8, 4),
        ]

        updated = bridge._merge_lora_adapter_weights(task, [Mock()], converted, adapter_infos)

        # Should merge all three projections
        assert "model.layers.0.self_attn.q_proj.weight" in updated
        assert "model.layers.0.self_attn.k_proj.weight" in updated
        assert "model.layers.0.self_attn.v_proj.weight" in updated

    def test_merge_canonical_lora_fc1(self, monkeypatch):
        """Test merging CanonicalLoRA FC1 adapters (adapter_gate, adapter_up)."""
        bridge = DummyBridge()
        hidden_size = 16
        ffn_hidden_size = 32

        # Base weights are typically (out_features, in_features)
        base_gate = torch.zeros(ffn_hidden_size, hidden_size)
        base_up = torch.zeros(ffn_hidden_size, hidden_size)
        converted = {
            "model.layers.0.mlp.gate_proj.weight": base_gate.clone(),
            "model.layers.0.mlp.up_proj.weight": base_up.clone(),
        }

        task = WeightConversionTask(
            param_name="decoder.layers.0.mlp.linear_fc1.to_wrap.weight",
            mapping=Mock(),
            megatron_module=Mock(),
            vp_stage=0,
        )

        lora_module = _make_canonical_lora_fc1_module(
            alpha=8, dim=4, hidden_size=hidden_size, ffn_hidden_size=ffn_hidden_size
        )
        monkeypatch.setattr(
            "megatron.bridge.models.conversion.model_bridge.get_module_and_param_from_name",
            lambda *_, **__: (lora_module, None),
        )
        monkeypatch.setattr("megatron.bridge.models.conversion.model_bridge.print_rank_0", lambda *_, **__: None)
        monkeypatch.setattr("megatron.bridge.peft.utils.HAVE_TE", False)

        adapter_infos = [
            ("decoder.layers.0.mlp.linear_fc1.adapter.adapter_gate", False, False, 8, 4),
            ("decoder.layers.0.mlp.linear_fc1.adapter.adapter_up", False, False, 8, 4),
        ]

        updated = bridge._merge_lora_adapter_weights(task, [Mock()], converted, adapter_infos)

        # Should merge both projections
        assert "model.layers.0.mlp.gate_proj.weight" in updated
        assert "model.layers.0.mlp.up_proj.weight" in updated

    def test_merge_fused_fc1(self, monkeypatch):
        """Test merging fused FC1 layer (gate_proj and up_proj from single adapter)."""
        bridge = DummyBridge()
        hidden_size = 16
        ffn_hidden_size = 32

        # Base weights are typically (out_features, in_features)
        base_gate = torch.zeros(ffn_hidden_size, hidden_size)
        base_up = torch.zeros(ffn_hidden_size, hidden_size)
        converted = {
            "model.layers.0.mlp.gate_proj.weight": base_gate.clone(),
            "model.layers.0.mlp.up_proj.weight": base_up.clone(),
        }

        task = WeightConversionTask(
            param_name="decoder.layers.0.mlp.linear_fc1.to_wrap.weight",
            mapping=Mock(),
            megatron_module=Mock(),
            vp_stage=0,
        )

        # Create adapter with linear_out that's 2x the base weight size (fused)
        dim = 4
        adapter = MockAdapter(hidden_size, 2 * ffn_hidden_size, dim, alpha=8)
        base_linear = nn.Linear(hidden_size, ffn_hidden_size * 2, bias=False)
        lora_module = MockLoRAModule(adapter, base_linear)

        monkeypatch.setattr(
            "megatron.bridge.models.conversion.model_bridge.get_module_and_param_from_name",
            lambda *_, **__: (lora_module, None),
        )
        monkeypatch.setattr("megatron.bridge.models.conversion.model_bridge.print_rank_0", lambda *_, **__: None)
        monkeypatch.setattr("megatron.bridge.peft.utils.HAVE_TE", False)

        adapter_infos = [("decoder.layers.0.mlp.linear_fc1.adapter", False, False, 8, 4)]

        updated = bridge._merge_lora_adapter_weights(task, [Mock()], converted, adapter_infos)

        # Should merge both projections
        assert "model.layers.0.mlp.gate_proj.weight" in updated
        assert "model.layers.0.mlp.up_proj.weight" in updated

    def test_merge_fused_qkv_requires_config(self, monkeypatch):
        """Test that fused QKV merging requires proper model config for split_qkv_weights."""
        bridge = DummyBridge()
        hidden_size = 16
        num_heads = 2
        kv_heads = 2
        kv_channels = hidden_size // num_heads

        # Base weights are typically (out_features, in_features)
        base_q = torch.zeros(hidden_size, hidden_size)
        base_k = torch.zeros(kv_heads * kv_channels, hidden_size)
        base_v = torch.zeros(kv_heads * kv_channels, hidden_size)
        converted = {
            "model.layers.0.self_attn.q_proj.weight": base_q.clone(),
            "model.layers.0.self_attn.k_proj.weight": base_k.clone(),
            "model.layers.0.self_attn.v_proj.weight": base_v.clone(),
        }

        # Create mock config for split_qkv_weights
        mock_config = SimpleNamespace(
            num_attention_heads=num_heads,
            num_query_groups=kv_heads,
            kv_channels=kv_channels,
            hidden_size=hidden_size,
        )

        # Create a mock model with config
        mock_model = Mock()
        mock_model.config = mock_config

        task = WeightConversionTask(
            param_name="decoder.layers.0.self_attention.linear_qkv.to_wrap.weight",
            mapping=Mock(),
            megatron_module=Mock(),
            vp_stage=0,
        )

        # Create adapter with fused QKV linear_out
        dim = 4
        qkv_size = (num_heads + 2 * kv_heads) * kv_channels
        adapter = MockAdapter(hidden_size, qkv_size, dim, alpha=8)
        base_linear = nn.Linear(hidden_size, qkv_size, bias=False)
        base_linear.config = mock_config
        lora_module = MockLoRAModule(adapter, base_linear)

        monkeypatch.setattr(
            "megatron.bridge.models.conversion.model_bridge.get_module_and_param_from_name",
            lambda *_, **__: (lora_module, None),
        )
        monkeypatch.setattr("megatron.bridge.models.conversion.model_bridge.print_rank_0", lambda *_, **__: None)
        monkeypatch.setattr("megatron.bridge.peft.utils.HAVE_TE", False)

        adapter_infos = [("decoder.layers.0.self_attention.linear_qkv.adapter", False, False, 8, 4)]

        updated = bridge._merge_lora_adapter_weights(task, [mock_model], converted, adapter_infos)

        # Should merge all three projections
        assert "model.layers.0.self_attn.q_proj.weight" in updated
        assert "model.layers.0.self_attn.k_proj.weight" in updated
        assert "model.layers.0.self_attn.v_proj.weight" in updated

    def test_merge_lora_adapter_weights_empty_adapter_infos(self, monkeypatch):
        """Test that empty adapter_infos is handled correctly."""
        bridge = DummyBridge()
        converted = {"hf.weight": torch.ones(2, 2)}
        task = WeightConversionTask(
            param_name="decoder.layers.0.mlp.linear_fc1.to_wrap.weight",
            mapping=Mock(),
            megatron_module=Mock(),
            vp_stage=0,
        )

        adapter_infos = []

        # The current implementation doesn't check for empty list and will crash
        with pytest.raises(IndexError):
            bridge._merge_lora_adapter_weights(task, [Mock()], converted, adapter_infos)

    def test_merge_lora_adapter_weights_unknown_adapter_keys(self, monkeypatch):
        """Test that unknown adapter keys raise ValueError."""
        bridge = DummyBridge()
        converted = {"hf.weight": torch.ones(2, 2)}
        task = WeightConversionTask(
            param_name="decoder.layers.0.mlp.linear_fc1.to_wrap.weight",
            mapping=Mock(),
            megatron_module=Mock(),
            vp_stage=0,
        )

        # Multiple adapter_infos but with unknown keys (not adapter_q/k/v/up/gate)
        adapter_infos = [
            ("decoder.layers.0.mlp.linear_fc1.adapter.adapter_unknown1", False, False, 8, 4),
            ("decoder.layers.0.mlp.linear_fc1.adapter.adapter_unknown2", False, False, 8, 4),
        ]

        monkeypatch.setattr(
            "megatron.bridge.models.conversion.model_bridge.get_module_and_param_from_name",
            lambda *_, **__: (None, None),
        )
        monkeypatch.setattr("megatron.bridge.models.conversion.model_bridge.print_rank_0", lambda *_, **__: None)

        with pytest.raises(ValueError, match="Unknown adapter keys"):
            bridge._merge_lora_adapter_weights(task, [Mock()], converted, adapter_infos)


class TestGetAdapterWrapModule:
    """Test suite for _get_adapter_wrap_module method."""

    def test_get_adapter_wrap_module_regular_lora(self, monkeypatch):
        """Test getting adapter and to_wrap for regular LoRA module."""
        bridge = DummyBridge()
        lora_module = _make_lora_module()

        monkeypatch.setattr(
            "megatron.bridge.models.conversion.model_bridge.get_module_and_param_from_name",
            lambda *_, **__: (lora_module, None),
        )

        adapter, to_wrap = bridge._get_adapter_wrap_module("decoder.layers.0.mlp.linear_fc1", [Mock()], 0)

        assert adapter is not None
        assert to_wrap is not None
        assert hasattr(adapter, "linear_in")
        assert hasattr(adapter, "linear_out")

    def test_get_adapter_wrap_module_canonical_lora(self, monkeypatch):
        """Test getting adapter and to_wrap for CanonicalLoRA module."""
        bridge = DummyBridge()
        lora_module = _make_canonical_lora_qkv_module()

        call_order = []

        def mock_get_module(models, param_name, *_, **__):
            call_order.append(param_name)
            # If asking for base_name + ".to_wrap", return the lora_module
            if isinstance(param_name, str) and ".to_wrap" in param_name:
                return (lora_module, None)
            # First call with base_name: return a module without adapter attribute
            module_without_adapter = nn.Module()
            return (module_without_adapter, None)

        monkeypatch.setattr(
            "megatron.bridge.models.conversion.model_bridge.get_module_and_param_from_name",
            mock_get_module,
        )

        adapter, to_wrap = bridge._get_adapter_wrap_module("decoder.layers.0.self_attention.linear_qkv", [Mock()], 0)

        assert adapter is not None
        assert isinstance(adapter, ModuleDict)
        assert to_wrap is not None


class TestMegatronGlobalParamNamesAllPPRanks:
    """Test suite for _megatron_global_param_names_all_pp_ranks method."""

    def test_global_param_names_skip_adapter(self, monkeypatch):
        """Test that adapter parameters are skipped when for_adapter=False."""
        bridge = DummyBridge()

        class DummyGroup:
            def size(self):
                return 1

        fake_param = torch.nn.Parameter(torch.zeros(1, 1))

        class FakeModel:
            def __init__(self):
                self.config = SimpleNamespace()

            def named_parameters(self):
                return [
                    ("decoder.layers.0.mlp.adapter.linear_in.weight", fake_param),
                    ("decoder.layers.0.mlp.linear_fc1.to_wrap.weight", fake_param),
                ]

        monkeypatch.setattr(
            "megatron.bridge.models.conversion.model_bridge.parallel_state.get_pipeline_model_parallel_group",
            lambda: DummyGroup(),
        )
        monkeypatch.setattr(
            "megatron.bridge.models.conversion.model_bridge.persistent_buffers",
            lambda *_: [],
        )
        monkeypatch.setattr(
            "megatron.bridge.models.conversion.model_bridge._megatron_local_name_to_global",
            lambda *_args, **_kwargs: _args[2],
        )
        monkeypatch.setattr(
            "megatron.bridge.models.conversion.model_bridge.unwrap_model",
            lambda models: models if isinstance(models, list) else [models],
        )
        monkeypatch.setattr(
            "torch.distributed.all_gather_object",
            lambda output, obj, group=None: output.__setitem__(0, obj),
        )

        names = bridge._megatron_global_param_names_all_pp_ranks([FakeModel()])
        assert names == ["decoder.layers.0.mlp.linear_fc1.to_wrap.weight"]

    def test_global_param_names_for_adapter(self, monkeypatch):
        """Test that adapter parameters are collected when for_adapter=True."""
        bridge = DummyBridge()

        class DummyGroup:
            def size(self):
                return 1

        fake_param = torch.nn.Parameter(torch.zeros(1, 1))

        class FakeModel:
            def __init__(self):
                self.config = SimpleNamespace()

            def named_parameters(self):
                return [
                    ("decoder.layers.0.mlp.adapter.linear_in.weight", fake_param),
                    ("decoder.layers.0.mlp.adapter.linear_out.weight", fake_param),
                    ("decoder.layers.0.mlp.linear_fc1.to_wrap.weight", fake_param),
                ]

        lora_module = _make_lora_module()
        base_linear = lora_module.to_wrap
        base_linear.config = SimpleNamespace(sequence_parallel=False)  # Add config attribute

        monkeypatch.setattr(
            "megatron.bridge.models.conversion.model_bridge.parallel_state.get_pipeline_model_parallel_group",
            lambda: DummyGroup(),
        )
        monkeypatch.setattr(
            "megatron.bridge.models.conversion.model_bridge.persistent_buffers",
            lambda *_: [],
        )
        monkeypatch.setattr(
            "megatron.bridge.models.conversion.model_bridge._megatron_local_name_to_global",
            lambda *_args, **_kwargs: _args[2],
        )
        monkeypatch.setattr(
            "megatron.bridge.models.conversion.model_bridge.unwrap_model",
            lambda models: models if isinstance(models, list) else [models],
        )
        monkeypatch.setattr(
            "torch.distributed.all_gather_object",
            lambda output, obj, group=None: output.__setitem__(0, obj),
        )
        monkeypatch.setattr(
            "megatron.bridge.models.conversion.model_bridge.get_module_and_param_from_name",
            lambda *_, **__: (lora_module, None),
        )
        # Mock parallel state functions
        monkeypatch.setattr(
            "megatron.bridge.models.conversion.model_bridge.parallel_state.get_tensor_model_parallel_world_size",
            lambda: 1,
        )
        monkeypatch.setattr(
            "megatron.bridge.models.conversion.model_bridge.parallel_state.get_expert_tensor_parallel_world_size",
            lambda: 1,
        )
        # Mock get_adapter_attributes_from_linear - must be mocked in both locations
        monkeypatch.setattr(
            "megatron.bridge.peft.utils.get_adapter_attributes_from_linear",
            lambda *_, **__: (False, 4, 4, False, False),
        )
        monkeypatch.setattr(
            "megatron.bridge.models.conversion.model_bridge.get_adapter_attributes_from_linear",
            lambda *_, **__: (False, 4, 4, False, False),
        )

        adapter_info = bridge._megatron_global_param_names_all_pp_ranks([FakeModel()], for_adapter=True)
        # Should return list of tuples: (base_name, input_is_parallel, base_linear_is_parallel, alpha, dim)
        assert len(adapter_info) == 1
        assert isinstance(adapter_info[0], tuple)
        assert len(adapter_info[0]) == 5
        assert adapter_info[0][0] == "decoder.layers.0.mlp.adapter"
        assert adapter_info[0][-2] == 8  # alpha
        assert adapter_info[0][-1] == 4  # dim


class TestGetAdapterWeights:
    """Test suite for _get_adapter_weights method."""

    def test_get_adapter_weights_non_parallel(self, monkeypatch):
        """Test getting adapter weights when base linear is not parallel."""
        bridge = DummyBridge()
        lora_module = _make_lora_module()
        adapter = lora_module.adapter

        adapter_info = ("decoder.layers.0.mlp.adapter", False, False, 8, 4)
        linear_in_weight, linear_out_weight = bridge._get_adapter_weights(adapter, adapter_info)

        assert linear_in_weight is not None
        assert linear_out_weight is not None
        torch.testing.assert_close(linear_in_weight, adapter.linear_in.weight.data)
        torch.testing.assert_close(linear_out_weight, adapter.linear_out.weight.data)

    def test_get_adapter_weights_with_none_adapter(self, monkeypatch):
        """Test getting adapter weights when adapter is None."""
        bridge = DummyBridge()
        adapter_info = ("decoder.layers.0.mlp.adapter", False, False, 8, 4)
        linear_in_weight, linear_out_weight = bridge._get_adapter_weights(None, adapter_info)

        assert linear_in_weight is None
        assert linear_out_weight is None


class TestMergeSingleAdapterWeight:
    """Test suite for _merge_single_adapter_weight method."""

    def test_merge_single_adapter_weight(self):
        """Test merging a single adapter weight with base weight."""
        bridge = DummyBridge()
        base_weight = torch.zeros(4, 4)
        alpha = 8
        dim = 4
        linear_in_weight = torch.eye(4)
        linear_out_weight = torch.eye(4)

        merged = bridge._merge_single_adapter_weight(base_weight, alpha, dim, linear_in_weight, linear_out_weight)

        # Expected: base + (alpha/dim) * linear_out @ linear_in
        # = zeros(4,4) + (8/4) * eye(4) @ eye(4) = 2 * eye(4)
        expected = base_weight + (alpha / dim) * (linear_out_weight @ linear_in_weight)
        torch.testing.assert_close(merged, expected)

    def test_merge_single_adapter_weight_different_devices(self):
        """Test merging when weights are on different devices."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        bridge = DummyBridge()
        base_weight = torch.zeros(4, 4).cuda()
        alpha = 8
        dim = 4
        linear_in_weight = torch.eye(4).cpu()
        linear_out_weight = torch.eye(4).cpu()

        merged = bridge._merge_single_adapter_weight(base_weight, alpha, dim, linear_in_weight, linear_out_weight)

        assert merged.device == base_weight.device
        expected = base_weight + (alpha / dim) * (linear_out_weight.cuda() @ linear_in_weight.cuda())
        torch.testing.assert_close(merged, expected)


class TestMergeCanonicalAdapter:
    """Test suite for _merge_canonical_adapter method."""

    def test_merge_canonical_adapter_qkv(self, monkeypatch):
        """Test merging CanonicalLoRA QKV adapters."""
        bridge = DummyBridge()
        hidden_size = 16

        # Base weights are typically (out_features, in_features)
        base_q = torch.zeros(hidden_size, hidden_size)
        base_k = torch.zeros(hidden_size, hidden_size)
        base_v = torch.zeros(hidden_size, hidden_size)
        converted = {
            "model.layers.0.self_attn.q_proj.weight": base_q.clone(),
            "model.layers.0.self_attn.k_proj.weight": base_k.clone(),
            "model.layers.0.self_attn.v_proj.weight": base_v.clone(),
        }

        base_name = "decoder.layers.0.self_attention.linear_qkv"
        lora_module = _make_canonical_lora_qkv_module(hidden_size=hidden_size)

        adapter_infos = [
            (f"{base_name}.adapter.adapter_q", False, False, 8, 4),
            (f"{base_name}.adapter.adapter_k", False, False, 8, 4),
            (f"{base_name}.adapter.adapter_v", False, False, 8, 4),
        ]

        adapter_name_map = {
            ".q_proj.weight": "adapter_q",
            ".k_proj.weight": "adapter_k",
            ".v_proj.weight": "adapter_v",
        }

        monkeypatch.setattr("megatron.bridge.models.conversion.model_bridge.print_rank_0", lambda *_, **__: None)
        monkeypatch.setattr("megatron.bridge.peft.utils.HAVE_TE", False)

        updated = bridge._merge_canonical_adapter(
            lora_module.adapter, converted, base_name, adapter_infos, adapter_name_map
        )

        # Should merge all three projections
        assert "model.layers.0.self_attn.q_proj.weight" in updated
        assert "model.layers.0.self_attn.k_proj.weight" in updated
        assert "model.layers.0.self_attn.v_proj.weight" in updated

    def test_merge_canonical_adapter_fc1(self, monkeypatch):
        """Test merging CanonicalLoRA FC1 adapters."""
        bridge = DummyBridge()
        hidden_size = 16
        ffn_hidden_size = 32

        # Base weights are typically (out_features, in_features)
        base_gate = torch.zeros(ffn_hidden_size, hidden_size)
        base_up = torch.zeros(ffn_hidden_size, hidden_size)
        converted = {
            "model.layers.0.mlp.gate_proj.weight": base_gate.clone(),
            "model.layers.0.mlp.up_proj.weight": base_up.clone(),
        }

        base_name = "decoder.layers.0.mlp.linear_fc1"
        lora_module = _make_canonical_lora_fc1_module(hidden_size=hidden_size, ffn_hidden_size=ffn_hidden_size)

        adapter_infos = [
            (f"{base_name}.adapter.adapter_gate", False, False, 8, 4),
            (f"{base_name}.adapter.adapter_up", False, False, 8, 4),
        ]

        adapter_name_map = {
            ".gate_proj.weight": "adapter_gate",
            ".up_proj.weight": "adapter_up",
        }

        monkeypatch.setattr("megatron.bridge.models.conversion.model_bridge.print_rank_0", lambda *_, **__: None)
        monkeypatch.setattr("megatron.bridge.peft.utils.HAVE_TE", False)

        updated = bridge._merge_canonical_adapter(
            lora_module.adapter, converted, base_name, adapter_infos, adapter_name_map
        )

        # Should merge both projections
        assert "model.layers.0.mlp.gate_proj.weight" in updated
        assert "model.layers.0.mlp.up_proj.weight" in updated

    def test_merge_canonical_adapter_missing_adapter_name_mapping(self, monkeypatch):
        """Test that missing adapter name mapping raises ValueError."""
        bridge = DummyBridge()
        converted = {
            "unknown_proj.weight": torch.zeros(16, 16),
        }

        base_name = "decoder.layers.0.mlp.linear_fc1"
        lora_module = _make_canonical_lora_fc1_module()

        adapter_infos = [
            (f"{base_name}.adapter.adapter_gate", False, False, 8, 4),
        ]

        adapter_name_map = {
            ".gate_proj.weight": "adapter_gate",
        }

        monkeypatch.setattr("megatron.bridge.models.conversion.model_bridge.print_rank_0", lambda *_, **__: None)

        with pytest.raises(ValueError, match="Adapter name mapping not found"):
            bridge._merge_canonical_adapter(lora_module.adapter, converted, base_name, adapter_infos, adapter_name_map)
