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

import datetime
import os
from unittest.mock import patch

import megatron.core.parallel_state as parallel_state
import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from megatron.core.transformer.module import MegatronModule

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.peft.canonical_lora import CanonicalLoRA, LoRALinearSplitFC1UpGate, LoRALinearSplitQKV, ModuleDict
from megatron.bridge.peft.lora_layers import LinearAdapter, LoRALinear


class SimpleModel(nn.Module):
    """Simple test model with various linear layers for testing."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(1000, 512)
        # Individual Q, K, V layers (canonical approach)
        self.linear_q = nn.Linear(512, 512)
        self.linear_k = nn.Linear(512, 512)
        self.linear_v = nn.Linear(512, 512)
        self.linear_proj = nn.Linear(512, 512)
        # Individual Up and Gate layers (canonical approach)
        self.linear_fc1_up = nn.Linear(512, 1024)
        self.linear_fc1_gate = nn.Linear(512, 1024)
        self.linear_fc2 = nn.Linear(2048, 512)
        self.output_projection = nn.Linear(512, 1000)  # Should NOT be matched
        self.layernorm = nn.LayerNorm(512)


class MockMegatronLinear(nn.Module):
    """Mock Megatron linear layer that's not nn.Linear to trigger parallel adapter path."""

    def __init__(self, in_features, out_features, kv_channels=None, num_query_groups=None):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.in_features = in_features
        self.out_features = out_features

        # Mock config
        class MockConfig:
            def __init__(self):
                self.kv_channels = kv_channels or 64
                self.num_query_groups = num_query_groups or 8
                self.sequence_parallel = False

        self.config = MockConfig()

    def forward(self, x):
        return self.linear(x), None  # Return tuple like Megatron layers


class MegatronStyleModel(nn.Module):
    """Model with Megatron-style fused layers for testing."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(1000, 512)
        # Mock Megatron-style layers with config (not nn.Linear to trigger parallel adapter path)
        self.linear_qkv = MockMegatronLinear(512, 1536, kv_channels=64, num_query_groups=8)
        self.linear_proj = MockMegatronLinear(512, 512)
        self.linear_fc1 = MockMegatronLinear(512, 2048)
        self.linear_fc2 = MockMegatronLinear(2048, 512)


class NestedModel(nn.Module):
    """Model with nested structure for testing pattern matching."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "attention": nn.ModuleDict(
                            {
                                "linear_q": nn.Linear(512, 512),
                                "linear_k": nn.Linear(512, 512),
                                "linear_v": nn.Linear(512, 512),
                                "linear_proj": nn.Linear(512, 512),
                            }
                        ),
                        "mlp": nn.ModuleDict(
                            {
                                "linear_fc1_up": nn.Linear(512, 1024),
                                "linear_fc1_gate": nn.Linear(512, 1024),
                                "linear_fc2": nn.Linear(2048, 512),
                            }
                        ),
                    }
                )
                for _ in range(2)
            ]
        )


class TestCanonicalLoRA:
    """Test suite for CanonicalLoRA PEFT implementation."""

    def test_canonical_lora_initialization(self):
        """Test CanonicalLoRA class initialization with default and custom parameters."""
        # Test default initialization
        lora = CanonicalLoRA()
        assert lora.target_modules == [
            "linear_q",
            "linear_k",
            "linear_v",
            "linear_proj",
            "linear_fc1_up",
            "linear_fc1_gate",
            "linear_fc2",
        ]
        assert lora.dim == 32
        assert lora.alpha == 32
        assert lora.dropout == 0.0
        assert lora.dropout_position == "pre"
        assert lora.lora_A_init_method == "xavier"
        assert lora.lora_B_init_method == "zero"
        assert hasattr(lora, "canonical_mapping")

        # Test custom initialization
        custom_lora = CanonicalLoRA(
            target_modules=["linear_q", "linear_k"],
            dim=16,
            alpha=16,
            dropout=0.1,
            dropout_position="post",
            lora_A_init_method="uniform",
        )
        assert custom_lora.target_modules == ["linear_q", "linear_k"]
        assert custom_lora.dim == 16
        assert custom_lora.alpha == 16
        assert custom_lora.dropout == 0.1
        assert custom_lora.dropout_position == "post"
        assert custom_lora.lora_A_init_method == "uniform"

    def test_canonical_lora_post_init_mapping(self):
        """Test the canonical mapping creation in __post_init__."""
        lora = CanonicalLoRA(target_modules=["linear_q", "linear_k", "linear_fc1_up", "linear_proj"])

        # Check that the mapping is created correctly
        expected_mapping = {
            "linear_qkv": {"linear_q", "linear_k"},
            "linear_fc1": {"linear_fc1_up"},
            "linear_proj": {"linear_proj"},
        }

        for key, expected_values in expected_mapping.items():
            assert key in lora.canonical_mapping
            assert lora.canonical_mapping[key] == expected_values

    def test_canonical_lora_invalid_targets(self):
        """Test that invalid target modules raise appropriate errors."""
        # Should raise error for fused targets
        with pytest.raises(AssertionError, match="does not support target 'linear_qkv'"):
            CanonicalLoRA(target_modules=["linear_qkv"])

        with pytest.raises(AssertionError, match="does not support target 'linear_fc1'"):
            CanonicalLoRA(target_modules=["linear_fc1"])

    def test_canonical_lora_wildcard_mapping(self):
        """Test wildcard pattern mapping in canonical LoRA."""
        lora = CanonicalLoRA(target_modules=["*.layers.0.*.linear_q", "*.layers.1.*.linear_k"])

        # Check wildcard patterns are mapped correctly
        assert "*.layers.0.*.linear_qkv" in lora.canonical_mapping
        assert "linear_q" in lora.canonical_mapping["*.layers.0.*.linear_qkv"]
        assert "*.layers.1.*.linear_qkv" in lora.canonical_mapping
        assert "linear_k" in lora.canonical_mapping["*.layers.1.*.linear_qkv"]

    def test_canonical_lora_transform_simple_model(self):
        """Test CanonicalLoRA transformation on a simple model with individual layers."""
        model = SimpleModel()
        # CanonicalLoRA with individual layers should only transform linear_proj and linear_fc2
        # since linear_q, linear_k, linear_v don't have corresponding fused layers to split
        lora = CanonicalLoRA(target_modules=["linear_proj", "linear_fc2"])

        # Apply CanonicalLoRA
        transformed_model = lora(model, training=True)

        # Check that target modules were transformed to LinearAdapter (simple nn.Linear case)
        assert isinstance(transformed_model.linear_proj, LinearAdapter)
        assert isinstance(transformed_model.linear_fc2, LinearAdapter)

        # Check that non-target modules were not transformed
        assert isinstance(transformed_model.linear_q, nn.Linear)
        assert isinstance(transformed_model.linear_k, nn.Linear)
        assert isinstance(transformed_model.linear_v, nn.Linear)
        assert isinstance(transformed_model.linear_fc1_up, nn.Linear)
        assert isinstance(transformed_model.linear_fc1_gate, nn.Linear)
        assert isinstance(transformed_model.output_projection, nn.Linear)
        assert isinstance(transformed_model.embedding, nn.Embedding)
        assert isinstance(transformed_model.layernorm, nn.LayerNorm)

    def test_canonical_lora_transform_fused_layers(self):
        """Test CanonicalLoRA transformation on fused layers (the primary use case)."""
        model = MegatronStyleModel()
        lora = CanonicalLoRA(target_modules=["linear_q", "linear_k", "linear_v", "linear_fc1_up", "linear_fc1_gate"])

        # Mock the get_adapter_attributes_from_linear function
        def mock_get_attrs(module):
            if hasattr(module, "out_features"):
                if module.out_features == 1536:  # linear_qkv
                    return (False, 512, 1536, False, True)
                elif module.out_features == 2048:  # linear_fc1
                    return (False, 512, 2048, False, True)
            return (False, 512, 512, False, True)  # default

        with patch(
            "megatron.bridge.peft.canonical_lora.get_adapter_attributes_from_linear", side_effect=mock_get_attrs
        ):
            # Mock ParallelLinearAdapter
            with patch("megatron.bridge.peft.canonical_lora.ParallelLinearAdapter") as mock_adapter:
                mock_adapter.return_value = nn.Linear(1, 1)  # Simple mock

                # Apply CanonicalLoRA
                transformed_model = lora(model, training=True)

                # Check that fused layers were transformed to canonical split wrappers
                assert isinstance(transformed_model.linear_qkv, LoRALinearSplitQKV)
                assert isinstance(transformed_model.linear_fc1, LoRALinearSplitFC1UpGate)

                # Check that adapters have the expected components
                assert hasattr(transformed_model.linear_qkv.adapter, "adapter_q")
                assert hasattr(transformed_model.linear_qkv.adapter, "adapter_k")
                assert hasattr(transformed_model.linear_qkv.adapter, "adapter_v")
                assert hasattr(transformed_model.linear_fc1.adapter, "adapter_up")
                assert hasattr(transformed_model.linear_fc1.adapter, "adapter_gate")

                # Check that non-target layers were not transformed
                assert isinstance(transformed_model.linear_proj, MockMegatronLinear)
                assert isinstance(transformed_model.linear_fc2, MockMegatronLinear)

    def test_canonical_lora_transform_nested_model(self):
        """Test CanonicalLoRA transformation on nested model structures."""
        model = NestedModel()
        lora = CanonicalLoRA(target_modules=["linear_proj", "linear_fc2"])

        # Apply CanonicalLoRA
        transformed_model = lora(model, training=True)

        # Check that nested target modules were transformed
        for layer in transformed_model.layers:
            assert isinstance(layer["attention"]["linear_proj"], LinearAdapter)
            assert isinstance(layer["mlp"]["linear_fc2"], LinearAdapter)

            # Check non-target modules remain unchanged
            assert isinstance(layer["attention"]["linear_q"], nn.Linear)
            assert isinstance(layer["attention"]["linear_k"], nn.Linear)
            assert isinstance(layer["attention"]["linear_v"], nn.Linear)
            assert isinstance(layer["mlp"]["linear_fc1_up"], nn.Linear)
            assert isinstance(layer["mlp"]["linear_fc1_gate"], nn.Linear)

    def test_canonical_lora_wildcard_matching(self):
        """Test CanonicalLoRA transformation with wildcard patterns."""
        model = NestedModel()
        # Only apply LoRA to first layer's attention modules
        lora = CanonicalLoRA(target_modules=["layers.0.attention.*"])

        # Apply CanonicalLoRA
        transformed_model = lora(model, training=True)

        # Check first layer attention modules are transformed
        assert isinstance(transformed_model.layers[0]["attention"]["linear_q"], LinearAdapter)
        assert isinstance(transformed_model.layers[0]["attention"]["linear_k"], LinearAdapter)
        assert isinstance(transformed_model.layers[0]["attention"]["linear_v"], LinearAdapter)
        assert isinstance(transformed_model.layers[0]["attention"]["linear_proj"], LinearAdapter)

        # Check first layer MLP modules are NOT transformed
        assert isinstance(transformed_model.layers[0]["mlp"]["linear_fc1_up"], nn.Linear)
        assert isinstance(transformed_model.layers[0]["mlp"]["linear_fc1_gate"], nn.Linear)
        assert isinstance(transformed_model.layers[0]["mlp"]["linear_fc2"], nn.Linear)

        # Check second layer modules are NOT transformed
        assert isinstance(transformed_model.layers[1]["attention"]["linear_q"], nn.Linear)
        assert isinstance(transformed_model.layers[1]["attention"]["linear_k"], nn.Linear)
        assert isinstance(transformed_model.layers[1]["attention"]["linear_v"], nn.Linear)
        assert isinstance(transformed_model.layers[1]["attention"]["linear_proj"], nn.Linear)

    def test_canonical_lora_adapter_properties(self):
        """Test that CanonicalLoRA adapters have correct properties."""
        model = SimpleModel()
        lora = CanonicalLoRA(target_modules=["linear_proj"], dim=16, alpha=32, dropout=0.1)

        # Apply CanonicalLoRA
        transformed_model = lora(model, training=True)

        # Check adapter properties
        adapter = transformed_model.linear_proj
        assert hasattr(adapter, "dim")
        assert hasattr(adapter, "scale")
        assert hasattr(adapter, "lora_a")
        assert hasattr(adapter, "lora_b")
        assert hasattr(adapter, "dropout")

        assert adapter.dim == 16
        assert adapter.scale == 32 / 16  # alpha / dim
        assert adapter.dropout.p == 0.1

    def test_canonical_lora_parameter_freezing(self):
        """Test that base model parameters are frozen and adapter parameters are trainable."""
        model = SimpleModel()
        lora = CanonicalLoRA(target_modules=["linear_proj"])

        # Apply CanonicalLoRA
        transformed_model = lora(model, training=True)

        # Check that original weights are frozen
        linear_adapter = transformed_model.linear_proj
        assert not linear_adapter.weight.requires_grad
        if linear_adapter.bias is not None:
            assert not linear_adapter.bias.requires_grad

        # Check that LoRA parameters are trainable
        assert linear_adapter.lora_a.weight.requires_grad
        assert linear_adapter.lora_b.weight.requires_grad

    def test_canonical_lora_forward_pass(self):
        """Test that CanonicalLoRA adapted models can perform forward passes."""
        model = SimpleModel()
        lora = CanonicalLoRA(target_modules=["linear_proj", "linear_fc2"], dim=8)

        # Apply CanonicalLoRA
        transformed_model = lora(model, training=True)

        # Test forward pass
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))

        with torch.no_grad():
            embeddings = transformed_model.embedding(input_ids)  # [batch, seq, 512]

            # Test adapted layers
            proj_out = transformed_model.linear_proj(embeddings)  # Should work
            fc2_out = transformed_model.linear_fc2(torch.randn(batch_size, seq_len, 2048))  # Should work

            assert proj_out.shape == (batch_size, seq_len, 512)
            assert fc2_out.shape == (batch_size, seq_len, 512)

            # Test non-adapted layers still work
            q_out = transformed_model.linear_q(embeddings)  # Should work (not adapted)
            k_out = transformed_model.linear_k(embeddings)  # Should work (not adapted)
            v_out = transformed_model.linear_v(embeddings)  # Should work (not adapted)

            assert q_out.shape == (batch_size, seq_len, 512)
            assert k_out.shape == (batch_size, seq_len, 512)
            assert v_out.shape == (batch_size, seq_len, 512)

    def test_canonical_lora_training_vs_inference_mode(self):
        """Test CanonicalLoRA behavior in training vs inference mode."""
        model = SimpleModel()
        lora = CanonicalLoRA()

        # Test training mode
        training_model = lora(model, training=True)
        assert training_model.training

        # Test inference mode
        inference_model = lora(model, training=False)
        assert not inference_model.training

    def test_canonical_lora_list_model_support(self):
        """Test CanonicalLoRA support for list of model chunks (pipeline parallelism)."""
        # Create list of model chunks
        model_chunks = [SimpleModel() for _ in range(3)]
        lora = CanonicalLoRA(target_modules=["linear_proj", "linear_fc2"])

        # Apply CanonicalLoRA to list of models
        transformed_chunks = lora(model_chunks, training=True)

        # Should return list of same length
        assert isinstance(transformed_chunks, list)
        assert len(transformed_chunks) == 3

        # Each chunk should have LoRA applied to target modules
        for chunk in transformed_chunks:
            assert isinstance(chunk.linear_proj, LinearAdapter)
            assert isinstance(chunk.linear_fc2, LinearAdapter)
            # Non-target modules should remain unchanged
            assert isinstance(chunk.linear_q, nn.Linear)
            assert isinstance(chunk.linear_k, nn.Linear)
            assert isinstance(chunk.linear_v, nn.Linear)

    def test_canonical_lora_parameter_efficiency(self):
        """Test that CanonicalLoRA significantly reduces trainable parameters."""
        model = SimpleModel()

        # Count original parameters
        original_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Apply CanonicalLoRA
        lora = CanonicalLoRA(target_modules=["linear_proj", "linear_fc2"], dim=8)  # Small rank for efficiency
        adapted_model = lora(model, training=True)

        # Count trainable parameters after LoRA
        lora_params = sum(p.numel() for p in adapted_model.parameters() if p.requires_grad)

        # LoRA should significantly reduce trainable parameters
        assert lora_params < original_params
        efficiency_ratio = lora_params / original_params
        assert efficiency_ratio < 0.5  # More relaxed since we're only adapting 2 layers

    def test_canonical_lora_reproducibility(self):
        """Test that CanonicalLoRA application is deterministic."""
        torch.manual_seed(42)
        model1 = SimpleModel()
        lora1 = CanonicalLoRA(target_modules=["linear_proj"], dim=8, alpha=16)
        adapted_model1 = lora1(model1, training=True)

        torch.manual_seed(42)
        model2 = SimpleModel()
        lora2 = CanonicalLoRA(target_modules=["linear_proj"], dim=8, alpha=16)
        adapted_model2 = lora2(model2, training=True)

        # LoRA weights should be identical with same seed
        lora_a_1 = adapted_model1.linear_proj.lora_a.weight.data
        lora_a_2 = adapted_model2.linear_proj.lora_a.weight.data
        assert torch.equal(lora_a_1, lora_a_2)

        lora_b_1 = adapted_model1.linear_proj.lora_b.weight.data
        lora_b_2 = adapted_model2.linear_proj.lora_b.weight.data
        assert torch.equal(lora_b_1, lora_b_2)

    def test_canonical_lora_transform_idempotent(self):
        """Test that CanonicalLoRA transform is idempotent (applying twice has same effect as applying once)."""
        model = SimpleModel()
        lora = CanonicalLoRA(target_modules=["linear_proj", "linear_fc2"], dim=8, alpha=16)

        # Apply CanonicalLoRA first time
        first_transform = lora(model, training=True)

        # Store references to the transformed modules
        first_linear_proj = first_transform.linear_proj
        first_linear_fc2 = first_transform.linear_fc2
        first_linear_q = first_transform.linear_q  # Should remain unchanged

        # Verify first transformation worked
        assert isinstance(first_linear_proj, LinearAdapter)
        assert isinstance(first_linear_fc2, LinearAdapter)
        assert isinstance(first_linear_q, nn.Linear)

        # Apply CanonicalLoRA second time to the already-transformed model
        second_transform = lora(first_transform, training=True)

        # Verify idempotency: second transformation should return identical objects
        assert second_transform.linear_proj is first_linear_proj
        assert second_transform.linear_fc2 is first_linear_fc2
        assert second_transform.linear_q is first_linear_q

        # Verify the module types are still correct
        assert isinstance(second_transform.linear_proj, LinearAdapter)
        assert isinstance(second_transform.linear_fc2, LinearAdapter)
        assert isinstance(second_transform.linear_q, nn.Linear)

        # Verify the LoRA parameters are identical
        assert torch.equal(
            first_transform.linear_proj.lora_a.weight.data, second_transform.linear_proj.lora_a.weight.data
        )
        assert torch.equal(
            first_transform.linear_proj.lora_b.weight.data, second_transform.linear_proj.lora_b.weight.data
        )

    def test_canonical_lora_transform_idempotent_fused_layers(self):
        """Test that CanonicalLoRA transform is idempotent for fused layers."""
        model = MegatronStyleModel()
        lora = CanonicalLoRA(target_modules=["linear_q", "linear_k", "linear_v"])

        # Mock the get_adapter_attributes_from_linear function
        with patch("megatron.bridge.peft.canonical_lora.get_adapter_attributes_from_linear") as mock_get_attrs:
            mock_get_attrs.return_value = (False, 512, 1536, False, True)

            # Mock ParallelLinearAdapter
            with patch("megatron.bridge.peft.canonical_lora.ParallelLinearAdapter") as mock_adapter:
                mock_adapter.return_value = nn.Linear(1, 1)  # Simple mock

                # Apply CanonicalLoRA first time
                first_transform = lora(model, training=True)

                # Store reference to the transformed module
                first_linear_qkv = first_transform.linear_qkv

                # Verify first transformation worked
                assert isinstance(first_linear_qkv, LoRALinearSplitQKV)

                # Apply CanonicalLoRA second time to the already-transformed model
                second_transform = lora(first_transform, training=True)

                # Verify idempotency: second transformation should return identical object
                assert second_transform.linear_qkv is first_linear_qkv
                assert isinstance(second_transform.linear_qkv, LoRALinearSplitQKV)

                # Verify the adapter structure is unchanged
                assert hasattr(second_transform.linear_qkv.adapter, "adapter_q")
                assert hasattr(second_transform.linear_qkv.adapter, "adapter_k")
                assert hasattr(second_transform.linear_qkv.adapter, "adapter_v")


class TestCanonicalLoRAMegatronLayers:
    """Test CanonicalLoRA with Megatron-style fused layers."""

    def test_megatron_style_qkv_transform(self):
        """Test that Megatron-style linear_qkv gets properly transformed."""
        model = MegatronStyleModel()
        lora = CanonicalLoRA(target_modules=["linear_q", "linear_k", "linear_v"])

        # Mock the get_adapter_attributes_from_linear function
        with patch("megatron.bridge.peft.canonical_lora.get_adapter_attributes_from_linear") as mock_get_attrs:
            mock_get_attrs.return_value = (False, 512, 1536, False, True)

            # Mock ParallelLinearAdapter
            with patch("megatron.bridge.peft.canonical_lora.ParallelLinearAdapter") as mock_adapter:
                mock_adapter.return_value = nn.Linear(1, 1)  # Simple mock

                # Apply CanonicalLoRA
                transformed_model = lora(model, training=True)

                # Check that linear_qkv was transformed to LoRALinearSplitQKV
                assert isinstance(transformed_model.linear_qkv, LoRALinearSplitQKV)
                assert hasattr(transformed_model.linear_qkv.adapter, "adapter_q")
                assert hasattr(transformed_model.linear_qkv.adapter, "adapter_k")
                assert hasattr(transformed_model.linear_qkv.adapter, "adapter_v")

    def test_megatron_style_fc1_transform(self):
        """Test that Megatron-style linear_fc1 gets properly transformed."""
        model = MegatronStyleModel()
        lora = CanonicalLoRA(target_modules=["linear_fc1_up", "linear_fc1_gate"])

        # Mock the get_adapter_attributes_from_linear function
        with patch("megatron.bridge.peft.canonical_lora.get_adapter_attributes_from_linear") as mock_get_attrs:
            mock_get_attrs.return_value = (False, 512, 2048, False, True)

            # Mock ParallelLinearAdapter
            with patch("megatron.bridge.peft.canonical_lora.ParallelLinearAdapter") as mock_adapter:
                mock_adapter.return_value = nn.Linear(1, 1)  # Simple mock

                # Apply CanonicalLoRA
                transformed_model = lora(model, training=True)

                # Check that linear_fc1 was transformed to LoRALinearSplitFC1UpGate
                assert isinstance(transformed_model.linear_fc1, LoRALinearSplitFC1UpGate)
                assert hasattr(transformed_model.linear_fc1.adapter, "adapter_up")
                assert hasattr(transformed_model.linear_fc1.adapter, "adapter_gate")


class TestCanonicalLoRAHelperClasses:
    """Test helper classes for CanonicalLoRA."""

    def test_module_dict_sharded_state_dict(self):
        """Test ModuleDict's sharded_state_dict method."""

        # Create mock modules that have sharded_state_dict method
        class MockModule(nn.Module):
            def __init__(self, name):
                super().__init__()
                self.name = name

            def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
                return {f"{prefix}{self.name}": f"mocked_state_{self.name}"}

        module_dict = ModuleDict({"layer1": MockModule("layer1"), "layer2": MockModule("layer2")})

        result = module_dict.sharded_state_dict(prefix="test_")

        # Should have entries from both layers with correct prefixes
        assert "test_layer1.layer1" in result
        assert "test_layer2.layer2" in result
        assert result["test_layer1.layer1"] == "mocked_state_layer1"
        assert result["test_layer2.layer2"] == "mocked_state_layer2"

    def test_lora_linear_split_qkv_forward(self):
        """Test LoRALinearSplitQKV forward pass."""
        # Create mock base layer
        base_layer = nn.Linear(512, 1536)
        base_layer.config = type("Config", (), {"kv_channels": 64, "num_query_groups": 8})()

        # Create mock adapters
        adapters = ModuleDict(
            {"adapter_q": nn.Linear(512, 512), "adapter_k": nn.Linear(512, 512), "adapter_v": nn.Linear(512, 512)}
        )

        # Create the wrapper
        wrapper = LoRALinearSplitQKV(base_layer, adapters)

        # Mock the forward methods to return tuples
        def mock_forward(x):
            return base_layer(x), None

        with patch.object(wrapper, "base_linear_forward", side_effect=lambda x: (base_layer(x), None, x)):
            # Test forward pass
            x = torch.randn(2, 10, 512)
            output, bias = wrapper(x)

            assert output.shape == (2, 10, 1536)
            assert bias is None

    def test_lora_linear_split_fc1_up_gate_forward(self):
        """Test LoRALinearSplitFC1UpGate forward pass."""
        # Create mock base layer
        base_layer = nn.Linear(512, 2048)

        # Create mock adapters
        adapters = ModuleDict({"adapter_up": nn.Linear(512, 1024), "adapter_gate": nn.Linear(512, 1024)})

        # Create the wrapper
        wrapper = LoRALinearSplitFC1UpGate(base_layer, adapters)

        with patch.object(wrapper, "base_linear_forward", side_effect=lambda x: (base_layer(x), None, x)):
            # Test forward pass
            x = torch.randn(2, 10, 512)
            output, bias = wrapper(x)

            assert output.shape == (2, 10, 2048)
            assert bias is None


class TestCanonicalLoRAMegatronIntegration:
    """Integration tests for CanonicalLoRA with real Megatron models."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown_parallel_state(self):
        """Setup and teardown parallel state for Megatron tests."""

        if not dist.is_initialized():
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = "29500"
            os.environ["RANK"] = "0"
            os.environ["LOCAL_RANK"] = "0"
            os.environ["WORLD_SIZE"] = "1"

            device_count = torch.cuda.device_count()
            if device_count > 0:
                torch.cuda.set_device(0)

            init_process_group_kwargs = {
                "backend": "nccl" if device_count > 0 else "gloo",
                "world_size": 1,
                "rank": 0,
                "timeout": datetime.timedelta(minutes=30),
            }

            dist.init_process_group(**init_process_group_kwargs)

        assert dist.is_initialized(), "Distributed backend not initialized"
        if not parallel_state.model_parallel_is_initialized():
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                virtual_pipeline_model_parallel_size=None,
                context_parallel_size=1,
            )

        assert parallel_state.model_parallel_is_initialized(), "Model parallel not initialized"
        from megatron.bridge.training.initialize import _set_random_seed

        _set_random_seed(
            seed_=1234,
            data_parallel_random_init=False,
            te_rng_tracker=True,
            inference_rng_tracker=False,
        )

        yield

        try:
            if parallel_state.model_parallel_is_initialized():
                parallel_state.destroy_model_parallel()
            if dist.is_initialized():
                dist.destroy_process_group()
                # Clean up environment variables
                for key in ["MASTER_ADDR", "MASTER_PORT", "RANK", "LOCAL_RANK", "WORLD_SIZE"]:
                    os.environ.pop(key, None)
        except (NameError, AttributeError, RuntimeError):
            pass

    def _create_canonical_lora_pre_wrap_hook(self, lora_config: CanonicalLoRA):
        """Create a pre-wrap hook that applies CanonicalLoRA to the model.

        Args:
            lora_config: CanonicalLoRA configuration instance

        Returns:
            A callable hook that can be registered with the model provider
        """

        def canonical_lora_pre_wrap_hook(model: list[MegatronModule]) -> list[MegatronModule]:
            """Pre-wrap hook that applies CanonicalLoRA transformation.

            Args:
                model: List of base model modules before distributed wrapping

            Returns:
                List of CanonicalLoRA-transformed model modules
            """
            return lora_config(model, training=True)

        return canonical_lora_pre_wrap_hook

    def test_canonical_lora_with_gpt_model(self):
        """Test CanonicalLoRA application to a real GPT model using pre-wrap hooks."""

        # Create a minimal GPT configuration
        model_provider = GPTModelProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=2,
            vocab_size=1000,
            ffn_hidden_size=256,
        )

        # Create CanonicalLoRA instance targeting linear layers
        lora = CanonicalLoRA(
            target_modules=[
                "linear_q",
                "linear_k",
                "linear_v",
                "linear_proj",
                "linear_fc1_up",
                "linear_fc1_gate",
                "linear_fc2",
            ],
            dim=8,
            alpha=16,
            dropout=0.0,
        )

        # Register CanonicalLoRA pre-wrap hook
        lora_hook = self._create_canonical_lora_pre_wrap_hook(lora)
        model_provider.register_pre_wrap_hook(lora_hook)
        model_provider.finalize()

        # Get the model with CanonicalLoRA applied via hook
        adapted_model = model_provider.provide_distributed_model(ddp_config=None, wrap_with_ddp=False)

        # Verify we got a list of Megatron modules
        assert isinstance(adapted_model, list)
        assert len(adapted_model) > 0
        assert all(isinstance(chunk, MegatronModule) for chunk in adapted_model)

        # Verify that LoRA was applied to target modules
        found_lora_modules = []
        for chunk in adapted_model:
            for name, module in chunk.named_modules():
                if isinstance(module, (LoRALinear, LoRALinearSplitQKV, LoRALinearSplitFC1UpGate)):
                    found_lora_modules.append(name)

        # Should have found some LoRA modules
        assert len(found_lora_modules) > 0, "No LoRA modules found in adapted model"

        # Verify parameter states
        total_params = 0
        trainable_params = 0
        for chunk in adapted_model:
            for param in chunk.parameters():
                total_params += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()

        # Should have significantly fewer trainable parameters than total
        assert trainable_params < total_params
        efficiency_ratio = trainable_params / total_params
        assert efficiency_ratio < 0.3, f"CanonicalLoRA should be parameter efficient, got ratio: {efficiency_ratio}"

    @pytest.mark.pleasefixme  # This test is too slow for unit tests (>Xs)
    def test_canonical_lora_forward_pass_with_megatron_model(self):
        """Test forward pass through CanonicalLoRA-adapted Megatron model using pre-wrap hooks."""

        # Create minimal config for fast testing
        model_provider = GPTModelProvider(
            num_layers=1,
            hidden_size=64,
            num_attention_heads=2,
            vocab_size=128,
            ffn_hidden_size=128,
        )

        # Create CanonicalLoRA and register hook
        lora = CanonicalLoRA(dim=4, alpha=8)
        lora_hook = self._create_canonical_lora_pre_wrap_hook(lora)
        model_provider.register_pre_wrap_hook(lora_hook)
        model_provider.finalize()

        # Get and adapt model using hook
        adapted_model = model_provider.provide_distributed_model(ddp_config=None, wrap_with_ddp=False)

        # Test forward pass with proper Megatron input format
        batch_size, seq_len = 2, 8

        # Get model device (model is on CUDA, inputs need to match)
        model_device = next(adapted_model[0].parameters()).device

        # Create input tensors in the format expected by Megatron models
        input_ids = torch.randint(0, model_provider.vocab_size, (batch_size, seq_len), device=model_device)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=model_device).unsqueeze(0).expand(batch_size, -1)

        # Create 4D causal attention mask [batch_size, 1, seq_len, seq_len]
        # True values are masked out (don't attend), False values attend
        attention_mask = torch.tril(torch.ones(seq_len, seq_len, device=model_device)) < 0.5
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)

        # Run forward pass using the standard codebase pattern
        forward_args = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }

        with torch.no_grad():
            for chunk in adapted_model:
                output = chunk(**forward_args)

                # Verify output shape and that LoRA is active
                if isinstance(output, tuple):
                    logits = output[0]
                else:
                    logits = output

                expected_shape = (batch_size, seq_len, model_provider.vocab_size)
                assert logits.shape == expected_shape, f"Expected {expected_shape}, got {logits.shape}"

                # Count LoRA adaptations
                lora_count = sum(
                    1
                    for _, m in chunk.named_modules()
                    if isinstance(m, (LoRALinear, LoRALinearSplitQKV, LoRALinearSplitFC1UpGate))
                )
                assert lora_count > 0, "Should have LoRA adaptations applied"


class TestCanonicalLoRAIntegration:
    """Integration tests for CanonicalLoRA functionality."""

    def test_canonical_lora_full_pipeline(self):
        """Test complete CanonicalLoRA application pipeline."""
        # Create base model
        model = SimpleModel()
        original_weights = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                original_weights[name] = module.weight.data.clone()

        # Apply CanonicalLoRA
        lora = CanonicalLoRA(target_modules=["linear_proj", "linear_fc2"], dim=4, alpha=8)
        adapted_model = lora(model, training=True)

        # Verify CanonicalLoRA was applied
        assert isinstance(adapted_model.linear_proj, LinearAdapter)
        assert isinstance(adapted_model.linear_fc2, LinearAdapter)
        # Verify non-target modules were not adapted
        assert isinstance(adapted_model.linear_q, nn.Linear)
        assert isinstance(adapted_model.linear_k, nn.Linear)
        assert isinstance(adapted_model.linear_v, nn.Linear)

        # Perform training step (mock)
        optimizer = torch.optim.Adam(adapted_model.parameters())

        # Forward pass
        input_ids = torch.randint(0, 1000, (2, 10))
        embeddings = adapted_model.embedding(input_ids)
        proj_output = adapted_model.linear_proj(embeddings)
        fc2_output = adapted_model.linear_fc2(torch.randn(2, 10, 2048))
        loss = proj_output.sum() + fc2_output.sum()

        # Backward pass
        loss.backward()
        optimizer.step()

        # Verify training worked
        assert loss.item() != 0.0
