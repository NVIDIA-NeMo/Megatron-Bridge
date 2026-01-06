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
from megatron.bridge.peft.lora import LoRA, LoRAMerge, VLMLoRA
from megatron.bridge.peft.lora_layers import LinearAdapter, LoRALinear


class SimpleModel(nn.Module):
    """Simple test model with various linear layers."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(1000, 512)
        self.linear_qkv = nn.Linear(512, 1536)  # Should be matched
        self.linear_proj = nn.Linear(512, 512)  # Should be matched
        self.linear_fc1 = nn.Linear(512, 2048)  # Should be matched
        self.linear_fc2 = nn.Linear(2048, 512)  # Should be matched
        self.output_projection = nn.Linear(512, 1000)  # Should NOT be matched (not in target_modules)
        self.layernorm = nn.LayerNorm(512)


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
                                "linear_qkv": nn.Linear(512, 1536),
                                "linear_proj": nn.Linear(512, 512),
                            }
                        ),
                        "mlp": nn.ModuleDict(
                            {
                                "linear_fc1": nn.Linear(512, 2048),
                                "linear_fc2": nn.Linear(2048, 512),
                            }
                        ),
                    }
                )
                for _ in range(2)
            ]
        )


class TestLoRA:
    """Test suite for LoRA PEFT implementation."""

    def test_lora_initialization(self):
        """Test LoRA class initialization with default and custom parameters."""
        # Test default initialization
        lora = LoRA()
        assert lora.target_modules == ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]
        assert lora.dim == 32
        assert lora.alpha == 32
        assert lora.dropout == 0.0
        assert lora.dropout_position == "pre"
        assert lora.lora_A_init_method == "xavier"
        assert lora.lora_B_init_method == "zero"

        # Test custom initialization
        custom_lora = LoRA(
            target_modules=["linear_qkv"],
            dim=16,
            alpha=16,
            dropout=0.1,
            dropout_position="post",
            lora_A_init_method="uniform",
        )
        assert custom_lora.target_modules == ["linear_qkv"]
        assert custom_lora.dim == 16
        assert custom_lora.alpha == 16
        assert custom_lora.dropout == 0.1
        assert custom_lora.dropout_position == "post"
        assert custom_lora.lora_A_init_method == "uniform"

    def test_lora_transform_simple_model(self):
        """Test LoRA transformation on a simple model."""
        model = SimpleModel()
        lora = LoRA(target_modules=["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"])

        # Apply LoRA
        transformed_model = lora(model, training=True)

        # Check that target modules were transformed to LinearAdapter
        assert isinstance(transformed_model.linear_qkv, LinearAdapter)
        assert isinstance(transformed_model.linear_proj, LinearAdapter)
        assert isinstance(transformed_model.linear_fc1, LinearAdapter)
        assert isinstance(transformed_model.linear_fc2, LinearAdapter)

        # Check that non-target modules were not transformed
        assert isinstance(transformed_model.output_projection, nn.Linear)
        assert isinstance(transformed_model.embedding, nn.Embedding)
        assert isinstance(transformed_model.layernorm, nn.LayerNorm)

    def test_lora_transform_with_exclude_modules(self):
        """Test LoRA transformation with exclude_modules parameter."""
        model = SimpleModel()
        # Use only exclude_modules (no target_modules) to test exclusion behavior
        lora = LoRA(
            target_modules=[],  # Empty target_modules to use exclude mode
            exclude_modules=["linear_fc2", "output_projection"],  # Exclude specific linear modules
        )

        # Apply LoRA
        transformed_model = lora(model, training=True)

        # Check that excluded linear modules were not transformed
        assert isinstance(transformed_model.linear_fc2, nn.Linear)
        assert isinstance(transformed_model.output_projection, nn.Linear)

        # Check that non-excluded linear modules were transformed
        # (In exclude mode, all linear layers except excluded ones should be transformed)
        assert isinstance(transformed_model.linear_qkv, LinearAdapter)
        assert isinstance(transformed_model.linear_proj, LinearAdapter)
        assert isinstance(transformed_model.linear_fc1, LinearAdapter)

        # Non-linear modules should never be transformed regardless
        assert isinstance(transformed_model.embedding, nn.Embedding)
        assert isinstance(transformed_model.layernorm, nn.LayerNorm)

    def test_lora_transform_nested_model(self):
        """Test LoRA transformation on nested model structures."""
        model = NestedModel()
        lora = LoRA(target_modules=["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"])

        # Apply LoRA
        transformed_model = lora(model, training=True)

        # Check that nested target modules were transformed
        for layer in transformed_model.layers:
            assert isinstance(layer["attention"]["linear_qkv"], LinearAdapter)
            assert isinstance(layer["attention"]["linear_proj"], LinearAdapter)
            assert isinstance(layer["mlp"]["linear_fc1"], LinearAdapter)
            assert isinstance(layer["mlp"]["linear_fc2"], LinearAdapter)

    def test_lora_wildcard_matching(self):
        """Test LoRA transformation with wildcard patterns."""
        model = NestedModel()
        # Only apply LoRA to first layer's attention modules
        lora = LoRA(target_modules=["layers.0.attention.*"])

        # Apply LoRA
        transformed_model = lora(model, training=True)

        # Check first layer attention modules are transformed
        assert isinstance(transformed_model.layers[0]["attention"]["linear_qkv"], LinearAdapter)
        assert isinstance(transformed_model.layers[0]["attention"]["linear_proj"], LinearAdapter)

        # Check first layer MLP modules are NOT transformed
        assert isinstance(transformed_model.layers[0]["mlp"]["linear_fc1"], nn.Linear)
        assert isinstance(transformed_model.layers[0]["mlp"]["linear_fc2"], nn.Linear)

        # Check second layer modules are NOT transformed
        assert isinstance(transformed_model.layers[1]["attention"]["linear_qkv"], nn.Linear)
        assert isinstance(transformed_model.layers[1]["attention"]["linear_proj"], nn.Linear)
        assert isinstance(transformed_model.layers[1]["mlp"]["linear_fc1"], nn.Linear)
        assert isinstance(transformed_model.layers[1]["mlp"]["linear_fc2"], nn.Linear)

    def test_lora_adapter_properties(self):
        """Test that LoRA adapters have correct properties."""
        model = SimpleModel()
        lora = LoRA(dim=16, alpha=32, dropout=0.1)

        # Apply LoRA
        transformed_model = lora(model, training=True)

        # Check adapter properties
        adapter = transformed_model.linear_qkv
        assert hasattr(adapter, "dim")
        assert hasattr(adapter, "alpha")
        assert hasattr(adapter, "scale")
        assert hasattr(adapter, "linear_in")
        assert hasattr(adapter, "linear_out")
        assert hasattr(adapter, "dropout")

        assert adapter.dim == 16
        assert adapter.scale == 32 / 16  # alpha / dim
        assert adapter.dropout.p == 0.1

    def test_lora_parameter_freezing(self):
        """Test that base model parameters are frozen and adapter parameters are trainable."""
        model = SimpleModel()
        lora = LoRA()

        # Apply LoRA
        transformed_model = lora(model, training=True)

        # Check that original weights are frozen
        linear_adapter = transformed_model.linear_qkv
        assert not linear_adapter.weight.requires_grad
        if linear_adapter.bias is not None:
            assert not linear_adapter.bias.requires_grad

        # Check that LoRA parameters are trainable
        assert linear_adapter.linear_in.weight.requires_grad
        assert linear_adapter.linear_out.weight.requires_grad

    def test_lora_forward_pass(self):
        """Test that LoRA adapted models can perform forward passes."""
        model = SimpleModel()
        lora = LoRA(dim=8)

        # Apply LoRA
        transformed_model = lora(model, training=True)

        # Test forward pass
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))

        with torch.no_grad():
            embeddings = transformed_model.embedding(input_ids)  # [batch, seq, 512]

            # Test each adapted layer
            qkv_out = transformed_model.linear_qkv(embeddings)  # Should work
            proj_out = transformed_model.linear_proj(embeddings)  # Should work
            fc1_out = transformed_model.linear_fc1(embeddings)  # Should work
            fc2_out = transformed_model.linear_fc2(fc1_out)  # Should work

            assert qkv_out.shape == (batch_size, seq_len, 1536)
            assert proj_out.shape == (batch_size, seq_len, 512)
            assert fc1_out.shape == (batch_size, seq_len, 2048)
            assert fc2_out.shape == (batch_size, seq_len, 512)

    def test_lora_training_vs_inference_mode(self):
        """Test LoRA behavior in training vs inference mode."""
        model = SimpleModel()
        lora = LoRA()

        # Test training mode
        training_model = lora(model, training=True)
        assert training_model.training

        # Test inference mode
        inference_model = lora(model, training=False)
        assert not inference_model.training

    @patch("megatron.bridge.peft.lora.te")
    def test_lora_te_linear_support(self, mock_te):
        """Test LoRA support for Transformer Engine Linear layers."""

        # Create the TE Linear type and an actual instance
        class MockTELinear(nn.Module):
            def __init__(self):
                super().__init__()

                # Create a simple weight mock that doesn't have _local_tensor
                class MockWeightData:
                    pass

                class MockWeight:
                    def __init__(self):
                        self.data = MockWeightData()

                self.weight = MockWeight()
                self.quant_state = None

        # Set the mock_te.Linear to our MockTELinear class
        mock_te.Linear = MockTELinear

        # Create an actual instance of our mock TE Linear
        te_linear_instance = MockTELinear()

        # Create model with mock TE linear
        model = nn.Module()
        model.te_linear = te_linear_instance

        lora = LoRA(target_modules=["te_linear"])

        # Create a mock class for TELinearAdapter to works with the isinstance() check
        class MockTELinearAdapter(nn.Module):
            def __init__(self, module, **kwargs):
                super().__init__()
                self.module = module

        # Import the module to patch the specific import
        from megatron.bridge.peft import lora as lora_module

        # Use patch.object to handle cases where TELinearAdapter might not exist
        # by creating it if necessary.
        with patch.object(lora_module, "TELinearAdapter", MockTELinearAdapter, create=True):
            # Should create TELinearAdapter
            result = lora(model, training=True)

            # Verify that te_linear was transformed to our mock adapter
            assert isinstance(result.te_linear, MockTELinearAdapter)

    @pytest.mark.timeout(10)
    def test_lora_list_model_support(self):
        """Test LoRA support for list of model chunks (pipeline parallelism)."""
        # Create list of model chunks
        model_chunks = [SimpleModel() for _ in range(3)]
        lora = LoRA()

        # Apply LoRA to list of models
        transformed_chunks = lora(model_chunks, training=True)

        # Should return list of same length
        assert isinstance(transformed_chunks, list)
        assert len(transformed_chunks) == 3

        # Each chunk should have LoRA applied
        for chunk in transformed_chunks:
            assert isinstance(chunk.linear_qkv, LinearAdapter)
            assert isinstance(chunk.linear_proj, LinearAdapter)
            assert isinstance(chunk.linear_fc1, LinearAdapter)
            assert isinstance(chunk.linear_fc2, LinearAdapter)


class TestLoRAMerge:
    """Test suite for LoRA merge functionality."""

    def test_lora_merge_initialization(self):
        """Test LoRAMerge class initialization."""
        merge = LoRAMerge()
        assert hasattr(merge, "transform")

    def test_lora_merge_transform(self):
        """Test LoRA weight merging behavior with LinearAdapter instances."""
        # Create model and apply LoRA
        model = SimpleModel()
        lora = LoRA(dim=8, alpha=16)
        adapted_model = lora(model, training=True)

        # Get original weights
        original_weight = adapted_model.linear_qkv.weight.data.clone()

        # Create merge instance and apply
        merge = LoRAMerge()
        merged_model = merge(adapted_model, training=False)

        # Note: LoRAMerge only handles LoRALinear instances (Megatron modules),
        # not LinearAdapter instances (regular nn.Linear modules).
        # So for SimpleModel, the modules should remain as LinearAdapter unchanged.
        assert isinstance(merged_model.linear_qkv, LinearAdapter)

        # Weights should be unchanged since merge doesn't apply to LinearAdapter
        merged_weight = merged_model.linear_qkv.weight.data
        assert torch.equal(original_weight, merged_weight)

    def test_lora_merge_with_lora_linear(self):
        """Test LoRA weight merging with LoRALinear instances (the intended use case)."""
        # Create a mock base module (representing a Megatron parallel module)
        base_module = nn.Linear(64, 128)
        original_weight = base_module.weight.data.clone()

        # Create a mock LoRA adapter that mimics ParallelLinearAdapter structure
        class MockAdapter(nn.Module):
            def __init__(self):
                super().__init__()
                self.alpha = 16
                self.dim = 8
                self.linear_in = nn.Linear(64, 8, bias=False)
                self.linear_out = nn.Linear(8, 128, bias=False)

                # Initialize with small non-zero values to see merge effect
                with torch.no_grad():
                    self.linear_in.weight.data.fill_(0.1)
                    self.linear_out.weight.data.fill_(0.05)

        adapter = MockAdapter()

        # Create LoRALinear instance (what LoRA creates for Megatron modules)
        lora_linear = LoRALinear(base_module, adapter)

        # Create merge instance and apply
        merge = LoRAMerge()
        merged_result = merge.transform(lora_linear)

        # Should return the LoRALinear wrapper (matches NeMo behavior)
        assert merged_result is lora_linear

        # The underlying weight should be modified (merged)
        merged_weight = lora_linear.to_wrap.weight.data
        assert not torch.equal(original_weight, merged_weight)

        # The change should equal the LoRA adaptation
        expected_lora_weight = (adapter.alpha / adapter.dim) * (adapter.linear_out.weight @ adapter.linear_in.weight)
        expected_merged = original_weight + expected_lora_weight
        assert torch.allclose(merged_weight, expected_merged, atol=1e-6)

    def test_lora_merge_non_lora_modules(self):
        """Test that non-LoRA modules are unchanged during merge."""
        model = SimpleModel()
        merge = LoRAMerge()

        # Apply merge to model without LoRA (should be no-op)
        original_linear = model.linear_qkv
        merged_model = merge(model, training=False)

        # Should be unchanged
        assert merged_model.linear_qkv is original_linear


class TestLoRAIntegration:
    """Integration tests for LoRA functionality."""

    def test_lora_full_pipeline(self):
        """Test complete LoRA application and merge pipeline."""
        # Create base model
        model = SimpleModel()
        original_weights = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                original_weights[name] = module.weight.data.clone()

        # Apply LoRA
        lora = LoRA(dim=4, alpha=8)
        adapted_model = lora(model, training=True)

        # Verify LoRA was applied
        assert isinstance(adapted_model.linear_qkv, LinearAdapter)

        # Perform training step (mock)
        optimizer = torch.optim.Adam(adapted_model.parameters())

        # Forward pass
        input_ids = torch.randint(0, 1000, (2, 10))
        embeddings = adapted_model.embedding(input_ids)
        output = adapted_model.linear_qkv(embeddings)
        loss = output.sum()

        # Backward pass
        loss.backward()
        optimizer.step()

        # Merge LoRA weights
        merge = LoRAMerge()
        merged_model = merge(adapted_model, training=False)

        # Note: LoRAMerge only handles LoRALinear instances (Megatron modules),
        # not LinearAdapter instances (regular nn.Linear modules).
        # So for SimpleModel, merge should be a no-op.
        assert isinstance(merged_model.linear_qkv, LinearAdapter)

        # The module should be unchanged since LoRAMerge doesn't affect LinearAdapter
        assert merged_model.linear_qkv is adapted_model.linear_qkv

    def test_lora_parameter_efficiency(self):
        """Test that LoRA significantly reduces trainable parameters."""
        model = SimpleModel()

        # Count original parameters
        original_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Apply LoRA
        lora = LoRA(dim=8)  # Small rank for efficiency
        adapted_model = lora(model, training=True)

        # Count trainable parameters after LoRA
        lora_params = sum(p.numel() for p in adapted_model.parameters() if p.requires_grad)

        # LoRA should significantly reduce trainable parameters
        assert lora_params < original_params
        efficiency_ratio = lora_params / original_params
        assert efficiency_ratio < 0.1

    def test_lora_reproducibility(self):
        """Test that LoRA application is deterministic."""
        torch.manual_seed(42)
        model1 = SimpleModel()
        lora1 = LoRA(dim=8, alpha=16)
        adapted_model1 = lora1(model1, training=True)

        torch.manual_seed(42)
        model2 = SimpleModel()
        lora2 = LoRA(dim=8, alpha=16)
        adapted_model2 = lora2(model2, training=True)

        # LoRA weights should be identical with same seed
        linear_in_1 = adapted_model1.linear_qkv.linear_in.weight.data
        linear_in_2 = adapted_model2.linear_qkv.linear_in.weight.data
        assert torch.equal(linear_in_1, linear_in_2)

        linear_out_1 = adapted_model1.linear_qkv.linear_out.weight.data
        linear_out_2 = adapted_model2.linear_qkv.linear_out.weight.data
        assert torch.equal(linear_out_1, linear_out_2)

    def test_lora_transform_idempotent(self):
        """Test that LoRA transform is idempotent (applying twice has same effect as applying once)."""
        model = SimpleModel()
        lora = LoRA(target_modules=["linear_qkv", "linear_proj"], dim=8, alpha=16)

        # Apply LoRA first time
        first_transform = lora(model, training=True)

        # Store references to the transformed modules
        first_linear_qkv = first_transform.linear_qkv
        first_linear_proj = first_transform.linear_proj
        first_linear_fc1 = first_transform.linear_fc1  # Should remain unchanged

        # Verify first transformation worked
        assert isinstance(first_linear_qkv, LinearAdapter)
        assert isinstance(first_linear_proj, LinearAdapter)
        assert isinstance(first_linear_fc1, nn.Linear)

        # Apply LoRA second time to the already-transformed model
        second_transform = lora(first_transform, training=True)

        # Verify idempotency: second transformation should return identical objects
        assert second_transform.linear_qkv is first_linear_qkv
        assert second_transform.linear_proj is first_linear_proj
        assert second_transform.linear_fc1 is first_linear_fc1

        # Verify the module types are still correct
        assert isinstance(second_transform.linear_qkv, LinearAdapter)
        assert isinstance(second_transform.linear_proj, LinearAdapter)
        assert isinstance(second_transform.linear_fc1, nn.Linear)

        # Verify the LoRA parameters are identical
        assert torch.equal(
            first_transform.linear_qkv.linear_in.weight.data, second_transform.linear_qkv.linear_in.weight.data
        )
        assert torch.equal(
            first_transform.linear_qkv.linear_out.weight.data, second_transform.linear_qkv.linear_out.weight.data
        )


class TestLoRAMegatronIntegration:
    """Integration tests for LoRA with real Megatron models."""

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

    def _create_lora_pre_wrap_hook(self, lora_config: LoRA):
        """Create a pre-wrap hook that applies LoRA to the model.

        Args:
            lora_config: LoRA configuration instance

        Returns:
            A callable hook that can be registered with the model provider
        """

        def lora_pre_wrap_hook(model: list[MegatronModule]) -> list[MegatronModule]:
            """Pre-wrap hook that applies LoRA transformation.

            Args:
                model: List of base model modules before distributed wrapping

            Returns:
                List of LoRA-transformed model modules
            """
            return lora_config(model, training=True)

        return lora_pre_wrap_hook

    def test_lora_with_gpt_model(self):
        """Test LoRA application to a real GPT model using pre-wrap hooks."""

        # Create a minimal GPT configuration
        model_provider = GPTModelProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=2,
            vocab_size=1000,
            ffn_hidden_size=256,
        )

        # Create LoRA instance targeting linear layers
        lora = LoRA(
            target_modules=["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"], dim=8, alpha=16, dropout=0.0
        )

        # Register LoRA pre-wrap hook
        lora_hook = self._create_lora_pre_wrap_hook(lora)
        model_provider.register_pre_wrap_hook(lora_hook)
        model_provider.finalize()

        # Get the model with LoRA applied via hook
        adapted_model = model_provider.provide_distributed_model(ddp_config=None, wrap_with_ddp=False)

        # Verify we got a list of Megatron modules
        assert isinstance(adapted_model, list)
        assert len(adapted_model) > 0
        assert all(isinstance(chunk, MegatronModule) for chunk in adapted_model)

        adapted_model = [chunk.cuda() for chunk in adapted_model]

        # Verify that LoRA was applied to target modules
        found_lora_modules = []
        for chunk in adapted_model:
            for name, module in chunk.named_modules():
                if isinstance(module, LoRALinear):
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
        assert efficiency_ratio < 0.3, f"LoRA should be parameter efficient, got ratio: {efficiency_ratio}"

    def test_lora_merge_with_megatron_model(self):
        """Test LoRA merge functionality with Megatron models using pre-wrap hooks."""

        # Create minimal config
        model_provider = GPTModelProvider(
            num_layers=1,
            hidden_size=64,
            num_attention_heads=2,
            vocab_size=100,
            ffn_hidden_size=128,
        )

        # Create LoRA and register hook
        lora = LoRA(dim=4, alpha=8)
        lora_hook = self._create_lora_pre_wrap_hook(lora)
        model_provider.register_pre_wrap_hook(lora_hook)
        model_provider.finalize()

        # Get LoRA-adapted model using hook
        adapted_model = model_provider.provide_distributed_model(ddp_config=None, wrap_with_ddp=False)
        adapted_model = [chunk.cuda() for chunk in adapted_model]

        # Count LoRA modules before merge
        lora_modules_before = 0
        original_weights = {}
        for chunk in adapted_model:
            for name, module in chunk.named_modules():
                if isinstance(module, LoRALinear):
                    lora_modules_before += 1
                    # Store original weights to verify they change after merge
                    original_weights[name] = module.to_wrap.weight.data.clone()

        assert lora_modules_before > 0, "Should have some LoRA modules before merge"

        # Simulate training by making adapter weights non-zero
        # (LoRA adapters start at zero, so merge would be no-op without this)
        for chunk in adapted_model:
            for name, module in chunk.named_modules():
                if isinstance(module, LoRALinear):
                    # Make adapter weights non-zero to simulate training
                    with torch.no_grad():
                        module.adapter.linear_in.weight.data.fill_(0.1)
                        module.adapter.linear_out.weight.data.fill_(0.05)

        # Apply merge
        merge = LoRAMerge()
        merged_model = merge(adapted_model, training=False)

        # Count LoRA modules after merge
        lora_modules_after = 0
        weights_changed = 0
        for chunk in merged_model:
            for name, module in chunk.named_modules():
                if isinstance(module, LoRALinear):
                    lora_modules_after += 1
                    # Check if weights were actually merged (changed)
                    if name in original_weights:
                        if not torch.equal(original_weights[name], module.to_wrap.weight.data):
                            weights_changed += 1

        # LoRAMerge keeps the LoRALinear wrappers but merges the weights
        assert lora_modules_after == lora_modules_before, "LoRAMerge keeps LoRALinear wrappers"
        assert weights_changed > 0, "LoRAMerge should change the underlying weights"

    def test_lora_different_targets(self):
        """Test LoRA with different target module configurations using pre-wrap hooks."""

        # Test different target configurations
        target_configs = [
            ["linear_qkv"],
            ["linear_proj"],
            ["linear_fc1", "linear_fc2"],
            ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"],
        ]

        for targets in target_configs:
            # Create fresh model provider for each configuration
            model_provider = GPTModelProvider(
                num_layers=2,
                hidden_size=64,
                num_attention_heads=2,
                vocab_size=100,
                ffn_hidden_size=128,
            )

            # Create LoRA and register hook
            lora = LoRA(target_modules=targets, dim=4, alpha=8)
            lora_hook = self._create_lora_pre_wrap_hook(lora)
            model_provider.register_pre_wrap_hook(lora_hook)
            model_provider.finalize()

            # Get adapted model using hook
            adapted_model = model_provider.provide_distributed_model(ddp_config=None, wrap_with_ddp=False)
            adapted_model = [chunk.cuda() for chunk in adapted_model]

            # Count LoRA modules
            lora_count = sum(
                1 for chunk in adapted_model for _, module in chunk.named_modules() if isinstance(module, LoRALinear)
            )

            # Should find some LoRA modules for each configuration
            assert lora_count > 0

    def test_lora_transform_idempotent_megatron_model(self):
        """Test that LoRA transform is idempotent when applied via pre-wrap hooks."""
        # Create a minimal GPT configuration
        model_provider = GPTModelProvider(
            num_layers=1,
            hidden_size=64,
            num_attention_heads=2,
            vocab_size=100,
            ffn_hidden_size=128,
        )

        # Create LoRA instance
        lora = LoRA(target_modules=["linear_qkv", "linear_proj"], dim=4, alpha=8)

        # Register hook and apply LoRA first time
        lora_hook = self._create_lora_pre_wrap_hook(lora)
        model_provider.register_pre_wrap_hook(lora_hook)
        model_provider.finalize()
        first_transform = model_provider.provide_distributed_model(ddp_config=None, wrap_with_ddp=False)

        first_transform = [chunk.cuda() for chunk in first_transform]

        # Store references to the transformed model chunks
        first_chunks = [chunk for chunk in first_transform]

        # Verify we got LoRA modules in the first transformation
        found_lora_modules_first = []
        for chunk in first_transform:
            for name, module in chunk.named_modules():
                if isinstance(module, LoRALinear):
                    found_lora_modules_first.append((chunk, name, module))

        assert len(found_lora_modules_first) > 0, "Should have found LoRA modules in first transformation"

        # Apply LoRA second time to the already-transformed model
        # Note: In the pre-wrap hook pattern, we need to apply LoRA directly since
        # the model provider has already been called
        second_transform = lora(first_transform, training=True)

        # Verify idempotency: should return the same model chunks
        assert len(second_transform) == len(first_transform)
        for i, (first_chunk, second_chunk) in enumerate(zip(first_chunks, second_transform)):
            assert second_chunk is first_chunk, f"Chunk {i} should be identical object"

        # Verify LoRA modules are identical objects
        found_lora_modules_second = []
        for chunk in second_transform:
            for name, module in chunk.named_modules():
                if isinstance(module, LoRALinear):
                    found_lora_modules_second.append((chunk, name, module))

        # Should have same number of LoRA modules
        assert len(found_lora_modules_second) == len(found_lora_modules_first)

        # Each LoRA module should be the identical object
        for (first_chunk, first_name, first_module), (second_chunk, second_name, second_module) in zip(
            found_lora_modules_first, found_lora_modules_second
        ):
            assert first_chunk is second_chunk
            assert first_name == second_name
            assert second_module is first_module, f"LoRA module {first_name} should be identical object"


class MockVLMModel(nn.Module):
    """Mock Vision-Language Model for testing VLMLoRA."""

    def __init__(self, has_vision=True, has_projection=True, has_language=True):
        super().__init__()
        self.vision_model = nn.Sequential(nn.Linear(512, 512), nn.ReLU()) if has_vision else None
        self.vision_projection = nn.Linear(512, 768) if has_projection else None
        self.language_model = nn.Sequential(nn.Linear(768, 768), nn.ReLU()) if has_language else None

    def parameters(self):
        """Override to provide all parameters."""
        params = []
        if self.vision_model:
            params.extend(self.vision_model.parameters())
        if self.vision_projection:
            params.extend(self.vision_projection.parameters())
        if self.language_model:
            params.extend(self.language_model.parameters())
        return iter(params)


class MockLLaVAWrapper(nn.Module):
    """Mock wrapper model with llava_model attribute."""

    def __init__(self, llava_model):
        super().__init__()
        self.llava_model = llava_model


class TestVLMLoRA:
    """Test suite for VLMLoRA PEFT implementation."""

    def test_vlmlora_initialization(self):
        """Test VLMLoRA class initialization with default and custom parameters."""
        # Test default initialization
        vlm_lora = VLMLoRA()
        assert vlm_lora.freeze_vision_model is True
        assert vlm_lora.freeze_vision_projection is True
        assert vlm_lora.freeze_language_model is True
        # Check inherited LoRA properties
        assert vlm_lora.target_modules == ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]
        assert vlm_lora.dim == 32
        assert vlm_lora.alpha == 32

        # Test custom initialization
        custom_vlm_lora = VLMLoRA(
            freeze_vision_model=False,
            freeze_vision_projection=False,
            freeze_language_model=True,
            dim=16,
            alpha=16,
        )
        assert custom_vlm_lora.freeze_vision_model is False
        assert custom_vlm_lora.freeze_vision_projection is False
        assert custom_vlm_lora.freeze_language_model is True
        assert custom_vlm_lora.dim == 16
        assert custom_vlm_lora.alpha == 16

    def test_freeze_all_components(self):
        """Test freezing all components of a VLM model."""
        model = MockVLMModel(has_vision=True, has_projection=True, has_language=True)
        vlm_lora = VLMLoRA(freeze_vision_model=True, freeze_vision_projection=True, freeze_language_model=True)

        # Set all parameters to require gradients initially
        for param in model.parameters():
            param.requires_grad = True

        # Apply freeze
        with patch("megatron.bridge.peft.lora.unwrap_model") as mock_unwrap:
            mock_unwrap.return_value = [model]
            vlm_lora.freeze_model(model, training=True)

        # Verify all parameters are frozen
        for param in model.vision_model.parameters():
            assert not param.requires_grad, "Vision model parameters should be frozen"
        for param in model.vision_projection.parameters():
            assert not param.requires_grad, "Vision projection parameters should be frozen"
        for param in model.language_model.parameters():
            assert not param.requires_grad, "Language model parameters should be frozen"

    def test_freeze_vision_only(self):
        """Test freezing only the vision model."""
        model = MockVLMModel(has_vision=True, has_projection=True, has_language=True)
        vlm_lora = VLMLoRA(freeze_vision_model=True, freeze_vision_projection=False, freeze_language_model=False)

        # Set all parameters to require gradients initially
        for param in model.parameters():
            param.requires_grad = True

        # Apply freeze
        with patch("megatron.bridge.peft.lora.unwrap_model") as mock_unwrap:
            mock_unwrap.return_value = [model]
            vlm_lora.freeze_model(model, training=True)

        # Verify only vision model is frozen
        for param in model.vision_model.parameters():
            assert not param.requires_grad, "Vision model parameters should be frozen"
        for param in model.vision_projection.parameters():
            assert param.requires_grad, "Vision projection parameters should NOT be frozen"
        for param in model.language_model.parameters():
            assert param.requires_grad, "Language model parameters should NOT be frozen"

    def test_freeze_language_only(self):
        """Test freezing only the language model."""
        model = MockVLMModel(has_vision=True, has_projection=True, has_language=True)
        vlm_lora = VLMLoRA(freeze_vision_model=False, freeze_vision_projection=False, freeze_language_model=True)

        # Set all parameters to require gradients initially
        for param in model.parameters():
            param.requires_grad = True

        # Apply freeze
        with patch("megatron.bridge.peft.lora.unwrap_model") as mock_unwrap:
            mock_unwrap.return_value = [model]
            vlm_lora.freeze_model(model, training=True)

        # Verify only language model is frozen
        for param in model.vision_model.parameters():
            assert param.requires_grad, "Vision model parameters should NOT be frozen"
        for param in model.vision_projection.parameters():
            assert param.requires_grad, "Vision projection parameters should NOT be frozen"
        for param in model.language_model.parameters():
            assert not param.requires_grad, "Language model parameters should be frozen"

    def test_freeze_no_components(self):
        """Test that no freezing occurs when all flags are False."""
        model = MockVLMModel(has_vision=True, has_projection=True, has_language=True)
        vlm_lora = VLMLoRA(freeze_vision_model=False, freeze_vision_projection=False, freeze_language_model=False)

        # Set all parameters to require gradients initially
        for param in model.parameters():
            param.requires_grad = True

        # Apply freeze
        with patch("megatron.bridge.peft.lora.unwrap_model") as mock_unwrap:
            mock_unwrap.return_value = [model]
            vlm_lora.freeze_model(model, training=True)

        # Verify no parameters are frozen
        for param in model.vision_model.parameters():
            assert param.requires_grad, "Vision model parameters should NOT be frozen"
        for param in model.vision_projection.parameters():
            assert param.requires_grad, "Vision projection parameters should NOT be frozen"
        for param in model.language_model.parameters():
            assert param.requires_grad, "Language model parameters should NOT be frozen"

    def test_freeze_with_missing_components(self):
        """Test freezing when some model components are None."""
        # Model with only language model
        model = MockVLMModel(has_vision=False, has_projection=False, has_language=True)
        vlm_lora = VLMLoRA(freeze_vision_model=True, freeze_vision_projection=True, freeze_language_model=True)

        # Set language model parameters to require gradients
        for param in model.language_model.parameters():
            param.requires_grad = True

        # Apply freeze (should not crash even though vision components are None)
        with patch("megatron.bridge.peft.lora.unwrap_model") as mock_unwrap:
            mock_unwrap.return_value = [model]
            vlm_lora.freeze_model(model, training=True)

        # Verify language model is frozen
        for param in model.language_model.parameters():
            assert not param.requires_grad, "Language model parameters should be frozen"

    def test_freeze_with_llava_wrapper(self):
        """Test that freeze_model correctly unwraps llava_model attribute."""
        inner_model = MockVLMModel(has_vision=True, has_projection=True, has_language=True)
        wrapper_model = MockLLaVAWrapper(inner_model)
        vlm_lora = VLMLoRA(freeze_vision_model=True, freeze_vision_projection=True, freeze_language_model=True)

        # Set all parameters to require gradients initially
        for param in inner_model.parameters():
            param.requires_grad = True

        # Apply freeze
        with patch("megatron.bridge.peft.lora.unwrap_model") as mock_unwrap:
            mock_unwrap.return_value = [wrapper_model]
            vlm_lora.freeze_model(wrapper_model, training=True)

        # Verify all parameters are frozen (should have unwrapped to inner_model)
        for param in inner_model.vision_model.parameters():
            assert not param.requires_grad, "Vision model parameters should be frozen"
        for param in inner_model.vision_projection.parameters():
            assert not param.requires_grad, "Vision projection parameters should be frozen"
        for param in inner_model.language_model.parameters():
            assert not param.requires_grad, "Language model parameters should be frozen"

    def test_training_mode_enabled(self):
        """Test that model is set to training mode when training=True."""
        model = MockVLMModel(has_vision=True, has_projection=True, has_language=True)
        model.eval()  # Start in eval mode
        vlm_lora = VLMLoRA()

        with patch("megatron.bridge.peft.lora.unwrap_model") as mock_unwrap:
            mock_unwrap.return_value = [model]
            vlm_lora.freeze_model(model, training=True)

        assert model.training, "Model should be in training mode after freeze_model with training=True"

    def test_training_mode_disabled(self):
        """Test that model training mode is not changed when training=False."""
        model = MockVLMModel(has_vision=True, has_projection=True, has_language=True)
        model.eval()  # Start in eval mode
        vlm_lora = VLMLoRA()

        with patch("megatron.bridge.peft.lora.unwrap_model") as mock_unwrap:
            mock_unwrap.return_value = [model]
            vlm_lora.freeze_model(model, training=False)

        assert not model.training, "Model should remain in eval mode after freeze_model with training=False"

    def test_vlmlora_inherits_transform(self):
        """Test that VLMLoRA inherits the transform method from LoRA."""
        vlm_lora = VLMLoRA(target_modules=["linear_qkv"])
        model = SimpleModel()

        # Apply transform to a target module
        transformed = vlm_lora.transform(model.linear_qkv, name="linear_qkv", prefix="model")

        # Should be wrapped with LoRA adapter
        assert isinstance(transformed, (LoRALinear, LinearAdapter)), (
            "VLMLoRA should inherit transform from LoRA and wrap modules"
        )

    def test_vlmlora_selective_freezing_combination(self):
        """Test various combinations of selective freezing."""
        test_cases = [
            (True, False, False),  # Freeze vision only
            (False, True, False),  # Freeze projection only
            (False, False, True),  # Freeze language only
            (True, True, False),  # Freeze vision + projection
            (True, False, True),  # Freeze vision + language
            (False, True, True),  # Freeze projection + language
        ]

        for freeze_vision, freeze_projection, freeze_language in test_cases:
            model = MockVLMModel(has_vision=True, has_projection=True, has_language=True)
            vlm_lora = VLMLoRA(
                freeze_vision_model=freeze_vision,
                freeze_vision_projection=freeze_projection,
                freeze_language_model=freeze_language,
            )

            # Set all parameters to require gradients
            for param in model.parameters():
                param.requires_grad = True

            with patch("megatron.bridge.peft.lora.unwrap_model") as mock_unwrap:
                mock_unwrap.return_value = [model]
                vlm_lora.freeze_model(model, training=True)

            # Verify freezing matches expectations
            for param in model.vision_model.parameters():
                assert param.requires_grad == (not freeze_vision), f"Vision params wrong for config {test_cases}"
            for param in model.vision_projection.parameters():
                assert param.requires_grad == (not freeze_projection), (
                    f"Projection params wrong for config {test_cases}"
                )
            for param in model.language_model.parameters():
                assert param.requires_grad == (not freeze_language), f"Language params wrong for config {test_cases}"
