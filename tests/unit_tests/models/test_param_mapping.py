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

from unittest.mock import patch

import pytest
import torch
from megatron.core.transformer.transformer_config import TransformerConfig

from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ColumnParallelMapping,
    DirectMapping,
    FusedExpertMapping,
    FusedGatedExpertMapping,
    GatedMLPMapping,
    KVMapping,
    QKVMapping,
    ReplicatedMapping,
    RowParallelMapping,
    merge_kv_biases,
    merge_kv_weights,
    merge_qkv_biases,
    merge_qkv_weights,
    split_kv_biases,
    split_kv_weights,
    split_qkv_biases,
    split_qkv_weights,
)


@pytest.fixture
def mock_distributed_env():
    """Mocks the distributed environment for single-process testing."""
    with (
        patch("megatron.bridge.models.conversion.param_mapping.mpu") as mock_mpu,
        patch("torch.distributed") as mock_dist,
        patch("torch.cuda.current_device", return_value=0),
    ):

        def setup_mocks(tp_size=1, tp_rank=0, pp_size=1, pp_rank=0):
            # Ensure Megatron and torch.distributed appear initialized
            mock_mpu.is_initialized.return_value = True
            mock_dist.is_initialized.return_value = True

            # Simple process group mock with size() and rank()
            class _MockGroup:
                def __init__(self, size, rank):
                    self._size = size
                    self._rank = rank

                def size(self):
                    return self._size

                def rank(self):
                    return self._rank

            tp_group = _MockGroup(tp_size, tp_rank)
            pp_group = _MockGroup(pp_size, pp_rank)

            mock_mpu.get_tensor_model_parallel_world_size.return_value = tp_size
            mock_mpu.get_tensor_model_parallel_rank.return_value = tp_rank
            mock_mpu.get_pipeline_model_parallel_world_size.return_value = pp_size
            mock_mpu.get_pipeline_model_parallel_rank.return_value = pp_rank
            mock_mpu.get_tensor_model_parallel_group.return_value = tp_group
            mock_mpu.get_pipeline_model_parallel_group.return_value = pp_group

            # Utility fns used by mapping helpers
            mock_dist.get_global_rank.side_effect = lambda group, group_rank: group_rank
            mock_dist.get_process_group_ranks.side_effect = lambda group: list(range(group.size()))
            return mock_mpu, mock_dist

        yield setup_mocks


@pytest.fixture
def transformer_config():
    """Provides a sample TransformerConfig."""
    return TransformerConfig(
        num_layers=2,
        hidden_size=32,
        num_attention_heads=4,
        kv_channels=8,
        ffn_hidden_size=128,
        use_cpu_initialization=True,
        num_query_groups=2,
    )


class MockModule(torch.nn.Module):
    """A mock nn.Module for testing purposes."""

    def __init__(self, config, weight_shape=(16, 16), has_bias=False, device="cpu"):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(weight_shape, device=device))
        if has_bias:
            self.bias = torch.nn.Parameter(torch.randn(weight_shape[0], device=device))
        self.config = config


class TestDirectMapping:
    def test_hf_to_megatron(self, mock_distributed_env, transformer_config):
        mock_distributed_env()
        mapping = DirectMapping("megatron.weight", "hf.weight")
        hf_weight = torch.randn(16, 16)
        megatron_module = MockModule(transformer_config)
        megatron_weight = mapping.hf_to_megatron(hf_weight, megatron_module)
        assert torch.equal(megatron_weight, hf_weight)

    def test_megatron_to_hf(self, mock_distributed_env):
        mock_distributed_env()
        mapping = DirectMapping("megatron.weight", "hf.weight")
        megatron_weight = torch.randn(16, 16)
        hf_weights = mapping.megatron_to_hf(megatron_weight, None)
        assert "hf.weight" in hf_weights
        assert torch.equal(hf_weights["hf.weight"], megatron_weight)


class TestReplicatedMapping:
    @pytest.mark.parametrize("tp_rank", [0, 1])
    def test_megatron_to_hf_tp_gt_1(self, mock_distributed_env, tp_rank):
        mock_mpu, _ = mock_distributed_env(tp_size=2, tp_rank=tp_rank)
        mapping = ReplicatedMapping("rep.weight", "hf.weight")
        megatron_weight = torch.randn(16, 16)
        result = mapping.megatron_to_hf(megatron_weight, None)

        assert "hf.weight" in result
        assert torch.equal(result["hf.weight"], megatron_weight)

    def test_hf_to_megatron_broadcast(self, mock_distributed_env, transformer_config):
        mock_mpu, mock_dist = mock_distributed_env(tp_size=2, tp_rank=0)
        mapping = ReplicatedMapping("rep.weight", "hf.weight")
        hf_weight = torch.randn(16, 16)
        megatron_module = MockModule(transformer_config, weight_shape=(16, 16))

        def mock_broadcast(tensor, src, group):
            pass  # Just pass through for testing

        mock_dist.broadcast.side_effect = mock_broadcast

        with patch.object(mapping, "broadcast_tensor_to_tp_ranks", return_value=hf_weight) as mock_broadcast_method:
            result = mapping.hf_to_megatron(hf_weight, megatron_module)
            # Verify the method was called once
            mock_broadcast_method.assert_called_once()

            # Check the arguments more robustly
            args, kwargs = mock_broadcast_method.call_args
            called_tensor = args[0]
            assert "src_rank" in kwargs
            assert kwargs["src_rank"] == 0

            # Verify the tensor shapes match and values are the same (accounting for device movement)
            assert called_tensor.shape == hf_weight.shape
            assert torch.equal(called_tensor.cpu(), hf_weight.cpu())
            assert torch.equal(result, hf_weight)


class TestColumnParallelMapping:
    @pytest.mark.parametrize("tp_rank", [0, 1])
    def test_hf_to_megatron(self, mock_distributed_env, transformer_config, tp_rank):
        _, mock_dist = mock_distributed_env(tp_size=2, tp_rank=tp_rank)
        mapping = ColumnParallelMapping("col.weight", "hf.weight")
        megatron_module = MockModule(transformer_config, weight_shape=(16, 16))
        # Create the full weight to simulate distributed scatter
        full_weight = torch.randn(32, 16)
        hf_weight = full_weight if tp_rank == 0 else None

        def mock_scatter(output, scatter_list, src, group):
            if tp_rank == 0:
                output.copy_(scatter_list[0])
            else:
                # On non-src ranks, scatter_list is None. The mapping handles this.
                # Here we simulate receiving the data.
                output.copy_(torch.chunk(full_weight, 2, dim=0)[tp_rank])

        mock_dist.scatter.side_effect = mock_scatter
        megatron_weight = mapping.hf_to_megatron(hf_weight, megatron_module)
        assert megatron_weight.shape == (16, 16)

        if tp_rank == 0:
            call_args = mock_dist.scatter.call_args[0]
            assert torch.equal(torch.cat(call_args[1], dim=0), hf_weight)

    @pytest.mark.parametrize("tp_rank", [0, 1])
    def test_megatron_to_hf(self, mock_distributed_env, tp_rank):
        mock_distributed_env(tp_size=2, tp_rank=tp_rank)
        mapping = ColumnParallelMapping("col.weight", "hf.weight")
        megatron_shard = torch.randn(16, 16)

        with patch.object(mapping, "gather_from_tp_ranks") as mock_gather:
            full_weight = torch.randn(32, 16)
            mock_gather.return_value = list(torch.chunk(full_weight, 2, dim=0))
            result = mapping.megatron_to_hf(megatron_shard, None)

            assert "hf.weight" in result
            assert torch.equal(result["hf.weight"], full_weight)


class TestRowParallelMapping:
    @pytest.mark.parametrize("tp_rank", [0, 1])
    def test_hf_to_megatron(self, mock_distributed_env, transformer_config, tp_rank):
        _, mock_dist = mock_distributed_env(tp_size=2, tp_rank=tp_rank)
        mapping = RowParallelMapping("row.weight", "hf.weight")
        megatron_module = MockModule(transformer_config, weight_shape=(16, 16))
        # Create the full weight to simulate distributed scatter
        full_weight = torch.randn(16, 32)
        hf_weight = full_weight if tp_rank == 0 else None

        def mock_scatter(output, scatter_list, src, group):
            if tp_rank == 0:
                output.copy_(scatter_list[0])
            else:
                output.copy_(torch.chunk(full_weight, 2, dim=1)[tp_rank])

        mock_dist.scatter.side_effect = mock_scatter
        megatron_weight = mapping.hf_to_megatron(hf_weight, megatron_module)
        assert megatron_weight.shape == (16, 16)

    @pytest.mark.parametrize("tp_rank", [0, 1])
    def test_megatron_to_hf(self, mock_distributed_env, tp_rank):
        mock_distributed_env(tp_size=2, tp_rank=tp_rank)
        mapping = RowParallelMapping("row.weight", "hf.weight")
        megatron_shard = torch.randn(16, 16)

        with patch.object(mapping, "gather_from_tp_ranks") as mock_gather:
            full_weight = torch.randn(16, 32)
            mock_gather.return_value = list(torch.chunk(full_weight, 2, dim=1))
            result = mapping.megatron_to_hf(megatron_shard, None)

            assert "hf.weight" in result
            assert torch.equal(result["hf.weight"], full_weight)


class TestAutoMapping:
    def test_detect_parallelism_type(self, mock_distributed_env, transformer_config):
        mock_distributed_env()
        mapping = AutoMapping(megatron_param="some.weight", hf_param="hf.weight")

        # Mock modules with different characteristics
        class MyCol(torch.nn.Module):
            tensor_model_parallel = True
            partition_dim = 0

        class MyRow(torch.nn.Module):
            tensor_model_parallel = True
            partition_dim = 1

        class MyRep(torch.nn.Module):
            tensor_model_parallel = False

        AutoMapping.register_module_type("MyCustomRow", "row")

        class MyCustomRow(torch.nn.Module):
            pass

        assert mapping._detect_parallelism_type(MyCol()) == "column"
        assert mapping._detect_parallelism_type(MyRow()) == "row"
        assert mapping._detect_parallelism_type(MyRep()) == "replicated"
        assert mapping._detect_parallelism_type(torch.nn.LayerNorm(5)) == "replicated"
        assert mapping._detect_parallelism_type(MyCustomRow()) == "row"

        with pytest.raises(ValueError):
            mapping._detect_parallelism_type(torch.nn.Linear(5, 5))

    def test_detect_parallelism_type_dynamic_module(self):
        mtq = pytest.importorskip("modelopt.torch.quantization")
        DynamicModule = pytest.importorskip("modelopt.torch.opt.dynamic").DynamicModule

        class Wrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                return self.linear(x)

        model = Wrapper()
        mtq.quantize(model, mtq.INT8_DEFAULT_CFG)
        quantized_linear = model.linear

        assert isinstance(quantized_linear, DynamicModule)
        assert type(quantized_linear).__name__ == "QuantLinear"
        assert quantized_linear.get_original_cls_by_level(level=0).__name__ == "Linear"

        AutoMapping.register_module_type("Linear", "column")
        try:
            mapping = AutoMapping(megatron_param="some.weight", hf_param="hf.weight")
            result = mapping._detect_parallelism_type(quantized_linear)
            assert result == "column"
        finally:
            AutoMapping._MODULE_TYPE_REGISTRY["column"].discard("Linear")


class TestHelperFunctions:
    def test_qkv_merge_split(self, transformer_config):
        q = torch.randn(32, 32)
        k = torch.randn(16, 32)
        v = torch.randn(16, 32)

        merged = merge_qkv_weights(transformer_config, q, k, v)
        assert merged.shape == (32 + 16 + 16, 32)

        q_s, k_s, v_s = split_qkv_weights(transformer_config, merged)
        assert torch.equal(q, q_s)
        assert torch.equal(k, k_s)
        assert torch.equal(v, v_s)

    def test_kv_merge_split(self, transformer_config):
        # k, v each [16, hidden_size] with hidden_size=32 in fixture
        k = torch.randn(16, 32)
        v = torch.randn(16, 32)

        merged = merge_kv_weights(transformer_config, k, v)
        # Expect stacked along dim 0: (16 + 16, 32)
        assert merged.shape == (32, 32)

        k_s, v_s = split_kv_weights(transformer_config, merged)
        assert torch.equal(k, k_s)
        assert torch.equal(v, v_s)


class TestQKVMapping:
    def test_hf_to_megatron(self, mock_distributed_env, transformer_config):
        mock_distributed_env()
        mapping = QKVMapping(megatron_param="qkv.weight", q="q.weight", k="k.weight", v="v.weight")
        weights = {
            "q": torch.randn(32, 32),
            "k": torch.randn(16, 32),
            "v": torch.randn(16, 32),
        }
        megatron_module = MockModule(transformer_config, weight_shape=(64, 32))

        with patch.object(mapping._tp_mapping, "hf_to_megatron") as mock_hf_to_megatron:
            mapping.hf_to_megatron(weights, megatron_module)
            mock_hf_to_megatron.assert_called_once()
            merged_weight = mock_hf_to_megatron.call_args[0][0]
            assert merged_weight.shape == (64, 32)


class TestKVMapping:
    def test_hf_to_megatron(self, mock_distributed_env, transformer_config):
        mock_distributed_env()
        mapping = KVMapping(megatron_param="kv.weight", k="k.weight", v="v.weight")
        weights = {
            "k": torch.randn(16, 32),
            "v": torch.randn(16, 32),
        }
        megatron_module = MockModule(transformer_config, weight_shape=(32, 32))

        with patch.object(mapping._tp_mapping, "hf_to_megatron") as mock_hf_to_megatron:
            mapping.hf_to_megatron(weights, megatron_module)
            mock_hf_to_megatron.assert_called_once()
            merged_weight = mock_hf_to_megatron.call_args[0][0]
            # Should match merge_kv_weights result and shape
            expected = merge_kv_weights(transformer_config, weights["k"], weights["v"])
            assert merged_weight.shape == (32, 32)
            assert torch.equal(merged_weight, expected)

    def test_megatron_to_hf(self, mock_distributed_env, transformer_config):
        mock_distributed_env()
        mapping = KVMapping(megatron_param="kv.weight", k="k.weight", v="v.weight")
        # Construct packed KV via helper to guarantee split reversibility
        k = torch.randn(16, 32)
        v = torch.randn(16, 32)
        packed_kv = merge_kv_weights(transformer_config, k, v)
        megatron_module = MockModule(transformer_config, weight_shape=(32, 32))

        with patch.object(mapping._tp_mapping, "megatron_to_hf", return_value={"kv.weight": packed_kv}):
            result = mapping.megatron_to_hf(packed_kv, megatron_module)

        assert "k.weight" in result and "v.weight" in result
        assert result["k.weight"].shape == (16, 32)
        assert result["v.weight"].shape == (16, 32)
        assert torch.equal(result["k.weight"], k)
        assert torch.equal(result["v.weight"], v)


class TestGatedMLPMapping:
    def test_hf_to_megatron_single_tp(self, mock_distributed_env, transformer_config):
        """Test gate+up merging with single TP rank."""
        mock_distributed_env()
        mapping = GatedMLPMapping(megatron_param="gated.weight", gate="gate.weight", up="up.weight")
        weights = {
            "gate": torch.randn(128, 32),
            "up": torch.randn(128, 32),
        }
        megatron_module = MockModule(transformer_config, weight_shape=(256, 32))

        # Test single TP case
        result = mapping.hf_to_megatron(weights, megatron_module)
        assert result.shape == (256, 32)

        # Verify that gate and up are properly concatenated
        expected = torch.cat([weights["gate"], weights["up"]], dim=0)
        assert torch.equal(result, expected)

    def test_hf_to_megatron_multi_tp(self, mock_distributed_env, transformer_config):
        """Test gate+up merging with multiple TP ranks."""
        # Test with TP size = 2, rank 0
        mock_mpu, mock_dist = mock_distributed_env(tp_size=2, tp_rank=0)
        mapping = GatedMLPMapping(megatron_param="gated.weight", gate="gate.weight", up="up.weight")

        # Create test weights - each component 128x32, so full concat will be 256x32
        # Each TP rank should get 128x32 (half of each component concatenated)
        weights = {
            "gate": torch.randn(128, 32),
            "up": torch.randn(128, 32),
        }
        megatron_module = MockModule(transformer_config, weight_shape=(128, 32))  # Each rank gets half

        with patch.object(mapping, "scatter_to_tp_ranks") as mock_scatter:
            mock_scatter.return_value = torch.randn(128, 32)
            mapping.hf_to_megatron(weights, megatron_module)

            # Verify scatter was called with proper splits
            mock_scatter.assert_called_once()
            call_args = mock_scatter.call_args[0]
            splits = call_args[0]  # First argument should be the splits

            # Should have 2 splits (one per TP rank)
            assert len(splits) == 2
            # Each split should be concatenated [gate_shard; up_shard]
            assert splits[0].shape == (
                128,
                32,
            )  # Half of 128 gate + half of 128 up = 64 + 64 = 128
            assert splits[1].shape == (128, 32)

    def test_megatron_to_hf_single_tp(self, mock_distributed_env, transformer_config):
        """Test splitting concatenated weights back to gate+up with single TP."""
        mock_distributed_env()
        mapping = GatedMLPMapping(megatron_param="gated.weight", gate="gate.weight", up="up.weight")

        # Create a concatenated tensor [gate; up]
        gate = torch.randn(128, 32)
        up = torch.randn(128, 32)
        merged_weight = torch.cat([gate, up], dim=0)
        megatron_module = MockModule(transformer_config, weight_shape=(256, 32))

        result = mapping.megatron_to_hf(merged_weight, megatron_module)

        assert "gate.weight" in result
        assert "up.weight" in result
        assert result["gate.weight"].shape == (128, 32)
        assert result["up.weight"].shape == (128, 32)

        # Verify the split is correct
        assert torch.equal(result["gate.weight"], gate)
        assert torch.equal(result["up.weight"], up)

    def test_hf_to_megatron_bias_single_tp(self, mock_distributed_env, transformer_config):
        """Test gate+up bias merging with single TP rank."""
        mock_distributed_env()
        mapping = GatedMLPMapping(megatron_param="gated.bias", gate="gate.bias", up="up.bias")
        weights = {
            "gate": torch.randn(128),  # 1D bias tensors
            "up": torch.randn(128),
        }
        megatron_module = MockModule(transformer_config, weight_shape=(256, 32), has_bias=True)
        # Override bias shape to match expected concatenated size
        megatron_module.bias = torch.nn.Parameter(torch.randn(256))

        # Test single TP case for bias
        result = mapping.hf_to_megatron(weights, megatron_module)
        assert result.shape == (256,)  # Concatenated bias shape

        # Verify that gate and up biases are properly concatenated
        expected = torch.cat([weights["gate"], weights["up"]], dim=0)
        assert torch.equal(result, expected)

    def test_megatron_to_hf_bias_single_tp(self, mock_distributed_env, transformer_config):
        """Test splitting concatenated bias back to gate+up with single TP."""
        mock_distributed_env()
        mapping = GatedMLPMapping(megatron_param="gated.bias", gate="gate.bias", up="up.bias")

        # Create concatenated bias tensor [gate_bias; up_bias]
        gate_bias = torch.randn(128)
        up_bias = torch.randn(128)
        merged_bias = torch.cat([gate_bias, up_bias], dim=0)
        megatron_module = MockModule(transformer_config, weight_shape=(256, 32), has_bias=True)
        megatron_module.bias = torch.nn.Parameter(torch.randn(256))

        result = mapping.megatron_to_hf(merged_bias, megatron_module)

        assert "gate.bias" in result
        assert "up.bias" in result
        assert result["gate.bias"].shape == (128,)
        assert result["up.bias"].shape == (128,)

        # Verify the split is correct
        assert torch.equal(result["gate.bias"], gate_bias)
        assert torch.equal(result["up.bias"], up_bias)


class TestMappingEdgeCases:
    """Test edge cases and error handling in param mappings."""

    def test_wildcard_pattern_validation(self):
        """Test that wildcard patterns are validated correctly."""
        # Valid patterns - should not raise
        DirectMapping("layer.*.weight", "model.*.weight")
        QKVMapping(
            megatron_param="*.qkv.weight",
            q="*.q_proj.weight",
            k="*.k_proj.weight",
            v="*.v_proj.weight",
        )

        # Invalid patterns - mismatched wildcard counts
        with pytest.raises(ValueError, match="Wildcard count mismatch"):
            DirectMapping("layer.*.*.weight", "model.*.weight")

        with pytest.raises(ValueError, match="Wildcard count mismatch"):
            QKVMapping(
                "*.qkv.weight",
                q="*.*.q_proj.weight",
                k="*.k_proj.weight",
                v="*.v_proj.weight",
            )

    def test_qkv_bias_handling(self, transformer_config):
        """Test QKV mapping handles biases correctly."""
        # Test bias merging
        q_bias = torch.randn(32)
        k_bias = torch.randn(16)
        v_bias = torch.randn(16)

        merged = merge_qkv_biases(transformer_config, q_bias, k_bias, v_bias)
        assert merged.shape == (64,)  # 32 + 16 + 16

        # Test bias splitting
        q_split, k_split, v_split = split_qkv_biases(transformer_config, merged)
        assert torch.equal(q_bias, q_split)
        assert torch.equal(k_bias, k_split)
        assert torch.equal(v_bias, v_split)

    def test_kv_bias_handling(self, transformer_config):
        """Test KV helpers handle biases correctly."""
        # num_query_groups=2, kv_channels=8 -> each bias length 16
        k_bias = torch.randn(16)
        v_bias = torch.randn(16)

        merged = merge_kv_biases(transformer_config, k_bias, v_bias)
        assert merged.shape == (32,)

        k_split, v_split = split_kv_biases(transformer_config, merged)
        assert torch.equal(k_bias, k_split)
        assert torch.equal(v_bias, v_split)

    def test_column_parallel_bias_handling(self, mock_distributed_env, transformer_config):
        """Test column parallel mapping handles biases correctly."""
        _, mock_dist = mock_distributed_env(tp_size=2, tp_rank=0)
        mapping = ColumnParallelMapping("col.bias", "hf.bias")

        # Create a module with bias
        class MockModuleWithBias(torch.nn.Module):
            def __init__(self, config):
                super().__init__()
                self.bias = torch.nn.Parameter(torch.randn(16))
                self.config = config

        megatron_module = MockModuleWithBias(transformer_config)
        hf_bias = torch.randn(32)

        def mock_scatter(output, scatter_list, src, group):
            if scatter_list:
                output.copy_(scatter_list[0])

        mock_dist.scatter.side_effect = mock_scatter

        # Test bias distribution
        megatron_bias = mapping.hf_to_megatron(hf_bias, megatron_module)
        assert megatron_bias.shape == (16,)

    def test_broadcast_from_pp_rank_error_handling(self, mock_distributed_env):
        """Test PP broadcast error handling."""
        mock_distributed_env(pp_size=2, pp_rank=0)
        mapping = DirectMapping("weight", "weight")

        # Test when no rank has the tensor
        with patch("torch.distributed.all_gather_object") as mock_gather:
            mock_gather.side_effect = lambda output, obj, group: output.__setitem__(slice(None), [None, None])

            with pytest.raises(ValueError, match="Object must exist on at least one PP rank"):
                mapping.broadcast_from_pp_rank(None)

    def test_broadcast_from_pp_rank_multi_owner(self, mock_distributed_env):
        """Test PP broadcast handles tensors present on multiple PP ranks.

        MLA (Multi-Latent Attention) architectures such as DeepSeek-V3 and
        MTP models can place the same weight tensor on more than one PP stage.
        broadcast_from_pp_rank must pick the first owner deterministically
        rather than raising ValueError.
        """
        _, mock_dist = mock_distributed_env(pp_size=2, pp_rank=0)
        mapping = DirectMapping("weight", "weight")

        tensor = torch.randn(16, 16)
        spec = (tensor.shape, tensor.dtype, None, None)

        # Simulate both PP ranks owning the tensor
        mock_dist.all_gather_object.side_effect = lambda output, obj, group: output.__setitem__(
            slice(None), [spec, spec]
        )
        mock_dist.broadcast.side_effect = lambda t, src, group: None

        # Must not raise — should pick rank 0 as source
        result = mapping.broadcast_from_pp_rank(tensor)
        assert result is not None

        # Verify broadcast was called with src=rank 0
        mock_dist.broadcast.assert_called_once()
        call_kwargs = mock_dist.broadcast.call_args
        assert call_kwargs[1]["src"] == 0 or call_kwargs[0][1] == 0

    def test_broadcast_from_pp_rank_multi_owner_with_cache(self, mock_distributed_env):
        """Test PP broadcast with cache_key when tensor exists on multiple ranks.

        Every real mapping calls broadcast_from_pp_rank with
        cache_key=str(self.hf_param). Verify that the cached path also
        handles multi-owner tensors correctly and that the second call
        skips the all_gather_object collective.
        """
        _, mock_dist = mock_distributed_env(pp_size=2, pp_rank=0)
        mapping = DirectMapping("weight", "weight")

        tensor = torch.randn(16, 16)
        spec = (tensor.shape, tensor.dtype, None, None)

        mock_dist.all_gather_object.side_effect = lambda output, obj, group: output.__setitem__(
            slice(None), [spec, spec]
        )
        mock_dist.broadcast.side_effect = lambda t, src, group: None

        cache_key = "model.layers.0.self_attn.kv_b_proj.weight"

        # First call — populates cache
        result1 = mapping.broadcast_from_pp_rank(tensor, cache_key=cache_key)
        assert result1 is not None
        assert mock_dist.all_gather_object.call_count == 1

        # Second call with same cache_key — must reuse cached spec
        result2 = mapping.broadcast_from_pp_rank(tensor, cache_key=cache_key)
        assert result2 is not None
        # all_gather_object should NOT be called again
        assert mock_dist.all_gather_object.call_count == 1
        # broadcast should be called twice (once per call)
        assert mock_dist.broadcast.call_count == 2

    def test_broadcast_obj_from_pp_rank_multi_owner(self, mock_distributed_env):
        """Test PP object broadcast handles objects present on multiple PP ranks.

        Similar to tensor broadcast, shared objects must not cause errors and
        the first owning rank must be selected deterministically.
        """
        _, mock_dist = mock_distributed_env(pp_size=2, pp_rank=0)
        mapping = DirectMapping("weight", "weight")

        test_obj = {"config": "value"}

        # Simulate both PP ranks owning the object
        mock_dist.all_gather_object.side_effect = lambda output, obj, group: output.__setitem__(
            slice(None), [True, True]
        )
        mock_dist.broadcast_object_list.side_effect = lambda obj_list, src, group: None

        # Must not raise — should pick rank 0 as source
        result = mapping.broadcast_obj_from_pp_rank(test_obj)
        assert result == test_obj

        # Verify broadcast_object_list was called with src=rank 0
        mock_dist.broadcast_object_list.assert_called_once()
        call_args = mock_dist.broadcast_object_list.call_args
        assert call_args[1].get("src", call_args[0][1] if len(call_args[0]) > 1 else None) == 0

    def test_tp_aware_unknown_module_error(self, transformer_config):
        """Test AutoMapping error for unknown module types."""
        mapping = AutoMapping("weight", "hf.weight")

        # Create an unknown module type
        unknown_module = torch.nn.Linear(10, 10)

        with pytest.raises(ValueError, match="Cannot determine parallelism type"):
            mapping._detect_parallelism_type(unknown_module)

    def test_resolve_wildcard_patterns(self):
        """Test wildcard pattern resolution."""
        # Test DirectMapping
        mapping = DirectMapping("layer.*.weight", "model.*.weight")
        resolved = mapping.resolve(("0",))
        assert resolved.megatron_param == "layer.0.weight"
        assert resolved.hf_param == "model.0.weight"

        # Test QKVMapping
        qkv_mapping = QKVMapping(
            "*.qkv.weight",
            q="*.q_proj.weight",
            k="*.k_proj.weight",
            v="*.v_proj.weight",
        )
        resolved_qkv = qkv_mapping.resolve(("layer0",))
        assert resolved_qkv.megatron_param == "layer0.qkv.weight"
        assert resolved_qkv.hf_param["q"] == "layer0.q_proj.weight"
        assert resolved_qkv.hf_param["k"] == "layer0.k_proj.weight"
        assert resolved_qkv.hf_param["v"] == "layer0.v_proj.weight"

        # Test GatedMLPMapping
        gated_mapping = GatedMLPMapping("*.mlp.weight", gate="*.gate_proj.weight", up="*.up_proj.weight")
        resolved_gated = gated_mapping.resolve(("layer1",))
        assert resolved_gated.megatron_param == "layer1.mlp.weight"
        assert resolved_gated.hf_param["gate"] == "layer1.gate_proj.weight"
        assert resolved_gated.hf_param["up"] == "layer1.up_proj.weight"

    def test_config_extraction_from_module(self, transformer_config):
        """Test config extraction from module hierarchy."""
        mapping = DirectMapping("weight", "weight")

        # Test direct config
        module_with_config = MockModule(transformer_config)
        assert mapping._get_config(module_with_config) == transformer_config

        # Test no config found
        module_without_config = torch.nn.Linear(10, 10)
        with pytest.raises(ValueError, match="Could not find config"):
            mapping._get_config(module_without_config)


class TestAutoMappingWithPermute:
    """Test AutoMapping functionality with permute_dims."""

    def test_basic_transpose_hf_to_megatron(self, mock_distributed_env, transformer_config):
        """Test basic transpose functionality from HF to Megatron."""
        mock_distributed_env()
        mapping = AutoMapping("transpose.weight", "hf.weight", permute_dims=(1, 0))

        # Create a test tensor [4, 8]
        hf_weight = torch.randn(4, 8)
        megatron_module = MockModule(transformer_config, weight_shape=(8, 4))

        with patch.object(mapping, "_mapping") as mock_delegate:
            mock_delegate.hf_to_megatron.return_value = torch.randn(8, 4)
            with patch.object(mapping, "_detect_parallelism_type", return_value="column"):
                mapping.hf_to_megatron(hf_weight, megatron_module)

            # Verify the tensor was transposed and made contiguous
            mock_delegate.hf_to_megatron.assert_called_once()
            transposed_tensor = mock_delegate.hf_to_megatron.call_args[0][0]
            expected_transposed = torch.permute(hf_weight, (1, 0)).contiguous()
            assert torch.equal(transposed_tensor, expected_transposed)
            assert transposed_tensor.shape == (8, 4)

    def test_basic_transpose_megatron_to_hf(self, mock_distributed_env, transformer_config):
        """Test basic transpose functionality from Megatron to HF."""
        mock_distributed_env()
        mapping = AutoMapping("transpose.weight", "hf.weight", permute_dims=(1, 0))

        # Create a test tensor [8, 4]
        megatron_weight = torch.randn(8, 4)
        megatron_module = MockModule(transformer_config, weight_shape=(8, 4))

        with patch.object(mapping, "_mapping") as mock_delegate:
            # Mock delegate to return the tensor under the megatron_param key
            mock_delegate.megatron_to_hf.return_value = {"transpose.weight": megatron_weight}
            with patch.object(mapping, "_detect_parallelism_type", return_value="column"):
                result = mapping.megatron_to_hf(megatron_weight, megatron_module)

            # Verify the result is transposed back and made contiguous
            assert "hf.weight" in result
            expected_transposed = torch.permute(megatron_weight, (1, 0)).contiguous()
            assert torch.equal(result["hf.weight"], expected_transposed)
            assert result["hf.weight"].shape == (4, 8)

    def test_transpose_with_different_dims(self, mock_distributed_env, transformer_config):
        """Test transpose with different dimension permutations."""
        mock_distributed_env()

        # Test 3D tensor transpose
        mapping = AutoMapping("transpose.weight", "hf.weight", permute_dims=(2, 0, 1))
        hf_weight = torch.randn(2, 3, 4)  # [2, 3, 4]
        megatron_module = MockModule(transformer_config, weight_shape=(4, 2, 3))

        with patch.object(mapping, "_mapping") as mock_delegate:
            mock_delegate.hf_to_megatron.return_value = torch.randn(4, 2, 3)
            with patch.object(mapping, "_detect_parallelism_type", return_value="column"):
                mapping.hf_to_megatron(hf_weight, megatron_module)

            # Verify the tensor was permuted correctly
            transposed_tensor = mock_delegate.hf_to_megatron.call_args[0][0]
            expected_shape = (4, 2, 3)  # dims=(2, 0, 1) applied to (2, 3, 4)
            assert transposed_tensor.shape == expected_shape

    def test_transpose_tp_distribution(self, mock_distributed_env, transformer_config):
        """Test AutoMapping with permute_dims and tensor parallelism."""
        mock_mpu, mock_dist = mock_distributed_env(tp_size=2, tp_rank=0)
        mapping = AutoMapping("transpose.weight", "hf.weight", permute_dims=(1, 0))

        hf_weight = torch.randn(4, 8)
        megatron_module = MockModule(transformer_config, weight_shape=(4, 4))  # Each rank gets half

        # Mock the AutoMapping's TP behavior
        with patch.object(mapping, "_mapping") as mock_delegate:
            mock_delegate.hf_to_megatron.return_value = torch.randn(4, 4)
            with patch.object(mapping, "_detect_parallelism_type", return_value="column"):
                mapping.hf_to_megatron(hf_weight, megatron_module)

            # Verify transpose happened before TP distribution
            mock_delegate.hf_to_megatron.assert_called_once()
            transposed_tensor = mock_delegate.hf_to_megatron.call_args[0][0]
            assert transposed_tensor.shape == (8, 4)  # Transposed from (4, 8)

    def test_transpose_tp_gathering(self, mock_distributed_env, transformer_config):
        """Test AutoMapping with permute_dims gathering from multiple TP ranks."""
        mock_distributed_env(tp_size=2, tp_rank=0)
        mapping = AutoMapping("transpose.weight", "hf.weight", permute_dims=(1, 0))

        megatron_weight = torch.randn(4, 4)  # Shard from current rank
        megatron_module = MockModule(transformer_config, weight_shape=(4, 4))

        # Mock AutoMapping to return gathered tensor
        full_gathered = torch.randn(8, 4)  # Full tensor after TP gathering
        with patch.object(mapping, "_mapping") as mock_delegate:
            mock_delegate.megatron_to_hf.return_value = {"transpose.weight": full_gathered}
            with patch.object(mapping, "_detect_parallelism_type", return_value="column"):
                result = mapping.megatron_to_hf(megatron_weight, megatron_module)

            # Verify the gathered tensor was transposed back
            assert "hf.weight" in result
            expected_shape = (4, 8)  # Transposed from (8, 4)
            assert result["hf.weight"].shape == expected_shape

    def test_transpose_empty_result_handling(self, mock_distributed_env, transformer_config):
        """Test AutoMapping with permute_dims handles empty results from delegate."""
        mock_distributed_env()
        mapping = AutoMapping("transpose.weight", "hf.weight", permute_dims=(1, 0))

        megatron_weight = torch.randn(8, 4)
        megatron_module = MockModule(transformer_config, weight_shape=(8, 4))

        # Mock delegate to return empty dict (e.g., from non-rank-0 in TP)
        with patch.object(mapping, "_mapping") as mock_delegate:
            mock_delegate.megatron_to_hf.return_value = {}
            with patch.object(mapping, "_detect_parallelism_type", return_value="column"):
                result = mapping.megatron_to_hf(megatron_weight, megatron_module)

            # Should return empty dict without error
            assert result == {}

    def test_transpose_wildcard_resolution(self, mock_distributed_env):
        """Test AutoMapping with permute_dims wildcard pattern resolution."""
        mock_distributed_env()
        mapping = AutoMapping("layer.*.transpose.weight", "model.*.hf.weight", permute_dims=(1, 0))

        # Test resolve method
        resolved = mapping.resolve(("0",))
        assert resolved.megatron_param == "layer.0.transpose.weight"
        assert resolved.hf_param == "model.0.hf.weight"
        assert resolved.permute_dims == (1, 0)  # permute_dims should be preserved
        assert isinstance(resolved, AutoMapping)

    def test_transpose_non_rank_zero_hf_to_megatron(self, mock_distributed_env, transformer_config):
        """Test AutoMapping with permute_dims on non-rank-0 during HF to Megatron conversion."""
        mock_distributed_env(tp_size=2, tp_rank=1)  # Non-rank-0
        mapping = AutoMapping("transpose.weight", "hf.weight", permute_dims=(1, 0))

        hf_weight = torch.randn(4, 8)
        megatron_module = MockModule(transformer_config, weight_shape=(4, 4))

        with patch.object(mapping, "_mapping") as mock_delegate:
            mock_delegate.hf_to_megatron.return_value = torch.randn(4, 4)
            with patch.object(mapping, "_detect_parallelism_type", return_value="column"):
                mapping.hf_to_megatron(hf_weight, megatron_module)

            # On non-rank-0, permutation is skipped, original tensor passed to delegate
            mock_delegate.hf_to_megatron.assert_called_once()
            passed_tensor = mock_delegate.hf_to_megatron.call_args[0][0]
            assert torch.equal(passed_tensor, hf_weight)

    def test_transpose_identity_permutation(self, mock_distributed_env, transformer_config):
        """Test AutoMapping with identity permutation."""
        mock_distributed_env()
        mapping = AutoMapping("transpose.weight", "hf.weight", permute_dims=(0, 1))  # Identity

        hf_weight = torch.randn(4, 8)
        megatron_module = MockModule(transformer_config, weight_shape=(4, 8))

        with patch.object(mapping, "_mapping") as mock_delegate:
            mock_delegate.hf_to_megatron.return_value = hf_weight.clone()
            with patch.object(mapping, "_detect_parallelism_type", return_value="column"):
                mapping.hf_to_megatron(hf_weight, megatron_module)

            # Even with identity permutation, tensor should be passed through (permuted then contiguous)
            transposed_tensor = mock_delegate.hf_to_megatron.call_args[0][0]
            assert torch.equal(transposed_tensor, hf_weight.contiguous())  # Should be unchanged except contiguous

    def test_transpose_higher_dimensional_tensors(self, mock_distributed_env, transformer_config):
        """Test AutoMapping with higher dimensional tensors."""
        mock_distributed_env()
        mapping = AutoMapping("transpose.weight", "hf.weight", permute_dims=(3, 1, 0, 2))

        # 4D tensor
        hf_weight = torch.randn(2, 3, 4, 5)  # [2, 3, 4, 5]
        expected_shape = (5, 3, 2, 4)  # After permutation (3, 1, 0, 2)
        megatron_module = MockModule(transformer_config, weight_shape=expected_shape)

        with patch.object(mapping, "_mapping") as mock_delegate:
            mock_delegate.hf_to_megatron.return_value = torch.randn(*expected_shape)
            with patch.object(mapping, "_detect_parallelism_type", return_value="column"):
                mapping.hf_to_megatron(hf_weight, megatron_module)

            transposed_tensor = mock_delegate.hf_to_megatron.call_args[0][0]
            assert transposed_tensor.shape == expected_shape

    def test_permute_without_permute_dims(self, mock_distributed_env, transformer_config):
        """Test that AutoMapping works normally without permute_dims."""
        mock_distributed_env()
        mapping = AutoMapping("weight", "hf.weight")  # No permute_dims

        hf_weight = torch.randn(4, 8)
        megatron_module = MockModule(transformer_config, weight_shape=(4, 8))

        with patch.object(mapping, "_mapping") as mock_delegate:
            mock_delegate.hf_to_megatron.return_value = hf_weight.clone()
            with patch.object(mapping, "_detect_parallelism_type", return_value="column"):
                mapping.hf_to_megatron(hf_weight, megatron_module)

            # Without permute_dims, tensor should be passed unchanged
            passed_tensor = mock_delegate.hf_to_megatron.call_args[0][0]
            assert torch.equal(passed_tensor, hf_weight)


class TestFusedGatedExpertMapping:
    """Tests for FusedGatedExpertMapping.hf_to_megatron with TP > 1.

    Regression: with TP=2 the target_param shape is already TP-sharded
    (e.g. (512, 2048) for moe_ffn=512, hidden=2048).  The old code compared
    the full HF gate_up_proj weight (1024, 2048) against that sharded shape
    and raised ValueError.  The fix computes the full (unsharded) shape before
    calling _align_expert_weight_to_shape so that _gated_mapping handles the
    TP scatter.
    """

    # Model dims matching Qwen3.5-35B-A3B: hidden=2048, moe_ffn=512
    HIDDEN = 2048
    MOE_FFN = 512
    NUM_EXPERTS = 4

    def _make_mapping(self):
        return FusedGatedExpertMapping(
            megatron_param="decoder.layers.0.mlp.experts.linear_fc1.weight3",
            hf_param="model.layers.0.mlp.experts.gate_up_proj",
        )

    def _make_hf_weights(self):
        """Fused gate_up_proj: [num_experts, 2*moe_ffn, hidden]."""
        return torch.randn(self.NUM_EXPERTS, 2 * self.MOE_FFN, self.HIDDEN)

    @pytest.mark.parametrize("tp_size", [1, 2, 4])
    def test_hf_to_megatron_tp_gt_1(self, mock_distributed_env, transformer_config, tp_size):
        """gate and up passed to _gated_mapping must be full (unsharded) tensors."""
        mock_mpu, _ = mock_distributed_env(tp_size=tp_size, tp_rank=0)

        # Expert weights use the expert TP group (etp_group) for tp_size/tp_rank.
        # Set etp_group to the same size as tp_group to match the test scenario.
        class _MockGroup:
            def __init__(self, size, rank):
                self._size, self._rank = size, rank

            def size(self):
                return self._size

            def rank(self):
                return self._rank

        mock_mpu.get_expert_tensor_parallel_group.return_value = _MockGroup(tp_size, 0)
        mapping = self._make_mapping()
        hf_weights = self._make_hf_weights()

        # Megatron linear_fc1.weight shape on one TP rank: (2*moe_ffn/tp, hidden)
        tp_sharded_shape = (2 * self.MOE_FFN // tp_size, self.HIDDEN)
        target_param = torch.nn.Parameter(torch.empty(tp_sharded_shape))

        with (
            patch(
                "megatron.bridge.models.conversion.param_mapping.get_module_and_param_from_name",
                return_value=(None, target_param),
            ),
            patch.object(mapping, "_gated_mapping") as mock_gated,
        ):
            mock_gated.hf_to_megatron.return_value = torch.zeros(tp_sharded_shape)
            mapping.hf_to_megatron(hf_weights, megatron_module=None)

        call_kwargs = mock_gated.hf_to_megatron.call_args
        gate = call_kwargs[0][0]["gate"]
        up = call_kwargs[0][0]["up"]

        # gate and up must be the full (unsharded) intermediate size
        expected_gate_shape = (self.MOE_FFN, self.HIDDEN)
        assert gate.shape == expected_gate_shape, (
            f"gate.shape={gate.shape} but expected {expected_gate_shape} "
            f"(TP={tp_size}; _gated_mapping is responsible for TP scatter)"
        )
        assert up.shape == expected_gate_shape, f"up.shape={up.shape} but expected {expected_gate_shape}"


class TestFusedExpertMapping:
    """Tests for FusedExpertMapping.hf_to_megatron with TP > 1.

    Regression: with TP=2 the target_param shape (linear_fc2.weight) is already
    TP-sharded (e.g. (2048, 256) for hidden=2048, moe_ffn=512).  The old code
    compared the full HF down_proj weight (2048, 512) against that sharded shape
    and raised ValueError.  The fix removes the _align call and passes the full
    expert weight directly to AutoMapping which handles TP scatter.
    """

    HIDDEN = 2048
    MOE_FFN = 512
    NUM_EXPERTS = 4

    def _make_mapping(self):
        return FusedExpertMapping(
            megatron_param="decoder.layers.0.mlp.experts.linear_fc2.weight3",
            hf_param="model.layers.0.mlp.experts.down_proj",
        )

    def _make_hf_weights(self):
        """Fused down_proj: [num_experts, hidden, moe_ffn]."""
        return torch.randn(self.NUM_EXPERTS, self.HIDDEN, self.MOE_FFN)

    @pytest.mark.parametrize("tp_size", [1, 2, 4])
    def test_hf_to_megatron_tp_gt_1(self, mock_distributed_env, transformer_config, tp_size):
        """Weight passed to AutoMapping.hf_to_megatron must be the full (unsharded) expert tensor."""
        mock_mpu, _ = mock_distributed_env(tp_size=tp_size, tp_rank=0)
        mapping = self._make_mapping()
        hf_weights = self._make_hf_weights()

        captured = {}

        def _capture_super(hf_w, megatron_module):
            captured["expert_weight"] = hf_w
            return torch.zeros(self.HIDDEN, self.MOE_FFN // tp_size)

        with patch.object(AutoMapping, "hf_to_megatron", side_effect=_capture_super):
            mapping.hf_to_megatron(hf_weights, megatron_module=None)

        expert_weight = captured["expert_weight"]
        expected_shape = (self.HIDDEN, self.MOE_FFN)
        assert expert_weight.shape == expected_shape, (
            f"expert_weight.shape={expert_weight.shape} but expected {expected_shape} "
            f"(TP={tp_size}; AutoMapping is responsible for TP scatter)"
        )
