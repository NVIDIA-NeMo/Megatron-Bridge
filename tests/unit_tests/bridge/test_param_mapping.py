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

from megatron.bridge.bridge.param_mapping import (
    ColumnParallelMapping,
    DirectMapping,
    GatedMLPMapping,
    QKVMapping,
    ReplicatedMapping,
    RowParallelMapping,
    TPAwareMapping,
    merge_gated_mlp_weights,
    merge_qkv_biases,
    merge_qkv_weights,
    split_gated_mlp_weights,
    split_qkv_biases,
    split_qkv_weights,
)


@pytest.fixture
def mock_distributed_env():
    """Mocks the distributed environment for single-process testing."""
    with (
        patch("megatron.bridge.bridge.param_mapping.mpu") as mock_mpu,
        patch("torch.distributed") as mock_dist,
        patch("torch.cuda.current_device", return_value=0),
    ):

        def setup_mocks(tp_size=1, tp_rank=0, pp_size=1, pp_rank=0):
            mock_mpu.get_tensor_model_parallel_world_size.return_value = tp_size
            mock_mpu.get_tensor_model_parallel_rank.return_value = tp_rank
            mock_mpu.get_pipeline_model_parallel_world_size.return_value = pp_size
            mock_mpu.get_pipeline_model_parallel_rank.return_value = pp_rank
            mock_mpu.get_tensor_model_parallel_group.return_value = "tp_group"
            mock_mpu.get_pipeline_model_parallel_group.return_value = "pp_group"
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

        if tp_rank == 0:
            assert "hf.weight" in result
            assert torch.equal(result["hf.weight"], megatron_weight)
        else:
            assert not result

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
            mock_broadcast_method.assert_called_once_with(hf_weight, src_rank=0)
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

            if tp_rank == 0:
                assert "hf.weight" in result
                assert torch.equal(result["hf.weight"], full_weight)
            else:
                assert not result


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

            if tp_rank == 0:
                assert "hf.weight" in result
                assert torch.equal(result["hf.weight"], full_weight)
            else:
                assert not result


class TestTPAwareMapping:
    def test_detect_parallelism_type(self, mock_distributed_env, transformer_config):
        mock_distributed_env()
        mapping = TPAwareMapping(megatron_param="some.weight", hf_param="hf.weight")

        # Mock modules with different characteristics
        class MyCol(torch.nn.Module):
            tensor_model_parallel = True
            partition_dim = 0

        class MyRow(torch.nn.Module):
            tensor_model_parallel = True
            partition_dim = 1

        class MyRep(torch.nn.Module):
            tensor_model_parallel = False

        TPAwareMapping.register_module_type("MyCustomRow", "row")

        class MyCustomRow(torch.nn.Module):
            pass

        assert mapping._detect_parallelism_type(MyCol()) == "column"
        assert mapping._detect_parallelism_type(MyRow()) == "row"
        assert mapping._detect_parallelism_type(MyRep()) == "replicated"
        assert mapping._detect_parallelism_type(torch.nn.LayerNorm(5)) == "replicated"
        assert mapping._detect_parallelism_type(MyCustomRow()) == "row"

        with pytest.raises(ValueError):
            mapping._detect_parallelism_type(torch.nn.Linear(5, 5))


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

    def test_gated_mlp_merge_split(self, transformer_config):
        gate = torch.randn(64, 32)
        up = torch.randn(64, 32)

        merged = merge_gated_mlp_weights(transformer_config, gate, up)
        assert merged.shape == (128, 32)

        gate_s, up_s = split_gated_mlp_weights(transformer_config, merged)
        assert torch.equal(gate, gate_s)
        assert torch.equal(up, up_s)


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


class TestGatedMLPMapping:
    def test_hf_to_megatron(self, mock_distributed_env, transformer_config):
        mock_distributed_env()
        mapping = GatedMLPMapping(megatron_param="gated.weight", gate="gate.weight", up="up.weight")
        weights = {
            "gate": torch.randn(128, 32),
            "up": torch.randn(128, 32),
        }
        megatron_module = MockModule(transformer_config, weight_shape=(256, 32))

        with patch.object(mapping._tp_mapping, "hf_to_megatron") as mock_hf_to_megatron:
            mapping.hf_to_megatron(weights, megatron_module)
            mock_hf_to_megatron.assert_called_once()
            merged_weight = mock_hf_to_megatron.call_args[0][0]
            assert merged_weight.shape == (256, 32)

    def test_megatron_to_hf(self, mock_distributed_env, transformer_config):
        mock_distributed_env()
        mapping = GatedMLPMapping(megatron_param="gated.weight", gate="gate.weight", up="up.weight")
        merged_weight = torch.randn(256, 32)
        megatron_module = MockModule(transformer_config, weight_shape=(256, 32))

        with patch.object(mapping._tp_mapping, "megatron_to_hf") as mock_megatron_to_hf:
            mock_megatron_to_hf.return_value = {"gated.weight": merged_weight}
            result = mapping.megatron_to_hf(merged_weight, megatron_module)

            assert "gate.weight" in result
            assert "up.weight" in result
            assert result["gate.weight"].shape == (128, 32)
            assert result["up.weight"].shape == (128, 32)


class TestMappingEdgeCases:
    """Test edge cases and error handling in param mappings."""

    def test_wildcard_pattern_validation(self):
        """Test that wildcard patterns are validated correctly."""
        # Valid patterns - should not raise
        DirectMapping("layer.*.weight", "model.*.weight")
        QKVMapping(megatron_param="*.qkv.weight", q="*.q_proj.weight", k="*.k_proj.weight", v="*.v_proj.weight")

        # Invalid patterns - mismatched wildcard counts
        with pytest.raises(ValueError, match="Wildcard count mismatch"):
            DirectMapping("layer.*.*.weight", "model.*.weight")

        with pytest.raises(ValueError, match="Wildcard count mismatch"):
            QKVMapping("*.qkv.weight", q="*.*.q_proj.weight", k="*.k_proj.weight", v="*.v_proj.weight")

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

    def test_tp_aware_unknown_module_error(self, transformer_config):
        """Test TPAwareMapping error for unknown module types."""
        mapping = TPAwareMapping("weight", "hf.weight")

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
        qkv_mapping = QKVMapping("*.qkv.weight", q="*.q_proj.weight", k="*.k_proj.weight", v="*.v_proj.weight")
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
