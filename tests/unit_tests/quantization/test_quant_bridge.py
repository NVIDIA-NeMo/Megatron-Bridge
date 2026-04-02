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
    GatedMLPMapping,
    QKVMapping,
    ReplicatedMapping,
    RowParallelMapping,
    merge_qkv_weights,
)

def scaled_fp8_blockwise(
    data_hp,
    weight_block_size,
):
    """
    This function is adopted from the VeRL project: https://github.com/verl-project/verl/blob/release/v0.7.0/verl/utils/vllm/vllm_fp8_utils.py#L109
    """
    # cast tensor from high precision to FP8 with 128*128 blockwise quantization.
    assert len(data_hp.shape) == 2, "Only 2d input tensor is supported"

    block_size1 = weight_block_size[1]
    block_size0 = weight_block_size[0]
    assert data_hp.shape[1] % block_size1 == 0, (
        f"data_hp.shape[1] {data_hp.shape[1]}  must be a multiple of block_size1: {block_size1}."
    )
    assert data_hp.shape[0] % block_size0 == 0, (
        f"data_hp.shape[0] {data_hp.shape[0]} must be a multiple of block_size0: {block_size0}."
    )

    # FP8
    max_dtype = torch.finfo(torch.float8_e4m3fn).max

    original_shape = data_hp.shape
    blk_m, blk_n = data_hp.shape[0] // block_size0, data_hp.shape[1] // block_size1

    assert block_size1 == block_size0
    data_hp = data_hp.reshape(blk_m, block_size0, blk_n, block_size1)

    # Permute to (BLK_M, BLK_N, BLOCK_SIZE_M, BLOCK_SIZE_N)
    data_hp = data_hp.permute(0, 2, 1, 3)
    # Flatten to (BLK_M, BLK_N, BLOCK_SIZE_M * BLOCK_SIZE_N)
    data_hp = data_hp.to(torch.float32).contiguous().flatten(start_dim=2)

    # Calculate max absolute value per block
    max_abs = torch.amax(torch.abs(data_hp), dim=-1, keepdim=True)

    # Use FP32 scale
    scale_fp = max_dtype / max_abs
    scale_fp = torch.where(max_abs == 0, 1.0, scale_fp)
    # preserve the behavior for 0 amax case
    scale_fp = torch.where(max_abs == torch.inf, 1.0, scale_fp)

    descale_fp = torch.reciprocal(scale_fp)

    # Scale and saturate cast the data elements to max of target dtype
    data_lp = torch.clamp(data_hp * scale_fp, min=-1 * max_dtype, max=max_dtype)

    fp_data = data_lp.to(torch.float8_e4m3fn)

    # (BLK_M, BLK_N, BLOCK_SIZE_M * BLOCK_SIZE_N) to (M, N)
    fp_data = fp_data.reshape(blk_m, blk_n, block_size0, block_size1).permute(0, 2, 1, 3).reshape(original_shape)

    # Convert to target format, but still in original precision container
    return fp_data, descale_fp


@pytest.fixture
def mock_distributed_env():
    """Mocks the distributed environment for single-process testing."""
    with (
        patch("megatron.bridge.models.conversion.param_mapping.mpu") as mock_mpu,
        patch("torch.distributed") as mock_dist,
        patch("torch.cuda.current_device", return_value=0),
    ):

        def setup_mocks(tp_size=1, tp_rank=0, pp_size=1, pp_rank=0, ep_size=1, ep_rank=0):
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
            ep_group = _MockGroup(ep_size, ep_rank)

            mock_mpu.get_tensor_model_parallel_world_size.return_value = tp_size
            mock_mpu.get_tensor_model_parallel_rank.return_value = tp_rank
            mock_mpu.get_pipeline_model_parallel_world_size.return_value = pp_size
            mock_mpu.get_pipeline_model_parallel_rank.return_value = pp_rank
            mock_mpu.get_expert_model_parallel_world_size.return_value = ep_size
            mock_mpu.get_expert_model_parallel_rank.return_value = ep_rank

            mock_mpu.get_tensor_model_parallel_group.return_value = tp_group
            mock_mpu.get_pipeline_model_parallel_group.return_value = pp_group
            mock_mpu.get_expert_model_parallel_group.return_value = ep_group

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


def dummy_quantization_checker(param_name):
    return True


class TestColumnParallelMappingQuant:
    @pytest.mark.parametrize("tp_rank", [0, 1])
    def test_megatron_to_hf_quant(self, mock_distributed_env, tp_rank):
        mock_distributed_env(tp_size=2, tp_rank=tp_rank)
        mapping = ColumnParallelMapping("col.weight", "hf.weight")
        megatron_shard = torch.randn(16, 16)

        with patch.object(mapping, "gather_from_tp_ranks") as mock_gather:
            full_weight = torch.randn(32, 16)
            q_full, scale_full = scaled_fp8_blockwise(full_weight)
            
            mock_gather.side_effect = [
                list(torch.chunk(q_full, 2, dim=0)),
                list(torch.chunk(scale_full, 2, dim=0))
            ]
            
            result = mapping.megatron_to_hf_quant(
                megatron_shard, 
                None, 
                quantization_checker=dummy_quantization_checker, 
                quant_fn=scaled_fp8_blockwise
            )

            assert "hf.weight" in result
            assert torch.equal(result["hf.weight"], q_full)
            assert torch.equal(result["hf.weight_scale_inv"], scale_full)


class TestRowParallelMappingQuant:
    @pytest.mark.parametrize("tp_rank", [0, 1])
    def test_megatron_to_hf_quant(self, mock_distributed_env, tp_rank):
        mock_distributed_env(tp_size=2, tp_rank=tp_rank)
        mapping = RowParallelMapping("row.weight", "hf.weight")
        megatron_shard = torch.randn(16, 16)

        with patch.object(mapping, "gather_from_tp_ranks") as mock_gather:
            full_weight = torch.randn(16, 32)
            q_full, scale_full = scaled_fp8_blockwise(full_weight, None)
            
            mock_gather.side_effect = [
                list(torch.chunk(q_full, 2, dim=1)),
                list(torch.chunk(scale_full, 2, dim=1))
            ]
            
            result = mapping.megatron_to_hf_quant(
                megatron_shard, 
                None, 
                quantization_checker=dummy_quantization_checker, 
                quant_fn=scaled_fp8_blockwise
            )

            assert "hf.weight" in result
            assert torch.equal(result["hf.weight"], q_full)
            assert torch.equal(result["hf.weight_scale_inv"], scale_full)


class TestReplicatedMappingQuant:
    @pytest.mark.parametrize("tp_rank", [0, 1])
    def test_megatron_to_hf_quant(self, mock_distributed_env, tp_rank):
        mock_distributed_env(tp_size=2, tp_rank=tp_rank)
        mapping = ReplicatedMapping("rep.weight", "hf.weight")
        megatron_weight = torch.randn(16, 16)
        
        # ReplicatedMapping doesn't quantize, it just calls megatron_to_hf
        result = mapping.megatron_to_hf_quant(
            megatron_weight, 
            None, 
            quantization_checker=dummy_quantization_checker, 
            quant_fn=scaled_fp8_blockwise
        )

        assert "hf.weight" in result
        assert torch.equal(result["hf.weight"], megatron_weight)


class TestAutoMappingQuant:
    def test_megatron_to_hf_quant(self, mock_distributed_env, transformer_config):
        mock_distributed_env()
        mapping = AutoMapping(megatron_param="some.weight", hf_param="hf.weight")
        
        class MyCol(torch.nn.Module):
            tensor_model_parallel = True
            partition_dim = 0
            
        megatron_module = MyCol()
        megatron_weight = torch.randn(16, 16)
        
        with patch.object(ColumnParallelMapping, "megatron_to_hf_quant") as mock_quant:
            q_full, _ = scaled_fp8_blockwise(megatron_weight, None)
            mock_quant.return_value = {"hf.weight": q_full}
            
            result = mapping.megatron_to_hf_quant(
                megatron_weight, 
                megatron_module, 
                quantization_checker=dummy_quantization_checker, 
                quant_fn=scaled_fp8_blockwise
            )
            
            assert "hf.weight" in result
            assert torch.equal(result["hf.weight"], q_full)


class TestQKVMappingQuant:
    def test_megatron_to_hf_quant(self, mock_distributed_env, transformer_config):
        mock_distributed_env()
        mapping = QKVMapping(megatron_param="qkv.weight", q="q.weight", k="k.weight", v="v.weight")

        q = torch.randn(32, 32)
        k = torch.randn(16, 32)
        v = torch.randn(16, 32)
        packed_qkv = merge_qkv_weights(transformer_config, q, k, v)
        megatron_module = MockModule(transformer_config, weight_shape=(64, 32))
        megatron_module.tensor_model_parallel = True
        megatron_module.partition_dim = 0

        quant_block_size = (16, 16)
        exp_q, exp_q_scale = scaled_fp8_blockwise(q, quant_block_size)
        exp_k, exp_k_scale = scaled_fp8_blockwise(k, quant_block_size)
        exp_v, exp_v_scale = scaled_fp8_blockwise(v, quant_block_size)

        result = mapping.megatron_to_hf_quant(
            packed_qkv,
            megatron_module,
            quantization_checker=dummy_quantization_checker,
            quant_fn=scaled_fp8_blockwise,
            quant_block_size=quant_block_size,
        )

        assert torch.equal(result["q.weight"], exp_q)
        assert torch.equal(result["k.weight"], exp_k)
        assert torch.equal(result["v.weight"], exp_v)
        assert torch.equal(result["q.weight_scale_inv"], exp_q_scale)
        assert torch.equal(result["k.weight_scale_inv"], exp_k_scale)
        assert torch.equal(result["v.weight_scale_inv"], exp_v_scale)


class TestGatedMLPMappingQuant:
    def test_megatron_to_hf_quant(self, mock_distributed_env, transformer_config):
        """Test splitting concatenated weights back to gate+up with single TP and quantization."""
        mock_distributed_env()
        mapping = GatedMLPMapping(megatron_param="gated.weight", gate="gate.weight", up="up.weight")

        # Create a concatenated tensor [gate; up]
        gate = torch.randn(128, 32)
        up = torch.randn(128, 32)
        merged_weight = torch.cat([gate, up], dim=0)
        megatron_module = MockModule(transformer_config, weight_shape=(256, 32))

        result = mapping.megatron_to_hf_quant(
            merged_weight, 
            megatron_module, 
            quantization_checker=dummy_quantization_checker, 
            quant_fn=scaled_fp8_blockwise,
            quant_block_size=(16, 16)
        )

        assert "gate.weight" in result
        assert "up.weight" in result
        assert result["gate.weight"].shape == (128, 32)
        assert result["up.weight"].shape == (128, 32)

        # Verify the split is correct
        q_gate, scale_gate = scaled_fp8_blockwise(gate, (16, 16))
        q_up, scale_up = scaled_fp8_blockwise(up, (16, 16))

        assert torch.equal(result["gate.weight"], q_gate)
        assert torch.equal(result["up.weight"], q_up)
        assert torch.equal(result["gate.weight_scale_inv"], scale_gate)
        assert torch.equal(result["up.weight_scale_inv"], scale_up)
