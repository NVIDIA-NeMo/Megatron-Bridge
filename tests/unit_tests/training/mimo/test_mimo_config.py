# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
import pytest

from megatron.bridge.models.mimo.mimo_config import MimoParallelismConfig, ModuleParallelismConfig


def test_module_parallelism_finalize_computes_dp():
    parallelism = ModuleParallelismConfig(tensor_model_parallel_size=2, pipeline_model_parallel_size=2)
    parallelism.finalize(world_size=16)
    assert parallelism.data_parallel_size == 4
    assert parallelism.total_model_parallel_size == 4
    assert parallelism.total_ranks == 16


def test_module_parallelism_finalize_invalid_world_size():
    parallelism = ModuleParallelismConfig(tensor_model_parallel_size=3, pipeline_model_parallel_size=2)
    with pytest.raises(ValueError, match="world_size .* not divisible"):
        parallelism.finalize(world_size=10)


def test_mimo_heterogeneous_rank_offset_overlap():
    """Test that overlapping rank ranges are detected in heterogeneous deployment."""
    module_parallelisms = {
        "encoder": ModuleParallelismConfig(tensor_model_parallel_size=1, data_parallel_size=4, rank_offset=0),
        "llm": ModuleParallelismConfig(tensor_model_parallel_size=1, data_parallel_size=4, rank_offset=2),
    }
    mimo_parallelism_config = MimoParallelismConfig(
        module_parallelisms=module_parallelisms,
    )
    with pytest.raises(ValueError, match="overlap"):
        mimo_parallelism_config.finalize(world_size=None)


def test_mimo_heterogeneous_valid_contiguous():
    """Test that contiguous rank allocation works correctly."""
    module_parallelisms = {
        "encoder": ModuleParallelismConfig(tensor_model_parallel_size=1, data_parallel_size=2, rank_offset=0),
        "llm": ModuleParallelismConfig(tensor_model_parallel_size=1, data_parallel_size=4, rank_offset=2),
    }
    mimo_parallelism_config = MimoParallelismConfig(
        module_parallelisms=module_parallelisms,
    )
    # No gaps, no overlap - should pass
    mimo_parallelism_config.finalize(world_size=None)
    assert mimo_parallelism_config.total_world_size == 6
