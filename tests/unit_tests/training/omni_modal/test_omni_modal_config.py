# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
import pytest

from megatron.bridge.models.omni_modal.omni_modal_config import ModuleParallelismConfig, OmniModalParallelismConfig


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


def test_omni_modal_heterogeneous_rank_offset_overlap():
    """Test that overlapping rank ranges are detected in heterogeneous deployment."""
    module_parallelisms = {
        "encoder": ModuleParallelismConfig(tensor_model_parallel_size=1, data_parallel_size=4, rank_offset=0),
        "language": ModuleParallelismConfig(tensor_model_parallel_size=1, data_parallel_size=4, rank_offset=2),
    }
    omni_modal_parallelism_config = OmniModalParallelismConfig(
        module_parallelisms=module_parallelisms,
    )
    with pytest.raises(ValueError, match="overlap"):
        omni_modal_parallelism_config.finalize(world_size=6)


def test_omni_modal_heterogeneous_valid_contiguous():
    """Test that contiguous rank allocation works correctly."""
    # Note: encoder DP must be >= LLM DP for embedding alignment
    module_parallelisms = {
        "encoder": ModuleParallelismConfig(tensor_model_parallel_size=1, data_parallel_size=4, rank_offset=0),
        "language": ModuleParallelismConfig(tensor_model_parallel_size=1, data_parallel_size=2, rank_offset=4),
    }
    omni_modal_parallelism_config = OmniModalParallelismConfig(
        module_parallelisms=module_parallelisms,
    )
    # No gaps, no overlap, encoder DP >= LLM DP - should pass
    omni_modal_parallelism_config.finalize(world_size=6)
    assert omni_modal_parallelism_config.total_world_size == 6
