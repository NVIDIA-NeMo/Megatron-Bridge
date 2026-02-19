# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import warnings

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


def test_module_parallelism_total_model_parallel_size_property():
    """Test total_model_parallel_size calculation."""
    parallelism = ModuleParallelismConfig(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=2,
        context_parallel_size=2,
        expert_tensor_parallel_size=2,
    )
    assert parallelism.total_model_parallel_size == 16  # 2 * 2 * 2 * 2


def test_module_parallelism_total_ranks_property():
    """Test total_ranks property."""
    parallelism = ModuleParallelismConfig(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=2,
        data_parallel_size=4,
    )
    assert parallelism.total_ranks == 16  # (2 * 2) * 4


def test_module_parallelism_total_ranks_raises_without_dp():
    """Test total_ranks raises error when data_parallel_size is None."""
    parallelism = ModuleParallelismConfig(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=2,
    )
    with pytest.raises(ValueError, match="data_parallel_size must be set"):
        _ = parallelism.total_ranks


def test_module_parallelism_expert_tensor_parallel_warning():
    """Test warning when using expert_tensor_parallel_size > 1 with pipeline > 1."""
    parallelism = ModuleParallelismConfig(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=2,
        expert_tensor_parallel_size=2,
        data_parallel_size=2,
    )

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        parallelism.finalize(world_size=None)

        # Check warning was raised
        assert len(w) == 1
        assert "expert_tensor_parallel_size > 1 with pipeline_model_parallel_size > 1" in str(w[0].message)


def test_module_parallelism_data_parallel_validation():
    """Test data_parallel_size validation."""
    parallelism = ModuleParallelismConfig(
        tensor_model_parallel_size=2,
        data_parallel_size=0,  # Invalid
    )
    with pytest.raises(ValueError, match="data_parallel_size must be positive"):
        parallelism.finalize(world_size=None)


def test_mimo_parallelism_total_world_size_property():
    """Test total_world_size calculation."""
    mimo_config = MimoParallelismConfig(
        module_parallelisms={
            "llm": ModuleParallelismConfig(
                tensor_model_parallel_size=2,
                data_parallel_size=2,
                rank_offset=0,
            ),
            "encoder": ModuleParallelismConfig(
                tensor_model_parallel_size=2,
                data_parallel_size=2,
                rank_offset=4,
            ),
        }
    )

    # Total world size should be 8 (ranks 0-7)
    assert mimo_config.total_world_size == 8


def test_mimo_parallelism_module_names_property():
    """Test module_names property."""
    mimo_config = MimoParallelismConfig(
        module_parallelisms={
            "llm": ModuleParallelismConfig(tensor_model_parallel_size=2),
            "clip_encoder": ModuleParallelismConfig(tensor_model_parallel_size=2),
            "dino_encoder": ModuleParallelismConfig(tensor_model_parallel_size=2),
        }
    )

    module_names = mimo_config.module_names
    assert "llm" in module_names
    assert "clip_encoder" in module_names
    assert "dino_encoder" in module_names
    assert len(module_names) == 3


def test_mimo_heterogeneous_edge_touching_ranges():
    """Test that edge-touching ranges (no overlap) are valid."""
    module_parallelisms = {
        "llm": ModuleParallelismConfig(
            tensor_model_parallel_size=2,
            data_parallel_size=2,
            rank_offset=0,  # ranks 0-3
        ),
        "encoder": ModuleParallelismConfig(
            tensor_model_parallel_size=2,
            data_parallel_size=2,
            rank_offset=4,  # ranks 4-7 (touching but not overlapping)
        ),
    }
    mimo_config = MimoParallelismConfig(module_parallelisms=module_parallelisms)

    # Should not raise an error
    mimo_config.finalize(world_size=None)


def test_mimo_heterogeneous_multiple_overlaps():
    """Test detection of multiple overlapping ranges."""
    module_parallelisms = {
        "llm": ModuleParallelismConfig(
            tensor_model_parallel_size=2,
            data_parallel_size=2,
            rank_offset=0,  # ranks 0-3
        ),
        "encoder1": ModuleParallelismConfig(
            tensor_model_parallel_size=2,
            data_parallel_size=2,
            rank_offset=2,  # ranks 2-5 (overlaps with llm)
        ),
        "encoder2": ModuleParallelismConfig(
            tensor_model_parallel_size=2,
            data_parallel_size=2,
            rank_offset=6,  # ranks 6-9
        ),
    }
    mimo_config = MimoParallelismConfig(module_parallelisms=module_parallelisms)

    # Should detect overlap
    with pytest.raises(ValueError, match="overlap"):
        mimo_config.finalize(world_size=None)


def test_mimo_finalize_missing_llm_module():
    """Test that finalize raises error when 'llm' module is missing."""
    mimo_config = MimoParallelismConfig(
        module_parallelisms={
            "encoder": ModuleParallelismConfig(tensor_model_parallel_size=2, data_parallel_size=2),
        }
    )

    with pytest.raises(ValueError, match="LLM module 'llm' must be in module_parallelisms"):
        mimo_config.finalize(world_size=None)


def test_mimo_finalize_world_size_mismatch():
    """Test that finalize detects world size mismatch."""
    mimo_config = MimoParallelismConfig(
        module_parallelisms={
            "llm": ModuleParallelismConfig(
                tensor_model_parallel_size=2,
                data_parallel_size=2,
                rank_offset=0,
            ),
        }
    )

    # Expected world size is 4, but providing 8
    with pytest.raises(ValueError, match="MIMO world size mismatch"):
        mimo_config.finalize(world_size=8)
