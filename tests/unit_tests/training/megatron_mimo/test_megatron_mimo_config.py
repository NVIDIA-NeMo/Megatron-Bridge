# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
import pytest

from megatron.bridge.models.megatron_mimo.megatron_mimo_config import (
    MegatronMIMOParallelismConfig,
    ModuleParallelismConfig,
)


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


def test_megatron_mimo_heterogeneous_rank_offset_overlap():
    """Test that overlapping rank ranges are detected in heterogeneous deployment."""
    module_parallelisms = {
        "encoder": ModuleParallelismConfig(tensor_model_parallel_size=1, data_parallel_size=4, rank_offset=0),
        "language": ModuleParallelismConfig(tensor_model_parallel_size=1, data_parallel_size=4, rank_offset=2),
    }
    megatron_mimo_parallelism_config = MegatronMIMOParallelismConfig(
        module_parallelisms=module_parallelisms,
    )
    with pytest.raises(ValueError, match="overlap"):
        megatron_mimo_parallelism_config.finalize(world_size=6)


def test_megatron_mimo_heterogeneous_valid_contiguous():
    """Test that contiguous rank allocation works correctly."""
    # Note: encoder DP must be >= LLM DP for embedding alignment
    module_parallelisms = {
        "encoder": ModuleParallelismConfig(tensor_model_parallel_size=1, data_parallel_size=4, rank_offset=0),
        "language": ModuleParallelismConfig(tensor_model_parallel_size=1, data_parallel_size=2, rank_offset=4),
    }
    megatron_mimo_parallelism_config = MegatronMIMOParallelismConfig(
        module_parallelisms=module_parallelisms,
    )
    # No gaps, no overlap, encoder DP >= LLM DP - should pass
    megatron_mimo_parallelism_config.finalize(world_size=6)
    assert megatron_mimo_parallelism_config.total_world_size == 6


def test_megatron_mimo_colocated_same_offset_same_size_accepted():
    """Colocated: modules share identical rank range (same offset AND same total_ranks)."""
    module_parallelisms = {
        "encoder": ModuleParallelismConfig(tensor_model_parallel_size=1, data_parallel_size=8, rank_offset=0),
        "language": ModuleParallelismConfig(tensor_model_parallel_size=1, data_parallel_size=8, rank_offset=0),
    }
    cfg = MegatronMIMOParallelismConfig(module_parallelisms=module_parallelisms)
    cfg.finalize(world_size=8)
    assert cfg.total_world_size == 8


def test_megatron_mimo_colocated_asymmetric_dp_rejected():
    """Colocated with asymmetric DP (encoder_dp != llm_dp) must fail fast.

    Short-term guard: the forward step currently slices the global micro-batch
    by a single DP geometry, so modality inputs and LLM keys cannot be routed
    independently. Full per-key per-module slicing is tracked as a follow-up;
    once it lands, this configuration should be accepted (fan-in case)."""
    module_parallelisms = {
        "encoder": ModuleParallelismConfig(tensor_model_parallel_size=1, data_parallel_size=8, rank_offset=0),
        "language": ModuleParallelismConfig(tensor_model_parallel_size=4, data_parallel_size=2, rank_offset=0),
    }
    cfg = MegatronMIMOParallelismConfig(module_parallelisms=module_parallelisms)
    with pytest.raises(ValueError, match="Colocated MegatronMIMO requires encoder DP == LLM DP"):
        cfg.finalize(world_size=8)


def test_megatron_mimo_colocated_asymmetric_tp_rejected():
    """Colocated with asymmetric TP (encoder_tp != llm_tp) must fail fast.

    Short-term guard: mcore's CUDA RNG tracker is seeded by a single tp_rank,
    so a rank whose encoder tp_rank and LLM tp_rank differ would run one
    module with the wrong TP-region dropout masks and weight-init RNG. Full
    per-module seeding (module-scoped tracker contexts) is an mcore-side
    change tracked as a follow-up; once it lands, this config should be OK."""
    # Equal DP with different TP requires PP variation to keep total_ranks equal.
    module_parallelisms = {
        "encoder": ModuleParallelismConfig(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=2,
            data_parallel_size=2,
            rank_offset=0,
        ),
        "language": ModuleParallelismConfig(
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=1,
            data_parallel_size=2,
            rank_offset=0,
        ),
    }
    cfg = MegatronMIMOParallelismConfig(module_parallelisms=module_parallelisms)
    with pytest.raises(ValueError, match="Colocated MegatronMIMO requires encoder TP == LLM TP"):
        cfg.finalize(world_size=4)


def test_megatron_mimo_rejects_partial_overlap_same_offset_different_size():
    """Same rank_offset but different total_ranks is neither colocated nor disjoint."""
    module_parallelisms = {
        "encoder": ModuleParallelismConfig(tensor_model_parallel_size=1, data_parallel_size=8, rank_offset=0),
        "language": ModuleParallelismConfig(tensor_model_parallel_size=1, data_parallel_size=4, rank_offset=0),
    }
    cfg = MegatronMIMOParallelismConfig(module_parallelisms=module_parallelisms)
    with pytest.raises(ValueError, match="partial overlap"):
        cfg.finalize(world_size=8)
