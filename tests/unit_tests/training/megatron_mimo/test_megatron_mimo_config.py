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
    assert parallelism.dense_model_parallel_size == 4
    assert parallelism.total_model_parallel_size == 4
    assert parallelism.total_ranks == 16
    assert parallelism.expert_data_parallel_size == 8


def test_module_parallelism_rank_algebra_excludes_expert_parallelism():
    parallelism = ModuleParallelismConfig(
        tensor_model_parallel_size=2,
        context_parallel_size=2,
        pipeline_model_parallel_size=2,
        data_parallel_size=4,
        expert_model_parallel_size=2,
        expert_tensor_parallel_size=4,
    )
    parallelism.finalize(world_size=None)

    assert parallelism.dense_model_parallel_size == 8
    assert parallelism.total_model_parallel_size == 8
    assert parallelism.total_ranks == 32
    assert parallelism.expert_data_parallel_size == 2


def test_module_parallelism_rejects_non_integer_expert_factorization():
    parallelism = ModuleParallelismConfig(
        tensor_model_parallel_size=2,
        data_parallel_size=1,
        expert_tensor_parallel_size=3,
    )

    with pytest.raises(ValueError, match="TP \\* CP \\* DP must be divisible"):
        parallelism.finalize(world_size=None)


@pytest.mark.parametrize(
    "field_name",
    [
        "tensor_model_parallel_size",
        "pipeline_model_parallel_size",
        "context_parallel_size",
        "expert_model_parallel_size",
        "expert_tensor_parallel_size",
        "data_parallel_size",
    ],
)
def test_module_parallelism_rejects_non_positive_sizes(field_name):
    kwargs = {"data_parallel_size": 1, field_name: 0}
    parallelism = ModuleParallelismConfig(**kwargs)

    with pytest.raises(ValueError, match=f"{field_name} must be positive"):
        parallelism.finalize(world_size=None)


def test_module_parallelism_rejects_negative_rank_offset():
    parallelism = ModuleParallelismConfig(data_parallel_size=1, rank_offset=-1)

    with pytest.raises(ValueError, match="rank_offset must be non-negative"):
        parallelism.finalize(world_size=None)


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


def test_megatron_mimo_heterogeneous_rank_offset_gap():
    """Test that disjoint rank ranges must still tile the world with no gaps."""
    module_parallelisms = {
        "encoder": ModuleParallelismConfig(tensor_model_parallel_size=1, data_parallel_size=1, rank_offset=0),
        "language": ModuleParallelismConfig(tensor_model_parallel_size=1, data_parallel_size=1, rank_offset=2),
    }
    megatron_mimo_parallelism_config = MegatronMIMOParallelismConfig(
        module_parallelisms=module_parallelisms,
    )
    with pytest.raises(ValueError, match="no gaps"):
        megatron_mimo_parallelism_config.finalize(world_size=3)


def test_megatron_mimo_heterogeneous_valid_contiguous():
    """Test that contiguous rank allocation works correctly."""
    module_parallelisms = {
        "encoder": ModuleParallelismConfig(tensor_model_parallel_size=1, data_parallel_size=4, rank_offset=0),
        "language": ModuleParallelismConfig(tensor_model_parallel_size=1, data_parallel_size=2, rank_offset=4),
    }
    megatron_mimo_parallelism_config = MegatronMIMOParallelismConfig(
        module_parallelisms=module_parallelisms,
    )
    megatron_mimo_parallelism_config.finalize(world_size=6)
    assert megatron_mimo_parallelism_config.total_world_size == 6


def test_megatron_mimo_heterogeneous_allows_encoder_dp_less_than_language_dp():
    """Test BridgeCommunicator fan-out layouts where encoder DP is smaller."""
    module_parallelisms = {
        "images": ModuleParallelismConfig(tensor_model_parallel_size=1, data_parallel_size=1, rank_offset=0),
        "language": ModuleParallelismConfig(tensor_model_parallel_size=1, data_parallel_size=4, rank_offset=1),
    }
    megatron_mimo_parallelism_config = MegatronMIMOParallelismConfig(
        module_parallelisms=module_parallelisms,
    )

    megatron_mimo_parallelism_config.finalize(world_size=5)

    assert megatron_mimo_parallelism_config.total_world_size == 5


def test_megatron_mimo_rejects_encoder_expert_parallelism():
    module_parallelisms = {
        "images": ModuleParallelismConfig(
            tensor_model_parallel_size=1,
            data_parallel_size=2,
            rank_offset=0,
            expert_model_parallel_size=2,
        ),
        "language": ModuleParallelismConfig(tensor_model_parallel_size=1, data_parallel_size=1, rank_offset=2),
    }
    megatron_mimo_parallelism_config = MegatronMIMOParallelismConfig(
        module_parallelisms=module_parallelisms,
    )

    with pytest.raises(ValueError, match="Encoder modules must remain dense"):
        megatron_mimo_parallelism_config.finalize(world_size=3)


def test_megatron_mimo_allows_language_expert_parallelism_after_runtime_wiring():
    module_parallelisms = {
        "images": ModuleParallelismConfig(tensor_model_parallel_size=1, data_parallel_size=1, rank_offset=0),
        "language": ModuleParallelismConfig(
            tensor_model_parallel_size=2,
            data_parallel_size=1,
            rank_offset=1,
            expert_model_parallel_size=2,
        ),
    }
    megatron_mimo_parallelism_config = MegatronMIMOParallelismConfig(
        module_parallelisms=module_parallelisms,
    )

    megatron_mimo_parallelism_config.finalize(world_size=3)

    language_parallelism = megatron_mimo_parallelism_config.get_parallelism("language")
    assert language_parallelism.expert_model_parallel_size == 2
    assert language_parallelism.expert_tensor_parallel_size == 1


def test_megatron_mimo_heterogeneous_rejects_non_divisible_dp():
    """Test asymmetric DP still requires pairwise divisibility."""
    module_parallelisms = {
        "images": ModuleParallelismConfig(tensor_model_parallel_size=1, data_parallel_size=3, rank_offset=0),
        "language": ModuleParallelismConfig(tensor_model_parallel_size=1, data_parallel_size=2, rank_offset=3),
    }
    megatron_mimo_parallelism_config = MegatronMIMOParallelismConfig(
        module_parallelisms=module_parallelisms,
    )

    with pytest.raises(ValueError, match="DP sizes must be divisible"):
        megatron_mimo_parallelism_config.finalize(world_size=5)
