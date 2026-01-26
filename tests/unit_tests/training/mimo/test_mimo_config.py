import pytest

from megatron.bridge.models.mimo.mimo_config import MimoParallelismConfig, ModuleParallelismConfig


def test_module_parallelism_finalize_computes_dp():
    parallelism = ModuleParallelismConfig(tensor_parallel=2, pipeline_parallel=2)
    parallelism.finalize(world_size=16)
    assert parallelism.data_parallel == 4
    assert parallelism.total_model_parallel_size == 4
    assert parallelism.total_ranks == 16


def test_module_parallelism_finalize_invalid_world_size():
    parallelism = ModuleParallelismConfig(tensor_parallel=3, pipeline_parallel=2)
    with pytest.raises(ValueError, match="world_size .* not divisible"):
        parallelism.finalize(world_size=10)


def test_mimo_colocated_mismatched_total_ranks():
    module_parallelisms = {
        "encoder": ModuleParallelismConfig(tensor_parallel=1, data_parallel=4),
        "language_module": ModuleParallelismConfig(tensor_parallel=2, data_parallel=4),
    }
    mimo_parallelism_config = MimoParallelismConfig(
        llm_module_name="language_module",
        module_parallelisms=module_parallelisms,
        deployment_mode="colocated",
    )
    with pytest.raises(ValueError, match="same total_ranks"):
        mimo_parallelism_config.finalize(world_size=8)


def test_mimo_homogeneous_mismatched_parallelism():
    module_parallelisms = {
        "encoder": ModuleParallelismConfig(tensor_parallel=1, data_parallel=2),
        "language_module": ModuleParallelismConfig(tensor_parallel=2, data_parallel=2),
    }
    mimo_parallelism_config = MimoParallelismConfig(
        llm_module_name="language_module",
        module_parallelisms=module_parallelisms,
        deployment_mode="homogeneous",
    )
    with pytest.raises(ValueError, match="identical parallelism"):
        mimo_parallelism_config.finalize(world_size=4)


def test_mimo_heterogeneous_rank_offset_overlap():
    module_parallelisms = {
        "encoder": ModuleParallelismConfig(tensor_parallel=1, data_parallel=4, rank_offset=0),
        "language_module": ModuleParallelismConfig(tensor_parallel=1, data_parallel=4, rank_offset=2),
    }
    mimo_parallelism_config = MimoParallelismConfig(
        llm_module_name="language_module",
        module_parallelisms=module_parallelisms,
        deployment_mode="heterogeneous",
    )
    with pytest.raises(ValueError, match="overlap"):
        mimo_parallelism_config.finalize(world_size=None)


