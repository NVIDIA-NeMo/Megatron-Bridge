from types import SimpleNamespace

import pytest

from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mimo_config import MIMOConfig, ModuleParallelismConfig


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
    mimo = MIMOConfig(
        llm_module_name="language_module",
        module_parallelisms=module_parallelisms,
        deployment_mode="colocated",
    )
    with pytest.raises(ValueError, match="same total_ranks"):
        mimo.finalize(world_size=8)


def test_mimo_homogeneous_mismatched_parallelism():
    module_parallelisms = {
        "encoder": ModuleParallelismConfig(tensor_parallel=1, data_parallel=2),
        "language_module": ModuleParallelismConfig(tensor_parallel=2, data_parallel=2),
    }
    mimo = MIMOConfig(
        llm_module_name="language_module",
        module_parallelisms=module_parallelisms,
        deployment_mode="homogeneous",
    )
    with pytest.raises(ValueError, match="identical parallelism"):
        mimo.finalize(world_size=4)


def test_mimo_heterogeneous_rank_offset_overlap():
    module_parallelisms = {
        "encoder": ModuleParallelismConfig(tensor_parallel=1, data_parallel=4, rank_offset=0),
        "language_module": ModuleParallelismConfig(tensor_parallel=1, data_parallel=4, rank_offset=2),
    }
    mimo = MIMOConfig(
        llm_module_name="language_module",
        module_parallelisms=module_parallelisms,
        deployment_mode="heterogeneous",
    )
    with pytest.raises(ValueError, match="overlap"):
        mimo.finalize(world_size=None)


def _make_cfg(mimo: MIMOConfig, encoder_providers=None) -> ConfigContainer:
    model = SimpleNamespace(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
    )
    train = SimpleNamespace(global_batch_size=8)
    placeholder = SimpleNamespace()
    return ConfigContainer(
        train=train,
        model=model,
        optimizer=placeholder,
        scheduler=placeholder,
        dataset=placeholder,
        logger=placeholder,
        tokenizer=placeholder,
        checkpoint=placeholder,
        mimo=mimo,
        encoder_providers=encoder_providers,
    )


def test_mimo_missing_encoder_providers(monkeypatch):
    module_parallelisms = {
        "encoder": ModuleParallelismConfig(tensor_parallel=1, data_parallel=8),
        "language_module": ModuleParallelismConfig(tensor_parallel=1, data_parallel=8),
    }
    mimo = MIMOConfig(
        llm_module_name="language_module",
        module_parallelisms=module_parallelisms,
        deployment_mode="colocated",
    )
    monkeypatch.setattr("megatron.bridge.training.config.get_world_size_safe", lambda: 1)
    cfg = _make_cfg(mimo=mimo, encoder_providers=None)
    with pytest.raises(ValueError, match="encoder_providers must be set"):
        cfg._validate_mimo()


def test_mimo_encoder_provider_unknown_key(monkeypatch):
    module_parallelisms = {
        "encoder": ModuleParallelismConfig(tensor_parallel=1, data_parallel=8),
        "language_module": ModuleParallelismConfig(tensor_parallel=1, data_parallel=8),
    }
    mimo = MIMOConfig(
        llm_module_name="language_module",
        module_parallelisms=module_parallelisms,
        deployment_mode="colocated",
    )
    monkeypatch.setattr("megatron.bridge.training.config.get_world_size_safe", lambda: 1)
    cfg = _make_cfg(mimo=mimo, encoder_providers={"other": object()})
    with pytest.raises(ValueError, match="unknown modules"):
        cfg._validate_mimo()
