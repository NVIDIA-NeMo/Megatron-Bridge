from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from megatron.bridge.models.common import unimodal
from megatron.bridge.models.deepseek import deepseek_v4_fsdp
from megatron.bridge.models.deepseek.deepseek_v4_hybrid_provider import DeepSeekV4HybridModelProvider


def provider_config(mhc):
    config = object.__new__(DeepSeekV4HybridModelProvider)
    config.enable_mhc_connections = mhc
    return config


@pytest.mark.unit
@pytest.mark.parametrize(
    "use_fsdp,version,mhc,deepseek",
    [(False, 2, True, True), (True, 1, True, True), (True, 2, False, True), (True, 2, True, False)],
)
def test_other_paths_delegate_unchanged(monkeypatch, use_fsdp, version, mhc, deepseek):
    core_wrap = Mock(return_value=["wrapped"])
    custom_wrap = Mock()
    monkeypatch.setattr(unimodal, "_core_ddp_wrap", core_wrap)
    monkeypatch.setattr(deepseek_v4_fsdp, "wrap_deepseek_v4_fsdp", custom_wrap)
    config = provider_config(mhc) if deepseek else SimpleNamespace(enable_mhc_connections=mhc)
    monkeypatch.setattr(unimodal, "get_model_config", lambda chunk: config)
    model, ddp_config, groups = [Mock()], SimpleNamespace(megatron_fsdp_version=version), Mock()
    assert unimodal._ddp_wrap(
        model,
        True,
        ddp_config,
        True,
        use_fsdp,
        pg_collection=groups,
        use_layer_wise_distributed_optimizer=True,
        use_layer_wise_param_layout=False,
    ) == ["wrapped"]
    core_wrap.assert_called_once_with(
        model,
        True,
        ddp_config,
        True,
        use_megatron_fsdp=use_fsdp,
        use_torch_fsdp2=False,
        pg_collection=groups,
        use_layer_wise_distributed_optimizer=True,
        use_layer_wise_param_layout=False,
    )
    custom_wrap.assert_not_called()


@pytest.mark.unit
def test_deepseek_v4_dispatch_forwards_conflicts_for_validation(monkeypatch):
    core_wrap, custom_wrap = Mock(), Mock(return_value=["custom"])
    monkeypatch.setattr(unimodal, "_core_ddp_wrap", core_wrap)
    monkeypatch.setattr(deepseek_v4_fsdp, "wrap_deepseek_v4_fsdp", custom_wrap)
    monkeypatch.setattr(unimodal, "get_model_config", lambda chunk: provider_config(True))
    model, ddp_config, groups = [Mock()], SimpleNamespace(megatron_fsdp_version=2), Mock()
    assert unimodal._ddp_wrap(
        model, True, ddp_config, True, True, True, pg_collection=groups, use_layer_wise_distributed_optimizer=True
    ) == ["custom"]
    custom_wrap.assert_called_once_with(
        model=model,
        data_parallel_random_init=True,
        ddp_config=ddp_config,
        overlap_param_gather_with_optimizer_step=True,
        use_torch_fsdp2=True,
        pg_collection=groups,
        use_layer_wise_distributed_optimizer=True,
    )
    core_wrap.assert_not_called()


@pytest.mark.unit
@pytest.mark.parametrize("mixed", [False, True])
def test_empty_or_mixed_chunks_do_not_select_custom_wrapper(monkeypatch, mixed):
    core_wrap, custom_wrap = Mock(return_value=[]), Mock()
    monkeypatch.setattr(unimodal, "_core_ddp_wrap", core_wrap)
    monkeypatch.setattr(deepseek_v4_fsdp, "wrap_deepseek_v4_fsdp", custom_wrap)
    models = (
        [Mock(config=provider_config(True)), Mock(config=SimpleNamespace(enable_mhc_connections=True))]
        if mixed
        else []
    )
    monkeypatch.setattr(unimodal, "get_model_config", lambda chunk: chunk.config)
    unimodal._ddp_wrap(models, False, SimpleNamespace(megatron_fsdp_version=2), False, True, pg_collection=Mock())
    core_wrap.assert_called_once()
    custom_wrap.assert_not_called()
