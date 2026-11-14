from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from megatron.bridge.models.common import unimodal


@pytest.mark.unit
@pytest.mark.parametrize("use_fsdp,version,mhc", [(False, 2, True), (True, 1, True), (True, 2, False)])
def test_non_mhc_v2_delegates(monkeypatch, use_fsdp, version, mhc):
    core_wrap = Mock(return_value=["wrapped"])
    monkeypatch.setattr(unimodal, "_core_ddp_wrap", core_wrap)
    monkeypatch.setattr(unimodal, "get_model_config", lambda chunk: SimpleNamespace(enable_mhc_connections=mhc))
    model = [Mock()]
    config = SimpleNamespace(megatron_fsdp_version=version)
    groups = Mock()
    assert unimodal._ddp_wrap(model, False, config, False, use_fsdp, pg_collection=groups) == ["wrapped"]
    core_wrap.assert_called_once_with(
        model,
        False,
        config,
        False,
        use_megatron_fsdp=use_fsdp,
        use_torch_fsdp2=False,
        pg_collection=groups,
        use_layer_wise_distributed_optimizer=False,
        use_layer_wise_param_layout=True,
    )


@pytest.mark.unit
@pytest.mark.parametrize("recompute", [False, True])
def test_mhc_replaces_inner_transformer_units(monkeypatch, recompute):
    from megatron.core.distributed.fsdp import mcore_fsdp_adapter
    from megatron.core.models.hybrid.hybrid_block import HybridStack
    from megatron.core.models.hybrid.layers.hybrid_hyper_connection import HyperConnectionHybridLayer

    constructor = Mock(return_value="wrapped")
    monkeypatch.setattr(mcore_fsdp_adapter, "FullyShardedDataParallelV2", constructor)
    transformer_config = SimpleNamespace(
        enable_mhc_connections=True,
        overlap_moe_expert_parallel_comm=False,
        recompute_granularity="selective" if recompute else None,
        recompute_modules=["mhc"] if recompute else None,
    )
    monkeypatch.setattr(unimodal, "get_model_config", lambda chunk: transformer_config)
    core_wrap = Mock()
    monkeypatch.setattr(unimodal, "_core_ddp_wrap", core_wrap)
    model = Mock()
    config = SimpleNamespace(megatron_fsdp_version=2)
    groups = Mock()
    assert unimodal._ddp_wrap([model], False, config, False, True, pg_collection=groups) == ["wrapped"]
    constructor.assert_called_once_with(
        config=transformer_config,
        ddp_config=config,
        module=model,
        fsdp_unit_modules=[HybridStack if recompute else HyperConnectionHybridLayer],
        disable_bucketing=False,
        pg_collection=groups,
    )
    core_wrap.assert_not_called()


@pytest.mark.unit
@pytest.mark.parametrize("conflict", ["torch_fsdp2", "layer_wise_optimizer", "optimizer_step_gather"])
def test_mhc_rejects_conflicting_wrapper(monkeypatch, conflict):
    monkeypatch.setattr(unimodal, "get_model_config", lambda chunk: SimpleNamespace(enable_mhc_connections=True))
    with pytest.raises(ValueError, match="native optimizer"):
        unimodal._ddp_wrap(
            [Mock()],
            False,
            SimpleNamespace(megatron_fsdp_version=2),
            conflict == "optimizer_step_gather",
            use_megatron_fsdp=True,
            use_torch_fsdp2=conflict == "torch_fsdp2",
            use_layer_wise_distributed_optimizer=conflict == "layer_wise_optimizer",
            pg_collection=Mock(),
        )


@pytest.mark.unit
@pytest.mark.parametrize("random_init,ep_overlap", [(True, False), (False, True)])
def test_mhc_rejects_unsupported_lifecycle(monkeypatch, random_init, ep_overlap):
    config = SimpleNamespace(enable_mhc_connections=True, overlap_moe_expert_parallel_comm=ep_overlap)
    monkeypatch.setattr(unimodal, "get_model_config", lambda chunk: config)
    with pytest.raises(ValueError):
        unimodal._ddp_wrap(
            [Mock()], random_init, SimpleNamespace(megatron_fsdp_version=2), False, True, pg_collection=Mock()
        )
