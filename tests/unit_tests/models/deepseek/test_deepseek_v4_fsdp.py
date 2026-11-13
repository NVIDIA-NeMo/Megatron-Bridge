from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from megatron.bridge.models.deepseek import deepseek_v4_fsdp


def wrap_arguments():
    return dict(
        model=[Mock()],
        data_parallel_random_init=False,
        ddp_config=SimpleNamespace(megatron_fsdp_version=2),
        overlap_param_gather_with_optimizer_step=False,
        use_torch_fsdp2=False,
        pg_collection=Mock(),
        use_layer_wise_distributed_optimizer=False,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "granularity,modules,stack",
    [(None, None, False), ("selective", ["mla_up_proj", "moe_act"], False), ("selective", ["mhc"], True)],
)
def test_unit_policy_and_constructor_arguments(monkeypatch, granularity, modules, stack):
    from megatron.core.distributed.fsdp import mcore_fsdp_adapter
    from megatron.core.models.hybrid.hybrid_block import HybridStack
    from megatron.core.models.hybrid.layers.hybrid_hyper_connection import HyperConnectionHybridLayer

    constructor = Mock(side_effect=["first", "second"])
    monkeypatch.setattr(mcore_fsdp_adapter, "FullyShardedDataParallelV2", constructor)
    config = SimpleNamespace(
        overlap_moe_expert_parallel_comm=False, recompute_granularity=granularity, recompute_modules=modules
    )
    monkeypatch.setattr(deepseek_v4_fsdp, "get_model_config", lambda chunk: config)
    arguments = wrap_arguments()
    arguments["model"] = [Mock(), Mock()]
    assert deepseek_v4_fsdp.wrap_deepseek_v4_fsdp(**arguments) == ["first", "second"]
    assert constructor.call_count == 2
    for chunk, call in zip(arguments["model"], constructor.call_args_list):
        assert call.kwargs == dict(
            config=config,
            ddp_config=arguments["ddp_config"],
            module=chunk,
            fsdp_unit_modules=[HybridStack if stack else HyperConnectionHybridLayer],
            disable_bucketing=False,
            pg_collection=arguments["pg_collection"],
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "conflict",
    [
        "use_torch_fsdp2",
        "use_layer_wise_distributed_optimizer",
        "overlap_param_gather_with_optimizer_step",
        "data_parallel_random_init",
        "ep_overlap",
        "version",
    ],
)
def test_rejects_incompatible_lifecycle(monkeypatch, conflict):
    arguments = wrap_arguments()
    if conflict == "version":
        arguments["ddp_config"].megatron_fsdp_version = 1
    elif conflict != "ep_overlap":
        arguments[conflict] = True
    config = SimpleNamespace(overlap_moe_expert_parallel_comm=conflict == "ep_overlap")
    monkeypatch.setattr(deepseek_v4_fsdp, "get_model_config", lambda chunk: config)
    with pytest.raises(ValueError):
        deepseek_v4_fsdp.wrap_deepseek_v4_fsdp(**arguments)
