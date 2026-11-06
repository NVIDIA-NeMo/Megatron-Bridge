# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from megatron.core import parallel_state as mpu

from megatron.bridge.models.megatron_mimo.build_model import (
    _bridge_parallel_state_globals,
    build_megatron_mimo_model,
)


pytestmark = pytest.mark.unit


@pytest.fixture
def parallel_groups(monkeypatch):
    groups = SimpleNamespace(
        tp=object(),
        dp=object(),
        dp_cp=object(),
        pp=object(),
        ep=object(),
        expt_tp=object(),
        tp_ep=object(),
        tp_ep_pp=object(),
        cp=object(),
        mp=object(),
    )
    for global_name in (
        "_GLOBAL_MEMORY_BUFFER",
        "_TENSOR_MODEL_PARALLEL_GROUP",
        "_DATA_PARALLEL_GROUP",
        "_DATA_PARALLEL_GROUP_WITH_CP",
        "_PIPELINE_MODEL_PARALLEL_GROUP",
        "_EXPERT_MODEL_PARALLEL_GROUP",
        "_EXPERT_TENSOR_PARALLEL_GROUP",
        "_EXPERT_DATA_PARALLEL_GROUP",
        "_INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP",
        "_EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP",
        "_EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP",
        "_CONTEXT_PARALLEL_GROUP",
        "_MODEL_PARALLEL_GROUP",
    ):
        monkeypatch.setattr(mpu, global_name, None)

    return groups


def test_bridge_parallel_state_uses_mcore_expert_group_fields(parallel_groups):
    groups = parallel_groups
    _bridge_parallel_state_globals(groups)

    assert mpu._EXPERT_MODEL_PARALLEL_GROUP is groups.ep
    assert mpu._EXPERT_TENSOR_PARALLEL_GROUP is groups.expt_tp
    assert mpu._EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP is groups.tp_ep
    assert mpu._EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP is groups.tp_ep_pp


@pytest.mark.parametrize("active_module", ["language", "images"])
def test_build_megatron_mimo_model_bridges_groups_before_construction(monkeypatch, parallel_groups, active_module):
    groups = parallel_groups
    infra = SimpleNamespace(
        module_to_grid_map={
            "language": SimpleNamespace(rank_offset=0, size=1),
            "images": SimpleNamespace(rank_offset=1, size=1),
        },
        pg_collections={name: groups if name == active_module else None for name in ("language", "images")},
    )
    target = "megatron.bridge.models.megatron_mimo.build_model"
    monkeypatch.setattr(f"{target}.dist.is_initialized", lambda: True)
    monkeypatch.setattr(f"{target}.dist.get_rank", lambda: 0 if active_module == "language" else 1)
    monkeypatch.setattr(f"{target}.dist.get_world_size", lambda: 2)
    monkeypatch.setattr(f"{target}._set_per_module_random_seeds", Mock())

    class ModelWithGroupChecks:
        def __init__(self):
            # Constructors must see the component groups, before any layers are built.
            assert mpu.get_tensor_model_parallel_group() is groups.tp
            assert mpu.get_pipeline_model_parallel_group() is groups.pp
            assert mpu.get_expert_model_parallel_group() is groups.ep
            assert mpu.get_expert_tensor_parallel_group() is groups.expt_tp

    provider = Mock(
        megatron_mimo_parallelism_config=object(),
        build_infra=Mock(return_value=infra),
        provide_distributed_model=Mock(side_effect=lambda **kwargs: [ModelWithGroupChecks()]),
    )

    model, built_infra = build_megatron_mimo_model(provider)

    assert isinstance(model, ModelWithGroupChecks)
    assert built_infra is infra
    provider.provide_distributed_model.assert_called_once()
