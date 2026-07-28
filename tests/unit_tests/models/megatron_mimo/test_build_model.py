# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

from megatron.core import parallel_state as mpu

from megatron.bridge.models.megatron_mimo.build_model import _bridge_parallel_state_globals


def test_bridge_parallel_state_uses_mcore_expert_group_fields(monkeypatch):
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

    _bridge_parallel_state_globals(groups)

    assert mpu._EXPERT_MODEL_PARALLEL_GROUP is groups.ep
    assert mpu._EXPERT_TENSOR_PARALLEL_GROUP is groups.expt_tp
    assert mpu._EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP is groups.tp_ep
    assert mpu._EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP is groups.tp_ep_pp
