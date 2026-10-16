# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Checkpoint metadata regressions for mixed layer-wise and distributed optimizers."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from megatron.core.models.mimo.optimizer import MimoOptimizer, ModuleOptimizerInfo
from megatron.core.optimizer import ChainedOptimizer, DistributedOptimizer
from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer

from megatron.bridge.training import checkpointing
from megatron.bridge.training.config import CheckpointConfig, OptimizerConfig


pytestmark = pytest.mark.unit


def _optimizer_tree(*, distributed: bool = True, active: bool = True):
    config = OptimizerConfig(
        optimizer="muon", use_distributed_optimizer=False, use_layer_wise_distributed_optimizer=True
    )
    layer_wise = Mock(spec=LayerWiseDistributedOptimizer, config=config, model_chunks=[])
    dist_opt = Mock(spec=DistributedOptimizer, config=config, model_chunks=[])
    children = [layer_wise, dist_opt] if distributed else [layer_wise]
    chain = ChainedOptimizer([ChainedOptimizer(children)])
    mimo = MimoOptimizer(
        {"encoder": ModuleOptimizerInfo(optimizer=chain, grid=None, pg_collection=None, is_active=active)},
        config,
    )
    return mimo, chain, layer_wise, dist_opt


def test_detects_distributed_instances_in_nested_mixed_optimizer():
    mimo, chain, layer_wise, dist_opt = _optimizer_tree()
    for optimizer in (mimo, chain, dist_opt):
        assert optimizer.config.use_distributed_optimizer is False
        assert checkpointing._has_distributed_optimizer(optimizer)
    assert not checkpointing._has_distributed_optimizer(layer_wise)
    assert not checkpointing._has_distributed_optimizer(None)


@pytest.mark.parametrize("distributed,active", [(False, True), (True, False)])
def test_layer_wise_only_and_inactive_components_do_not_enable_distopt(distributed, active):
    mimo, *_ = _optimizer_tree(distributed=distributed, active=active)
    assert not checkpointing._has_distributed_optimizer(mimo)


@pytest.mark.parametrize("distributed", [False, True])
def test_checkpoint_metadata_follows_actual_optimizer_tree(distributed):
    optimizer, *_ = _optimizer_tree(distributed=distributed)
    metadata = checkpointing._build_sharded_state_dict_metadata(
        checkpointing._uses_distributed_optimizer(optimizer.config, optimizer), CheckpointConfig()
    )
    assert metadata["distrib_optim_sharding_type"] == "dp_reshardable"


def test_runtime_tree_recovers_metadata_when_shared_config_is_stale():
    optimizer, *_ = _optimizer_tree()
    optimizer.config.use_layer_wise_distributed_optimizer = False
    assert checkpointing._uses_distributed_optimizer(optimizer.config, optimizer)


def test_mixed_metadata_selects_distopt_dp_reshardable_handler():
    mimo, _, _, dist_opt = _optimizer_tree()
    metadata = checkpointing._build_sharded_state_dict_metadata(
        checkpointing._has_distributed_optimizer(mimo), CheckpointConfig()
    )
    dist_opt.ddp_config = SimpleNamespace(use_megatron_fsdp=False)
    dist_opt.state_dict.return_value = {}
    dist_opt.checkpoint_fully_reshardable_formats = DistributedOptimizer.checkpoint_fully_reshardable_formats

    result = DistributedOptimizer.sharded_state_dict(dist_opt, {}, metadata=metadata)

    assert result["param_state_sharding_type"] == "dp_reshardable"
    dist_opt.sharded_param_state_dp_reshardable.assert_called_once_with({}, False, metadata)
    dist_opt.sharded_param_state_fs_model_space.assert_not_called()
