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


@pytest.mark.parametrize("runtime_distopt", [False, True])
@pytest.mark.parametrize("saved_distopt", [False, True])
@pytest.mark.parametrize("sharding_type", ["dp_reshardable", "fully_reshardable"])
@pytest.mark.parametrize("resume_mode", ["optimizer", "model_only", "finetune"])
@pytest.mark.parametrize("declared_distopt", ["neither", "target", "source"])
def test_topology_change_checks_source_and_target_optimizer_state(
    monkeypatch: pytest.MonkeyPatch,
    runtime_distopt: bool,
    saved_distopt: bool,
    sharding_type: str,
    resume_mode: str,
    declared_distopt: str,
) -> None:
    """Exercise the load gate without weakening ordinary or mixed DistOpt checks."""
    optimizer, *_ = _optimizer_tree(distributed=runtime_distopt)
    optimizer.config.use_distributed_optimizer = declared_distopt == "target"
    cfg = SimpleNamespace(
        optimizer=optimizer.config,
        model=SimpleNamespace(tensor_model_parallel_size=2, pipeline_model_parallel_size=1),
        checkpoint=CheckpointConfig(
            load_optim=resume_mode != "model_only", load_rng=False, finetune=resume_mode == "finetune"
        ),
    )
    saved_optimizer: dict[int, object] = {0: {"optimizer": {"state": {}, "param_groups": []}}}
    if saved_distopt:
        saved_optimizer[1] = {"param_state_sharding_type": sharding_type}
    common_state = {"optimizer": saved_optimizer, "checkpoint_version": checkpointing._CHECKPOINT_FORMAT_VERSION}
    run_config = {
        "model": {"tensor_model_parallel_size": 1, "pipeline_model_parallel_size": 1},
        "optimizer": {"use_distributed_optimizer": declared_distopt == "source"},
        "checkpoint": {"save_optim": True, "save_rng": False, "fully_parallel_save": True},
    }
    metadata = {"distrib_optim_sharding_type": sharding_type}
    monkeypatch.setattr(checkpointing, "unwrap_model", lambda model: model)
    monkeypatch.setattr(checkpointing, "is_hf_checkpoint_dir", lambda path: False)
    monkeypatch.setattr(
        checkpointing,
        "_load_base_checkpoint",
        Mock(return_value=(common_state, "checkpoint", False, checkpointing.CheckpointType.GLOBAL)),
    )
    monkeypatch.setattr(checkpointing, "file_exists", lambda path: True)
    monkeypatch.setattr(checkpointing, "read_run_config", lambda path: run_config)
    monkeypatch.setattr(checkpointing, "_get_checkpoint_weight_topology", lambda config: None)
    monkeypatch.setattr(checkpointing, "_validate_checkpoint_weight_topology", lambda **kwargs: None)
    monkeypatch.setattr(checkpointing.dist_checkpointing, "load_content_metadata", lambda **kwargs: metadata)
    monkeypatch.setattr(checkpointing, "_checkpoint_dp_cp_group", lambda pg: None)

    class ReachedStateConstruction(Exception):
        pass

    build_state = Mock(side_effect=ReachedStateConstruction)
    monkeypatch.setattr(checkpointing, "generate_state_dict", build_state)
    rejected = (
        resume_mode == "optimizer"
        and (runtime_distopt or saved_distopt or declared_distopt != "neither")
        and sharding_type == "dp_reshardable"
    )
    with pytest.raises(
        RuntimeError if rejected else ReachedStateConstruction,
        match="not supported for DistributedOptimizer" if rejected else None,
    ):
        checkpointing._load_checkpoint_from_path(
            "checkpoint", SimpleNamespace(cfg=cfg), [SimpleNamespace()], optimizer, None, pg_collection=Mock()
        )
    if rejected:
        build_state.assert_not_called()
    else:
        assert (
            build_state.call_args.kwargs["optim_sd_kwargs"]["metadata"]["distrib_optim_sharding_type"] == sharding_type
        )


@pytest.mark.parametrize(
    "state,expected",
    [
        ({0: {"optimizer": {"state": {}}}}, False),
        ({0: {"param_state_sharding_type": "dp_reshardable"}}, True),
        ({"optimizer": [{}, ({"param_state_sharding_type": "fully_reshardable"},)]}, True),
        (None, False),
    ],
)
def test_detects_saved_distributed_optimizer_state(state: object, expected: bool) -> None:
    assert checkpointing._has_distributed_optimizer_state(state) is expected
