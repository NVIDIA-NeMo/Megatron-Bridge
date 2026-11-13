"""Native config identities and Bridge runtime behavior survive direct imports."""

from copy import deepcopy
from dataclasses import asdict

import pytest
import torch
from megatron.core.distributed import DistributedDataParallelConfig, TorchFullyShardedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig
from megatron.training import config as native

from megatron.bridge.training import config as bridge
from megatron.bridge.training.utils.config_utils import create_ddp_config
from tests.unit_tests.training.test_config import (
    create_test_config_container,
    create_test_gpt_config,
    restore_get_world_size_safe,
)


pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "name",
    [
        "CheckpointConfig",
        "LoggerConfig",
        "RNGConfig",
        "TrainingConfig",
        "SchedulerConfig",
        "ProfilingConfig",
        "RerunStateMachineConfig",
        "StragglerDetectionConfig",
        "ValidationConfig",
    ],
)
def test_training_types_are_native(name):
    assert getattr(bridge, name) is getattr(native, name)


def test_core_types_are_native():
    assert bridge.OptimizerConfig is OptimizerConfig
    assert bridge.DistributedDataParallelConfig is DistributedDataParallelConfig


@pytest.mark.parametrize("update", [bridge.runtime_config_update, bridge.megatron_mimo_runtime_config_update])
def test_native_configs_resolve_recipe_edits_at_runtime(update):
    cfg, original, module = create_test_config_container(1, create_test_gpt_config())
    try:
        cfg.optimizer.main_params_dtype = "fp32"
        cfg.optimizer.exp_avg_dtype = "fp32"
        cfg.logger.log_interval = 10
        cfg.logger.log_memory_interval = 15
        with pytest.raises(AssertionError):
            update(cfg)
        cfg.logger.log_memory_interval = 20
        update(cfg)
        assert cfg.optimizer.main_params_dtype is torch.float32
        assert cfg.optimizer.exp_avg_dtype is torch.float32
        before = asdict(cfg.optimizer)
        update(cfg)
        assert asdict(cfg.optimizer) == before
    finally:
        restore_get_world_size_safe(original, module)


def test_legacy_bridge_targets_reload_as_native_configs():
    cfg, original, module = create_test_config_container(1, create_test_gpt_config())
    try:
        cfg.checkpoint.hf_source_path = "org/source"
        cfg.checkpoint.stage_precision_aware_optimizer_state_on_cpu = True
        cfg.dist.external_gpu_device_mapping = True
        cfg.logger.mlflow_tags = {"model": "tiny"}
        cfg.scheduler.max_steps = 2000
        saved = cfg.to_dict()
        sections = [
            "checkpoint",
            "dist",
            "logger",
            "optimizer",
            "ddp",
            "train",
            "scheduler",
            "rng",
            "rerun_state_machine",
        ]
        for name in sections:
            saved[name]["_target_"] = "megatron.bridge.training.config." + type(getattr(cfg, name)).__name__
        loaded = bridge.ConfigContainer.from_dict(saved)
        for name in sections:
            assert type(getattr(loaded, name)) is type(getattr(cfg, name))
            assert loaded.to_dict()[name] == cfg.to_dict()[name]
        bridge.runtime_config_update(loaded)
        assert loaded.scheduler.lr_decay_iters == 2000
        assert loaded.checkpoint.hf_source_path == "org/source"
    finally:
        restore_get_world_size_safe(original, module)


def test_external_ddp_factory_applies_bridge_precision_policy():
    config = create_ddp_config(
        use_megatron_fsdp=True,
        overrides={
            "grad_reduce_in_fp32": True,
            "megatron_fsdp_main_grads_dtype": torch.bfloat16,
            "megatron_fsdp_grad_comm_dtype": torch.bfloat16,
        },
    )
    assert type(config) is DistributedDataParallelConfig
    assert config.grad_reduce_in_fp32 is False
    assert config.megatron_fsdp_main_grads_dtype is torch.bfloat16


@pytest.mark.parametrize("public_ddp_target", [False, True])
@pytest.mark.parametrize("torch_fsdp2", [False, True])
@pytest.mark.parametrize("fp32_accumulation", [False, True])
def test_legacy_distributed_settings_reload_under_canonical_ddp(torch_fsdp2, fp32_accumulation, public_ddp_target):
    cfg, original, module = create_test_config_container(1, create_test_gpt_config())
    try:
        cfg.straggler = native.StragglerDetectionConfig(disable_straggler_on_startup=True)
        cfg.dist.eval_context_parallel_size = 2
        cfg.dist.enable_megatron_core_experimental = True
        cfg.dist.use_decentralized_pg = True
        saved = cfg.to_dict()
        if public_ddp_target:
            saved["ddp"]["_target_"] = "megatron.core.distributed.DistributedDataParallelConfig"
        saved["dist"]["_target_"] = f"{native.DistributedInitConfig.__module__}.DistributedInitConfig"
        saved["dist"]["use_torch_fsdp2"] = torch_fsdp2
        saved["dist"]["gtp_remat_reduce_scatter_with_fp32_accumulation"] = fp32_accumulation
        saved["ddp"]["gtp_remat_reduce_scatter_with_fp32_accumulation"] = not fp32_accumulation
        saved["straggler"]["enable_straggler_on_startup"] = True
        unchanged = deepcopy(saved)
        loaded = bridge.ConfigContainer.from_dict(saved)
        assert isinstance(loaded.ddp, TorchFullyShardedDataParallelConfig) is torch_fsdp2
        assert loaded.ddp.gtp_remat_reduce_scatter_with_fp32_accumulation is fp32_accumulation
        assert loaded.straggler.disable_straggler_on_startup is True
        assert "enable_straggler_on_startup" not in loaded.to_dict()["straggler"]
        assert loaded.dist.eval_context_parallel_size == 2
        assert loaded.dist.enable_megatron_core_experimental is True
        assert loaded.dist.use_decentralized_pg is True
        assert saved == unchanged
        reloaded = bridge.ConfigContainer.from_dict(loaded.to_dict())
        assert type(reloaded.ddp) is type(loaded.ddp)
        assert reloaded.to_dict() == loaded.to_dict()
    finally:
        restore_get_world_size_safe(original, module)


def test_saved_conversion_config_keeps_ddp_disabled():
    cfg, original, module = create_test_config_container(1, create_test_gpt_config())
    try:
        saved = cfg.to_dict()
        saved["ddp"] = None
        saved["dist"]["use_torch_fsdp2"] = False
        saved["dist"]["gtp_remat_reduce_scatter_with_fp32_accumulation"] = False
        loaded = bridge.ConfigContainer.from_dict(saved)
        assert loaded.ddp is None
        saved["dist"]["use_torch_fsdp2"] = True
        with pytest.raises(ValueError, match="active legacy distributed settings with ddp=None"):
            bridge.ConfigContainer.from_dict(saved)
    finally:
        restore_get_world_size_safe(original, module)
