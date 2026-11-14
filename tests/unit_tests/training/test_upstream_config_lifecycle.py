"""Recipe edits must reach upstream validation and runtime derivation."""

import copy
from dataclasses import asdict, fields, replace

import pytest
import torch
from megatron.core.distributed import DistributedDataParallelConfig as CoreDDPConfig
from megatron.core.optimizer import OptimizerConfig as CoreOptimizerConfig
from megatron.training.config import RNGConfig as TrainingRNGConfig

from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedDataParallelConfig,
    DistributedInitConfig,
    LoggerConfig,
    OptimizerConfig,
    RNGConfig,
    megatron_mimo_runtime_config_update,
    runtime_config_update,
)
from tests.unit_tests.training.test_config import (
    create_test_config_container,
    create_test_gpt_config,
    restore_get_world_size_safe,
)


pytestmark = pytest.mark.unit


@pytest.mark.parametrize("update", [runtime_config_update, megatron_mimo_runtime_config_update])
@pytest.mark.parametrize("optimizer_type", [CoreOptimizerConfig, OptimizerConfig])
@pytest.mark.parametrize("ddp_type", [CoreDDPConfig, DistributedDataParallelConfig])
def test_recipe_edits_reach_native_optimizer_finalization(optimizer_type, ddp_type, update):
    optimizer = optimizer_type()
    # Older dev pins accept torch.dtype only; main also accepts CLI strings.
    dtype = "fp32" if hasattr(optimizer, "finalize_optimizer_config") else torch.float32
    optimizer.main_params_dtype = dtype
    optimizer.exp_avg_dtype = dtype
    ddp = ddp_type()
    cfg, original, module = create_test_config_container(
        world_size_override=1,
        model_config=create_test_gpt_config(),
        optimizer_config=optimizer,
        ddp_config=ddp,
    )
    try:
        update(cfg)
        assert optimizer.main_params_dtype is torch.float32
        assert optimizer.exp_avg_dtype is torch.float32
        state = asdict(optimizer)
        update(cfg)
        assert asdict(optimizer) == state
    finally:
        restore_get_world_size_safe(original, module)


def test_optimizer_invalid_override_is_checked_before_resolution():
    optimizer = CoreOptimizerConfig()
    optimizer.main_params_dtype = "bf16"
    cfg, original, module = create_test_config_container(1, create_test_gpt_config(), optimizer_config=optimizer)
    try:
        with pytest.raises(AssertionError, match="main_params_dtype"):
            runtime_config_update(cfg)
        assert optimizer.main_params_dtype == "bf16"
    finally:
        restore_get_world_size_safe(original, module)


def test_ddp_override_is_validated_after_recipe_construction():
    ddp = DistributedDataParallelConfig(num_buckets=2, bucket_size=1024)
    with pytest.raises(AssertionError):
        ddp.finalize()
    ddp.num_buckets = None
    ddp.finalize()
    before = asdict(ddp)
    ddp.finalize()
    assert asdict(ddp) == before


def test_logger_override_reaches_upstream_validator():
    cfg = LoggerConfig(log_interval=10)
    cfg.log_memory_interval = 15
    with pytest.raises(AssertionError):
        cfg.finalize()
    cfg.log_memory_interval = 20
    cfg.finalize()


@pytest.mark.parametrize(
    "section",
    [
        CheckpointConfig(use_persistent_ckpt_worker=True, strict_fsdp_dtensor_load=False),
        DistributedInitConfig(external_gpu_device_mapping=True, use_decentralized_pg=True),
        LoggerConfig(timing_log_level=-1, mlflow_tags={"run": "test"}),
        OptimizerConfig(use_layer_wise_param_layout=False),
        DistributedDataParallelConfig(
            overlap_grad_reduce=True,
            param_name_patterns_for_fp32_local_accumulation=("*.bias",),
        ),
        RNGConfig(seed=42),
        CoreOptimizerConfig(),
        CoreDDPConfig(),
    ],
)
def test_section_replace_copy_and_saved_config_round_trip(section):
    cfg, original, module = create_test_config_container(1, create_test_gpt_config())
    name = {
        CheckpointConfig: "checkpoint",
        DistributedInitConfig: "dist",
        LoggerConfig: "logger",
        OptimizerConfig: "optimizer",
        DistributedDataParallelConfig: "ddp",
        RNGConfig: "rng",
        CoreOptimizerConfig: "optimizer",
        CoreDDPConfig: "ddp",
    }[type(section)]
    try:
        for cloned in (replace(section), copy.deepcopy(section)):
            assert asdict(cloned) == asdict(section)
        assert name in {field.name for field in fields(cfg)}
        setattr(cfg, name, section)
        loaded = ConfigContainer.from_dict(cfg.to_dict())
        assert type(getattr(loaded, name)) is type(section)
        # Core serializes tuple-valued DDP patterns as YAML-compatible lists.
        # Compare the saved section contract, including all public fields and targets.
        assert loaded.to_dict()[name] == cfg.to_dict()[name]
    finally:
        restore_get_world_size_safe(original, module)


def test_rng_reexport_keeps_recipe_failure_at_validation():
    assert RNGConfig is TrainingRNGConfig
    cfg, original, module = create_test_config_container(1, create_test_gpt_config())
    try:
        cfg.rng.seed = 0
        with pytest.raises(ValueError, match="Seed"):
            cfg.validate()
    finally:
        restore_get_world_size_safe(original, module)
