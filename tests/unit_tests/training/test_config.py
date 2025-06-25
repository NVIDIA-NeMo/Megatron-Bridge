# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for configuration validation in megatron.hub.training.config."""

from typing import Any, Optional, Union

import pytest
import torch
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig

from megatron.hub.models.gpt import GPTConfig
from megatron.hub.models.t5 import T5Config
from megatron.hub.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedInitConfig,
    FinetuningDatasetConfig,
    GPTDatasetConfig,
    LoggerConfig,
    MockGPTDatasetConfig,
    NVRxStragglerDetectionConfig,
    ProfilingConfig,
    RerunStateMachineConfig,
    RNGConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainingConfig,
)


def mock_get_world_size_safe(world_size_to_return: int):
    """
    Factory for a mock version of `get_world_size_safe`.

    Args:
        world_size_to_return: The integer value the mock function should return.

    Returns:
        A function that, when called, returns `world_size_to_return`.
    """

    def _mock():
        return world_size_to_return

    return _mock


def create_test_gpt_config(**kwargs: Any) -> GPTConfig:
    """Creates an instance of GPTConfig for testing."""
    defaults = {
        "num_layers": 1,
        "hidden_size": 128,
        "num_attention_heads": 4,
        "seq_length": 512,
        "apply_rope_fusion": False,
    }
    defaults.update(kwargs)
    return GPTConfig(**defaults)


def create_test_t5_config(**kwargs: Any) -> T5Config:
    """Creates an instance of T5Config with sensible defaults for testing."""
    defaults = {
        "num_layers": 1,
        "hidden_size": 128,
        "num_attention_heads": 4,
        "seq_length": 512,
        "apply_rope_fusion": False,
    }
    defaults.update(kwargs)
    return T5Config(**defaults)


def create_test_training_config(**kwargs: Any) -> TrainingConfig:
    """Creates an instance of TrainingConfig with defaults for testing."""
    defaults = {
        "global_batch_size": 32,
        "train_iters": 1000,
    }
    defaults.update(kwargs)
    return TrainingConfig(**defaults)


def create_test_optimizer_config(**kwargs: Any) -> OptimizerConfig:
    """Creates an instance of OptimizerConfig with defaults for testing."""
    defaults = {
        "lr": 0.0001,
        "use_distributed_optimizer": False,
    }
    defaults.update(kwargs)
    return OptimizerConfig(**defaults)


def create_test_scheduler_config(**kwargs: Any) -> SchedulerConfig:
    """Creates an instance of SchedulerConfig with defaults for testing."""
    defaults = {
        "lr_decay_style": "linear",
        "lr_warmup_iters": 0,
    }
    defaults.update(kwargs)
    return SchedulerConfig(**defaults)


def create_test_gpt_dataset_config(sequence_length: int) -> GPTDatasetConfig:
    """Creates an instance of GPTDatasetConfig with defaults for testing."""
    return GPTDatasetConfig(
        random_seed=1234,
        sequence_length=sequence_length,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
    )


def create_test_finetuning_dataset_config(sequence_length: int) -> FinetuningDatasetConfig:
    """Creates an instance of FinetuningDatasetConfig with defaults for testing."""
    return FinetuningDatasetConfig(sequence_length=sequence_length)


def create_test_logger_config(**kwargs: Any) -> LoggerConfig:
    """Creates an instance of LoggerConfig with defaults for testing."""
    return LoggerConfig(**kwargs)


def create_test_tokenizer_config(**kwargs: Any) -> TokenizerConfig:
    """Creates an instance of TokenizerConfig with defaults for testing."""
    return TokenizerConfig(**kwargs)


def create_test_checkpoint_config(**kwargs: Any) -> CheckpointConfig:
    """Creates an instance of CheckpointConfig with defaults for testing."""
    defaults = {
        "ckpt_format": "torch_dist",
    }
    defaults.update(kwargs)
    return CheckpointConfig(**defaults)


def create_test_distributed_init_config(**kwargs: Any) -> DistributedInitConfig:
    """Creates an instance of DistributedInitConfig with defaults for testing."""
    defaults = {
        "use_gloo_process_groups": True,
        "lazy_init": False,
    }
    defaults.update(kwargs)
    return DistributedInitConfig(**defaults)


def create_test_profiling_config(**kwargs: Any) -> ProfilingConfig:
    """Creates an instance of ProfilingConfig with defaults for testing."""
    defaults = {
        "use_pytorch_profiler": False,
        "use_nsys_profiler": False,
    }
    defaults.update(kwargs)
    return ProfilingConfig(**defaults)


def create_test_nvrx_straggler_config(**kwargs: Any) -> NVRxStragglerDetectionConfig:
    """Creates an instance of NVRxStragglerDetectionConfig with defaults for testing."""
    defaults = {
        "calc_relative_gpu_perf": True,
        "calc_individual_gpu_perf": True,
    }
    defaults.update(kwargs)
    return NVRxStragglerDetectionConfig(**defaults)


def create_test_config_container(
    world_size_override: int,
    model_config: Union[GPTConfig, T5Config],
    train_config: Optional[TrainingConfig] = None,
    optimizer_config: Optional[OptimizerConfig] = None,
    scheduler_config: Optional[SchedulerConfig] = None,
    dataset_config_override: Optional[Union[GPTDatasetConfig, FinetuningDatasetConfig]] = None,
    logger_config: Optional[LoggerConfig] = None,
    tokenizer_config: Optional[TokenizerConfig] = None,
    checkpoint_config: Optional[CheckpointConfig] = None,
    dist_config: Optional[DistributedInitConfig] = None,
    profiling_config: Optional[ProfilingConfig] = None,
):
    """
    Helper to create a ConfigContainer with specified or default test configurations.
    Monkeypatches `get_world_size_safe` for the duration of the test.

    Args:
        world_size_override: The world size for the mock `get_world_size_safe`.
        model_config: The model configuration (GPTConfig or T5Config).
        train_config: Optional override for training configuration.
        optimizer_config: Optional override for optimizer configuration.
        scheduler_config: Optional override for scheduler configuration.
        dataset_config_override: Optional override for dataset configuration.
        logger_config: Optional override for logger configuration.
        tokenizer_config: Optional override for tokenizer configuration.
        checkpoint_config: Optional override for checkpoint configuration.
        dist_config: Optional override for distributed initialization configuration.
        profiling_config: Optional override for profiling configuration.


    Returns:
        A tuple containing the ConfigContainer instance, the original
        `get_world_size_safe` function, and the config module reference.
    """

    final_dataset_config: Union[GPTDatasetConfig, FinetuningDatasetConfig]
    if dataset_config_override:
        final_dataset_config = dataset_config_override
    elif isinstance(model_config, (GPTConfig, T5Config)):  # T5 also uses GPTDataset for these tests
        final_dataset_config = create_test_gpt_dataset_config(sequence_length=model_config.seq_length)
    else:
        raise ValueError(f"Unsupported model_config type for default dataset_config: {type(model_config)}")

    container = ConfigContainer(
        train=train_config or create_test_training_config(),
        model=model_config,
        optimizer=optimizer_config or create_test_optimizer_config(),
        scheduler=scheduler_config or create_test_scheduler_config(),
        dataset=final_dataset_config,
        logger=logger_config or create_test_logger_config(),
        tokenizer=tokenizer_config or create_test_tokenizer_config(),
        checkpoint=checkpoint_config or create_test_checkpoint_config(),
        dist=dist_config or create_test_distributed_init_config(),
        ddp=DistributedDataParallelConfig(),
        rng=RNGConfig(),
        rerun_state_machine=RerunStateMachineConfig(),
        profiling=profiling_config,
    )

    # Monkeypatch get_world_size_safe for this test
    import megatron.hub.training.config as config_module

    original_get_world_size = getattr(config_module, "get_world_size_safe", None)
    config_module.get_world_size_safe = mock_get_world_size_safe(world_size_override)

    return container, original_get_world_size, config_module


def restore_get_world_size_safe(original_func, module_ref):
    """
    Restores the original `get_world_size_safe` function in the given module.

    Args:
        original_func: The original function to restore.
        module_ref: The module where the function was patched.
    """
    if original_func is not None:
        module_ref.get_world_size_safe = original_func


class TestMockGPTDatasetConfig:
    """Tests desired behavior for MockGPTDatasetConfig."""

    def test_initialization(self):
        """Test that blend and blend_per_split fields are always None in MockGPTDatasetConfig."""
        config = MockGPTDatasetConfig(
            random_seed=1234,
            sequence_length=512,
            reset_position_ids=False,
            reset_attention_mask=False,
            eod_mask_loss=False,
        )

        # Should be an instance of both MockGPTDatasetConfig and GPTDatasetConfig
        from megatron.core.datasets.blended_megatron_dataset_config import BlendedMegatronDatasetConfig
        from megatron.core.datasets.gpt_dataset import GPTDatasetConfig as MCoreGPTDatasetConfig

        assert isinstance(config, MockGPTDatasetConfig)
        assert isinstance(config, GPTDatasetConfig)
        assert isinstance(config, MCoreGPTDatasetConfig)
        assert isinstance(config, BlendedMegatronDatasetConfig)

        # Should have all the expected fields from parent class
        assert hasattr(config, "random_seed")
        assert hasattr(config, "sequence_length")
        assert hasattr(config, "path_to_cache")

        # Verify blend fields are None and cannot be accessed via __dict__
        assert config.blend is None
        assert config.blend_per_split is None
        assert config.mock  # should be set by BlendedMegatronDatasetConfig post-init
        assert "blend" not in config.__dict__
        assert "blend_per_split" not in config.__dict__

    def test_cannot_set_blend_fields(self):
        """Test that blend and blend_per_split fields cannot be set during initialization."""
        # These should raise a TypeError because blend and blend_per_split are marked as init=False
        with pytest.raises(TypeError, match="got an unexpected keyword argument 'blend'"):
            MockGPTDatasetConfig(
                random_seed=1234,
                sequence_length=512,
                reset_position_ids=False,
                reset_attention_mask=False,
                eod_mask_loss=False,
                blend=(["some", "data", "paths"], None),  # This should fail
            )

        with pytest.raises(TypeError, match="got an unexpected keyword argument 'blend_per_split'"):
            MockGPTDatasetConfig(
                random_seed=1234,
                sequence_length=512,
                reset_position_ids=False,
                reset_attention_mask=False,
                eod_mask_loss=False,
                blend_per_split=[
                    (["train", "paths"], None),
                    (["valid", "paths"], None),
                    (["test", "paths"], None),
                ],  # This should fail
            )

        with pytest.raises(TypeError, match="got an unexpected keyword argument"):
            MockGPTDatasetConfig(
                random_seed=1234,
                sequence_length=512,
                reset_position_ids=False,
                reset_attention_mask=False,
                eod_mask_loss=False,
                blend=(["some", "data", "paths"], None),
                blend_per_split=[(["train", "paths"], None), (["valid", "paths"], None), (["test", "paths"], None)],
            )


class TestConfigContainerValidation:
    """Tests for the `validate` method of the `ConfigContainer` class."""

    @pytest.mark.parametrize(
        "world_size, expect_assertion_error",
        [
            (8, False),
            (7, True),
        ],
    )
    def test_world_size_divisibility_gpt(self, monkeypatch, world_size, expect_assertion_error):
        """Test world size divisibility by model_size for GPT."""
        gpt_model_cfg = create_test_gpt_config(
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=2,
            context_parallel_size=1,
            pipeline_dtype=torch.bfloat16,
        )
        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=world_size, model_config=gpt_model_cfg
        )

        try:
            if expect_assertion_error:
                with pytest.raises(AssertionError, match="is not divisible by"):
                    container.validate()
            else:
                container.validate()  # Should pass without error
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    @pytest.mark.parametrize(
        "world_size, expect_assertion_error",
        [
            (10, False),
            (9, True),
        ],
    )
    def test_world_size_divisibility_t5(self, monkeypatch, world_size, expect_assertion_error):
        """Test world size divisibility by model_size for GPT."""
        gpt_model_cfg = create_test_t5_config(
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=1,
            encoder_pipeline_model_parallel_size=2,
            context_parallel_size=1,
            pipeline_dtype=torch.bfloat16,
        )
        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=world_size, model_config=gpt_model_cfg
        )

        try:
            if expect_assertion_error:
                with pytest.raises(AssertionError, match="is not divisible by"):
                    container.validate()
            else:
                container.validate()  # Should pass without error
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_cpu_initialization_with_lazy_init(self, monkeypatch):
        """Test `use_cpu_initialization` is True if `lazy_init` is True."""
        gpt_model_cfg = create_test_gpt_config(use_cpu_initialization=False)
        dist_cfg = create_test_distributed_init_config(lazy_init=True)

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=4, model_config=gpt_model_cfg, dist_config=dist_cfg
        )
        try:
            container.validate()
            assert container.model.use_cpu_initialization is True
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_cpu_initialization_persists_if_true(self, monkeypatch):
        """Test `use_cpu_initialization` remains True if initially True."""
        gpt_model_cfg_true = create_test_gpt_config(use_cpu_initialization=True)

        # Case 1: lazy_init is False
        dist_cfg_lazy_false = create_test_distributed_init_config(lazy_init=False)
        container1, og1, mod1 = create_test_config_container(
            world_size_override=4, model_config=gpt_model_cfg_true, dist_config=dist_cfg_lazy_false
        )
        try:
            container1.validate()
            assert container1.model.use_cpu_initialization is True
        finally:
            restore_get_world_size_safe(og1, mod1)

        # Case 2: lazy_init is True
        dist_cfg_lazy_true = create_test_distributed_init_config(lazy_init=True)
        gpt_model_cfg_true_case2 = create_test_gpt_config(use_cpu_initialization=True)
        container2, og2, mod2 = create_test_config_container(
            world_size_override=4, model_config=gpt_model_cfg_true_case2, dist_config=dist_cfg_lazy_true
        )
        try:
            container2.validate()
            assert container2.model.use_cpu_initialization is True
        finally:
            restore_get_world_size_safe(og2, mod2)

    def test_distributed_optimizer_with_torch_dist_checkpointing_passes(self, monkeypatch):
        """Test validation passes: distributed optimizer, no gloo, torch_dist checkpoint."""
        gpt_model_cfg = create_test_gpt_config()
        dist_cfg = create_test_distributed_init_config(use_gloo_process_groups=False)
        opt_cfg = create_test_optimizer_config(use_distributed_optimizer=True)
        chkpt_cfg = create_test_checkpoint_config(ckpt_format="torch_dist")

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=4,
            model_config=gpt_model_cfg,
            dist_config=dist_cfg,
            optimizer_config=opt_cfg,
            checkpoint_config=chkpt_cfg,
        )
        try:
            container.validate()
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_scheduler_lr_decay_iters_default(self, monkeypatch):
        """Test `lr_decay_iters` defaults to `train_iters` and `lr_decay_steps` calculation."""
        gpt_model_cfg = create_test_gpt_config()
        train_cfg = create_test_training_config(train_iters=2000, global_batch_size=32)
        sched_cfg = create_test_scheduler_config(lr_decay_iters=None)

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1, model_config=gpt_model_cfg, train_config=train_cfg, scheduler_config=sched_cfg
        )
        try:
            container.validate()
            assert container.scheduler.lr_decay_iters == train_cfg.train_iters
            assert container.scheduler.lr_decay_steps == train_cfg.train_iters * train_cfg.global_batch_size
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_scheduler_lr_decay_iters_custom(self, monkeypatch):
        """Test custom `lr_decay_iters` and `lr_decay_steps` calculation."""
        gpt_model_cfg = create_test_gpt_config()
        train_cfg = create_test_training_config(train_iters=2000, global_batch_size=32)
        custom_lr_decay_iters = 1500
        sched_cfg = create_test_scheduler_config(lr_decay_iters=custom_lr_decay_iters)

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1, model_config=gpt_model_cfg, train_config=train_cfg, scheduler_config=sched_cfg
        )
        try:
            container.validate()
            assert container.scheduler.lr_decay_iters == custom_lr_decay_iters
            assert container.scheduler.lr_decay_steps == custom_lr_decay_iters * train_cfg.global_batch_size
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_scheduler_wd_incr_steps(self, monkeypatch):
        """Test `wd_incr_steps` calculation."""
        gpt_model_cfg = create_test_gpt_config()
        train_cfg = create_test_training_config(train_iters=500, global_batch_size=16)
        sched_cfg = create_test_scheduler_config()

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1, model_config=gpt_model_cfg, train_config=train_cfg, scheduler_config=sched_cfg
        )
        try:
            container.validate()
            expected_wd_incr_steps = train_cfg.train_iters * train_cfg.global_batch_size
            assert container.scheduler.wd_incr_steps == expected_wd_incr_steps
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_scheduler_wsd_decay_steps(self, monkeypatch):
        """Test `wsd_decay_steps` calculation when `lr_wsd_decay_iters` is set."""
        gpt_model_cfg = create_test_gpt_config()
        # train_iters is needed for lr_decay_iters default in scheduler validation if not set
        train_cfg = create_test_training_config(global_batch_size=8, train_iters=100)
        lr_wsd_decay_iters = 100
        sched_cfg = create_test_scheduler_config(lr_wsd_decay_iters=lr_wsd_decay_iters)

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1, model_config=gpt_model_cfg, train_config=train_cfg, scheduler_config=sched_cfg
        )
        try:
            container.validate()
            expected_wsd_decay_steps = lr_wsd_decay_iters * train_cfg.global_batch_size
            assert container.scheduler.wsd_decay_steps == expected_wsd_decay_steps
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_scheduler_wsd_decay_steps_none(self, monkeypatch):
        """Test `wsd_decay_steps` is None when `lr_wsd_decay_iters` is None."""
        gpt_model_cfg = create_test_gpt_config()
        train_cfg = create_test_training_config()
        sched_cfg = create_test_scheduler_config(lr_wsd_decay_iters=None)

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1, model_config=gpt_model_cfg, train_config=train_cfg, scheduler_config=sched_cfg
        )
        try:
            container.validate()
            assert container.scheduler.wsd_decay_steps is None
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_scheduler_lr_warmup_steps_from_fraction(self, monkeypatch):
        """Test `lr_warmup_steps` calculation from `lr_warmup_fraction`."""
        gpt_model_cfg = create_test_gpt_config()
        train_cfg = create_test_training_config(train_iters=1000)
        lr_warmup_fraction = 0.1
        sched_cfg = create_test_scheduler_config(
            lr_warmup_fraction=lr_warmup_fraction, lr_warmup_iters=0
        )  # lr_decay_iters defaults to train_iters

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1, model_config=gpt_model_cfg, train_config=train_cfg, scheduler_config=sched_cfg
        )
        try:
            container.validate()
            # lr_decay_iters in scheduler_config defaults to train_config.train_iters
            expected_lr_warmup_steps = lr_warmup_fraction * train_cfg.train_iters
            assert container.scheduler.lr_warmup_steps == expected_lr_warmup_steps
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_scheduler_lr_warmup_steps_from_iters(self, monkeypatch):
        """Test `lr_warmup_steps` calculation from `lr_warmup_iters`."""
        gpt_model_cfg = create_test_gpt_config()
        train_cfg = create_test_training_config(global_batch_size=10)
        lr_warmup_iters = 50
        sched_cfg = create_test_scheduler_config(lr_warmup_fraction=None, lr_warmup_iters=lr_warmup_iters)

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1, model_config=gpt_model_cfg, train_config=train_cfg, scheduler_config=sched_cfg
        )
        try:
            container.validate()
            expected_lr_warmup_steps = lr_warmup_iters * train_cfg.global_batch_size
            assert container.scheduler.lr_warmup_steps == expected_lr_warmup_steps
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_scheduler_lr_warmup_steps_fraction_precedence(self, monkeypatch):
        """Test `lr_warmup_fraction` takes precedence over `lr_warmup_iters`."""
        gpt_model_cfg = create_test_gpt_config()
        train_cfg = create_test_training_config(train_iters=1000, global_batch_size=10)
        lr_warmup_fraction = 0.05
        lr_warmup_iters = 50  # This should be ignored
        sched_cfg = create_test_scheduler_config(
            lr_warmup_fraction=lr_warmup_fraction, lr_warmup_iters=lr_warmup_iters
        )
        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1, model_config=gpt_model_cfg, train_config=train_cfg, scheduler_config=sched_cfg
        )
        try:
            container.validate()
            expected_lr_warmup_steps = lr_warmup_fraction * train_cfg.train_iters
            assert container.scheduler.lr_warmup_steps == expected_lr_warmup_steps
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    @pytest.mark.parametrize(
        "use_pytorch_profiler, use_nsys_profiler, expect_assertion_error",
        [
            (True, False, False),  # Only PyTorch enabled
            (False, True, False),  # Only Nsys enabled
            (True, True, True),  # Both enabled (Error)
            (False, False, False),  # Neither enabled
        ],
    )
    def test_profiling_config_instantiation_validation(
        self, monkeypatch, use_pytorch_profiler, use_nsys_profiler, expect_assertion_error
    ):
        """Test ProfilingConfig __post_init__ validation for profiler exclusivity."""

        if expect_assertion_error:
            with pytest.raises(AssertionError, match="Exactly one of pytorch or nsys profiler should be enabled"):
                prof_cfg = create_test_profiling_config(
                    use_pytorch_profiler=use_pytorch_profiler, use_nsys_profiler=use_nsys_profiler
                )
        else:
            # No error expected at instantiation
            prof_cfg = create_test_profiling_config(
                use_pytorch_profiler=use_pytorch_profiler, use_nsys_profiler=use_nsys_profiler
            )
            gpt_model_cfg = create_test_gpt_config()
            container, og_ws, cfg_mod = create_test_config_container(
                world_size_override=1, model_config=gpt_model_cfg, profiling_config=prof_cfg
            )
            try:
                container.validate()
            finally:
                restore_get_world_size_safe(og_ws, cfg_mod)


class TestRerunConfigValidation:
    """
    Test that any assertions or modifications done by __post_init__() functions
    are idempotent when the config is unchanged. Tests the same for ConfigContainer.validate().
    """

    def _check_post_init_idempotency(self, cfg_init_fn):
        import copy

        cfg = cfg_init_fn()
        cfg_copy = copy.deepcopy(cfg)
        assert cfg == cfg_copy

        # rerun post-init
        cfg.__post_init__()
        assert cfg == cfg_copy

    def test_scheduler_config(self):
        self._check_post_init_idempotency(create_test_scheduler_config)

        # Test rerun of post-init with valid and invalid changes
        cfg = create_test_scheduler_config(lr_decay_iters=10)
        cfg.lr_decay_iters = 20
        cfg.__post_init__()

        with pytest.raises(AssertionError, match="start_weight_decay"):
            cfg.start_weight_decay = -5.2
            cfg.__post_init__()

    def test_gptdataset_config(self):
        self._check_post_init_idempotency(create_test_gpt_dataset_config)

        # Test rerun of post-init with valid and invalid changes
        cfg = create_test_gpt_dataset_config(1024)
        cfg.random_seed = 2468
        cfg.__post_init__()

        with pytest.raises(AssertionError, match="reset_position_ids"):
            cfg.reset_position_ids = None
            cfg.__post_init__()

    def test_profiling_config(self):
        self._check_post_init_idempotency(create_test_profiling_config)

        # Test rerun of post-init with valid and invalid changes
        cfg = create_test_profiling_config()
        cfg.profile_step_end = 1000
        cfg.__post_init__()

        with pytest.raises(AssertionError, match="one of pytorch or nsys profiler should be enabled"):
            cfg.use_nsys_profiler = True
            cfg.use_pytorch_profiler = True
            cfg.__post_init__()

    def test_nvrx_straggler_config(self):
        self._check_post_init_idempotency(create_test_nvrx_straggler_config)

        # Test rerun of post-init with valid and invalid changes
        cfg = create_test_nvrx_straggler_config()
        cfg.num_gpu_perf_scores_to_print = 2
        cfg.__post_init__()

        with pytest.raises(AssertionError, match="report_time_interval must be positive"):
            cfg.report_time_interval = -100.0
            cfg.__post_init__()

    def test_rerun_validate_config_container(self):
        import copy
        from dataclasses import fields

        gpt_cfg = create_test_gpt_config()
        full_cfg, og_ws, cfg_mod = create_test_config_container(world_size_override=8, model_config=gpt_cfg)

        def check_container_state_matches(cfg1, cfg2):
            for f1 in fields(cfg1):
                sub_cfg1 = getattr(cfg1, f1.name)
                assert hasattr(cfg2, f1.name)
                sub_cfg2 = getattr(cfg2, f1.name)
                assert sub_cfg1 == sub_cfg2
            for f2 in fields(cfg2):
                sub_cfg2 = getattr(cfg2, f2.name)
                assert hasattr(cfg1, f2.name)
                sub_cfg1 = getattr(cfg2, f2.name)
                assert sub_cfg1 == sub_cfg2

        try:
            # idempotency
            full_cfg.validate()
            full_cfg_copy = copy.deepcopy(full_cfg)
            check_container_state_matches(full_cfg, full_cfg_copy)
            full_cfg.validate()
            check_container_state_matches(full_cfg, full_cfg_copy)

            # test rerun of validate with valid and invalid changes
            full_cfg.scheduler.lr_decay_iters = 20
            full_cfg.validate()

            with pytest.raises(AssertionError, match="start_weight_decay"):
                full_cfg.scheduler.start_weight_decay = -5.2
                full_cfg.validate()
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)
