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

from typing import Any, Optional, Union
from unittest.mock import MagicMock, patch

import pytest
import torch
from megatron.core.optimizer import OptimizerConfig

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.t5_provider import T5ModelProvider
from megatron.bridge.training.config import (
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


def create_test_gpt_config(**kwargs: Any) -> GPTModelProvider:
    """Creates an instance of GPTConfig for testing."""
    defaults = {
        "num_layers": 1,
        "hidden_size": 128,
        "num_attention_heads": 4,
        "seq_length": 512,
        "apply_rope_fusion": False,
    }
    defaults.update(kwargs)
    return GPTModelProvider(**defaults)


def create_test_t5_config(**kwargs: Any) -> T5ModelProvider:
    """Creates an instance of T5Config with sensible defaults for testing."""
    defaults = {
        "num_layers": 1,
        "hidden_size": 128,
        "num_attention_heads": 4,
        "seq_length": 512,
        "apply_rope_fusion": False,
    }
    defaults.update(kwargs)
    return T5ModelProvider(**defaults)


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
    return FinetuningDatasetConfig(seq_length=sequence_length)


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
    model_config: Union[GPTModelProvider, T5ModelProvider],
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
    elif isinstance(model_config, (GPTModelProvider, T5ModelProvider)):  # T5 also uses GPTDataset for these tests
        final_dataset_config = create_test_gpt_dataset_config(sequence_length=model_config.seq_length)
    else:
        raise ValueError(f"Unsupported model_config type for default dataset_config: {type(model_config)}")

    from megatron.core.distributed import DistributedDataParallelConfig

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
    import megatron.bridge.training.config as config_module

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


def create_test_cp_config_container(cp_size, calc_per_token_loss, avg_in_collective, dataset_type="finetuning"):
    """Helper to create config container for context parallel tests."""
    from megatron.core.distributed import DistributedDataParallelConfig

    gpt_model_cfg = create_test_gpt_config(
        seq_length=512,
        context_parallel_size=cp_size,
        calculate_per_token_loss=calc_per_token_loss,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
    )

    dataset_cfg = (
        create_test_finetuning_dataset_config(sequence_length=512)
        if dataset_type == "finetuning"
        else create_test_gpt_dataset_config(sequence_length=512)
    )

    ddp_cfg = DistributedDataParallelConfig(average_in_collective=avg_in_collective)

    container, og_ws, cfg_mod = create_test_config_container(
        world_size_override=cp_size,
        model_config=gpt_model_cfg,
        dataset_config_override=dataset_cfg,
    )
    container.ddp = ddp_cfg
    return container, og_ws, cfg_mod


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

    def test_packed_sequence_micro_batch_size_validation_error(self, monkeypatch):
        """Test validation error when micro_batch_size > 1 with packed sequences."""
        from megatron.bridge.data.datasets.packed_sequence import PackedSequenceSpecs

        # Create config with micro_batch_size > 1 and packed sequences
        gpt_model_cfg = create_test_gpt_config()
        train_cfg = create_test_training_config(micro_batch_size=4, global_batch_size=32)

        # Create packed sequence specs with packed_sequence_size > 0
        packed_specs = PackedSequenceSpecs(packed_sequence_size=512)
        dataset_cfg = create_test_finetuning_dataset_config(sequence_length=512)
        dataset_cfg.packed_sequence_specs = packed_specs

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=gpt_model_cfg,
            train_config=train_cfg,
            dataset_config_override=dataset_cfg,
        )

        try:
            with pytest.raises(ValueError, match="Micro batch size should be 1 when training with packed sequence"):
                container.validate()
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_packed_sequence_micro_batch_size_validation_passes(self, monkeypatch):
        """Test validation passes when micro_batch_size = 1 with packed sequences."""
        from megatron.bridge.data.datasets.packed_sequence import PackedSequenceSpecs

        # Create config with micro_batch_size = 1 and packed sequences
        gpt_model_cfg = create_test_gpt_config()
        train_cfg = create_test_training_config(micro_batch_size=1, global_batch_size=32)

        # Create packed sequence specs with packed_sequence_size > 0
        packed_specs = PackedSequenceSpecs(packed_sequence_size=512)
        dataset_cfg = create_test_finetuning_dataset_config(sequence_length=512)
        dataset_cfg.packed_sequence_specs = packed_specs

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=gpt_model_cfg,
            train_config=train_cfg,
            dataset_config_override=dataset_cfg,
        )

        try:
            container.validate()  # Should pass without error
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_packed_sequence_validation_skipped_when_specs_none(self, monkeypatch):
        """Test validation skipped when packed_sequence_specs is None."""
        # Create config with micro_batch_size > 1 but no packed sequences
        gpt_model_cfg = create_test_gpt_config()
        train_cfg = create_test_training_config(micro_batch_size=4, global_batch_size=32)
        dataset_cfg = create_test_finetuning_dataset_config(sequence_length=512)
        # packed_sequence_specs defaults to None

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=gpt_model_cfg,
            train_config=train_cfg,
            dataset_config_override=dataset_cfg,
        )

        try:
            container.validate()  # Should pass without error
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_packed_sequence_validation_skipped_for_gpt_dataset(self, monkeypatch):
        """Test validation skipped when using GPTDatasetConfig instead of FinetuningDatasetConfig."""
        # Create config with micro_batch_size > 1 and GPTDatasetConfig
        gpt_model_cfg = create_test_gpt_config()
        train_cfg = create_test_training_config(micro_batch_size=4, global_batch_size=32)
        dataset_cfg = create_test_gpt_dataset_config(sequence_length=512)
        # GPTDatasetConfig doesn't have packed_sequence_specs

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=gpt_model_cfg,
            train_config=train_cfg,
            dataset_config_override=dataset_cfg,
        )

        try:
            container.validate()  # Should pass without error
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    @pytest.mark.parametrize(
        "seq_length, context_parallel_size, expect_assertion_error",
        [
            (512, 2, False),  # 512 % (2 * 2) == 0, valid
            (510, 2, True),  # 510 % (2 * 2) != 0, invalid
            (256, 3, True),  # 256 % (3 * 2) != 0, invalid
        ],
    )
    def test_context_parallel_seq_length_divisibility(
        self, monkeypatch, seq_length, context_parallel_size, expect_assertion_error
    ):
        """Test sequence length must be divisible by 2 * context_parallel_size when CP > 1."""
        gpt_model_cfg = create_test_gpt_config(
            seq_length=seq_length,
            context_parallel_size=context_parallel_size,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=context_parallel_size, model_config=gpt_model_cfg
        )

        try:
            if expect_assertion_error:
                with pytest.raises(
                    AssertionError, match="Sequence length must be divisible by 2 \\* context parallel size"
                ):
                    container.validate()
            else:
                container.validate()
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    @pytest.mark.parametrize(
        "dataset_type, cp_size, calc_per_token_loss, avg_in_collective, expect_error, error_match",
        [
            # FinetuningDatasetConfig with CP > 1 - both checks should trigger
            ("finetuning", 2, False, False, True, "calculate_per_token_loss must be True"),
            ("finetuning", 2, True, True, True, "average_in_collective must be False"),
            ("finetuning", 2, True, False, False, None),  # Valid case
            # GPTDatasetConfig with CP > 1 - checks should be skipped
            ("gpt", 2, False, True, False, None),
            # CP = 1 - checks should be skipped regardless of dataset type
            ("finetuning", 1, False, True, False, None),
        ],
    )
    def test_context_parallel_finetuning_validations(
        self, monkeypatch, dataset_type, cp_size, calc_per_token_loss, avg_in_collective, expect_error, error_match
    ):
        """Test context parallel validations for finetuning configurations."""
        container, og_ws, cfg_mod = create_test_cp_config_container(
            cp_size, calc_per_token_loss, avg_in_collective, dataset_type
        )

        try:
            if expect_error:
                with pytest.raises(AssertionError, match=error_match):
                    container.validate()
            else:
                container.validate()
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    @pytest.mark.parametrize(
        "gpu_major, moe_enable_deepep, expect_error",
        [
            (8, True, False),  # Ampere GPU with DeepEP enabled - should pass
            (9, True, False),  # Hopper GPU with DeepEP enabled - should pass
            (7, True, True),  # Volta GPU with DeepEP enabled - should raise ValueError
            (6, True, True),  # Pascal GPU with DeepEP enabled - should raise ValueError
            (10, True, True),  # Future unsupported GPU with DeepEP enabled - should raise ValueError
            (7, False, False),  # Volta GPU with DeepEP disabled - should pass
            (6, False, False),  # Pascal GPU with DeepEP disabled - should pass
        ],
    )
    @patch("torch.cuda.get_device_properties")
    def test_deepep_validation(
        self, mock_get_device_properties, monkeypatch, gpu_major, moe_enable_deepep, expect_error
    ):
        """Test DeepEP validation during config container validation."""
        # Mock GPU device properties
        mock_properties = MagicMock()
        mock_properties.major = gpu_major
        mock_get_device_properties.return_value = mock_properties

        # Create a GPT model config with MoE settings
        gpt_model_cfg = create_test_gpt_config(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            moe_token_dispatcher_type="flex" if moe_enable_deepep else "alltoall",
            moe_enable_deepep=moe_enable_deepep,
            moe_shared_expert_overlap=not moe_enable_deepep,  # DeepEP requires this to be False
        )

        container, og_ws, cfg_mod = create_test_config_container(world_size_override=1, model_config=gpt_model_cfg)

        try:
            if expect_error:
                with pytest.raises(ValueError, match="DeepEP is supported for Ampere"):
                    container.validate()
            else:
                container.validate()  # Should pass without error
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    @patch("torch.cuda.get_device_properties")
    def test_deepep_validation_disabled_skips_hardware_check(self, mock_get_device_properties, monkeypatch):
        """Test that DeepEP validation is skipped when DeepEP is disabled, even on unsupported hardware."""
        # Mock unsupported GPU (should not be called since DeepEP is disabled)
        mock_properties = MagicMock()
        mock_properties.major = 7  # Volta
        mock_get_device_properties.return_value = mock_properties

        # Create a GPT model config with DeepEP disabled
        gpt_model_cfg = create_test_gpt_config(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            moe_enable_deepep=False,  # DeepEP disabled
        )

        container, og_ws, cfg_mod = create_test_config_container(world_size_override=1, model_config=gpt_model_cfg)

        try:
            # Should pass without error and without calling get_device_properties
            container.validate()
            # Verify get_device_properties was not called since DeepEP is disabled
            mock_get_device_properties.assert_not_called()
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_megatron_fsdp_config(self, monkeypatch):
        """Test MegatronFSDP config."""
        gpt_model_cfg = create_test_gpt_config()
        train_cfg = create_test_training_config(train_iters=500, global_batch_size=16)
        sched_cfg = create_test_scheduler_config()
        dist_cfg = create_test_distributed_init_config(use_megatron_fsdp=True)

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=gpt_model_cfg,
            train_config=train_cfg,
            scheduler_config=sched_cfg,
            dist_config=dist_cfg,
        )
        try:
            container.model.gradient_accumulation_fusion = True
            container.ddp.average_in_collective = True
            container.validate()
            assert container.model.gradient_accumulation_fusion is False
            assert container.ddp.average_in_collective is False
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_megatron_fsdp_config_with_torch_fsdp2(self, monkeypatch):
        """Test MegatronFSDP config with torch_fsdp2, should raise ValueError."""
        gpt_model_cfg = create_test_gpt_config()
        train_cfg = create_test_training_config(train_iters=500, global_batch_size=16)
        sched_cfg = create_test_scheduler_config()
        dist_cfg = create_test_distributed_init_config(use_megatron_fsdp=True, use_torch_fsdp2=True)

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=gpt_model_cfg,
            train_config=train_cfg,
            scheduler_config=sched_cfg,
            dist_config=dist_cfg,
        )
        try:
            with pytest.raises(ValueError):
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
        def gpt_dataset_seqlen_1024():
            return create_test_gpt_dataset_config(1024)

        self._check_post_init_idempotency(gpt_dataset_seqlen_1024)

        # Test rerun of post-init with valid and invalid changes
        cfg = gpt_dataset_seqlen_1024()
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
        cfg = create_test_nvrx_straggler_config(enabled=True)
        cfg.num_gpu_perf_scores_to_print = 2
        cfg.__post_init__()

        with pytest.raises(ValueError, match="report_time_interval must be positive"):
            cfg.report_time_interval = -100.0
            cfg.__post_init__()

    def test_checkpoint_config(self):
        self._check_post_init_idempotency(create_test_checkpoint_config)

        # Test rerun of post-init with valid and invalid changes
        cfg = create_test_checkpoint_config(ckpt_format="torch_dist")
        cfg.save = "/tmp/test_checkpoint_config"
        cfg.__post_init__()

        with pytest.raises(AssertionError, match="load_main_params_from_ckpt must be used with load_optim=False"):
            cfg.load_main_params_from_ckpt = True
            cfg.load_optim = True
            cfg.__post_init__()

    def test_mixed_precision_config(self):
        from megatron.bridge.training.mixed_precision import bf16_with_mxfp8_mixed

        self._check_post_init_idempotency(bf16_with_mxfp8_mixed)
        cfg = bf16_with_mxfp8_mixed()
        cfg.grad_reduce_in_fp32 = False
        cfg.__post_init__()

    def test_rerun_validate_config_container(self):
        import copy
        from dataclasses import fields

        def patched_init_method():
            return torch.nn.init.normal_(mean=0.0, std=0.02)

        gpt_cfg = create_test_gpt_config(init_method=patched_init_method, output_layer_init_method=patched_init_method)
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


class TestCheckpointConfig:
    """Tests for CheckpointConfig class."""

    @pytest.mark.parametrize(
        "load_main_params_from_ckpt, load_optim, expect_assertion_error",
        [
            (True, False, False),  # Valid combination
            (True, True, True),  # Invalid combination - should raise error
            (False, False, False),  # Valid combination
            (False, True, False),  # Valid combination
        ],
    )
    def test_load_main_params_from_ckpt_validation_parametrized(
        self, load_main_params_from_ckpt, load_optim, expect_assertion_error
    ):
        """Parametrized test for load_main_params_from_ckpt validation."""
        if expect_assertion_error:
            with pytest.raises(AssertionError, match="load_main_params_from_ckpt must be used with load_optim=False"):
                create_test_checkpoint_config(
                    load_main_params_from_ckpt=load_main_params_from_ckpt, load_optim=load_optim
                )
        else:
            create_test_checkpoint_config(load_main_params_from_ckpt=load_main_params_from_ckpt, load_optim=load_optim)

    def test_async_save_validation_error(self):
        """Test that async_save requires both a save path and use_persistent_ckpt_worker=True."""
        # Test that async_save requires a save path
        with pytest.raises(
            AssertionError, match="async_save is enabled, but save is not set. Set save to a valid path."
        ):
            create_test_checkpoint_config(async_save=True, save=None)

        # Test that async_save requires use_persistent_ckpt_worker=True
        with pytest.raises(AssertionError, match="async_save requires use_persistent_ckpt_worker=True."):
            create_test_checkpoint_config(
                async_save=True, save="/tmp/test_checkpoint_config", use_persistent_ckpt_worker=False
            )

        # should not raise an error when both conditions are met
        create_test_checkpoint_config(
            async_save=True, save="/tmp/test_checkpoint_config", use_persistent_ckpt_worker=True
        )

    def test_async_save_format_validation_torch_dist(self, monkeypatch):
        """Test that async_save works with torch_dist format."""
        gpt_model_cfg = create_test_gpt_config()
        train_cfg = create_test_training_config(train_iters=500, global_batch_size=16)
        sched_cfg = create_test_scheduler_config()
        ckpt_cfg = create_test_checkpoint_config(
            async_save=True, save="/tmp/test_checkpoint", ckpt_format="torch_dist"
        )

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=gpt_model_cfg,
            train_config=train_cfg,
            scheduler_config=sched_cfg,
            checkpoint_config=ckpt_cfg,
        )
        try:
            # Should not raise error - async_save with torch_dist is allowed
            container.validate()
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_async_save_format_validation_fsdp_dtensor_fails(self, monkeypatch):
        """Test that async_save fails with fsdp_dtensor format."""
        gpt_model_cfg = create_test_gpt_config()
        train_cfg = create_test_training_config(train_iters=500, global_batch_size=16)
        sched_cfg = create_test_scheduler_config()
        ckpt_cfg = create_test_checkpoint_config(
            async_save=True, save="/tmp/test_checkpoint", ckpt_format="fsdp_dtensor"
        )
        # Enable Megatron FSDP so the format validation passes and we reach the async_save check
        dist_cfg = create_test_distributed_init_config(use_megatron_fsdp=True)

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=gpt_model_cfg,
            train_config=train_cfg,
            scheduler_config=sched_cfg,
            checkpoint_config=ckpt_cfg,
            dist_config=dist_cfg,
        )
        try:
            # Should raise error - async_save with fsdp_dtensor is not allowed
            with pytest.raises(AssertionError, match="async_save is only supported with ckpt_format='torch_dist'"):
                container.validate()
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_fsdp_dtensor_format_validation_with_megatron_fsdp(self, monkeypatch):
        """Test that fsdp_dtensor format requires Megatron FSDP."""
        gpt_model_cfg = create_test_gpt_config()
        train_cfg = create_test_training_config(train_iters=500, global_batch_size=16)
        sched_cfg = create_test_scheduler_config()
        ckpt_cfg = create_test_checkpoint_config(save="/tmp/test_checkpoint", ckpt_format="fsdp_dtensor")
        dist_cfg = create_test_distributed_init_config(use_megatron_fsdp=True)

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=gpt_model_cfg,
            train_config=train_cfg,
            scheduler_config=sched_cfg,
            checkpoint_config=ckpt_cfg,
            dist_config=dist_cfg,
        )
        try:
            # Should not raise error - fsdp_dtensor with Megatron FSDP is allowed
            container.validate()
            assert container.checkpoint.ckpt_format == "fsdp_dtensor"
            assert container.dist.use_megatron_fsdp is True
            assert container.ddp.use_megatron_fsdp is True
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_fsdp_dtensor_format_validation_without_megatron_fsdp_fails(self, monkeypatch):
        """Test that fsdp_dtensor format fails without Megatron FSDP."""
        gpt_model_cfg = create_test_gpt_config()
        train_cfg = create_test_training_config(train_iters=500, global_batch_size=16)
        sched_cfg = create_test_scheduler_config()
        ckpt_cfg = create_test_checkpoint_config(save="/tmp/test_checkpoint", ckpt_format="fsdp_dtensor")
        dist_cfg = create_test_distributed_init_config(use_megatron_fsdp=False)

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=gpt_model_cfg,
            train_config=train_cfg,
            scheduler_config=sched_cfg,
            checkpoint_config=ckpt_cfg,
            dist_config=dist_cfg,
        )
        try:
            # Should raise error - fsdp_dtensor without Megatron FSDP is not allowed
            with pytest.raises(AssertionError, match="fsdp_dtensor checkpoint format only supports Megatron FSDP"):
                container.validate()
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_megatron_fsdp_with_precision_aware_optimizer(self, monkeypatch):
        """Test that Megatron FSDP with precision aware optimizer sets preserve_fp32_weights=False."""
        gpt_model_cfg = create_test_gpt_config()
        train_cfg = create_test_training_config(train_iters=500, global_batch_size=16)
        sched_cfg = create_test_scheduler_config()

        # Create optimizer config with precision aware optimizer enabled
        optim_cfg = create_test_optimizer_config()
        optim_cfg.use_precision_aware_optimizer = True
        optim_cfg.use_distributed_optimizer = True  # Required for precision aware optimizer

        dist_cfg = create_test_distributed_init_config(use_megatron_fsdp=True)

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=gpt_model_cfg,
            train_config=train_cfg,
            scheduler_config=sched_cfg,
            optimizer_config=optim_cfg,
            dist_config=dist_cfg,
        )
        try:
            container.validate()
            # Should automatically set preserve_fp32_weights=False when using precision aware optimizer with FSDP
            assert container.ddp.preserve_fp32_weights is False
            assert container.optimizer.use_precision_aware_optimizer is True
            assert container.dist.use_megatron_fsdp is True
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_megatron_fsdp_without_precision_aware_optimizer(self, monkeypatch):
        """Test that Megatron FSDP without precision aware optimizer doesn't modify preserve_fp32_weights."""
        gpt_model_cfg = create_test_gpt_config()
        train_cfg = create_test_training_config(train_iters=500, global_batch_size=16)
        sched_cfg = create_test_scheduler_config()

        # Create optimizer config with precision aware optimizer disabled
        optim_cfg = create_test_optimizer_config()
        optim_cfg.use_precision_aware_optimizer = False
        optim_cfg.use_distributed_optimizer = True  # Enable distributed optimizer for consistency

        dist_cfg = create_test_distributed_init_config(use_megatron_fsdp=True)

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=gpt_model_cfg,
            train_config=train_cfg,
            scheduler_config=sched_cfg,
            optimizer_config=optim_cfg,
            dist_config=dist_cfg,
        )
        try:
            container.validate()
            # preserve_fp32_weights should keep its default value when precision aware optimizer is disabled
            assert container.optimizer.use_precision_aware_optimizer is False
            assert container.dist.use_megatron_fsdp is True
            # preserve_fp32_weights should remain at its default
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)
