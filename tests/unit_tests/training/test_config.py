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

import os
from dataclasses import fields
from typing import Any, Optional, Union
from unittest.mock import MagicMock, patch

import pytest
import torch

from megatron.bridge.data.builders import (
    DirectHFSFTDatasetConfig,
    EnergonDatasetConfig,
    GPTSFTDatasetConfig,
    HFDatasetSourceConfig,
    HFEnergonTaskEncoderConfig,
    MockVLMSFTDatasetConfig,
)
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.mla_provider import MLAModelProvider
from megatron.bridge.models.t5_provider import T5ModelProvider
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DatasetProvider,
    DistributedDataParallelConfig,
    DistributedInitConfig,
    GPTDatasetConfig,
    GPTFIMDatasetConfig,
    LoggerConfig,
    MockGPTDatasetConfig,
    NVRxStragglerDetectionConfig,
    OptimizerConfig,
    ProfilingConfig,
    RerunStateMachineConfig,
    RNGConfig,
    SchedulerConfig,
    TrainingConfig,
    ValidationConfig,
    _validate_and_sync_distributed_optimizer_settings,
    _validate_mixed_precision_consistency,
    apply_environment_variables,
    megatron_mimo_runtime_config_update,
)
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig
from megatron.bridge.training.tokenizers.config import TokenizerConfig
from megatron.bridge.utils.cuda_graph import (
    cuda_graph_module_names,
    set_cuda_graph_modules,
    set_full_iteration_cuda_graph,
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


def create_test_deepseek_config(**kwargs: Any) -> MLAModelProvider:
    """Creates an instance of MLAModelProvider for testing."""
    defaults = {
        "num_layers": 1,
        "hidden_size": 128,
        "num_attention_heads": 4,
        "seq_length": 512,
        "apply_rope_fusion": False,
    }
    defaults.update(kwargs)
    return MLAModelProvider(**defaults)


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
        "micro_batch_size": 1,
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
        seq_length=sequence_length,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
    )


def create_test_gpt_sft_dataset_config(sequence_length: int) -> GPTSFTDatasetConfig:
    """Create a GPTSFTDatasetConfig with defaults for testing."""
    return GPTSFTDatasetConfig(seq_length=sequence_length, dataset_root="/tmp/dataset")


def create_test_direct_hf_sft_dataset_config(sequence_length: int) -> DirectHFSFTDatasetConfig:
    """Create a DirectHFSFTDatasetConfig with defaults for testing."""
    return DirectHFSFTDatasetConfig(
        seq_length=sequence_length,
        source=HFDatasetSourceConfig(path_or_dataset="json"),
    )


def create_test_energon_dataset_config(sequence_length: int, micro_batch_size: int = 1) -> EnergonDatasetConfig:
    """Create a serializable Energon config with generic HF task encoding."""
    return EnergonDatasetConfig(
        path="/tmp/energon",
        seq_length=sequence_length,
        micro_batch_size=micro_batch_size,
        task_encoder=HFEnergonTaskEncoderConfig(hf_processor_path="org/model"),
    )


def create_test_mock_vlm_dataset_config(sequence_length: int) -> MockVLMSFTDatasetConfig:
    """Create a synthetic VLM config with no runtime processor."""
    return MockVLMSFTDatasetConfig(seq_length=sequence_length, hf_processor_path="org/model", num_images=0)


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
        "lazy_mpu_init": False,
    }
    defaults.update(kwargs)
    return DistributedInitConfig(**defaults)


def create_test_ddp_config(**kwargs: Any) -> DistributedDataParallelConfig:
    """Creates an instance of DistributedDataParallelConfig with defaults for testing."""
    return DistributedDataParallelConfig(**kwargs)


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
    dataset_config_override: GPTDatasetConfig | DatasetProvider | None = None,
    logger_config: Optional[LoggerConfig] = None,
    tokenizer_config: Optional[TokenizerConfig] = None,
    checkpoint_config: Optional[CheckpointConfig] = None,
    dist_config: Optional[DistributedInitConfig] = None,
    profiling_config: Optional[ProfilingConfig] = None,
    ddp_config: Optional[DistributedDataParallelConfig] = None,
    validation_config: Optional[ValidationConfig] = None,
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

    final_dataset_config: GPTDatasetConfig | DatasetProvider
    if dataset_config_override:
        final_dataset_config = dataset_config_override
    elif isinstance(model_config, (GPTModelProvider, T5ModelProvider)):  # T5 also uses GPTDataset for these tests
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
        ddp=ddp_config or create_test_ddp_config(),
        rng=RNGConfig(),
        rerun_state_machine=RerunStateMachineConfig(),
        profiling=profiling_config,
        validation=validation_config or ValidationConfig(),
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
    gpt_model_cfg = create_test_gpt_config(
        seq_length=512,
        context_parallel_size=cp_size,
        calculate_per_token_loss=calc_per_token_loss,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
    )

    if dataset_type == "finetuning":
        dataset_cfg = create_test_gpt_sft_dataset_config(sequence_length=512)
    elif dataset_type == "conversation":
        dataset_cfg = create_test_direct_hf_sft_dataset_config(sequence_length=512)
    elif dataset_type == "energon":
        dataset_cfg = create_test_energon_dataset_config(sequence_length=512)
    elif dataset_type == "mock_vlm":
        dataset_cfg = create_test_mock_vlm_dataset_config(sequence_length=512)
    else:
        dataset_cfg = create_test_gpt_dataset_config(sequence_length=512)

    ddp_cfg = DistributedDataParallelConfig(average_in_collective=avg_in_collective)

    container, og_ws, cfg_mod = create_test_config_container(
        world_size_override=cp_size,
        model_config=gpt_model_cfg,
        dataset_config_override=dataset_cfg,
    )
    container.ddp = ddp_cfg
    return container, og_ws, cfg_mod


class TestGPTFIMDatasetConfig:
    """Tests desired behavior for GPTFIMDatasetConfig."""

    def test_initialization(self):
        config = GPTFIMDatasetConfig(
            random_seed=1234,
            seq_length=512,
            fim_rate=0.1,
            fim_no_prefix="test",
            fim_extra_tokens={"middle": "<middle>"},
            fim_split_sample="test sample",
            reset_position_ids=False,
            reset_attention_mask=False,
            eod_mask_loss=False,
        )
        config.finalize()

        # Should be an instance GPTFIMDatasetConfig
        from megatron.core.datasets.blended_megatron_dataset_config import BlendedMegatronDatasetConfig

        assert isinstance(config, GPTFIMDatasetConfig)
        assert isinstance(config, GPTDatasetConfig)
        assert isinstance(config, BlendedMegatronDatasetConfig)

        # Should have all the expected fields from parent class
        assert hasattr(config, "random_seed")
        assert hasattr(config, "seq_length")
        assert hasattr(config, "path_to_cache")

        # Verify have all the expected fields were set proeprly
        assert config.fim_data
        assert config.fim_rate == 0.1
        assert config.fim_no_prefix == "test"
        assert config.fim_split_sample == "test sample"
        assert config.fim_extra_tokens["middle"] == "<middle>"


class TestMockGPTDatasetConfig:
    """Tests desired behavior for MockGPTDatasetConfig."""

    def test_initialization(self):
        """Test that blend and blend_per_split fields are always None in MockGPTDatasetConfig."""
        config = MockGPTDatasetConfig(
            random_seed=1234,
            seq_length=512,
            reset_position_ids=False,
            reset_attention_mask=False,
            eod_mask_loss=False,
        )
        config.finalize()

        # Should be an instance of both MockGPTDatasetConfig and GPTDatasetConfig
        from megatron.core.datasets.blended_megatron_dataset_config import BlendedMegatronDatasetConfig
        from megatron.core.datasets.gpt_dataset import GPTDatasetConfig as MCoreGPTDatasetConfig

        assert isinstance(config, MockGPTDatasetConfig)
        assert isinstance(config, GPTDatasetConfig)
        assert isinstance(config, MCoreGPTDatasetConfig)
        assert isinstance(config, BlendedMegatronDatasetConfig)

        # Should have all the expected fields from parent class
        assert hasattr(config, "random_seed")
        assert hasattr(config, "seq_length")
        assert hasattr(config, "path_to_cache")

        # Verify blend fields are None and cannot be accessed via __dict__
        assert config.blend is None
        assert config.blend_per_split is None
        assert config.mock  # should be set by BlendedMegatronDatasetConfig post-init
        print(config.__dict__)
        assert "blend" not in config.__dict__
        assert "blend_per_split" not in config.__dict__

    def test_cannot_set_blend_fields(self):
        """Test that blend and blend_per_split fields cannot be set during initialization."""
        # These should raise a TypeError because blend and blend_per_split are marked as init=False
        with pytest.raises(TypeError, match="got an unexpected keyword argument 'blend'"):
            MockGPTDatasetConfig(
                random_seed=1234,
                seq_length=512,
                reset_position_ids=False,
                reset_attention_mask=False,
                eod_mask_loss=False,
                blend=(["some", "data", "paths"], None),  # This should fail
            ).finalize()

        with pytest.raises(TypeError, match="got an unexpected keyword argument 'blend_per_split'"):
            MockGPTDatasetConfig(
                random_seed=1234,
                seq_length=512,
                reset_position_ids=False,
                reset_attention_mask=False,
                eod_mask_loss=False,
                blend_per_split=[
                    (["train", "paths"], None),
                    (["valid", "paths"], None),
                    (["test", "paths"], None),
                ],  # This should fail
            ).finalize()

        with pytest.raises(TypeError, match="got an unexpected keyword argument"):
            MockGPTDatasetConfig(
                random_seed=1234,
                seq_length=512,
                reset_position_ids=False,
                reset_attention_mask=False,
                eod_mask_loss=False,
                blend=(["some", "data", "paths"], None),
                blend_per_split=[(["train", "paths"], None), (["valid", "paths"], None), (["test", "paths"], None)],
            ).finalize()


class TestConfigContainerValidation:
    def test_deterministic_mode_disallows_ce_fusion(self, monkeypatch):
        """Test that deterministic mode disallows cross-entropy loss fusion."""
        gpt_model_cfg = create_test_gpt_config(
            deterministic_mode=True,
            cross_entropy_loss_fusion=True,
        )

        # Ensure NCCL_ALGO present but valid, so we fail on CE fusion
        monkeypatch.setenv("NCCL_ALGO", "Tree")

        container, og_ws, cfg_mod = create_test_config_container(world_size_override=1, model_config=gpt_model_cfg)

        try:
            with pytest.raises(AssertionError, match="Cross Entropy Fusion is currently not deterministic"):
                container.validate()
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_deterministic_mode_requires_nccl_algo_and_sets_torch(self, monkeypatch):
        """Test that deterministic mode requires NCCL_ALGO and sets torch.use_deterministic_algorithms."""
        gpt_model_cfg = create_test_gpt_config(
            deterministic_mode=True,
            cross_entropy_loss_fusion=False,
            transformer_impl="transformer_engine",
        )

        container, og_ws, cfg_mod = create_test_config_container(world_size_override=1, model_config=gpt_model_cfg)

        try:
            # Missing NCCL_ALGO
            monkeypatch.delenv("NCCL_ALGO", raising=False)
            with pytest.raises(AssertionError, match="NCCL_ALGO must be one of"):
                container.validate()

            # Invalid NCCL_ALGO
            monkeypatch.setenv("NCCL_ALGO", "AllReduce")
            with pytest.raises(AssertionError, match="NCCL_ALGO must be one of"):
                container.validate()

            # Valid NCCL_ALGO -> should pass and call torch deterministic
            monkeypatch.setenv("NCCL_ALGO", "Ring")

            called = {"det": False}

            def _mock_use_deterministic(flag):
                called["det"] = flag

            with patch.object(torch, "use_deterministic_algorithms", side_effect=_mock_use_deterministic):
                container.validate()
                assert called["det"] is True
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

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
            world_size_override=world_size,
            model_config=gpt_model_cfg,
            validation_config=ValidationConfig(eval_global_batch_size=10, eval_micro_batch_size=1),
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
        """Test `use_cpu_initialization` is True if `lazy_mpu_init` is True."""
        gpt_model_cfg = create_test_gpt_config(use_cpu_initialization=False)
        dist_cfg = create_test_distributed_init_config(lazy_mpu_init=True)

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

        # Case 1: lazy_mpu_init is False
        dist_cfg_lazy_false = create_test_distributed_init_config(lazy_mpu_init=False)
        container1, og1, mod1 = create_test_config_container(
            world_size_override=4, model_config=gpt_model_cfg_true, dist_config=dist_cfg_lazy_false
        )
        try:
            container1.validate()
            assert container1.model.use_cpu_initialization is True
        finally:
            restore_get_world_size_safe(og1, mod1)

        # Case 2: lazy_mpu_init is True
        dist_cfg_lazy_true = create_test_distributed_init_config(lazy_mpu_init=True)
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
        ddp_cfg = create_test_ddp_config(use_distributed_optimizer=True)

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=4,
            model_config=gpt_model_cfg,
            dist_config=dist_cfg,
            optimizer_config=opt_cfg,
            checkpoint_config=chkpt_cfg,
            ddp_config=ddp_cfg,
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

    def test_scheduler_max_steps_preserves_full_run_schedules(self, monkeypatch):
        """A short test run can use the schedules from the full training run."""
        gpt_model_cfg = create_test_gpt_config()
        train_cfg = create_test_training_config(train_iters=1000, global_batch_size=32)
        sched_cfg = create_test_scheduler_config(max_steps=48000)

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1, model_config=gpt_model_cfg, train_config=train_cfg, scheduler_config=sched_cfg
        )
        try:
            container.validate()
            assert container.train.train_iters == 1000
            assert container.scheduler.max_steps == 48000
            assert container.scheduler.lr_decay_iters == 48000
            assert container.scheduler.lr_decay_steps == 48000 * train_cfg.global_batch_size
            assert container.scheduler.wd_incr_steps == 48000 * train_cfg.global_batch_size
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_scheduler_max_steps_rejects_value_shorter_than_training(self, monkeypatch):
        gpt_model_cfg = create_test_gpt_config()
        train_cfg = create_test_training_config(train_iters=1000)
        sched_cfg = create_test_scheduler_config(max_steps=999)

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1, model_config=gpt_model_cfg, train_config=train_cfg, scheduler_config=sched_cfg
        )
        try:
            with pytest.raises(ValueError, match="must be greater than or equal to train.train_iters"):
                container.validate()
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
        train_cfg = create_test_training_config(train_iters=1000, global_batch_size=32)
        lr_warmup_fraction = 0.1
        sched_cfg = create_test_scheduler_config(
            lr_warmup_fraction=lr_warmup_fraction, lr_warmup_iters=0
        )  # lr_decay_iters defaults to train_iters

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1, model_config=gpt_model_cfg, train_config=train_cfg, scheduler_config=sched_cfg
        )
        try:
            container.validate()
            expected_lr_warmup_steps = lr_warmup_fraction * (train_cfg.train_iters * train_cfg.global_batch_size)
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

    def test_scheduler_lr_warmup_steps_capped_when_exceeds_lr_decay_steps(self, monkeypatch):
        """Test lr_warmup_steps is capped to lr_decay_steps - 1 with a warning when it would exceed lr_decay_steps."""
        gpt_model_cfg = create_test_gpt_config()
        # train_iters=10 gives lr_decay_steps=10*32=320; lr_warmup_iters=2000 gives lr_warmup_steps=2000*32=64000
        train_cfg = create_test_training_config(train_iters=10, global_batch_size=32)
        sched_cfg = create_test_scheduler_config(lr_warmup_fraction=None, lr_warmup_iters=2000)

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1, model_config=gpt_model_cfg, train_config=train_cfg, scheduler_config=sched_cfg
        )
        try:
            with pytest.warns(UserWarning, match="capping lr_warmup_steps"):
                container.validate()
            assert container.scheduler.lr_warmup_steps == container.scheduler.lr_decay_steps - 1
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_scheduler_lr_decay_steps_zero_raises_value_error(self, monkeypatch):
        """Test that lr_decay_steps <= 0 raises ValueError."""
        gpt_model_cfg = create_test_gpt_config()
        # train_iters=0 gives lr_decay_steps=0*32=0, which must be rejected
        train_cfg = create_test_training_config(train_iters=0, global_batch_size=32)
        sched_cfg = create_test_scheduler_config(lr_warmup_fraction=None, lr_warmup_iters=0)

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1, model_config=gpt_model_cfg, train_config=train_cfg, scheduler_config=sched_cfg
        )
        try:
            with pytest.raises(ValueError, match="lr_decay_steps must be > 0"):
                container.validate()
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_scheduler_lr_warmup_fraction_and_iters_mutual_exclusivity(self, monkeypatch):
        """Test that lr_warmup_fraction and lr_warmup_iters cannot both be specified."""
        gpt_model_cfg = create_test_gpt_config()
        train_cfg = create_test_training_config(train_iters=1000, global_batch_size=10)
        lr_warmup_fraction = 0.05
        lr_warmup_iters = 50  # This should not be allowed with lr_warmup_fraction
        sched_cfg = create_test_scheduler_config(
            lr_warmup_fraction=lr_warmup_fraction, lr_warmup_iters=lr_warmup_iters
        )
        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1, model_config=gpt_model_cfg, train_config=train_cfg, scheduler_config=sched_cfg
        )
        try:
            # This should fail validation due to mutual exclusivity at scheduler finalize level
            with pytest.raises(AssertionError, match="Cannot specify lr_warmup_fraction=0.05 with lr_warmup_iters=50"):
                container.validate()
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
        """Test ProfilingConfig finalize validation for profiler exclusivity."""

        prof_cfg = create_test_profiling_config(
            use_pytorch_profiler=use_pytorch_profiler, use_nsys_profiler=use_nsys_profiler
        )
        gpt_model_cfg = create_test_gpt_config()
        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1, model_config=gpt_model_cfg, profiling_config=prof_cfg
        )

        try:
            if expect_assertion_error:
                with pytest.raises(AssertionError, match="Exactly one of pytorch or nsys profiler should be enabled"):
                    container.validate()  # Validation error should occur here during finalize
            else:
                container.validate()  # Should pass without error
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    @pytest.mark.parametrize(
        "profile_step_start, profile_step_end, expect_assertion_error, expected_error_match",
        [
            (10, 20, False, None),  # Valid: end > start
            (10, 10, False, None),  # Valid: end == start (single step)
            (0, 5, False, None),  # Valid: start at 0
            (20, 10, True, "profile_step_end .* must be >= profile_step_start"),  # Invalid: end < start
            (-1, 10, True, "profile_step_start must be >= 0"),  # Invalid: start < 0
            (10, -1, True, "profile_step_end must be >= 0"),  # Invalid: end < 0
            (-5, -1, True, "profile_step_start must be >= 0"),  # Invalid: both < 0
        ],
    )
    def test_profiling_config_step_range_validation(
        self, profile_step_start, profile_step_end, expect_assertion_error, expected_error_match
    ):
        """Test ProfilingConfig validation for profile step ranges."""
        prof_cfg = create_test_profiling_config(
            use_pytorch_profiler=True,
            profile_step_start=profile_step_start,
            profile_step_end=profile_step_end,
        )

        if expect_assertion_error:
            with pytest.raises(AssertionError, match=expected_error_match):
                prof_cfg.finalize()
        else:
            prof_cfg.finalize()  # Should pass without error

    def test_packed_sequence_micro_batch_size_validation_error(self, monkeypatch):
        """Test validation error when micro_batch_size > 1 with packed sequences."""
        from megatron.bridge.data.packing import PackedSequenceSpecs

        # Create config with micro_batch_size > 1 and packed sequences
        gpt_model_cfg = create_test_gpt_config()
        train_cfg = create_test_training_config(micro_batch_size=4, global_batch_size=32)

        # Create packed sequence specs with packed_sequence_size > 0
        packed_specs = PackedSequenceSpecs(packed_sequence_size=512)
        dataset_cfg = create_test_gpt_sft_dataset_config(sequence_length=512)
        dataset_cfg.enable_offline_packing = True
        dataset_cfg.offline_packing_specs = packed_specs

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
        from megatron.bridge.data.packing import PackedSequenceSpecs

        # Create config with micro_batch_size = 1 and packed sequences
        gpt_model_cfg = create_test_gpt_config()
        train_cfg = create_test_training_config(micro_batch_size=1, global_batch_size=32)

        # Create packed sequence specs with packed_sequence_size > 0
        packed_specs = PackedSequenceSpecs(packed_sequence_size=512)
        dataset_cfg = create_test_gpt_sft_dataset_config(sequence_length=512)
        dataset_cfg.enable_offline_packing = True
        dataset_cfg.offline_packing_specs = packed_specs

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

    def test_packed_sequence_micro_batch_size_validation_error_for_dataset_provider(self, monkeypatch):
        """Test packed sequence validation for DatasetProvider configs."""
        from dataclasses import dataclass
        from typing import Optional, Tuple

        from megatron.bridge.data.packing import PackedSequenceSpecs
        from megatron.bridge.training.config import DatasetBuildContext, DatasetProvider

        @dataclass
        class PackedDatasetProvider(DatasetProvider):
            seq_length: int = 512
            enable_offline_packing: bool = False
            offline_packing_specs: PackedSequenceSpecs | None = None

            def build_datasets(
                self, context: DatasetBuildContext
            ) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
                return None, None, None

        gpt_model_cfg = create_test_gpt_config()
        train_cfg = create_test_training_config(micro_batch_size=4, global_batch_size=32)
        dataset_cfg = PackedDatasetProvider(
            enable_offline_packing=True,
            offline_packing_specs=PackedSequenceSpecs(packed_sequence_size=512),
        )

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

    def test_packed_sequence_validation_skipped_when_specs_none(self, monkeypatch):
        """Test validation skipped when offline_packing_specs is None."""
        # Create config with micro_batch_size > 1 but no packed sequences
        gpt_model_cfg = create_test_gpt_config()
        train_cfg = create_test_training_config(micro_batch_size=4, global_batch_size=32)
        dataset_cfg = create_test_gpt_sft_dataset_config(sequence_length=512)
        # offline_packing_specs defaults to None

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
        """Test validation skipped when using GPTDatasetConfig instead of GPTSFTDatasetConfig."""
        # Create config with micro_batch_size > 1 and GPTDatasetConfig
        gpt_model_cfg = create_test_gpt_config()
        train_cfg = create_test_training_config(micro_batch_size=4, global_batch_size=32)
        dataset_cfg = create_test_gpt_dataset_config(sequence_length=512)
        # GPTDatasetConfig doesn't have offline_packing_specs

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

    def test_enable_in_batch_packing_requires_micro_batch_size_gt_1(self, monkeypatch):
        """Test validation error when micro_batch_size == 1 with enable_in_batch_packing=True."""
        gpt_model_cfg = create_test_gpt_config()
        train_cfg = create_test_training_config(micro_batch_size=1, global_batch_size=32)
        dataset_cfg = create_test_direct_hf_sft_dataset_config(sequence_length=512)
        dataset_cfg.enable_in_batch_packing = True

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=gpt_model_cfg,
            train_config=train_cfg,
            dataset_config_override=dataset_cfg,
        )
        error_msg = (
            "micro_batch_size should be greater than 1 when using enable_in_batch_packing=True. "
            "In-batch packing concatenates multiple sequences within a microbatch, so at least 2 sequences "
            "are required per micro-batch."
        )
        try:
            with pytest.raises(
                ValueError,
                match=error_msg,
            ):
                container.validate()
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_enable_in_batch_packing_passes_with_micro_batch_size_gt_1(self, monkeypatch):
        """Test validation passes when micro_batch_size > 1 with enable_in_batch_packing=True."""
        gpt_model_cfg = create_test_gpt_config()
        train_cfg = create_test_training_config(micro_batch_size=4, global_batch_size=32)
        dataset_cfg = create_test_direct_hf_sft_dataset_config(sequence_length=512)
        dataset_cfg.enable_in_batch_packing = True

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

    def test_enable_in_batch_packing_sets_collate_padding_multiple(self, monkeypatch):
        """Test in-batch packing forwards CP/SP divisibility requirements to collate-time packers."""
        gpt_model_cfg = create_test_gpt_config(
            context_parallel_size=2,
            tensor_model_parallel_size=4,
            sequence_parallel=True,
            calculate_per_token_loss=True,
        )
        train_cfg = create_test_training_config(micro_batch_size=2, global_batch_size=8)
        dataset_cfg = create_test_direct_hf_sft_dataset_config(sequence_length=512)
        dataset_cfg.enable_in_batch_packing = True

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=8,
            model_config=gpt_model_cfg,
            train_config=train_cfg,
            dataset_config_override=dataset_cfg,
        )
        container.ddp.average_in_collective = False

        try:
            container.validate()
            assert dataset_cfg.in_batch_packing_pad_to_multiple_of == 8
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_direct_hf_in_batch_padding_includes_train_eval_cp_and_sp(self, monkeypatch):
        """Test direct-HF packing reserves one shape for train/eval CP with SP."""
        gpt_model_cfg = create_test_gpt_config(
            context_parallel_size=2,
            tensor_model_parallel_size=2,
            sequence_parallel=True,
            calculate_per_token_loss=True,
        )
        train_cfg = create_test_training_config(micro_batch_size=2, global_batch_size=8)
        dataset_cfg = create_test_direct_hf_sft_dataset_config(sequence_length=512)
        dataset_cfg.enable_in_batch_packing = True

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=8,
            model_config=gpt_model_cfg,
            train_config=train_cfg,
            dataset_config_override=dataset_cfg,
        )
        container.dist.eval_context_parallel_size = 4
        container.ddp.average_in_collective = False

        try:
            container.validate()
            assert dataset_cfg.in_batch_packing_pad_to_multiple_of == 8
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_direct_hf_non_packed_padding_multiple_includes_cp_and_sp_requirements(self, monkeypatch):
        """Test non-packed direct-HF batches are divisible for CP/SP slicing."""
        gpt_model_cfg = create_test_gpt_config(
            context_parallel_size=2,
            tensor_model_parallel_size=4,
            sequence_parallel=True,
            calculate_per_token_loss=True,
        )
        train_cfg = create_test_training_config(micro_batch_size=1, global_batch_size=8)
        dataset_cfg = create_test_direct_hf_sft_dataset_config(sequence_length=512)
        dataset_cfg.pad_to_multiple_of = 3

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=8,
            model_config=gpt_model_cfg,
            train_config=train_cfg,
            dataset_config_override=dataset_cfg,
        )
        container.ddp.average_in_collective = False

        try:
            container.validate()
            assert dataset_cfg.pad_to_multiple_of == 24
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_energon_packing_and_non_packed_padding_include_cp_sp_requirements(self, monkeypatch):
        """Test Energon receives the same CP/SP-safe collate multiples as direct HF."""
        model_cfg = create_test_gpt_config(
            context_parallel_size=2,
            tensor_model_parallel_size=4,
            sequence_parallel=True,
            calculate_per_token_loss=True,
        )
        train_cfg = create_test_training_config(micro_batch_size=2, global_batch_size=8)
        dataset_cfg = create_test_energon_dataset_config(sequence_length=512, micro_batch_size=2)
        dataset_cfg.enable_in_batch_packing = True
        dataset_cfg.defer_in_batch_packing_to_step = True
        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=8,
            model_config=model_cfg,
            train_config=train_cfg,
            dataset_config_override=dataset_cfg,
        )
        container.ddp.average_in_collective = False

        try:
            container.validate()
            assert dataset_cfg.in_batch_packing_pad_to_multiple_of == 8
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

        model_cfg = create_test_gpt_config(
            context_parallel_size=2,
            tensor_model_parallel_size=4,
            sequence_parallel=True,
            calculate_per_token_loss=True,
        )
        train_cfg = create_test_training_config(micro_batch_size=1, global_batch_size=8)
        dataset_cfg = create_test_energon_dataset_config(sequence_length=512)
        dataset_cfg.pad_to_multiple_of = 3
        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=8,
            model_config=model_cfg,
            train_config=train_cfg,
            dataset_config_override=dataset_cfg,
        )
        container.ddp.average_in_collective = False

        try:
            container.validate()
            assert dataset_cfg.pad_to_multiple_of == 24
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_mock_vlm_padding_includes_cp_sp_with_sft_loss_safeguards(self, monkeypatch):
        """Mock conversation data gets safe shapes and valid CP loss reduction."""
        model_cfg = create_test_gpt_config(
            context_parallel_size=2,
            tensor_model_parallel_size=4,
            sequence_parallel=True,
            calculate_per_token_loss=True,
        )
        train_cfg = create_test_training_config(micro_batch_size=2, global_batch_size=8)
        dataset_cfg = create_test_mock_vlm_dataset_config(sequence_length=512)
        dataset_cfg.enable_in_batch_packing = True
        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=8,
            model_config=model_cfg,
            train_config=train_cfg,
            dataset_config_override=dataset_cfg,
        )
        container.ddp.average_in_collective = False

        try:
            container.validate()
            assert dataset_cfg.in_batch_packing_pad_to_multiple_of == 8
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

        model_cfg = create_test_gpt_config(
            context_parallel_size=2,
            tensor_model_parallel_size=4,
            sequence_parallel=True,
            calculate_per_token_loss=True,
        )
        train_cfg = create_test_training_config(micro_batch_size=1, global_batch_size=8)
        dataset_cfg = create_test_mock_vlm_dataset_config(sequence_length=512)
        dataset_cfg.pad_to_multiple_of = 3
        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=8,
            model_config=model_cfg,
            train_config=train_cfg,
            dataset_config_override=dataset_cfg,
        )
        container.ddp.average_in_collective = False

        try:
            container.validate()
            assert dataset_cfg.pad_to_multiple_of == 24
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_direct_hf_preserves_explicit_fixed_length_padding(self, monkeypatch):
        """Test explicit fixed-length padding remains enabled without PP or EP."""
        gpt_model_cfg = create_test_gpt_config(
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=1,
        )
        dataset_cfg = create_test_direct_hf_sft_dataset_config(sequence_length=512)
        dataset_cfg.pad_to_max_length = True

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=gpt_model_cfg,
            dataset_config_override=dataset_cfg,
        )

        try:
            container.validate()
            assert dataset_cfg.pad_to_max_length is True
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_direct_hf_seq_length_must_support_cp_and_sp_collate_slicing(self, monkeypatch):
        """Test the sequence cap cannot undo CP/SP-safe collate padding."""
        gpt_model_cfg = create_test_gpt_config(
            seq_length=20,
            context_parallel_size=2,
            tensor_model_parallel_size=4,
            sequence_parallel=True,
            calculate_per_token_loss=True,
        )
        train_cfg = create_test_training_config(micro_batch_size=1, global_batch_size=8)
        dataset_cfg = create_test_direct_hf_sft_dataset_config(sequence_length=20)

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=8,
            model_config=gpt_model_cfg,
            train_config=train_cfg,
            dataset_config_override=dataset_cfg,
        )
        container.ddp.average_in_collective = False

        try:
            with pytest.raises(ValueError, match="seq_length must be divisible by the CP/SP collate padding"):
                container.validate()
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_direct_hf_padding_multiple_is_validated_before_runtime_derivation(self, monkeypatch):
        """Test runtime LCM derivation cannot normalize an invalid declarative value."""
        gpt_model_cfg = create_test_gpt_config(
            context_parallel_size=2,
            calculate_per_token_loss=True,
        )
        train_cfg = create_test_training_config(micro_batch_size=1, global_batch_size=2)
        dataset_cfg = create_test_direct_hf_sft_dataset_config(sequence_length=512)
        dataset_cfg.pad_to_multiple_of = -3

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=2,
            model_config=gpt_model_cfg,
            train_config=train_cfg,
            dataset_config_override=dataset_cfg,
        )
        container.ddp.average_in_collective = False

        try:
            with pytest.raises(ValueError, match="pad_to_multiple_of must be greater than 0"):
                container.validate()
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_direct_hf_padding_multiple_includes_eval_context_parallel_size(self, monkeypatch):
        """Test validation batches remain sliceable with a different eval CP degree."""
        gpt_model_cfg = create_test_gpt_config(seq_length=24)
        train_cfg = create_test_training_config(micro_batch_size=1, global_batch_size=2)
        dataset_cfg = create_test_direct_hf_sft_dataset_config(sequence_length=24)
        dataset_cfg.pad_to_multiple_of = 3

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=2,
            model_config=gpt_model_cfg,
            train_config=train_cfg,
            dataset_config_override=dataset_cfg,
        )
        container.dist.eval_context_parallel_size = 2

        try:
            container.validate()
            assert dataset_cfg.pad_to_multiple_of == 12
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_enable_offline_packing_requires_specs(self, monkeypatch):
        """Test validation error when offline packing is enabled without specs."""
        gpt_model_cfg = create_test_gpt_config()
        train_cfg = create_test_training_config(micro_batch_size=1, global_batch_size=32)
        dataset_cfg = create_test_gpt_sft_dataset_config(sequence_length=512)
        dataset_cfg.enable_offline_packing = True
        dataset_cfg.offline_packing_specs = None

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=gpt_model_cfg,
            train_config=train_cfg,
            dataset_config_override=dataset_cfg,
        )

        try:
            with pytest.raises(ValueError, match="offline_packing_specs must be set"):
                container.validate()
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_offline_packing_specs_require_enable_offline_packing(self, monkeypatch):
        """Test validation error when offline specs are set without enabling offline packing."""
        from megatron.bridge.data.packing import PackedSequenceSpecs

        gpt_model_cfg = create_test_gpt_config()
        train_cfg = create_test_training_config(micro_batch_size=1, global_batch_size=32)
        dataset_cfg = create_test_gpt_sft_dataset_config(sequence_length=512)
        dataset_cfg.offline_packing_specs = PackedSequenceSpecs(packed_sequence_size=512)

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=gpt_model_cfg,
            train_config=train_cfg,
            dataset_config_override=dataset_cfg,
        )

        try:
            with pytest.raises(ValueError, match="enable_offline_packing must be True"):
                container.validate()
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_offline_and_in_batch_packing_are_mutually_exclusive(self, monkeypatch):
        """Test validation error when both packing modes are enabled."""
        from megatron.bridge.data.packing import PackedSequenceSpecs

        gpt_model_cfg = create_test_gpt_config()
        train_cfg = create_test_training_config(micro_batch_size=4, global_batch_size=32)
        dataset_cfg = create_test_gpt_sft_dataset_config(sequence_length=512)
        dataset_cfg.enable_offline_packing = True
        dataset_cfg.offline_packing_specs = PackedSequenceSpecs(packed_sequence_size=512)
        dataset_cfg.enable_in_batch_packing = True

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=gpt_model_cfg,
            train_config=train_cfg,
            dataset_config_override=dataset_cfg,
        )

        try:
            with pytest.raises(
                ValueError,
                match="enable_offline_packing and enable_in_batch_packing are mutually exclusive",
            ):
                container.validate()
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
            # GPTSFTDatasetConfig with CP > 1 - both checks should trigger
            ("finetuning", 2, False, False, True, "calculate_per_token_loss must be True"),
            ("finetuning", 2, True, True, True, "average_in_collective must be False"),
            ("finetuning", 2, True, False, False, None),  # Valid case
            # Direct HF conversation SFT uses the same CP loss-reduction safeguards.
            ("conversation", 2, False, False, True, "calculate_per_token_loss must be True"),
            ("conversation", 2, True, True, True, "average_in_collective must be False"),
            ("conversation", 2, True, False, False, None),
            # Energon multimodal SFT uses the same CP loss-reduction safeguards.
            ("energon", 2, False, False, True, "calculate_per_token_loss must be True"),
            ("energon", 2, True, True, True, "average_in_collective must be False"),
            ("energon", 2, True, False, False, None),
            # Synthetic VLM conversations have the same masked SFT loss semantics.
            ("mock_vlm", 2, False, False, True, "calculate_per_token_loss must be True"),
            ("mock_vlm", 2, True, True, True, "average_in_collective must be False"),
            ("mock_vlm", 2, True, False, False, None),
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
        "gpu_major, gpu_name, moe_enable_deepep, expect_error",
        [
            (8, "NVIDIA A100", True, False),  # Ampere GPU with DeepEP enabled - should pass
            (9, "NVIDIA H100", True, False),  # Hopper GPU with DeepEP enabled - should pass
            (10, "NVIDIA B200", True, False),  # Blackwell B200 GPU with DeepEP enabled - should pass
            (10, "NVIDIA B200 SXM6 AC", True, False),  # Blackwell B200 variant with DeepEP enabled - should pass
            (10, "NVIDIA B300", True, False),  # Blackwell B300 GPU with DeepEP enabled - should pass
            (7, "NVIDIA V100", True, True),  # Volta GPU with DeepEP enabled - should raise ValueError
            (6, "NVIDIA P100", True, True),  # Pascal GPU with DeepEP enabled - should raise ValueError
            (
                10,
                "NVIDIA B100",
                True,
                True,
            ),  # Unsupported Blackwell variant with DeepEP enabled - should raise ValueError
            (7, "NVIDIA V100", False, False),  # Volta GPU with DeepEP disabled - should pass
            (6, "NVIDIA P100", False, False),  # Pascal GPU with DeepEP disabled - should pass
        ],
    )
    @patch("torch.cuda.get_device_properties")
    def test_deepep_validation(
        self, mock_get_device_properties, monkeypatch, gpu_major, gpu_name, moe_enable_deepep, expect_error
    ):
        """Test DeepEP validation during config container validation."""
        # Mock GPU device properties
        mock_properties = MagicMock()
        mock_properties.major = gpu_major
        mock_properties.name = gpu_name
        mock_get_device_properties.return_value = mock_properties

        # Create a GPT model config with MoE settings
        gpt_model_cfg = create_test_gpt_config(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            moe_flex_dispatcher_backend="deepep" if moe_enable_deepep else None,
            moe_token_dispatcher_type="flex" if moe_enable_deepep else "alltoall",
            moe_shared_expert_overlap=not moe_enable_deepep,  # DeepEP requires this to be False
        )

        container, og_ws, cfg_mod = create_test_config_container(world_size_override=1, model_config=gpt_model_cfg)

        try:
            if expect_error:
                with pytest.raises(ValueError, match="DeepEP is supported for Ampere, Hopper, and Blackwell"):
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
            moe_flex_dispatcher_backend=None,  # DeepEP disabled
            moe_token_dispatcher_type="alltoall",  # Disable flex dispatcher
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
            container.ddp.average_in_collective = True
            container.validate()
            assert container.ddp.average_in_collective is False
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_megatron_fsdp_forces_reuse_grad_buf_false(self, monkeypatch):
        """Test that Megatron FSDP forces reuse_grad_buf_for_mxfp8_param_ag=False on ddp and optimizer."""
        gpt_model_cfg = create_test_gpt_config()
        train_cfg = create_test_training_config(train_iters=500, global_batch_size=16)
        sched_cfg = create_test_scheduler_config()
        dist_cfg = create_test_distributed_init_config(use_megatron_fsdp=True)
        # Create optimizer config with reuse_grad_buf_for_mxfp8_param_ag=True
        optimizer_cfg = create_test_optimizer_config(reuse_grad_buf_for_mxfp8_param_ag=True)
        # Create ddp config with reuse_grad_buf_for_mxfp8_param_ag=True
        # fp8_param_gather=True is required for reuse_grad_buf in DDP config validation
        ddp_cfg = create_test_ddp_config(
            use_megatron_fsdp=True, reuse_grad_buf_for_mxfp8_param_ag=True, fp8_param_gather=True
        )

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=gpt_model_cfg,
            train_config=train_cfg,
            scheduler_config=sched_cfg,
            dist_config=dist_cfg,
            optimizer_config=optimizer_cfg,
            ddp_config=ddp_cfg,
        )
        try:
            # Verify the values are True before validation
            assert container.ddp.reuse_grad_buf_for_mxfp8_param_ag is True
            assert container.optimizer.reuse_grad_buf_for_mxfp8_param_ag is True

            container.validate()

            # After validation, both should be forced to False due to FSDP
            assert container.ddp.reuse_grad_buf_for_mxfp8_param_ag is False
            assert container.optimizer.reuse_grad_buf_for_mxfp8_param_ag is False
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_reuse_grad_buf_for_mxfp8_param_ag_required_without_fsdp(self, monkeypatch):
        """Test MXFP8 parameter gather requirements when FSDP is disabled."""
        gpt_model_cfg = create_test_gpt_config()
        train_cfg = create_test_training_config(train_iters=500, global_batch_size=16)
        sched_cfg = create_test_scheduler_config()

        # Case 1: Should raise when reuse_grad_buf_for_mxfp8_param_ag=False
        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=gpt_model_cfg,
            train_config=train_cfg,
            scheduler_config=sched_cfg,
        )
        try:
            container.mixed_precision = MixedPrecisionConfig(
                fp8_param_gather=True, fp8_recipe="mxfp8", reuse_grad_buf_for_mxfp8_param_ag=False
            )
            with pytest.raises(AssertionError, match="reuse_grad_buf_for_mxfp8_param_ag must be set to True"):
                container.validate()
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

        # Case 2: Should raise when reuse_grad_buf_for_mxfp8_param_ag=True but overlap_param_gather=False
        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=gpt_model_cfg,
            train_config=train_cfg,
            scheduler_config=sched_cfg,
            ddp_config=create_test_ddp_config(overlap_param_gather=False),
        )
        try:
            container.mixed_precision = MixedPrecisionConfig(
                fp8_param_gather=True, fp8_recipe="mxfp8", reuse_grad_buf_for_mxfp8_param_ag=True
            )
            with pytest.raises(AssertionError, match="ddp.overlap_param_gather must be set to True"):
                container.validate()
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

        # Case 3: Should pass when both reuse_grad_buf_for_mxfp8_param_ag and overlap_param_gather are True
        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=gpt_model_cfg,
            train_config=train_cfg,
            scheduler_config=sched_cfg,
            ddp_config=create_test_ddp_config(overlap_param_gather=True),
        )
        try:
            container.mixed_precision = MixedPrecisionConfig(
                fp8_param_gather=True, fp8_recipe="mxfp8", reuse_grad_buf_for_mxfp8_param_ag=True
            )
            container.validate()
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

        # Case 4: Should pass when fp8_param_gather=False (guard skips)
        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=gpt_model_cfg,
            train_config=train_cfg,
            scheduler_config=sched_cfg,
        )
        try:
            container.mixed_precision = MixedPrecisionConfig(
                fp8_param_gather=False, fp8_recipe="mxfp8", reuse_grad_buf_for_mxfp8_param_ag=False
            )
            container.validate()
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

        # Case 5: Should pass when fp8_recipe is not mxfp8 (guard skips)
        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=gpt_model_cfg,
            train_config=train_cfg,
            scheduler_config=sched_cfg,
        )
        try:
            container.mixed_precision = MixedPrecisionConfig(
                fp8_param_gather=True, fp8_recipe="delayed", reuse_grad_buf_for_mxfp8_param_ag=False
            )
            container.validate()
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

    def test_megatron_fsdp_config_with_dp_last_dim(self, monkeypatch):
        """Test MegatronFSDP config with use_tp_pp_dp_mapping, should raise ValueError."""
        gpt_model_cfg = create_test_gpt_config()
        train_cfg = create_test_training_config(train_iters=500, global_batch_size=16)
        sched_cfg = create_test_scheduler_config()
        dist_cfg = create_test_distributed_init_config(use_megatron_fsdp=True, use_tp_pp_dp_mapping=True)

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=gpt_model_cfg,
            train_config=train_cfg,
            scheduler_config=sched_cfg,
            dist_config=dist_cfg,
        )
        try:
            with pytest.raises(AssertionError):
                container.validate()
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_cuda_graph_full_iteration_requires_check_for_nan_disabled(self, monkeypatch):
        """Test that full_iteration CUDA graph requires check_for_nan_in_loss=False."""
        gpt_model_cfg = create_test_gpt_config(
            use_te_rng_tracker=True,
        )
        set_full_iteration_cuda_graph(gpt_model_cfg)

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=gpt_model_cfg,
        )

        try:
            # Default check_for_nan_in_loss is True - should fail validation
            assert container.rerun_state_machine.check_for_nan_in_loss is True
            with pytest.raises(
                AssertionError,
                match="check_for_nan_in_loss must be disabled when using full_iteration CUDA graph",
            ):
                container.validate()

            # Setting check_for_nan_in_loss=False should pass validation
            container.rerun_state_machine.check_for_nan_in_loss = False
            container.validate()  # Should pass without error
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_cuda_graph_non_full_iteration_allows_check_for_nan(self, monkeypatch):
        """Test that non-full_iteration CUDA graph allows check_for_nan_in_loss=True."""
        gpt_model_cfg = create_test_gpt_config(
            cuda_graph_impl="transformer_engine",
            use_te_rng_tracker=True,
        )
        set_cuda_graph_modules(gpt_model_cfg, ["attn", "mlp"])

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=gpt_model_cfg,
        )

        try:
            # check_for_nan_in_loss=True should be allowed
            assert container.rerun_state_machine.check_for_nan_in_loss is True
            container.validate()  # Should pass without error
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    @pytest.mark.parametrize("inference_scope", ["layer", "block"])
    def test_cuda_graph_local_inference_scope_allows_validation(self, inference_scope, monkeypatch):
        """Test that MCore local inference graphs are not rejected as training scopes."""
        gpt_model_cfg = create_test_gpt_config(
            cuda_graph_impl="local",
            inference_cuda_graph_scope=inference_scope,
            use_te_rng_tracker=True,
        )

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=gpt_model_cfg,
        )

        try:
            container.validate()
            assert container.model.cuda_graph_impl == "local"
            assert container.model.inference_cuda_graph_scope.name == inference_scope
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_cuda_graph_local_inference_scope_rejects_training_modules(self, monkeypatch):
        """Test that an inference scope does not bypass local training-scope validation."""
        gpt_model_cfg = create_test_gpt_config(
            cuda_graph_impl="local",
            inference_cuda_graph_scope="block",
            use_te_rng_tracker=True,
        )
        set_cuda_graph_modules(gpt_model_cfg, ["mlp"])

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=gpt_model_cfg,
        )

        try:
            with pytest.raises(ValueError, match='cuda_graph_impl="local"'):
                container.validate()
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    @pytest.mark.parametrize("graph_modules", [["mlp"], ["moe_router"]])
    def test_cuda_graph_local_scoped_modules_raise_clear_error(self, graph_modules, monkeypatch):
        """Test that Bridge rejects local scoped graphs before MCore layer construction."""
        gpt_model_cfg = create_test_gpt_config(
            cuda_graph_impl="local",
            use_te_rng_tracker=True,
        )
        set_cuda_graph_modules(gpt_model_cfg, graph_modules)

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=gpt_model_cfg,
        )

        try:
            with pytest.raises(
                ValueError,
                match='cuda_graph_impl="local".*cuda_graph_impl="transformer_engine"',
            ):
                container.validate()
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_cuda_graph_local_deprecated_scope_raise_clear_error(self, monkeypatch):
        """Test post-construction cuda_graph_scope overrides fail clearly for local scoped graphs."""
        gpt_model_cfg = create_test_gpt_config(use_te_rng_tracker=True)
        gpt_model_cfg.cuda_graph_impl = "local"
        gpt_model_cfg.cuda_graph_scope = ["mlp"]

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=gpt_model_cfg,
        )

        try:
            with pytest.raises(
                ValueError,
                match='cuda_graph_impl="local".*cuda_graph_impl="transformer_engine"',
            ):
                container.validate()
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_cuda_graph_local_deprecated_scope_direct_provider_raise_clear_error(self):
        """Test direct provider construction validates local scoped graphs before MCore build."""
        gpt_model_cfg = create_test_gpt_config(use_te_rng_tracker=True, vocab_size=128)
        gpt_model_cfg.cuda_graph_impl = "local"
        gpt_model_cfg.cuda_graph_scope = ["mlp"]

        with pytest.raises(
            ValueError,
            match='cuda_graph_impl="local".*cuda_graph_impl="transformer_engine"',
        ):
            gpt_model_cfg.provide()

    def test_cuda_graph_local_full_iteration_module_allows_validation(self, monkeypatch):
        """Test local full_iteration compatibility input is not treated as scoped local graphs."""
        gpt_model_cfg = create_test_gpt_config(
            cuda_graph_impl="local",
            use_te_rng_tracker=True,
        )
        gpt_model_cfg.cuda_graph_modules = "full_iteration"

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=gpt_model_cfg,
        )

        try:
            container.rerun_state_machine.check_for_nan_in_loss = False
            container.validate()
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_cuda_graph_impl_none_clears_modules(self, monkeypatch):
        """Test that cuda_graph_impl=none clears module-scoped CUDA graph settings."""
        gpt_model_cfg = create_test_gpt_config(
            cuda_graph_impl="none",
            use_te_rng_tracker=True,
        )
        set_cuda_graph_modules(gpt_model_cfg, ["attn", "mlp"])

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=gpt_model_cfg,
        )

        try:
            container.validate()
            assert cuda_graph_module_names(container.model) == []
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_cuda_graph_impl_none_clears_deprecated_full_iteration_scope(self, monkeypatch):
        """Test cuda_graph_impl=none ignores deprecated full_iteration scope overrides."""
        gpt_model_cfg = create_test_gpt_config(
            cuda_graph_impl="none",
            use_te_rng_tracker=True,
        )
        gpt_model_cfg.cuda_graph_scope = ["full_iteration"]

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=gpt_model_cfg,
        )

        try:
            container.validate()
            assert container.model.cuda_graph_impl == "none"
            assert cuda_graph_module_names(container.model) == []
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    @pytest.mark.parametrize("model_factory", [create_test_gpt_config, create_test_deepseek_config])
    def test_default_pipeline_dtype(self, model_factory, monkeypatch):
        """
        Test pipeline_dtype is automatically set if None and PP enabled.
        Test for both GPT and Deepseek to test both TransformerConfig types.
        """

        gpt_model_cfg1 = model_factory(params_dtype=torch.bfloat16, pipeline_model_parallel_size=2)

        container1, og_ws, cfg_mod = create_test_config_container(
            world_size_override=2,
            model_config=gpt_model_cfg1,
        )

        try:
            container1.validate()
            assert container1.model.pipeline_dtype == torch.bfloat16
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

        # Do not change if already set
        gpt_model_cfg2 = model_factory(
            params_dtype=torch.bfloat16, pipeline_dtype=torch.float32, pipeline_model_parallel_size=2
        )

        container2, og_ws, cfg_mod = create_test_config_container(
            world_size_override=2,
            model_config=gpt_model_cfg2,
        )

        try:
            container2.validate()
            assert container2.model.pipeline_dtype == torch.float32
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

        # Do not change if no PP
        gpt_model_cfg3 = model_factory(params_dtype=torch.bfloat16, pipeline_model_parallel_size=1)

        container3, og_ws, cfg_mod = create_test_config_container(
            world_size_override=2,
            model_config=gpt_model_cfg3,
        )

        try:
            container3.validate()
            assert container3.model.pipeline_dtype is None
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_modelopt_with_gradient_accumulation_fusion_fails(self, monkeypatch):
        """Test that restore_modelopt_state with gradient_accumulation_fusion raises AssertionError."""
        gpt_model_cfg = create_test_gpt_config(
            gradient_accumulation_fusion=True,
            restore_modelopt_state=True,
        )
        train_cfg = create_test_training_config(train_iters=500, global_batch_size=16)
        sched_cfg = create_test_scheduler_config()

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=gpt_model_cfg,
            train_config=train_cfg,
            scheduler_config=sched_cfg,
        )
        try:
            with pytest.raises(
                AssertionError,
                match="Gradient accumulation fusion is not supported with ModelOpt/Quantized models",
            ):
                container.validate()
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_modelopt_without_gradient_accumulation_fusion_passes(self, monkeypatch):
        """Test that restore_modelopt_state without gradient_accumulation_fusion passes validation."""
        gpt_model_cfg = create_test_gpt_config(
            gradient_accumulation_fusion=False,
            restore_modelopt_state=True,
        )
        train_cfg = create_test_training_config(train_iters=500, global_batch_size=16)
        sched_cfg = create_test_scheduler_config()

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=gpt_model_cfg,
            train_config=train_cfg,
            scheduler_config=sched_cfg,
        )
        try:
            container.validate()  # Should pass without error
            assert container.model.restore_modelopt_state is True
            assert container.model.gradient_accumulation_fusion is False
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_modelopt_requires_no_gradient_accumulation_fusion(self, monkeypatch):
        """Test that restore_modelopt_state requires gradient_accumulation_fusion to be explicitly set to False."""
        # When restore_modelopt_state=True but gradient_accumulation_fusion is not set (defaults to True),
        # validation should fail
        gpt_model_cfg = create_test_gpt_config(restore_modelopt_state=True)
        # Don't explicitly set gradient_accumulation_fusion - let it use default (which is True)
        train_cfg = create_test_training_config(train_iters=500, global_batch_size=16)
        sched_cfg = create_test_scheduler_config()

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=gpt_model_cfg,
            train_config=train_cfg,
            scheduler_config=sched_cfg,
        )
        try:
            # Should fail because gradient_accumulation_fusion defaults to True
            with pytest.raises(
                AssertionError,
                match="Gradient accumulation fusion is not supported with ModelOpt/Quantized models",
            ):
                container.validate()
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    @patch("megatron.core.utils.is_te_min_version")
    def test_fine_grained_activation_offloading_requires_transformer_engine(self, mock_is_te_min_version, monkeypatch):
        """Test that fine_grained_activation_offloading requires transformer_engine implementation."""
        mock_is_te_min_version.return_value = False  # Pretend TE < 2.10.0

        gpt_model_cfg = create_test_gpt_config(
            fine_grained_activation_offloading=True,
            offload_modules=["attn_norm"],  # Required when fine_grained_activation_offloading=True
            transformer_impl="local",  # Using local instead of transformer_engine
        )
        container, og_ws, cfg_mod = create_test_config_container(world_size_override=1, model_config=gpt_model_cfg)

        try:
            with pytest.raises(
                ValueError,
                match="Fine-grained activation offloading is only supported with transformer_engine implementation",
            ):
                container.validate()
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    @patch("megatron.core.utils.is_te_min_version")
    def test_fine_grained_activation_offloading_with_transformer_engine_passes(
        self, mock_is_te_min_version, monkeypatch
    ):
        """Test that fine_grained_activation_offloading passes with transformer_engine implementation."""
        mock_is_te_min_version.return_value = False  # Pretend TE < 2.10.0 to skip env var check

        gpt_model_cfg = create_test_gpt_config(
            fine_grained_activation_offloading=True,
            offload_modules=["attn_norm"],  # Required when fine_grained_activation_offloading=True
            transformer_impl="transformer_engine",
        )
        container, og_ws, cfg_mod = create_test_config_container(world_size_override=1, model_config=gpt_model_cfg)

        try:
            container.validate()  # Should pass without error
            assert container.model.fine_grained_activation_offloading is True
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    @patch.dict("os.environ", {"NVTE_CPU_OFFLOAD_V1": "0"})
    @patch("megatron.core.utils.is_te_min_version")
    def test_fine_grained_activation_offloading_te_2_10_requires_env_var(self, mock_is_te_min_version, monkeypatch):
        """Test that fine_grained_activation_offloading with TE >= 2.10.0 requires NVTE_CPU_OFFLOAD_V1=1."""
        mock_is_te_min_version.return_value = True  # Pretend TE >= 2.10.0

        gpt_model_cfg = create_test_gpt_config(
            fine_grained_activation_offloading=True,
            offload_modules=["attn_norm"],  # Required when fine_grained_activation_offloading=True
            transformer_impl="transformer_engine",
        )
        container, og_ws, cfg_mod = create_test_config_container(world_size_override=1, model_config=gpt_model_cfg)

        try:
            with pytest.raises(
                ValueError,
                match="NVTE_CPU_OFFLOAD_V1 environment variable should be set to 1",
            ):
                container.validate()
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    @patch.dict("os.environ", {"NVTE_CPU_OFFLOAD_V1": "1"})
    @patch("megatron.core.utils.is_te_min_version")
    def test_fine_grained_activation_offloading_te_2_10_with_env_var_passes(self, mock_is_te_min_version, monkeypatch):
        """Test that fine_grained_activation_offloading with TE >= 2.10.0 and NVTE_CPU_OFFLOAD_V1=1 passes."""
        mock_is_te_min_version.return_value = True  # Pretend TE >= 2.10.0

        gpt_model_cfg = create_test_gpt_config(
            fine_grained_activation_offloading=True,
            offload_modules=["attn_norm"],  # Required when fine_grained_activation_offloading=True
            transformer_impl="transformer_engine",
        )
        container, og_ws, cfg_mod = create_test_config_container(world_size_override=1, model_config=gpt_model_cfg)

        try:
            container.validate()  # Should pass without error
            assert container.model.fine_grained_activation_offloading is True
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_fine_grained_activation_offloading_disabled_skips_validation(self, monkeypatch):
        """Test that validation is skipped when fine_grained_activation_offloading is disabled."""
        gpt_model_cfg = create_test_gpt_config(
            fine_grained_activation_offloading=False,
            transformer_impl="local",  # Would fail if validation was run
        )
        container, og_ws, cfg_mod = create_test_config_container(world_size_override=1, model_config=gpt_model_cfg)

        try:
            container.validate()  # Should pass without error since offloading is disabled
            assert container.model.fine_grained_activation_offloading is False
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)


class TestEvalBatchSizeConfig:
    """Tests for eval batch size default resolution and validation in ConfigContainer.validate()."""

    def test_eval_batch_sizes_default_to_training_values(self, monkeypatch):
        """When eval batch sizes are not set, they should be resolved from training config."""
        gpt_model_cfg = create_test_gpt_config()
        train_cfg = create_test_training_config(global_batch_size=64, micro_batch_size=4)
        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=8, model_config=gpt_model_cfg, train_config=train_cfg
        )
        try:
            container.validate()
            assert container.validation.eval_global_batch_size == 64
            assert container.validation.eval_micro_batch_size == 4
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_eval_batch_sizes_explicit_override(self, monkeypatch):
        """When eval batch sizes are explicitly set, they should not be overridden."""
        gpt_model_cfg = create_test_gpt_config()
        train_cfg = create_test_training_config(global_batch_size=64, micro_batch_size=4)
        val_cfg = ValidationConfig(eval_global_batch_size=16, eval_micro_batch_size=2)
        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=8,
            model_config=gpt_model_cfg,
            train_config=train_cfg,
            validation_config=val_cfg,
        )
        try:
            container.validate()
            assert container.validation.eval_global_batch_size == 16
            assert container.validation.eval_micro_batch_size == 2
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_eval_global_batch_size_partial_override(self, monkeypatch):
        """When only eval_global_batch_size is set, eval_micro_batch_size defaults to training."""
        gpt_model_cfg = create_test_gpt_config()
        train_cfg = create_test_training_config(global_batch_size=64, micro_batch_size=4)
        val_cfg = ValidationConfig(eval_global_batch_size=32)
        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=8,
            model_config=gpt_model_cfg,
            train_config=train_cfg,
            validation_config=val_cfg,
        )
        try:
            container.validate()
            assert container.validation.eval_global_batch_size == 32
            assert container.validation.eval_micro_batch_size == 4
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_eval_micro_batch_size_partial_override(self, monkeypatch):
        """When only eval_micro_batch_size is set, eval_global_batch_size defaults to training."""
        gpt_model_cfg = create_test_gpt_config()
        train_cfg = create_test_training_config(global_batch_size=64, micro_batch_size=4)
        val_cfg = ValidationConfig(eval_micro_batch_size=2)
        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=8,
            model_config=gpt_model_cfg,
            train_config=train_cfg,
            validation_config=val_cfg,
        )
        try:
            container.validate()
            assert container.validation.eval_global_batch_size == 64
            assert container.validation.eval_micro_batch_size == 2
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_energon_micro_batch_size_must_match_train_and_validation(self, monkeypatch):
        """Energon external loaders use one physical micro-batch size for both splits."""
        dataset_cfg = create_test_energon_dataset_config(sequence_length=512, micro_batch_size=2)
        train_cfg = create_test_training_config(global_batch_size=8, micro_batch_size=1)
        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=create_test_gpt_config(),
            train_config=train_cfg,
            dataset_config_override=dataset_cfg,
        )
        try:
            with pytest.raises(ValueError, match="must match train.micro_batch_size"):
                container.validate()
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

        dataset_cfg = create_test_energon_dataset_config(sequence_length=512, micro_batch_size=1)
        dataset_cfg.do_validation = False
        train_cfg = create_test_training_config(global_batch_size=8, micro_batch_size=1)
        validation_cfg = ValidationConfig(eval_global_batch_size=8, eval_micro_batch_size=2)
        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=create_test_gpt_config(),
            train_config=train_cfg,
            validation_config=validation_cfg,
            dataset_config_override=dataset_cfg,
        )
        try:
            container.validate()
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

        dataset_cfg = create_test_energon_dataset_config(sequence_length=512, micro_batch_size=1)
        train_cfg = create_test_training_config(global_batch_size=8, micro_batch_size=1)
        validation_cfg = ValidationConfig(eval_global_batch_size=8, eval_micro_batch_size=2)
        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=create_test_gpt_config(),
            train_config=train_cfg,
            validation_config=validation_cfg,
            dataset_config_override=dataset_cfg,
        )
        try:
            with pytest.raises(ValueError, match="must match validation.eval_micro_batch_size"):
                container.validate()
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_eval_batch_size_divisibility_check_passes(self, monkeypatch):
        """Eval GBS divisible by (eval_MBS * DP) should pass validation."""
        gpt_model_cfg = create_test_gpt_config()
        train_cfg = create_test_training_config(global_batch_size=64, micro_batch_size=4)
        # world_size=8, TP=1, PP=1 => DP=8; eval_GBS=16, eval_MBS=2 => 16 / (2*8) = 1 OK
        val_cfg = ValidationConfig(eval_global_batch_size=16, eval_micro_batch_size=2)
        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=8,
            model_config=gpt_model_cfg,
            train_config=train_cfg,
            validation_config=val_cfg,
        )
        try:
            container.validate()
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_eval_batch_size_divisibility_check_fails(self, monkeypatch):
        """Eval GBS not divisible by (eval_MBS * DP) should fail validation."""
        gpt_model_cfg = create_test_gpt_config()
        train_cfg = create_test_training_config(global_batch_size=64, micro_batch_size=4)
        # world_size=8, TP=1, PP=1 => DP=8; eval_GBS=10, eval_MBS=2 => 10 / (2*8) not integer
        val_cfg = ValidationConfig(eval_global_batch_size=10, eval_micro_batch_size=2)
        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=8,
            model_config=gpt_model_cfg,
            train_config=train_cfg,
            validation_config=val_cfg,
        )
        try:
            with pytest.raises(AssertionError, match="eval_global_batch_size.*must be divisible by"):
                container.validate()
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_eval_batch_size_divisibility_uses_eval_data_parallel_size(self, monkeypatch):
        """Eval-time CP changes the DP degree that owns validation batches."""
        gpt_model_cfg = create_test_gpt_config()
        train_cfg = create_test_training_config(global_batch_size=8, micro_batch_size=1)
        val_cfg = ValidationConfig(eval_global_batch_size=2, eval_micro_batch_size=1)
        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=4,
            model_config=gpt_model_cfg,
            train_config=train_cfg,
            validation_config=val_cfg,
        )
        container.dist.use_decentralized_pg = True
        container.dist.use_gloo_process_groups = False
        container.dist.eval_context_parallel_size = 2

        try:
            container.validate()
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_eval_default_divisibility_check_fails(self, monkeypatch):
        """Even with defaults from training, the divisibility check should catch mismatches."""
        gpt_model_cfg = create_test_gpt_config()
        # global_batch_size=10, micro_batch_size=4, DP=8 => 10 / (4*8) not integer
        train_cfg = create_test_training_config(global_batch_size=10, micro_batch_size=4)
        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=8, model_config=gpt_model_cfg, train_config=train_cfg
        )
        try:
            with pytest.raises(AssertionError, match="eval_global_batch_size.*must be divisible by"):
                container.validate()
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_eval_micro_batch_size_none_without_training_value_fails(self, monkeypatch):
        """If train.micro_batch_size is None and eval is not explicitly set, assert should fire."""
        gpt_model_cfg = create_test_gpt_config()
        # micro_batch_size must be explicitly None to test this path
        train_cfg = create_test_training_config(global_batch_size=32, micro_batch_size=None)
        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=8, model_config=gpt_model_cfg, train_config=train_cfg
        )
        try:
            with pytest.raises(AssertionError, match="train.micro_batch_size must be set"):
                container.validate()
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_eval_global_batch_size_none_without_training_value_fails(self, monkeypatch):
        """If train.global_batch_size is None and eval is not explicitly set, assert should fire."""
        gpt_model_cfg = create_test_gpt_config()
        # global_batch_size=None with train_samples triggers assertion in TrainingConfig.finalize()
        train_cfg = TrainingConfig(micro_batch_size=4, train_samples=1000)
        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=8, model_config=gpt_model_cfg, train_config=train_cfg
        )
        try:
            with pytest.raises(AssertionError, match="global_batch_size must be set"):
                container.validate()
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_eval_global_batch_size_none_with_train_iters_fails(self, monkeypatch):
        """If train.global_batch_size is None (train_iters mode), eval default resolution should fail."""
        gpt_model_cfg = create_test_gpt_config()
        # In train_iters mode, global_batch_size=None won't crash in finalize() but will
        # fail when resolving eval defaults
        train_cfg = TrainingConfig(micro_batch_size=4, train_iters=1000, global_batch_size=None)
        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=8, model_config=gpt_model_cfg, train_config=train_cfg
        )
        try:
            with pytest.raises(AssertionError, match="train.global_batch_size must be set"):
                container.validate()
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    @pytest.mark.parametrize(
        "eval_gbs, eval_mbs, world_size, expect_error",
        [
            (32, 4, 8, False),  # 32 / (4*8) = 1
            (64, 8, 8, False),  # 64 / (8*8) = 1
            (32, 4, 4, False),  # 32 / (4*4) = 2
            (32, 3, 8, True),  # 32 / (3*8) not integer
            (15, 2, 8, True),  # 15 / (2*8) not integer
        ],
    )
    def test_eval_batch_size_divisibility_parametrized(
        self, monkeypatch, eval_gbs, eval_mbs, world_size, expect_error
    ):
        """Parametrized test for eval batch size divisibility."""
        gpt_model_cfg = create_test_gpt_config()
        train_cfg = create_test_training_config(global_batch_size=64, micro_batch_size=4)
        val_cfg = ValidationConfig(eval_global_batch_size=eval_gbs, eval_micro_batch_size=eval_mbs)
        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=world_size,
            model_config=gpt_model_cfg,
            train_config=train_cfg,
            validation_config=val_cfg,
        )
        try:
            if expect_error:
                with pytest.raises(AssertionError, match="eval_global_batch_size.*must be divisible by"):
                    container.validate()
            else:
                container.validate()
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)


class TestRerunConfigValidation:
    """
    Test that finalize() functions behave correctly when called multiple times:
    - All configs now use finalize() method for validation and computed field calculation to handle deferred overrides.
    - finalize() may change computed fields on first call, but subsequent calls are idempotent
    - Tests the same behavior for ConfigContainer.validate().
    """

    def _check_finalize_idempotency(self, cfg_init_fn):
        import copy

        cfg = cfg_init_fn()
        cfg_copy = copy.deepcopy(cfg)
        assert cfg == cfg_copy

        # All configs now use finalize() method
        cfg.finalize()
        # For configs that may change computed fields, take a new snapshot after first finalization
        cfg_after_finalize = copy.deepcopy(cfg)
        # Second finalize() should be idempotent (no further changes)
        cfg.finalize()
        assert cfg == cfg_after_finalize

    def test_scheduler_config(self):
        self._check_finalize_idempotency(create_test_scheduler_config)

        # Test rerun of finalize with valid and invalid changes
        cfg = create_test_scheduler_config(lr_decay_iters=10)
        cfg.lr_decay_iters = 20
        cfg.finalize()

        with pytest.raises(AssertionError, match="start_weight_decay"):
            cfg.start_weight_decay = -5.2
            cfg.finalize()

    def test_gptdataset_config(self):
        def gpt_dataset_seqlen_1024():
            return create_test_gpt_dataset_config(1024)

        self._check_finalize_idempotency(gpt_dataset_seqlen_1024)

        # Test rerun of finalize with valid and invalid changes
        cfg = gpt_dataset_seqlen_1024()
        cfg.random_seed = 2468
        cfg.finalize()

        with pytest.raises(AssertionError, match="reset_position_ids"):
            cfg.reset_position_ids = None
            cfg.finalize()

    def test_profiling_config(self):
        self._check_finalize_idempotency(create_test_profiling_config)

        # Test rerun of finalize with valid and invalid changes
        cfg = create_test_profiling_config()
        cfg.profile_step_end = 1000
        cfg.finalize()

        with pytest.raises(AssertionError, match="one of pytorch or nsys profiler should be enabled"):
            cfg.use_nsys_profiler = True
            cfg.use_pytorch_profiler = True
            cfg.finalize()

    def test_nvrx_straggler_config(self):
        self._check_finalize_idempotency(create_test_nvrx_straggler_config)

        # Test rerun of finalize with valid and invalid changes
        cfg = create_test_nvrx_straggler_config(enabled=True)
        cfg.num_gpu_perf_scores_to_print = 2
        cfg.finalize()

        with pytest.raises(ValueError, match="report_time_interval must be positive"):
            cfg.report_time_interval = -100.0
            cfg.finalize()

    def test_checkpoint_config(self):
        self._check_finalize_idempotency(create_test_checkpoint_config)

        # Test rerun of finalize with valid and invalid changes
        cfg = create_test_checkpoint_config(ckpt_format="torch_dist")
        cfg.save = "/tmp/test_checkpoint_config"
        cfg.finalize()

        with pytest.raises(AssertionError, match="load_main_params_from_ckpt must be used with load_optim=False"):
            cfg.load_main_params_from_ckpt = True
            cfg.load_optim = True
            cfg.finalize()

    def test_mixed_precision_config(self):
        from megatron.bridge.training.mixed_precision import bf16_with_mxfp8_mixed

        self._check_finalize_idempotency(bf16_with_mxfp8_mixed)
        cfg = bf16_with_mxfp8_mixed()
        cfg.grad_reduce_in_fp32 = False
        cfg.finalize()

    def test_comm_overlap_config(self):
        """Test that CommOverlapConfig.finalize() is idempotent and preserves user configuration."""

        def create_comm_overlap_config():
            return CommOverlapConfig(
                tp_comm_overlap=True,
                tp_comm_bootstrap_backend="nccl",
            )

        # Use the standard idempotency check
        self._check_finalize_idempotency(create_comm_overlap_config)

        cfg = create_comm_overlap_config()
        cfg.finalize()
        assert cfg.user_comm_overlap_cfg.tp_comm_bootstrap_backend == "nccl"
        assert cfg.user_comm_overlap_cfg.tp_comm_overlap is True
        cfg.finalize()

        # The user configuration should be preserved across all re-runs
        assert cfg.user_comm_overlap_cfg.tp_comm_bootstrap_backend == "nccl"
        assert cfg.user_comm_overlap_cfg.tp_comm_overlap is True

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
        ckpt_cfg = create_test_checkpoint_config(
            load_main_params_from_ckpt=load_main_params_from_ckpt, load_optim=load_optim
        )
        gpt_model_cfg = create_test_gpt_config()
        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1, model_config=gpt_model_cfg, checkpoint_config=ckpt_cfg
        )

        try:
            if expect_assertion_error:
                with pytest.raises(
                    AssertionError, match="load_main_params_from_ckpt must be used with load_optim=False"
                ):
                    container.validate()  # Validation error should occur here during finalize
            else:
                container.validate()  # Should pass without error
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_ckpt_step_requires_load_directory(self):
        """Test that ckpt_step requires checkpoint.load to be set."""
        # Test that ckpt_step without load fails
        ckpt_cfg = create_test_checkpoint_config(ckpt_step=5000, load=None)

        with pytest.raises(ValueError) as exc_info:
            ckpt_cfg.finalize()

        assert "ckpt_step=5000 specified but checkpoint.load is None" in str(exc_info.value)
        assert "Please set checkpoint.load to the base checkpoint directory" in str(exc_info.value)

    def test_ckpt_step_with_load_directory_passes(self):
        """Test that ckpt_step with checkpoint.load passes validation."""
        ckpt_cfg = create_test_checkpoint_config(ckpt_step=5000, load="/checkpoints")

        # Should not raise any errors
        ckpt_cfg.finalize()
        assert ckpt_cfg.ckpt_step == 5000
        assert ckpt_cfg.load == "/checkpoints"

    def test_save_weight_format_field_is_removed(self):
        """Test that the old save_weight_format alias is no longer part of CheckpointConfig."""
        assert "save_weight_format" not in {field.name for field in fields(CheckpointConfig)}

    def test_also_save_hf_checkpoint_rejects_fsdp_dtensor(self):
        """Test that HF extra export is not allowed with fsdp_dtensor checkpoints."""
        ckpt_cfg = create_test_checkpoint_config(also_save_hf_checkpoint=True, ckpt_format="fsdp_dtensor")

        with pytest.raises(ValueError, match="also_save_hf_checkpoint=True is not supported"):
            ckpt_cfg.finalize()

    def test_also_save_hf_checkpoint_rejects_local_non_persistent_checkpoint(self):
        """Test that HF extra export is not allowed for local non-persistent checkpoints."""
        ckpt_cfg = create_test_checkpoint_config(also_save_hf_checkpoint=True, non_persistent_ckpt_type="local")

        with pytest.raises(ValueError, match="also_save_hf_checkpoint=True is not compatible"):
            ckpt_cfg.finalize()

    def test_also_save_hf_checkpoint_requires_hf_source_during_container_validation(self):
        """Test that HF extra export requires a source before training starts."""
        checkpoint_cfg = create_test_checkpoint_config(also_save_hf_checkpoint=True)
        model_cfg = create_test_gpt_config(hf_model_id=None)
        tokenizer_cfg = create_test_tokenizer_config(tokenizer_model=None)
        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=model_cfg,
            tokenizer_config=tokenizer_cfg,
            checkpoint_config=checkpoint_cfg,
        )

        try:
            with pytest.raises(ValueError, match="also_save_hf_checkpoint=True requires an HF source"):
                container.validate()
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_also_save_hf_checkpoint_accepts_hf_source_path_during_container_validation(self):
        """Test that explicit hf_source_path satisfies HF extra export validation."""
        checkpoint_cfg = create_test_checkpoint_config(
            also_save_hf_checkpoint=True,
            hf_source_path="/hf/source",
        )
        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=create_test_gpt_config(hf_model_id=None),
            tokenizer_config=create_test_tokenizer_config(tokenizer_model=None),
            checkpoint_config=checkpoint_cfg,
        )

        try:
            container.validate()
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_also_save_hf_checkpoint_accepts_model_hf_model_id_during_container_validation(self):
        """Test that model.hf_model_id satisfies HF extra export validation."""
        checkpoint_cfg = create_test_checkpoint_config(also_save_hf_checkpoint=True)
        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=create_test_gpt_config(hf_model_id="hf/model"),
            tokenizer_config=create_test_tokenizer_config(tokenizer_model=None),
            checkpoint_config=checkpoint_cfg,
        )

        try:
            container.validate()
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_also_save_hf_checkpoint_accepts_tokenizer_model_during_container_validation(self):
        """Test that tokenizer.tokenizer_model satisfies HF extra export validation."""
        checkpoint_cfg = create_test_checkpoint_config(also_save_hf_checkpoint=True)
        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=create_test_gpt_config(hf_model_id=None),
            tokenizer_config=create_test_tokenizer_config(tokenizer_model="hf/tokenizer-or-model"),
            checkpoint_config=checkpoint_cfg,
        )

        try:
            container.validate()
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_async_save_validation_error(self):
        """Test that async_save requires both a save path and use_persistent_ckpt_worker=True."""

        # Test that async_save requires a save path
        ckpt_cfg1 = create_test_checkpoint_config(async_save=True, save=None)
        gpt_model_cfg1 = create_test_gpt_config()
        container1, og_ws1, cfg_mod1 = create_test_config_container(
            world_size_override=1, model_config=gpt_model_cfg1, checkpoint_config=ckpt_cfg1
        )

        try:
            with pytest.raises(
                AssertionError, match="async_save is enabled, but save is not set. Set save to a valid path."
            ):
                container1.validate()
        finally:
            restore_get_world_size_safe(og_ws1, cfg_mod1)

        # Test that async_save requires use_persistent_ckpt_worker=True
        ckpt_cfg2 = create_test_checkpoint_config(
            async_save=True, save="/tmp/test_checkpoint_config", use_persistent_ckpt_worker=False
        )
        gpt_model_cfg2 = create_test_gpt_config()
        container2, og_ws2, cfg_mod2 = create_test_config_container(
            world_size_override=1, model_config=gpt_model_cfg2, checkpoint_config=ckpt_cfg2
        )

        try:
            with pytest.raises(AssertionError, match="async_save requires use_persistent_ckpt_worker=True."):
                container2.validate()
        finally:
            restore_get_world_size_safe(og_ws2, cfg_mod2)

        # should not raise an error when both conditions are met
        ckpt_cfg3 = create_test_checkpoint_config(
            async_save=True, save="/tmp/test_checkpoint_config", use_persistent_ckpt_worker=True
        )
        gpt_model_cfg3 = create_test_gpt_config()
        container3, og_ws3, cfg_mod3 = create_test_config_container(
            world_size_override=1, model_config=gpt_model_cfg3, checkpoint_config=ckpt_cfg3
        )

        try:
            container3.validate()  # Should pass without error
        finally:
            restore_get_world_size_safe(og_ws3, cfg_mod3)

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

    def test_pretrained_checkpoint_none_skips_validation(self):
        """Test that finalize succeeds when pretrained_checkpoint is None (no file existence check)."""
        ckpt_cfg = create_test_checkpoint_config(pretrained_checkpoint=None)
        # Should not raise any errors
        ckpt_cfg.finalize()

    @patch("megatron.bridge.training.utils.checkpoint_utils.file_exists", return_value=True)
    def test_pretrained_checkpoint_exists_passes(self, mock_file_exists):
        """Test that finalize succeeds when pretrained_checkpoint path exists."""
        ckpt_cfg = create_test_checkpoint_config(pretrained_checkpoint="/path/to/valid/checkpoint")
        # Should not raise any errors
        ckpt_cfg.finalize()
        mock_file_exists.assert_called_once_with("/path/to/valid/checkpoint")

    @patch("megatron.bridge.training.utils.checkpoint_utils.file_exists", return_value=False)
    def test_pretrained_checkpoint_not_exists_raises(self, mock_file_exists):
        """Test that finalize raises AssertionError when pretrained_checkpoint path does not exist."""
        ckpt_cfg = create_test_checkpoint_config(pretrained_checkpoint="/path/to/missing/checkpoint")
        with pytest.raises(AssertionError, match="Pretrained checkpoint /path/to/missing/checkpoint does not exist"):
            ckpt_cfg.finalize()
        mock_file_exists.assert_called_once_with("/path/to/missing/checkpoint")


class TestMixedPrecisionConsistencyValidation:
    """Tests for _validate_mixed_precision_consistency function.

    These tests verify that precision settings (bf16/fp16) are properly validated
    between model and optimizer configs, especially when use_precision_aware_optimizer=True.
    """

    def test_bf16_model_bf16_optimizer_with_precision_aware_passes(self):
        """Test that bf16 model + bf16 optimizer + precision_aware passes validation."""
        gpt_model_cfg = create_test_gpt_config(bf16=True, fp16=False)
        optim_cfg = create_test_optimizer_config(
            bf16=True,
            fp16=False,
            use_precision_aware_optimizer=True,
            use_distributed_optimizer=True,
        )

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=gpt_model_cfg,
            optimizer_config=optim_cfg,
        )
        try:
            # Should pass without error
            _validate_mixed_precision_consistency(container)
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_fp16_model_fp16_optimizer_with_precision_aware_passes(self):
        """Test that fp16 model + fp16 optimizer + precision_aware passes validation."""
        gpt_model_cfg = create_test_gpt_config(bf16=False, fp16=True)
        optim_cfg = create_test_optimizer_config(
            bf16=False,
            fp16=True,
            use_precision_aware_optimizer=True,
            use_distributed_optimizer=True,
        )

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=gpt_model_cfg,
            optimizer_config=optim_cfg,
        )
        try:
            # Should pass without error
            _validate_mixed_precision_consistency(container)
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_fp32_model_fp32_optimizer_with_precision_aware_passes(self):
        """Test that fp32 model + fp32 optimizer + precision_aware passes validation."""
        gpt_model_cfg = create_test_gpt_config(bf16=False, fp16=False)
        optim_cfg = create_test_optimizer_config(
            bf16=False,
            fp16=False,
            use_precision_aware_optimizer=True,
            use_distributed_optimizer=True,
        )

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=gpt_model_cfg,
            optimizer_config=optim_cfg,
        )
        try:
            # Should pass without error
            _validate_mixed_precision_consistency(container)
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_bf16_model_fp16_optimizer_with_precision_aware_fails(self):
        """Test that bf16 model + fp16 optimizer + precision_aware fails validation."""
        gpt_model_cfg = create_test_gpt_config(bf16=True, fp16=False)
        optim_cfg = create_test_optimizer_config(
            bf16=False,
            fp16=True,
            use_precision_aware_optimizer=True,
            use_distributed_optimizer=True,
        )

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=gpt_model_cfg,
            optimizer_config=optim_cfg,
        )
        try:
            with pytest.raises(AssertionError, match="optimizer.bf16=True must be set when model.bf16=True"):
                _validate_mixed_precision_consistency(container)
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_bf16_model_fp32_optimizer_with_precision_aware_fails(self):
        """Test that bf16 model + fp32 optimizer + precision_aware fails validation."""
        gpt_model_cfg = create_test_gpt_config(bf16=True, fp16=False)
        optim_cfg = create_test_optimizer_config(
            bf16=False,
            fp16=False,
            use_precision_aware_optimizer=True,
            use_distributed_optimizer=True,
        )

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=gpt_model_cfg,
            optimizer_config=optim_cfg,
        )
        try:
            with pytest.raises(AssertionError, match="optimizer.bf16=True must be set when model.bf16=True"):
                _validate_mixed_precision_consistency(container)
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_fp16_model_bf16_optimizer_with_precision_aware_fails(self):
        """Test that fp16 model + bf16 optimizer + precision_aware fails validation."""
        gpt_model_cfg = create_test_gpt_config(bf16=False, fp16=True)
        optim_cfg = create_test_optimizer_config(
            bf16=True,
            fp16=False,
            use_precision_aware_optimizer=True,
            use_distributed_optimizer=True,
        )

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=gpt_model_cfg,
            optimizer_config=optim_cfg,
        )
        try:
            with pytest.raises(AssertionError, match="optimizer.fp16=True must be set when model.fp16=True"):
                _validate_mixed_precision_consistency(container)
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_fp16_model_fp32_optimizer_with_precision_aware_fails(self):
        """Test that fp16 model + fp32 optimizer + precision_aware fails validation."""
        gpt_model_cfg = create_test_gpt_config(bf16=False, fp16=True)
        optim_cfg = create_test_optimizer_config(
            bf16=False,
            fp16=False,
            use_precision_aware_optimizer=True,
            use_distributed_optimizer=True,
        )

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=gpt_model_cfg,
            optimizer_config=optim_cfg,
        )
        try:
            with pytest.raises(AssertionError, match="optimizer.fp16=True must be set when model.fp16=True"):
                _validate_mixed_precision_consistency(container)
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_fp32_model_bf16_optimizer_with_precision_aware_fails(self):
        """Test that fp32 model + bf16 optimizer + precision_aware fails validation."""
        gpt_model_cfg = create_test_gpt_config(bf16=False, fp16=False)
        optim_cfg = create_test_optimizer_config(
            bf16=True,
            fp16=False,
            use_precision_aware_optimizer=True,
            use_distributed_optimizer=True,
        )

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=gpt_model_cfg,
            optimizer_config=optim_cfg,
        )
        try:
            with pytest.raises(AssertionError, match="optimizer.bf16 and optimizer.fp16 must both be False"):
                _validate_mixed_precision_consistency(container)
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_fp32_model_fp16_optimizer_with_precision_aware_fails(self):
        """Test that fp32 model + fp16 optimizer + precision_aware fails validation."""
        gpt_model_cfg = create_test_gpt_config(bf16=False, fp16=False)
        optim_cfg = create_test_optimizer_config(
            bf16=False,
            fp16=True,
            use_precision_aware_optimizer=True,
            use_distributed_optimizer=True,
        )

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=gpt_model_cfg,
            optimizer_config=optim_cfg,
        )
        try:
            with pytest.raises(AssertionError, match="optimizer.bf16 and optimizer.fp16 must both be False"):
                _validate_mixed_precision_consistency(container)
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_mismatch_without_precision_aware_optimizer_passes(self):
        """Test that mismatched settings pass when use_precision_aware_optimizer=False."""
        gpt_model_cfg = create_test_gpt_config(bf16=True, fp16=False)
        optim_cfg = create_test_optimizer_config(
            bf16=False,
            fp16=False,
            use_precision_aware_optimizer=False,
            use_distributed_optimizer=False,
        )

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=gpt_model_cfg,
            optimizer_config=optim_cfg,
        )
        try:
            # Should pass without error when precision_aware_optimizer is disabled
            _validate_mixed_precision_consistency(container)
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_model_both_bf16_fp16_true_fails(self):
        """Test that model with both bf16=True and fp16=True fails validation."""
        gpt_model_cfg = create_test_gpt_config(bf16=True, fp16=True)
        optim_cfg = create_test_optimizer_config()

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=gpt_model_cfg,
            optimizer_config=optim_cfg,
        )
        try:
            with pytest.raises(AssertionError, match="Model config cannot have both bf16=True and fp16=True"):
                _validate_mixed_precision_consistency(container)
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_optimizer_both_bf16_fp16_true_fails(self):
        """Test that optimizer with both bf16=True and fp16=True fails validation."""
        gpt_model_cfg = create_test_gpt_config(bf16=False, fp16=False)
        optim_cfg = create_test_optimizer_config(bf16=True, fp16=True)

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=gpt_model_cfg,
            optimizer_config=optim_cfg,
        )
        try:
            with pytest.raises(AssertionError, match="Optimizer config cannot have both bf16=True and fp16=True"):
                _validate_mixed_precision_consistency(container)
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_validation_called_during_container_validate(self):
        """Test that mixed precision validation is called during ConfigContainer.validate()."""
        gpt_model_cfg = create_test_gpt_config(bf16=True, fp16=False)
        train_cfg = create_test_training_config(train_iters=500, global_batch_size=16)
        sched_cfg = create_test_scheduler_config()
        optim_cfg = create_test_optimizer_config(
            bf16=False,  # Mismatch with model
            fp16=False,
            use_precision_aware_optimizer=True,
            use_distributed_optimizer=True,
        )
        ddp_cfg = create_test_ddp_config(use_distributed_optimizer=True)

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=gpt_model_cfg,
            train_config=train_cfg,
            scheduler_config=sched_cfg,
            optimizer_config=optim_cfg,
            ddp_config=ddp_cfg,
        )
        try:
            # Should fail during validate() because of precision mismatch
            with pytest.raises(AssertionError, match="optimizer.bf16=True must be set when model.bf16=True"):
                container.validate()
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)


class TestRuntimeConfigUpdate:
    """Tests for the runtime_config_update function."""

    def test_recipe_environment_variables_are_applied_before_runtime_update(self, monkeypatch):
        """Recipe env defaults should be stringified without replacing launcher values."""
        gpt_cfg = create_test_gpt_config()
        full_cfg, og_ws, cfg_mod = create_test_config_container(world_size_override=1, model_config=gpt_cfg)
        full_cfg.env_vars = {
            "NVTE_FWD_LAYERNORM_SM_MARGIN": 16,
            "NVTE_BWD_LAYERNORM_SM_MARGIN": 16,
            "TORCHINDUCTOR_WORKER_START": "fork",
            "QUANTIZATION_TYPE_DEBUG": 1,
            "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 64,
            "USE_MNNVL": 1,
        }
        for name in full_cfg.env_vars:
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("USE_MNNVL", "launcher-value")

        try:
            from megatron.bridge.training.config import runtime_config_update

            runtime_config_update(full_cfg)

            assert os.environ["NVTE_FWD_LAYERNORM_SM_MARGIN"] == "16"
            assert os.environ["NVTE_BWD_LAYERNORM_SM_MARGIN"] == "16"
            assert os.environ["TORCHINDUCTOR_WORKER_START"] == "fork"
            assert os.environ["QUANTIZATION_TYPE_DEBUG"] == "1"
            assert os.environ["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] == "64"
            assert os.environ["USE_MNNVL"] == "launcher-value"
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    @pytest.mark.parametrize("env_vars", [{"": "1"}, {1: "bad-name"}, {"VALID_NAME": ["not", "scalar"]}])
    def test_recipe_environment_variables_reject_invalid_values(self, env_vars):
        """Invalid recipe environment mappings should fail before training starts."""
        config = MagicMock(spec=ConfigContainer)
        config.env_vars = env_vars

        with pytest.raises((TypeError, ValueError)):
            apply_environment_variables(config)

    def test_recipe_environment_variables_support_hydra_overrides(self):
        """The top-level mapping should be replaceable through the shared recipe override path."""
        from megatron.bridge.training.utils.omegaconf_utils import process_config_with_overrides

        gpt_cfg = create_test_gpt_config()
        full_cfg, og_ws, cfg_mod = create_test_config_container(world_size_override=1, model_config=gpt_cfg)

        try:
            updated_cfg = process_config_with_overrides(
                full_cfg,
                cli_overrides=[
                    "++env_vars={NVTE_FWD_LAYERNORM_SM_MARGIN:16,TORCHINDUCTOR_WORKER_START:fork,USE_MNNVL:1}"
                ],
            )

            assert updated_cfg.env_vars == {
                "NVTE_FWD_LAYERNORM_SM_MARGIN": 16,
                "TORCHINDUCTOR_WORKER_START": "fork",
                "USE_MNNVL": 1,
            }
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_recipe_environment_variables_round_trip_through_yaml(self, tmp_path):
        """Environment mappings should be preserved in saved recipe configs."""
        gpt_cfg = create_test_gpt_config()
        full_cfg, og_ws, cfg_mod = create_test_config_container(world_size_override=1, model_config=gpt_cfg)
        full_cfg.env_vars = {
            "TORCHINDUCTOR_WORKER_START": "fork",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
        config_path = tmp_path / "recipe.yaml"

        try:
            full_cfg.to_yaml(str(config_path))
            restored_cfg = ConfigContainer.from_yaml(str(config_path))

            assert restored_cfg.env_vars == full_cfg.env_vars
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_hf_model_revision_round_trip_through_yaml(self, tmp_path):
        """Immutable Hugging Face model provenance should survive runtime config persistence."""
        revision = "b968826d9c46dd6066d109eabc6255188de91218"  # pragma: allowlist secret
        gpt_cfg = create_test_gpt_config(hf_model_id="Qwen/Qwen3-8B", hf_model_revision=revision)
        full_cfg, og_ws, cfg_mod = create_test_config_container(world_size_override=1, model_config=gpt_cfg)
        config_path = tmp_path / "recipe.yaml"

        try:
            full_cfg.to_yaml(str(config_path))
            restored_cfg = ConfigContainer.from_yaml(str(config_path))

            assert restored_cfg.model.hf_model_id == "Qwen/Qwen3-8B"
            assert restored_cfg.model.hf_model_revision == revision
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_runtime_config_update_with_mixed_precision_string(self):
        """Test runtime_config_update with mixed precision as string."""
        from megatron.bridge.training.config import runtime_config_update

        def patched_init_method():
            return torch.nn.init.normal_(mean=0.0, std=0.02)

        gpt_cfg = create_test_gpt_config(init_method=patched_init_method, output_layer_init_method=patched_init_method)
        full_cfg, og_ws, cfg_mod = create_test_config_container(world_size_override=4, model_config=gpt_cfg)

        # Set mixed precision as string
        full_cfg.mixed_precision = "bf16_mixed"

        try:
            # Verify initial state
            assert isinstance(full_cfg.mixed_precision, str)
            assert not hasattr(full_cfg, "data_parallel_size")

            # Run runtime config update
            runtime_config_update(full_cfg)

            # Verify results
            assert not isinstance(full_cfg.mixed_precision, str)  # Should be resolved to config object
            assert hasattr(full_cfg, "data_parallel_size")
            assert full_cfg.data_parallel_size == 4  # world_size / model_parallel_size
            assert full_cfg.model.bf16 is True  # Mixed precision should be applied

        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_runtime_config_update_with_comm_overlap(self):
        """Test runtime_config_update with communication overlap configuration."""
        from megatron.bridge.training.comm_overlap import CommOverlapConfig
        from megatron.bridge.training.config import runtime_config_update

        def patched_init_method():
            return torch.nn.init.normal_(mean=0.0, std=0.02)

        gpt_cfg = create_test_gpt_config(init_method=patched_init_method, output_layer_init_method=patched_init_method)
        full_cfg, og_ws, cfg_mod = create_test_config_container(world_size_override=8, model_config=gpt_cfg)

        full_cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)

        try:
            # Verify initial state
            assert not hasattr(full_cfg, "data_parallel_size")
            assert full_cfg.comm_overlap.data_parallel_size is None  # Field exists but is None

            # Run runtime config update
            runtime_config_update(full_cfg)

            # Verify results
            assert hasattr(full_cfg, "data_parallel_size")
            assert full_cfg.data_parallel_size == 8  # world_size / model_parallel_size
            assert full_cfg.comm_overlap.data_parallel_size == 8  # Should be set by runtime_config_update

        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_runtime_config_update_finalization(self):
        """Test that runtime_config_update properly finalizes configs."""
        from megatron.bridge.training.config import runtime_config_update

        def patched_init_method():
            return torch.nn.init.normal_(mean=0.0, std=0.02)

        gpt_cfg = create_test_gpt_config(init_method=patched_init_method, output_layer_init_method=patched_init_method)
        full_cfg, og_ws, cfg_mod = create_test_config_container(world_size_override=4, model_config=gpt_cfg)

        try:
            # Verify configs are not finalized initially (for configs that inherit from MCore)
            if isinstance(full_cfg.dataset, GPTDatasetConfig):
                # GPTDatasetConfig inherits from MCore, should have deferred post-init
                assert getattr(full_cfg.dataset, "split", None) is None  # Computed field not set yet

            # Run runtime config update
            runtime_config_update(full_cfg)

            # Verify configs are finalized
            if isinstance(full_cfg.dataset, GPTDatasetConfig):
                # Computed fields should now be set
                assert getattr(full_cfg.dataset, "split", None) is not None

            # Verify model config is finalized (computed fields set)
            assert full_cfg.model.num_query_groups is not None

        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_runtime_config_update_no_mixed_precision_or_comm_overlap(self):
        """Test runtime_config_update with no mixed precision or comm overlap."""
        from megatron.bridge.training.config import runtime_config_update

        def patched_init_method():
            return torch.nn.init.normal_(mean=0.0, std=0.02)

        gpt_cfg = create_test_gpt_config(init_method=patched_init_method, output_layer_init_method=patched_init_method)
        full_cfg, og_ws, cfg_mod = create_test_config_container(world_size_override=2, model_config=gpt_cfg)

        # Ensure no mixed precision or comm overlap
        full_cfg.mixed_precision = None
        full_cfg.comm_overlap = None

        try:
            # Run runtime config update
            runtime_config_update(full_cfg)

            # Verify basic functionality works
            assert hasattr(full_cfg, "data_parallel_size")
            assert full_cfg.data_parallel_size == 2

        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_runtime_config_update_idempotency(self):
        """Test that runtime_config_update can be called multiple times safely."""
        from megatron.bridge.training.config import runtime_config_update

        def patched_init_method():
            return torch.nn.init.normal_(mean=0.0, std=0.02)

        gpt_cfg = create_test_gpt_config(init_method=patched_init_method, output_layer_init_method=patched_init_method)
        full_cfg, og_ws, cfg_mod = create_test_config_container(world_size_override=4, model_config=gpt_cfg)

        try:
            # Run runtime config update twice
            runtime_config_update(full_cfg)
            first_state = {
                "data_parallel_size": full_cfg.data_parallel_size,
                "model_num_query_groups": full_cfg.model.num_query_groups,
            }

            runtime_config_update(full_cfg)
            second_state = {
                "data_parallel_size": full_cfg.data_parallel_size,
                "model_num_query_groups": full_cfg.model.num_query_groups,
            }

            # Verify idempotency - second call should not change anything
            assert first_state == second_state

        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)


class TestDistributedOptimizerValidation:
    """Tests for the _validate_and_sync_distributed_optimizer_settings function."""

    @pytest.mark.parametrize(
        "ddp_setting, optimizer_setting, expected_final_state, should_print_message, expected_message_parts",
        [
            # Cases where sync is needed
            (
                True,
                False,
                True,
                True,
                ["ddp.use_distributed_optimizer=True", "optimizer.use_distributed_optimizer=False"],
            ),
            (
                False,
                True,
                True,
                True,
                ["ddp.use_distributed_optimizer=False", "optimizer.use_distributed_optimizer=True"],
            ),
            # Cases where no sync is needed
            (True, True, True, False, []),
            (False, False, False, False, []),
        ],
    )
    @patch("megatron.bridge.training.config.warn_rank_0")
    def test_distributed_optimizer_sync_scenarios(
        self,
        mock_warn_rank_0,
        ddp_setting,
        optimizer_setting,
        expected_final_state,
        should_print_message,
        expected_message_parts,
    ):
        """Test various distributed optimizer sync scenarios."""
        gpt_model_cfg = create_test_gpt_config()
        ddp_cfg = create_test_ddp_config(use_distributed_optimizer=ddp_setting)
        optimizer_cfg = create_test_optimizer_config(use_distributed_optimizer=optimizer_setting)

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=gpt_model_cfg,
            ddp_config=ddp_cfg,
            optimizer_config=optimizer_cfg,
        )

        try:
            # Before validation
            assert container.ddp.use_distributed_optimizer is ddp_setting
            assert container.optimizer.use_distributed_optimizer is optimizer_setting

            # Call the validation function directly
            _validate_and_sync_distributed_optimizer_settings(container)

            # After validation - both should match expected final state
            assert container.ddp.use_distributed_optimizer is expected_final_state
            assert container.optimizer.use_distributed_optimizer is expected_final_state

            # Check warning behavior
            if should_print_message:
                mock_warn_rank_0.assert_called_once()
                call_args = mock_warn_rank_0.call_args[0][0]
                assert "Distributed optimizer settings were not in sync" in call_args
                assert "Automatically enabling distributed optimizer for both settings" in call_args
                for expected_part in expected_message_parts:
                    assert expected_part in call_args
            else:
                mock_warn_rank_0.assert_not_called()

        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    @patch("megatron.bridge.training.config.warn_rank_0")
    def test_integration_with_config_container_validation(self, mock_warn_rank_0):
        """Test that the function is properly called during ConfigContainer.validate()."""
        gpt_model_cfg = create_test_gpt_config()
        ddp_cfg = create_test_ddp_config(use_distributed_optimizer=True)
        optimizer_cfg = create_test_optimizer_config(use_distributed_optimizer=False)

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=gpt_model_cfg,
            ddp_config=ddp_cfg,
            optimizer_config=optimizer_cfg,
        )

        try:
            # Before validation
            assert container.ddp.use_distributed_optimizer is True
            assert container.optimizer.use_distributed_optimizer is False

            # Call container.validate() which should trigger our function
            container.validate()

            # After validation - both should be True
            assert container.ddp.use_distributed_optimizer is True
            assert container.optimizer.use_distributed_optimizer is True

            # Should have issued the sync warning
            mock_warn_rank_0.assert_called()
            call_args = mock_warn_rank_0.call_args[0][0]
            assert "Distributed optimizer settings were not in sync" in call_args

        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    @pytest.mark.parametrize(
        "ddp_overlap, optimizer_overlap, expected_final_state, should_print_message, expected_message_parts",
        [
            (True, False, True, True, ["ddp.overlap_param_gather=True", "optimizer.overlap_param_gather=False"]),
            (False, True, True, True, ["ddp.overlap_param_gather=False", "optimizer.overlap_param_gather=True"]),
            (True, True, True, False, []),
            (False, False, False, False, []),
        ],
    )
    @patch("megatron.bridge.training.config.warn_rank_0")
    def test_overlap_param_gather_sync_scenarios(
        self,
        mock_warn_rank_0,
        ddp_overlap,
        optimizer_overlap,
        expected_final_state,
        should_print_message,
        expected_message_parts,
    ):
        """Test overlap_param_gather sync between DDP and optimizer configs."""
        gpt_model_cfg = create_test_gpt_config()
        ddp_cfg = create_test_ddp_config(overlap_param_gather=ddp_overlap)
        optimizer_cfg = create_test_optimizer_config(overlap_param_gather=optimizer_overlap)

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=gpt_model_cfg,
            ddp_config=ddp_cfg,
            optimizer_config=optimizer_cfg,
        )

        try:
            assert container.ddp.overlap_param_gather is ddp_overlap
            assert container.optimizer.overlap_param_gather is optimizer_overlap

            _validate_and_sync_distributed_optimizer_settings(container)

            assert container.ddp.overlap_param_gather is expected_final_state
            assert container.optimizer.overlap_param_gather is expected_final_state

            overlap_warnings = [
                call
                for call in mock_warn_rank_0.call_args_list
                if call[0] and "overlap_param_gather settings were not in sync" in call[0][0]
            ]
            if should_print_message:
                assert len(overlap_warnings) == 1
                call_args = overlap_warnings[0][0][0]
                assert "Automatically enabling overlap_param_gather for both settings" in call_args
                for expected_part in expected_message_parts:
                    assert expected_part in call_args
            else:
                assert len(overlap_warnings) == 0

        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)


class TestSampleBasedTraining:
    """Tests for sample-based training configuration and validation."""

    def test_sample_based_training_config_creation(self):
        """Test creating a valid sample-based training configuration."""
        train_cfg = create_test_training_config(train_samples=10000, train_iters=None, global_batch_size=32)
        sched_cfg = create_test_scheduler_config(
            lr_decay_samples=8000, lr_warmup_samples=1000, lr_decay_iters=None, lr_warmup_iters=0
        )

        gpt_model_cfg = create_test_gpt_config()
        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1, model_config=gpt_model_cfg, train_config=train_cfg, scheduler_config=sched_cfg
        )

        try:
            container.validate()
            # Verify train_iters was calculated from train_samples
            expected_train_iters = train_cfg.train_samples // train_cfg.global_batch_size
            assert container.train.train_iters == expected_train_iters

            # Verify scheduler steps for sample-based training
            assert container.scheduler.lr_decay_steps == sched_cfg.lr_decay_samples
            assert container.scheduler.wd_incr_steps == train_cfg.train_samples
            assert container.scheduler.lr_warmup_steps == sched_cfg.lr_warmup_samples
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_sample_based_training_with_warmup_fraction(self):
        """Test sample-based training with lr_warmup_fraction."""
        train_cfg = create_test_training_config(train_samples=10000, train_iters=None, global_batch_size=32)
        sched_cfg = create_test_scheduler_config(
            lr_decay_samples=8000, lr_warmup_fraction=0.1, lr_warmup_samples=0, lr_decay_iters=None, lr_warmup_iters=0
        )

        gpt_model_cfg = create_test_gpt_config()
        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1, model_config=gpt_model_cfg, train_config=train_cfg, scheduler_config=sched_cfg
        )

        try:
            container.validate()
            # Verify warmup steps calculated from fraction of decay steps (sample count)
            expected_lr_warmup_steps = sched_cfg.lr_warmup_fraction * sched_cfg.lr_decay_samples
            assert container.scheduler.lr_warmup_steps == expected_lr_warmup_steps
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_training_mode_mutual_exclusivity(self):
        """Test that train_iters and train_samples cannot both be specified."""
        train_cfg = create_test_training_config(train_iters=1000, train_samples=10000)

        gpt_model_cfg = create_test_gpt_config()
        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1, model_config=gpt_model_cfg, train_config=train_cfg
        )

        try:
            with pytest.raises(AssertionError, match="Cannot specify more than one"):
                container.validate()
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_training_mode_required(self):
        """Test that either train_iters or train_samples must be specified."""
        train_cfg = create_test_training_config(train_iters=None)
        # train_samples defaults to None

        gpt_model_cfg = create_test_gpt_config()
        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1, model_config=gpt_model_cfg, train_config=train_cfg
        )

        try:
            with pytest.raises(
                AssertionError, match="One of train_iters, train_samples, or num_epochs must be provided"
            ):
                container.validate()
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_sample_based_scheduler_field_validation(self):
        """Test that sample-based training rejects iteration-based scheduler fields."""
        train_cfg = create_test_training_config(train_samples=10000, train_iters=None)
        sched_cfg = create_test_scheduler_config(lr_decay_iters=500)  # Should not be used with sample-based

        gpt_model_cfg = create_test_gpt_config()
        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1, model_config=gpt_model_cfg, train_config=train_cfg, scheduler_config=sched_cfg
        )

        try:
            with pytest.raises(
                AssertionError, match="Use lr_decay_samples for sample-based training, not lr_decay_iters"
            ):
                container.validate()
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_sample_based_training_rejects_scheduler_max_steps(self):
        """scheduler.max_steps applies only to iteration-based training."""
        train_cfg = create_test_training_config(train_samples=10000, train_iters=None)
        sched_cfg = create_test_scheduler_config(max_steps=1000)
        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=create_test_gpt_config(),
            train_config=train_cfg,
            scheduler_config=sched_cfg,
        )

        try:
            with pytest.raises(AssertionError, match="only supported for iteration-based training"):
                container.validate()
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_iteration_based_scheduler_field_validation(self):
        """Test that iteration-based training rejects sample-based scheduler fields."""
        train_cfg = create_test_training_config(train_iters=1000)
        sched_cfg = create_test_scheduler_config(lr_decay_samples=8000)  # Should not be used with iteration-based

        gpt_model_cfg = create_test_gpt_config()
        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1, model_config=gpt_model_cfg, train_config=train_cfg, scheduler_config=sched_cfg
        )

        try:
            with pytest.raises(
                AssertionError, match="Use lr_decay_iters for iteration-based training, not lr_decay_samples"
            ):
                container.validate()
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_sample_based_warmup_mutual_exclusivity(self):
        """Test mutual exclusivity between lr_warmup_fraction and lr_warmup_samples."""
        train_cfg = create_test_training_config(train_samples=10000, train_iters=None)
        sched_cfg = create_test_scheduler_config(
            lr_warmup_fraction=0.1,
            lr_warmup_samples=1000,  # Both specified - should fail
        )

        gpt_model_cfg = create_test_gpt_config()
        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1, model_config=gpt_model_cfg, train_config=train_cfg, scheduler_config=sched_cfg
        )

        try:
            # This should now fail at scheduler finalize level with detailed field values
            with pytest.raises(
                AssertionError, match="Cannot specify lr_warmup_fraction=0.1 with.*lr_warmup_samples=1000"
            ):
                container.validate()
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_sample_based_with_rampup_batch_size_fails(self):
        """Test that sample-based training with rampup_batch_size raises ValueError."""
        train_cfg = create_test_training_config(train_samples=10000, train_iters=None, rampup_batch_size=[16, 8, 5000])

        gpt_model_cfg = create_test_gpt_config()
        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1, model_config=gpt_model_cfg, train_config=train_cfg
        )

        try:
            with pytest.raises(AssertionError, match="Batch size rampup not supported with sample-based training yet"):
                container.validate()
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_sample_based_lr_decay_samples_defaults(self):
        """Test that lr_decay_samples defaults to train_samples."""
        train_cfg = create_test_training_config(train_samples=10000, train_iters=None)
        sched_cfg = create_test_scheduler_config(lr_decay_samples=None)  # Should default to train_samples

        gpt_model_cfg = create_test_gpt_config()
        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1, model_config=gpt_model_cfg, train_config=train_cfg, scheduler_config=sched_cfg
        )

        try:
            container.validate()
            assert container.scheduler.lr_decay_samples == train_cfg.train_samples
            assert container.scheduler.lr_decay_steps == train_cfg.train_samples
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_sample_based_wsd_decay_steps(self):
        """Test WSD decay steps calculation for sample-based training."""
        train_cfg = create_test_training_config(train_samples=10000, train_iters=None)
        sched_cfg = create_test_scheduler_config(lr_wsd_decay_samples=5000)

        gpt_model_cfg = create_test_gpt_config()
        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1, model_config=gpt_model_cfg, train_config=train_cfg, scheduler_config=sched_cfg
        )

        try:
            container.validate()
            assert container.scheduler.wsd_decay_steps == sched_cfg.lr_wsd_decay_samples
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_sample_based_vs_iteration_based_config_equivalence(self):
        """Test that equivalent sample-based and iteration-based configs produce same scheduler steps."""
        from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing_samples

        # Sample-based config
        sample_train_cfg = create_test_training_config(train_samples=32, train_iters=None, global_batch_size=4)
        sample_optimizer_cfg, sample_scheduler_cfg = distributed_fused_adam_with_cosine_annealing_samples(
            lr_warmup_samples=8,
            lr_decay_samples=24,
            max_lr=1e-3,
        )

        sample_model_cfg = create_test_gpt_config()
        sample_container, og_ws1, cfg_mod1 = create_test_config_container(
            world_size_override=1,
            model_config=sample_model_cfg,
            train_config=sample_train_cfg,
            scheduler_config=sample_scheduler_cfg,
        )

        # Equivalent iteration-based config
        iter_train_cfg = create_test_training_config(train_iters=8, global_batch_size=4)  # 32 samples / 4 batch_size
        iter_scheduler_cfg = create_test_scheduler_config(
            lr_warmup_iters=2,  # 8 samples / 4 batch_size
            lr_decay_iters=6,  # 24 samples / 4 batch_size
        )

        iter_model_cfg = create_test_gpt_config()
        iter_container, og_ws2, cfg_mod2 = create_test_config_container(
            world_size_override=1,
            model_config=iter_model_cfg,
            train_config=iter_train_cfg,
            scheduler_config=iter_scheduler_cfg,
        )

        try:
            # Validate both configurations
            sample_container.validate()
            iter_container.validate()

            # Both should have the same final train_iters
            assert sample_container.train.train_iters == iter_container.train.train_iters == 8

            # Both should have equivalent scheduler steps (different calculation, same result)
            assert sample_container.scheduler.lr_decay_steps == 24  # Direct sample count
            assert iter_container.scheduler.lr_decay_steps == 6 * 4  # lr_decay_iters * global_batch_size = 24
            assert sample_container.scheduler.lr_decay_steps == iter_container.scheduler.lr_decay_steps

            # Both should have equivalent warmup steps
            assert sample_container.scheduler.lr_warmup_steps == 8  # Direct sample count
            assert iter_container.scheduler.lr_warmup_steps == 2 * 4  # lr_warmup_iters * global_batch_size = 8
            assert sample_container.scheduler.lr_warmup_steps == iter_container.scheduler.lr_warmup_steps

        finally:
            restore_get_world_size_safe(og_ws1, cfg_mod1)
            restore_get_world_size_safe(og_ws2, cfg_mod2)

    def test_scheduler_field_mixing_validation(self):
        """Test that mixing iteration-based and sample-based scheduler fields fails in scheduler finalize."""
        # This should fail at the SchedulerConfig.finalize() level, before cross-validation
        sched_cfg = create_test_scheduler_config(
            lr_decay_iters=100,  # iteration-based
            lr_decay_samples=1000,  # sample-based - mixing not allowed
        )

        with pytest.raises(AssertionError, match="Cannot mix iteration-based and sample-based scheduler fields"):
            sched_cfg.finalize()

    def test_scheduler_warmup_fraction_with_iters_validation(self):
        """Test that lr_warmup_fraction with lr_warmup_iters fails in scheduler finalize."""
        sched_cfg = create_test_scheduler_config(
            lr_warmup_fraction=0.1,
            lr_warmup_iters=100,  # Should not be mixed with lr_warmup_fraction
        )

        with pytest.raises(AssertionError, match="Cannot specify lr_warmup_fraction=0.1 with lr_warmup_iters=100"):
            sched_cfg.finalize()

    def test_scheduler_warmup_fraction_with_samples_validation(self):
        """Test that lr_warmup_fraction with lr_warmup_samples fails in scheduler finalize."""
        sched_cfg = create_test_scheduler_config(
            lr_warmup_fraction=0.1,
            lr_warmup_samples=1000,  # Should not be mixed with lr_warmup_fraction
        )

        with pytest.raises(AssertionError, match="Cannot specify lr_warmup_fraction=0.1 with.*lr_warmup_samples=1000"):
            sched_cfg.finalize()


class TestEpochBasedTraining:
    """Tests for epoch-based training configuration and resolution."""

    def test_epoch_based_training_sentinel_is_declared_field(self):
        assert "_train_iters_from_num_epochs" in {field.name for field in fields(TrainingConfig)}

    def test_epoch_based_training_resolves_fractional_epochs(self):
        train_cfg = create_test_training_config(train_iters=None, num_epochs=1.5, global_batch_size=32)
        dataset_cfg = GPTSFTDatasetConfig(dataset_root="/tmp/dataset", seq_length=512)
        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=create_test_gpt_config(),
            train_config=train_cfg,
            dataset_config_override=dataset_cfg,
        )

        try:
            container.validate()
            assert container.train.train_iters is None

            container._resolve_num_epochs(train_dataset_size=100)

            assert container.train.train_iters == 6
            assert container.train._train_iters_from_num_epochs is True
            container.train.finalize()
            assert container.scheduler.lr_decay_iters == 6
            assert container.scheduler.lr_decay_steps == 192
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    @pytest.mark.parametrize(
        "training_overrides",
        [
            {"train_iters": 10, "num_epochs": 1.0},
            {"train_iters": None, "train_samples": 100, "num_epochs": 1.0},
        ],
    )
    def test_epoch_based_training_rejects_other_training_modes(self, training_overrides):
        train_cfg = create_test_training_config(**training_overrides)

        with pytest.raises(AssertionError, match="Cannot specify more than one"):
            train_cfg.finalize()

    def test_epoch_based_training_rejects_rampup_batch_size(self):
        train_cfg = create_test_training_config(
            train_iters=None,
            num_epochs=1.0,
            rampup_batch_size=[16, 8, 5000],
        )

        with pytest.raises(AssertionError, match="Batch size rampup not supported with epoch-based training"):
            train_cfg.finalize()

    @pytest.mark.parametrize("num_epochs", [0.0, -1.0])
    def test_epoch_based_training_rejects_non_positive_num_epochs(self, num_epochs):
        train_cfg = create_test_training_config(train_iters=None, num_epochs=num_epochs)

        with pytest.raises(AssertionError, match="num_epochs must be a positive number"):
            train_cfg.finalize()

    def test_epoch_based_training_requires_finite_gpt_sft_dataset(self):
        train_cfg = create_test_training_config(train_iters=None, num_epochs=1.0)
        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=create_test_gpt_config(),
            train_config=train_cfg,
        )

        try:
            with pytest.raises(ValueError, match="num_epochs is only supported for finite GPTSFTDatasetConfig"):
                container.validate()
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_epoch_based_training_rejects_direct_hf_sft_dataset(self):
        train_cfg = create_test_training_config(train_iters=None, num_epochs=1.0)
        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=create_test_gpt_config(),
            train_config=train_cfg,
            dataset_config_override=create_test_direct_hf_sft_dataset_config(sequence_length=512),
        )

        try:
            with pytest.raises(ValueError, match="num_epochs is only supported for finite GPTSFTDatasetConfig"):
                container.validate()
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    @pytest.mark.parametrize("dataloader_type", ["single", "cyclic"])
    def test_epoch_based_training_rejects_non_batch_dataloader(self, dataloader_type):
        train_cfg = create_test_training_config(train_iters=None, num_epochs=1.0)
        dataset_cfg = GPTSFTDatasetConfig(
            dataset_root="/tmp/dataset",
            seq_length=512,
            dataloader_type=dataloader_type,
        )
        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=create_test_gpt_config(),
            train_config=train_cfg,
            dataset_config_override=dataset_cfg,
        )

        try:
            with pytest.raises(ValueError, match='dataloader_type="batch"'):
                container.validate()
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_megatron_mimo_runtime_config_update_rejects_num_epochs(self, monkeypatch):
        cfg = MagicMock()
        cfg.env_vars = {"TORCHINDUCTOR_WORKER_START": "fork"}
        cfg.train.num_epochs = 1.0
        monkeypatch.delenv("TORCHINDUCTOR_WORKER_START", raising=False)

        with pytest.raises(ValueError, match="num_epochs is not supported for MegatronMIMO datasets"):
            megatron_mimo_runtime_config_update(cfg)
        assert os.environ["TORCHINDUCTOR_WORKER_START"] == "fork"


class TestDatasetSequenceLengthValidation:
    """Tests for dataset sequence length validation with different dataset types."""

    def test_custom_dataset_provider_without_seq_length_passes(self, monkeypatch):
        """Test that a custom DatasetProvider without seq_length passes validation."""
        from dataclasses import dataclass
        from typing import Any, Optional, Tuple

        from megatron.bridge.training.config import DatasetBuildContext, DatasetProvider

        @dataclass
        class CustomDatasetProvider(DatasetProvider):
            """Custom dataset provider without seq_length attribute."""

            data_path: str = "/path/to/data"

            def build_datasets(
                self, context: DatasetBuildContext
            ) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
                # Mock implementation
                return None, None, None

        gpt_model_cfg = create_test_gpt_config(seq_length=512)
        custom_dataset = CustomDatasetProvider()

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=gpt_model_cfg,
            dataset_config_override=custom_dataset,
        )

        try:
            # Should pass without trying to access seq_length.
            container.validate()
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_gpt_dataset_sequence_length_mismatch_fails(self, monkeypatch):
        """Test that GPTDatasetConfig with mismatched sequence length fails validation."""
        gpt_model_cfg = create_test_gpt_config(seq_length=512)
        dataset_cfg = create_test_gpt_dataset_config(sequence_length=1024)  # Mismatch!

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=gpt_model_cfg,
            dataset_config_override=dataset_cfg,
        )

        try:
            with pytest.raises(
                AssertionError, match="sequence length configuration in model config and dataset config match"
            ):
                container.validate()
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_gpt_dataset_sequence_length_match_passes(self, monkeypatch):
        """Test that GPTDatasetConfig with matching sequence length passes validation."""
        gpt_model_cfg = create_test_gpt_config(seq_length=512)
        dataset_cfg = create_test_gpt_dataset_config(sequence_length=512)  # Match!

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=gpt_model_cfg,
            dataset_config_override=dataset_cfg,
        )

        try:
            container.validate()  # Should pass
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_gpt_sft_dataset_sequence_length_mismatch_fails(self, monkeypatch):
        """Test that GPTSFTDatasetConfig with mismatched sequence length fails validation."""
        gpt_model_cfg = create_test_gpt_config(seq_length=512)
        dataset_cfg = create_test_gpt_sft_dataset_config(sequence_length=1024)  # Mismatch!

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=gpt_model_cfg,
            dataset_config_override=dataset_cfg,
        )

        try:
            with pytest.raises(
                AssertionError, match="sequence length configuration in model config and dataset config match"
            ):
                container.validate()
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_gpt_sft_dataset_sequence_length_match_passes(self, monkeypatch):
        """Test that GPTSFTDatasetConfig with matching sequence length passes validation."""
        gpt_model_cfg = create_test_gpt_config(seq_length=512)
        dataset_cfg = create_test_gpt_sft_dataset_config(sequence_length=512)  # Match!

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=gpt_model_cfg,
            dataset_config_override=dataset_cfg,
        )

        try:
            container.validate()  # Should pass
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_direct_hf_sft_dataset_sequence_length_mismatch_fails(self, monkeypatch):
        """Test that direct HF conversation configs enforce model sequence length."""
        gpt_model_cfg = create_test_gpt_config(seq_length=512)
        dataset_cfg = create_test_direct_hf_sft_dataset_config(sequence_length=1024)

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=gpt_model_cfg,
            dataset_config_override=dataset_cfg,
        )

        try:
            with pytest.raises(
                AssertionError, match="sequence length configuration in model config and dataset config match"
            ):
                container.validate()
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_direct_hf_sft_dataset_sequence_length_match_passes(self, monkeypatch):
        """Test that direct HF conversation configs accept matching sequence length."""
        gpt_model_cfg = create_test_gpt_config(seq_length=512)
        dataset_cfg = create_test_direct_hf_sft_dataset_config(sequence_length=512)

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=gpt_model_cfg,
            dataset_config_override=dataset_cfg,
        )

        try:
            container.validate()
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_custom_dataset_provider_with_seq_length_validates(self, monkeypatch):
        """Test that GPTSFTDatasetConfig subclasses retain sequence-length validation."""
        from dataclasses import dataclass

        @dataclass
        class CustomGPTSFTDatasetConfig(GPTSFTDatasetConfig):
            """Custom GPT SFT dataset that extends GPTSFTDatasetConfig."""

            custom_field: str = "custom"

        gpt_model_cfg = create_test_gpt_config(seq_length=512)
        custom_dataset = CustomGPTSFTDatasetConfig(
            seq_length=1024,
            dataset_root="/tmp/dataset",
        )

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=gpt_model_cfg,
            dataset_config_override=custom_dataset,
        )

        try:
            # Should still validate sequence length since it's a GPTSFTDatasetConfig.
            with pytest.raises(
                AssertionError, match="sequence length configuration in model config and dataset config match"
            ):
                container.validate()
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)


@pytest.mark.unit
class TestLoggerConfigFinalize:
    """Tests for LoggerConfig.finalize() method."""

    def test_finalize_no_mlflow_settings(self):
        """Test finalize succeeds when no MLFlow settings are configured."""
        config = LoggerConfig()
        # Should not raise
        config.finalize()

    def test_finalize_with_mlflow_experiment_only_raises_error(self):
        """Test finalize raises error when mlflow_experiment is set but mlflow_run_name is missing."""
        config = LoggerConfig(mlflow_experiment="my_experiment")

        with pytest.raises(ValueError, match="Set logger.mlflow_run_name"):
            config.finalize()

    def test_finalize_with_mlflow_experiment_and_empty_run_name_raises_error(self):
        """Test finalize raises error when mlflow_run_name is empty string."""
        config = LoggerConfig(mlflow_experiment="my_experiment", mlflow_run_name="")

        with pytest.raises(ValueError, match="Set logger.mlflow_run_name"):
            config.finalize()

    def test_finalize_with_mlflow_experiment_and_run_name_succeeds(self):
        """Test finalize succeeds when both mlflow_experiment and mlflow_run_name are set."""
        config = LoggerConfig(mlflow_experiment="my_experiment", mlflow_run_name="my_run")
        # Mock mlflow import to avoid slow actual import
        with patch("importlib.import_module"):
            config.finalize()  # Should not raise

    def test_finalize_mlflow_not_installed_raises_module_not_found(self):
        """Test finalize raises ModuleNotFoundError when mlflow is configured but not installed."""
        config = LoggerConfig(mlflow_experiment="my_experiment", mlflow_run_name="my_run")

        with patch.dict("sys.modules", {"mlflow": None}):
            with patch("importlib.import_module", side_effect=ModuleNotFoundError("No module named 'mlflow'")):
                with pytest.raises(ModuleNotFoundError, match="mlflow"):
                    config.finalize()

    def test_finalize_with_mlflow_tags_only(self):
        """Test finalize with only mlflow_tags triggers MLFlow validation."""
        config = LoggerConfig(mlflow_tags={"env": "test"})

        # mlflow_tags without mlflow_experiment should still try to import mlflow
        # but not require mlflow_run_name since experiment is not set
        # Mock mlflow import to avoid slow actual import
        with patch("importlib.import_module"):
            config.finalize()  # Should not raise

    def test_finalize_with_mlflow_tracking_uri_only(self):
        """Test finalize with only mlflow_tracking_uri triggers MLFlow validation."""
        config = LoggerConfig(mlflow_tracking_uri="http://localhost:5000")

        # Mock mlflow import to avoid slow actual import
        with patch("importlib.import_module"):
            config.finalize()  # Should not raise

    def test_finalize_with_all_mlflow_settings(self):
        """Test finalize with all MLFlow settings configured."""
        config = LoggerConfig(
            mlflow_experiment="my_experiment",
            mlflow_run_name="my_run",
            mlflow_tracking_uri="http://localhost:5000",
            mlflow_tags={"env": "test", "version": "1.0"},
        )

        # Mock mlflow import to avoid slow actual import
        with patch("importlib.import_module"):
            config.finalize()  # Should not raise

    def test_finalize_no_comet_settings(self):
        """Test finalize succeeds when no Comet settings are configured."""
        config = LoggerConfig()
        config.finalize()

    def test_finalize_with_comet_project_only_raises_error(self):
        """Test finalize raises error when comet_project is set but comet_experiment_name is missing."""
        config = LoggerConfig(comet_project="my_project")

        with pytest.raises(ValueError, match="comet_experiment_name"):
            config.finalize()

    def test_finalize_with_comet_project_and_empty_experiment_name_raises_error(self):
        """Test finalize raises error when comet_experiment_name is empty string."""
        config = LoggerConfig(comet_project="my_project", comet_experiment_name="")

        with pytest.raises(ValueError, match="comet_experiment_name"):
            config.finalize()

    def test_finalize_with_comet_project_and_experiment_name_succeeds(self):
        """Test finalize succeeds when both comet_project and comet_experiment_name are set."""
        config = LoggerConfig(comet_project="my_project", comet_experiment_name="my_experiment")
        with patch("importlib.import_module"):
            config.finalize()

    def test_finalize_comet_not_installed_raises_module_not_found(self):
        """Test finalize raises ModuleNotFoundError when comet_ml is configured but not installed."""
        config = LoggerConfig(comet_project="my_project", comet_experiment_name="my_experiment")

        with patch("importlib.import_module", side_effect=ModuleNotFoundError("No module named 'comet_ml'")):
            with pytest.raises(ModuleNotFoundError, match="comet_ml"):
                config.finalize()

    def test_finalize_with_comet_workspace_only(self):
        """Test finalize with only comet_workspace triggers Comet validation."""
        config = LoggerConfig(comet_workspace="my_workspace")
        with patch("importlib.import_module"):
            config.finalize()

    def test_finalize_with_all_comet_settings(self):
        """Test finalize with all Comet settings configured."""
        config = LoggerConfig(
            comet_project="my_project",
            comet_experiment_name="my_experiment",
            comet_workspace="my_workspace",
            comet_api_key="my_key",
            comet_tags=["sft", "qwen3"],
        )
        with patch("importlib.import_module"):
            config.finalize()


class TestTokenizerConfig:
    def test_config_success(self):
        tokenizer_model = "/path/to/tokenizer"
        tokenizer_type = "HuggingFaceTokenizer"
        metadata_path = "/path/to/metadata.json"
        use_fast = False
        legacy = True

        config = TokenizerConfig(
            tokenizer_model=tokenizer_model,
            tokenizer_type=tokenizer_type,
            metadata_path=metadata_path,
            hf_tokenizer_kwargs={"use_fast": use_fast},
            sp_tokenizer_kwargs={"legacy": legacy},
        )

        assert config.tokenizer_model == tokenizer_model
        assert config.metadata_path == metadata_path
        assert config.tokenizer_hf_no_use_fast == (not use_fast)
        assert config.tokenizer_sentencepiece_legacy == legacy

    def test_config_failure(self):
        tokenizer_model = "/path/to/tokenizer"
        tokenizer_type = "HuggingFaceTokenizer"
        metadata_path = "/path/to/metadata.json"

        with pytest.raises(TypeError, match="got an unexpected keyword argument"):
            TokenizerConfig(
                tokenizer_model=tokenizer_model,
                tokenizer_type=tokenizer_type,
                metadata_path=metadata_path,
                random_arg=True,
            )
