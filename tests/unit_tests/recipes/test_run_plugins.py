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
from unittest.mock import MagicMock, patch

import pytest
import torch


try:
    import nemo_run as run

    HAS_NEMO_RUN = True
except ImportError:
    HAS_NEMO_RUN = False

if HAS_NEMO_RUN:
    from megatron.bridge.recipes.run_plugins import (
        FaultTolerancePlugin,
        FaultTolerancePluginScriptArgs,
        NsysPlugin,
        NsysPluginScriptArgs,
        PerfEnvPlugin,
        PerfEnvPluginScriptArgs,
        PreemptionPlugin,
        PreemptionPluginScriptArgs,
        PyTorchProfilerPlugin,
        PyTorchProfilerPluginScriptArgs,
        WandbPlugin,
        WandbPluginScriptArgs,
    )
    from megatron.bridge.recipes.utils.nemo_run_utils import prepare_config_for_nemo_run
    from megatron.bridge.training.config import ProfilingConfig


def create_test_config(**kwargs):
    """Create a test config that works without TransformerEngine/Apex dependencies."""
    from megatron.core.distributed import DistributedDataParallelConfig
    from megatron.core.optimizer import OptimizerConfig

    from megatron.bridge.models.llama import Llama3ModelProvider8B
    from megatron.bridge.training.config import (
        CheckpointConfig,
        ConfigContainer,
        GPTDatasetConfig,
        LoggerConfig,
        RNGConfig,
        SchedulerConfig,
        TokenizerConfig,
        TrainingConfig,
    )

    # Extract model-specific args
    tensor_parallelism = kwargs.pop("tensor_parallelism", 1)
    pipeline_parallelism = kwargs.pop("pipeline_parallelism", 1)
    pipeline_parallelism_dtype = kwargs.pop(
        "pipeline_parallelism_dtype", torch.float32 if pipeline_parallelism > 1 else None
    )
    virtual_pipeline_parallelism = kwargs.pop("virtual_pipeline_parallelism", None)
    context_parallelism = kwargs.pop("context_parallelism", 2)
    sequence_parallelism = kwargs.pop("sequence_parallelism", False)

    # Extract training args with defaults
    train_iters = kwargs.pop("train_iters", 100)
    global_batch_size = kwargs.pop("global_batch_size", 32)
    micro_batch_size = kwargs.pop("micro_batch_size", 1)
    seq_length = kwargs.pop("seq_length", 512)
    lr = kwargs.pop("lr", 1e-4)
    min_lr = kwargs.pop("min_lr", 1e-5)

    # Create model config with apply_rope_fusion=False
    model_cfg = Llama3ModelProvider8B(
        apply_rope_fusion=False,  # Disable to avoid TE/Apex requirement
        tensor_model_parallel_size=tensor_parallelism,
        pipeline_model_parallel_size=pipeline_parallelism,
        pipeline_dtype=pipeline_parallelism_dtype,
        virtual_pipeline_model_parallel_size=virtual_pipeline_parallelism,
        context_parallel_size=context_parallelism,
        sequence_parallel=sequence_parallelism,
    )

    # Create a minimal ConfigContainer
    config = ConfigContainer(
        model=model_cfg,
        train=TrainingConfig(
            train_iters=train_iters,
            eval_interval=2000,
            eval_iters=32,
            global_batch_size=global_batch_size,
            micro_batch_size=micro_batch_size,
            exit_signal_handler=False,
            exit_signal_handler_for_dataloader=False,
            manual_gc=False,
            manual_gc_interval=100,
        ),
        optimizer=OptimizerConfig(
            optimizer="adam",
            lr=lr,
            min_lr=min_lr,
            weight_decay=0.1,
            bf16=True,
            fp16=False,
            adam_beta1=0.9,
            adam_beta2=0.95,
            adam_eps=1e-5,
            use_distributed_optimizer=True,
            clip_grad=1.0,
        ),
        scheduler=SchedulerConfig(
            start_weight_decay=0.033,
            end_weight_decay=0.033,
            weight_decay_incr_style="constant",
            lr_decay_style="cosine",
            lr_warmup_iters=2000,
            lr_warmup_init=0.0,
            lr_decay_iters=train_iters,
            override_opt_param_scheduler=True,
        ),
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            average_in_collective=True,
            use_distributed_optimizer=True,
        ),
        dataset=GPTDatasetConfig(
            random_seed=1234,
            reset_attention_mask=False,
            reset_position_ids=False,
            eod_mask_loss=False,
            sequence_length=seq_length,
            num_dataset_builder_threads=1,
            blend=None,  # Mock data
            blend_per_split=None,
            split="1,1,1",
            data_sharding=True,
            dataloader_type="single",
            num_workers=1,
        ),
        logger=LoggerConfig(
            log_interval=10,
            tensorboard_dir="/tmp/tb_logs",
            wandb_project=None,
            wandb_entity=None,
            wandb_exp_name="test",
            wandb_save_dir="/tmp/wandb",
        ),
        tokenizer=TokenizerConfig(tokenizer_type="NullTokenizer"),
        checkpoint=CheckpointConfig(
            save_interval=2000,
            save="/tmp/checkpoints",
            ckpt_format="torch_dist",
            fully_parallel_save=True,
            async_save=True,
        ),
        rng=RNGConfig(seed=1234),
    )

    # Don't include profiling config by default - let plugins set it up
    # This ensures tests can properly test plugin behavior
    return config


@pytest.mark.skipif(not HAS_NEMO_RUN, reason="nemo_run not installed")
class TestPreemptionPlugin:
    """Test PreemptionPlugin functionality."""

    def test_default_initialization(self):
        """Test plugin initialization with default values."""
        plugin = PreemptionPlugin()
        assert plugin.preempt_time == 60
        assert plugin.enable_exit_handler is True
        assert plugin.enable_exit_handler_for_data_loader is False

    def test_setup_with_partial_task(self):
        """Test setup with run.Partial task."""
        plugin = PreemptionPlugin(enable_exit_handler=True, enable_exit_handler_for_data_loader=True)

        # Create a config using llama3_8b recipe
        config = create_test_config(train_iters=100)
        prepared_config = prepare_config_for_nemo_run(config)

        # Create mock task and executor
        task = MagicMock(spec=run.Partial)
        task.config = prepared_config
        executor = MagicMock(spec=run.SlurmExecutor)

        # Run setup
        plugin.setup(task, executor)

        # Verify config was modified
        assert task.config.train.exit_signal_handler is True
        assert task.config.train.exit_signal_handler_for_dataloader is True

        # Verify SLURM signal was set
        assert executor.signal == "TERM@60"

    def test_setup_with_script_task(self):
        """Test setup with run.Script task."""
        plugin = PreemptionPlugin(preempt_time=120)

        # Create mock script task
        task = MagicMock(spec=run.Script)
        task.args = []

        # Create mock executor
        executor = MagicMock(spec=run.SlurmExecutor)

        # Run setup
        plugin.setup(task, executor)

        # Verify CLI overrides were added
        assert "train.exit_signal_handler=True" in task.args
        assert "train.exit_signal_handler_for_dataloader=False" in task.args

        # Verify SLURM signal was set with custom preempt time
        assert executor.signal == "TERM@120"

    def test_setup_with_non_slurm_executor(self):
        """Test setup with non-SLURM executor."""
        plugin = PreemptionPlugin()

        # Create config and task
        config = create_test_config()
        task = MagicMock(spec=run.Partial)
        task.config = prepare_config_for_nemo_run(config)

        # Create non-SLURM executor
        executor = MagicMock()  # Not a SlurmExecutor

        # Run setup
        plugin.setup(task, executor)

        # Verify signal was NOT set (since it's not SLURM)
        # The signal attribute should not be set on non-SLURM executors
        # But MagicMock might have it as a mock attribute, so check if it was actually set by the plugin
        if hasattr(executor, "signal"):
            # If it exists, it should be a Mock object, not an actual string
            assert isinstance(executor.signal, MagicMock)

    def test_custom_script_args_converter(self):
        """Test setup with custom script args converter."""

        # Define a custom converter that uses argparse-style arguments
        def custom_converter(args: PreemptionPluginScriptArgs):
            result = []
            if args.enable_exit_handler:
                result.append("--enable-exit-handler")
            if args.enable_exit_handler_for_data_loader:
                result.append("--enable-exit-handler-dataloader")
            return result

        plugin = PreemptionPlugin(
            enable_exit_handler=True,
            enable_exit_handler_for_data_loader=True,
            script_args_converter_fn=custom_converter,
        )

        # Create mock script task
        task = MagicMock(spec=run.Script)
        task.args = []

        # Create mock executor
        executor = MagicMock(spec=run.SlurmExecutor)

        # Run setup
        plugin.setup(task, executor)

        # Verify custom converter was used
        assert "--enable-exit-handler" in task.args
        assert "--enable-exit-handler-dataloader" in task.args
        # Verify hydra-style args are NOT present
        assert "train.exit_signal_handler=True" not in task.args

    def test_default_converter_used_when_none_provided(self):
        """Test that default converter is used when script_args_converter_fn is None."""
        plugin = PreemptionPlugin(enable_exit_handler=True, script_args_converter_fn=None)

        # Create mock script task
        task = MagicMock(spec=run.Script)
        task.args = []

        # Create mock executor
        executor = MagicMock()

        # Run setup
        plugin.setup(task, executor)

        # Verify default hydra-style args are present
        assert "train.exit_signal_handler=True" in task.args
        assert "train.exit_signal_handler_for_dataloader=False" in task.args


@pytest.mark.skipif(not HAS_NEMO_RUN, reason="nemo_run not installed")
class TestFaultTolerancePlugin:
    """Test FaultTolerancePlugin functionality."""

    def test_default_initialization(self):
        """Test plugin initialization with default values."""
        plugin = FaultTolerancePlugin()
        assert plugin.enable_ft_package is True
        assert plugin.calc_ft_timeouts is True
        assert plugin.num_in_job_restarts == 3
        assert plugin.num_job_retries_on_failure == 2
        assert plugin.initial_rank_heartbeat_timeout == 1800
        assert plugin.rank_heartbeat_timeout == 300

    def test_setup_with_partial_task(self):
        """Test setup with run.Partial task."""
        plugin = FaultTolerancePlugin(
            enable_ft_package=True, calc_ft_timeouts=False, num_in_job_restarts=5, num_job_retries_on_failure=3
        )

        # Create config using llama3_8b recipe
        config = create_test_config()
        prepared_config = prepare_config_for_nemo_run(config)

        # Create mock task and executor
        task = MagicMock(spec=run.Partial)
        task.config = prepared_config

        # Mock executor and launcher
        executor = MagicMock()
        mock_ft_launcher = MagicMock(spec=run.FaultTolerance)

        with patch.object(run, "FaultTolerance", return_value=mock_ft_launcher) as mock_ft_class:
            plugin.setup(task, executor)

        # Verify FaultTolerance launcher was created with correct params
        mock_ft_class.assert_called_once_with(
            max_restarts=5, initial_rank_heartbeat_timeout=1800, rank_heartbeat_timeout=300
        )

        # Verify executor settings
        assert executor.launcher == mock_ft_launcher
        assert executor.retries == 3

        # Verify fault tolerance config was set
        assert hasattr(task.config, "ft")
        assert task.config.ft.enable_ft_package is True
        assert task.config.ft.calc_ft_timeouts is False

    def test_setup_with_script_task(self):
        """Test setup with run.Script task."""
        plugin = FaultTolerancePlugin()

        # Create mock script task
        task = MagicMock(spec=run.Script)
        task.args = []

        # Create mock executor
        executor = MagicMock()
        mock_ft_launcher = MagicMock(spec=run.FaultTolerance)

        with patch.object(run, "FaultTolerance", return_value=mock_ft_launcher):
            plugin.setup(task, executor)

        # Verify CLI overrides were added
        assert "ft.enable_ft_package=true" in task.args
        assert "ft.calc_ft_timeouts=true" in task.args

    def test_nsys_profiler_warning(self):
        """Test that nsys profiler is disabled when fault tolerance is enabled."""
        plugin = FaultTolerancePlugin()

        # Create config without profiling
        config = create_test_config()

        # Remove any existing profiling config
        if hasattr(config, "profiling"):
            delattr(config, "profiling")

        prepared_config = prepare_config_for_nemo_run(config)

        # Add profiling config with nsys enabled
        prepared_config.profiling = ProfilingConfig(use_nsys_profiler=True)

        # Create mock task
        task = MagicMock(spec=run.Partial)
        task.config = prepared_config

        # Create mock executor
        executor = MagicMock()

        with patch.object(run, "FaultTolerance", return_value=MagicMock()):
            # Capture logger warning
            from megatron.bridge.recipes import run_plugins

            with patch.object(run_plugins.logger, "warning") as mock_warning:
                plugin.setup(task, executor)

        # Verify nsys profiler was disabled
        assert task.config.profiling.use_nsys_profiler is False

        # Verify warning was logged
        mock_warning.assert_called_once()
        assert "Nsys not supported with the FaultTolerancePlugin" in mock_warning.call_args[0][0]

    def test_custom_script_args_converter(self):
        """Test setup with custom script args converter."""

        # Define a custom converter
        def custom_converter(args: FaultTolerancePluginScriptArgs):
            return [
                f"--ft-enabled={str(args.enable_ft_package).lower()}",
                f"--ft-calc-timeouts={str(args.calc_ft_timeouts).lower()}",
            ]

        plugin = FaultTolerancePlugin(
            enable_ft_package=True, calc_ft_timeouts=False, script_args_converter_fn=custom_converter
        )

        # Create mock script task
        task = MagicMock(spec=run.Script)
        task.args = []

        # Create mock executor
        executor = MagicMock()

        with patch.object(run, "FaultTolerance", return_value=MagicMock()):
            plugin.setup(task, executor)

        # Verify custom converter was used
        assert "--ft-enabled=true" in task.args
        assert "--ft-calc-timeouts=false" in task.args
        # Verify hydra-style args are NOT present
        assert "ft.enable_ft_package=true" not in task.args


@pytest.mark.skipif(not HAS_NEMO_RUN, reason="nemo_run not installed")
class TestNsysPlugin:
    """Test NsysPlugin functionality."""

    def test_initialization(self):
        """Test plugin initialization."""
        plugin = NsysPlugin(
            profile_step_start=10,
            profile_step_end=20,
            profile_ranks=[0, 1],
            nsys_trace=["nvtx", "cuda", "cudnn"],
            record_shapes=True,
        )
        assert plugin.profile_step_start == 10
        assert plugin.profile_step_end == 20
        assert plugin.profile_ranks == [0, 1]
        assert plugin.nsys_trace == ["nvtx", "cuda", "cudnn"]
        assert plugin.record_shapes is True

    def test_setup_with_partial_task(self):
        """Test setup with run.Partial task."""
        plugin = NsysPlugin(profile_step_start=100, profile_step_end=200)

        # Create config using llama3_8b recipe
        config = create_test_config()
        prepared_config = prepare_config_for_nemo_run(config)

        # Ensure config has profiling attribute (it might not by default)
        if not hasattr(prepared_config, "profiling"):
            prepared_config.profiling = None

        # Create mock task and executor
        task = MagicMock(spec=run.Partial)
        task.config = prepared_config
        # Ensure task has hasattr method that works properly
        task.hasattr = lambda attr: hasattr(task, attr)

        # Create mock executor with launcher
        executor = MagicMock()
        mock_launcher = MagicMock()
        executor.get_launcher.return_value = mock_launcher

        # Run setup
        plugin.setup(task, executor)

        # Verify launcher settings
        assert mock_launcher.nsys_profile is True
        assert mock_launcher.nsys_trace == ["nvtx", "cuda"]

        # Verify profiling config
        assert hasattr(task.config, "profiling")
        assert task.config.profiling is not None
        assert task.config.profiling.use_nsys_profiler is True
        assert task.config.profiling.profile_step_start == 100
        assert task.config.profiling.profile_step_end == 200
        assert task.config.profiling.profile_ranks == [0]
        assert task.config.profiling.record_shapes is False

    def test_setup_with_slurm_executor(self):
        """Test setup with SlurmExecutor sets proper filename."""
        plugin = NsysPlugin(profile_step_start=1, profile_step_end=10)

        # Create config and ensure it has profiling attribute
        config = create_test_config()
        prepared_config = prepare_config_for_nemo_run(config)
        if not hasattr(prepared_config, "profiling"):
            prepared_config.profiling = None

        # Create mock task
        task = MagicMock(spec=run.Partial)
        task.config = prepared_config
        task.hasattr = lambda attr: hasattr(task, attr)

        # Create SLURM executor
        executor = MagicMock(spec=run.SlurmExecutor)
        mock_launcher = MagicMock()
        executor.get_launcher.return_value = mock_launcher

        # Run setup
        plugin.setup(task, executor)

        # Verify SLURM-specific filename was set
        assert mock_launcher.nsys_filename == "profile_%p_%q{SLURM_JOB_ID}_node%q{SLURM_NODEID}_rank%q{SLURM_PROCID}"

    def test_setup_with_script_task(self):
        """Test setup with run.Script task."""
        plugin = NsysPlugin(profile_step_start=50, profile_step_end=100, profile_ranks=[0, 1, 2], record_shapes=True)

        # Create mock script task
        task = MagicMock(spec=run.Script)
        task.args = []

        # Create mock executor
        executor = MagicMock()
        mock_launcher = MagicMock()
        executor.get_launcher.return_value = mock_launcher

        # Run setup
        plugin.setup(task, executor)

        # Verify CLI overrides
        expected_args = [
            "profiling.use_nsys_profiler=true",
            "profiling.profile_step_start=50",
            "profiling.profile_step_end=100",
            "profiling.profile_ranks=[0,1,2]",
            "profiling.record_shapes=true",
        ]
        for arg in expected_args:
            assert arg in task.args

    def test_custom_script_args_converter(self):
        """Test setup with custom script args converter."""

        # Define a custom converter that uses JSON format
        def custom_converter(args: NsysPluginScriptArgs):
            import json

            config = {
                "nsys_enabled": True,
                "start": args.profile_step_start,
                "end": args.profile_step_end,
                "ranks": args.profile_ranks,
                "shapes": args.record_shapes,
            }
            return ["--nsys-config", json.dumps(config)]

        plugin = NsysPlugin(
            profile_step_start=10,
            profile_step_end=20,
            profile_ranks=[0, 1],
            record_shapes=True,
            script_args_converter_fn=custom_converter,
        )

        # Create mock script task
        task = MagicMock(spec=run.Script)
        task.args = []

        # Create mock executor
        executor = MagicMock()
        mock_launcher = MagicMock()
        executor.get_launcher.return_value = mock_launcher

        # Run setup
        plugin.setup(task, executor)

        # Verify custom converter was used
        assert "--nsys-config" in task.args
        # Find the JSON arg
        json_arg = None
        for i, arg in enumerate(task.args):
            if arg == "--nsys-config" and i + 1 < len(task.args):
                json_arg = task.args[i + 1]
                break

        assert json_arg is not None
        import json

        config = json.loads(json_arg)
        assert config["nsys_enabled"] is True
        assert config["start"] == 10
        assert config["end"] == 20
        # Verify hydra-style args are NOT present
        assert "profiling.use_nsys_profiler=true" not in task.args


@pytest.mark.skipif(not HAS_NEMO_RUN, reason="nemo_run not installed")
class TestPyTorchProfilerPlugin:
    """Test PyTorchProfilerPlugin functionality."""

    def test_initialization(self):
        """Test plugin initialization."""
        plugin = PyTorchProfilerPlugin(
            profile_step_start=5,
            profile_step_end=15,
            profile_ranks=[0],
            record_memory_history=True,
            memory_snapshot_path="/tmp/memory.pickle",
            record_shapes=True,
        )
        assert plugin.profile_step_start == 5
        assert plugin.profile_step_end == 15
        assert plugin.profile_ranks == [0]
        assert plugin.record_memory_history is True
        assert plugin.memory_snapshot_path == "/tmp/memory.pickle"
        assert plugin.record_shapes is True

    def test_setup_with_partial_task(self):
        """Test setup with run.Partial task."""
        plugin = PyTorchProfilerPlugin(profile_step_start=10, profile_step_end=20, record_memory_history=True)

        # Create config
        config = create_test_config()
        prepared_config = prepare_config_for_nemo_run(config)

        # Ensure config has profiling attribute
        if not hasattr(prepared_config, "profiling"):
            prepared_config.profiling = None

        # Create mock task
        task = MagicMock(spec=run.Partial)
        task.config = prepared_config
        task.hasattr = lambda attr: hasattr(task, attr)

        # Create mock executor
        executor = MagicMock()

        # Run setup
        plugin.setup(task, executor)

        # Verify profiling config
        assert hasattr(task.config, "profiling")
        assert task.config.profiling is not None
        assert task.config.profiling.use_pytorch_profiler is True
        assert task.config.profiling.profile_step_start == 10
        assert task.config.profiling.profile_step_end == 20
        assert task.config.profiling.profile_ranks == [0]
        assert task.config.profiling.record_memory_history is True
        assert task.config.profiling.memory_snapshot_path == "snapshot.pickle"
        assert task.config.profiling.record_shapes is False

    def test_custom_script_args_converter(self):
        """Test setup with custom script args converter."""

        # Define a custom converter
        def custom_converter(args: PyTorchProfilerPluginScriptArgs):
            return [
                "--pytorch-profiler",
                f"--profile-start={args.profile_step_start}",
                f"--profile-end={args.profile_step_end}",
                f"--profile-ranks={','.join(map(str, args.profile_ranks))}",
                f"--memory-history={'true' if args.record_memory_history else 'false'}",
            ]

        plugin = PyTorchProfilerPlugin(
            profile_step_start=5,
            profile_step_end=15,
            profile_ranks=[0, 1],
            record_memory_history=True,
            script_args_converter_fn=custom_converter,
        )

        # Create mock script task
        task = MagicMock(spec=run.Script)
        task.args = []

        # Create mock executor
        executor = MagicMock()

        # Run setup
        plugin.setup(task, executor)

        # Verify custom converter was used
        assert "--pytorch-profiler" in task.args
        assert "--profile-start=5" in task.args
        assert "--profile-end=15" in task.args
        assert "--profile-ranks=0,1" in task.args
        assert "--memory-history=true" in task.args
        # Verify hydra-style args are NOT present
        assert "profiling.use_pytorch_profiler=true" not in task.args


@pytest.mark.skipif(not HAS_NEMO_RUN, reason="nemo_run not installed")
class TestWandbPlugin:
    """Test WandbPlugin functionality."""

    def test_initialization(self):
        """Test plugin initialization."""
        plugin = WandbPlugin(
            project="test_project",
            name="test_run",
            entity="test_entity",
            save_dir="/custom/wandb",
            log_task_config=False,
        )
        assert plugin.project == "test_project"
        assert plugin.name == "test_run"
        assert plugin.entity == "test_entity"
        assert plugin.save_dir == "/custom/wandb"
        assert plugin.log_task_config is False

    @patch.dict(os.environ, {"WANDB_API_KEY": "test_api_key"})
    def test_setup_with_partial_task_with_api_key(self):
        """Test setup with run.Partial task when WANDB_API_KEY is set."""
        plugin = WandbPlugin(project="llama_training", name="experiment_1", entity="nvidia")

        # Create config
        config = create_test_config()
        prepared_config = prepare_config_for_nemo_run(config)

        # Create mock task
        task = MagicMock(spec=run.Partial)
        task.config = prepared_config

        # Create mock executor
        executor = MagicMock()
        executor.env_vars = {}

        # Run setup
        plugin.setup(task, executor)

        # Verify env var was set
        assert executor.env_vars["WANDB_API_KEY"] == "test_api_key"

        # Verify logger config
        assert task.config.logger.wandb_project == "llama_training"
        assert task.config.logger.wandb_entity == "nvidia"
        assert task.config.logger.wandb_exp_name == "experiment_1"
        assert task.config.logger.wandb_save_dir == "/nemo_run/wandb"

    @patch.dict(os.environ, {}, clear=True)
    def test_setup_without_api_key(self):
        """Test setup when WANDB_API_KEY is not set."""
        plugin = WandbPlugin(project="test_project")

        # Create mock task
        task = MagicMock(spec=run.Partial)
        task.config = prepare_config_for_nemo_run(create_test_config())

        # Create mock executor
        executor = MagicMock()
        executor.env_vars = {}

        # Capture logger warning
        import megatron.bridge.recipes.run_plugins

        with patch.object(megatron.bridge.recipes.run_plugins.logger, "warning") as mock_warning:
            plugin.setup(task, executor)

        # Verify warning was logged
        mock_warning.assert_called_once()
        call_args = mock_warning.call_args[0][0]
        assert "WANDB_API_KEY environment variable is not set" in call_args

        # Verify env var was NOT set
        assert "WANDB_API_KEY" not in executor.env_vars

    @patch.dict(os.environ, {"WANDB_API_KEY": "test_api_key"})
    def test_setup_with_script_task(self):
        """Test setup with run.Script task."""
        plugin = WandbPlugin(
            project="script_project", entity="test_entity", name="script_run", save_dir="/script/wandb"
        )

        # Create mock script task
        task = MagicMock(spec=run.Script)
        task.args = []

        # Create mock executor
        executor = MagicMock()
        executor.env_vars = {}

        # Run setup
        plugin.setup(task, executor)

        # Verify CLI overrides
        expected_args = [
            "logger.wandb_project=script_project",
            "logger.wandb_entity=test_entity",
            "logger.wandb_exp_name=script_run",
            "logger.wandb_save_dir=/script/wandb",
        ]
        for arg in expected_args:
            assert arg in task.args

    @patch.dict(os.environ, {"WANDB_API_KEY": "test_api_key"})
    def test_custom_script_args_converter(self):
        """Test setup with custom script args converter."""

        # Define a custom converter
        def custom_converter(args: WandbPluginScriptArgs):
            result = ["--wandb"]
            result.append(f"--wandb-project={args.project}")
            if args.entity:
                result.append(f"--wandb-entity={args.entity}")
            if args.name:
                result.append(f"--wandb-name={args.name}")
            return result

        plugin = WandbPlugin(
            project="custom_project",
            entity="custom_entity",
            name="custom_run",
            script_args_converter_fn=custom_converter,
        )

        # Create mock script task
        task = MagicMock(spec=run.Script)
        task.args = []

        # Create mock executor
        executor = MagicMock()
        executor.env_vars = {}

        # Run setup
        plugin.setup(task, executor)

        # Verify custom converter was used
        assert "--wandb" in task.args
        assert "--wandb-project=custom_project" in task.args
        assert "--wandb-entity=custom_entity" in task.args
        assert "--wandb-name=custom_run" in task.args
        # Verify hydra-style args are NOT present
        assert "logger.wandb_project=custom_project" not in task.args


@pytest.mark.skipif(not HAS_NEMO_RUN, reason="nemo_run not installed")
class TestPerfEnvPlugin:
    """Test PerfEnvPlugin functionality."""

    def test_initialization(self):
        """Test plugin initialization with default values."""
        plugin = PerfEnvPlugin()
        assert plugin.enable_layernorm_sm_margin is True
        assert plugin.layernorm_sm_margin == 16
        assert plugin.enable_vboost is False
        assert plugin.nccl_pp_comm_chunksize is None
        assert plugin.gpu_sm100_or_newer is False
        assert plugin.enable_manual_gc is True
        assert plugin.manual_gc_interval == 100

    def test_setup_environment_variables(self):
        """Test environment variable setup."""
        plugin = PerfEnvPlugin(
            enable_layernorm_sm_margin=True,
            layernorm_sm_margin=32,
            gpu_sm100_or_newer=True,
            tp_size=4,
            cp_size=2,
            pp_size=4,
            nccl_pp_comm_chunksize=2048,
        )

        # Create mock task and executor
        task = MagicMock(spec=run.Partial)
        task.config = prepare_config_for_nemo_run(create_test_config())

        executor = MagicMock()
        executor.env_vars = {}

        # Run setup
        plugin.setup(task, executor)

        # Verify environment variables
        assert executor.env_vars["CUDA_DEVICE_MAX_CONNECTIONS"] == "32"  # sm100 with tp>1
        assert executor.env_vars["NVTE_FWD_LAYERNORM_SM_MARGIN"] == "32"
        assert executor.env_vars["NVTE_BWD_LAYERNORM_SM_MARGIN"] == "32"
        assert executor.env_vars["NCCL_P2P_NET_CHUNKSIZE"] == "2048"

    def test_setup_with_older_gpu(self):
        """Test setup with older GPU architecture."""
        plugin = PerfEnvPlugin(gpu_sm100_or_newer=False, tp_size=2, cp_size=1)

        # Create mock task and executor
        task = MagicMock(spec=run.Partial)
        task.config = prepare_config_for_nemo_run(create_test_config())

        executor = MagicMock()
        executor.env_vars = {}

        # Run setup
        plugin.setup(task, executor)

        # Verify CUDA_DEVICE_MAX_CONNECTIONS is 1 for older GPUs
        assert executor.env_vars["CUDA_DEVICE_MAX_CONNECTIONS"] == "1"

    def test_manual_gc_config(self):
        """Test manual garbage collection configuration."""
        plugin = PerfEnvPlugin(enable_manual_gc=True, manual_gc_interval=200)

        # Test with Partial task
        config = create_test_config()
        prepared_config = prepare_config_for_nemo_run(config)

        task = MagicMock(spec=run.Partial)
        task.config = prepared_config

        executor = MagicMock()
        executor.env_vars = {}

        plugin.setup(task, executor)

        # Verify manual GC was enabled
        assert task.config.train.manual_gc is True
        assert task.config.train.manual_gc_interval == 200

    def test_vboost_with_slurm_executor(self):
        """Test vboost setup with SlurmExecutor."""
        plugin = PerfEnvPlugin(enable_vboost=True)

        # Create mock task
        task = MagicMock(spec=run.Partial)
        task.config = prepare_config_for_nemo_run(create_test_config())

        # Create SLURM executor
        executor = MagicMock(spec=run.SlurmExecutor)
        executor.env_vars = {}
        executor.nodes = 2
        executor.tunnel = MagicMock()
        executor.tunnel.job_dir = "/job/dir"
        executor.setup_lines = ""

        # Run setup
        plugin.setup(task, executor)

        # Verify vboost command was added to setup lines
        assert "sudo nvidia-smi boost-slider --vboost 1" in executor.setup_lines
        assert "srun" in executor.setup_lines
        assert "--ntasks=2" in executor.setup_lines

    def test_setup_with_script_task(self):
        """Test setup with run.Script task."""
        plugin = PerfEnvPlugin(enable_manual_gc=True, manual_gc_interval=150)

        # Create mock script task
        task = MagicMock(spec=run.Script)
        task.args = []

        # Create mock executor
        executor = MagicMock()
        executor.env_vars = {}

        # Run setup
        plugin.setup(task, executor)

        # Verify CLI overrides for manual GC
        assert "train.manual_gc=true" in task.args
        assert "train.manual_gc_interval=150" in task.args

    def test_custom_script_args_converter(self):
        """Test setup with custom script args converter."""

        # Define a custom converter
        def custom_converter(args: PerfEnvPluginScriptArgs):
            result = []
            if args.enable_manual_gc:
                result.append("--enable-gc")
                result.append(f"--gc-interval={args.manual_gc_interval}")
            return result

        plugin = PerfEnvPlugin(
            enable_manual_gc=True, manual_gc_interval=250, script_args_converter_fn=custom_converter
        )

        # Create mock script task
        task = MagicMock(spec=run.Script)
        task.args = []

        # Create mock executor
        executor = MagicMock()
        executor.env_vars = {}

        # Run setup
        plugin.setup(task, executor)

        # Verify custom converter was used
        assert "--enable-gc" in task.args
        assert "--gc-interval=250" in task.args
        # Verify hydra-style args are NOT present
        assert "train.manual_gc=true" not in task.args


@pytest.mark.skipif(not HAS_NEMO_RUN, reason="nemo_run not installed")
class TestPluginIntegration:
    """Test integration of multiple plugins with llama3_8b recipe."""

    def test_multiple_plugins_with_llama_config(self):
        """Test using multiple plugins together with llama3_8b config."""
        # Create plugins
        preemption_plugin = PreemptionPlugin(preempt_time=300)
        ft_plugin = FaultTolerancePlugin(num_in_job_restarts=5)
        perf_plugin = PerfEnvPlugin(tp_size=2, pp_size=2, enable_manual_gc=True)

        # Create config from llama3_8b recipe
        config = create_test_config(
            name="test_llama_training",
            tensor_parallelism=2,
            pipeline_parallelism=2,
            train_iters=1000,
            global_batch_size=32,
            micro_batch_size=2,
        )
        prepared_config = prepare_config_for_nemo_run(config)

        # Create mock task
        task = MagicMock(spec=run.Partial)
        task.config = prepared_config

        # Create SLURM executor
        executor = MagicMock(spec=run.SlurmExecutor)
        executor.env_vars = {}
        mock_launcher = MagicMock()

        with patch.object(run, "FaultTolerance", return_value=mock_launcher):
            # Apply all plugins
            preemption_plugin.setup(task, executor)
            ft_plugin.setup(task, executor)
            perf_plugin.setup(task, executor)

        # Verify all configurations were applied
        # From preemption plugin
        assert task.config.train.exit_signal_handler is True
        assert executor.signal == "TERM@300"

        # From fault tolerance plugin
        assert hasattr(task.config, "ft")
        assert task.config.ft.enable_ft_package is True
        assert executor.launcher == mock_launcher
        assert executor.retries == 2

        # From perf plugin
        assert task.config.train.manual_gc is True

    def test_script_task_with_multiple_plugins(self):
        """Test multiple plugins with Script task."""
        # Create plugins
        nsys_plugin = NsysPlugin(profile_step_start=100, profile_step_end=200, profile_ranks=[0, 1])
        wandb_plugin = WandbPlugin(project="llama_profiling", name="profile_run")

        # Create script task
        task = MagicMock(spec=run.Script)
        task.args = []

        # Create executor
        executor = MagicMock()
        executor.env_vars = {}
        mock_launcher = MagicMock()
        executor.get_launcher.return_value = mock_launcher

        # Apply plugins
        with patch.dict(os.environ, {"WANDB_API_KEY": "test_key"}):
            nsys_plugin.setup(task, executor)
            wandb_plugin.setup(task, executor)

        # Verify both plugins added their CLI overrides
        assert "profiling.use_nsys_profiler=true" in task.args
        assert "profiling.profile_step_start=100" in task.args
        assert "logger.wandb_project=llama_profiling" in task.args
        assert "logger.wandb_exp_name=profile_run" in task.args

        # Verify wandb env var was set
        assert executor.env_vars["WANDB_API_KEY"] == "test_key"
