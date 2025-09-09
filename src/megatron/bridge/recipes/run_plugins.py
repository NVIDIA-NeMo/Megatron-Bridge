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

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Union

from megatron.bridge.utils.import_utils import MISSING_NEMO_RUN_MSG


try:
    import nemo_run as run
    from nemo_run import Partial, Plugin, Script, SlurmExecutor

    HAVE_NEMO_RUN = True
except (ImportError, ModuleNotFoundError):
    Partial, Plugin, Script, SlurmExecutor = object, object, object, object
    HAVE_NEMO_RUN = False

if TYPE_CHECKING:
    import nemo_run as run


logger: logging.Logger = logging.getLogger(__name__)

# This file contains plugins based on NeMo-Run's run.Plugin API.
# Plugins operate both on a configured task and an executor at the same time, and are specific to NeMo-Run.
# These plugins work by modifying the ConfigContainer configuration overrides.


def _format_list_for_override(values: List | int):
    """Render a Python list into a Hydra/CLI-safe list string without spaces.

    Example: [0, 3] -> "[0,3]"
    """
    if isinstance(values, int):
        values = [values]
    return "[" + ",".join(str(v) for v in values) + "]"


@dataclass(kw_only=True)
class PreemptionPlugin(Plugin):
    """
    A plugin for setting up preemption handling and signals.

    Args:
        preempt_time (int): The time, in seconds, before the task's time limit at which the executor
                             will send a SIGTERM preemption signal. This allows tasks to be gracefully
                             stopped before reaching their time limit, reducing waste and
                             promoting fair resource usage. The default value is 60 seconds (1 minute).
                             This is only supported for ``run.SlurmExecutor``.
        enable_exit_handler (bool): Whether to enable the exit signal handler in training config.
    """

    preempt_time: int = 60
    enable_exit_handler: bool = True
    enable_exit_handler_for_data_loader: bool = False

    def setup(self, task: Union["run.Partial", "run.Script"], executor: "run.Executor"):
        if not HAVE_NEMO_RUN:
            raise ImportError(MISSING_NEMO_RUN_MSG)

        """Set up the preemption plugin."""
        if isinstance(task, Script):
            # For run.Script, append CLI overrides to the script arguments
            if self.enable_exit_handler:
                task.args.append(f"train.exit_signal_handler={str(self.enable_exit_handler)}")
                task.args.append(
                    f"train.exit_signal_handler_for_dataloader={str(self.enable_exit_handler_for_data_loader)}"
                )
                logger.info(
                    f"{self.__class__.__name__} added CLI override: train.exit_signal_handler={str(self.enable_exit_handler)}"
                )
                logger.info(
                    f"{self.__class__.__name__} added CLI override: train.exit_signal_handler_for_dataloader={str(self.enable_exit_handler_for_data_loader)}"
                )
        else:
            # Enable exit signal handler in training config
            if self.enable_exit_handler and hasattr(task, "config"):
                task.config.train.exit_signal_handler = self.enable_exit_handler
                task.config.train.exit_signal_handler_for_dataloader = self.enable_exit_handler_for_data_loader

        # Apply signal configuration for both task types when using SlurmExecutor
        if isinstance(executor, SlurmExecutor):
            # Sends a SIGTERM self.preempt_time seconds before hitting time limit
            logger.info(
                f"{self.__class__.__name__} will send a SIGTERM {self.preempt_time} seconds before the job's time limit for your Slurm executor."
            )
            executor.signal = f"TERM@{self.preempt_time}"


@dataclass(kw_only=True)
class FaultTolerancePlugin(Plugin):
    """
    A plugin for setting up fault tolerance configuration.
    This plugin enables workload hang detection, automatic calculation of timeouts used for hang detection,
    detection of rank(s) terminated due to an error and workload respawning in case of a failure.

    Args:
        enable_ft_package (bool): Enable the fault tolerance package. Default is True.
        calc_ft_timeouts (bool): Automatically compute timeouts. Default is True.
        num_in_job_restarts (int): Max number of restarts on failure, within the same job. Default is 3.
        num_job_retries_on_failure (int): Max number of new job restarts on failure. Default is 2.
        initial_rank_heartbeat_timeout (int): Timeouts are time intervals used by a rank monitor to detect
            that a rank is not alive. This is the max timeout for the initial heartbeat. Default is 1800.
        rank_heartbeat_timeout (int): This is the timeout for subsequent hearbeats after the initial heartbeat.
            Default is 300.
    """

    enable_ft_package: bool = True
    calc_ft_timeouts: bool = True
    num_in_job_restarts: int = 3
    num_job_retries_on_failure: int = 2
    initial_rank_heartbeat_timeout: int = 1800
    rank_heartbeat_timeout: int = 300

    def setup(self, task: Union["run.Partial", "run.Script"], executor: "run.Executor"):
        if not HAVE_NEMO_RUN:
            raise ImportError(MISSING_NEMO_RUN_MSG)

        """Set up the fault tolerance plugin."""
        # Set up fault tolerance launcher for both task types
        executor.launcher = run.FaultTolerance(
            max_restarts=self.num_in_job_restarts,
            initial_rank_heartbeat_timeout=self.initial_rank_heartbeat_timeout,
            rank_heartbeat_timeout=self.rank_heartbeat_timeout,
        )
        executor.retries = self.num_job_retries_on_failure

        if isinstance(task, run.Script):
            # For run.Script, append CLI overrides to the script arguments
            cli_overrides = [
                f"ft.enable_ft_package={str(self.enable_ft_package).lower()}",
                f"ft.calc_ft_timeouts={str(self.calc_ft_timeouts).lower()}",
            ]
            task.args.extend(cli_overrides)
            logger.info(f"{self.__class__.__name__} added CLI overrides: {', '.join(cli_overrides)}")
        else:
            # For run.Partial, modify the task config directly
            # Configure fault tolerance in task config
            if not hasattr(task.config, "ft") or task.config.ft is None:
                from megatron.bridge.training.config import FaultToleranceConfig

                task.config.ft = FaultToleranceConfig()

            task.config.ft.enable_ft_package = self.enable_ft_package
            task.config.ft.calc_ft_timeouts = self.calc_ft_timeouts

            # Check if nsys profiling is enabled and warn if so
            if hasattr(task.config, "profiling") and task.config.profiling and task.config.profiling.use_nsys_profiler:
                logger.warning("Warning: Nsys not supported with the FaultTolerancePlugin.")
                task.config.profiling.use_nsys_profiler = False


@dataclass(kw_only=True)
class NsysPlugin(Plugin):
    """
    A plugin for nsys profiling configuration.

    The NsysPlugin allows you to profile your run using nsys.
    You can specify when to start and end the profiling, on which ranks to run the profiling,
    and what to trace during profiling.

    Args:
        profile_step_start (int): The step at which to start the nsys profiling.
        profile_step_end (int): The step at which to end the nsys profiling.
        profile_ranks (Optional[list[int]]): The ranks on which to run the nsys profiling. If not specified,
            profiling will be run on rank 0.
        nsys_trace (Optional[list[str]]): The events to trace during profiling. If not specified,
            'nvtx' and 'cuda' events will be traced.
        record_shapes (bool): Whether to record tensor shapes. Default is False.
    """

    profile_step_start: int
    profile_step_end: int
    profile_ranks: Optional[list[int]] = None
    nsys_trace: Optional[list[str]] = None
    record_shapes: bool = False
    nsys_gpu_metrics: bool = False

    def setup(self, task: Union["run.Partial", "run.Script"], executor: "run.Executor"):
        if not HAVE_NEMO_RUN:
            raise ImportError(MISSING_NEMO_RUN_MSG)
        """Set up the nsys profiling plugin."""
        launcher = executor.get_launcher()
        launcher.nsys_profile = True
        launcher.nsys_trace = self.nsys_trace or ["nvtx", "cuda"]

        if isinstance(executor, SlurmExecutor):
            # NOTE: DO NOT change to f-string, `%q{}` is Slurm placeholder
            launcher.nsys_filename = "profile_%p_%q{SLURM_JOB_ID}_node%q{SLURM_NODEID}_rank%q{SLURM_PROCID}"

        if self.nsys_gpu_metrics:
            if hasattr(launcher, "nsys_gpu_metrics"):
                launcher.nsys_gpu_metrics = self.nsys_gpu_metrics
            else:
                logger.warning(
                    "Unable to enable nsys gpu metrics collection. Please upgrade Nemo-Run to include commit 70a0df4."
                )

        # Configure profiling in task config
        if isinstance(task, Script):
            # For run.Script, append CLI overrides to the script arguments
            cli_overrides = [
                "profiling.use_nsys_profiler=true",
                f"profiling.profile_step_start={self.profile_step_start}",
                f"profiling.profile_step_end={self.profile_step_end}",
                f"profiling.profile_ranks={_format_list_for_override(self.profile_ranks or [0])}",
                f"profiling.record_shapes={str(self.record_shapes).lower()}",
            ]
            task.args.extend(cli_overrides)
            logger.info(f"{self.__class__.__name__} added CLI overrides: {', '.join(cli_overrides)}")
        elif isinstance(task, Partial):
            # For run.Partial, modify the task config directly
            if not hasattr(task.config, "profiling") or task.config.profiling is None:
                from megatron.bridge.training.config import ProfilingConfig

                task.config.profiling = ProfilingConfig()

            task.config.profiling.use_nsys_profiler = True
            task.config.profiling.profile_step_start = self.profile_step_start
            task.config.profiling.profile_step_end = self.profile_step_end
            task.config.profiling.profile_ranks = self.profile_ranks or [0]
            task.config.profiling.record_shapes = self.record_shapes


@dataclass(kw_only=True)
class PyTorchProfilerPlugin(Plugin):
    """
    A plugin for PyTorch profiler configuration.

    The PyTorchProfilerPlugin allows you to use the built-in PyTorch profiler
    which can be viewed in TensorBoard.

    Args:
        profile_step_start (int): The step at which to start profiling.
        profile_step_end (int): The step at which to end profiling.
        profile_ranks (Optional[list[int]]): The ranks on which to run the profiling. If not specified,
            profiling will be run on rank 0.
        record_memory_history (bool): Whether to record memory history. Default is False.
        memory_snapshot_path (str): Path to save memory snapshots. Default is "snapshot.pickle".
        record_shapes (bool): Whether to record tensor shapes. Default is False.
    """

    profile_step_start: int
    profile_step_end: int
    profile_ranks: Optional[list[int]] = None
    record_memory_history: bool = False
    memory_snapshot_path: str = "snapshot.pickle"
    record_shapes: bool = False

    def setup(self, task: Union["run.Partial", "run.Script"], executor: "run.Executor"):
        if not HAVE_NEMO_RUN:
            raise ImportError(MISSING_NEMO_RUN_MSG)

        """Set up the PyTorch profiler plugin."""
        if isinstance(task, Script):
            # For run.Script, append CLI overrides to the script arguments
            cli_overrides = [
                "profiling.use_pytorch_profiler=true",
                f"profiling.profile_step_start={self.profile_step_start}",
                f"profiling.profile_step_end={self.profile_step_end}",
                f"profiling.profile_ranks={_format_list_for_override(self.profile_ranks or [0])}",
                f"profiling.record_memory_history={str(self.record_memory_history).lower()}",
                f"profiling.memory_snapshot_path={self.memory_snapshot_path}",
                f"profiling.record_shapes={str(self.record_shapes).lower()}",
            ]
            task.args.extend(cli_overrides)
            logger.info(f"{self.__class__.__name__} added CLI overrides: {', '.join(cli_overrides)}")
        else:
            # For run.Partial, modify the task config directly
            # Configure profiling in task config
            if not hasattr(task.config, "profiling") or task.config.profiling is None:
                from megatron.bridge.training.config import ProfilingConfig

                task.config.profiling = ProfilingConfig()

            task.config.profiling.use_pytorch_profiler = True
            task.config.profiling.profile_step_start = self.profile_step_start
            task.config.profiling.profile_step_end = self.profile_step_end
            task.config.profiling.profile_ranks = self.profile_ranks or [0]
            task.config.profiling.record_memory_history = self.record_memory_history
            task.config.profiling.memory_snapshot_path = self.memory_snapshot_path
            task.config.profiling.record_shapes = self.record_shapes


@dataclass(kw_only=True)
class WandbPlugin(Plugin):
    """
    A plugin for setting up Weights & Biases configuration.

    This plugin sets up Weights & Biases logging configuration. The plugin is only activated
    if the ``WANDB_API_KEY`` environment variable is set.
    The ``WANDB_API_KEY`` environment variables will also be set in the executor's environment variables.
    Follow https://docs.wandb.ai/quickstart to retrieve your ``WANDB_API_KEY``.

    Args:
        project (str): The Weights & Biases project name.
        name (Optional[str]): The name for the Weights & Biases run. If not provided, uses experiment name.
        entity (Optional[str]): The Weights & Biases entity name.
        save_dir (str): Directory to save wandb logs. Default is "/nemo_run/wandb".
        log_task_config (bool, optional): Whether to log the task configuration to wandb.
            Defaults to True.
    """

    project: str
    name: Optional[str] = None
    entity: Optional[str] = None
    save_dir: str = "/nemo_run/wandb"
    log_task_config: bool = True

    def setup(self, task: Union["run.Partial", "run.Script"], executor: "run.Executor"):
        if not HAVE_NEMO_RUN:
            raise ImportError(MISSING_NEMO_RUN_MSG)

        """Set up the wandb plugin."""
        if "WANDB_API_KEY" in os.environ:
            executor.env_vars["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]

            if isinstance(task, Script):
                # For run.Script, append CLI overrides to the script arguments
                cli_overrides = [
                    f"logger.wandb_project={self.project}",
                ]
                if self.entity:
                    cli_overrides.append(f"logger.wandb_entity={self.entity}")
                if self.name:
                    cli_overrides.append(f"logger.wandb_exp_name={self.name}")
                cli_overrides.append(f"logger.wandb_save_dir={self.save_dir}")

                task.args.extend(cli_overrides)
                logger.info(f"{self.__class__.__name__} added CLI overrides: {', '.join(cli_overrides)}")
            else:
                # For run.Partial, modify the task config directly
                if hasattr(task, "config"):
                    # Use provided name or fall back to experiment name
                    exp_name = self.name or task.config.logger.wandb_exp_name

                    task.config.logger.wandb_project = self.project
                    task.config.logger.wandb_entity = self.entity
                    task.config.logger.wandb_exp_name = exp_name
                    task.config.logger.wandb_save_dir = self.save_dir
        else:
            logger.warning(
                f"Warning: The {self.__class__.__name__} will have no effect as WANDB_API_KEY environment variable is not set."
            )


@dataclass(kw_only=True)
class PerfEnvPlugin(Plugin):
    """
    A plugin for setting up performance optimized environments.

    Attributes:
        enable_layernorm_sm_margin (bool): Set SM margin for TransformerEngine's Layernorm, so
            in order to not block DP level communication overlap.
        layernorm_sm_margin (int): The SM margin for TransformerEngine Layernorm.
        enable_vboost (bool): Whether to steer more power towards tensor cores via
            `sudo nvidia-smi boost-slider --vboost 1`. May not work on all systems.
        nccl_pp_comm_chunksize (Optional[int]): Chunk size for P2P communications.
        gpu_sm100_or_newer (bool): Whether GPU is SM100 or newer architecture.
        enable_manual_gc (bool): Enable manual garbage collection for better performance.
        manual_gc_interval (int): Interval for manual garbage collection. Default is 100.
    """

    enable_layernorm_sm_margin: bool = True
    layernorm_sm_margin: int = 16
    enable_vboost: bool = False
    nccl_pp_comm_chunksize: Optional[int] = None
    gpu_sm100_or_newer: bool = False
    enable_manual_gc: bool = True
    manual_gc_interval: int = 100
    tp_size: int = 1
    cp_size: int = 1
    pp_size: int = 1

    def get_vboost_srun_cmd(self, nodes, job_dir):
        """Create the vboost `sudo nvidia-smi boost-slider --vboost 1` command"""
        import shlex

        vboost_cmd = " ".join(
            [
                "\n# Command 0: enable vboost\n\n",
                "srun",
                f"--ntasks={nodes}",
                "--output",
                os.path.join(job_dir, "vboost.out"),
                "--error",
                os.path.join(job_dir, "vboost.err"),
                "bash -c ",
                shlex.quote("sudo nvidia-smi boost-slider --vboost 1"),
            ],
        )

        return vboost_cmd

    def setup(self, task: Union["run.Partial", "run.Script"], executor: "run.Executor"):
        """Enable the performance environment settings"""

        if not HAVE_NEMO_RUN:
            raise ImportError(MISSING_NEMO_RUN_MSG)

        # Environment variables work for both task types

        # Force program order kernel launch for TP, CP overlap
        if self.gpu_sm100_or_newer and (self.tp_size > 1 or self.cp_size > 1):
            executor.env_vars["CUDA_DEVICE_MAX_CONNECTIONS"] = "32"
        elif (not self.gpu_sm100_or_newer) and (self.tp_size > 1 or self.cp_size > 1):
            executor.env_vars["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

        # Set LayerNorm SM margin to support the overlap with LayerNorm kernel
        if self.enable_layernorm_sm_margin:
            executor.env_vars["NVTE_FWD_LAYERNORM_SM_MARGIN"] = str(self.layernorm_sm_margin)
            executor.env_vars["NVTE_BWD_LAYERNORM_SM_MARGIN"] = str(self.layernorm_sm_margin)

        # Set the chunk size of P2P communications
        if self.pp_size > 1 and self.nccl_pp_comm_chunksize is not None:
            assert isinstance(self.nccl_pp_comm_chunksize, int) and self.nccl_pp_comm_chunksize > 1
            executor.env_vars["NCCL_P2P_NET_CHUNKSIZE"] = str(self.nccl_pp_comm_chunksize)

        # Configure manual garbage collection
        if self.enable_manual_gc:
            if isinstance(task, Script):
                # For run.Script, append CLI overrides
                cli_overrides = [
                    f"train.manual_gc={str(self.enable_manual_gc).lower()}",
                    f"train.manual_gc_interval={self.manual_gc_interval}",
                ]
                task.args.extend(cli_overrides)
                logger.info(f"{self.__class__.__name__} added CLI overrides: {', '.join(cli_overrides)}")
            elif hasattr(task, "config"):
                # For run.Partial, modify the task config directly
                task.config.train.manual_gc = True
                task.config.train.manual_gc_interval = self.manual_gc_interval

        # Improve perf by steering power to tensor cores, may not work on all systems
        if self.enable_vboost and isinstance(executor, SlurmExecutor):
            vboost_cmd = self.get_vboost_srun_cmd(executor.nodes, executor.tunnel.job_dir)
            executor.setup_lines = (
                executor.setup_lines + vboost_cmd
                if (executor.setup_lines and len(executor.setup_lines) > 0)
                else vboost_cmd
            )
