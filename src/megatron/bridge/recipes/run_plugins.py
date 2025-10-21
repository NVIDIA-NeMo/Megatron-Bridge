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

"""
This file contains plugins based on NeMo-Run's run.Plugin API.
Plugins operate both on a configured task and an executor at the same time, and are specific to NeMo-Run.
These plugins work by modifying the ConfigContainer configuration overrides.

For run.Script tasks, each plugin supports custom argument conversion via the `script_args_converter_fn`
parameter. This allows users to specify their own conversion function if their training scripts don't
use hydra-style overrides.

Example usage with custom converter:

    from megatron.bridge.recipes.run_plugins import (
        PreemptionPlugin,
        PreemptionPluginScriptArgs,
    )

    # Define a custom converter for argparse-style arguments
    def argparse_preemption_converter(args: PreemptionPluginScriptArgs) -> list[str]:
        result = []
        if args.enable_exit_handler:
            result.append("--enable-exit-handler")
        if args.enable_exit_handler_for_data_loader:
            result.append("--enable-exit-handler-dataloader")
        return result

    # Use the plugin with the custom converter
    plugin = PreemptionPlugin(
        preempt_time=120,
        enable_exit_handler=True,
        script_args_converter_fn=argparse_preemption_converter,
    )

If no converter is provided, the plugin will use the default hydra-style converter.
"""

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

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


def _format_list_for_override(values: list | int):
    """Render a Python list into a Hydra/CLI-safe list string without spaces.

    Example: [0, 3] -> "[0,3]"
    """
    if isinstance(values, int):
        values = [values]
    return "[" + ",".join(str(v) for v in values) + "]"


@dataclass
class PreemptionPluginScriptArgs:
    """Arguments for PreemptionPlugin to pass to run.Script."""

    enable_exit_handler: bool
    enable_exit_handler_for_data_loader: bool


def _default_preemption_converter(args: PreemptionPluginScriptArgs) -> list[str]:
    """Default converter for PreemptionPlugin that generates hydra-style overrides."""
    return [
        f"train.exit_signal_handler={str(args.enable_exit_handler)}",
        f"train.exit_signal_handler_for_dataloader={str(args.enable_exit_handler_for_data_loader)}",
    ]


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
        enable_exit_handler_for_data_loader (bool): Whether to enable the exit signal handler for data loader.
        script_args_converter_fn (Callable | None): A function that takes PreemptionPluginScriptArgs
                                                        and returns a list of CLI arguments. If not provided,
                                                        uses the default hydra-style converter.
    """

    preempt_time: int = 60
    enable_exit_handler: bool = True
    enable_exit_handler_for_data_loader: bool = False
    script_args_converter_fn: Callable[[PreemptionPluginScriptArgs], list[str]] | None = None

    def setup(self, task: "run.Partial" | "run.Script", executor: "run.Executor"):
        if not HAVE_NEMO_RUN:
            raise ImportError(MISSING_NEMO_RUN_MSG)

        """Set up the preemption plugin."""
        if isinstance(task, Script):
            # For run.Script, append CLI overrides to the script arguments
            if self.enable_exit_handler:
                # Create args dataclass
                script_args = PreemptionPluginScriptArgs(
                    enable_exit_handler=self.enable_exit_handler,
                    enable_exit_handler_for_data_loader=self.enable_exit_handler_for_data_loader,
                )

                # Use custom converter or default
                converter = self.script_args_converter_fn or _default_preemption_converter
                cli_overrides = converter(script_args)

                task.args.extend(cli_overrides)
                logger.info(f"{self.__class__.__name__} added CLI overrides: {', '.join(cli_overrides)}")
        else:
            raise NotImplementedError("PreemptionPlugin is only supported for run.Script tasks")

        # Apply signal configuration for both task types when using SlurmExecutor
        if isinstance(executor, SlurmExecutor):
            # Sends a SIGTERM self.preempt_time seconds before hitting time limit
            logger.info(
                f"{self.__class__.__name__} will send a SIGTERM {self.preempt_time} seconds before the job's time limit for your Slurm executor."
            )
            executor.signal = f"TERM@{self.preempt_time}"


@dataclass
class FaultTolerancePluginScriptArgs:
    """Arguments for FaultTolerancePlugin to pass to run.Script."""

    enable_ft_package: bool
    calc_ft_timeouts: bool


def _default_fault_tolerance_converter(args: FaultTolerancePluginScriptArgs) -> list[str]:
    """Default converter for FaultTolerancePlugin that generates hydra-style overrides."""
    return [
        f"ft.enable_ft_package={str(args.enable_ft_package).lower()}",
        f"ft.calc_ft_timeouts={str(args.calc_ft_timeouts).lower()}",
    ]


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
        script_args_converter_fn (Callable | None): A function that takes FaultTolerancePluginScriptArgs
                                                        and returns a list of CLI arguments. If not provided,
                                                        uses the default hydra-style converter.

    Note:
        This plugin is incompatible with NsysPlugin. Nsys profiling cannot be used when fault tolerance
        is enabled.
    """

    enable_ft_package: bool = True
    calc_ft_timeouts: bool = True
    num_in_job_restarts: int = 3
    num_job_retries_on_failure: int = 2
    initial_rank_heartbeat_timeout: int = 1800
    rank_heartbeat_timeout: int = 300
    script_args_converter_fn: Callable[[FaultTolerancePluginScriptArgs], list[str]] | None = None

    def setup(self, task: "run.Partial" | "run.Script", executor: "run.Executor"):
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
            # Create args dataclass
            script_args = FaultTolerancePluginScriptArgs(
                enable_ft_package=self.enable_ft_package,
                calc_ft_timeouts=self.calc_ft_timeouts,
            )

            # Use custom converter or default
            converter = self.script_args_converter_fn or _default_fault_tolerance_converter
            cli_overrides = converter(script_args)

            task.args.extend(cli_overrides)
            logger.info(f"{self.__class__.__name__} added CLI overrides: {', '.join(cli_overrides)}")
        else:
            raise NotImplementedError("FaultTolerancePlugin is only supported for run.Script tasks")


@dataclass
class NsysPluginScriptArgs:
    """Arguments for NsysPlugin to pass to run.Script."""

    profile_step_start: int
    profile_step_end: int
    profile_ranks: list[int]
    record_shapes: bool


def _default_nsys_converter(args: NsysPluginScriptArgs) -> list[str]:
    """Default converter for NsysPlugin that generates hydra-style overrides."""
    return [
        "profiling.use_nsys_profiler=true",
        f"profiling.profile_step_start={args.profile_step_start}",
        f"profiling.profile_step_end={args.profile_step_end}",
        f"profiling.profile_ranks={_format_list_for_override(args.profile_ranks)}",
        f"profiling.record_shapes={str(args.record_shapes).lower()}",
    ]


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
        profile_ranks (list[int] | None): The ranks on which to run the nsys profiling. If not specified,
            profiling will be run on rank 0.
        nsys_trace (list[str] | None): The events to trace during profiling. If not specified,
            'nvtx' and 'cuda' events will be traced.
        record_shapes (bool): Whether to record tensor shapes. Default is False.
        nsys_gpu_metrics (bool): Whether to enable GPU metrics collection. Default is False.
        script_args_converter_fn (Callable | None): A function that takes NsysPluginScriptArgs
                                                        and returns a list of CLI arguments. If not provided,
                                                        uses the default hydra-style converter.

    Note:
        This plugin is incompatible with FaultTolerancePlugin. Nsys profiling cannot be used when
        fault tolerance is enabled, as the profiler interferes with the fault tolerance mechanisms.
    """

    profile_step_start: int
    profile_step_end: int
    profile_ranks: list[int] | None = None
    nsys_trace: list[str] | None = None
    record_shapes: bool = False
    nsys_gpu_metrics: bool = False
    script_args_converter_fn: Callable[[NsysPluginScriptArgs], list[str]] | None = None

    def setup(self, task: "run.Partial" | "run.Script", executor: "run.Executor"):
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
            # Create args dataclass
            script_args = NsysPluginScriptArgs(
                profile_step_start=self.profile_step_start,
                profile_step_end=self.profile_step_end,
                profile_ranks=self.profile_ranks or [0],
                record_shapes=self.record_shapes,
            )

            # Use custom converter or default
            converter = self.script_args_converter_fn or _default_nsys_converter
            cli_overrides = converter(script_args)

            task.args.extend(cli_overrides)
            logger.info(f"{self.__class__.__name__} added CLI overrides: {', '.join(cli_overrides)}")
        else:
            raise NotImplementedError("NsysPlugin is only supported for run.Script tasks")


@dataclass
class PyTorchProfilerPluginScriptArgs:
    """Arguments for PyTorchProfilerPlugin to pass to run.Script."""

    profile_step_start: int
    profile_step_end: int
    profile_ranks: list[int]
    record_memory_history: bool
    memory_snapshot_path: str
    record_shapes: bool


def _default_pytorch_profiler_converter(args: PyTorchProfilerPluginScriptArgs) -> list[str]:
    """Default converter for PyTorchProfilerPlugin that generates hydra-style overrides."""
    return [
        "profiling.use_pytorch_profiler=true",
        f"profiling.profile_step_start={args.profile_step_start}",
        f"profiling.profile_step_end={args.profile_step_end}",
        f"profiling.profile_ranks={_format_list_for_override(args.profile_ranks)}",
        f"profiling.record_memory_history={str(args.record_memory_history).lower()}",
        f"profiling.memory_snapshot_path={args.memory_snapshot_path}",
        f"profiling.record_shapes={str(args.record_shapes).lower()}",
    ]


@dataclass(kw_only=True)
class PyTorchProfilerPlugin(Plugin):
    """
    A plugin for PyTorch profiler configuration.

    The PyTorchProfilerPlugin allows you to use the built-in PyTorch profiler
    which can be viewed in TensorBoard.

    Args:
        profile_step_start (int): The step at which to start profiling.
        profile_step_end (int): The step at which to end profiling.
        profile_ranks (list[int] | None): The ranks on which to run the profiling. If not specified,
            profiling will be run on rank 0.
        record_memory_history (bool): Whether to record memory history. Default is False.
        memory_snapshot_path (str): Path to save memory snapshots. Default is "snapshot.pickle".
        record_shapes (bool): Whether to record tensor shapes. Default is False.
        script_args_converter_fn (Callable | None): A function that takes PyTorchProfilerPluginScriptArgs
                                                        and returns a list of CLI arguments. If not provided,
                                                        uses the default hydra-style converter.
    """

    profile_step_start: int
    profile_step_end: int
    profile_ranks: list[int] | None = None
    record_memory_history: bool = False
    memory_snapshot_path: str = "snapshot.pickle"
    record_shapes: bool = False
    script_args_converter_fn: Callable[[PyTorchProfilerPluginScriptArgs], list[str]] | None = None

    def setup(self, task: "run.Partial" | "run.Script", executor: "run.Executor"):
        if not HAVE_NEMO_RUN:
            raise ImportError(MISSING_NEMO_RUN_MSG)

        """Set up the PyTorch profiler plugin."""
        if isinstance(task, Script):
            # For run.Script, append CLI overrides to the script arguments
            # Create args dataclass
            script_args = PyTorchProfilerPluginScriptArgs(
                profile_step_start=self.profile_step_start,
                profile_step_end=self.profile_step_end,
                profile_ranks=self.profile_ranks or [0],
                record_memory_history=self.record_memory_history,
                memory_snapshot_path=self.memory_snapshot_path,
                record_shapes=self.record_shapes,
            )

            # Use custom converter or default
            converter = self.script_args_converter_fn or _default_pytorch_profiler_converter
            cli_overrides = converter(script_args)

            task.args.extend(cli_overrides)
            logger.info(f"{self.__class__.__name__} added CLI overrides: {', '.join(cli_overrides)}")
        else:
            raise NotImplementedError("NsysPlugin is only supported for run.Script tasks")


@dataclass
class WandbPluginScriptArgs:
    """Arguments for WandbPlugin to pass to run.Script."""

    project: str
    entity: str | None
    name: str | None
    save_dir: str


def _default_wandb_converter(args: WandbPluginScriptArgs) -> list[str]:
    """Default converter for WandbPlugin that generates hydra-style overrides."""
    cli_overrides = [f"logger.wandb_project={args.project}"]
    if args.entity:
        cli_overrides.append(f"logger.wandb_entity={args.entity}")
    if args.name:
        cli_overrides.append(f"logger.wandb_exp_name={args.name}")
    cli_overrides.append(f"logger.wandb_save_dir={args.save_dir}")
    return cli_overrides


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
        name (str | None): The name for the Weights & Biases run. If not provided, uses experiment name.
        entity (str | None): The Weights & Biases entity name.
        save_dir (str): Directory to save wandb logs. Default is "/nemo_run/wandb".
        log_task_config (bool, optional): Whether to log the task configuration to wandb.
            Defaults to True.
        script_args_converter_fn (Callable | None): A function that takes WandbPluginScriptArgs
                                                        and returns a list of CLI arguments. If not provided,
                                                        uses the default hydra-style converter.
    """

    project: str
    name: str | None = None
    entity: str | None = None
    save_dir: str = "/nemo_run/wandb"
    log_task_config: bool = True
    script_args_converter_fn: Callable[[WandbPluginScriptArgs], list[str]] | None = None

    def setup(self, task: "run.Partial" | "run.Script", executor: "run.Executor"):
        if not HAVE_NEMO_RUN:
            raise ImportError(MISSING_NEMO_RUN_MSG)

        """Set up the wandb plugin."""
        if "WANDB_API_KEY" in os.environ:
            executor.env_vars["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]

            if isinstance(task, Script):
                # For run.Script, append CLI overrides to the script arguments
                # Create args dataclass
                script_args = WandbPluginScriptArgs(
                    project=self.project,
                    entity=self.entity,
                    name=self.name,
                    save_dir=self.save_dir,
                )

                # Use custom converter or default
                converter = self.script_args_converter_fn or _default_wandb_converter
                cli_overrides = converter(script_args)

                task.args.extend(cli_overrides)
                logger.info(f"{self.__class__.__name__} added CLI overrides: {', '.join(cli_overrides)}")
            else:
                raise NotImplementedError("WandbPlugin is only supported for run.Script tasks")
        else:
            logger.warning(
                f"Warning: The {self.__class__.__name__} will have no effect as WANDB_API_KEY environment variable is not set."
            )


@dataclass
class PerfEnvPluginScriptArgs:
    """Arguments for PerfEnvPlugin to pass to run.Script."""

    enable_manual_gc: bool
    manual_gc_interval: int


def _default_perf_env_converter(args: PerfEnvPluginScriptArgs) -> list[str]:
    """Default converter for PerfEnvPlugin that generates hydra-style overrides."""
    return [
        f"train.manual_gc={str(args.enable_manual_gc).lower()}",
        f"train.manual_gc_interval={args.manual_gc_interval}",
    ]


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
        nccl_pp_comm_chunksize (int | None): Chunk size for P2P communications.
        gpu_sm100_or_newer (bool): Whether GPU is SM100 or newer architecture.
        enable_manual_gc (bool): Enable manual garbage collection for better performance.
        manual_gc_interval (int): Interval for manual garbage collection. Default is 100.
        tp_size (int): Tensor parallelism size. Default is 1.
        cp_size (int): Context parallelism size. Default is 1.
        pp_size (int): Pipeline parallelism size. Default is 1.
        script_args_converter_fn (Callable | None): A function that takes PerfEnvPluginScriptArgs
                                                        and returns a list of CLI arguments. If not provided,
                                                        uses the default hydra-style converter.
    """

    enable_layernorm_sm_margin: bool = True
    layernorm_sm_margin: int = 16
    enable_vboost: bool = False
    nccl_pp_comm_chunksize: int | None = None
    gpu_sm100_or_newer: bool = False
    enable_manual_gc: bool = True
    manual_gc_interval: int = 100
    tp_size: int = 1
    cp_size: int = 1
    pp_size: int = 1
    script_args_converter_fn: Callable[[PerfEnvPluginScriptArgs], list[str]] | None = None
    num_gpus: int = 8
    deepep_enabled: bool = False
    a2a_overlap: bool = False

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

    def _set_num_cuda_device_max_connections(self, task: "run.Partial" | "run.Script", executor: "run.Executor"):
        self.dp_size = self.num_gpus // (self.tp_size * self.cp_size * self.pp_size)

        cuda_device_max_connections = 8
        if self.deepep_enabled:
            cuda_device_max_connections = 32
        if self.gpu_sm100_or_newer:
            if (self.tp_size > 1 or self.cp_size > 1) and (self.dp_size > 1 or self.pp_size > 1):
                """
                We need extra connections to avoid serialization of streams, so we use max connections of 32 instead
                of the default device connection of 8.
                """
                cuda_device_max_connections = 32
        else:
            # Hopper or earlier generation GPUs
            if (self.tp_size > 1 or self.cp_size > 1) and not self.a2a_overlap:
                """
                Set the device connection to 1 to enforce kernel queuing order from host to execution order on GPU.
                This is needed to schedule a communication kernel before the overlapping persistent GEMM kernel.
                Otherwise, communication kernel will be pushed to the end of the GEMM kernel, failing to overlap the
                kernels.
                """
                cuda_device_max_connections = 1

        executor.env_vars["CUDA_DEVICE_MAX_CONNECTIONS"] = str(cuda_device_max_connections)
        logger.info(f"Set CUDA_DEVICE_MAX_CONNECTIONS to {cuda_device_max_connections}")

    def setup(self, task: "run.Partial" | "run.Script", executor: "run.Executor"):
        """Enable the performance environment settings"""

        if not HAVE_NEMO_RUN:
            raise ImportError(MISSING_NEMO_RUN_MSG)

        # Force program order kernel launch for TP, CP overlap
        self._set_num_cuda_device_max_connections(task, executor)

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
                # Create args dataclass
                script_args = PerfEnvPluginScriptArgs(
                    enable_manual_gc=self.enable_manual_gc,
                    manual_gc_interval=self.manual_gc_interval,
                )

                # Use custom converter or default
                converter = self.script_args_converter_fn or _default_perf_env_converter
                cli_overrides = converter(script_args)

                task.args.extend(cli_overrides)
                logger.info(f"{self.__class__.__name__} added CLI overrides: {', '.join(cli_overrides)}")
            elif hasattr(task, "config"):
                raise NotImplementedError("PerfEnvPlugin is only supported for run.Script tasks")

        # Improve perf by steering power to tensor cores, may not work on all systems
        if self.enable_vboost and isinstance(executor, SlurmExecutor):
            vboost_cmd = self.get_vboost_srun_cmd(executor.nodes, executor.tunnel.job_dir)
            executor.setup_lines = (
                executor.setup_lines + vboost_cmd
                if (executor.setup_lines and len(executor.setup_lines) > 0)
                else vboost_cmd
            )
