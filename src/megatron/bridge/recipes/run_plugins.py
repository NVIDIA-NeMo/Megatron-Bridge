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
    def argparse_preemption_converter(args: PreemptionPluginScriptArgs) -> List[str]:
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
from typing import TYPE_CHECKING, Callable, List, Optional, Union

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


def _format_list_for_override(values: List | int):
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


def _default_preemption_converter(args: PreemptionPluginScriptArgs) -> List[str]:
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
        script_args_converter_fn (Optional[Callable]): A function that takes PreemptionPluginScriptArgs
                                                        and returns a list of CLI arguments. If not provided,
                                                        uses the default hydra-style converter.
    """

    preempt_time: int = 60
    enable_exit_handler: bool = True
    enable_exit_handler_for_data_loader: bool = False
    script_args_converter_fn: Optional[Callable[[PreemptionPluginScriptArgs], List[str]]] = None

    def setup(self, task: Union["run.Partial", "run.Script"], executor: "run.Executor"):
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


def _default_fault_tolerance_converter(args: FaultTolerancePluginScriptArgs) -> List[str]:
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
        script_args_converter_fn (Optional[Callable]): A function that takes FaultTolerancePluginScriptArgs
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
    script_args_converter_fn: Optional[Callable[[FaultTolerancePluginScriptArgs], List[str]]] = None

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
    profile_ranks: List[int]
    record_shapes: bool


def _default_nsys_converter(args: NsysPluginScriptArgs) -> List[str]:
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
        profile_ranks (Optional[list[int]]): The ranks on which to run the nsys profiling. If not specified,
            profiling will be run on rank 0.
        nsys_trace (Optional[list[str]]): The events to trace during profiling. If not specified,
            'nvtx' and 'cuda' events will be traced.
        record_shapes (bool): Whether to record tensor shapes. Default is False.
        nsys_gpu_metrics (bool): Whether to enable GPU metrics collection. Default is False.
        script_args_converter_fn (Optional[Callable]): A function that takes NsysPluginScriptArgs
                                                        and returns a list of CLI arguments. If not provided,
                                                        uses the default hydra-style converter.

    Note:
        This plugin is incompatible with FaultTolerancePlugin. Nsys profiling cannot be used when
        fault tolerance is enabled, as the profiler interferes with the fault tolerance mechanisms.
    """

    profile_step_start: int
    profile_step_end: int
    profile_ranks: Optional[list[int]] = None
    nsys_trace: Optional[list[str]] = None
    record_shapes: bool = False
    nsys_gpu_metrics: bool = False
    script_args_converter_fn: Optional[Callable[[NsysPluginScriptArgs], List[str]]] = None

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
    profile_ranks: List[int]
    record_memory_history: bool
    memory_snapshot_path: str
    record_shapes: bool


def _default_pytorch_profiler_converter(args: PyTorchProfilerPluginScriptArgs) -> List[str]:
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
        profile_ranks (Optional[list[int]]): The ranks on which to run the profiling. If not specified,
            profiling will be run on rank 0.
        record_memory_history (bool): Whether to record memory history. Default is False.
        memory_snapshot_path (str): Path to save memory snapshots. Default is "snapshot.pickle".
        record_shapes (bool): Whether to record tensor shapes. Default is False.
        script_args_converter_fn (Optional[Callable]): A function that takes PyTorchProfilerPluginScriptArgs
                                                        and returns a list of CLI arguments. If not provided,
                                                        uses the default hydra-style converter.
    """

    profile_step_start: int
    profile_step_end: int
    profile_ranks: Optional[list[int]] = None
    record_memory_history: bool = False
    memory_snapshot_path: str = "snapshot.pickle"
    record_shapes: bool = False
    script_args_converter_fn: Optional[Callable[[PyTorchProfilerPluginScriptArgs], List[str]]] = None

    def setup(self, task: Union["run.Partial", "run.Script"], executor: "run.Executor"):
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
    entity: Optional[str]
    name: Optional[str]
    save_dir: str


def _default_wandb_converter(args: WandbPluginScriptArgs) -> List[str]:
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
        name (Optional[str]): The name for the Weights & Biases run. If not provided, uses experiment name.
        entity (Optional[str]): The Weights & Biases entity name.
        save_dir (str): Directory to save wandb logs. Default is "/nemo_run/wandb".
        log_task_config (bool, optional): Whether to log the task configuration to wandb.
            Defaults to True.
        script_args_converter_fn (Optional[Callable]): A function that takes WandbPluginScriptArgs
                                                        and returns a list of CLI arguments. If not provided,
                                                        uses the default hydra-style converter.
    """

    project: str
    name: Optional[str] = None
    entity: Optional[str] = None
    save_dir: str = "/nemo_run/wandb"
    log_task_config: bool = True
    script_args_converter_fn: Optional[Callable[[WandbPluginScriptArgs], List[str]]] = None

    def setup(self, task: Union["run.Partial", "run.Script"], executor: "run.Executor"):
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
class CometPluginScriptArgs:
    """Arguments for CometPlugin to pass to run.Script."""

    project: str
    workspace: Optional[str]
    name: Optional[str]


def _default_comet_converter(args: CometPluginScriptArgs) -> List[str]:
    """Default converter for CometPlugin that generates CLI overrides."""
    cli_overrides = [f"logger.comet_project={args.project}"]
    if args.workspace:
        cli_overrides.append(f"logger.comet_workspace={args.workspace}")
    if args.name:
        cli_overrides.append(f"logger.comet_experiment_name={args.name}")
    return cli_overrides


@dataclass(kw_only=True)
class CometPlugin(Plugin):
    """
    A plugin for setting up Comet ML configuration.

    This plugin sets up Comet ML logging configuration. The plugin is only activated
    if the ``COMET_API_KEY`` environment variable is set.
    The ``COMET_API_KEY`` environment variable will also be set in the executor's environment variables.
    Follow https://www.comet.com/docs/v2/guides/getting-started/quickstart/ to retrieve your ``COMET_API_KEY``.

    Args:
        project (str): The Comet ML project name.
        name (Optional[str]): The name for the Comet ML experiment.
        workspace (Optional[str]): The Comet ML workspace.
        script_args_converter_fn (Optional[Callable]): A function that takes CometPluginScriptArgs
                                                        and returns a list of CLI arguments.
    """

    project: str
    name: Optional[str] = None
    workspace: Optional[str] = None
    script_args_converter_fn: Optional[Callable[[CometPluginScriptArgs], List[str]]] = None

    def setup(self, task: Union["run.Partial", "run.Script"], executor: "run.Executor"):
        if not HAVE_NEMO_RUN:
            raise ImportError(MISSING_NEMO_RUN_MSG)

        if "COMET_API_KEY" in os.environ:
            executor.env_vars["COMET_API_KEY"] = os.environ["COMET_API_KEY"]

            if isinstance(task, Script):
                script_args = CometPluginScriptArgs(
                    project=self.project,
                    workspace=self.workspace,
                    name=self.name,
                )

                converter = self.script_args_converter_fn or _default_comet_converter
                cli_overrides = converter(script_args)

                task.args.extend(cli_overrides)
                logger.info(f"{self.__class__.__name__} added CLI overrides: {', '.join(cli_overrides)}")
            else:
                raise NotImplementedError("CometPlugin is only supported for run.Script tasks")
        else:
            logger.warning(
                f"Warning: The {self.__class__.__name__} will have no effect as COMET_API_KEY environment variable is not set."
            )
