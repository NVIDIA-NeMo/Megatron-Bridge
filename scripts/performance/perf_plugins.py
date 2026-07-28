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
Note: This file is a copy from megatron/bridge/recipes/run_plugins.py.
      This is being cloned to not require installing Megatron-Bridge to run the perf scripts.



This file contains plugins based on NeMo-Run's run.Plugin API.
Plugins operate both on a configured task and an executor at the same time, and are specific to NeMo-Run.
These plugins work by modifying the ConfigContainer configuration overrides.

For run.Script tasks, each plugin supports custom argument conversion via the `script_args_converter_fn`
parameter. This allows users to specify their own conversion function if their training scripts don't
use hydra-style overrides.
"""

import logging
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import nemo_run as run
from nemo_run import Plugin, Script, SlurmExecutor


logger: logging.Logger = logging.getLogger(__name__)
NSYS_SQLITE_EXPORT_ARG = "--export=sqlite"


def _format_list_for_override(values: List | int):
    """Render a Python list into a Hydra/CLI-safe list string without spaces.

    Example: [0, 3] -> "[0,3]"
    """
    if isinstance(values, int):
        values = [values]
    return "[" + ",".join(str(v) for v in values) + "]"


def _ensure_sqlite_nsys_export(nsys_extra_args: list[str]) -> list[str]:
    """Add SQLite export unless nsys export args already request it."""
    for index, arg in enumerate(nsys_extra_args):
        export_values = None
        if arg == "--export" and index + 1 < len(nsys_extra_args):
            export_values = nsys_extra_args[index + 1]
        elif arg.startswith("--export="):
            export_values = arg.split("=", 1)[1]

        if export_values is not None and "sqlite" in export_values.split(","):
            return nsys_extra_args

    return nsys_extra_args + [NSYS_SQLITE_EXPORT_ARG]


def _nsys_arg_name(arg: str) -> Optional[str]:
    """Return the option name for a flag-like token (e.g. '--cuda-graph-trace'), else None."""
    if not arg.startswith("-"):
        return None
    return arg.split("=", 1)[0]


def _merge_nsys_extra_args(user_args: list[str], default_args: list[str]) -> list[str]:
    """Merge user-provided nsys args over defaults, deduplicating by option name.

    nemo_run seeds ``launcher.nsys_extra_args`` with defaults such as
    ``--cuda-graph-trace=node``. Naively concatenating user args with these defaults
    emits the same option twice (e.g. both ``--cuda-graph-trace=node`` and
    ``--cuda-graph-trace=graph``), which nsys rejects. Here, any default whose option
    name is also supplied by the user is dropped so the user value wins.

    Both ``--flag=value`` and space-separated ``--flag value`` forms are handled.
    """
    user_names = {name for name in (_nsys_arg_name(arg) for arg in user_args) if name is not None}

    retained_defaults: list[str] = []
    skip_next = False
    for index, arg in enumerate(default_args):
        if skip_next:
            skip_next = False
            continue
        name = _nsys_arg_name(arg)
        if name is not None and name in user_names:
            # Drop this default; also drop its space-separated value, if any.
            if "=" not in arg and index + 1 < len(default_args) and not default_args[index + 1].startswith("-"):
                skip_next = True
            continue
        retained_defaults.append(arg)

    return user_args + retained_defaults


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
        export_sqlite (bool): Whether to export a SQLite report after profiling finishes. Default is False.
        script_args_converter_fn (Optional[Callable]): A function that takes NsysPluginScriptArgs
                                                        and returns a list of CLI arguments. If not provided,
                                                        uses the default hydra-style converter.

    Note:
        This plugin is incompatible with fault tolerance. Nsys profiling cannot be used when
        fault tolerance is enabled, as the profiler interferes with the fault tolerance mechanisms.

    """

    profile_step_start: int
    profile_step_end: int
    profile_ranks: Optional[list[int]] = None
    nsys_trace: Optional[list[str]] = None
    nsys_extra_args: Optional[list[str]] = None
    record_shapes: bool = False
    nsys_gpu_metrics: bool = False
    export_sqlite: bool = False
    script_args_converter_fn: Optional[Callable[[NsysPluginScriptArgs], List[str]]] = None

    def setup(self, task: Union["run.Partial", "run.Script"], executor: "run.Executor"):
        """Set up the nsys profiling plugin."""
        launcher = executor.get_launcher()
        launcher.nsys_profile = True

        # Set nsys_trace if provided, otherwise use nemo_run defaults
        if self.nsys_trace is not None:
            launcher.nsys_trace = self.nsys_trace

        # Combine default extra args with user-provided extra args
        if self.nsys_extra_args is not None:
            # Get existing launcher extra args (nemo_run defaults)
            existing_extra_args = launcher.nsys_extra_args or []
            # Merge, letting user args override any default sharing the same option
            # name (e.g. --cuda-graph-trace) instead of emitting a duplicate flag.
            launcher.nsys_extra_args = _merge_nsys_extra_args(self.nsys_extra_args, existing_extra_args)
            logger.info(f"Combined nsys_extra_args: {launcher.nsys_extra_args}")

        if self.export_sqlite:
            launcher.nsys_extra_args = _ensure_sqlite_nsys_export(launcher.nsys_extra_args or [])

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
    record_memory_history: bool = True
    memory_snapshot_path: str = "/nemo_run/pytorch_profile/snapshot.pickle"
    record_shapes: bool = False
    script_args_converter_fn: Optional[Callable[[PyTorchProfilerPluginScriptArgs], List[str]]] = None

    def setup(self, task: Union["run.Partial", "run.Script"], executor: "run.Executor"):
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
            raise NotImplementedError("PyTorchProfilerPlugin is only supported for run.Script tasks")


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
    """A plugin for setting up preemption handling and signals.

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
                f"{self.__class__.__name__} will send a SIGTERM {self.preempt_time} seconds before the "
                "job's time limit for your Slurm executor."
            )
            executor.signal = f"TERM@{self.preempt_time}"
