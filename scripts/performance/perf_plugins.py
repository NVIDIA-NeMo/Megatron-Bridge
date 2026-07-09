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
import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import nemo_run as run
from nemo_run import Plugin, Script, SlurmExecutor


try:
    from utils.utils import WorkloadBaseConfig
except (ImportError, ModuleNotFoundError):
    from .utils.utils import WorkloadBaseConfig

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
            # Combine user args with existing args (user args first for precedence)
            launcher.nsys_extra_args = self.nsys_extra_args + existing_extra_args
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
class PerfEnvPluginScriptArgs:
    """Arguments for PerfEnvPlugin to pass to run.Script."""

    enable_manual_gc: bool
    manual_gc_interval: int


def _default_perf_env_converter(args: PerfEnvPluginScriptArgs) -> List[str]:
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
        enable_vboost (bool): Whether to steer more power towards tensor cores via
            `sudo nvidia-smi boost-slider --vboost 1`. May not work on all systems.
        lock_gpu_freq (int | None): Lock GPU graphics clock to the specified frequency in MHz via
            `sudo nvidia-smi -lgc <freq>`. Runs once per node before training. None to disable.
        enable_manual_gc (bool): Enable manual garbage collection for better performance.
        manual_gc_interval (int): Interval for manual garbage collection. Default is 100.
        script_args_converter_fn (Optional[Callable]): A function that takes PerfEnvPluginScriptArgs
            and returns a list of CLI arguments. If not provided, uses the
            default hydra-style converter.
        deterministic (bool): Add runtime determinism environment variables.
    """

    enable_vboost: bool = False
    lock_gpu_freq: int | None = None
    enable_manual_gc: bool = True
    manual_gc_interval: int = 100
    script_args_converter_fn: Optional[Callable[[PerfEnvPluginScriptArgs], List[str]]] = None
    deterministic: bool = False

    @staticmethod
    def _sync_slurm_container_env(executor: "run.Executor") -> None:
        """Make plugin-added variables override matching container-image values."""
        if isinstance(executor, SlurmExecutor):
            executor.container_env = sorted(executor.env_vars)

    def _set_determinism_env_vars(self, executor: "run.Executor") -> None:
        """Set env vars required for bit-exact reproducibility."""
        executor.env_vars["NCCL_ALGO"] = "Ring"
        executor.env_vars["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"
        executor.env_vars["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        logger.info("Deterministic mode enabled")

    def _set_manual_gc(
        self,
        task: Union["run.Partial", "run.Script"],
        executor: "run.Executor",
        enable_manual_gc: bool,
        manual_gc_interval: int,
    ):
        if enable_manual_gc:
            if isinstance(task, Script):
                # For run.Script, append CLI overrides
                # Create args dataclass
                script_args = PerfEnvPluginScriptArgs(
                    enable_manual_gc=enable_manual_gc,
                    manual_gc_interval=manual_gc_interval,
                )

                # Use custom converter or default
                converter = self.script_args_converter_fn or _default_perf_env_converter
                cli_overrides = converter(script_args)

                task.args.extend(cli_overrides)
                logger.info(f"{self.__class__.__name__} added CLI overrides: {', '.join(cli_overrides)}")
            else:
                raise NotImplementedError("PerfEnvPlugin is only supported for run.Script tasks")

    def _set_vboost(self, task: Union["run.Partial", "run.Script"], executor: "run.Executor", enable_vboost: bool):
        def get_vboost_srun_cmd(nodes, job_dir):
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

        if enable_vboost and isinstance(executor, SlurmExecutor):
            vboost_cmd = get_vboost_srun_cmd(executor.nodes, executor.tunnel.job_dir)
            executor.setup_lines = (
                executor.setup_lines + vboost_cmd
                if (executor.setup_lines and len(executor.setup_lines) > 0)
                else vboost_cmd
            )

    def _set_lock_gpu_freq(
        self, task: Union["run.Partial", "run.Script"], executor: "run.Executor", lock_gpu_freq: int | None
    ):
        """Lock GPU graphics clocks to a fixed frequency before training.

        Used for silicon simulation correlation studies where a fixed GPU
        clock frequency is required to match simulation assumptions.
        """

        def get_lock_gpu_freq_srun_cmd(job_dir, freq_mhz):
            import shlex

            lock_freq_cmd = "\n".join(
                [
                    "",
                    "# Command 0: lock GPU graphics clock",
                    " ".join(
                        [
                            "srun",
                            "--ntasks-per-node=1",
                            "--output",
                            os.path.join(job_dir, "lock_gpu_freq.out"),
                            "--error",
                            os.path.join(job_dir, "lock_gpu_freq.err"),
                            "bash -c",
                            shlex.quote(f"sudo nvidia-smi -lgc {freq_mhz}"),
                        ]
                    ),
                    "",
                ]
            )

            return lock_freq_cmd

        if lock_gpu_freq is not None and isinstance(executor, SlurmExecutor):
            lock_freq_cmd = get_lock_gpu_freq_srun_cmd(executor.tunnel.job_dir, lock_gpu_freq)
            executor.setup_lines = (
                executor.setup_lines + lock_freq_cmd
                if (executor.setup_lines and len(executor.setup_lines) > 0)
                else lock_freq_cmd
            )

    def setup_recipe_environment(
        self,
        task: Union["run.Partial", "run.Script", None],
        executor: "run.Executor",
        workload_base_config: WorkloadBaseConfig,
        protected_recipe_env_names: set[str] | None = None,
    ) -> None:
        """Project resolved recipe settings onto the worker environment.

        Flat recipes already carry the composed values. Existing launcher,
        shell, and explicit Hydra values retain precedence.
        """
        del task, protected_recipe_env_names
        for name, value in workload_base_config.env_vars.items():
            executor.env_vars.setdefault(name, str(value))

        if self.deterministic:
            self._set_determinism_env_vars(executor)

        self._sync_slurm_container_env(executor)

    def setup(self, task: Union["run.Partial", "run.Script"], executor: "run.Executor"):
        """Apply launcher-only settings without importing Megatron-Bridge on the login node."""
        # Recipe-dependent environment variables are applied by run_script_with_env.py
        # inside the training container, where the flat recipe can be imported safely.
        self._set_manual_gc(task, executor, self.enable_manual_gc, self.manual_gc_interval)
        self._set_vboost(task, executor, self.enable_vboost)
        self._set_lock_gpu_freq(task, executor, self.lock_gpu_freq)


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
