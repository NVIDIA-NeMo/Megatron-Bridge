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
from typing import TYPE_CHECKING, Callable


MISSING_NEMO_RUN_MSG = "nemo-run is not available. Please install it with `pip install nemo-run`."


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
        This plugin is incompatible with fault tolerance. Nsys profiling cannot be used when
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
            else:
                raise NotImplementedError("PerfEnvPlugin is only supported for run.Script tasks")

        # Improve perf by steering power to tensor cores, may not work on all systems
        if self.enable_vboost and isinstance(executor, SlurmExecutor):
            vboost_cmd = self.get_vboost_srun_cmd(executor.nodes, executor.tunnel.job_dir)
            executor.setup_lines = (
                executor.setup_lines + vboost_cmd
                if (executor.setup_lines and len(executor.setup_lines) > 0)
                else vboost_cmd
            )
