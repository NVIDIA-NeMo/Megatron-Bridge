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
    from utils.utils import get_workload_base_config
except (ImportError, ModuleNotFoundError):
    from .utils.utils import get_workload_base_config

logger: logging.Logger = logging.getLogger(__name__)


def _format_list_for_override(values: List | int):
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
    script_args_converter_fn: Optional[Callable[[NsysPluginScriptArgs], List[str]]] = None

    def setup(self, task: Union["run.Partial", "run.Script"], executor: "run.Executor"):
        """Set up the nsys profiling plugin."""
        launcher = executor.get_launcher()
        launcher.nsys_profile = True
        launcher.nsys_trace = self.nsys_trace or ["nvtx", "cuda"]
        launcher.nsys_extra_args = self.nsys_extra_args or launcher.nsys_extra_args

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
        enable_layernorm_sm_margin (bool): Set SM margin for TransformerEngine's Layernorm, so
            in order to not block DP level communication overlap.
        enable_vboost (bool): Whether to steer more power towards tensor cores via
            `sudo nvidia-smi boost-slider --vboost 1`. May not work on all systems.
        enable_manual_gc (bool): Enable manual garbage collection for better performance.
        manual_gc_interval (int): Interval for manual garbage collection. Default is 100.
        tp_size (int): Tensor parallelism size. Default is 1.
        cp_size (int): Context parallelism size. Default is 1.
        pp_size (int): Pipeline parallelism size. Default is 1.
        script_args_converter_fn (Optional[Callable]): A function that takes PerfEnvPluginScriptArgs
                                                        and returns a list of CLI arguments. If not provided,
                                                        uses the default hydra-style converter.
    """

    enable_layernorm_sm_margin: bool = True
    enable_vboost: bool = False
    enable_manual_gc: bool = True
    manual_gc_interval: int = 100
    tp_size: int = 1
    cp_size: int = 1
    pp_size: int = 1
    ep_size: int = 1
    script_args_converter_fn: Optional[Callable[[PerfEnvPluginScriptArgs], List[str]]] = None
    moe_a2a_overlap: bool = False
    model_family_name: str
    model_recipe_name: str
    gpu: str
    compute_dtype: str
    train_task: str

    def _set_num_cuda_device_max_connections(
        self,
        task: Union["run.Partial", "run.Script"],
        executor: "run.Executor",
        tp_size: int,
        cp_size: int,
        moe_a2a_overlap: bool,
        moe_flex_dispatcher_backend: str,
        gpu_sm100_or_newer: bool,
    ):
        cuda_device_max_connections = 8
        if moe_flex_dispatcher_backend in ["deepep", "hybridep"]:
            cuda_device_max_connections = 32
        if gpu_sm100_or_newer:
            """
            We need extra connections to avoid serialization of streams, so we use max connections of 32 instead
            of the default device connection of 8.
            """
            cuda_device_max_connections = 32
        else:
            # Hopper or earlier generation GPUs
            if (tp_size > 1 or cp_size > 1) and not moe_a2a_overlap:
                """
                Set the device connection to 1 to enforce kernel queuing order from host to execution order on GPU.
                This is needed to schedule a communication kernel before the overlapping persistent GEMM kernel.
                Otherwise, communication kernel will be pushed to the end of the GEMM kernel, failing to overlap the
                kernels.
                """
                cuda_device_max_connections = 1

        executor.env_vars["CUDA_DEVICE_MAX_CONNECTIONS"] = str(cuda_device_max_connections)
        logger.info(f"Set CUDA_DEVICE_MAX_CONNECTIONS to {cuda_device_max_connections}")

    def _set_model_specific_environment_variables(
        self,
        task: Union["run.Partial", "run.Script"],
        executor: "run.Executor",
        model_family_name: str,
        model_recipe_name: str,
        gpu: str,
        compute_dtype: str,
        train_task: str,
    ):
        """Set model-specific environment variables"""
        if (
            model_family_name in ["llama31"]
            and model_recipe_name in ["llama31_405b"]
            and train_task == "pretrain"
            and gpu in ["gb200"]
        ):
            if compute_dtype in ["fp8_cs", "fp8_mx"]:
                executor.env_vars["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        del_cudnn_ln = True
        if gpu in ["h100"]:
            if model_family_name == "llama3" and model_recipe_name == "llama3_8b" and train_task == "pretrain":
                if compute_dtype == "fp8_cs":
                    # executor.env_vars["NCCL_NVLS_ENABLE"] = "1" # This causes OOM; worked fine with NeMo2 and 25.09
                    executor.env_vars["NCCL_CTA_POLICY"] = "1"
                    del_cudnn_ln = False
        if gpu in ["gb200", "gb300"]:
            if model_family_name == "llama3" and model_recipe_name == "llama3_70b" and train_task == "pretrain":
                if compute_dtype == "bf16" or (compute_dtype == "fp8_cs"):
                    del_cudnn_ln = False
            if model_family_name == "llama31" and model_recipe_name == "llama31_405b" and train_task == "pretrain":
                if compute_dtype == "fp8_cs":
                    del_cudnn_ln = False
        if del_cudnn_ln:
            if "NVTE_NORM_FWD_USE_CUDNN" in executor.env_vars:
                executor.env_vars.pop("NVTE_NORM_FWD_USE_CUDNN")
            if "NVTE_NORM_BWD_USE_CUDNN" in executor.env_vars:
                executor.env_vars.pop("NVTE_NORM_BWD_USE_CUDNN")

    def _set_layernorm_sm_margin(
        self,
        task: Union["run.Partial", "run.Script"],
        executor: "run.Executor",
        enable_layernorm_sm_margin: bool,
        layernorm_sm_margin: int,
    ):
        if enable_layernorm_sm_margin:
            executor.env_vars["NVTE_FWD_LAYERNORM_SM_MARGIN"] = str(layernorm_sm_margin)
            executor.env_vars["NVTE_BWD_LAYERNORM_SM_MARGIN"] = str(layernorm_sm_margin)

    def _set_nvl_domain_size(
        self,
        task: Union["run.Partial", "run.Script"],
        executor: "run.Executor",
        moe_flex_dispatcher_backend: str,
        gpu: str,
        ep_size: int,
    ):
        if moe_flex_dispatcher_backend == "hybridep":
            assert ep_size <= 72, "ep_size must be less than or equal to 72"
            executor.env_vars["NVLINK_DOMAIN_SIZE"] = "72"
            executor.env_vars["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] = str(ep_size)
            executor.env_vars["USE_MNNVL"] = "1"

    def _set_nccl_pp_comm_chunksize(
        self,
        task: Union["run.Partial", "run.Script"],
        executor: "run.Executor",
        nccl_pp_comm_chunksize: int,
        pp_size: int,
    ):
        if pp_size > 1 and nccl_pp_comm_chunksize is not None:
            assert isinstance(nccl_pp_comm_chunksize, int) and nccl_pp_comm_chunksize > 1
            executor.env_vars["NCCL_P2P_NET_CHUNKSIZE"] = str(nccl_pp_comm_chunksize)

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

    def setup(self, task: Union["run.Partial", "run.Script"], executor: "run.Executor"):
        """Enable the performance environment settings"""
        workload_base_config = get_workload_base_config(
            self.model_family_name, self.model_recipe_name, self.gpu, self.compute_dtype, self.train_task
        )
        tp_size = self.tp_size if self.tp_size is not None else workload_base_config.tensor_model_parallel_size
        pp_size = self.pp_size if self.pp_size is not None else workload_base_config.pipeline_model_parallel_size
        cp_size = self.cp_size if self.cp_size is not None else workload_base_config.context_parallel_size
        ep_size = self.ep_size if self.ep_size is not None else workload_base_config.ep_size

        # Force program order kernel launch for TP, CP overlap
        moe_flex_dispatcher_backend = getattr(workload_base_config, "moe_flex_dispatcher_backend", None)
        moe_a2a_overlap = (
            self.moe_a2a_overlap
            if self.moe_a2a_overlap is not None
            else getattr(workload_base_config, "moe_a2a_overlap", False)
        )
        self._set_num_cuda_device_max_connections(
            task,
            executor,
            tp_size,
            cp_size,
            moe_a2a_overlap=moe_a2a_overlap,
            moe_flex_dispatcher_backend=moe_flex_dispatcher_backend,
            gpu_sm100_or_newer=self.gpu in ["b300", "b200", "gb200", "gb300"],
        )

        # Set LayerNorm SM margin to support the overlap with LayerNorm kernel
        layernorm_sm_margin = 20 if moe_flex_dispatcher_backend in ["deepep", "hybridep"] else 16
        self._set_layernorm_sm_margin(
            task, executor, self.enable_layernorm_sm_margin, layernorm_sm_margin=layernorm_sm_margin
        )

        # Set NVL domain size when using HybridEP
        self._set_nvl_domain_size(
            task,
            executor,
            moe_flex_dispatcher_backend,
            self.gpu,
            ep_size,
        )

        # Set the chunk size of P2P communications
        nccl_pp_comm_chunksize = (
            2097152
            if self.model_recipe_name in ["llama3_70b", "llama31_405b"] and self.train_task == "pretrain"
            else None
        )
        self._set_nccl_pp_comm_chunksize(task, executor, nccl_pp_comm_chunksize, pp_size)

        # Configure manual garbage collection
        self._set_manual_gc(task, executor, self.enable_manual_gc, self.manual_gc_interval)

        # Improve perf by steering power to tensor cores, may not work on all systems
        self._set_vboost(task, executor, self.enable_vboost)

        # Set model-specific environment variables
        self._set_model_specific_environment_variables(
            task,
            executor,
            self.model_family_name,
            self.model_recipe_name,
            self.gpu,
            self.compute_dtype,
            self.train_task,
        )


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
