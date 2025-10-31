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

import sys
from pathlib import Path
from typing import List

from omegaconf import OmegaConf


try:
    from argument_parser import parse_additional_slurm_params, parse_cli_args
    from utils.common import get_perf_matrix_overrides
    from utils.executors import slurm_executor
except (ImportError, ModuleNotFoundError):
    from .argument_parser import parse_additional_slurm_params, parse_cli_args
    from .utils.common import get_perf_matrix_overrides
    from .utils.executors import slurm_executor


try:
    import nemo_run as run

    HAS_NEMO_RUN = True
except ImportError:
    HAS_NEMO_RUN = False

if HAS_NEMO_RUN:
    try:
        from perf_plugins import NsysPlugin, PerfEnvPlugin
    except (ImportError, ModuleNotFoundError):
        from .perf_plugins import NsysPlugin, PerfEnvPlugin

    # TEMPORARY WORKAROUND - Remove in next release when upstream srun issue is fixed
    try:
        from utils.slurm_exit_code_override import *  # Monkey-patch for false-positive job failures
    except (ImportError, ModuleNotFoundError):
        from .utils.slurm_exit_code_override import *  # Monkey-patch for false-positive job failures

import logging


def main(
    model_name: str,
    model_size: str,
    domain: str,
    task: str,
    compute_dtype: str,
    fp8_recipe: str,
    gpu: str,
    num_gpus: int,
    gpus_per_node: int,
    hf_token: str,
    custom_mounts: List[str],
    detach: bool,
    dryrun: bool,
    enable_vboost: bool,
    enable_nsys: bool,
    use_tokendrop: bool,
    executor: run.Executor,
):
    """Sets up the experiment and runs it."""

    if model_name in ["qwen3"] and model_size in ["30b_a3b", "235b_a22b"]:
        assert hf_token is not None, "HF token is required for Qwen3 tokenizer. NullTokenizer to be used soon."

    SCRIPT_DIR: Path = Path(__file__).parent.resolve()
    RUN_SCRIPT_FILENAME: str = "run_script.py"
    RUN_SCRIPT_PATH: Path = SCRIPT_DIR / RUN_SCRIPT_FILENAME
    logger.info(f"Run script path: {RUN_SCRIPT_PATH}")
    if not RUN_SCRIPT_PATH.is_file():
        logger.error(f"Specified run script not found: {RUN_SCRIPT_PATH}")
        logger.error("Ensure the path passed to --run_script is correct.")
        sys.exit(1)
    config_filename = f"{model_name}_{model_size}_{domain}_{task}.yaml"
    config_filepath = SCRIPT_DIR / "configs" / f"{model_name}" / config_filename
    logger.info(f"Config file path: {config_filepath}")
    if not config_filepath.is_file():
        logger.error(f"Specified YAML config file not found: {config_filepath}")
        logger.error("Ensure the path passed to --config_file is correct.")
        sys.exit(1)

    yaml_overrides_omega = OmegaConf.load(config_filepath)
    preset = get_perf_matrix_overrides(yaml_overrides_omega, args)
    if not preset:
        num_gpus_yaml_key = f"num_gpus_{num_gpus or gpus_per_node}"
        logger.debug(f"No preset found for {gpu}.{num_gpus_yaml_key} in perf_matrix")

    common = preset.get("common") or {}
    compute_dtype, fp8_recipe = compute_dtype.lower(), fp8_recipe.lower()
    compute_dtype = compute_dtype if compute_dtype == "bf16" else f"{compute_dtype}_{fp8_recipe}"
    dtype_cfg = preset.get(compute_dtype) if compute_dtype in preset else None
    # Deep-merge so dtype-specific values override common
    merged_perf = OmegaConf.merge(OmegaConf.create(common), OmegaConf.create(dtype_cfg or {}))
    perf_overrides = OmegaConf.to_container(merged_perf, resolve=True)  #

    tp = perf_overrides.get("tp", 1)
    cp = perf_overrides.get("cp", 1)
    pp = perf_overrides.get("pp", 1)

    enable_deepep, a2a_overlap = False, False
    if gpu.lower() in ["h100"]:
        if model_name == "deepseek" and model_size == "v3":
            enable_deepep = True
            a2a_overlap = True

    plugins = (
        [
            PerfEnvPlugin(
                enable_vboost=args.enable_vboost,
                nccl_pp_comm_chunksize=2097152 if args.model_size in ["70b", "405b"] else None,
                gpu_sm100_or_newer=args.gpu.lower() in ["b200", "gb200", "gb300"],
                layernorm_sm_margin=20 if enable_deepep else 16,
                tp_size=tp,
                cp_size=cp,
                pp_size=pp,
                num_gpus=num_gpus,
                deepep_enabled=enable_deepep,
                a2a_overlap=a2a_overlap,
            )
        ]
        if HAS_NEMO_RUN
        else []
    )
    if HAS_NEMO_RUN and args.enable_nsys:
        profile_cfg = yaml_overrides_omega["ConfigContainer"]["profiling"]
        start_step = profile_cfg["profile_step_start"]
        end_step = profile_cfg["profile_step_end"]
        ranks = list(range(num_nodes * args.gpus_per_node))
        plugins.append(NsysPlugin(profile_step_start=start_step,
            profile_step_end=end_step,
            profile_ranks=ranks,
            nsys_gpu_metrics=args.profiling_gpu_metrics,
            nsys_trace=['cuda']))
        
    # Parse additional SLURM parameters if provided
    additional_slurm_params = None
    if hasattr(args, 'additional_slurm_params') and args.additional_slurm_params:
        additional_slurm_params = parse_additional_slurm_params(args.additional_slurm_params)

    custom_mounts = args.custom_mounts + [
        f"{config_filepath}:{config_filepath}",
        f"{RUN_SCRIPT_PATH}:{RUN_SCRIPT_PATH}",
        f"{SCRIPT_DIR}:{SCRIPT_DIR}",
    ]
    executor.container_mounts.extend(custom_mounts)
    logger.info(f"Custom mounts: {executor.container_mounts}")

    if model_name in ["llama31"] and model_size in ["405b"] and gpu.lower() in ["gb200", "gb300"]:
        if compute_dtype == "fp8" and fp8_recipe in ["cs", "mx"]:
            executor.env_vars["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            executor.env_vars["CUDA_DEVICE_MAX_CONNECTIONS"] = "32
    if model_name in ["deepseek"] and model_size in ["v3"] and gpu.lower() in ["gb200"] and args.num_gpus == 128:
        if compute_dtype == "bf16" and (not use_tokendrop):
            executor.env_vars["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # OOM if not set
    del_cudnn_ln = True
    if gpu.lower() in ["h100"]:
        if model_name == "llama3" and model_size == "8b":
            if compute_dtype == "fp8" and fp8_recipe == "cs":
                executor.env_vars["NCCL_NVLS_ENABLE"] = "1"
                executor.env_vars["NCCL_CTA_POLICY"] = "1"
                del_cudnn_ln = False
    if gpu.lower() in ["gb200"]:
        if model_name == "llama3" and model_size == "70b":
            if compute_dtype == "bf16" or (compute_dtype == "fp8" and fp8_recipe == "cs"):
                del_cudnn_ln = False
        if model_name == ["llama31"] and model_size == "405b":
            if compute_dtype == "fp8" and fp8_recipe == "cs":
                del_cudnn_ln = False
    if del_cudnn_ln:
        if "NVTE_NORM_FWD_USE_CUDNN" in executor.env_vars:
            executor.env_vars.pop("NVTE_NORM_FWD_USE_CUDNN")
        if "NVTE_NORM_BWD_USE_CUDNN" in executor.env_vars:
            executor.env_vars.pop("NVTE_NORM_BWD_USE_CUDNN")

    target_script_args = [
        "--config_file",
        str(config_filepath),
    ]
    # Forward relevant args that run_script.py needs
    args_to_forward = ["model_name", "model_size", "compute_dtype", "fp8_recipe", "gpu", "use_tokendrop", "micro_batch_size", "global_batch_size", "tensor_parallel_size", "pipeline_parallel_size", "virtual_pipeline_parallel_size", "context_parallel_size", "expert_parallel_size", "expert_tensor_parallel_size"]
    for arg_name in args_to_forward:
        if hasattr(args, arg_name):
            arg_value = getattr(args, arg_name)
            if arg_value is not None:
                target_script_args.extend([f"--{arg_name}", str(arg_value)])
    target_script_args.extend(["-a", "dummy", "-p", "dummy", "-ng", str(num_gpus)])

    train_script = run.Script(
        path=str(RUN_SCRIPT_PATH),
        entrypoint="python",
        args=target_script_args,
    )
    
    base_config = preset["common"]
    extra_config = preset[compute_dtype]
    train_config = OmegaConf.merge(base_config, extra_config) if extra_config else base_config

    exp_config = (
        f"gpus{args.num_gpus}_"
        f"tp{train_config['tp']}_"
        f"pp{train_config['pp']}_"
        f"cp{train_config['cp']}_"
        f"vp{train_config['vp']}_"
        f"ep{train_config['ep']}_"
        f"mbs{train_config['mbs']}_"
        f"gbs{train_config['gbs']}"
    )
    exp_name = f"pretrain_{args.model_name}_{args.model_size}_{compute_dtype}_{exp_config}"

    run.run(train_script, executor=executor, plugins=plugins, dryrun=dryrun, detach=detach, name=exp_name)

    experiment = run.Experiment.from_title(exp_name)
    result_dict = experiment.status(return_dict=True)
    for exp_name_result, job_dict in result_dict.items():
        job_status = str(job_dict["status"])

        if job_status != "SUCCEEDED":
            raise Exception(f"Megatron-Bridge experiment failed for {exp_name_result} with status: {job_status}.")


logger: logging.Logger = logging.getLogger(__name__)

if __name__ == "__main__":
    args, _ = parse_cli_args()
    main(
        model_name=args.model_name,
        model_size=args.model_size,
        domain=args.domain,
        task=args.task,
        compute_dtype=args.compute_dtype,
        fp8_recipe=args.fp8_recipe,
        gpu=args.gpu,
        num_gpus=args.num_gpus,
        gpus_per_node=args.gpus_per_node,
        hf_token=args.hf_token,
        custom_mounts=args.custom_mounts,
        detach=args.detach,
        dryrun=args.dryrun,
        enable_vboost=args.enable_vboost,
        enable_nsys=args.enable_nsys,
        use_tokendrop=args.use_tokendrop,
        executor=slurm_executor(
            args.gpu.lower(),
            args.account,
            args.partition,
            args.log_dir,
            -(args.num_gpus // -args.gpus_per_node),
            args.gpus_per_node,
            args.time_limit,
            args.container_image,
            custom_env_vars={},
            hf_token=args.hf_token,
            nemo_home=args.nemo_home,
            wandb_key=args.wandb_key,
            additional_slurm_params=additional_slurm_params,
        ),
    )
