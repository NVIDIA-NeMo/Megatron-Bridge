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
    from megatron.bridge.recipes.run_plugins import NsysPlugin, PerfEnvPlugin

import logging


logger: logging.Logger = logging.getLogger(__name__)

if __name__ == "__main__":
    args, _ = parse_cli_args()
    dtype = "bf16" if args.compute_dtype == "bf16" else f"{args.compute_dtype}_{args.fp8_recipe}"

    if args.model_name in ["qwen3"] and args.model_size in ["30b_a3b", "235b_a22b"]:
        assert args.hf_token is not None, "HF token is required for Qwen3 tokenizer. NullTokenizer to be used soon."

    if args.model_name in ["qwen3"] and args.model_size in ["30b_a3b", "235b_a22b"]:
        assert args.hf_token is not None, "HF token is required for Qwen3 tokenizer. NullTokenizer to be used soon."

    SCRIPT_DIR: Path = Path(__file__).parent.resolve()
    RUN_SCRIPT_FILENAME: str = "run_script.py"
    RUN_SCRIPT_PATH: Path = SCRIPT_DIR / RUN_SCRIPT_FILENAME
    logger.info(f"Run script path: {RUN_SCRIPT_PATH}")
    if not RUN_SCRIPT_PATH.is_file():
        logger.error(f"Specified run script not found: {RUN_SCRIPT_PATH}")
        logger.error("Ensure the path passed to --run_script is correct.")
        sys.exit(1)
    config_filename = f"{args.model_name}_{args.model_size}_{args.domain}_{args.task}.yaml"
    config_filepath = SCRIPT_DIR / "configs" / f"{args.model_name}" / config_filename
    logger.info(f"Config file path: {config_filepath}")
    if not config_filepath.is_file():
        logger.error(f"Specified YAML config file not found: {config_filepath}")
        logger.error("Ensure the path passed to --config_file is correct.")
        sys.exit(1)

    enable_deepep = bool(args.gpu.lower() in ["h100"])
    plugins = (
        [
            PerfEnvPlugin(
                enable_vboost=args.enable_vboost,
                nccl_pp_comm_chunksize=2097152 if args.model_size in ["70b", "405b"] else None,
                gpu_sm100_or_newer=args.gpu.lower() in ["b200", "gb200", "gb300"],
                layernorm_sm_margin=20 if enable_deepep else 16,
            )
        ]
        if HAS_NEMO_RUN
        else []
    )
    custom_mounts = args.custom_mounts + [
        f"{config_filepath}:{config_filepath}",
        f"{RUN_SCRIPT_PATH}:{RUN_SCRIPT_PATH}",
        f"{SCRIPT_DIR}:{SCRIPT_DIR}",
    ]
    logger.info(f"Custom mounts: {custom_mounts}")

    num_gpus_per_node = args.gpus_per_node
    yaml_overrides_omega = OmegaConf.load(config_filepath)
    preset = get_perf_matrix_overrides(yaml_overrides_omega, args)
    if preset:
        num_gpus_per_node = preset.get("num_gpus_per_node", args.gpus_per_node)

    num_nodes = -(args.num_gpus // -num_gpus_per_node)

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

    executor = slurm_executor(
        args.gpu.lower(),
        args.account,
        args.partition,
        args.log_dir,
        num_nodes,
        num_gpus_per_node,
        args.time_limit,
        args.container_image,
        custom_mounts=custom_mounts,
        custom_env_vars={},
        hf_token=args.hf_token,
        nemo_home=args.nemo_home,
        wandb_key=args.wandb_key,
        additional_slurm_params=additional_slurm_params,
    )

    if args.model_name in ["llama31"] and args.model_size in ["405b"] and args.gpu.lower() in ["gb200", "gb300"]:
        if args.compute_dtype == "fp8" and args.fp8_recipe in ["cs", "mx"]:
            executor.env_vars["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            executor.env_vars["CUDA_DEVICE_MAX_CONNECTIONS"] = "32"
    del_cudnn_ln = False
    if args.gpu.lower() in ["h100"]:
        if args.model_name == "llama31" and args.model_size == "405b":
            del_cudnn_ln = True
        elif args.model_name == "llama3" and args.model_size == "70b" and args.compute_dtype == "bf16":
            del_cudnn_ln = True
    if args.model_name in ["deepseek"] and args.model_size == "v3" and args.gpu.lower() in ["b200"]:
        del_cudnn_ln = True
    if args.model_name in ["qwen3"] and args.model_size in ["30b_a3b"] and args.gpu.lower() in ["gb200"]:
        del_cudnn_ln = True
    if del_cudnn_ln:
        if "NVTE_NORM_FWD_USE_CUDNN" in executor.env_vars:
            executor.env_vars.pop("NVTE_NORM_FWD_USE_CUDNN")
        if "NVTE_NORM_BWD_USE_CUDNN" in executor.env_vars:
            executor.env_vars.pop("NVTE_NORM_BWD_USE_CUDNN")

    if args.model_name in ["deepseek"] and args.model_size == "v3" and args.gpu.lower() in ["gb200"] and args.num_gpus == 128:
        executor.env_vars["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    target_script_args = [
        "--config_file",
        str(config_filepath),
    ]
    # Forward relevant args that run_script.py needs
    args_to_forward = ["model_name", "model_size", "compute_dtype", "fp8_recipe", "gpu", "use_tokendrop"]
    for arg_name in args_to_forward:
        if hasattr(args, arg_name):
            arg_value = getattr(args, arg_name)
            if arg_value is not None:
                target_script_args.extend([f"--{arg_name}", str(arg_value)])
    target_script_args.extend(["-a", "dummy", "-p", "dummy", "-ng", str(args.num_gpus)])

    train_script = run.Script(
        path=str(RUN_SCRIPT_PATH),
        entrypoint="python",
        args=target_script_args,
    )

    base_config = preset["common"]
    extra_config = preset[dtype]
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
    exp_name = f"pretrain_{args.model_name}_{args.model_size}_{dtype}_{exp_config}"

    run.run(train_script, executor=executor, plugins=plugins, dryrun=args.dryrun, detach=True, name=exp_name)
