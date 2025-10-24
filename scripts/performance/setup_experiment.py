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

import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf


try:
    from argument_parser import parse_cli_args
    from utils.common import get_perf_matrix_overrides
    from utils.executors import slurm_executor
except (ImportError, ModuleNotFoundError):
    from .argument_parser import parse_cli_args
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

import logging


logger: logging.Logger = logging.getLogger(__name__)


def add_perf_environment_to_executor(
    executor: run.SlurmExecutor,
    model_name: str,
    model_size: str,
    gpu: str,
    compute_dtype: str,
    fp8_recipe: str,
    use_tokendrop: bool,
) -> run.SlurmExecutor:
    if model_name in ["llama31"] and model_size in ["405b"] and gpu in ["gb200"]:
        if compute_dtype == "fp8" and fp8_recipe in ["cs", "mx"]:
            executor.env_vars["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    if model_name in ["deepseek"] and model_size in ["v3"] and gpu in ["gb200"]:
        if compute_dtype == "bf16" and (not use_tokendrop):
            executor.env_vars["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # OOM if not set

    del_cudnn_ln = True
    if gpu in ["h100"]:
        if model_name == "llama3" and model_size == "8b":
            if compute_dtype == "fp8" and fp8_recipe == "cs":
                executor.env_vars["NCCL_NVLS_ENABLE"] = "1"
                executor.env_vars["NCCL_CTA_POLICY"] = "1"
                del_cudnn_ln = False

    if gpu in ["gb200"]:
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

    return executor


def build_script_args(
    config_filepath: str,
    model_name: str | None,
    model_size: str | None,
    compute_dtype: str | None,
    fp8_recipe: str | None,
    gpu: str | None,
    use_tokendrop: bool | None,
    num_gpus: int | None,
) -> list[str]:
    script_args = [
        "--config_file",
        str(config_filepath),
    ]
    # Forward relevant args that run_script.py needs
    if model_name is not None:
        script_args.extend(["--model_name", model_name])
    if model_size is not None:
        script_args.extend(["--model_size", model_size])
    if compute_dtype is not None:
        script_args.extend(["--compute_dtype", compute_dtype])
    if fp8_recipe is not None:
        script_args.extend(["--fp8_recipe", fp8_recipe])
    if gpu is not None:
        script_args.extend(["--gpu", gpu])
    if use_tokendrop is not None:
        script_args.extend(["--use_tokendrop", use_tokendrop])
    if num_gpus is not None:
        script_args.extend(["--num_gpus", num_gpus])
    script_args.extend(["-a", "dummy", "-p", "dummy", "-ng", str(num_gpus)])

    return script_args


def build_perf_env_plugin(
    config_filepath: str,
    num_gpus: int,
    gpus_per_node: int,
    gpu: str,
    compute_dtype: str,
    fp8_recipe: str,
    model_size: str,
) -> list[run.Plugin]:
    preset = get_perf_matrix_overrides(OmegaConf.load(config_filepath), args)
    if not preset:
        num_gpus_yaml_key = f"num_gpus_{num_gpus or gpus_per_node}"
        logger.debug(f"No preset found for {gpu}.{num_gpus_yaml_key} in perf_matrix")

    common = preset.get("common") or {}
    compute_dtype_mapped = compute_dtype if compute_dtype == "bf16" else f"{compute_dtype}_{fp8_recipe}"
    dtype_cfg = preset.get(compute_dtype_mapped) if compute_dtype_mapped in preset else None
    # Deep-merge so dtype-specific values override common
    merged_perf = OmegaConf.merge(OmegaConf.create(common), OmegaConf.create(dtype_cfg or {}))
    perf_overrides = OmegaConf.to_container(merged_perf, resolve=True)  #

    tp = perf_overrides.get("tp", 1)
    cp = perf_overrides.get("cp", 1)
    pp = perf_overrides.get("pp", 1)

    enable_deepep, a2a_overlap = False, False
    if gpu in ["h100"]:
        if args.model_name == "deepseek" and model_size == "v3":
            enable_deepep = True
            a2a_overlap = True

    return PerfEnvPlugin(
        enable_vboost=args.enable_vboost,
        nccl_pp_comm_chunksize=2097152 if model_size in ["70b", "405b"] else None,
        gpu_sm100_or_newer=gpu in ["b200", "gb200"],
        layernorm_sm_margin=20 if enable_deepep else 16,
        tp_size=tp,
        cp_size=cp,
        pp_size=pp,
        num_gpus=args.num_gpus,
        deepep_enabled=enable_deepep,
        a2a_overlap=a2a_overlap,
    )


def launch_performance_experiment(
    model_name: str,
    model_size: str,
    domain: str,
    task: str,
    compute_dtype: str,
    fp8_recipe: str,
    gpu: str,
    use_tokendrop: bool,
    num_gpus: int,
    gpus_per_node: int,
    hf_token: str,
    executor: run.SlurmExecutor,
    enable_nsys: bool,
    dryrun: bool,
    detach: bool,
):
    if model_name in ["qwen3"] and model_size in ["30b_a3b", "235b_a22b"]:
        assert hf_token is not None, "HF token is required for Qwen3 tokenizer. NullTokenizer to be used soon."

    SCRIPT_DIR = Path(__file__).parent.resolve()
    RUN_SCRIPT_PATH = SCRIPT_DIR / "run_script.py"
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

    executor.custom_mounts.extend(
        [
            f"{config_filepath}:{config_filepath}",
            f"{RUN_SCRIPT_PATH}:{RUN_SCRIPT_PATH}",
            f"{SCRIPT_DIR}:{SCRIPT_DIR}",
        ]
    )
    logger.info(f"Custom mounts: {executor.custom_mounts}")

    plugins = [
        build_perf_env_plugin(
            config_filepath=str(config_filepath),
            num_gpus=num_gpus,
            gpus_per_node=gpus_per_node,
            gpu=gpu,
            compute_dtype=compute_dtype,
            fp8_recipe=fp8_recipe,
            model_size=model_size,
        )
    ]

    if enable_nsys:
        plugins.append(NsysPlugin(profile_step_start=10, profile_step_end=11))

    run.run(
        run.Script(
            path=str(RUN_SCRIPT_PATH),
            entrypoint="python",
            args=build_script_args(
                config_filepath=str(config_filepath),
                model_name=model_name,
                model_size=model_size,
                compute_dtype=compute_dtype,
                fp8_recipe=fp8_recipe,
                gpu=gpu,
                use_tokendrop=use_tokendrop,
                num_gpus=num_gpus,
            ),
        ),
        executor=add_perf_environment_to_executor(
            executor,
            model_name=model_name,
            model_size=model_size,
            gpu=gpu,
            compute_dtype=compute_dtype,
            fp8_recipe=fp8_recipe,
            use_tokendrop=use_tokendrop,
        ),
        plugins=plugins,
        dryrun=dryrun,
        detach=detach,
        name=(
            (f"{model_name}_{model_size}_{domain}_{task}")
            + ("_bf16" if compute_dtype == "bf16" else f"_{compute_dtype}_{fp8_recipe}")
        ),
    )


if __name__ == "__main__":
    parser = parse_cli_args()
    args, _ = parser.parse_known_args()

    launch_performance_experiment(
        model_name=args.model_name,
        model_size=args.model_size,
        domain=args.domain,
        task=args.task,
        compute_dtype=args.compute_dtype,
        fp8_recipe=args.fp8_recipe,
        gpu=args.gpu,
        use_tokendrop=args.use_tokendrop,
        num_gpus=args.num_gpus,
        gpus_per_node=args.gpus_per_node,
        hf_token=args.hf_token,
        executor=slurm_executor(
            gpu=args.gpu,
            account=args.account,
            partition=args.partition,
            log_dir=args.log_dir,
            nodes=-(args.num_gpus // -args.gpus_per_node),
            num_gpus_per_node=args.gpus_per_node,
            time_limit=args.time_limit,
            container_image=args.container_image,
            custom_mounts=args.custom_mounts,
            custom_env_vars={},
            hf_token=args.hf_token,
            nemo_home=args.nemo_home,
            wandb_key=args.wandb_key,
        ),
        enable_nsys=args.enable_nsys,
        dryrun=args.dryrun,
        detach=args.detach,
    )
