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


try:
    from argument_parser import parse_cli_args
    from utils.executors import slurm_executor
    from utils.utils import get_parallelism_defaults
except (ImportError, ModuleNotFoundError):
    from .argument_parser import parse_cli_args
    from .utils.executors import slurm_executor
    from .utils.utils import get_parallelism_defaults

import nemo_run as run


try:
    from perf_plugins import NsysPlugin, PerfEnvPlugin
except (ImportError, ModuleNotFoundError):
    from .perf_plugins import NsysPlugin, PerfEnvPlugin

import logging


logger: logging.Logger = logging.getLogger(__name__)


def main(
    model_name: str,
    model_size: str,
    domain: str,
    task: str,
    compute_dtype: str,
    fp8_recipe: str,
    gpu: str,
    num_gpus: int,
    hf_token: str,
    custom_mounts: List[str],
    detach: bool,
    dryrun: bool,
    enable_vboost: bool,
    enable_nsys: bool,
    use_tokendrop: bool,
    moe_a2a_overlap: bool,
    tp_size: int,
    pp_size: int,
    cp_size: int,
    wandb_key: str,
    wandb_prj_name: str,
    wandb_exp_name: str,
    executor: run.Executor,
):
    """Sets up the experiment and runs it."""
    exp_name = f"{model_name}_{model_size}_{domain}_{task}"
    exp_name += "_bf16" if compute_dtype == "bf16" else f"_{compute_dtype}_{fp8_recipe}"

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

    enable_deepep = False
    moe_a2a_overlap = False if moe_a2a_overlap is None else moe_a2a_overlap
    if gpu in ["h100"] and model_name == "deepseek" and model_size == "v3":
        enable_deepep, moe_a2a_overlap = True, True

    parallelism_defaults = get_parallelism_defaults(model_name, model_size, gpu, num_gpus, compute_dtype, fp8_recipe)

    tp_size = tp_size if tp_size is not None else parallelism_defaults.tensor_model_parallel_size
    pp_size = pp_size if pp_size is not None else parallelism_defaults.pipeline_model_parallel_size
    cp_size = cp_size if cp_size is not None else parallelism_defaults.context_parallel_size

    plugins = [
        PerfEnvPlugin(
            enable_vboost=enable_vboost,
            nccl_pp_comm_chunksize=2097152 if model_size in ["70b", "405b"] else None,
            gpu_sm100_or_newer=gpu in ["b200", "gb200", "gb300"],
            layernorm_sm_margin=20 if enable_deepep else 16,
            num_gpus=num_gpus,
            deepep_enabled=enable_deepep,
            a2a_overlap=moe_a2a_overlap,
            tp_size=tp_size,
            pp_size=pp_size,
            cp_size=cp_size,
        )
    ]

    if enable_nsys:
        plugins.append(NsysPlugin(profile_step_start=10, profile_step_end=11))

    if wandb_key is not None:
        assert wandb_prj_name is not None and wandb_exp_name is not None, (
            "both wandb_prj_name and wandb_exp_name are required for logging with WandB"
        )

    custom_mounts = custom_mounts + [
        f"{RUN_SCRIPT_PATH}:{RUN_SCRIPT_PATH}",
        f"{SCRIPT_DIR}:{SCRIPT_DIR}",
    ]
    executor.container_mounts.extend(custom_mounts)
    logger.info(f"Custom mounts: {executor.container_mounts}")

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
    if gpu in ["gb200", "gb300"]:
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

    target_script_args = list(sys.argv[1:])
    train_script = run.Script(
        path=str(RUN_SCRIPT_PATH),
        entrypoint="python",
        env={"PYTHONPATH": f"{SCRIPT_DIR}:$PYTHONPATH"},
        args=target_script_args,
    )

    run.run(train_script, executor=executor, plugins=plugins, dryrun=dryrun, detach=detach, name=exp_name)

    experiment = run.Experiment.from_title(exp_name)
    result_dict = experiment.status(return_dict=True)
    for exp_name_result, job_dict in result_dict.items():
        job_status = str(job_dict["status"])

        if job_status not in ["SUCCEEDED", "SUBMITTED", "PENDING"]:
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
        hf_token=args.hf_token,
        custom_mounts=args.custom_mounts,
        detach=args.detach,
        dryrun=args.dryrun,
        enable_vboost=args.enable_vboost,
        enable_nsys=args.enable_nsys,
        use_tokendrop=args.use_tokendrop,
        moe_a2a_overlap=args.moe_a2a_overlap,
        tp_size=args.tensor_model_parallel_size,
        pp_size=args.pipeline_model_parallel_size,
        cp_size=args.context_parallel_size,
        wandb_key=args.wandb_key,
        wandb_prj_name=args.wandb_prj_name,
        wandb_exp_name=args.wandb_exp_name,
        executor=slurm_executor(
            args.gpu,
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
        ),
    )
