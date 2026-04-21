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
from typing import List, Optional


try:
    from argument_parser import parse_cli_args
    from utils.executors import slurm_executor
except (ImportError, ModuleNotFoundError):
    from .argument_parser import parse_cli_args
    from .utils.executors import slurm_executor

import nemo_run as run


try:
    from perf_plugins import NsysPlugin, PerfEnvPlugin
except (ImportError, ModuleNotFoundError):
    from .perf_plugins import NsysPlugin, PerfEnvPlugin

import logging


logging.basicConfig(level=logging.DEBUG)
logger: logging.Logger = logging.getLogger(__name__)

SCRIPT_DIR: Path = Path(__file__).parent.resolve()
SCRIPT_NAME: str = "run_script.py"


def main(
    script_name: str,
    model_name: str,
    model_size: str,
    domain: str,
    task: str,
    compute_dtype: str,
    gpu: str,
    hf_token: str,
    custom_mounts: List[str],
    detach: bool,
    dryrun: bool,
    enable_vboost: bool,
    enable_nsys: bool,
    use_tokendrop: bool,
    moe_a2a_overlap: bool,
    tp_size: Optional[int],
    pp_size: Optional[int],
    cp_size: Optional[int],
    wandb_key: str,
    wandb_prj_name: str,
    wandb_exp_name: str,
    executor: run.Executor,
):
    """Sets up the experiment and runs it."""
    if model_name in ["qwen3"] and model_size in ["30b_a3b", "235b_a22b"]:
        assert hf_token is not None, "HF token is required for Qwen3 tokenizer. NullTokenizer to be used soon."

    if wandb_key is not None:
        assert wandb_prj_name is not None and wandb_exp_name is not None, (
            "both wandb_prj_name and wandb_exp_name are required for logging with WandB"
        )

    RUN_SCRIPT_PATH: Path = SCRIPT_DIR / script_name
    logger.info(f"Run script path: {RUN_SCRIPT_PATH}")
    if not RUN_SCRIPT_PATH.is_file():
        logger.error(f"Specified run script not found: {RUN_SCRIPT_PATH}")
        sys.exit(1)

    plugins = []

    plugins.append(
        PerfEnvPlugin(
            enable_vboost=enable_vboost,
            moe_a2a_overlap=moe_a2a_overlap,
            tp_size=tp_size,
            pp_size=pp_size,
            cp_size=cp_size,
            model_name=model_name,
            model_size=model_size,
            gpu=gpu,
            compute_dtype=compute_dtype,
            use_tokendrop=use_tokendrop,
        )
    )
    if enable_nsys:
        plugins.append(NsysPlugin(profile_step_start=10, profile_step_end=11))

    executor.container_mounts.extend(
        custom_mounts
        + [
            f"{RUN_SCRIPT_PATH}:{RUN_SCRIPT_PATH}",
            f"{SCRIPT_DIR}:{SCRIPT_DIR}",
        ]
    )
    logger.info(f"Custom mounts: {executor.container_mounts}")

    exp_name = f"{model_name}_{model_size}_{domain}_{task}" + (
        "_bf16" if compute_dtype == "bf16" else f"_{compute_dtype}"
    )
    logger.debug(
        run.Script(
            path=str(RUN_SCRIPT_PATH),
            entrypoint="python",
            env={"PYTHONPATH": f"{SCRIPT_DIR}:$PYTHONPATH"},
            args=list(sys.argv[1:]),
        )
    )
    run.run(
        run.Script(
            path=str(RUN_SCRIPT_PATH),
            entrypoint="python",
            env={"PYTHONPATH": f"{SCRIPT_DIR}:$PYTHONPATH"},
            args=list(sys.argv[1:]),
        ),
        executor=executor,
        plugins=plugins,
        dryrun=dryrun,
        detach=detach,
        name=exp_name,
    )

    exp_name_result, job_dict = list(run.Experiment.from_title(exp_name).status(return_dict=True).items()).pop()
    job_status = str(job_dict["status"])

    if job_status not in ["SUCCEEDED", "SUBMITTED", "PENDING", "RUNNING"]:
        raise Exception(f"Megatron-Bridge experiment failed for {exp_name_result} with status: {job_status}.")


logger: logging.Logger = logging.getLogger(__name__)

if __name__ == "__main__":
    args, _ = parse_cli_args()

    main(
        script_name=SCRIPT_NAME,
        model_name=args.model_name,
        model_size=args.model_size,
        domain=args.domain,
        task=args.task,
        compute_dtype=args.compute_dtype,
        gpu=args.gpu,
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
