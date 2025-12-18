#!/usr/bin/env python3

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

import glob
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import nemo_run as run
from nemo_run.config import get_nemorun_home


try:
    from argument_parser import parse_cli_args
    from utils.evaluate import calc_convergence_and_performance
    from utils.executors import dgxc_executor, slurm_executor
except (ImportError, ModuleNotFoundError):
    from .argument_parser import parse_cli_args
    from .utils.evaluate import calc_convergence_and_performance
    from .utils.executors import dgxc_executor, slurm_executor

try:
    import wandb

    HAVE_WANDB = True
except (ImportError, ModuleNotFoundError):
    HAVE_WANDB = False

try:
    from perf_plugins import NsysPlugin, PerfEnvPlugin
    from resiliency_plugins import FaultTolerancePlugin
except (ImportError, ModuleNotFoundError):
    from .perf_plugins import NsysPlugin, PerfEnvPlugin
    from .resiliency_plugins import FaultTolerancePlugin

import logging


SCRIPT_DIR = Path(__file__).parent.resolve()
ENTRYPOINT_PEFORMANCE = "run_script.py"
ENTRYPOINT_RECIPE = "run_recipe.py"

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def check_training_finished(log_file_path: str) -> bool:
    """Check if training is finished."""
    with open(log_file_path, "r") as f:
        log_lines = f.readlines()
    log = "\n".join(log_lines)
    return "StopIteration" in log or "after training is done" in log or "exiting program at iteration" in log


def check_slurm_timeout(log_file_path: str) -> bool:
    """Check if Slurm job timed out."""
    with open(log_file_path, "r") as f:
        log_lines = f.readlines()
    log = "\n".join(log_lines)
    return "DUE TO TIME LIMIT" in log


def is_flaky_failure(log_file_path: str) -> bool:
    """Check if Slurm job failed due to flaky failure."""
    with open(log_file_path, "r") as f:
        log_lines = f.readlines()
    log = "\n".join(log_lines)

    return (
        "The server socket has failed to listen on any local network address." in log
        or "Some NCCL operations have failed or timed out." in log
        or "uncorrectable ECC error encountered" in log
        or "illegal memory access" in log
        or "illegal instruction" in log
        or "torch.distributed.DistNetworkError" in log
        or "Segmentation fault" in log
        or "found NaN in" in log
        or "For debugging consider passing CUDA_LAUNCH_BLOCKING=1" in log
        or "double free or corruption" in log
        or "Call to CUDA function failed." in log
        or "Connection reset by peer" in log
        or "invalid pointer" in log
        or "malloc(): unaligned tcache chunk detected" in log
        or "zmq.error.ZMQError: Address already in use" in log
        or "We couldn't connect to 'https://huggingface.co'" in log
        or "Unpack failed: incomplete input" in log
        or "unspecified launch failure" in log
        or "free(): corrupted unsorted chunks" in log
        or "Segfault encountered" in log
        or "Fatal glibc error" in log
    )


def build_performance_config(args) -> Optional[Dict[str, Any]]:
    """Build performance configuration from command-line arguments.

    Args:
        args: Parsed command-line arguments

    Returns:
        Dictionary with performance configuration or None if performance is disabled
    """
    config = {}

    performance_params = {
        "timing_threshold": args.timing_threshold,
        "skip_first_percent_time": args.skip_first_percent_time,
    }

    for key, value in performance_params.items():
        if value is not None:
            config[key] = value

    return config if config else None


def ensure_logs_where_written(log_file_paths: List[str]):
    """Ensure logs were written to disk."""
    if len(log_file_paths) != 1:
        raise FileNotFoundError(
            f"Unexpected number of log files found: {log_file_paths}. Expected 1, got {len(log_file_paths)}"
        )


def get_job_dir_and_status_from_run(exp_name: str):
    """Get job directory and status from run."""
    result_dict = run.Experiment.from_title(exp_name).status(return_dict=True)
    _, job_dict = list(result_dict.items())[0]
    job_dir = job_dict["local_dir"]
    job_status = str(job_dict["status"])
    return job_dir, job_status


def maybe_increase_n_attempts_on_flaky_failure(
    n_attempts: int,
    max_retries: int,
    is_finished_experiment: bool,
    is_long_convergence_run: bool,
    log_file_paths: List[str],
):
    """Maybe increase number of attempts."""
    if not is_finished_experiment and not is_long_convergence_run:
        if is_flaky_failure(log_file_paths[-1]):
            n_attempts += 1
        else:
            n_attempts = max_retries  # On non-flaky failures, we don't need to restart the experiment.

    return n_attempts


def main(
    use_recipes: bool,
    model_family_name: str,
    model_recipe_name: str,
    task: str,
    compute_dtype: str,
    gpu: str,
    hf_token: str,
    detach: bool,
    dryrun: bool,
    enable_vboost: bool,
    enable_nsys: bool,
    moe_a2a_overlap: bool,
    tp_size: Optional[int],
    pp_size: Optional[int],
    cp_size: Optional[int],
    wandb_key: str,
    wandb_project_name: str,
    wandb_experiment_name: str,
    wandb_entity_name: str,
    profiling_start_step: int,
    profiling_stop_step: int,
    profiling_gpu_metrics: bool,
    profiling_ranks: Optional[List[int]],
    nemo_home: str,
    account: str,
    partition: str,
    log_dir: str,
    gpus_per_node: int,
    time_limit: str,
    container_image: str,
    custom_mounts: List[str],
    custom_env_vars: List[str],
    custom_srun_args: List[str],
    pretrained_checkpoint: Optional[str],
    num_gpus: int,
    is_long_convergence_run: bool,
    additional_slurm_params: Optional[Dict[str, Any]],
    golden_values_path: str,
    convergence_params: Dict[str, Any],
    performance_params: Dict[str, Any],
    max_retries: int,
    dgxc_base_url: str,
    dgxc_cluster: str,
    dgxc_kube_apiserver_url: str,
    dgxc_app_id: str,
    dgxc_app_secret: str,
    dgxc_project_name: str,
    dgxc_pvc_claim_name: str,
    dgxc_pvc_mount_path: str,
):
    """Sets up the experiment and runs it."""
    if (
        model_family_name in ["qwen3"]
        and model_recipe_name
        in [
            "qwen3_30b_a3b",
            "qwen3_235b_a22b",
        ]
        and task == "pretrain"
    ):
        assert hf_token is not None, "HF token is required for Qwen3 tokenizer. NullTokenizer to be used soon."

    if wandb_key is not None:
        assert wandb_project_name is not None and wandb_experiment_name is not None, (
            "both wandb_project_name and wandb_experiment_name are required for logging with WandB"
        )

    if use_recipes:
        script_name = ENTRYPOINT_RECIPE
        exp_name = (
            wandb_experiment_name
            if wandb_experiment_name is not None
            else f"{model_recipe_name}_{task}_{num_gpus}gpu_{gpu}"
        )

    else:
        script_name = ENTRYPOINT_PEFORMANCE
        exp_name = (
            wandb_experiment_name
            if wandb_experiment_name is not None
            else f"{model_recipe_name}_{task}_{num_gpus}gpu_{gpu}_{compute_dtype}"
        )

    if pretrained_checkpoint is not None:
        custom_mounts.append(f"{pretrained_checkpoint}:{pretrained_checkpoint}")

    run_script_path = SCRIPT_DIR / script_name
    logger.info(f"Run script path: {run_script_path}")
    if not run_script_path.is_file():
        logger.error(f"Specified run script not found: {run_script_path}")
        sys.exit(1)

    custom_mounts.extend(
        [
            f"{run_script_path}:{run_script_path}",
            f"{SCRIPT_DIR}:{SCRIPT_DIR}",
        ]
    )

    if not dgxc_cluster:
        executor = slurm_executor(
            gpu=gpu,
            account=account,
            partition=partition,
            log_dir=log_dir,
            nodes=-(num_gpus // -gpus_per_node),
            num_gpus_per_node=gpus_per_node,
            time_limit=time_limit,
            container_image=container_image,
            custom_mounts=custom_mounts,
            custom_env_vars=custom_env_vars,
            custom_srun_args=custom_srun_args,
            gres=args.gres,
            hf_token=hf_token,
            nemo_home=nemo_home,
            additional_slurm_params=additional_slurm_params,
            wandb_key=wandb_key,
        )
    else:
        executor = dgxc_executor(
            dgxc_base_url=dgxc_base_url,
            dgxc_cluster=dgxc_cluster,
            dgxc_kube_apiserver_url=dgxc_kube_apiserver_url,
            dgxc_app_id=dgxc_app_id,
            dgxc_app_secret=dgxc_app_secret,
            dgxc_project_name=dgxc_project_name,
            dgxc_pvc_claim_name=dgxc_pvc_claim_name,
            dgxc_pvc_mount_path=dgxc_pvc_mount_path,
            custom_env_vars=custom_env_vars,
            nodes=-(num_gpus // -gpus_per_node),
            num_gpus_per_node=gpus_per_node,
            container_image=container_image,
            wandb_key=wandb_key,
            hf_token=hf_token,
        )

    plugins = []

    if not use_recipes:
        plugins.append(
            PerfEnvPlugin(
                enable_vboost=enable_vboost,
                moe_a2a_overlap=moe_a2a_overlap,
                tp_size=tp_size,
                pp_size=pp_size,
                cp_size=cp_size,
                model_family_name=model_family_name,
                model_recipe_name=model_recipe_name,
                gpu=gpu,
                compute_dtype=compute_dtype,
                train_task=task,
            )
        )

    if enable_nsys:
        plugins.append(
            NsysPlugin(
                profile_step_start=profiling_start_step,
                profile_step_end=profiling_stop_step,
                nsys_gpu_metrics=profiling_gpu_metrics,
                profile_ranks=profiling_ranks,
            )
        )

    if use_recipes:
        plugins.append(
            FaultTolerancePlugin(
                enable_ft_package=True,
                calc_ft_timeouts=True,
                num_in_job_restarts=10,
                num_job_retries_on_failure=10,
                initial_rank_heartbeat_timeout=1800,
                rank_heartbeat_timeout=300,
            )
        )

    nemorun_script = run.Script(
        path=str(run_script_path),
        entrypoint="python",
        env={"PYTHONPATH": f"{SCRIPT_DIR}:$PYTHONPATH"},
        args=list(sys.argv[1:]),
    )

    logger.info("Will launch the following command with Nemo-Run: %s", " ".join(nemorun_script.to_command()))

    is_finished_experiment = False  # An experiment might consist of multiple training runs, due to restarts.
    is_testing_passed = False  # Whether the testing passed convergence and performance validation.
    error_msg = None
    n_attempts = 0
    exp_name = (
        exp_name[:37] if dgxc_cluster is not None else exp_name
    )  # Some k8s clusters have a limit on the length of the experiment name.
    wandb_run_id = None
    while n_attempts <= max_retries:
        while is_finished_experiment is False:
            if HAVE_WANDB:
                wandb_run_id = (
                    (wandb_run_id or wandb.util.generate_id()) if is_long_convergence_run else wandb.util.generate_id()
                )
                executor.env_vars.update(
                    {
                        "WANDB_RUN_ID": wandb_run_id,
                        "WANDB_RESUME": "allow",
                    }
                )
            if wandb_key is not None:
                executor.env_vars["WANDB_API_KEY"] = wandb_key

            run.run(
                nemorun_script,
                executor=executor,
                plugins=plugins,
                dryrun=dryrun,
                detach=detach,
                name=exp_name,
            )
            if dryrun:
                logger.info("dryrun requested: exiting")
                return

            job_dir, job_status = get_job_dir_and_status_from_run(exp_name)

            if job_status not in ["SUCCEEDED", "SUBMITTED", "PENDING", "RUNNING"]:
                raise Exception(f"Experiment failed for {exp_name} with status: {job_status}.")

            if detach:
                is_finished_experiment = True
                is_testing_passed = True
                break

            log_file_paths = list(Path(f"{job_dir}").glob("log-*_0.out"))
            ensure_logs_where_written(log_file_paths)

            is_finished_experiment = (
                check_training_finished(log_file_paths[-1]) if is_long_convergence_run else (job_status == "SUCCEEDED")
            )

            n_attempts = maybe_increase_n_attempts_on_flaky_failure(
                n_attempts=n_attempts,
                max_retries=max_retries,
                is_finished_experiment=is_finished_experiment,
                is_long_convergence_run=is_long_convergence_run,
                log_file_paths=log_file_paths,
            )

            if not is_finished_experiment and n_attempts <= max_retries:
                logger.error(f"Starting attempt {n_attempts + 1} of {max_retries + 1} for {exp_name}")

            if not is_finished_experiment:
                break

        if is_finished_experiment is True and detach is False:
            log_paths = sorted(
                list(glob.glob(f"{get_nemorun_home()}/experiments/{exp_name}/{exp_name}_*/{exp_name}/log-*_0.out"))
            )

            if not is_long_convergence_run:
                log_paths = [log_paths[-1]]

            logger.info(f"Starting convergence check for {model_family_name}_{model_recipe_name}")
            wandb_run = None
            if HAVE_WANDB and wandb_key:
                wandb_run = wandb.init(
                    project=wandb_project_name, entity=wandb_entity_name, id=wandb_run_id, resume="allow"
                )

            logger.info("Waiting 10 seconds for I/O to settle")
            time.sleep(10)

            is_testing_passed, error_msg = calc_convergence_and_performance(
                model_family_name=model_family_name,
                model_recipe_name=model_recipe_name,
                assets_dir=os.path.join(job_dir, exp_name),
                log_paths=log_paths,
                loss_metric="lm loss",
                timing_metric="elapsed time per iteration (ms)",
                golden_values_path=golden_values_path,
                convergence_config=convergence_params,
                performance_config=performance_params,
                wandb_run=wandb_run,
            )

            if wandb_run:
                wandb_run.finish()
                wandb.teardown(exit_code=int(not is_testing_passed))

            if not is_testing_passed and not is_long_convergence_run:
                if n_attempts < max_retries:
                    logger.error(f"Starting attempt {n_attempts + 2} of {max_retries + 1} for {exp_name}")
                n_attempts += 1
                is_finished_experiment = False

        if is_finished_experiment and is_testing_passed:
            break

    if not is_testing_passed and error_msg is not None:
        raise AssertionError(error_msg)

    if not is_finished_experiment:
        raise Exception("Megatron-Bridge CI test job failed")
    elif is_finished_experiment and not detach:
        logger.info("Megatron-Bridge CI test job completed successfully!")


if __name__ == "__main__":
    parser = parse_cli_args()
    args, unknown_args = parser.parse_known_args()

    # probably better to use parser.parse_args() and make unknowns an error,
    # but for now we'll just issue a warning.
    if unknown_args:
        logger.warning(f"Ignoring unrecognized arguments: {' '.join(unknown_args)}")

    main(
        use_recipes=args.use_recipes,
        model_family_name=args.model_family_name,
        model_recipe_name=args.model_recipe_name,
        task=args.task,
        compute_dtype=args.compute_dtype,
        gpu=args.gpu,
        hf_token=args.hf_token,
        detach=args.detach,
        dryrun=args.dryrun,
        enable_vboost=args.enable_vboost,
        enable_nsys=args.enable_nsys,
        moe_a2a_overlap=args.moe_a2a_overlap,
        tp_size=args.tensor_model_parallel_size,
        pp_size=args.pipeline_model_parallel_size,
        cp_size=args.context_parallel_size,
        wandb_key=args.wandb_key,
        wandb_project_name=args.wandb_project_name,
        wandb_experiment_name=args.wandb_experiment_name,
        wandb_entity_name=args.wandb_entity_name,
        profiling_start_step=args.profiling_start_step,
        profiling_stop_step=args.profiling_stop_step,
        profiling_gpu_metrics=args.profiling_gpu_metrics,
        profiling_ranks=args.profiling_ranks,
        nemo_home=args.nemo_home,
        account=args.account,
        partition=args.partition,
        log_dir=args.log_dir,
        gpus_per_node=args.gpus_per_node,
        time_limit=args.time_limit,
        container_image=args.container_image,
        custom_mounts=args.custom_mounts,
        custom_env_vars=args.custom_env_vars,
        custom_srun_args=args.custom_srun_args,
        pretrained_checkpoint=args.pretrained_checkpoint,
        num_gpus=args.num_gpus,
        is_long_convergence_run=args.is_long_convergence_run,
        additional_slurm_params=args.additional_slurm_params,
        golden_values_path=args.golden_values_path,
        convergence_params={
            "correlation_threshold": args.correlation_threshold,
            "high_loss_tolerance": args.high_loss_tolerance,
            "medium_loss_tolerance": args.medium_loss_tolerance,
            "low_loss_tolerance": args.low_loss_tolerance,
            "final_loss_tolerance": args.final_loss_tolerance,
            "max_outlier_ratio": args.max_outlier_ratio,
            "outlier_threshold": args.outlier_threshold,
            "skip_first_percent_loss": args.skip_first_percent_loss,
        },
        performance_params={
            "timing_threshold": args.timing_threshold,
            "skip_first_percent_time": args.skip_first_percent_time,
        },
        max_retries=args.max_retries,
        dgxc_base_url=args.dgxc_base_url,
        dgxc_cluster=args.dgxc_cluster,
        dgxc_kube_apiserver_url=args.dgxc_kube_apiserver_url,
        dgxc_app_id=args.dgxc_app_id,
        dgxc_app_secret=args.dgxc_app_secret,
        dgxc_project_name=args.dgxc_project_name,
        dgxc_pvc_claim_name=args.dgxc_pvc_claim_name,
        dgxc_pvc_mount_path=args.dgxc_pvc_mount_path,
    )
