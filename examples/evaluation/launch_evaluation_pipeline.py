#!/usr/bin/env python3
"""
Launch Megatron-Bridge Evaluation

Parse arguments early to catch unknown args before other libraries
(like nemo_run) can consume them during import.
"""

import logging
import os
import signal
import sys
from dataclasses import dataclass

import nemo_run as run
import yaml
from nemo_run.core.execution.slurm import SlurmJobDetails


try:
    import wandb

    HAVE_WANDB = True
except (ImportError, ModuleNotFoundError):
    HAVE_WANDB = False
    wandb = None

try:
    from argument_parser import ENDPOINT_TYPES, parse_cli_args
    from utils.executors import dgxc_executor, slurm_executor
except (ImportError, ModuleNotFoundError):
    from .argument_parser import ENDPOINT_TYPES, parse_cli_args
    from .utils.executors import dgxc_executor, slurm_executor

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def register_pipeline_terminator(exp: run.Experiment, job_id: str):
    def sigterm_handler(_signo, _stack_frame):
        logger.info(f"Trying to terminate job {job_id}")
        exp.cancel(job_id=job_id)
        logger.info(f"Job {job_id} terminated")
        sys.exit(0)

    signal.signal(signal.SIGINT, sigterm_handler)
    signal.signal(signal.SIGTERM, sigterm_handler)


@dataclass(kw_only=True)
class CustomJobDetailsRay(SlurmJobDetails):
    # ray jobs have a custom logs structure

    @property
    def ls_term(self) -> str:
        """This term will be used to fetch the logs.

        The command used to list the files is ls -1 {ls_term} 2> /dev/null
        """
        assert self.folder
        return os.path.join(self.folder, "ray-overlap-*")


RAY_DEPLOY_SCRIPT = """
# Unset SLURM/PMI/PMIX env vars to prevent MPI initialization issues
for i in $(env | grep ^SLURM_ | cut -d"=" -f 1); do unset -v $i; done
for i in $(env | grep ^PMI_ | cut -d"=" -f 1); do unset -v $i; done
for i in $(env | grep ^PMIX_ | cut -d"=" -f 1); do unset -v $i; done

python \
  /opt/Export-Deploy/scripts/deploy/nlp/deploy_ray_inframework.py \
  --megatron_checkpoint {megatron_checkpoint} \
  --model_id megatron_model \
  --host {host} \
  --port {port} \
  --num_gpus {num_gpus} \
  --num_replicas {num_replicas} \
  --tensor_model_parallel_size {tensor_model_parallel_size} \
  --pipeline_model_parallel_size {pipeline_model_parallel_size} \
  --context_parallel_size {context_model_parallel_size}
"""

EVAL_SCRIPT = """
# Unset SLURM/PMI/PMIX env vars to prevent MPI initialization issues
for i in $(env | grep ^SLURM_ | cut -d"=" -f 1); do unset -v $i; done
for i in $(env | grep ^PMI_ | cut -d"=" -f 1); do unset -v $i; done
for i in $(env | grep ^PMIX_ | cut -d"=" -f 1); do unset -v $i; done

# Install missing dependency for lm-evaluation-harness
pip install math_verify --quiet

python << 'EVAL_EOF'
import subprocess
import time

from nemo_evaluator.api.api_dataclasses import (
    ApiEndpoint,
    ConfigParams,
    EvaluationConfig,
    EvaluationTarget,
)
from nemo_evaluator.api import check_endpoint, evaluate

# Configuration
endpoint_url = "{endpoint_url}"
endpoint_type = "{endpoint_type}"
model_id = "megatron_model"
eval_task = "{eval_task}"
limit_samples = {limit_samples}
parallelism = {parallelism}
request_timeout = {request_timeout}
temperature = {temperature}
top_p = {top_p}
top_k = {top_k}
output_dir = "/nemo_run/results/"

# Check server readiness
server_ready = check_endpoint(
    endpoint_url=endpoint_url,
    endpoint_type=endpoint_type,
    model_name=model_id,
)
if not server_ready:
    raise RuntimeError(
        "Server is not ready to accept requests. Check the deployment logs for errors."
    )

# Build configs
api_endpoint = ApiEndpoint(
    url=endpoint_url,
    type=endpoint_type,
    model_id=model_id,
)
target_cfg = EvaluationTarget(api_endpoint=api_endpoint)
eval_params = ConfigParams(
    limit_samples=limit_samples,
    parallelism=parallelism,
    request_timeout=request_timeout,
    temperature=temperature,
    top_p=top_p,
    top_k=top_k,
)
eval_cfg = EvaluationConfig(
    type=eval_task,
    params=eval_params,
    output_dir=output_dir,
)

# Run evaluation
result = evaluate(target_cfg=target_cfg, eval_cfg=eval_cfg)

# Shutdown Ray server
print("Evaluation completed. Shutting down Ray server...")
subprocess.run(["ray", "stop", "--force"], check=False, timeout=30)
print("Ray server shutdown command sent.")
time.sleep(5)
EVAL_EOF
"""


def main(args):
    """Deploys the inference and evaluation server with NemoRun."""
    # Deployment script
    deploy_run_script = run.Script(
        inline=RAY_DEPLOY_SCRIPT.format(
            megatron_checkpoint=args.megatron_checkpoint,
            host=args.host,
            port=args.port,
            num_gpus=args.num_gpus,
            num_replicas=args.num_replicas,
            tensor_model_parallel_size=args.tensor_model_parallel_size,
            pipeline_model_parallel_size=args.pipeline_model_parallel_size,
            context_model_parallel_size=args.context_model_parallel_size,
        ),
        metadata={"use_with_ray_cluster": True},
    )
    logger.info("Deploy script: %s", deploy_run_script)

    # Evaluation script - using inline script to avoid serialization issues
    eval_run_script = run.Script(
        inline=EVAL_SCRIPT.format(
            endpoint_url=f"http://{args.host}:{args.port}/v1/{ENDPOINT_TYPES[args.endpoint_type]}",
            endpoint_type=args.endpoint_type,
            eval_task=args.eval_task,
            limit_samples=args.limit_samples if args.limit_samples is not None else "None",
            parallelism=args.parallelism,
            request_timeout=args.request_timeout,
            temperature=args.temperature if args.temperature is not None else "None",
            top_p=args.top_p if args.top_p is not None else "None",
            top_k=args.top_k if args.top_k is not None else "None",
        )
    )
    logger.info("Evaluation script: %s", eval_run_script)

    if not args.dgxc_cluster:
        deploy_executor = slurm_executor(
            account=args.account,
            partition=args.partition,
            nodes=-(args.num_gpus // -args.gpus_per_node),
            num_gpus_per_node=args.gpus_per_node,
            time_limit=args.time_limit,
            container_image=args.container_image,
            custom_mounts=args.custom_mounts,
            custom_env_vars=args.custom_env_vars,
            hf_token=args.hf_token,
        )
    else:
        deploy_executor = dgxc_executor(
            dgxc_base_url=args.dgxc_base_url,
            dgxc_cluster=args.dgxc_cluster,
            dgxc_kube_apiserver_url=args.dgxc_kube_apiserver_url,
            dgxc_app_id=args.dgxc_app_id,
            dgxc_app_secret=args.dgxc_app_secret,
            dgxc_project_name=args.dgxc_project_name,
            dgxc_pvc_claim_name=args.dgxc_pvc_claim_name,
            dgxc_pvc_mount_path=args.dgxc_pvc_mount_path,
            custom_env_vars=args.custom_env_vars,
            nodes=-(args.num_gpus // -args.gpus_per_node),
            num_gpus_per_node=args.gpus_per_node,
            container_image=args.container_image,
        )

    eval_executor = deploy_executor.clone()
    eval_executor.srun_args = [
        "--mpi=none",
        "--ntasks-per-node=1",
        "--nodes=1",
    ]

    deploy_executor.job_details = CustomJobDetailsRay()
    eval_executor.job_details = CustomJobDetailsRay()

    with run.Experiment(args.experiment_name) as exp:
        exp.add(
            [deploy_run_script, eval_run_script],
            executor=[deploy_executor, eval_executor],
            name=args.experiment_name,
        )
        if args.dryrun:
            exp.dryrun()
        else:
            exp.run(detach=True)

    register_pipeline_terminator(exp=exp, job_id=exp.jobs[0].id)

    exp.logs(job_id=exp.jobs[0].id)

    result_dict = exp.status(return_dict=True)
    _, job_dict = list(result_dict.items())[0]

    with open(os.path.join(job_dict["local_dir"], "results", "results.yml"), "r") as f:
        results = yaml.safe_load(f)

    logger.info("Results: %s", results)

    if HAVE_WANDB and args.wandb_key:
        wandb_run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity_name,
            id=args.wandb_experiment_name,
            resume="allow",
        )
        artifact = wandb.Artifact(name="evaluation_results", type="evaluation_results")
        artifact.add_file(
            local_path=os.path.join(job_dict["local_dir"], "results", "results.json"), name="results.json"
        )
        wandb_run.log_artifact(artifact)

        for category in ["tasks", "groups"]:
            for task_or_group_name, result in results[category].items():
                for metric_name, metric_result in result["metrics"].items():
                    field_key = f"{category.rstrip('s')}/{task_or_group_name}/{metric_name}"
                    wandb_run.log(
                        {
                            f"{field_key}/value": metric_result["scores"][metric_name]["value"],
                            f"{field_key}/stderr": metric_result["scores"][metric_name]["stats"]["stderr"],
                        }
                    )

        wandb_run.finish()


if __name__ == "__main__":
    main(args=parse_cli_args().parse_args())
