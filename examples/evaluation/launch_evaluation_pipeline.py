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
from nemo_run.run.ray.cluster import RayCluster
from nemo_run.run.ray.job import RayJob


try:
    import wandb

    HAVE_WANDB = True
except (ImportError, ModuleNotFoundError):
    HAVE_WANDB = False
    wandb = None

try:
    from argument_parser import ENDPOINT_TYPES, parse_cli_args
    from utils.executors import kuberay_executor, slurm_executor
except (ImportError, ModuleNotFoundError):
    from .argument_parser import ENDPOINT_TYPES, parse_cli_args
    from .utils.executors import kuberay_executor, slurm_executor

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def register_pipeline_terminator(job: RayJob):
    """Register a signal handler to terminate the job."""

    def sigterm_handler(_signo, _stack_frame):
        logger.info(f"Trying to terminate job {job.name}")
        job.stop(wait=True)
        logger.info(f"Job {job.name} terminated")
        sys.exit(0)

    signal.signal(signal.SIGINT, sigterm_handler)
    signal.signal(signal.SIGTERM, sigterm_handler)


@dataclass(kw_only=True)
class CustomJobDetailsRay(SlurmJobDetails):
    """Custom job details for Ray jobs."""

    @property
    def ls_term(self) -> str:
        """This term will be used to fetch the logs.

        The command used to list the files is ls -1 {ls_term} 2> /dev/null
        """
        assert self.folder
        return os.path.join(self.folder, "ray-job.log")


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
        executor = slurm_executor(
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
        executor = kuberay_executor(
            nodes=-(args.num_gpus // -args.gpus_per_node),
            num_gpus_per_node=args.gpus_per_node,
            dgxc_pvc_claim_name=args.dgxc_pvc_claim_name,
            dgxc_pvc_mount_path=args.dgxc_pvc_mount_path,
            custom_env_vars=args.custom_env_vars,
            container_image=args.container_image,
            namespace=args.dgxc_namespace,
            hf_token=args.hf_token,
        )

    executor.job_details = CustomJobDetailsRay()

    pre_ray_start = [
        "cp -a /nemo-workspace/Export-Deploy/. /opt/Export-Deploy/ || true",
        "cp -a /nemo-workspace/Megatron-Bridge/. /opt/Megatron-Bridge/ || true",
    ]

    job = RayJob(
        name="demo-slurm-ray-deploy",
        executor=executor,
    )
    job.start(
        command="bash /home/okoenig/.nemo_run/experiments/demo-slurm-ray-deploy/code/deploy.sh | tee -a deploy.log & bash /home/okoenig/.nemo_run/experiments/demo-slurm-ray-deploy/code/eval.sh | tee -a eval.log",
        workdir=os.path.dirname(os.path.abspath(__file__)),  # rsync'ed via SSH to the cluster_dir/code/
        pre_ray_start_commands=pre_ray_start,
    )
    job.logs(follow=True, timeout=10 * 60)

    register_pipeline_terminator(job=job)

    with open(
        os.path.join("/home/okoenig/.nemo_run/experiments/demo-slurm-ray-deploy", "results", "results.yml"), "r"
    ) as f:
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
            local_path=os.path.join(
                "/home/okoenig/.nemo_run/experiments/demo-slurm-ray-deploy", "results", "results.json"
            ),
            name="results.json",
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
