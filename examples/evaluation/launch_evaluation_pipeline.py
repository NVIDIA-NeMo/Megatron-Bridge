# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
import time
from dataclasses import dataclass

import yaml
from nemo_run.core.execution.slurm import SlurmJobDetails
from nemo_run.run.ray.job import RayJob


try:
    import wandb

    HAVE_WANDB = True
except (ImportError, ModuleNotFoundError):
    HAVE_WANDB = False
    wandb = None

try:
    from argument_parser import parse_cli_args
    from utils.executors import kuberay_executor, slurm_executor
except (ImportError, ModuleNotFoundError):
    from .argument_parser import parse_cli_args
    from .utils.executors import kuberay_executor, slurm_executor

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def register_pipeline_terminator(job: RayJob):
    """Register a signal handler to terminate the job."""

    def sigterm_handler(_signo, _stack_frame):
        logger.info(f"Trying to terminate job {job.name}")
        job.stop()
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


def main(args):
    """Deploys the inference and evaluation server with NemoRun."""

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

    job = RayJob(
        name="demo-slurm-ray-deploy",
        executor=executor,
    )
    job.start(
        command=f"""
        bash /opt/Megatron-Bridge/examples/evaluation/deploy.sh \
            {args.megatron_checkpoint} \
            {args.num_replicas} \
            {args.num_gpus}| tee -a deploy.log & \
        sleep 120; \
        bash /opt/Megatron-Bridge/examples/evaluation/eval.sh \
            {args.output_dir} \
            {args.parallelism} | tee -a eval.log
        """,
        workdir=None,
    )

    register_pipeline_terminator(job=job)

    job_deployment_status = "Initializing"
    job_status = "UNKNOWN"
    while job_deployment_status != "Running" or job_status != "RUNNING":
        status = job.status(display=False)
        job_deployment_status = status["jobDeploymentStatus"]
        job_status = status["jobStatus"]
        time.sleep(1)
        if job_deployment_status == "Failed":
            raise RuntimeError("Job failed")

    job.logs(follow=True, timeout=10 * 60 * 60)
    job.stop()

    with open(os.path.join(args.output_dir, "results", "results.yml"), "r") as f:
        results = yaml.safe_load(f)

    logger.info("Results: %s", results)

    if HAVE_WANDB and args.wandb_key:
        wandb.login(key=args.wandb_key)
        api = wandb.Api()
        runs = api.runs(
            path=f"{args.wandb_entity_name}/{args.wandb_project_name}",
            filters={"display_name": args.wandb_experiment_name},
        )

        if runs:
            run_id = runs[0].id
            print(f"Found run with ID: {run_id}")

        wandb_run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity_name,
            id=run_id,
            resume="allow",
        )
        artifact = wandb.Artifact(name="evaluation_results", type="evaluation_results")
        artifact.add_file(
            local_path=os.path.join(args.output_dir, "results", "results.yml"),
            name="results.yml",
        )
        wandb_run.log_artifact(artifact)

        for category in ["tasks", "groups"]:
            for task_or_group_name, result in results["results"][category].items():
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
