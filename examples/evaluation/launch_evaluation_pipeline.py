#!/usr/bin/env python3
"""
Launch Megatron-Bridge Evaluation

Parse arguments early to catch unknown args before other libraries
(like nemo_run) can consume them during import.
"""

import argparse


try:
    from utils.executors import slurm_executor
except (ImportError, ModuleNotFoundError):
    from .utils.executors import slurm_executor


ENDPOINT_TYPES = {"chat": "chat/completions/", "completions": "completions/"}


def list_of_strings(arg):
    """Split a comma-separated string into a list of substrings."""
    return arg.split(",")


def to_dict(arg):
    """Split a comma-separated string into a dictionary of key-value pairs."""
    return dict(item.split("=") for item in arg.split(","))


def get_parser():
    parser = argparse.ArgumentParser(description="Launch Megatron-Bridge Evaluation")

    parser.add_argument("--megatron_checkpoint", type=str, help="Megatron checkpoint to evaluate")
    parser.add_argument(
        "--host",
        type=str,
        help="Server address to use for evaluation",
        default="0.0.0.0",
    )
    parser.add_argument("--port", type=int, help="Server port to use for evaluation", default=8000)
    parser.add_argument("--gpus_per_node", type=int, help="Number of GPUs per node", default=8)
    parser.add_argument("--num_gpus", type=int, help="Number of nodes to use for evaluation", default=8)
    parser.add_argument("--num_replicas", type=int, default=1, help="Num of replicas for Ray server")
    parser.add_argument(
        "--tensor_model_parallel_size",
        type=int,
        help="Tensor model parallel size to use for evaluation",
        default=1,
    )
    parser.add_argument(
        "--pipeline_model_parallel_size",
        type=int,
        help="Pipeline model parallel size to use for evaluation",
        default=1,
    )
    parser.add_argument(
        "--context_model_parallel_size",
        type=int,
        help="Context model parallel size to use for evaluation",
        default=1,
    )
    parser.add_argument(
        "--endpoint_type",
        type=str,
        default="completions",
        help="Whether to use completions or chat endpoint. Refer to the docs for details on tasks that are completions"
        "v/s chat.",
        choices=list(ENDPOINT_TYPES),
    )
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help="Limit evaluation to `limit` samples. Default: use all samples.",
    )
    parser.add_argument(
        "--parallel_requests",
        type=int,
        default=8,
        help="Number of parallel requests to send to server. Default: use default for the task.",
    )
    parser.add_argument(
        "--request_timeout",
        type=int,
        default=1000,
        help="Time in seconds for the eval client. Default: 1000s",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature for generation. Higher values = more random. Default: use task default.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="Top-p (nucleus) sampling threshold. Default: use task default.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Top-k sampling threshold. Default: use task default.",
    )
    parser.add_argument(
        "--eval_task",
        type=str,
        default="mmlu",
        help="Evaluation benchmark to run. Refer to the docs for more details on the tasks/benchmarks.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        help="Optional tag for your experiment title which will be appended after the model/exp name.",
        required=False,
        default="",
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Dry run the experiment.",
        default=False,
    )
    parser.add_argument(
        "--custom_mounts", type=list_of_strings, help="Comma separated string of mounts", default=[], required=False
    )
    parser.add_argument(
        "--custom_env_vars",
        type=to_dict,
        help="Comma separated string of environment variables",
        default={},
        required=False,
    )
    parser.add_argument("--account", type=str, help="Cluster account to run test")
    parser.add_argument("--partition", type=str, help="Cluster partition to run test")
    parser.add_argument("--time_limit", type=str, default="04:00:00", help="Time limit of run")
    parser.add_argument("--container_image", type=str, default="", help="Container image to run")
    parser.add_argument("--hf_token", type=str, help="HuggingFace token", default=None)
    parser.add_argument("--wandb_key", type=str, help="WandB key", default=None)
    return parser


# Parse arguments EARLY, before importing nemo_run/nemo_evaluator which may
# consume sys.argv during import and silently ignore unknown arguments.
if __name__ == "__main__":
    _ARGS = get_parser().parse_args()
else:
    _ARGS = None

# Now safe to import libraries that may parse sys.argv during import
import nemo_run as run


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
output_dir = "/results/"

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
    # Deployment script
    commons_args = {
        "megatron_checkpoint": args.megatron_checkpoint,
        "host": args.host,
        "port": args.port,
        "num_gpus": args.num_gpus,
        "num_replicas": args.num_replicas,
        "tensor_model_parallel_size": args.tensor_model_parallel_size,
        "pipeline_model_parallel_size": args.pipeline_model_parallel_size,
        "context_model_parallel_size": args.context_model_parallel_size,
    }

    deploy_script = RAY_DEPLOY_SCRIPT.format(**commons_args)
    deploy_run_script = run.Script(inline=deploy_script, metadata={"use_with_ray_cluster": True})

    print("Deploy script: %s", deploy_script)

    # Evaluation script - using inline script to avoid serialization issues
    eval_script = EVAL_SCRIPT.format(
        endpoint_url=f"http://{args.host}:{args.port}/v1/{ENDPOINT_TYPES[args.endpoint_type]}",
        endpoint_type=args.endpoint_type,
        eval_task=args.eval_task,
        limit_samples=args.limit if args.limit is not None else "None",
        parallelism=args.parallel_requests,
        request_timeout=args.request_timeout,
        temperature=args.temperature if args.temperature is not None else "None",
        top_p=args.top_p if args.top_p is not None else "None",
        top_k=args.top_k if args.top_k is not None else "None",
    )
    eval_run_script = run.Script(inline=eval_script)

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
        wandb_key=args.wandb_key,
    )

    print(executor)
    print(deploy_run_script)

    executor_eval = executor.clone()
    executor_eval.srun_args = [
        "--mpi=none",
        "--ntasks-per-node=1",
        "--nodes=1",
    ]

    exp_name = "Megatron-Bridge-Evaluation"
    with run.Experiment(f"{exp_name}-{args.tag}") as exp:
        exp.add(
            [deploy_run_script, eval_run_script],
            executor=[executor, executor_eval],
            name=exp_name,
            tail_logs=False,
        )
        if args.dryrun:
            exp.dryrun()
        else:
            exp.run()


if __name__ == "__main__":
    main(_ARGS)
