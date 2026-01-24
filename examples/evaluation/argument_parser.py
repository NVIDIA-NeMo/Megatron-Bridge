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
import argparse


def list_of_strings(arg):
    """Split a comma-separated string into a list of substrings."""
    return arg.split(",")


def to_dict(arg):
    """Split a comma-separated string into a dictionary of key-value pairs."""
    return dict(item.split("=") for item in arg.split(","))


ENDPOINT_TYPES = {"chat": "chat/completions/", "completions": "completions/"}


def parse_cli_args():
    """Parse command line arguments for launching Megatron-Bridge Evaluation."""
    parser = argparse.ArgumentParser(description="Launch Megatron-Bridge Evaluation")
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Dry run the experiment.",
        default=False,
    )

    # Deployment args
    deployment_args = parser.add_argument_group("Deployment arguments")
    deployment_args.add_argument("--megatron_checkpoint", type=str, help="Megatron checkpoint to evaluate")
    deployment_args.add_argument(
        "--host",
        type=str,
        help="Server address to use for evaluation",
        default="0.0.0.0",
    )
    deployment_args.add_argument("--port", type=int, help="Server port to use for evaluation", default=8000)
    deployment_args.add_argument("--gpus_per_node", type=int, help="Number of GPUs per node", default=8)
    deployment_args.add_argument("--num_gpus", type=int, help="Number of nodes to use for evaluation", default=8)
    deployment_args.add_argument("--num_replicas", type=int, default=1, help="Num of replicas for Ray server")
    deployment_args.add_argument(
        "--tensor_model_parallel_size",
        type=int,
        help="Tensor model parallel size to use for evaluation",
        default=1,
    )
    deployment_args.add_argument(
        "--pipeline_model_parallel_size",
        type=int,
        help="Pipeline model parallel size to use for evaluation",
        default=1,
    )
    deployment_args.add_argument(
        "--context_model_parallel_size",
        type=int,
        help="Context model parallel size to use for evaluation",
        default=1,
    )

    # Evaluation args
    evaluation_args = parser.add_argument_group("Evaluation arguments")
    evaluation_args.add_argument(
        "--endpoint_type",
        type=str,
        default="completions",
        help="Whether to use completions or chat endpoint. Refer to the docs for details on tasks that are completions"
        "v/s chat.",
        choices=list(ENDPOINT_TYPES),
    )
    evaluation_args.add_argument(
        "--limit_samples",
        type=float,
        default=None,
        help="Limit evaluation to `limit` samples. Default: use all samples.",
    )
    evaluation_args.add_argument(
        "--parallelism",
        type=int,
        default=8,
        help="Number of parallel requests to send to server. Default: use default for the task.",
    )
    evaluation_args.add_argument(
        "--request_timeout",
        type=int,
        default=1000,
        help="Time in seconds for the eval client. Default: 1000s",
    )
    evaluation_args.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature for generation. Higher values = more random. Default: use task default.",
    )
    evaluation_args.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="Top-p (nucleus) sampling threshold. Default: use task default.",
    )
    evaluation_args.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Top-k sampling threshold. Default: use task default.",
    )
    evaluation_args.add_argument(
        "--eval_task",
        type=str,
        default="mmlu",
        help="Evaluation benchmark to run. Refer to the docs for more details on the tasks/benchmarks.",
    )

    # Slurm args
    slurm_args = parser.add_argument_group("Slurm arguments")
    slurm_args.add_argument(
        "--custom_mounts", type=list_of_strings, help="Comma separated string of mounts", default=[], required=False
    )
    slurm_args.add_argument(
        "--custom_env_vars",
        type=to_dict,
        help="Comma separated string of environment variables",
        default={},
        required=False,
    )
    slurm_args.add_argument("--account", type=str, help="Cluster account to run test")
    slurm_args.add_argument("--partition", type=str, help="Cluster partition to run test")
    slurm_args.add_argument("--time_limit", type=str, default="04:00:00", help="Time limit of run")
    slurm_args.add_argument("--container_image", type=str, default="", help="Container image to run")

    # Logging args
    logging_args = parser.add_argument_group("Logging arguments")
    logging_args.add_argument(
        "--output_dir",
        type=str,
        help="Output directory to save the results",
        required=False,
    )
    logging_args.add_argument(
        "--experiment_name",
        type=str,
        help="wandb job name",
        required=False,
    )
    logging_args.add_argument(
        "--wandb_key",
        type=str,
        help="wandb key. Needed for wandb logger projection to server",
        required=False,
    )
    logging_args.add_argument(
        "--wandb_project_name",
        type=str,
        help="wandb project name",
        required=False,
    )
    logging_args.add_argument(
        "--wandb_entity_name",
        type=str,
        help="wandb entity name",
        required=False,
    )
    logging_args.add_argument(
        "--wandb_experiment_name",
        type=str,
        help="wandb job name",
        required=False,
    )

    # Tokenizer args
    tokenizer_args = parser.add_argument_group("Tokenizer arguments")
    tokenizer_args.add_argument(
        "-hf",
        "--hf_token",
        type=str,
        help="HuggingFace token. Defaults to None. Required for accessing tokenizers and checkpoints.",
    )

    # DGXCloud
    dgxc_args = parser.add_argument_group("DGXCloud arguments")
    dgxc_args.add_argument(
        "--dgxc_cluster",
        type=str,
        help="DGXCloud cluster to use for experiment",
        required=False,
    )
    dgxc_args.add_argument(
        "--dgxc_base_url",
        type=str,
        help="DGXCloud base url",
        required=False,
    )
    dgxc_args.add_argument(
        "--dgxc_kube_apiserver_url",
        type=str,
        help="DGXCloud kube apiserver url",
        required=False,
    )
    dgxc_args.add_argument(
        "--dgxc_app_id",
        type=str,
        help="DGXCloud app id",
        required=False,
    )
    dgxc_args.add_argument(
        "--dgxc_app_secret",
        type=str,
        help="DGXCloud app secret",
        required=False,
    )
    dgxc_args.add_argument(
        "--dgxc_project_name",
        type=str,
        help="DGXCloud project name",
        required=False,
    )
    dgxc_args.add_argument(
        "--dgxc_pvc_claim_name",
        type=str,
        help="DGXCloud pvc claim name",
        required=False,
    )
    dgxc_args.add_argument(
        "--dgxc_pvc_mount_path",
        type=str,
        help="DGXCloud pvc mount path",
        required=False,
    )
    dgxc_args.add_argument(
        "--dgxc_namespace",
        type=str,
        help="DGXCloud namespace",
        required=False,
    )

    return parser
