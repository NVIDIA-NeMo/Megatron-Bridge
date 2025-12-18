#
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import logging
import os
from pathlib import Path

from nemo_run.config import get_nemorun_home


logger: logging.Logger = logging.getLogger(__name__)

DEFAULT_NEMO_HOME = os.getenv("NEMO_HOME", Path.home() / ".cache" / "nemo")
VALID_CUDA_GRAPH_IMPLS = ["none", "local", "transformer_engine"]
VALID_CUDA_GRAPH_SCOPES = ["full_iteration", "attn", "mlp", "moe", "moe_router", "moe_preprocess", "mamba"]


def list_of_strings(arg):
    """Split a comma-separated string into a list of substrings."""
    return arg.split(",")

def list_of_ints(arg):
    """Split a comma-separated string into a list of integers."""
    if arg is None:
        raise argparse.ArgumentTypeError("empty argument list")
    try:
        result = [int(p, 10) for p in list_of_strings(arg)]
    except Exception:
        raise argparse.ArgumentTypeError(f"invalid comma-separated integer list: {arg!r}") from None

    return result

def to_dict(arg):
    """Split a comma-separated string into a dictionary of key-value pairs."""
    return dict(item.split("=") for item in arg.split(","))


def lower_str(arg):
    """Lowercase a CLI string argument with a runtime type check."""
    assert isinstance(arg, str), f"Argument {arg} is not a string"
    return arg.lower()


def bool_arg(arg):
    """Convert a string CLI value to a boolean."""
    if arg.lower() in ["true", "1", "t", "yes", "y"]:
        return True
    elif arg.lower() in ["false", "0", "f", "no", "n"]:
        return False
    else:
        raise ValueError(f"Invalid value for boolean argument: {arg}")


def is_cuda_graph_impl_valid(arg):
    """Validate and normalize the CUDA graph implementation argument."""
    if arg in VALID_CUDA_GRAPH_IMPLS:
        return arg
    else:
        raise ValueError(f"Invalid value for cuda_graph_impl: {arg}. Valid options are: {VALID_CUDA_GRAPH_IMPLS}")


def is_cuda_graph_scope_valid(arg):
    """Validate the CUDA graph scope argument."""
    args = arg.split(",")
    if all(a in VALID_CUDA_GRAPH_SCOPES for a in args):
        return args
    else:
        raise ValueError(
            f"Invalid value for cuda_graph_scope: {arg}. Valid options are: {VALID_CUDA_GRAPH_SCOPES}. "
            "Comma separated list of scopes is allowed."
        )


def parse_additional_slurm_params(params_str):
    """
    Parse additional SLURM parameters from a string of key=value pairs.
    This function handles different separator formats:
    1. Semicolon-separated: "key1=value1;key2=value2" (recommended for multiple parameters)
    2. Space-separated: "key1=value1 key2=value2"
    3. Single parameter: "key1=value1,value2" (no separators = single parameter)
    Args:
        params_str (str): String with parameters
    Returns:
        dict: Dictionary of parameters, or None if params_str is None/empty
    Example:
        parse_additional_slurm_params("nodelist=node001,node002")
        returns {"nodelist": "node001,node002"}
        parse_additional_slurm_params("nodelist=node001,node002;constraint=gpu")
        returns {"nodelist": "node001,node002", "constraint": "gpu"}
        parse_additional_slurm_params("reservation=my_res;constraint=gpu")
        returns {"reservation": "my_res", "constraint": "gpu"}
    """
    if not params_str:
        return None

    params = {}

    # Try semicolon separation first (most reliable for complex values)
    if ";" in params_str:
        parts = params_str.split(";")
    # Try space separation next
    elif " " in params_str:
        parts = params_str.split()
    # No separators found - treat as single parameter
    else:
        parts = [params_str]

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if "=" in part:
            key, value = part.split("=", 1)
            params[key.strip()] = value.strip()
        else:
            # Boolean flag (no value)
            params[part] = True

    return params if params else None


def parse_cli_args():
    """
    Command line arguments for running pre-training and fine-tuning experiments.
    """
    parser = argparse.ArgumentParser(
        description="Megatron-Bridge Pretraining and Fine-Tuning",
        argument_default=None,
    )
    parser.add_argument(
        "-m",
        "--model_family_name",
        type=lower_str,
        help="Model family name to use for experiment. E.g. `--model_family_name llama` (not llama3)",
        required=True,
    )
    parser.add_argument(
        "-mr",
        "--model_recipe_name",
        type=lower_str,
        help="Model recipe name to use for experiment. E.g. `--model_recipe_name llama31_405b`",
        required=True,
    )
    parser.add_argument(
        "--use_recipes",
        action="store_true",
        help="Use library recipes. Disabled by default.",
        default=False,
    )
    parser.add_argument(
        "-hf",
        "--hf_token",
        type=str,
        help="HuggingFace token. Defaults to None. Required for accessing tokenizers and checkpoints.",
    )
    parser.add_argument(
        "-nh",
        "--nemo_home",
        type=str,
        help=" ".join(
            [
                "Sets env var `NEMO_HOME` (on compute node using sbatch script)- directory where NeMo searches",
                "for models and checkpoints. This saves a lot of time (especially for bigger models) if checkpoints already",
                f"exist here. Missing files will be downloaded here from HuggingFace. Defaults to {DEFAULT_NEMO_HOME}",
            ]
        ),
        default=DEFAULT_NEMO_HOME,
    )
    parser.add_argument(
        "--detach",
        help="Detach the experiment from the terminal. Enabled by default",
        type=bool_arg,
        default=True,
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        help="Maximum number of retries. Defaults to 2",
        default=2,
    )
    parser.add_argument(
        "-ng",
        "--num_gpus",
        type=int,
        help="Number of gpus.",
        required=True,
    )

    # Training
    training_args = parser.add_argument_group("Training arguments")
    training_args.add_argument(
        "--task",
        choices=["pretrain", "sft", "lora"],
        help="Task to run. Defaults to 'pretrain'",
        default="pretrain",
    )
    training_args.add_argument(
        "-ms",
        "--max_steps",
        type=int,
        help="Maximum number of steps to run the experiment for. Defaults to 50.",
    )
    training_args.add_argument(
        "-gb",
        "--global_batch_size",
        type=int,
    )
    training_args.add_argument(
        "-mb",
        "--micro_batch_size",
        type=int,
    )
    training_args.add_argument(
        "-sl",
        "--seq_length",
        type=int,
    )

    # Optimizer
    optimizer_args = parser.add_argument_group("Optimizer arguments")
    optimizer_args.add_argument("--lr", type=float, help="Learning rate")
    optimizer_args.add_argument("--min_lr", type=float, help="Minimum learning rate")
    optimizer_args.add_argument("--warmup_iters", type=int, help="Warmup iterations", default=10)

    # Checkpointing
    checkpointing_args = parser.add_argument_group("Checkpointing arguments")
    checkpointing_args.add_argument("--pretrained_checkpoint", type=str, help="Path to pretrained checkpoint")
    checkpointing_args.add_argument("--save_dir", type=str, help="Directory to save checkpoints")
    checkpointing_args.add_argument("--load_dir", type=str, help="Directory to load checkpoints")
    checkpointing_args.add_argument("--save_interval", type=int, help="Number of iterations between checkpoint saves")
    checkpointing_args.add_argument("--most_recent_k", type=int, help="Number of latest checkpoints to keep")
    checkpointing_args.add_argument(
        "--save_config_filepath", type=str, help="Path to save the task configuration file"
    )

    # Data
    data_args = parser.add_argument_group("Data arguments")
    data_args.add_argument(
        "--data",
        type=str,
        default="mock",
        choices=["mock", "rp2", "squad", "squad_packed"],
        help="Dataset type to use",
    )
    data_args.add_argument("--dataset_paths", nargs="*", help="Dataset paths (for rp2 dataset)")
    data_args.add_argument("--dataset_root", type=str, help="Dataset root directory (for squad datasets)")
    parser.add_argument("--index_mapping_dir", type=str, help="Index mapping directory (for rp2 dataset)")
    data_args.add_argument("--dataset_name", type=str, help="Dataset name (deprecated)")
    data_args.add_argument("--packed_sequence", action="store_true", help="Use packed sequences")
    data_args.add_argument("--head_only", action="store_true", help="Use only head data (for rp2 dataset)")

    # Tokenizer configuration
    tokenizer_args = parser.add_argument_group("Tokenizer arguments")
    data_args.add_argument(
        "--tokenizer_type",
        type=str,
        choices=["NullTokenizer", "HuggingFaceTokenizer", "SentencePieceTokenizer"],
    )
    tokenizer_args.add_argument(
        "--tokenizer_model", type=str, help="Path to tokenizer model (automatically provided by launcher)"
    )
    tokenizer_args.add_argument("--vocab_size", type=int, default=32000, help="Vocabulary size for NullTokenizer")

    # Parallelism
    parallelism_args = parser.add_argument_group("Parallelism arguments")
    parallelism_args.add_argument(
        "-tp",
        "--tensor_model_parallel_size",
        type=int,
        help="Intra-layer model parallelism. Splits tensors across GPU ranks.",
    )
    parallelism_args.add_argument(
        "-pp",
        "--pipeline_model_parallel_size",
        type=int,
        help="Inter-layer model parallelism. Splits transformer layers across GPU ranks.",
    )
    parallelism_args.add_argument(
        "-cp",
        "--context_parallel_size",
        type=int,
        help="Splits network input along sequence dimension across GPU ranks.",
    )
    parallelism_args.add_argument(
        "-vp",
        "--virtual_pipeline_model_parallel_size",
        type=lambda x: None if x == "None" else int(x),
        help="Number of virtual blocks per pipeline model parallel rank is the virtual model parallel size.",
        default=-1,
    )
    parallelism_args.add_argument(
        "-ep",
        "--expert_model_parallel_size",
        type=int,
        help="Distributes Moe Experts across sub data parallel dimension.",
    )
    parallelism_args.add_argument(
        "-et",
        "--expert_tensor_parallel_size",
        type=lambda x: int(x) if x is not None else None,
        nargs="?",
        const=None,
        help="Intra-layer tensor model parallelsm for expert layer. Splits tensors across GPU ranks.\
            Use -et/--expert_tensor_parallel_size <space> for None or -et/--expert_tensor_parallel_size <int>",
    )

    # Slurm
    slurm_args = parser.add_argument_group("Slurm arguments")
    slurm_args.add_argument(
        "-a",
        "--account",
        type=str,
        help="Slurm account to use for experiment",
    )
    slurm_args.add_argument(
        "-p",
        "--partition",
        type=str,
        help="Slurm partition to use for experiment",
    )
    slurm_args.add_argument(
        "-t",
        "--time_limit",
        type=str,
        help="Maximum time limit to run experiment for. Defaults to 30 minutes (format- 'HH:MM:SS')",
        default="00:30:00",
    )
    slurm_args.add_argument(
        "-gn",
        "--gpus_per_node",
        type=int,
        help="Number of gpus per node. Defaults to 8",
        default=8,
    )
    slurm_args.add_argument(
        "-i",
        "--container_image",
        type=str,
        help=" ".join(
            [
                "NeMo container to use for experiment. Defaults to latest dev container- 'nvcr.io/nvidia/nemo:dev'",
                "Make sure your NGC credentials are accessible in your environment.",
            ]
        ),
        default="nvcr.io/nvidia/nemo:dev",
    )
    slurm_args.add_argument(
        "-cm",
        "--custom_mounts",
        type=list_of_strings,
        help="Comma separated string of mounts",
        default=[],
    )
    slurm_args.add_argument(
        "-ce",
        "--custom_env_vars",
        type=to_dict,
        help="Comma separated string of environment variables",
        default=[],
    )
    slurm_args.add_argument(
        "-cs",
        "--custom_srun_args",
        type=list_of_strings,
        help="Comma separated string of srun arguments",
        default=[],
    )
    slurm_args.add_argument(
        "--gres",
        type=str,
        help="Slurm generic resources to request (e.g., 'gpu:4').",
        required=False,
        default=None,
    )
    slurm_args.add_argument(
        "--additional_slurm_params",
        type=parse_additional_slurm_params,
        help="Additional SLURM parameters as key=value pairs. "
        "Use semicolons (;) to separate parameters when values contain commas. "
        "Examples: 'nodelist=node001,node002;constraint=gpu' or 'reservation=my_res;exclusive'",
        required=False,
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

    # For performance
    performance_args = parser.add_argument_group("Performance arguments")
    performance_args.add_argument(
        "-g",
        "--gpu",
        type=str,
        choices=["h100", "b200", "gb200", "gb300"],
        help="Target gpu type.",
        required=False,
    )
    performance_args.add_argument(
        "-c",
        "--compute_dtype",
        type=str,
        choices=["bf16", "fp8_cs", "fp8_mx", "fp8_sc", "nvfp4"],
        help="Compute precision. Options- bf16 or fp8. Defaults to bf16",
        required=False,
        default="bf16",
    )
    performance_args.add_argument(
        "-vb",
        "--enable_vboost",
        help="Enable VBoost which steers more power towards tensor cores. Disabled by default",
        type=bool_arg,
        required=False,
    )
    performance_args.add_argument(
        "-en",
        "--enable_nsys",
        help="Enable Nsys profiling. Disabled by default",
        action="store_true",
    )
    performance_args.add_argument(
        "--profiling_start_step",
        type=int,
        help="Defines start step for profiling",
        required=False,
        default=10,
    )
    performance_args.add_argument(
        "--profiling_stop_step",
        type=int,
        help="Defines stop step for profiling",
        required=False,
        default=11,
    )
    performance_args.add_argument(
        "--profiling_gpu_metrics",
        help="Enable nsys gpu metrics. Disabled by default.",
        action="store_true",
    )
    performance_args.add_argument(
        "--profiling_ranks",
        type=list_of_ints,
        help="List of ranks to target for profiling (defaults to just first rank)",
        required=False,
        default=None
    )
    performance_args.add_argument(
        "--use_tokendrop",
        help="Use token drop. Disabled by default. Currently only supported for DeepSeek v3",
        type=bool_arg,
        required=False,
    )
    performance_args.add_argument(
        "--use_megatron_fsdp",
        help="Use Megatron FSDP. Disabled by default.",
        type=bool_arg,
        required=False,
    )
    performance_args.add_argument(
        "--cuda_graph_impl",
        help=f"Cuda graph implementation. Options- {', '.join(VALID_CUDA_GRAPH_IMPLS)}.",
        type=is_cuda_graph_impl_valid,
        required=False,
    )
    performance_args.add_argument(
        "--cuda_graph_scope",
        help=f"Cuda graph scope. Options- {VALID_CUDA_GRAPH_SCOPES}. Comma separated list of scopes is allowed.",
        type=is_cuda_graph_scope_valid,
        required=False,
    )
    performance_args.add_argument(
        "--moe_a2a_overlap",
        type=bool_arg,
        required=False,
    )
    performance_args.add_argument(
        "-rl",
        "--recompute_num_layers",
        type=int,
        help="Number of Transformer layers to recompute, where all the intermediate "
        "activations of a Transformer layer are computed. Defaults to None",
        required=False,
    )
    performance_args.add_argument(
        "-ol",
        "--activation_offload_layers",
        type=int,
        help="Number of Transformer layers to offload to the CPU memory. Defaults to None",
        required=False,
    )
    performance_args.add_argument(
        "--recompute_modules",
        type=list_of_strings,
        help="Comma separated list of modules to recompute. Defaults to None",
        required=False,
    )

    # Logging
    logging_args = parser.add_argument_group("Logging arguments")
    logging_args.add_argument(
        "-wdk",
        "--wandb_key",
        type=str,
        help="wandb key. Needed for wandb logger projection to server",
        required=False,
    )
    logging_args.add_argument(
        "-wdp",
        "--wandb_project_name",
        type=str,
        help="wandb project name",
        required=False,
    )
    logging_args.add_argument(
        "-wde",
        "--wandb_entity_name",
        type=str,
        help="wandb project name",
        required=False,
    )
    logging_args.add_argument(
        "-wdj",
        "--wandb_experiment_name",
        type=str,
        help="wandb job name",
        required=False,
    )
    logging_args.add_argument(
        "-wds",
        "--wandb_save_dir",
        type=str,
        help="wandb save directory",
        required=False,
    )
    logging_args.add_argument(
        "-l",
        "--log_dir",
        type=str,
        help=f"Directory for logging experiment results. Defaults to {get_nemorun_home()} or NEMORUN_HOME envvar",
        required=False,
        default=None,
    )

    parser.add_argument(
        "-d",
        "--dryrun",
        help="If true, prints sbatch script to terminal without launching experiment.",
        required=False,
        action="store_true",
    )

    # Testing parameters
    testing_args = parser.add_argument_group("Testing arguments")
    testing_args.add_argument(
        "--is_long_convergence_run",
        action="store_true",
        help="If true, runs a long convergence run.",
        required=False,
        default=False,
    )
    testing_args.add_argument(
        "--golden_values_path",
        type=str,
        help="Path to golden values file",
        required=False,
    )
    testing_args.add_argument(
        "--timing_threshold", type=float, default=0.05, help="Step timing validation threshold (default: 0.05 = 5%%)"
    )
    testing_args.add_argument(
        "--skip_first_percent_time",
        type=float,
        default=0.70,
        help="Percentage of iterations to skip for timing comparison (default: 0.75 = 75%%)",
    )

    # Convergence loss validation parameters
    testing_args.add_argument(
        "--correlation_threshold", type=float, default=0.95, help="Correlation threshold for loss curve validation"
    )
    testing_args.add_argument(
        "--high_loss_tolerance", type=float, default=0.10, help="Tolerance for high loss values (>2.0)"
    )
    testing_args.add_argument(
        "--medium_loss_tolerance", type=float, default=0.05, help="Tolerance for medium loss values (0.5-2.0)"
    )
    testing_args.add_argument(
        "--low_loss_tolerance", type=float, default=0.02, help="Tolerance for low loss values (<0.5)"
    )
    testing_args.add_argument(
        "--final_loss_tolerance", type=float, default=0.05, help="Tolerance for final loss value"
    )
    testing_args.add_argument("--max_outlier_ratio", type=float, default=0.1, help="Maximum ratio of outliers allowed")
    testing_args.add_argument(
        "--outlier_threshold", type=float, default=3.0, help="Outlier detection threshold (sigma)"
    )
    testing_args.add_argument(
        "--skip_first_percent_loss",
        type=float,
        default=0.20,
        help="Percentage of loss points to skip from beginning for convergence analysis",
    )

    return parser
