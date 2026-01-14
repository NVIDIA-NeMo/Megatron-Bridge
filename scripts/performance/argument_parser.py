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
import logging
import os
from pathlib import Path

from nemo_run.config import get_nemorun_home


logger: logging.Logger = logging.getLogger(__name__)

DEFAULT_NEMO_CACHE_HOME = Path.home() / ".cache" / "nemo"
DEFAULT_NEMO_HOME = os.getenv("NEMO_HOME", DEFAULT_NEMO_CACHE_HOME)

VALID_CUDA_GRAPH_IMPLS = ["none", "local", "transformer_engine"]
VALID_CUDA_GRAPH_SCOPES = ["full_iteration", "attn", "mlp", "moe", "moe_router", "moe_preprocess", "mamba"]


def bool_arg(arg):
    """Convert a string CLI value to a boolean."""
    if arg.lower() in ["true", "1", "t", "yes", "y"]:
        return True
    elif arg.lower() in ["false", "0", "f", "no", "n"]:
        return False
    else:
        raise ValueError(f"Invalid value for boolean argument: {arg}")


def list_of_strings(arg):
    """Split a comma-separated string into a list of substrings."""
    return arg.split(",")


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


def lower_str(arg):
    """Lowercase a CLI string argument with a runtime type check."""
    assert isinstance(arg, str), f"Argument {arg} is not a string"
    return arg.lower()


def parse_cli_args():
    """
    Command line arguments correspong to Slurm cluster and NeMo2.0 for running pre-training and
    fine-tuning experiments.
    """
    parser = argparse.ArgumentParser(
        description="NeMo2.0 Performance Pretraining and Fine-Tuning",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "-a",
        "--account",
        type=str,
        help="Slurm account to use for experiment",
        required=True,
    )
    parser.add_argument(
        "-p",
        "--partition",
        type=str,
        help="Slurm partition to use for experiment",
        required=True,
    )
    parser.add_argument(
        "-g",
        "--gpu",
        type=str,
        choices=["h100", "b200", "gb200", "gb300"],
        help="Target gpu type.",
        required=True,
    )
    parser.add_argument(
        "-l",
        "--log_dir",
        type=str,
        help=f"Directory for logging experiment results. Defaults to {get_nemorun_home()}",
        required=False,
        default=get_nemorun_home(),
    )
    parser.add_argument(
        "-t",
        "--time_limit",
        type=str,
        help="Maximum time limit to run experiment for. Defaults to 30 minutes (format- 'HH:MM:SS')",
        required=False,
        default="00:30:00",
    )
    container_img_msg = [
        "NeMo container to use for experiment. Defaults to latest dev container- 'nvcr.io/nvidia/nemo:dev'",
        "Make sure your NGC credentials are accessible in your environment.",
    ]
    parser.add_argument(
        "-i",
        "--container_image",
        type=str,
        help=" ".join(container_img_msg),
        required=False,
        default="nvcr.io/nvidia/nemo:dev",
    )
    parser.add_argument(
        "-c",
        "--compute_dtype",
        type=str,
        choices=["bf16", "fp8_cs", "fp8_mx", "fp8_sc", "nvfp4"],
        help="Compute precision. Options- bf16 or fp8. Defaults to bf16",
        required=False,
        default="bf16",
    )
    parser.add_argument(
        "--task",
        choices=["pretrain", "sft", "lora"],
        help="Task to run. Defaults to 'pretrain'",
        default="pretrain",
    )
    parser.add_argument(
        "-hf",
        "--hf_token",
        type=str,
        help="HuggingFace token. Defaults to None. Required for accessing tokenizers and checkpoints.",
        default=None,
    )
    nemo_home_msg = [
        "Sets env var `NEMO_HOME` (on compute node using sbatch script)- directory where NeMo searches",
        "for models and checkpoints. This saves a lot of time (especially for bigger models) if checkpoints already",
        f"exist here. Missing files will be downloaded here from HuggingFace. Defaults to {DEFAULT_NEMO_HOME}",
    ]
    parser.add_argument(
        "-nh",
        "--nemo_home",
        type=str,
        help=" ".join(nemo_home_msg),
        default=DEFAULT_NEMO_HOME,
    )
    parser.add_argument(
        "-wdk",
        "--wandb_key",
        type=str,
        help="wandb key. Needed for wandb logger projetion to server",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-wdp",
        "--wandb_prj_name",
        type=str,
        help="wandb project name",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-wdj",
        "--wandb_exp_name",
        type=str,
        help="wandb job name",
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
    parser.add_argument(
        "-ng",
        "--num_gpus",
        type=int,
        help="Number of gpus.",
        required=True,
    )
    parser.add_argument(
        "-gn",
        "--gpus_per_node",
        type=int,
        help="Number of gpus per node. Defaults to 8",
        required=False,
        default=8,
    )

    parser.add_argument(
        "-cm",
        "--custom_mounts",
        type=list_of_strings,
        help="Comma separated string of mounts",
        required=False,
        default=[],
    )
    parser.add_argument(
        "-vb",
        "--enable_vboost",
        help="Enable VBoost which steers more power towards tensor cores. Disabled by default",
        type=bool_arg,
        required=False,
        default=None,
    )
    parser.add_argument(
        "-m",
        "--model_name",
        type=lower_str,
        help="Model to use for experiment.",
        required=True,
    )
    parser.add_argument(
        "-s",
        "--model_size",
        type=lower_str,
        help="Model size to use for experiment.",
        required=True,
    )
    parser.add_argument(
        "-en",
        "--enable_nsys",
        help="Enable Nsys profiling. Diabled by default",
        action="store_true",
    )
    parser.add_argument(
        "--domain",
        type=str,
        help="Domain to use for the experiment- llm, vlm, diffusion. Default: llm",
        required=False,
        default="llm",
    )
    parser.add_argument(
        "--use_tokendrop",
        help="Use token drop. Disabled by default. Currently only supported for DeepSeek v3",
        type=bool_arg,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--use_megatron_fsdp",
        help="Use Megatron FSDP. Disabled by default.",
        type=bool_arg,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--cuda_graph_impl",
        help=f"Cuda graph implementation. Options- {', '.join(VALID_CUDA_GRAPH_IMPLS)}.",
        type=is_cuda_graph_impl_valid,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--cuda_graph_scope",
        help=f"Cuda graph scope. Options- {VALID_CUDA_GRAPH_SCOPES}. Comma separated list of scopes is allowed.",
        type=is_cuda_graph_scope_valid,
        required=False,
        default=None,
    )
    parser.add_argument(
        "-tp",
        "--tensor_model_parallel_size",
        type=int,
        help="Intra-layer model parallelism. Splits tensors across GPU ranks.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-pp",
        "--pipeline_model_parallel_size",
        type=int,
        help="Inter-layer model parallelism. Splits transformer layers across GPU ranks.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-cp",
        "--context_parallel_size",
        type=int,
        help="Splits network input along sequence dimension across GPU ranks.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-vp",
        "--virtual_pipeline_model_parallel_size",
        type=lambda x: None if x == "None" else int(x),
        help="Number of virtual blocks per pipeline model parallel rank is the virtual model parallel size.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-ep",
        "--expert_model_parallel_size",
        type=int,
        help="Distributes Moe Experts across sub data parallel dimension.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-et",
        "--expert_tensor_parallel_size",
        type=lambda x: int(x) if x is not None else None,
        nargs="?",
        const=None,
        help="Intra-layer tensor model parallelsm for expert layer. Splits tensors across GPU ranks.\
            Use -et/--expert_tensor_parallel_size <space> for None or -et/--expert_tensor_parallel_size <int>",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-mb",
        "--micro_batch_size",
        type=int,
        required=False,
        default=None,
    )
    parser.add_argument(
        "-gb",
        "--global_batch_size",
        type=int,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--moe_a2a_overlap",
        type=bool_arg,
        required=False,
        default=None,
    )
    parser.add_argument(
        "-ms",
        "--max_steps",
        type=int,
        help="Maximum number of steps to run the experiment for. Defaults to 50.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-rl",
        "--recompute_num_layers",
        type=int,
        help="Number of Transformer layers to recompute, where all the intermediate "
        "activations of a Transformer layer are computed. Defaults to None",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-ol",
        "--activation_offload_layers",
        type=int,
        help="Number of Transformer layers to offload to the CPU memory. Defaults to None",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--recompute_modules",
        type=list_of_strings,
        help="Comma separated list of modules to recompute. Defaults to None",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--megatron_ckpt",
        type=str,
        help=" ".join(
            [
                "Megatron checkpoint directory to use for LoRA. Defaults to None.",
                "Must be in Megatron checkpoint format and required for LoRA.",
            ]
        ),
        required=False,
        default=None,
    )
    parser.add_argument(
        "--detach",
        help="Detach the experiment from the terminal. Disabled by default",
        action="store_true",
        dest="detach",
        default=True,
    )
    parser.add_argument(
        "--no-detach",
        help="Do not detach the experiment from the terminal. Enabled by default",
        action="store_false",
        dest="detach",
    )
    parser.add_argument(
        "--profiling_start_step", type=int, help="Defines start step for profiling", required=False, default=10
    )
    parser.add_argument(
        "--profiling_stop_step", type=int, help="Defines stop step for profiling", required=False, default=11
    )
    parser.add_argument(
        "--profiling_gpu_metrics",
        help="Enable nsys gpu metrics. Disabled by default.",
        action="store_true",
    )
    parser.add_argument(
        "-cps",
        "--checkpoint_save",
        type=bool_arg,
        help="When enabled will trigger checkpoint save operation at the end of training",
        required=False,
        default=None, 
    )
    parser.add_argument(
        "-cpi",
        "--checkpoint_interval",
        type=int,
        help="Number of iterations between checkpoint saves. Defaults to 500. Set to smaller value for shorter runs.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-cpd",
        "--checkpoint_dir",
        type=str,
        help="Directory where checkpoints will be saved. If not specified, checkpoints will be saved into default location under experiment folder (/nemo_run/code/nemo_experiments/default/checkpoints).",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-cpl", 
        "--checkpoint_load_path",
        type=str,
        help="Path to checkpoint to load prior to training start",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--additional_slurm_params",
        type=str,
        help="Additional SLURM parameters as key=value pairs. "
        "Use semicolons (;) to separate parameters when values contain commas. "
        "Examples: 'nodelist=node001,node002;constraint=gpu' or 'reservation=my_res;exclusive'",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        help="Hidden size to use for the experiment. Defaults to None.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        help="Number of layers to use for the experiment. Defaults to None.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--pipeline_model_parallel_layout",
        type=str,
        help="Pipeline model parallel layout to use for the experiment. Defaults to None.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--first_k_dense_replace",
        type=int,
        help="Number of MoE layers to be converted to dense layers. Defaults to None.",
        required=False,
        default=None,
    )
    args, cli_dotlist_overrides = parser.parse_known_args()
    return args, cli_dotlist_overrides


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
