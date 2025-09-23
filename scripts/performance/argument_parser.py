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
import os
from pathlib import Path

from nemo_run.config import get_nemorun_home


DEFAULT_NEMO_CACHE_HOME = Path.home() / ".cache" / "nemo"
DEFAULT_NEMO_HOME = os.getenv("NEMO_HOME", DEFAULT_NEMO_CACHE_HOME)


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
        choices=["h100", "b200", "gb200"],
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
        choices=["bf16", "fp8"],
        help="Compute precision. Options- bf16 or fp8. Defaults to bf16",
        required=False,
        default="bf16",
    )
    fp8_recipe_msg = (
        "FP8 recipe. Options- ds (per-tensor delayed scaling), cs (per-tensor current scaling), "
        "mxfp8, ss (subchannel scaling). Defaults to ds"
    )
    parser.add_argument(
        "-fr",
        "--fp8_recipe",
        type=str,
        choices=["ds", "cs", "mx", "ss"],
        help=fp8_recipe_msg,
        required=False,
        default="ds",
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

    def bool_arg(arg):
        if arg.lower() in ["true", "1", "t", "yes", "y"]:
            return True
        elif arg.lower() in ["false", "0", "f", "no", "n"]:
            return False
        else:
            raise ValueError(f"Invalid value for boolean argument: {arg}")

    def list_of_strings(arg):
        return arg.split(",")

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
        type=str,
        help="Model to use for experiment. Default: llama3",
        required=True,
    )
    parser.add_argument(
        "-s",
        "--model_size",
        type=str,
        help="Model size to use for experiment. Default: 8b",
        required=True,
    )
    parser.add_argument(
        "-en",
        "--enable_nsys",
        help="Enable Nsys profiling. Diabled by default",
        action="store_true",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        help="Path to the config yaml file to use for the experiment.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--domain",
        type=str,
        help="Domain to use for the experiment- llm, vlm, diffusion. Default: llm",
        required=False,
        default="llm",
    )

    args, cli_dotlist_overrides = parser.parse_known_args()
    return args, cli_dotlist_overrides
