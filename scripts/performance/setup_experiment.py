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
from os.path import basename, splitext
from pathlib import Path

from .argument_parser import parse_cli_args
from .utils.executors import slurm_executor


try:
    import nemo_run as run

    HAS_NEMO_RUN = True
except ImportError:
    HAS_NEMO_RUN = False

if HAS_NEMO_RUN:
    from megatron.bridge.recipes.run_plugins import NsysPlugin, PerfEnvPlugin

import logging


logger: logging.Logger = logging.getLogger(__name__)

if __name__ == "__main__":
    args, _ = parse_cli_args()
    exp_name = f"{splitext(basename(__file__))[0]}_{args.compute_dtype}"

    SCRIPT_DIR: Path = Path(__file__).parent.resolve()
    RUN_SCRIPT_FILENAME: str = "run_script.py"
    RUN_SCRIPT_PATH: Path = SCRIPT_DIR / RUN_SCRIPT_FILENAME
    logger.info(f"Run script path: {RUN_SCRIPT_PATH}")
    if not RUN_SCRIPT_PATH.is_file():
        logger.error(f"Specified run script not found: {RUN_SCRIPT_PATH}")
        logger.error("Ensure the path passed to --run_script is correct.")
        sys.exit(1)
    config_filename = f"{args.model_name}_{args.model_size}_{args.task}_overrides.yaml"
    config_filepath = SCRIPT_DIR / "llm" / "configs" / config_filename
    logger.info(f"Config file path: {config_filepath}")
    if not config_filepath.is_file():
        logger.error(f"Specified YAML config file not found: {config_filepath}")
        logger.error("Ensure the path passed to --config_file is correct.")
        sys.exit(1)

    plugins = (
        [
            PerfEnvPlugin(
                enable_vboost=args.enable_vboost,
                nccl_pp_comm_chunksize=2097152 if args.model_size in ["70b", "405b"] else None,
                gpu_sm100_or_newer=args.gpu.lower() in ["b200", "gb200"],
            )
        ]
        if HAS_NEMO_RUN
        else []
    )
    if HAS_NEMO_RUN and args.enable_nsys and not args.localrun:
        plugins.append(NsysPlugin(profile_step_start=10, profile_step_end=11, profile_ranks=[0]))

    custom_mounts = args.custom_mounts + [
        f"{config_filepath}:{config_filepath}",
        f"{RUN_SCRIPT_PATH}:{RUN_SCRIPT_PATH}",
        f"{SCRIPT_DIR}:{SCRIPT_DIR}",
    ]
    logger.info(f"Custom mounts: {custom_mounts}")

    num_nodes = -(args.num_gpus // -args.gpus_per_node)
    if not args.localrun:
        executor = slurm_executor(
            args.gpu.lower(),
            args.account,
            args.partition,
            args.log_dir,
            num_nodes,
            args.gpus_per_node,
            args.time_limit,
            args.container_image,
            custom_mounts=custom_mounts,
            custom_env_vars={},
            hf_token=args.hf_token,
            nemo_home=args.nemo_home,
            wandb_key=args.wandb_key,
        )
    else:
        executor = run.LocalExecutor(ntasks_per_node=8, launcher="torchrun", env_vars={})

    target_script_args = [
        "--config_file",
        str(config_filepath),
    ]
    # Forward relevant args that run_script.py needs
    args_to_forward = ["model_name", "model_size", "compute_dtype", "fp8_recipe", "gpu"]
    for arg_name in args_to_forward:
        if hasattr(args, arg_name):
            arg_value = getattr(args, arg_name)
            if arg_value is not None:
                target_script_args.extend([f"--{arg_name}", str(arg_value)])
    target_script_args.extend(["-a", "dummy", "-p", "dummy"])

    train_script = run.Script(
        path=str(RUN_SCRIPT_PATH),
        entrypoint="python",
        args=target_script_args,
    )

    run.run(train_script, executor=executor, plugins=plugins, dryrun=args.dryrun, detach=True)
