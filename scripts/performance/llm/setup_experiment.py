import sys
from os.path import basename, splitext

from ..argument_parser import parse_cli_args
from ..executors import slurm_executor

try:
    import nemo_run as run

    HAS_NEMO_RUN = True
except ImportError:
    HAS_NEMO_RUN = False

if HAS_NEMO_RUN:
    from megatron.bridge.recipes.run_plugins import PerfEnvPlugin, NsysPlugin

import fiddle as fdl
import fiddle._src.experimental.dataclasses as fdl_dc

import logging
logger: logging.Logger = logging.getLogger(__name__)

if __name__ == "__main__":
    args = parse_cli_args().parse_args()
    exp_name = f"{splitext(basename(__file__))[0]}_{args.compute_dtype}"

    from pathlib import Path
    SCRIPT_DIR: Path = Path(__file__).parent.resolve()
    RUN_SCRIPT_FILENAME: str = "run_script.py"
    RUN_SCRIPT_PATH: Path = SCRIPT_DIR / RUN_SCRIPT_FILENAME
    print(f"Run script path: {RUN_SCRIPT_PATH}")
    if not RUN_SCRIPT_PATH.is_file():
        logger.error(f"Specified run script not found: {RUN_SCRIPT_PATH}")
        logger.error("Ensure the path passed to --run_script is correct.")
        sys.exit(1)

    config_file_to_use = SCRIPT_DIR / f"{args.model_name}_{args.model_size}_pretrain_overrides.yaml"
    print(f"Config file path: {config_file_to_use}")
    if not config_file_to_use.is_file():
        logger.error(f"Specified YAML config file not found: {config_file_to_use}")
        logger.error("Ensure the path passed to --config_file is correct.")
        sys.exit(1)

    plugins = [
        PerfEnvPlugin(
        enable_vboost = args.enable_vboost,
        nccl_pp_comm_chunksize = 2097152 if args.model_size in ["70b", "405b"] else None,
        gpu_sm100_or_newer = args.gpu.lower() in ["b200", "gb200"],
        )
    ] if HAS_NEMO_RUN else []
    if HAS_NEMO_RUN and args.enable_nsys:
        plugins.append(NsysPlugin(profile_step_start=10, profile_step_end=11))

    custom_mounts = args.custom_mounts + [
        f"{config_file_to_use}:{config_file_to_use}",
        f"{RUN_SCRIPT_PATH}:{RUN_SCRIPT_PATH}",
    ]
    print(f"Custom mounts: {custom_mounts}")

    num_nodes = -(args.num_gpus // -args.gpus_per_node)
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

    target_script_args = [
        "--config_file",
        str(config_file_to_use),
        "--gpu",
        args.gpu,
        "--compute_dtype",
        args.compute_dtype,
        "--fp8_recipe",
        args.fp8_recipe,
        "--model_name",
        args.model_name,
        "--model_size",
        args.model_size,
    ]

    train_script = run.Script(
        path=str(RUN_SCRIPT_PATH),
        entrypoint="python",
        args=target_script_args,
    )

    run.run(train_script, executor=executor, plugins=plugins, detach=True)
