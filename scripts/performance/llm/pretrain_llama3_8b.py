from os.path import basename, splitext

from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.recipes.llama.llama3_8b import pretrain_config as recipe
from megatron.bridge.training.mixed_precision import bf16_mixed, bf16_with_fp8_mixed
from megatron.bridge.recipes.utils.nemo_run_utils import get_partial_fn

from ..argument_parser import parse_cli_args
from ..executors import slurm_executor

try:
    import nemo_run as run

    HAS_NEMO_RUN = True
except ImportError:
    HAS_NEMO_RUN = False

if HAS_NEMO_RUN:
    from megatron.bridge.recipes.run_plugins import (
        PerfEnvPlugin,
        NsysPlugin,
    )

import fiddle as fdl
import fiddle._src.experimental.dataclasses as fdl_dc

if __name__ == "__main__":
    args = parse_cli_args().parse_args()

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
        custom_mounts=args.custom_mounts,
        custom_env_vars={},
        hf_token=args.hf_token,
        nemo_home=args.nemo_home,
        wandb_key=args.wandb_key,
        network='sharp' if args.use_sharp else None,
    )

    plugins = [
        PerfEnvPlugin(
        enable_vboost = args.enable_vboost,
        gpu_sm100_or_newer = args.gpu.lower() in ["b200", "gb200"],
        )
    ]
    if args.enable_nsys:
        plugins.append(NsysPlugin(start_step=5, end_step=6))

    exp_name = f"{splitext(basename(__file__))[0]}_{args.compute_dtype}"
    if args.compute_dtype == "fp8":
        exp_name += f"_{args.fp8_recipe}"
        precision_config = bf16_with_fp8_mixed()
    else:
        precision_config = bf16_mixed()
    precision_config.grad_reduce_in_fp32 = False

    with run.Experiment(exp_name) as exp:
        exp.add(
            get_partial_fn(
                pretrain,
                config=recipe(
                    mock=True, global_batch_size=128, precision_config=precision_config),
                forward_step_func=forward_step,
            ),
            executor=executor,
            name = exp_name,
            plugins=plugins,
        )

        if not args.dryrun:
            exp.run(sequential=True, detach=True)
        else:
            exp.dryrun()
