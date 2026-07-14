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

"""Bootstrap the recipe environment and run performance training."""

import logging
import os
import sys
from pathlib import Path

from argument_parser import parse_cli_args
from perf_plugins import PerfEnvPlugin
from utils.utils import _workload_base_config_from_recipe
from utils.utils import get_perf_recipe_by_name as get_perf_recipe_for_environment


logger = logging.getLogger(__name__)
ENV_BOOTSTRAP_MARKER = "_MB_PERF_ENV_BOOTSTRAPPED"
_TRAINING_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "training"


def _load_recipe_runner():
    """Import the shared runner after the perf environment bootstrap."""
    if str(_TRAINING_SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(_TRAINING_SCRIPT_DIR))

    import recipe_runner

    return recipe_runner


def _step_function_name(domain: str) -> str:
    """Map the perf domain selector to the shared forward-step registry name."""
    if domain == "vlm":
        return "vlm_step"
    if domain == "qwen3vl":
        return "qwen3_vl_step"
    if domain == "diffusion":
        return "wan_step"
    return "llm_step"


def _apply_perf_recipe_overrides(recipe, cli_overrides: list[str], args):
    """Apply performance CLI overrides while preserving main-branch behavior."""
    from utils.overrides import set_cli_overrides, set_user_overrides

    recipe = set_cli_overrides(recipe, cli_overrides)
    recipe = set_user_overrides(recipe, args)

    if args.compute_dtype == "bf16" and recipe.optimizer.optimizer == "adam":
        recipe.optimizer.use_precision_aware_optimizer = True
    if getattr(recipe.ddp, "nccl_ub", False):
        os.environ["NCCL_NVLS_ENABLE"] = "1"
        os.environ["NCCL_CTA_POLICY"] = "1"
    return recipe


class _EnvironmentExecutor:
    """Minimal executor adapter for ``PerfEnvPlugin`` environment setup."""

    def __init__(self) -> None:
        self.env_vars = os.environ


def _bootstrap_recipe_environment(args) -> None:
    """Install recipe env and re-exec this script in a clean interpreter."""
    recipe = get_perf_recipe_for_environment(
        model_recipe_name=args.model_recipe_name,
        task=args.task,
        num_gpus=args.num_gpus,
        gpu=args.gpu,
        precision=args.compute_dtype,
        config_variant=args.config_variant,
    )
    workload_base_config = _workload_base_config_from_recipe(recipe, num_gpus=args.num_gpus)
    plugin = PerfEnvPlugin(
        moe_a2a_overlap=args.moe_a2a_overlap,
        tp_size=args.tensor_model_parallel_size,
        pp_size=args.pipeline_model_parallel_size,
        cp_size=args.context_parallel_size,
        ep_size=args.expert_model_parallel_size,
        model_family_name=args.model_family_name,
        model_recipe_name=args.model_recipe_name,
        gpu=args.gpu,
        compute_dtype=args.compute_dtype,
        train_task=args.task,
        config_variant=args.config_variant,
        deterministic=args.deterministic,
    )
    plugin.setup_recipe_environment(None, _EnvironmentExecutor(), workload_base_config)

    environment = dict(os.environ)
    # exec preserves the PID. Binding the marker to it prevents an inherited
    # or stale variable from skipping environment setup in a new process.
    environment[ENV_BOOTSTRAP_MARKER] = str(os.getpid())
    os.execvpe(sys.executable, [sys.executable, __file__, *sys.argv[1:]], environment)


def _run_training(args, cli_overrides: list[str]) -> None:
    """Import the training stack after env bootstrap and run the workload."""
    recipe_runner = _load_recipe_runner()
    recipe = recipe_runner.load_perf_recipe_by_name(
        model_recipe_name=args.model_recipe_name,
        task=args.task,
        num_gpus=args.num_gpus,
        gpu=args.gpu,
        precision=args.compute_dtype,
        config_variant=getattr(args, "config_variant", None),
    )

    recipe = _apply_perf_recipe_overrides(recipe, cli_overrides, args)
    forward_step_func = recipe_runner.load_forward_step(_step_function_name(args.domain), mode="pretrain")
    recipe_runner.run_config(
        config=recipe,
        mode="pretrain",
        step_func=forward_step_func,
        dryrun=args.dryrun,
        save_config_filepath=args.save_config_filepath,
        barrier_before_destroy=True,
        dryrun_num_gpus=args.num_gpus,
        dump_environment=args.dump_env,
    )


def main() -> None:
    """Resolve recipe env on the first pass and train in the self-exec process."""
    parser = parse_cli_args()
    args, cli_overrides = parser.parse_known_args()

    if os.environ.get(ENV_BOOTSTRAP_MARKER) != str(os.getpid()):
        _bootstrap_recipe_environment(args)
    else:
        _run_training(args, cli_overrides)


if __name__ == "__main__":
    main()
