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

"""Resolve recipe-dependent perf environment variables, then exec run_script.py.

This entrypoint runs inside the training container.  It keeps recipe imports off
the dependency-light login node while ensuring environment variables are present
before the training Python process imports Torch, Transformer Engine, or Megatron.
"""

import os
import sys
from pathlib import Path

from argument_parser import parse_cli_args
from perf_plugins import PerfEnvPlugin
from utils.utils import _workload_base_config_from_recipe, get_perf_recipe_by_name


class _EnvironmentExecutor:
    """Minimal executor adapter used by PerfEnvPlugin's environment helpers."""

    def __init__(self) -> None:
        self.env_vars = os.environ


def main() -> None:
    """Apply the selected recipe's environment and replace this process with run_script.py."""
    parser = parse_cli_args()
    args, _ = parser.parse_known_args()

    recipe = get_perf_recipe_by_name(
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

    run_script = Path(__file__).with_name("run_script.py")
    os.execvpe(sys.executable, [sys.executable, str(run_script), *sys.argv[1:]], dict(os.environ))


if __name__ == "__main__":
    main()
