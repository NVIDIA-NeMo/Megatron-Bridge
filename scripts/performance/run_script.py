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

import logging

import torch
from argument_parser import parse_cli_args
from utils.overrides import set_post_overrides, set_user_overrides
from utils.utils import get_perf_optimized_recipe

from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain


logger = logging.getLogger(__name__)


def main():
    """Main function to run the pretraining/finetuning script."""
    parser = parse_cli_args()
    args, _ = parser.parse_known_args()

    args.model_recipe_name = (
        f"{args.model_recipe_name}_pretrain_config"
        if args.task == "pretrain"
        else f"{args.model_recipe_name}_finetune_config"
    )

    if args.model_recipe_name == "deepseek_v3_32nodes_pretrain_config":
        args.model_recipe_name = "deepseek_v3_pretrain_config_32nodes"

    recipe = get_perf_optimized_recipe(
        model_family_name=args.model_family_name,
        model_recipe_name=args.model_recipe_name,
        gpu=args.gpu,
        compute_dtype=args.compute_dtype,
        mock=args.data == "mock",
    )

    recipe = set_user_overrides(recipe, args)

    recipe = set_post_overrides(
        recipe,
        args.model_family_name,
        args.model_recipe_name,
        args.gpu,
        args.num_gpus,
        args.compute_dtype,
        args.task,
        user_gbs=args.global_batch_size,
    )

    pretrain(config=recipe, forward_step_func=forward_step)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
