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
from utils.overrides import set_cli_overrides, set_user_overrides

from megatron.bridge.models.qwen_vl.qwen3_vl_step import forward_step as qwen3_vl_forward_step
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.vlm_step import forward_step as vlm_forward_step


logger = logging.getLogger(__name__)


def get_perf_recipe_by_name(model_recipe_name, task, num_gpus, gpu, precision, config_variant="v1"):
    """Load a flat perf recipe from megatron.bridge.recipes by convention name."""
    import importlib

    precision_map = {
        "bf16": "bf16",
        "fp8_cs": "fp8cs",
        "fp8_mx": "fp8mx",
        "fp8_sc": "fp8cs",
        "nvfp4": "nvfp4",
    }
    prec = precision_map.get(precision.lower(), precision.lower().replace("_", ""))

    v2_prefix = "_v2" if config_variant == "v2" else ""
    name = f"{model_recipe_name}_{task}{v2_prefix}_{num_gpus}gpu_{gpu}_{prec}_config"

    family_map = {
        "llama3_8b": "llama",
        "llama3_70b": "llama",
        "llama31_405b": "llama",
        "qwen3_235b_a22b": "qwen",
        "qwen3_30b_a3b": "qwen",
        "qwen3_next_80b_a3b": "qwen",
        "deepseek_v3": "deepseek",
        "nemotronh_56b": "nemotronh",
        "nemotron_3_nano": "nemotronh",
        "kimi_k2": "kimi",
        "gpt_oss_120b": "gpt_oss",
        "qwen3_vl_235b_a22b": "qwen_vl",
        "qwen3_vl_30b_a3b": "qwen_vl",
    }

    family = family_map.get(model_recipe_name)
    if not family:
        raise ValueError(
            f"Unknown model_recipe_name {model_recipe_name!r}. Add it to family_map in get_perf_recipe_by_name."
        )

    mod = importlib.import_module(f"megatron.bridge.recipes.{family}")
    recipe_fn = getattr(mod, name, None)
    if recipe_fn is None:
        raise ValueError(f"No perf recipe {name!r} found in megatron.bridge.recipes.{family}.")
    return recipe_fn()


def main():
    """Main function to run the pretraining/finetuning script."""
    # Parse known args and treat any unknown args as Hydra-style config overrides.
    # `argparse.parse_known_args()` returns the unknown args as a `list[str]`.
    parser = parse_cli_args()
    args, cli_overrides = parser.parse_known_args()

    recipe = get_perf_recipe_by_name(
        model_recipe_name=args.model_recipe_name,
        task=args.task,
        num_gpus=args.num_gpus,
        gpu=args.gpu,
        precision=args.compute_dtype,
        config_variant=args.config_variant,
    )

    recipe = set_cli_overrides(recipe, cli_overrides)
    recipe = set_user_overrides(recipe, args)

    # Select forward step function based on the model family name.
    if args.domain == "vlm":
        forward_step_func = vlm_forward_step
    elif args.domain == "qwen3vl":
        forward_step_func = qwen3_vl_forward_step
    else:
        forward_step_func = forward_step

    pretrain(config=recipe, forward_step_func=forward_step_func)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
