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
import os
import sys

import torch
from argument_parser import parse_cli_args
from omegaconf import OmegaConf
from utils.helpers import COMM_OVERLAP_CONFIG_MAP, apply_perf_matrix_overrides, get_precision_config

from megatron.bridge.recipes.deepseek.deepseek_v3 import pretrain_config as deepseek_v3_pretrain_config
from megatron.bridge.recipes.llama import (
    llama3_8b_pretrain_config,
    llama3_70b_pretrain_config,
    llama31_405b_pretrain_config,
)
from megatron.bridge.recipes.qwen import (
    qwen3_30b_a3b_pretrain_config,
    qwen3_235b_a22b_pretrain_config,
)
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.utils.moe_token_drop import apply_moe_token_drop
from megatron.bridge.training.utils.omegaconf_utils import (
    apply_overrides,
    create_omegaconf_dict_config,
    parse_hydra_overrides,
)


logger: logging.Logger = logging.getLogger(__name__)


def main():
    """Main function to run the pretraining/finetuning script."""
    args, cli_overrides = parse_cli_args()

    precision_config = get_precision_config(args.compute_dtype, args.fp8_recipe)

    if args.model_name == "llama3" and args.model_size == "8b":
        recipe = llama3_8b_pretrain_config(mock=True, precision_config=precision_config)
    elif args.model_name == "llama3" and args.model_size == "70b":
        recipe = llama3_70b_pretrain_config(mock=True, precision_config=precision_config)
    elif args.model_name == "llama31" and args.model_size == "405b":
        recipe = llama31_405b_pretrain_config(mock=True, precision_config=precision_config)
    elif args.model_name == "deepseek" and args.model_size == "v3":
        enable_deepep = bool(args.gpu.lower() in ["h100"])
        use_tokendrop = bool(args.gpu.lower() in ["b200", "gb200"])
        use_tokendrop = args.use_tokendrop if args.use_tokendrop is not None else use_tokendrop
        if use_tokendrop:
            enable_deepep = False
            logger.info("Using token drop, disabling DeepEP")
        A2A_1F1B = bool(args.gpu.lower() in ["h100"])

        pp, vp = (8, 4) if args.gpu.lower() in ["h100"] else (4, 8)
        recipe = deepseek_v3_pretrain_config(
            mock=True,
            precision_config=precision_config,
            # NOTE: IMPORTANT: PLEASE SET PP-VP size here to correctly set the pp-vp layout
            pipeline_parallelism=pp,
            virtual_pipeline_parallelism=vp,
            enable_deepep=enable_deepep,
            layout="Et|(tt|)*30mL",
        )

        if enable_deepep:
            recipe.model.moe_router_force_load_balancing = True
        if use_tokendrop:
            recipe.model = apply_moe_token_drop(recipe.model)

        if A2A_1F1B:
            recipe.comm_overlap.overlap_moe_expert_parallel_comm = True
            recipe.comm_overlap.delay_wgrad_compute = True
            recipe.model.moe_shared_expert_overlap = False
        else:
            recipe.comm_overlap.overlap_moe_expert_parallel_comm = False
            recipe.comm_overlap.delay_wgrad_compute = False
            recipe.model.moe_shared_expert_overlap = True
        if args.gpu.lower() in ["h100"]:
            recipe.model.recompute_modules = ["mla_up_proj", "mlp"]
        elif args.gpu.lower() in ["gb200"]:
            recipe.model.recompute_modules = ["mla_up_proj", "mlp", "moe_act"]
        if args.gpu.lower() in ["gb200", "b200"]:
            recipe.comm_overlap.overlap_grad_reduce = True
        elif args.gpu.lower() in ["h100"]:
            recipe.comm_overlap.overlap_grad_reduce = False
    elif args.model_name == "qwen3" and args.model_size == "30b_a3b":
        recipe = qwen3_30b_a3b_pretrain_config(
            mock=True,
            precision_config=precision_config,
            comm_overlap_config=CommOverlapConfig(tp_comm_overlap=True),
        )
        recipe.model = apply_moe_token_drop(recipe.model)
    elif args.model_name == "qwen3" and args.model_size == "235b_a22b":
        recipe = qwen3_235b_a22b_pretrain_config(
            mock=True,
            precision_config=precision_config,
            comm_overlap_config=CommOverlapConfig(tp_comm_overlap=True),
        )
        recipe.model = apply_moe_token_drop(recipe.model)
    else:
        raise ValueError(f"Model {args.model_name} {args.model_size} not supported")

    if (
        f"{args.model_name}_{args.model_size}" in COMM_OVERLAP_CONFIG_MAP
        and args.gpu in COMM_OVERLAP_CONFIG_MAP[f"{args.model_name}_{args.model_size}"]
    ):
        ub_cfg = COMM_OVERLAP_CONFIG_MAP[f"{args.model_name}_{args.model_size}"][args.gpu][args.compute_dtype]
        recipe.comm_overlap.tp_comm_overlap_cfg = ub_cfg

    if args.compute_dtype == "bf16":
        recipe.optimizer.use_precision_aware_optimizer = True

    merged_omega_conf, excluded_fields = create_omegaconf_dict_config(recipe)
    # Load and merge YAML overrides if a config file is provided
    yaml_overrides_omega = None
    if args.config_file:
        logger.debug(f"Loading YAML overrides from: {args.config_file}")
        if not os.path.exists(args.config_file):
            logger.error(f"Override YAML file not found: {args.config_file}")
            sys.exit(1)
        yaml_overrides_omega = OmegaConf.load(args.config_file)
        # If YAML contains a nested ConfigContainer, merge only that subtree.
        yaml_cfg_overrides = (
            yaml_overrides_omega["ConfigContainer"]
            if OmegaConf.is_dict(yaml_overrides_omega) and "ConfigContainer" in yaml_overrides_omega
            else yaml_overrides_omega
        )
        merged_omega_conf = OmegaConf.merge(merged_omega_conf, yaml_cfg_overrides)
        logger.debug("YAML overrides merged successfully.")
    if cli_overrides:
        logger.debug(f"Applying Hydra-style command-line overrides: {cli_overrides}")
        merged_omega_conf = parse_hydra_overrides(merged_omega_conf, cli_overrides)
        logger.debug("Hydra-style command-line overrides applied successfully.")

    # Apply the final merged OmegaConf configuration back to the original ConfigContainer
    logger.debug("Applying final merged configuration back to Python ConfigContainer...")
    final_overrides_as_dict = OmegaConf.to_container(merged_omega_conf, resolve=True)
    # Apply overrides while preserving excluded fields
    apply_overrides(recipe, final_overrides_as_dict, excluded_fields)

    # Apply GPU/precision-specific performance overrides from perf_matrix, if present
    if yaml_overrides_omega is not None:
        apply_perf_matrix_overrides(yaml_overrides_omega, recipe, args, excluded_fields)
    recipe.model.gradient_accumulation_fusion = True

    if recipe.model.use_transformer_engine_op_fuser:
        if args.fp8_recipe == "mx" or recipe.ddp.use_megatron_fsdp:
            logger.warning("Disabling model.use_transformer_engine_op_fuser as it cannot work with MXFP8 or FSDP.")
            recipe.model.use_transformer_engine_op_fuser = False

    if recipe.ddp.use_megatron_fsdp:
        if args.model_name in ["llama3", "llama31"] and args.model_size in ["70b", "405b"]:
            recipe.ddp.fsdp_double_buffer = True
    recipe.model.apply_rope_fusion = True

    pretrain(config=recipe, forward_step_func=forward_step)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
