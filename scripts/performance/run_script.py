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
from configs.llama3.llama3_8b_llm_pretrain import (
    llama3_8b_h100_bf16_config, 
    llama3_8b_h100_fp8_config,
    llama3_8b_b200_bf16_config,
    llama3_8b_b200_fp8_config,
    llama3_8b_gb200_bf16_config,
    llama3_8b_gb200_fp8_config,
)
from configs.deepseek.deepseek_v3_llm_pretrain import (
    deepseek_v3_gb200_bf16_config, 
    deepseek_v3_gb200_fp8_config, 
    deepseek_v3_h100_bf16_config, 
    deepseek_v3_h100_fp8_config,
    deepseek_v3_b200_bf16_config,
    deepseek_v3_b200_fp8_config,
)
from configs.qwen3.qwen3_30b_a3b_llm_pretrain import (
    qwen3_30b_a3b_h100_bf16_config,
    qwen3_30b_a3b_h100_fp8_config,
    qwen3_30b_a3b_b200_bf16_config,
    qwen3_30b_a3b_b200_fp8_config,
    qwen3_30b_a3b_gb200_bf16_config,
    qwen3_30b_a3b_gb200_fp8_config,
)
from configs.qwen3.qwen3_235b_a22b_llm_pretrain import (
    qwen3_235b_a22b_h100_bf16_config,
    qwen3_235b_a22b_h100_fp8_config,
    qwen3_235b_a22b_b200_bf16_config,
    qwen3_235b_a22b_b200_fp8_config,
    qwen3_235b_a22b_gb200_bf16_config,
    qwen3_235b_a22b_gb200_fp8_config,
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

    if args.model_name in ["llama3", "llama31"]:
        cfg_str = f"{args.model_name}_{args.model_size}_{args.gpu.lower()}_{args.compute_dtype.lower()}_config"
        recipe = globals()[cfg_str](fp8_recipe=args.fp8_recipe)
    elif args.model_name == "deepseek" and args.model_size == "v3":
        enable_deepep = bool(args.gpu.lower() in ["h100"])
        use_tokendrop = bool(args.gpu.lower() in ["b200", "gb200"])
        use_tokendrop = args.use_tokendrop if args.use_tokendrop is not None else use_tokendrop
        if use_tokendrop:
            enable_deepep = False
            logger.info("Using token drop, disabling DeepEP")
        A2A_1F1B = bool(args.gpu.lower() in ["h100"])
        
        cfg_str = f"deepseek_v3_{args.gpu.lower()}_{args.compute_dtype.lower()}_config"
        recipe = globals()[cfg_str](
            fp8_recipe=args.fp8_recipe,
            use_tokendrop=use_tokendrop,
            enable_deepep=enable_deepep,
            a2a_1f1b=A2A_1F1B,
        )
    elif args.model_name == "qwen3":
        cfg_str = f"qwen3_{args.model_size}_{args.gpu.lower()}_{args.compute_dtype.lower()}_config"
        recipe = globals()[cfg_str](fp8_recipe=args.fp8_recipe)
    else:
        raise ValueError(f"Model {args.model_name} {args.model_size} not supported")

    merged_omega_conf, excluded_fields = create_omegaconf_dict_config(recipe)
    if cli_overrides:
        logger.debug(f"Applying Hydra-style command-line overrides: {cli_overrides}")
        merged_omega_conf = parse_hydra_overrides(merged_omega_conf, cli_overrides)
        logger.debug("Hydra-style command-line overrides applied successfully.")

    # Apply the final merged OmegaConf configuration back to the original ConfigContainer
    logger.debug("Applying final merged configuration back to Python ConfigContainer...")
    final_overrides_as_dict = OmegaConf.to_container(merged_omega_conf, resolve=True)
    # Apply overrides while preserving excluded fields
    apply_overrides(recipe, final_overrides_as_dict, excluded_fields)

    if args.compute_dtype == "bf16":
        recipe.optimizer.use_precision_aware_optimizer = True
    if recipe.model.use_transformer_engine_op_fuser:
        recipe.model.use_transformer_engine_op_fuser = False
    recipe.model.gradient_accumulation_fusion = True
    recipe.model.apply_rope_fusion = True

    tp = recipe.model.tensor_model_parallel_size
    pp = recipe.model.pipeline_model_parallel_size
    cp = recipe.model.context_parallel_size
    vp = recipe.model.virtual_pipeline_model_parallel_size or 1

    dp = int(args.num_gpus / (tp * pp * cp))
    logger.info(f"DP: {dp}; TP: {tp}; PP: {pp}; CP: {cp}; VP: {vp}")
    if dp > 1 and pp > 1 and vp > 1:
        recipe.optimizer.overlap_param_gather_with_optimizer_step = True
        recipe.comm_overlap.overlap_param_gather_with_optimizer_step = True

    pretrain(config=recipe, forward_step_func=forward_step)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
