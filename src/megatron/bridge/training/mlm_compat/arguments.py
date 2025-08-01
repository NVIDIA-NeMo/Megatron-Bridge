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
import dataclasses

import torch
import torch.nn.functional as F
from megatron.core import dist_checkpointing
from megatron.core.quantization.utils import kitchen_quantization_recipe_config, load_quantization_recipe
from megatron.core.transformer import MLATransformerConfig, TransformerConfig
from megatron.core.transformer.heterogeneous.heterogeneous_config import HeterogeneousTransformerConfig

from megatron.bridge.training.config import TokenizerConfig
from megatron.bridge.training.mlm_compat.activations import squared_relu


def _load_args_from_checkpoint(checkpoint_path: str) -> argparse.Namespace:
    """Obtain argparse args object from an MLM checkpoint."""
    state_dict = dist_checkpointing.load_common_state_dict(checkpoint_path)
    assert state_dict is not None, f"Could not load state from checkpoint at {checkpoint_path}"
    assert "args" in state_dict, "Provided checkpoint does not have arguments saved."

    return state_dict["args"]


def _tokenizer_config_from_args(args: argparse.Namespace) -> TokenizerConfig:
    """Build TokenizerConfig from content of MLM argparse args object."""
    kw_args = {}
    for f in dataclasses.fields(TokenizerConfig):
        if hasattr(args, f.name):
            kw_args[f.name] = getattr(args, f.name)

    return TokenizerConfig(**kw_args)


def _transformer_config_from_args(
    args: argparse.Namespace, config_class: type[TransformerConfig] = TransformerConfig
) -> TransformerConfig:
    """Build a variant of TransformerConfig based on contents of the MLM argparse args object."""
    if args.multi_latent_attention:
        config_class = MLATransformerConfig

    if args.heterogeneous_layers_config_path is not None:
        assert not args.multi_latent_attention, "Multi latent attention with heterogeneous layers is not supported."
        config_class = HeterogeneousTransformerConfig

    # Translate args to core transformer configuration
    kw_args = {}
    for f in dataclasses.fields(config_class):
        if hasattr(args, f.name):
            kw_args[f.name] = getattr(args, f.name)
    kw_args["persist_layer_norm"] = not args.no_persist_layer_norm
    kw_args["layernorm_zero_centered_gamma"] = args.apply_layernorm_1p
    kw_args["layernorm_epsilon"] = args.norm_epsilon
    kw_args["deallocate_pipeline_outputs"] = True
    kw_args["pipeline_dtype"] = args.params_dtype
    kw_args["batch_p2p_comm"] = not args.overlap_p2p_comm
    kw_args["num_moe_experts"] = args.num_experts
    kw_args["rotary_interleaved"] = args.rotary_interleaved
    kw_args["num_layers_in_first_pipeline_stage"] = args.decoder_first_pipeline_num_layers
    kw_args["num_layers_in_last_pipeline_stage"] = args.decoder_last_pipeline_num_layers
    kw_args["fp8_param"] = args.fp8_param_gather
    if args.swiglu:
        kw_args["activation_func"] = F.silu
        kw_args["gated_linear_unit"] = True
        kw_args["bias_activation_fusion"] = args.bias_swiglu_fusion
    else:
        kw_args["bias_activation_fusion"] = args.bias_gelu_fusion
    if args.squared_relu:
        assert not args.swiglu
        kw_args["activation_func"] = squared_relu
    if args.init_method_xavier_uniform:
        kw_args["init_method"] = torch.nn.init.xavier_uniform_
    if args.group_query_attention:
        kw_args["num_query_groups"] = args.num_query_groups
    else:
        kw_args["num_query_groups"] = None
    kw_args["config_logger_dir"] = args.config_logger_dir

    if len(args.cp_comm_type) == 1:
        kw_args["cp_comm_type"] = args.cp_comm_type[0]
    if args.is_hybrid_model:
        kw_args["is_hybrid_model"] = args.is_hybrid_model

    # handle quantization config
    # NOTE: Kitchen arguments are only added to the namespace when
    # Kitchen library is available.
    if hasattr(args, "kitchen_config_file") and args.kitchen_config_file is not None:
        kw_args["use_kitchen"] = True
        kw_args["quant_recipe"] = load_quantization_recipe(args.kitchen_config_file)
    elif hasattr(args, "kitchen_recipe_number") and args.kitchen_recipe_number is not None:
        kw_args["use_kitchen"] = True
        kw_args["quant_recipe"] = kitchen_quantization_recipe_config(args.kitchen_recipe_number)

    # Return config.
    return config_class(**kw_args)
