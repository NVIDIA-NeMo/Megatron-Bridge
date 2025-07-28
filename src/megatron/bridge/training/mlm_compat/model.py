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
from typing import Optional

import torch
import torch.nn.functional as F
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.heterogeneous.heterogeneous_layer_specs import (
    get_gpt_heterogeneous_layer_spec,
)
from megatron.core.models.mamba import MambaModel
from megatron.core.quantization.utils import kitchen_quantization_recipe_config, load_quantization_recipe
from megatron.core.transformer import MLATransformerConfig, TransformerConfig
from megatron.core.transformer.heterogeneous.heterogeneous_config import HeterogeneousTransformerConfig
from megatron.core.transformer.spec_utils import import_module

from megatron.bridge.training.mlm_compat.activations import squared_relu


def _transformer_config_from_args(args, config_class=TransformerConfig) -> TransformerConfig:
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
        kw_args["scaled_init_method"] = torch.nn.init.xavier_uniform_
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


def _get_transformer_layer_spec(args, use_te, config):
    """Get transformer layer specification based on configuration.

    Args:
        args: Training arguments
        use_te (bool): Whether to use Transformer Engine
        config: Model configuration

    Returns:
        transformer_layer_spec: The transformer layer specification
    """
    if use_te:
        return get_gpt_layer_with_transformer_engine_spec(
            args.num_experts,
            args.moe_grouped_gemm,
            args.qk_layernorm,
            args.multi_latent_attention,
            args.moe_use_legacy_grouped_gemm,
            qk_l2_norm=args.qk_l2_norm,
            use_kitchen=config.use_kitchen,
        )
    else:
        return get_gpt_layer_local_spec(
            args.num_experts,
            args.moe_grouped_gemm,
            args.qk_layernorm,
            args.multi_latent_attention,
            args.moe_use_legacy_grouped_gemm,
            normalization=args.normalization,
            use_kitchen=config.use_kitchen,
        )


def _gpt_provider(
    args: argparse.Namespace, pre_process=True, post_process=True, vp_stage: Optional[int] = None
) -> GPTModel:
    use_te = args.transformer_impl == "transformer_engine"
    config = _transformer_config_from_args(args)

    if args.num_experts:
        # Define the decoder block spec
        transformer_layer_spec = get_gpt_decoder_block_spec(
            config,
            use_transformer_engine=use_te,
            normalization=args.normalization,
            qk_l2_norm=args.qk_l2_norm,
            vp_stage=vp_stage,
        )
    elif args.heterogeneous_layers_config_path is not None:
        transformer_layer_spec = get_gpt_heterogeneous_layer_spec(config, use_te)
    else:
        # Define the decoder layer spec
        transformer_layer_spec = _get_transformer_layer_spec(args, use_te, config)

    mtp_block_spec = None
    if args.mtp_num_layers is not None:
        if hasattr(transformer_layer_spec, "layer_specs") and len(transformer_layer_spec.layer_specs) == 0:
            # Get the decoder layer spec explicitly if no decoder layer in the last stage,
            # Only happens with block spec (TransformerBlockSubmodules) when using MoE.
            transformer_layer_spec_for_mtp = _get_transformer_layer_spec(args, use_te, config)
        else:
            transformer_layer_spec_for_mtp = transformer_layer_spec
        mtp_block_spec = get_gpt_mtp_block_spec(
            config, transformer_layer_spec_for_mtp, use_transformer_engine=use_te, vp_stage=vp_stage
        )

    return GPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
        rope_scaling=args.use_rope_scaling,
        mtp_block_spec=mtp_block_spec,
        vp_stage=vp_stage,
    )


def _mamba_provider(args: argparse.Namespace, pre_process=True, post_process=True) -> MambaModel:
    config = _transformer_config_from_args(args)
    mamba_stack_spec = import_module(args.spec)

    model = MambaModel(
        config=config,
        mamba_stack_spec=mamba_stack_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        hybrid_attention_ratio=args.hybrid_attention_ratio,
        hybrid_mlp_ratio=args.hybrid_mlp_ratio,
        hybrid_override_pattern=args.hybrid_override_pattern,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
    )

    return model
