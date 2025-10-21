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

import megatron.core.parallel_state as mpu
import torch
from megatron.core import tensor_parallel
from megatron.core.enums import ModelType
from megatron.core.fp8_utils import correct_amax_history_if_needed
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
from megatron.core.transformer import MegatronModule, ModuleSpec, TransformerConfig
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.spec_utils import import_module
from megatron.core.utils import get_model_config

from megatron.bridge.training.mlm_compat.arguments import _transformer_config_from_args


def _get_transformer_layer_spec(args: argparse.Namespace, use_te: bool, use_kitchen: bool) -> ModuleSpec:
    """Get transformer layer specification based on configuration.

    Args:
        args: Training arguments
        use_te: Whether to use Transformer Engine
        use_kitchen: Whether to use kitchen extension

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
            use_kitchen=use_kitchen,
        )
    else:
        return get_gpt_layer_local_spec(
            args.num_experts,
            args.moe_grouped_gemm,
            args.qk_layernorm,
            args.multi_latent_attention,
            args.moe_use_legacy_grouped_gemm,
            normalization=args.normalization,
            use_kitchen=use_kitchen,
        )


def _gpt_provider(
    args: argparse.Namespace,
    config: TransformerConfig | None = None,
    pre_process: bool = True,
    post_process: bool = True,
    vp_stage: int | None = None,
) -> GPTModel:
    """Provide the GPTModel exactly as done by MLM using an argparse args object.

    May need to set `args` and `config` with functools.partial.
    """
    use_te = args.transformer_impl == "transformer_engine"

    if config is None:
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
        transformer_layer_spec = _get_transformer_layer_spec(args, use_te, config.use_kitchen)

    mtp_block_spec = None
    if args.mtp_num_layers is not None:
        if hasattr(transformer_layer_spec, "layer_specs") and len(transformer_layer_spec.layer_specs) == 0:
            # Get the decoder layer spec explicitly if no decoder layer in the last stage,
            # Only happens with block spec (TransformerBlockSubmodules) when using MoE.
            transformer_layer_spec_for_mtp = _get_transformer_layer_spec(args, use_te, config.use_kitchen)
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


def _mamba_provider(
    args: argparse.Namespace,
    config: TransformerConfig | None = None,
    pre_process: bool = True,
    post_process: bool = True,
    vp_stage: int | None = None,
) -> MambaModel:
    """Provide the MambaModel exactly as done by MLM using an argparse args object.

    May need to set `args` and `config` with functools.partial.
    """
    if config is None:
        config = _transformer_config_from_args(args)

    assert args.spec is not None, "You must provide a valid Mamba layer spec!"
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


def _get_model(
    args: argparse.Namespace,
    model_provider_func,
    model_cfg: TransformerConfig,
    model_type=ModelType.encoder_or_decoder,
) -> list[MegatronModule]:
    """Build the model vp stages and wrappers for inference. In the same style as MLM."""

    # Build model.
    def build_model():
        if (
            mpu.get_pipeline_model_parallel_world_size() > 1
            and mpu.get_virtual_pipeline_model_parallel_world_size() is not None
        ):
            model = []
            for i in range(mpu.get_virtual_pipeline_model_parallel_world_size()):
                # Set pre_process and post_process only after virtual rank is set.
                pre_process = mpu.is_pipeline_first_stage(ignore_virtual=False, vp_stage=i)
                post_process = mpu.is_pipeline_last_stage(ignore_virtual=False, vp_stage=i)
                this_model = model_provider_func(
                    args, model_cfg, pre_process=pre_process, post_process=post_process, vp_stage=i
                )
                this_model.model_type = model_type
                this_model.vp_stage = i
                model.append(this_model)
        else:
            pre_process = mpu.is_pipeline_first_stage()
            post_process = mpu.is_pipeline_last_stage()
            model = model_provider_func(args, model_cfg, pre_process=pre_process, post_process=post_process)
            model.model_type = model_type
        return model

    if args.init_model_with_meta_device:
        with torch.device("meta"):
            model = build_model()
    else:
        model = build_model()

    if not isinstance(model, list):
        model = [model]

    # Set tensor model parallel attributes if not set.
    # Only parameters that are already tensor model parallel have these
    # attributes set for them. We should make sure the default attributes
    # are set for all params so the optimizer can use them.
    for model_module in model:
        for param in model_module.parameters():
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)

    # Print number of parameters.
    num_parameters = sum([sum([p.nelement() for p in model_module.parameters()]) for model_module in model])
    if mpu.get_data_parallel_rank() == 0 and mpu.get_context_parallel_rank() == 0:
        print(
            " > number of parameters on (tensor, pipeline) model parallel rank ({}, {}): {}".format(
                mpu.get_tensor_model_parallel_rank(),
                mpu.get_pipeline_model_parallel_rank(),
                num_parameters,
            ),
            flush=True,
        )

    if not args.init_model_with_meta_device:
        for model_module in model:
            model_module.cuda(torch.cuda.current_device())

    # Fp16 conversion.
    if args.fp16 or args.bf16:
        config = get_model_config(model[0])
        model = [Float16Module(config, model_module) for model_module in model]

    # Before TE2.x: The model_module.bfloat16()/model_module.half() above will call the inplace
    #               copy of TE's Float8Tensor, which will write an unwanted value (amax calculated
    #               from the current fp8 param) to its amax_history. The below function will correct
    #               the amax_history back.
    # After TE2.x: Below function is an empty function and does nothing.
    correct_amax_history_if_needed(model)

    return model
