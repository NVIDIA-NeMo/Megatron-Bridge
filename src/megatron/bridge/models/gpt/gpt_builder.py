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

import inspect
import logging

from megatron.core.post_training.modelopt.gpt.model_specs import get_gpt_modelopt_spec
from megatron.core.transformer import ModuleSpec
from megatron.training.models.gpt import GPTModelBuilder, GPTModelConfig, mtp_block_spec

from megatron.bridge.models.transformer_config import TransformerConfig


logger = logging.getLogger(__name__)


from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)

from megatron.bridge.models.transformer_config import TransformerConfig


def transformer_engine_layer_spec(config: "GPTModelConfig") -> ModuleSpec:
    """Create a Transformer Engine layer specification based on the provided config."""
    if "use_te_op_fuser" in inspect.signature(get_gpt_layer_with_transformer_engine_spec).parameters:
        kwargs = {"use_te_op_fuser": config.use_transformer_engine_op_fuser}
    else:
        kwargs = {}
    return get_gpt_layer_with_transformer_engine_spec(
        num_experts=config.transformer.num_moe_experts,
        moe_grouped_gemm=config.transformer.moe_grouped_gemm,
        qk_layernorm=config.transformer.qk_layernorm,
        fp8=bool(config.transformer.num_moe_experts and (config.transformer.fp8 is not None)),
        **kwargs,
    )


def transformer_engine_full_layer_spec(config: TransformerConfig) -> ModuleSpec:
    """Create a full Transformer Engine layer specification with autocast support.

    Args:
        config: GPT configuration object

    Returns:
        ModuleSpec: Module specification for full TE layers
    """
    from megatron.bridge.models.gpt_full_te_layer_autocast_spec import get_gpt_full_te_layer_autocast_spec

    return get_gpt_full_te_layer_autocast_spec(transformer_config=config)


def local_layer_spec(config: TransformerConfig) -> ModuleSpec:
    """Create a local layer specification without Transformer Engine.

    Args:
        config: GPT configuration object

    Returns:
        ModuleSpec: Module specification for local implementation layers
    """
    return get_gpt_layer_local_spec(
        num_experts=config.num_moe_experts,
        moe_grouped_gemm=config.moe_grouped_gemm,
        qk_layernorm=config.qk_layernorm,
        normalization=config.normalization,
    )


def modelopt_transformer_layer_spec(config: "GPTModelConfig") -> ModuleSpec:
    """Layer specification for quantization with ModelOpt."""
    # arbitrary attention mask is used for speculative decoding training
    # When context parallel > 1, only causal mask type is supported
    from megatron.core import parallel_state

    use_arbitrary_attention_mask = (
        config.use_arbitrary_attention_mask
        if config.use_arbitrary_attention_mask is not None
        else parallel_state.get_context_parallel_world_size() == 1
    )
    return get_gpt_modelopt_spec(
        config=config.transformer,
        local_core_attention=False,
        remap_te_layernorm=True,
        real_quant_cfg="None",
        use_arbitrary_attention_mask=use_arbitrary_attention_mask,
    )


def default_layer_spec(config: "GPTModelConfig") -> ModuleSpec:
    """Determine the most appropriate layer specification based on availability."""
    if config.restore_modelopt_state:
        return modelopt_transformer_layer_spec(config)
    elif config.use_transformer_engine_full_layer_spec:
        return transformer_engine_full_layer_spec(config.transformer)
    else:
        return transformer_engine_layer_spec(config)


__all__ = [
    "GPTModelConfig",
    "GPTModelBuilder",
    "mtp_block_spec",
    "transformer_engine_layer_spec",
    "transformer_engine_full_layer_spec",
    "local_layer_spec",
    "modelopt_transformer_layer_spec",
    "default_layer_spec",
]
