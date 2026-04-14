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

"""
Decoder layer spec for ERNIE 4.5 VL MoE.

Creates heterogeneous transformer block specs where:
- Layer 0: dense MLP
- Layers 1+: ErnieMultiTypeMoE (dual-pool MoE with text + vision expert pools)

The text and vision MoE pools each use standard Megatron MoELayer with
SequentialMLP experts, enabling full TP/EP compatibility through standard
Megatron-Core infrastructure.
"""

from typing import Optional

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.experts import SequentialMLP
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import (
    TransformerBlockSubmodules,
    get_num_layers_to_build,
)
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
    get_transformer_layer_offset,
)

from megatron.bridge.models.ernie_vl.modeling_ernie45_vl.ernie_moe_layer import (
    ErnieMultiTypeMoE,
    MultiTypeMoeSubmodules,
)

try:
    import transformer_engine  # noqa: F401

    HAVE_TE = True
except (ImportError, ModuleNotFoundError):
    HAVE_TE = False


def _get_linear_modules():
    """Get appropriate linear module classes based on TE availability."""
    if HAVE_TE:
        from megatron.core.extensions.transformer_engine import (
            TEColumnParallelLinear,
            TEDotProductAttention,
            TELayerNormColumnParallelLinear,
            TENorm,
            TERowParallelLinear,
        )

        return (
            TEColumnParallelLinear,
            TERowParallelLinear,
            TEDotProductAttention,
            TELayerNormColumnParallelLinear,
            TENorm,
        )
    else:
        from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
        from megatron.core.transformer.dot_product_attention import DotProductAttention
        from megatron.core.transformer.torch_norm import WrappedTorchNorm

        return (
            ColumnParallelLinear,
            RowParallelLinear,
            DotProductAttention,
            ColumnParallelLinear,  # No fused LN+Linear without TE
            WrappedTorchNorm,
        )


def _get_mlp_module_spec(
    num_experts: Optional[int] = None,
    moe_grouped_gemm: bool = False,
) -> ModuleSpec:
    """Get MLP module spec for dense or dual-pool MoE layers.

    Args:
        num_experts: Number of experts per pool. None for dense MLP.
        moe_grouped_gemm: Whether to use grouped GEMM for experts.

    Returns:
        ModuleSpec for either dense MLP or ErnieMultiTypeMoE.
    """
    ColumnParallel, RowParallel, _, LayerNormColumnParallel, _ = _get_linear_modules()

    if num_experts is None:
        # Dense MLP (layer 0): uses TELayerNormColumnParallelLinear to fuse
        # post_attention_layernorm with linear_fc1
        return ModuleSpec(
            module=MLP,
            submodules=MLPSubmodules(
                linear_fc1=LayerNormColumnParallel,
                linear_fc2=RowParallel,
            ),
        )

    # Expert MLP spec (used by both text and vision pools)
    if moe_grouped_gemm:
        try:
            from megatron.core.transformer.moe.experts import TEGroupedMLP

            experts_spec = ModuleSpec(module=TEGroupedMLP)
        except ImportError:
            experts_spec = ModuleSpec(
                module=SequentialMLP,
                submodules=MLPSubmodules(
                    linear_fc1=ColumnParallel,
                    linear_fc2=RowParallel,
                ),
            )
    else:
        experts_spec = ModuleSpec(
            module=SequentialMLP,
            submodules=MLPSubmodules(
                linear_fc1=ColumnParallel,
                linear_fc2=RowParallel,
            ),
        )

    # Each pool is a standard MoELayer
    base_moe_spec = ModuleSpec(
        module=MoELayer,
        submodules=MoESubmodules(
            experts=experts_spec,
        ),
    )

    # Shared experts MLP
    shared_experts_spec = ModuleSpec(
        module=SharedExpertMLP,
        submodules=MLPSubmodules(
            linear_fc1=ColumnParallel,
            linear_fc2=RowParallel,
        ),
    )

    # Dual-pool MoE
    return ModuleSpec(
        module=ErnieMultiTypeMoE,
        submodules=MultiTypeMoeSubmodules(
            text_moe_layer=base_moe_spec,
            vision_moe_layer=base_moe_spec,
            shared_experts=shared_experts_spec,
        ),
    )


def _get_ernie_decoder_layer_spec(
    num_experts: Optional[int] = None,
    moe_grouped_gemm: bool = False,
) -> ModuleSpec:
    """Get a single transformer layer spec.

    Args:
        num_experts: Number of experts per pool. None for dense layer.
        moe_grouped_gemm: Whether to use grouped GEMM.

    Returns:
        ModuleSpec for a TransformerLayer.
    """
    _, RowParallel, DotProductAttention, LayerNormColumnParallel, Norm = (
        _get_linear_modules()
    )

    mlp_spec = _get_mlp_module_spec(
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
    )

    # For dense layers, the post_attention_layernorm is fused into
    # TELayerNormColumnParallelLinear (linear_fc1), so pre_mlp_layernorm = IdentityOp (default).
    # For MoE layers, a separate pre_mlp_layernorm is needed since the MoE
    # does not have a fused layernorm path.
    layer_submodules = TransformerLayerSubmodules(
        self_attention=ModuleSpec(
            module=SelfAttention,
            params={"attn_mask_type": AttnMaskType.causal},
            submodules=SelfAttentionSubmodules(
                linear_qkv=LayerNormColumnParallel,
                core_attention=DotProductAttention,
                linear_proj=RowParallel,
            ),
        ),
        self_attn_bda=get_bias_dropout_add,
        mlp=mlp_spec,
        mlp_bda=get_bias_dropout_add,
    )

    # MoE layers need separate pre_mlp_layernorm (not fused into MLP)
    if num_experts is not None:
        layer_submodules.pre_mlp_layernorm = Norm

    return ModuleSpec(
        module=TransformerLayer,
        submodules=layer_submodules,
    )


def get_ernie45_vl_decoder_block_spec(
    config,
    use_transformer_engine: bool = True,
) -> TransformerBlockSubmodules:
    """Get the full decoder block spec for ERNIE 4.5 VL MoE.

    Creates a heterogeneous block where layer types are determined by
    config.moe_layer_freq (list of 0/1 per layer):
    - 0: dense MLP layer
    - 1: ErnieMultiTypeMoE layer (dual-pool MoE)

    Args:
        config: TransformerConfig with moe_layer_freq, num_moe_experts, etc.
        use_transformer_engine: Whether to use TE modules.

    Returns:
        TransformerBlockSubmodules with heterogeneous layer specs.
    """
    num_experts = getattr(config, "num_moe_experts", None)
    moe_grouped_gemm = getattr(config, "moe_grouped_gemm", False)

    # Dense layer spec (no MoE)
    dense_layer_spec = _get_ernie_decoder_layer_spec(
        num_experts=None,
        moe_grouped_gemm=False,
    )

    # MoE layer spec (dual-pool)
    moe_layer_spec = _get_ernie_decoder_layer_spec(
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
    )

    # Build per-layer specs based on moe_layer_freq
    moe_layer_freq = getattr(config, "moe_layer_freq", None)
    if moe_layer_freq is None:
        # Default: all MoE
        moe_layer_freq = [1] * config.num_layers

    layer_specs = []
    for i in range(config.num_layers):
        if isinstance(moe_layer_freq, list):
            is_moe = moe_layer_freq[i]
        else:
            is_moe = moe_layer_freq
        layer_specs.append(moe_layer_spec if is_moe else dense_layer_spec)

    # Slice for pipeline parallelism
    offset = get_transformer_layer_offset(config)
    num_layers_to_build = get_num_layers_to_build(config)
    layer_specs = layer_specs[offset: offset + num_layers_to_build]

    # Get the Norm class for final_layernorm (TENorm or WrappedTorchNorm).
    # Without this, TransformerBlock.final_layernorm would be None because
    # TransformerBlockSubmodules.layer_norm defaults to None.
    _, _, _, _, Norm = _get_linear_modules()

    return TransformerBlockSubmodules(layer_specs=layer_specs, layer_norm=Norm)
