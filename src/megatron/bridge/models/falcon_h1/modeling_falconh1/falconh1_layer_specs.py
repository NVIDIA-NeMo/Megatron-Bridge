# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TENorm,
    TERowParallelLinear,
)
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.ssm.mamba_layer import MambaLayer, MambaLayerSubmodules
from megatron.core.ssm.mamba_mixer import MambaMixerSubmodules
from megatron.core.ssm.mlp_layer import MLPLayer
from megatron.core.transformer.attention import SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules

from megatron.bridge.models.falcon_h1.modeling_falconh1.falconh1_block import (
    FalconH1Stack,
    FalconH1StackSubmodules,
)
from megatron.bridge.models.falcon_h1.modeling_falconh1.falconh1_layer import (
    FalconH1Layer,
    FalconH1MambaMixer,
    FalconH1MLP,
    FalconH1SelfAttention,
    FalconH1Submodules,
)


falconh1_stack_spec = ModuleSpec(
    module=FalconH1Stack,
    submodules=FalconH1StackSubmodules(
        mamba_layer=ModuleSpec(
            module=MambaLayer,
            submodules=MambaLayerSubmodules(
                mixer=ModuleSpec(
                    module=FalconH1MambaMixer,
                    submodules=MambaMixerSubmodules(
                        in_proj=TELayerNormColumnParallelLinear, out_proj=TERowParallelLinear
                    ),
                ),
                mamba_bda=get_bias_dropout_add,
            ),
        ),
        # Started with spec from gpt_layer_specs.py (with MLP removed)
        # Using the TE spec because we had problems getting the non-TE spec
        # working
        attention_layer=ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                self_attention=ModuleSpec(
                    module=FalconH1SelfAttention,
                    params={"attn_mask_type": AttnMaskType.causal},
                    submodules=SelfAttentionSubmodules(
                        linear_qkv=TELayerNormColumnParallelLinear,
                        core_attention=TEDotProductAttention,
                        linear_proj=TERowParallelLinear,
                    ),
                ),
                self_attn_bda=get_bias_dropout_add,
            ),
        ),
        # Started with spec from gpt_layer_specs.py
        # Using the TE spec because we had problems getting the non-TE spec
        # working
        mlp_layer=ModuleSpec(
            module=MLPLayer,
            submodules=TransformerLayerSubmodules(
                mlp=ModuleSpec(
                    module=FalconH1MLP,
                    submodules=MLPSubmodules(
                        linear_fc1=TELayerNormColumnParallelLinear, linear_fc2=TERowParallelLinear
                    ),
                ),
                mlp_bda=get_bias_dropout_add,
            ),
        ),
        # Updated FalconH1 Hybrid Mixer specification
        falconh1_layer=ModuleSpec(
            module=FalconH1Layer,
            params={"attn_mask_type": AttnMaskType.causal},
            submodules=FalconH1Submodules(
                norm=TENorm,
                # SSM component - complete MambaMixer spec
                mamba_mixer=ModuleSpec(
                    module=FalconH1MambaMixer,
                    submodules=MambaMixerSubmodules(
                        in_proj=TEColumnParallelLinear,
                        out_proj=TERowParallelLinear,
                    ),
                ),
                # Attention component - complete SelfAttention spec
                self_attention=ModuleSpec(
                    module=FalconH1SelfAttention,
                    submodules=SelfAttentionSubmodules(
                        linear_qkv=TEColumnParallelLinear,
                        core_attention=TEDotProductAttention,
                        linear_proj=TERowParallelLinear,
                    ),
                ),
                falconh1_bda=get_bias_dropout_add,
                # MLP component
                mlp=ModuleSpec(
                    module=FalconH1MLP,
                    submodules=MLPSubmodules(
                        linear_fc1=TELayerNormColumnParallelLinear, linear_fc2=TERowParallelLinear
                    ),
                ),
                mlp_bda=get_bias_dropout_add,
            ),
        ),
    ),
)
