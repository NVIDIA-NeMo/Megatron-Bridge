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
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from megatron.bridge.models.mamba.mamba_provider import MambaProvider


logger = logging.getLogger(__name__)


def nemotron_h_activation_func(x: torch.Tensor) -> torch.Tensor:
    """Nemotron-H activation function: torch.pow(F.relu(x), 2)"""
    return torch.pow(F.relu(x), 2)

def nemotron_h_mamba_stack_spec(config):
    """Custom mamba stack spec that properly uses the activation_func from config."""
    from megatron.core.models.mamba.mamba_layer_specs import mamba_stack_spec
    from megatron.core.transformer.spec_utils import ModuleSpec
    from megatron.core.transformer.mlp import MLPSubmodules
    from megatron.core.transformer.transformer_layer import TransformerLayerSubmodules
    import copy
    
    # Get the default spec - it might be a function or already a ModuleSpec
    if callable(mamba_stack_spec):
        spec = mamba_stack_spec()
    else:
        spec = mamba_stack_spec
    
    # Create a deep copy to avoid modifying the original spec
    spec = copy.deepcopy(spec)
    
    # Ensure we have a valid activation function
    activation_func = getattr(config, 'activation_func', None)
    if activation_func is None:
        print("WARNING: activation_func is None in mamba_stack_spec, using nemotron_h_activation_func")
        activation_func = nemotron_h_activation_func
    else:
        print(f"INFO: Using activation_func from config: {activation_func}")
    
    # Update the MLP layer to use the activation function from config
    if hasattr(spec, 'submodules') and hasattr(spec.submodules, 'mlp_layer'):
        mlp_layer = spec.submodules.mlp_layer
        if hasattr(mlp_layer, 'submodules') and hasattr(mlp_layer.submodules, 'mlp'):
            mlp_spec = mlp_layer.submodules.mlp
            if hasattr(mlp_spec, 'submodules'):
                old_submodules = mlp_spec.submodules
                if hasattr(old_submodules, '_asdict'):
                    mlp_dict = old_submodules._asdict()
                    mlp_dict['activation_func'] = activation_func
                    mlp_spec.submodules = type(old_submodules)(**mlp_dict)
                else:
                    mlp_spec.submodules.activation_func = activation_func
                print(f"INFO: Set activation_func in MLP submodules via: {activation_func}")
    
    return spec


@dataclass
class NemotronHModelProvider(MambaProvider):
    """Configuration for Nemotron-H models."""

    seq_length: int = 8192
    mamba_num_groups: int = 8
    mamba_head_dim: int = 64
    num_query_groups: int = 8
    make_vocab_size_divisible_by: int = 128
    activation_func: callable = nemotron_h_activation_func
    masked_softmax_fusion: bool = True
    apply_query_key_layer_scaling: bool = False
    persist_layer_norm: bool = True
    attention_softmax_in_fp32: bool = False
    first_last_layers_bf16: bool = True
    is_hybrid_model: bool = True

    def __post_init__(self):
        super().__post_init__()
        # Override the mamba_stack_spec to use our custom one that properly handles activation_func
        self.mamba_stack_spec = lambda: nemotron_h_mamba_stack_spec(self)
        
        # Ensure activation_func is not None
        if self.activation_func is None:
            print("WARNING: activation_func is None in __post_init__, setting to nemotron_h_activation_func")
            self.activation_func = nemotron_h_activation_func
        else:
            print(f"INFO: activation_func is properly set to {self.activation_func}")


@dataclass
class NemotronHModel4BProvider(NemotronHModelProvider):
    """Configuration for a 4B parameter Nemotron-H model."""

    hybrid_override_pattern: str = "M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-"
    num_layers: int = 52
    hidden_size: int = 3072
    mamba_num_heads: int = 112
    kv_channels: int = 128
    mamba_state_dim: int = 128
    ffn_hidden_size: int = 12288
    num_attention_heads: int = 32
    use_mamba_mem_eff_path: bool = False


@dataclass
class NemotronHModel8BProvider(NemotronHModelProvider):
    """Configuration for a 8B parameter Nemotron-H model."""

    hybrid_override_pattern: str = "M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-"
    num_layers: int = 52
    hidden_size: int = 4096
    mamba_state_dim: int = 128
    mamba_num_heads: int = 128
    ffn_hidden_size: int = 21504
    num_attention_heads: int = 32


@dataclass
class NemotronHModel47BProvider(NemotronHModelProvider):
    """Configuration for a 47B parameter Nemotron-H model."""

    hybrid_override_pattern: str = (
        "M-M-M-M-M-M-M-M-M*-M-M-M-M-M-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-M-M---MM---M-M*-M-M-M-M-M-"
    )
    num_layers: int = 98
    hidden_size: int = 8192
    mamba_state_dim: int = 256
    mamba_num_heads: int = 256
    ffn_hidden_size: int = 30720
    num_attention_heads: int = 64


@dataclass
class NemotronHModel56BProvider(NemotronHModelProvider):
    """Configuration for a 56B parameter Nemotron-H model."""

    hybrid_override_pattern: str = (
        "M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-"
        "M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-"
    )
    num_layers: int = 118
    hidden_size: int = 8192
    mamba_state_dim: int = 256
    mamba_num_heads: int = 256
    ffn_hidden_size: int = 32768
    num_attention_heads: int = 64


@dataclass
class NemotronNano9Bv2Provider(NemotronHModelProvider):
    """Configuration for a 9B parameter Nemotron Nano v2 model."""

    hybrid_override_pattern: str = "M-M-M-MM-M-M-M*-M-M-M*-M-M-M-M*-M-M-M-M*-M-MM-M-M-M-M-M-"
    num_layers: int = 56
    hidden_size: int = 4480
    mamba_num_heads: int = 128
    mamba_state_dim: int = 128
    ffn_hidden_size: int = 15680
    num_attention_heads: int = 40
    mamba_head_dim: int = 80
    seq_length: int = 131072


@dataclass
class NemotronNano12Bv2Provider(NemotronHModelProvider):
    """Configuration for a 12B parameter Nemotron Nano v2 model."""

    hybrid_override_pattern: str = "M-M-M-M*-M-M-M-M*-M-M-M-M*-M-M-M-M*-M-M-M-M*-M-M-M-M*-M-M-M-M-"
    num_layers: int = 62
    hidden_size: int = 5120
    mamba_num_heads: int = 128
    mamba_state_dim: int = 128
    ffn_hidden_size: int = 20480
    num_attention_heads: int = 40
    mamba_head_dim: int = 80
    seq_length: int = 131072
