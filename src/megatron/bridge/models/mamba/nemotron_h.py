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


@dataclass
class NemotronHModelProvider(MambaProvider):
    """Configuration for Nemotron-H models."""

    seq_length: int = 8192
    mamba_num_groups: int = 8
    mamba_head_dim: int = 64
    num_query_groups: int = 8
    make_vocab_size_divisible_by: int = 128
    activation_func: callable = lambda x: torch.pow(F.relu(x), 2)
    masked_softmax_fusion: bool = True
    apply_query_key_layer_scaling: bool = False
    persist_layer_norm: bool = True
    attention_softmax_in_fp32: bool = False
    vocab_size: int = 131072
    first_last_layers_bf16: bool = True
    is_hybrid_model: bool = True


@dataclass
class NemotronHModel8BProvider(NemotronHModelProvider):
    """Configuration for a 8B parameter Nemotron-H model."""

    hybrid_override_pattern: str = "M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-"
    num_layers: int = 52
    hidden_size: int = 4096
    mamba_state_dim: int = 128
    ffn_hidden_size: int = 21504
    num_attention_heads: int = 32
