# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Model provider for the Kimi K3 language backbone."""

from dataclasses import dataclass

from megatron.bridge.models.mla_provider import MLAModelProvider


@dataclass
class KimiK3ModelProvider(MLAModelProvider):
    """Megatron configuration and provider for Kimi K3."""

    variable_seq_lengths: bool = True
    kimi_kda_layers: tuple[int, ...] = ()
    kimi_linear_num_heads: int = 96
    kimi_linear_head_dim: int = 128
    kimi_linear_conv_kernel_size: int = 4
    kimi_kda_gate_lower_bound: float = -5.0
    kimi_attn_res_block_size: int = 12
