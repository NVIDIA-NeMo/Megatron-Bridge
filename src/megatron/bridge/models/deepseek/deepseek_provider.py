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
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from megatron.bridge.models.mla_provider import MLAModelProvider
from megatron.bridge.utils.common_utils import get_rank_safe


def _warn_deprecated(old_cls: str, new_cls: str = "MLAModelProvider") -> None:
    if get_rank_safe() == 0:
        warnings.warn(
            f"{old_cls} is deprecated and will be removed in a future release. "
            f"Use {new_cls} with MEGATRON_DEFAULTS in the bridge instead.",
            DeprecationWarning,
            stacklevel=3,
        )


@dataclass
class DeepSeekModelProvider(MLAModelProvider):
    """Deprecated alias for ``MLAModelProvider``.

    Deprecated:
        This alias remains for backward compatibility and will be removed in a
        future release. Use ``MLAModelProvider`` instead.
    """

    # Common DeepSeek defaults
    normalization: str = "RMSNorm"
    activation_func: Callable = F.silu
    gated_linear_unit: bool = True
    position_embedding_type: str = "rope"
    add_bias_linear: bool = False
    share_embeddings_and_output_weights: bool = False
    qk_layernorm: bool = True
    bf16: bool = True
    params_dtype: torch.dtype = torch.bfloat16
    moe_grouped_gemm: bool = True
    moe_token_dispatcher_type: str = "alltoall"
    # MLA defaults
    q_lora_rank: int | None = 1536
    kv_lora_rank: int = 512

    def __post_init__(self) -> None:
        _warn_deprecated("DeepSeekModelProvider")
        super().__post_init__()


@dataclass
class DeepSeekV2ModelProvider(MLAModelProvider):
    """
    DeepSeek-V2 Model: https://github.com/deepseek-ai/DeepSeek-V2
    """

    num_layers: int = 60
    hidden_size: int = 5120
    ffn_hidden_size: int = 12288
    num_moe_experts: int = 160
    moe_ffn_hidden_size: int = 1536
    moe_shared_expert_intermediate_size: int = 3072  # 1536 * 2 shared experts
    moe_layer_freq: int | list[int] = field(default_factory=lambda: [0] + [1] * 59)  # first layer is dense
    moe_router_topk: int = 6
    moe_router_num_groups: int = 8
    moe_router_group_topk: int = 3
    moe_router_topk_scaling_factor: float = 16.0
    moe_aux_loss_coeff: float = 1e-3
    mscale: float = 0.707
    mscale_all_dim: float = 0.707
    vocab_size: int = 102400

    def __post_init__(self) -> None:
        _warn_deprecated("DeepSeekV2ModelProvider")
        super().__post_init__()


@dataclass
class DeepSeekV2LiteModelProvider(MLAModelProvider):
    """
    DeepSeek-V2-Lite Model: https://github.com/deepseek-ai/DeepSeek-V2
    HuggingFace: https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite
    """

    num_layers: int = 27
    hidden_size: int = 2048
    ffn_hidden_size: int = 10944
    num_attention_heads: int = 16
    kv_channels: int = 16
    q_lora_rank: int | None = None
    num_moe_experts: int = 64
    moe_ffn_hidden_size: int = 1408
    moe_shared_expert_intermediate_size: int = 2816  # 1408 * 2 shared experts
    moe_layer_freq: int | list[int] = field(default_factory=lambda: [0] + [1] * 26)  # first layer is dense
    moe_router_topk: int = 6
    moe_router_num_groups: int = 1
    moe_router_group_topk: int = 1
    moe_router_topk_scaling_factor: float = 1.0
    mscale: float = 0.707
    mscale_all_dim: float = 0.707
    vocab_size: int = 102400

    def __post_init__(self) -> None:
        _warn_deprecated("DeepSeekV2LiteModelProvider")
        super().__post_init__()


@dataclass
class DeepSeekV3ModelProvider(MLAModelProvider):
    """
    DeepSeek-V3 Model: https://github.com/deepseek-ai/DeepSeek-V3
    """

    num_layers: int = 61
    hidden_size: int = 7168
    ffn_hidden_size: int = 18432
    kv_channels: int = 128
    num_moe_experts: int = 256
    moe_ffn_hidden_size: int = 2048
    moe_shared_expert_intermediate_size: int = 2048  # 2048 * 1 shared expert
    moe_layer_freq: int | list[int] = field(default_factory=lambda: [0] * 3 + [1] * 58)  # first three layers are dense
    moe_router_topk: int = 8
    moe_router_num_groups: int = 8
    moe_router_group_topk: int = 4
    moe_router_topk_scaling_factor: float = 2.5
    moe_aux_loss_coeff: float = 1e-4
    make_vocab_size_divisible_by: int = 1280
    moe_router_score_function: str = "sigmoid"
    moe_router_enable_expert_bias: bool = True
    moe_router_bias_update_rate: float = 1e-3
    mscale: float = 1.0
    mscale_all_dim: float = 1.0
    vocab_size: int = 129280

    def __post_init__(self) -> None:
        _warn_deprecated("DeepSeekV3ModelProvider")
        super().__post_init__()


@dataclass
class MoonlightModelProvider16B(MLAModelProvider):
    """
    Moonlight-16B-A3B Model: https://github.com/moonshotai/Moonlight-16B-A3B

    Moonlight is based on DeepSeek-V3.
    """

    max_position_embeddings: int = 4096
    num_layers: int = 27
    hidden_size: int = 2048
    ffn_hidden_size: int = 11264
    num_attention_heads: int = 16
    kv_channels: int = 16
    num_moe_experts: int = 64
    moe_ffn_hidden_size: int = 1408
    moe_shared_expert_intermediate_size: int = 2816  # 1408 * 2 shared expert
    moe_layer_freq: int | list[int] = field(default_factory=lambda: [0] * 1 + [1] * 26)  # first layer is dense
    moe_router_topk: int = 6
    moe_router_num_groups: int = 1
    moe_router_group_topk: int = 1
    moe_router_topk_scaling_factor: float = 2.446
    moe_aux_loss_coeff: float = 0.001
    make_vocab_size_divisible_by: int = 1280
    moe_router_score_function: str = "sigmoid"
    moe_router_enable_expert_bias: bool = True
    rotary_scaling_factor: float = 1.0
    mscale: float = 1.0
    mscale_all_dim: float = 1.0
    rotary_base: float = 50000
    layernorm_epsilon: float = 1e-5
    q_lora_rank: int = None
    init_method_std: float = 0.02
    moe_router_bias_update_rate: float = 1e-3
    rotary_percent: float = 1.0
    vocab_size: int = 163842

    def __post_init__(self) -> None:
        _warn_deprecated("MoonlightModelProvider16B")
        super().__post_init__()


# Legacy aliases for backward compatibility
DeepSeekProvider = DeepSeekModelProvider
DeepSeekV2Provider = DeepSeekV2ModelProvider
DeepSeekV2LiteProvider = DeepSeekV2LiteModelProvider
DeepSeekV3Provider = DeepSeekV3ModelProvider
MoonlightProvider = MoonlightModelProvider16B
