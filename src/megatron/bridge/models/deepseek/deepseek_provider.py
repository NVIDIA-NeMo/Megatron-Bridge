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
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Callable

import torch
import torch.nn.functional as F
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.transformer_config import MLATransformerConfig
from megatron.bridge.utils.common_utils import get_rank_safe


try:
    import transformer_engine  # type: ignore  # noqa: F401

    HAVE_TE = True
except (ImportError, ModuleNotFoundError):
    HAVE_TE = False

if TYPE_CHECKING:
    from megatron.core.transformer import ModuleSpec

if HAVE_TE:
    from megatron.core.utils import is_te_min_version


@dataclass
class DeepSeekModelProvider(MLATransformerConfig, GPTModelProvider):
    """
    Base config for DeepSeek V2 and V3 models.
    """

    transformer_layer_spec: "ModuleSpec" | Callable[["GPTModelProvider"], "ModuleSpec"] = partial(
        get_gpt_decoder_block_spec, use_transformer_engine=HAVE_TE
    )

    # Model
    normalization: str = "RMSNorm"
    activation_func: Callable = F.silu
    gated_linear_unit: bool = True  # swiglu
    position_embedding_type: str = "rope"
    add_bias_linear: bool = False
    share_embeddings_and_output_weights: bool = False
    num_attention_heads: int = 128
    kv_channels: int = 128
    max_position_embeddings: int = 4096
    seq_length: int = 4096
    rotary_base: float = 10000.0
    make_vocab_size_divisible_by: int = 3200
    mtp_num_layers: int | None = None
    mtp_loss_scaling_factor: float | None = None

    # Regularization
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    qk_layernorm: bool = True

    # MoE
    moe_grouped_gemm: bool = True
    moe_router_pre_softmax: bool = True
    moe_token_dispatcher_type: str = "alltoall"
    moe_router_load_balancing_type: str = "seq_aux_loss"
    moe_shared_expert_overlap: bool = True
    moe_router_dtype: str | None = "fp32"

    # MLA
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_head_dim: int = 128
    qk_pos_emb_head_dim: int = 64
    v_head_dim: int = 128
    rotary_scaling_factor: float = 40
    mscale: float = 1.0
    mscale_all_dim: float = 1.0

    # Miscellaneous
    init_method_std: float = 0.006
    layernorm_epsilon: float = 1e-6
    bf16: bool = True
    params_dtype: torch.dtype = torch.bfloat16
    async_tensor_model_parallel_allreduce: bool = True
    attention_softmax_in_fp32: bool = False
    persist_layer_norm: bool = True
    num_layers_in_first_pipeline_stage: int | None = None
    num_layers_in_last_pipeline_stage: int | None = None
    account_for_embedding_in_pipeline_split: bool = False
    account_for_loss_in_pipeline_split: bool = False

    # MLA specific
    multi_latent_attention: bool = True

    # fusions
    apply_rope_fusion: bool = False
    bias_activation_fusion: bool = True
    bias_dropout_fusion: bool = True
    masked_softmax_fusion: bool = True
    cross_entropy_loss_fusion: bool = True
    cross_entropy_fusion_impl: str = "te"
    moe_permute_fusion: bool = is_te_min_version("2.1.0") if HAVE_TE else False


@dataclass
class DeepSeekV2ModelProvider(DeepSeekModelProvider):
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


@dataclass
class DeepSeekV2LiteModelProvider(DeepSeekV2ModelProvider):
    """
    DeepSeek-V2-Lite Model: https://github.com/deepseek-ai/DeepSeek-V2
    HuggingFace: https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite
    """

    num_layers: int = 27
    hidden_size: int = 2048
    ffn_hidden_size: int = 10944
    num_attention_heads: int = 16
    kv_channels: int = 16
    q_lora_rank: int = None
    num_moe_experts: int = 64
    moe_ffn_hidden_size: int = 1408
    moe_shared_expert_intermediate_size: int = 2816  # 1408 * 2 shared experts
    moe_layer_freq: int | list[int] = field(default_factory=lambda: [0] + [1] * 26)  # first layer is dense
    moe_router_topk: int = 6
    moe_router_num_groups: int = 1
    moe_router_group_topk: int = 1
    moe_router_topk_scaling_factor: float = 1.0
    vocab_size: int = 102400


@dataclass
class DeepSeekV3ModelProvider(DeepSeekModelProvider):
    """
    DeepSeek-V3 Model: https://github.com/deepseek-ai/DeepSeek-V3
    """

    num_layers: int = 61
    hidden_size: int = 7168
    ffn_hidden_size: int = 18432
    num_moe_experts: int = 256
    moe_ffn_hidden_size: int = 2048
    moe_shared_expert_intermediate_size: int = 2048  # 2048 * 1 shared expert
    moe_layer_freq: int | list[int] = field(default_factory=lambda: [0] * 3 + [1] * 58)  # first three layers are dense
    moe_router_topk: int = 8
    moe_router_num_groups: int = 8
    moe_router_group_topk: int = 4
    moe_router_topk_scaling_factor: float = 2.5
    make_vocab_size_divisible_by: int = 1280
    moe_router_score_function: str = "sigmoid"
    moe_router_enable_expert_bias: bool = True
    moe_router_bias_update_rate: float = 1e-3
    mscale: float = 1.0
    mscale_all_dim: float = 1.0
    vocab_size: int = 129280


@dataclass
class MoonlightModelProvider16B(DeepSeekModelProvider):
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
    vocab_size: int = 163840


# -----------------------------------------------------------------------------
# Deprecated aliases (to be removed in a future release)
# -----------------------------------------------------------------------------


def _warn_deprecated(old_cls: str, new_cls: str) -> None:
    if get_rank_safe() == 0:
        warnings.warn(
            f"{old_cls} is deprecated and will be removed in a future release. Use {new_cls} instead.",
            DeprecationWarning,
            stacklevel=2,
        )


@dataclass
class DeepSeekProvider(DeepSeekModelProvider):
    """Deprecated alias for ``DeepSeekModelProvider``.

    Deprecated:
        This alias remains for backward compatibility and will be removed in a
        future release. Import and use ``DeepSeekModelProvider`` instead.
    """

    def __post_init__(self) -> None:
        _warn_deprecated("DeepSeekProvider", "DeepSeekModelProvider")
        super().__post_init__()


@dataclass
class DeepSeekV2Provider(DeepSeekV2ModelProvider):
    """Deprecated alias for ``DeepSeekV2ModelProvider``.

    Deprecated:
        This alias remains for backward compatibility and will be removed in a
        future release. Import and use ``DeepSeekV2ModelProvider`` instead.
    """

    def __post_init__(self) -> None:
        _warn_deprecated("DeepSeekV2Provider", "DeepSeekV2ModelProvider")
        super().__post_init__()


@dataclass
class DeepSeekV2LiteProvider(DeepSeekV2LiteModelProvider):
    """Deprecated alias for ``DeepSeekV2LiteModelProvider``.

    Deprecated:
        This alias remains for backward compatibility and will be removed in a
        future release. Import and use ``DeepSeekV2LiteModelProvider`` instead.
    """

    def __post_init__(self) -> None:
        _warn_deprecated("DeepSeekV2LiteProvider", "DeepSeekV2LiteModelProvider")
        super().__post_init__()


@dataclass
class DeepSeekV3Provider(DeepSeekV3ModelProvider):
    """Deprecated alias for ``DeepSeekV3ModelProvider``.

    Deprecated:
        This alias remains for backward compatibility and will be removed in a
        future release. Import and use ``DeepSeekV3ModelProvider`` instead.
    """

    def __post_init__(self) -> None:
        _warn_deprecated("DeepSeekV3Provider", "DeepSeekV3ModelProvider")
        super().__post_init__()


@dataclass
class MoonlightProvider(MoonlightModelProvider16B):
    """Deprecated alias for ``MoonlightModelProvider16B``.

    Deprecated:
        This alias remains for backward compatibility and will be removed in a
        future release. Import and use ``MoonlightModelProvider16B`` instead.
    """

    def __post_init__(self) -> None:
        _warn_deprecated("MoonlightProvider", "MoonlightModelProvider16B")
        super().__post_init__()
