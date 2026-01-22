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

from collections.abc import Callable
from dataclasses import dataclass, field

import torch.nn.functional as F
from megatron.core.transformer import ModuleSpec

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.llama.llama4_utils import get_llama4_layer_spec


@dataclass
class LlamaModelProvider(GPTModelProvider):
    """Configuration class for Llama models.

    Extends GPTConfig with specific settings optimized for Llama architectures.
    Includes configurations for normalization, activation functions, and various
    architecture-specific options.
    """

    # configs that are common across model sizes
    normalization: str = "RMSNorm"
    activation_func: Callable = F.silu
    gated_linear_unit: bool = True
    position_embedding_type: str = "rope"
    add_bias_linear: bool = False
    seq_length: int = 4096
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    share_embeddings_and_output_weights: bool = False
    # Fusions
    bias_activation_fusion: bool = True
    masked_softmax_fusion: bool = True
    persist_layer_norm: bool = True
    bias_dropout_fusion: bool = True
    apply_rope_fusion: bool = True
    use_transformer_engine_op_fuser: bool | None = None


@dataclass
class Llama2ModelProvider7B(LlamaModelProvider):
    """Configuration for a 7B parameter Llama 2 model.

    Specific configuration for the 7B Llama 2 model with 32 layers,
    4096 hidden size, and 32 attention heads.
    """

    num_layers: int = 32
    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_query_groups: int = 32
    ffn_hidden_size: int = 11008


@dataclass
class Llama2ModelProvider13B(LlamaModelProvider):
    """Configuration for a 13B parameter Llama 2 model.

    Specific configuration for the 13B Llama 2 model with 40 layers,
    5120 hidden size, and 40 attention heads.
    """

    num_layers: int = 40
    hidden_size: int = 5120
    num_attention_heads: int = 40
    num_query_groups: int = 40
    ffn_hidden_size: int = 13824


@dataclass
class Llama2ModelProvider70B(LlamaModelProvider):
    """Configuration for a 70B parameter Llama 2 model.

    Specific configuration for the 70B Llama 2 model with 80 layers,
    8192 hidden size, and 64 attention heads with 8 query groups.
    """

    num_layers: int = 80
    hidden_size: int = 8192
    num_attention_heads: int = 64
    num_query_groups: int = 8
    ffn_hidden_size: int = 28672


@dataclass
class Llama3ModelProvider(LlamaModelProvider):
    """Configuration for Llama 3 models.

    Base configuration for Llama 3 architecture with common settings
    across different model sizes, including group query attention (GQA)
    and architecture-specific settings.
    """

    num_query_groups: int = 8
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    normalization: str = "RMSNorm"
    init_method_std: float = 0.01
    layernorm_epsilon: float = 1.0e-05
    add_bias_linear: bool = False
    activation_func: Callable = F.silu
    gated_linear_unit: bool = True
    # Fusions
    bias_activation_fusion: bool = True
    masked_softmax_fusion: bool = True
    persist_layer_norm: bool = True
    bias_dropout_fusion: bool = True
    apply_rope_fusion: bool = True
    share_embeddings_and_output_weights: bool = False
    position_embedding_type: str = "rope"
    rotary_percent: float = 1.0


@dataclass
class Llama31ModelProvider(Llama3ModelProvider):
    """Configuration for Llama 3.1 models.

    Extends Llama3ModelProvider with specific settings for Llama 3.1 models,
    including RoPE scaling via Megatron Core's built-in support.
    """

    # RoPE scaling is now handled by Megatron Core's RotaryEmbedding
    rope_scaling: bool = True
    rope_scaling_factor: float = 8.0
    init_method_std: float = 0.02


@dataclass
class Llama3ModelProvider8B(Llama3ModelProvider):
    """Configuration for an 8B parameter Llama 3 model.

    Specific configuration for the 8B Llama 3 model with 32 layers,
    4096 hidden size, and 32 attention heads.
    """

    rotary_base: int = 500_000
    seq_length: int = 8192
    num_layers: int = 32
    hidden_size: int = 4096
    ffn_hidden_size: int = 14336
    num_attention_heads: int = 32
    cross_entropy_fusion_impl: str = "te"


@dataclass
class Llama3ModelProvider70B(Llama3ModelProvider):
    """Configuration for a 70B parameter Llama 3 model.

    Specific configuration for the 70B Llama 3 model with 80 layers,
    8192 hidden size, and 64 attention heads.
    """

    rotary_base: int = 500_000
    seq_length: int = 8192
    num_layers: int = 80
    hidden_size: int = 8192
    ffn_hidden_size: int = 28672
    num_attention_heads: int = 64
    init_method_std: float = 0.008944
    make_vocab_size_divisible_by: int = 128
    cross_entropy_fusion_impl: str = "te"


@dataclass
class Llama31ModelProvider8B(Llama31ModelProvider):
    """Configuration for an 8B parameter Llama 3.1 model.

    Specific configuration for the 8B Llama 3.1 model with 32 layers,
    4096 hidden size, and 32 attention heads, supporting a longer context
    length of 131K tokens.
    """

    rotary_base: int = 500_000
    seq_length: int = 131072
    num_layers: int = 32
    hidden_size: int = 4096
    ffn_hidden_size: int = 14336
    num_attention_heads: int = 32


@dataclass
class Llama31ModelProvider70B(Llama31ModelProvider):
    """Configuration for a 70B parameter Llama 3.1 model.

    Specific configuration for the 70B Llama 3.1 model with 80 layers,
    8192 hidden size, and 64 attention heads, supporting a longer context
    length of 131K tokens.
    """

    rotary_base: int = 500_000
    seq_length: int = 131072
    num_layers: int = 80
    hidden_size: int = 8192
    ffn_hidden_size: int = 28672
    num_attention_heads: int = 64
    make_vocab_size_divisible_by: int = 128


@dataclass
class Llama31ModelProvider405B(Llama31ModelProvider):
    """Configuration for a 405B parameter Llama 3.1 model.

    Specific configuration for the 405B Llama 3.1 model with 126 layers,
    16384 hidden size, and 128 attention heads, supporting a longer context
    length of 131K tokens.
    """

    rotary_base: int = 500_000
    seq_length: int = 131072
    num_layers: int = 126
    hidden_size: int = 16384
    ffn_hidden_size: int = 53248
    num_attention_heads: int = 128
    make_vocab_size_divisible_by: int = 128
    cross_entropy_fusion_impl: str = "te"


@dataclass
class Llama32ModelProvider1B(Llama31ModelProvider):
    """Configuration for a 1B parameter Llama 3.2 model.

    Specific configuration for the 1B Llama 3.2 model with 16 layers,
    2048 hidden size, and 32 attention heads (8 query groups).
    """

    rope_scaling_factor: float = 32.0
    share_embeddings_and_output_weights: bool = True
    rotary_base: int = 500_000
    seq_length: int = 131072
    num_layers: int = 16
    hidden_size: int = 2048
    ffn_hidden_size: int = 8192
    num_attention_heads: int = 32
    num_query_groups: int = 8
    make_vocab_size_divisible_by: int = 128


@dataclass
class Llama32ModelProvider3B(Llama31ModelProvider):
    """Configuration for a 3B parameter Llama 3.2 model.

    Specific configuration for the 3B Llama 3.2 model with 28 layers,
    3072 hidden size, and 24 attention heads (8 query groups).
    """

    rope_scaling_factor: float = 32.0
    share_embeddings_and_output_weights: bool = True
    rotary_base: int = 500_000
    seq_length: int = 131072
    num_layers: int = 28
    hidden_size: int = 3072
    ffn_hidden_size: int = 8192
    num_attention_heads: int = 24
    num_query_groups: int = 8
    make_vocab_size_divisible_by: int = 128


@dataclass
class CodeLlamaModelProvider7B(Llama2ModelProvider7B):
    """Configuration for a 7B parameter CodeLlama model.

    Extends Llama2ModelProvider7B with modified settings specifically for code generation,
    including longer context length and different rotary base.
    """

    rotary_base: int = 1_000_000
    seq_length: int = 16384


@dataclass
class CodeLlamaModelProvider13B(Llama2ModelProvider13B):
    """Configuration for a 13B parameter CodeLlama model.

    Extends Llama2ModelProvider13B with modified settings specifically for code generation,
    including longer context length and different rotary base.
    """

    rotary_base: int = 1_000_000
    seq_length: int = 16384


@dataclass
class CodeLlamaModelProvider34B(LlamaModelProvider):
    """Configuration for a 34B parameter CodeLlama model.

    Specific configuration for the 34B CodeLlama model with 48 layers,
    8192 hidden size, and 64 attention heads (8 query groups).
    """

    num_layers: int = 48
    hidden_size: int = 8192
    num_attention_heads: int = 64
    num_query_groups: int = 8
    ffn_hidden_size: int = 22016
    rotary_base: int = 1_000_000
    seq_length: int = 16384


@dataclass
class CodeLlamaModelProvider70B(Llama2ModelProvider70B):
    """Configuration for a 70B parameter CodeLlama model.

    Extends Llama2ModelProvider70B with settings specifically for code generation.
    """

    pass


@dataclass
class Llama4ModelProvider(Llama3ModelProvider):
    """
    Configuration for Llama4 language model.
    """

    rotary_base: int = 500_000
    seq_length: int = 8192
    num_layers: int = 48
    hidden_size: int = 5120
    ffn_hidden_size: int = 16384
    num_attention_heads: int = 40
    vocab_size: int = 25256 * 8
    add_bias_linear: bool = False
    gated_linear_unit: bool = True
    rotary_interleaved: bool = True
    apply_rope_fusion: bool = False
    nope_layer_interval: int = 4
    transformer_layer_spec: ModuleSpec | Callable[["LlamaModelProvider"], ModuleSpec] = field(
        default_factory=lambda: get_llama4_layer_spec
    )
    # MOE
    moe_grouped_gemm: bool = True
    moe_shared_expert_intermediate_size: int = 8192
    moe_ffn_hidden_size: int = 8192
    moe_router_topk: int = 1
    moe_router_pre_softmax: bool = False
    moe_router_score_function: str = "sigmoid"
    moe_token_dispatcher_type: str = "alltoall"
    moe_router_dtype: str | None = None
    moe_apply_probs_on_input: bool = True
    moe_shared_expert_overlap: bool = True
    moe_permute_fusion: bool = False
    # Configs that are overwritten in subclass models
    qk_l2_norm: bool = True
    rope_scaling: bool = True
    rope_scaling_factor: float = 8.0
    attention_chunk_size: int = 8192


@dataclass
class Llama4Experts16ModelProvider(Llama4ModelProvider):
    """
    Configuration for llama4 16-experts model.
    """

    num_moe_experts: int = 16
    rope_scaling: bool = True
    rope_scaling_factor: float = 8.0
    qk_l2_norm: bool = True


@dataclass
class Llama4Experts128ModelProvider(Llama4ModelProvider):
    """
    Configuration for llama4 128-experts model.
    """

    num_moe_experts: int = 128
    rope_scaling: bool = False
    moe_layer_freq: int | list[int] = field(default_factory=lambda: [0, 1] * 24)
    qk_l2_norm: bool = False
