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

"""Shensi configuration for HuggingFace checkpoints and Megatron-Core."""

from dataclasses import dataclass
from typing import List, Optional

from megatron.core.ssm.kda_layer_config import KDALayerConfig

from megatron.bridge.models.transformer_config import MLATransformerConfig


SHENSI_LAYER_TYPES = ("indexed_attention", "linear_attention")
SHENSI_MLP_LAYER_TYPES = ("dense", "sparse")
SHENSI_INDEXER_TYPES = ("full", "shared")
SHENSI_ATTN_RES_BLOCK_TYPES = ("block_write_layer", "block_read_layer")

# Shensi-only fields carried on the transformer config for tooling and the bridge-side model.
SHENSI_ONLY_FIELDS: dict[str, str] = {
    "latent_size": "width of the routed-expert and attention-residual latent space",
    "deepemb_rank": "rank of the token-conditioned deep-embedding transform",
    "deepemb_kv_rank": "rank of the cached key/value latent",
    "kv_emb_rank": "rank of the token-embedding key/value factor",
    "index_block_size": "number of tokens averaged into one HISA block",
    "index_topk_blocks": "number of HISA blocks kept before token refinement",
    "index_topk_freq": "interval between full indexer layers",
    "index_latent_rank": "rank of the cached indexer keys",
    "indexer_loss_coeff": "weight of the indexer KL auxiliary loss",
    "hc_mult": "number of hyper-connection streams",
    "hc_active_streams": "streams updated by each sublayer",
    "hc_adapter_rank": "rank of the low-rank adapter attached to each routed stream",
    "attn_res_rank": "rank of the attention-residual depth-routing keys and channel gates",
    "hc_fixed_streams": "streams always updated",
    "hc_conv_kernels": "depthwise kernel sizes that create the streams",
    "hc_stream_rank": "rank of the stream basis used between layers",
    "attn_res_block_size": "attention-residual block length",
    "attn_res_block_summary_rank": "rank of the attention-residual block bank",
    "rosa_embed_dim": "width of the ROSA token embeddings",
    "rosa_layer_ids": "layers that inject ROSA predictions",
    "rosa_reset_token_id": "token id that resets the ROSA suffix automaton",
    "rosa_loss_coef": "weight of the ROSA next-token auxiliary loss",
    "erc_loss_alpha": "ERC specialization threshold",
    "erc_loss_coef": "ERC loss weight",
    "fp32_keep": "keep the HF `_keep_in_fp32_modules_strict` hits in fp32 while training in bf16",
}

SHENSI_PENDING_FIELDS: dict[str, str] = {}


def resolve_mlp_layer_types(mlp_layer_types: List[str], first_k_dense_replace: int, num_layers: int) -> List[str]:
    """Per-layer FFN plan, expanding the dense prefix when only the count is given."""
    if mlp_layer_types is None:
        return ["dense"] * int(first_k_dense_replace) + ["sparse"] * (int(num_layers) - int(first_k_dense_replace))
    types = list(mlp_layer_types)
    if len(types) != int(num_layers):
        raise ValueError(f"mlp_layer_types needs {num_layers} entries, got {len(types)}")
    bad = [t for t in types if t not in SHENSI_MLP_LAYER_TYPES]
    if bad:
        raise ValueError(f"mlp_layer_types entries must be one of {SHENSI_MLP_LAYER_TYPES}, got {bad}")
    return types


def resolve_attn_res_block_layer_types(
    mlp_layer_types: List[str], attn_res_block_size: int, num_layers: int
) -> List[str]:
    """Attention-residual block write/read role per layer."""
    n_dense = sum(1 for t in mlp_layer_types if t == "dense")
    return [
        "block_write_layer"
        if i == 0 or (i >= n_dense and (i - n_dense) % int(attn_res_block_size) == 0)
        else "block_read_layer"
        for i in range(int(num_layers))
    ]


def resolve_layer_types(attn_res_block_layer_types: List[str]) -> List[str]:
    """Attention plan: indexed attention on every attention-residual write layer."""
    return [
        "indexed_attention" if block_type == "block_write_layer" else "linear_attention"
        for block_type in attn_res_block_layer_types
    ]


def resolve_indexer_types(
    layer_types: List[str], indexer_types: Optional[List[str]], index_topk_freq: int
) -> List[str]:
    """Per-layer indexer mode; full indexers recur every ``index_topk_freq`` indexed layers."""
    num_layers = len(layer_types)
    write_layers = [i for i, layer_type in enumerate(layer_types) if layer_type == "indexed_attention"]
    if indexer_types is None:
        resolved = ["shared"] * num_layers
        for pos, layer_idx in enumerate(write_layers):
            if pos % int(index_topk_freq) == 0:
                resolved[layer_idx] = "full"
        return resolved

    types = list(indexer_types)
    if len(types) == len(write_layers) and len(write_layers) != num_layers:
        resolved = ["shared"] * num_layers
        for pos, layer_idx in enumerate(write_layers):
            resolved[layer_idx] = types[pos]
        types = resolved
    if len(types) != num_layers:
        raise ValueError(f"indexer_types needs {num_layers} entries, got {len(types)}")
    bad = [t for t in types if t not in SHENSI_INDEXER_TYPES]
    if bad:
        raise ValueError(f"indexer_types entries must be one of {SHENSI_INDEXER_TYPES}, got {bad}")
    return types


def resolve_rosa_layer_ids(mlp_layer_types: List[str], rosa_layer_ids: Optional[List[int]]) -> List[int]:
    """ROSA sits at the first and third quartiles of the sparse stack by default."""
    if rosa_layer_ids is not None:
        return [int(i) for i in rosa_layer_ids]
    num_layers = len(mlp_layer_types)
    n_dense = sum(1 for t in mlp_layer_types if t == "dense")
    n_sparse = num_layers - n_dense
    first_rosa = n_dense + n_sparse // 4
    middle_rosa = n_dense + 3 * n_sparse // 4
    return sorted({first_rosa, middle_rosa})


@dataclass
class ShensiTransformerConfig(MLATransformerConfig, KDALayerConfig):
    """Megatron-Core transformer config with the Shensi-specific fields."""

    experimental_attention_variant: str = "dsa"
    multi_latent_attention: bool = True
    q_lora_rank: int = 320
    qk_head_dim: int = 512
    qk_pos_emb_head_dim: int = 0
    v_head_dim: int = 512
    kv_lora_rank: int = 64
    normalization: str = "RMSNorm"
    add_bias_linear: bool = False
    gated_linear_unit: bool = True
    qk_layernorm: bool = True
    apply_rope_fusion: bool = False
    rotary_interleaved: bool = False
    position_embedding_type: str = "none"

    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 2048
    index_block_size: int = 128
    index_topk_blocks: int = 64
    index_topk_freq: int = 4
    index_latent_rank: int = 32
    # The indexer trains through a KL auxiliary loss against the attention distribution
    # (DeepSeek-V3 lightning-indexer style); the top-k selection itself stays non-differentiable.
    indexer_loss_coeff: float = 0.01
    linear_num_heads: int = 16
    linear_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    linear_lower_bound: float = -5.0

    layer_types: Optional[List[str]] = None
    mlp_layer_types: Optional[List[str]] = None
    indexer_types: Optional[List[str]] = None
    attn_res_block_layer_types: Optional[List[str]] = None
    first_k_dense_replace: int = 3

    latent_size: int = 320
    num_moe_experts: int = 256
    moe_ffn_hidden_size: int = 640
    moe_router_topk: int = 6
    moe_router_score_function: str = "sqrtsoftplus"
    moe_router_topk_scaling_factor: float = 1.5
    moe_router_pre_softmax: bool = False
    moe_router_enable_expert_bias: bool = False
    moe_router_dtype: str = "fp32"
    moe_router_load_balancing_type: str = "none"
    moe_grouped_gemm: bool = True
    moe_latent_size: int = 320
    moe_use_norm_before_up_proj: bool = True
    norm_topk_prob: bool = True
    activation_func_clamp_value: float = 10.0

    hc_mult: int = 4
    hc_active_streams: int = 4
    hc_adapter_rank: int = 8
    hc_fixed_streams: int = 1
    hc_conv_kernels: Optional[List[int]] = None
    hc_stream_rank: int = 320
    hc_eps: float = 1.0e-6
    attn_res_rank: int = 32
    attn_res_block_size: int = 2
    attn_res_block_summary_rank: int = 320

    deepemb_rank: int = 32
    deepemb_kv_rank: int = 64
    kv_emb_rank: int = 128

    rosa_embed_dim: Optional[int] = None
    rosa_layer_ids: Optional[List[int]] = None
    rosa_reset_token_id: Optional[int] = None

    rosa_loss_coef: float = 0.0
    erc_loss_alpha: float = 0.5
    erc_loss_coef: float = 1.0
    fp32_keep: bool = True
    actual_vocab_size: Optional[int] = None


@dataclass
class ShensiConfig:
    """HuggingFace-side Shensi configuration."""

    vocab_size: int = 129280
    hidden_size: int = 1280
    latent_size: int = 320
    num_hidden_layers: int = 19
    num_mtp_layers: int = 1
    num_attention_heads: int = 16
    num_key_value_heads: int = 1
    head_dim: int = 512
    hidden_act: str = "silu"
    max_position_embeddings: int = 1048576
    initializer_range: float = 0.02
    rms_norm_eps: float = 1.0e-6
    use_cache: bool = True
    pad_token_id: Optional[int] = None
    bos_token_id: Optional[int] = 0
    eos_token_id: Optional[int] = 1
    tie_word_embeddings: bool = True
    attention_bias: bool = False
    attention_dropout: float = 0.0

    layer_types: Optional[List[str]] = None
    q_lora_rank: int = 320
    qk_nope_head_dim: Optional[int] = None
    qk_rope_head_dim: int = 0
    v_head_dim: Optional[int] = None

    deepemb_kv_rank: Optional[int] = None
    kv_emb_rank: Optional[int] = None
    deepemb_rank: int = 32

    index_block_size: int = 128
    index_head_dim: int = 128
    index_latent_rank: Optional[int] = None
    index_n_heads: int = 64
    index_topk: int = 2048
    index_topk_blocks: int = 64
    index_topk_freq: int = 4
    indexer_types: Optional[List[str]] = None
    indexer_loss_coeff: float = 0.01

    linear_conv_kernel_dim: int = 4
    linear_head_dim: int = 128
    linear_lower_bound: float = -5.0
    linear_num_heads: int = 16

    first_k_dense_replace: int = 3
    mlp_bias: bool = False
    mlp_layer_types: Optional[List[str]] = None
    swiglu_limit: float = 10.0

    moe_intermediate_size: int = 640
    n_routed_experts: int = 256
    norm_topk_prob: bool = True
    num_experts_per_tok: int = 6
    routed_scaling_factor: float = 1.5
    scoring_func: str = "sqrtsoftplus"

    hc_active_streams: int = 4
    hc_adapter_rank: int = 8
    hc_conv_kernels: Optional[List[int]] = None
    hc_eps: float = 1.0e-6
    hc_fixed_streams: int = 1
    hc_mult: int = 4
    hc_stream_rank: Optional[int] = None

    attn_res_rank: int = 32
    attn_res_block_layer_types: Optional[List[str]] = None
    attn_res_block_size: int = 2
    attn_res_block_summary_rank: Optional[int] = None

    rosa_embed_dim: Optional[int] = None
    rosa_layer_ids: Optional[List[int]] = None
    rosa_reset_token_id: Optional[int] = None

    rosa_loss_coef: float = 0.0
    erc_loss_alpha: float = 0.5
    erc_loss_coef: float = 1.0
    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.001

    def __post_init__(self) -> None:
        n = self.num_hidden_layers

        if self.num_key_value_heads is None:
            self.num_key_value_heads = 1
        if self.hc_conv_kernels is None:
            self.hc_conv_kernels = [4 * (i + 1) for i in range(self.hc_mult - 1)]
        else:
            self.hc_mult = 1 + len(self.hc_conv_kernels)
        if self.rosa_embed_dim is None:
            self.rosa_embed_dim = self.hidden_size
        if self.qk_nope_head_dim is None:
            self.qk_nope_head_dim = self.head_dim
        if self.v_head_dim is None:
            self.v_head_dim = self.head_dim

        if self.deepemb_kv_rank is None:
            self.deepemb_kv_rank = self.head_dim // 8
        if self.kv_emb_rank is None:
            self.kv_emb_rank = self.head_dim // 4
        if self.index_latent_rank is None:
            self.index_latent_rank = self.index_head_dim // 4
        if self.hc_stream_rank is None:
            self.hc_stream_rank = self.hidden_size // 4
        if self.attn_res_block_summary_rank is None:
            self.attn_res_block_summary_rank = self.hidden_size // 4

        self.mlp_layer_types = resolve_mlp_layer_types(self.mlp_layer_types, self.first_k_dense_replace, n)

        if self.attn_res_block_layer_types is None:
            self.attn_res_block_layer_types = resolve_attn_res_block_layer_types(
                self.mlp_layer_types, self.attn_res_block_size, n
            )

        if self.layer_types is None:
            self.layer_types = resolve_layer_types(self.attn_res_block_layer_types)
        self._validate_plans()

        self.indexer_types = resolve_indexer_types(self.layer_types, self.indexer_types, self.index_topk_freq)

        self.rosa_layer_ids = resolve_rosa_layer_ids(self.mlp_layer_types, self.rosa_layer_ids)

    def _validate_plans(self) -> None:
        n = self.num_hidden_layers
        for name, types, allowed in (
            ("layer_types", self.layer_types, SHENSI_LAYER_TYPES),
            ("mlp_layer_types", self.mlp_layer_types, SHENSI_MLP_LAYER_TYPES),
            ("attn_res_block_layer_types", self.attn_res_block_layer_types, SHENSI_ATTN_RES_BLOCK_TYPES),
        ):
            if types is None:
                continue
            if len(types) != n:
                raise ValueError(f"`{name}` needs `num_hidden_layers` ({n}) entries, got {len(types)}")
            bad = [t for t in types if t not in allowed]
            if bad:
                raise ValueError(f"`{name}` entries must be one of {allowed}; got {bad}.")
        if self.attn_res_block_size < 1:
            raise ValueError("attn_res_block_size must be at least 1")
        if self.hc_mult < 1:
            raise ValueError("hc_mult must be at least 1")
        if self.hc_active_streams > self.hc_mult:
            raise ValueError("hc_active_streams cannot exceed hc_mult")

    @property
    def num_attn_res_blocks(self) -> int:
        return sum(1 for t in self.attn_res_block_layer_types if t == "block_write_layer")
