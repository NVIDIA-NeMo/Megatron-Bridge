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

"""Shensi configuration: the HF-side field surface plus its validation rules.

A Shensi checkpoint ships three per-layer plans that the rest of the stack derives from:

* ``layer_types`` — attention plan, one entry per decoder layer, either ``sliding_attention``,
  ``compressed_sparse_attention`` (compress ratio 4) or ``heavily_compressed_attention`` (128);
* ``mlp_layer_types`` — FFN plan, ``hash_moe`` for the first ``default_num_hash_layers`` layers
  and ``moe`` afterwards;
* ``compress_ratios`` — the same attention plan in numeric form, extended with ``mtp_num_layers``
  entries when multi-token prediction is enabled.

Attention residuals are laid on top: a layer writes a new residual block when it is a hash-MoE
layer boundary, otherwise it reads the block written before it.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from megatron.bridge.models.transformer_config import MLATransformerConfig


SHENSI_LAYER_TYPES = (
    "sliding_attention",
    "compressed_sparse_attention",
    "heavily_compressed_attention",
)
SHENSI_MLP_LAYER_TYPES = ("hash_moe", "moe")

CSA_RATIO_BY_LAYER_TYPE = {
    "sliding_attention": 0,
    "compressed_sparse_attention": 4,
    "heavily_compressed_attention": 128,
}
_COMPRESS_RATIO_TO_LAYER_TYPE = {v: k for k, v in CSA_RATIO_BY_LAYER_TYPE.items()}

# Shensi fields that have no Megatron-Core counterpart; carried on the transformer config for tooling.
SHENSI_ONLY_FIELDS: dict[str, str] = {
    "routed_expert_hidden_size": "low-rank bottleneck of the routed experts (mapped to moe_latent_size)",
    "attn_res_block_size": "attention-residual block length (block write / block read layers)",
    "erc_loss_alpha": "ERC anchor coefficient",
    "erc_loss_coef": "ERC loss weight",
    "fp32_keep": "keep the HF `_keep_in_fp32_modules_strict` hits in fp32 while training in bf16",
}

# Known gaps between the Shensi config surface and Megatron-Core; surfaced for tooling.
# Empty since the attention-residual state rides the stage payload: a block that crosses a
# pipeline stage boundary needs no Megatron-Core slot of its own.
SHENSI_PENDING_FIELDS: dict[str, str] = {}


def resolve_csa_compress_ratios(layer_types: list[str]) -> list[int]:
    """Per-layer attention compress ratios from the layer type list."""
    bad = [t for t in layer_types if t not in CSA_RATIO_BY_LAYER_TYPE]
    if bad:
        raise ValueError(f"unknown layer_types {bad}, expected one of {tuple(CSA_RATIO_BY_LAYER_TYPE)}")
    return [CSA_RATIO_BY_LAYER_TYPE[t] for t in layer_types]


def resolve_moe_n_hash_layers(mlp_layer_types: list[str]) -> int:
    """Number of leading hash-MoE layers from the MLP layer type list."""
    n_hash = sum(1 for t in mlp_layer_types if t == "hash_moe")
    expected = ["hash_moe"] * n_hash + ["moe"] * (len(mlp_layer_types) - n_hash)
    if list(mlp_layer_types) != expected:
        raise NotImplementedError(
            "mlp_layer_types must be `n_hash` hash-MoE layers followed by plain MoE layers to map "
            f"onto moe_n_hash_layers; got {list(mlp_layer_types)}"
        )
    return n_hash


@dataclass
class ShensiTransformerConfig(MLATransformerConfig):
    """Megatron-Core transformer config with the Shensi-specific fields."""

    experimental_attention_variant: str = "dsv4_hybrid"
    routed_expert_hidden_size: int = 320
    attn_res_block_size: int = 2
    erc_loss_alpha: float = 0.5
    erc_loss_coef: float = 1.0
    csa_compress_rotary_base: float = 160000.0
    fp32_keep: bool = True
    attn_res_pp_state_transfer: bool = True
    add_bias_linear: bool = False
    gated_linear_unit: bool = True
    qk_layernorm: bool = True
    moe_grouped_gemm: bool = True
    moe_router_score_function: str = "sqrtsoftplus"
    moe_router_topk_scaling_factor: float = 1.5
    ademamix_betas: tuple[float, float, float] = (0.9, 0.999, 0.9999)
    ademamix_alpha: float = 5.0
    ademamix_t_alpha_beta3: int = 0
    # Shensi fields the model reads off its config. They are declared here (not only carried as
    # provider attributes) so the builder-backed model config can map them onto the transformer.
    o_groups: int = 2
    o_lora_rank: int = 320
    moe_n_hash_layers: int = 0
    actual_vocab_size: Optional[int] = None
    apply_dsa_kernel_fusion: bool = False


@dataclass
class ShensiConfig:
    """HF-side configuration of a Shensi checkpoint.

    The defaults describe the reference geometry used by the toy/CI checkpoints; released
    checkpoints always carry every field explicitly in ``config.json``.
    """

    vocab_size: int = 129280
    hidden_size: int = 1280
    moe_intermediate_size: int = 800
    num_hidden_layers: int = 19
    num_attention_heads: int = 16
    num_key_value_heads: int = 1
    head_dim: int = 512
    q_lora_rank: int = 320
    default_partial_rotary_factor: float = 64.0 / 512.0
    num_experts_per_tok: int = 6
    n_routed_experts: int = 256
    scoring_func: str = "sqrtsoftplus"
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.5
    max_position_embeddings: int = 1048576
    rope_theta: float = 10000.0
    layer_types: Optional[List[str]] = None
    compress_ratios: Optional[List[int]] = None
    compress_rates: Optional[Dict[str, int]] = None
    default_compress_rates: Dict[str, int] = field(
        default_factory=lambda: {
            "compressed_sparse_attention": 4,
            "heavily_compressed_attention": 128,
        }
    )
    compress_rope_theta: float = 160000.0
    mlp_layer_types: Optional[List[str]] = None
    default_num_hash_layers: int = 3
    swiglu_limit: float = 10.0
    routed_expert_hidden_size: int = 320
    router_aux_loss_coef: float = 0.001
    output_router_logits: bool = False
    sliding_window: int = 128
    o_groups: int = 2
    o_lora_rank: int = 320
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    num_nextn_predict_layers: int = 1
    attn_res_block_size: int = 2
    erc_loss_alpha: float = 0.5
    erc_loss_coef: float = 1.0
    indexer_loss_coeff: float = 0.01
    hidden_act: str = "silu"
    initializer_range: float = 0.02
    rms_norm_eps: float = 1.0e-6
    attention_dropout: float = 0.0
    mlp_bias: bool = False
    attention_bias: bool = False
    tie_word_embeddings: bool = False
    # RoPE is split per attention path: `main` for the sliding/HCA path, `compress` for CSA.
    partial_rotary_factor: Optional[float] = None
    rope_parameters: Optional[dict] = None
    compress_rope_scaling_factor: float = 1.0
    compress_rope_original_max_position_embeddings: int = 4096
    compress_rope_beta_fast: float = 32.0
    compress_rope_beta_slow: float = 1.0
    compress_rope_mscale: float = 1.0
    compress_rope_mscale_all_dim: float = 0.0

    def __post_init__(self) -> None:
        n = self.num_hidden_layers
        if self.compress_rates is None:
            self.compress_rates = dict(self.default_compress_rates)
        self.compress_rates = dict(self.compress_rates)
        if self.compress_ratios is not None:
            ratios = [int(r) for r in self.compress_ratios]
            want = n + int(self.num_nextn_predict_layers or 0)
            if len(ratios) != want:
                raise ValueError(
                    "compress_ratios needs num_hidden_layers + num_nextn_predict_layers = "
                    f"{want} entries, got {len(ratios)}"
                )
            bad = sorted({r for r in ratios if r not in _COMPRESS_RATIO_TO_LAYER_TYPE})
            if bad:
                raise ValueError(
                    f"compress_ratios entries must be one of {tuple(sorted(_COMPRESS_RATIO_TO_LAYER_TYPE))}, got {bad}"
                )
            self.compress_ratios = ratios
            if self.layer_types is None:
                self.layer_types = [_COMPRESS_RATIO_TO_LAYER_TYPE[r] for r in ratios[:n]]
        if self.layer_types is None:
            interleave = [
                "compressed_sparse_attention" if i % 2 else "heavily_compressed_attention"
                for i in range(max(n - 2, 0))
            ]
            self.layer_types = ["heavily_compressed_attention"] * min(n, 2) + interleave
        self.layer_types = list(self.layer_types[:n])
        if self.mlp_layer_types is None:
            n_hash = self.default_num_hash_layers
            self.mlp_layer_types = ["hash_moe"] * min(n, n_hash) + ["moe"] * max(0, n - n_hash)
        self.mlp_layer_types = list(self.mlp_layer_types[:n])
        if self.partial_rotary_factor is None:
            self.partial_rotary_factor = self.default_partial_rotary_factor
        self.qk_rope_head_dim = int(self.head_dim * self.partial_rotary_factor)
        rp = self.rope_parameters or {}
        if isinstance(rp.get("main"), dict) and isinstance(rp.get("compress"), dict):
            self.rope_parameters = {"main": rp["main"], "compress": rp["compress"]}
        else:
            yarn = {k: v for k, v in rp.items() if k not in ("main", "compress")}
            main = {
                "rope_type": "default",
                "rope_theta": self.rope_theta,
                "partial_rotary_factor": self.partial_rotary_factor,
            }
            compress = {
                **yarn,
                "rope_theta": self.compress_rope_theta,
                "partial_rotary_factor": self.partial_rotary_factor,
            }
            compress.setdefault("rope_type", "default")
            if compress["rope_type"] == "yarn":
                compress.setdefault("attention_factor", 1.0)
            self.rope_parameters = {"main": main, "compress": compress}
        self._resolve_compress_rope_scaling()
        self.validate_layer_type()

    def _resolve_compress_rope_scaling(self) -> None:
        """Fold the compress-path RoPE settings into the flat fields used by the provider."""
        p = dict(self.rope_parameters.get("compress", {}))
        rtype = p.get("rope_type", "default")
        if rtype == "default":
            self.compress_rope_scaling_factor = 1.0
            self.compress_rope_mscale = 1.0
            self.compress_rope_mscale_all_dim = 0.0
            return
        if rtype == "yarn":
            if "factor" not in p:
                raise ValueError("compress rope_type=yarn requires rope_parameters['compress']['factor']")
            self.compress_rope_scaling_factor = float(p["factor"])
            self.compress_rope_original_max_position_embeddings = int(
                p.get("original_max_position_embeddings", self.max_position_embeddings)
            )
            self.compress_rope_beta_fast = float(p.get("beta_fast", 32.0))
            self.compress_rope_beta_slow = float(p.get("beta_slow", 1.0))
            self.compress_rope_mscale = 1.0
            self.compress_rope_mscale_all_dim = 0.0
            return
        raise NotImplementedError(
            f"compress rope_type={rtype!r} has no Megatron-Core mapping; add it explicitly, the "
            "default rotary_scaling_factor does not apply"
        )

    def validate_layer_type(self) -> None:
        for name, types, allowed in (
            ("layer_types", self.layer_types, SHENSI_LAYER_TYPES),
            ("mlp_layer_types", self.mlp_layer_types, SHENSI_MLP_LAYER_TYPES),
        ):
            if types is None:
                continue
            if len(types) != self.num_hidden_layers:
                raise ValueError(
                    f"`num_hidden_layers` ({self.num_hidden_layers}) must equal `len({name})` ({len(types)})."
                )
            bad = [t for t in types if t not in allowed]
            if bad:
                raise ValueError(f"`{name}` entries must be one of {allowed}; got {bad}.")

    @property
    def attn_res_block_layer_types(self) -> List[str]:
        """Attention-residual role per layer: the first layer and every hash-MoE boundary write."""
        n_hash = self.mlp_layer_types.count("hash_moe")
        return [
            (
                "block_write_layer"
                if i == 0 or (i >= n_hash and (i - n_hash) % self.attn_res_block_size == 0)
                else "block_read_layer"
            )
            for i in range(self.num_hidden_layers)
        ]

    @property
    def num_attn_res_blocks(self) -> int:
        return self.attn_res_block_layer_types.count("block_write_layer")
