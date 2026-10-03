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
from typing import Dict, List, Optional, Tuple

from megatron.core.transformer.transformer_config import MLATransformerConfig


SHENSI_LAYER_TYPES = (
    "sliding_attention",
    "compressed_sparse_attention",
    "heavily_compressed_attention",
)
SHENSI_MLP_LAYER_TYPES = ("hash_moe", "moe")
_COMPRESS_RATIO_TO_LAYER_TYPE = {
    0: "sliding_attention",
    4: "compressed_sparse_attention",
    128: "heavily_compressed_attention",
}


CSA_RATIO_BY_LAYER_TYPE = {
    "sliding_attention": 0,
    "compressed_sparse_attention": 4,
    "heavily_compressed_attention": 128,
}
CSA_LAYER_TYPE_BY_RATIO = {v: k for k, v in CSA_RATIO_BY_LAYER_TYPE.items()}

# Shensi fields that have no Megatron-Core counterpart; carried on the transformer config for tooling.
SHENSI_ONLY_FIELDS: dict[str, str] = {
    "hc_active_streams": "number of streams refreshed from the current token by the mHC module",
    "hc_fixed_streams": "number of streams refreshed from a fixed slot (the rest follow top-k routing)",
    "hc_conv_kernels": "kernel sizes of the MLP-side mHC depth convolutions",
    "routed_expert_hidden_size": "low-rank bottleneck of the routed experts (mapped to moe_latent_size)",
    "attn_res_block_size": "attention-residual block length (block write / block read layers)",
    "erc_loss_alpha": "ERC anchor coefficient",
    "erc_loss_coef": "ERC loss weight",
    "hc_fp32_keep": "keep the HF `_keep_in_fp32_modules_strict` hits in fp32 while training in bf16",
}

# Known gaps between the Shensi config surface and Megatron-Core; surfaced for tooling.
SHENSI_PENDING_FIELDS: dict[str, str] = {
    "attn_res_pp_gt_1_extra_tensor_channel": (
        "Attention-residual state is shipped across pipeline stages on a dedicated group instead of "
        "occupying a second Megatron-Core pipeline tensor."
    )
}


def _to_int_tuple(value) -> tuple[int, ...]:
    """Accept ``"4 8 12"``, ``"4,8,12"`` or any iterable of ints."""
    if isinstance(value, str):
        return tuple(int(v) for v in value.replace(",", " ").split() if v)
    return tuple(int(v) for v in value)


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
    enable_hyper_connections: bool = True
    num_residual_streams: int = 16
    hc_active_streams: int = 4
    hc_fixed_streams: int = 2
    hc_conv_kernels: tuple[int, ...] = (4, 8, 12)
    routed_expert_hidden_size: int = 640
    attn_res_block_size: int = 4
    erc_loss_alpha: float = 0.5
    erc_loss_coef: float = 1.0
    csa_compress_rotary_base: float = 160000.0
    hc_fp32_keep: bool = True
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


ShensiTransformerConfig.attn_res_read_heads = 8


@dataclass
class ShensiConfig:
    """HF-side configuration of a Shensi checkpoint.

    The defaults describe the reference geometry used by the toy/CI checkpoints; released
    checkpoints always carry every field explicitly in ``config.json``.
    """

    vocab_size: int = 128
    hidden_size: int = 128
    moe_intermediate_size: int = 32
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    num_key_value_heads: int = 1
    head_dim: int = 64
    q_lora_rank: int = 64
    default_partial_rotary_factor: float = 8.0 / 64.0
    num_experts_per_tok: int = 2
    n_routed_experts: int = 8
    scoring_func: str = "sqrtsoftplus"
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.5
    max_position_embeddings: int = 128
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
    hc_mult: int = 16
    hc_active_streams: int = 4
    hc_fixed_streams: int = 2
    hc_conv_kernels: Tuple[int, ...] = (4, 8, 12)
    mlp_layer_types: Optional[List[str]] = None
    default_num_hash_layers: int = 3
    swiglu_limit: float = 10.0
    routed_expert_hidden_size: int = 32
    router_aux_loss_coef: float = 0.001
    output_router_logits: bool = False
    sliding_window: int = 64
    o_groups: int = 4
    o_lora_rank: int = 32
    index_n_heads: int = 8
    index_head_dim: int = 16
    index_topk: int = 32
    num_nextn_predict_layers: int = 0
    attn_res_block_size: int = 4
    attn_res_read_heads: int = 8
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

    def describe(self) -> str:
        return (
            "ShensiConfig("
            f"L={self.num_hidden_layers}, H={self.hidden_size}, heads={self.num_attention_heads}, "
            f"head_dim={self.head_dim}, rope_dim={self.qk_rope_head_dim}, "
            f"q_lora={self.q_lora_rank}, o_groups={self.o_groups}, o_lora={self.o_lora_rank}, "
            f"hc_mult={self.hc_mult}(active={self.hc_active_streams},fixed={self.hc_fixed_streams}), "
            f"experts={self.n_routed_experts}x{self.num_experts_per_tok}@{self.moe_intermediate_size} "
            f"rank={self.routed_expert_hidden_size}, "
            f"layer_types={self.layer_types}, mlp_layer_types={self.mlp_layer_types}, "
            f"attn_res={self.attn_res_block_layer_types}, sw={self.sliding_window}, "
            f"erc_coef={self.erc_loss_coef})"
        )


def build_shensi_config(args) -> ShensiConfig:
    """Build a Shensi config from training arguments or checkpoint metadata."""
    kwargs = dict(
        vocab_size=int(getattr(args, "padded_vocab_size", None) or args.vocab_size),
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_attention_heads,
        max_position_embeddings=args.max_position_embeddings,
    )

    def _pick(name: str, default: int) -> int:
        val = getattr(args, name, None)
        return int(default if val in (None, 0) else val)

    hidden = kwargs["hidden_size"]
    kwargs.update(
        head_dim=_pick("shensi_head_dim", hidden // 2),
        q_lora_rank=_pick("shensi_q_lora_rank", hidden // 2),
        o_lora_rank=_pick("shensi_o_lora_rank", hidden // 4),
        o_groups=_pick("shensi_o_groups", 4),
        moe_intermediate_size=_pick("shensi_moe_intermediate_size", hidden // 4),
        routed_expert_hidden_size=_pick("shensi_routed_expert_hidden_size", hidden // 4),
        index_n_heads=_pick("shensi_index_n_heads", 8),
        index_head_dim=_pick("shensi_index_head_dim", 16),
        default_partial_rotary_factor=float(
            getattr(args, "shensi_partial_rotary_factor", None)
            or ShensiConfig.__dataclass_fields__["default_partial_rotary_factor"].default
        ),
        sliding_window=int(min(args.seq_length, _pick("shensi_sliding_window", 64))),
    )
    layers = str(getattr(args, "shensi_attn_layer_types", "") or "")
    ratios = getattr(args, "csa_compress_ratios", None)
    if layers and ratios:
        raise ValueError(
            "--csa-compress-ratios (numeric form, includes MTP layers) and --shensi-attn-layer-types "
            "(type-name form, decoder layers only) are mutually exclusive"
        )
    if layers:
        target = [t for t in layers.split(",") if t]
        assert len(target) == kwargs["num_hidden_layers"], (
            f"--shensi-attn-layer-types needs {kwargs['num_hidden_layers']} entries, got {len(target)}"
        )
        kwargs["layer_types"] = target
    elif ratios:
        kwargs["compress_ratios"] = [int(r) for r in ratios]
    kwargs["num_nextn_predict_layers"] = int(getattr(args, "mtp_num_layers", 0) or 0)
    factor = float(getattr(args, "rotary_scaling_factor", 1.0) or 1.0)
    if factor != 1.0:
        kwargs["rope_parameters"] = {
            "rope_type": "yarn",
            "factor": factor,
            "original_max_position_embeddings": int(
                getattr(args, "original_max_position_embeddings", 0) or kwargs["max_position_embeddings"]
            ),
            "beta_fast": float(getattr(args, "beta_fast", 32.0) or 32.0),
            "beta_slow": float(getattr(args, "beta_slow", 1.0) or 1.0),
        }
    mlp = [m for m in (getattr(args, "shensi_mlp_layer_types", "") or "").split(",") if m]
    if mlp:
        kwargs["mlp_layer_types"] = mlp
    kwargs["hc_mult"] = int(getattr(args, "shensi_hc_mult", 16))
    kwargs["hc_active_streams"] = int(getattr(args, "shensi_hc_active_streams", 4))
    kwargs["hc_fixed_streams"] = int(getattr(args, "shensi_hc_fixed_streams", 2))
    kwargs["index_topk"] = int(getattr(args, "shensi_index_topk", 32))
    kwargs["n_routed_experts"] = int(getattr(args, "shensi_num_experts", 8))
    kwargs["num_experts_per_tok"] = int(getattr(args, "shensi_num_experts_per_tok", 2))
    kwargs["attn_res_block_size"] = int(getattr(args, "shensi_attn_res_block_size", 4))
    kwargs["attn_res_read_heads"] = int(getattr(args, "attn_res_read_heads", 8) or 8)
    kwargs["erc_loss_coef"] = float(getattr(args, "shensi_erc_loss_coef", 1.0))
    kwargs["erc_loss_alpha"] = float(getattr(args, "shensi_erc_loss_alpha", 0.5))
    explicit_aux = getattr(args, "shensi_router_aux_loss_coef", None)
    if explicit_aux is not None:
        kwargs["router_aux_loss_coef"] = float(explicit_aux)
    elif float(getattr(args, "moe_aux_loss_coeff", 0.0) or 0.0):
        kwargs["router_aux_loss_coef"] = float(args.moe_aux_loss_coeff)
    explicit_idx = getattr(args, "shensi_indexer_loss_coeff", None)
    if explicit_idx is not None:
        kwargs["indexer_loss_coeff"] = float(explicit_idx)
    return ShensiConfig(**kwargs)
