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

"""Shensi model provider and the Shensi → Megatron-Core field mapping.

`ShensiModelProvider` is a `TransformerConfig` that can build a `ShensiModel` through
`provide_distributed_model()`. Shensi keeps its own attention plan (CSA / HCA / sliding-window),
mHC stream counts, attention-residual blocking and ERC coefficients, so the provider carries a
few extra fields next to the MLA ones and derives the Megatron-Core values from a `ShensiConfig`
(``shensi_derived_field_values``) or from parsed training arguments (``shensi_config_from_args``).
"""

import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.shensi.configuration_shensi import (
    SHENSI_ONLY_FIELDS,
    SHENSI_PENDING_FIELDS,
    ShensiConfig,
    ShensiTransformerConfig,
    _to_int_tuple,
    resolve_csa_compress_ratios,
    resolve_moe_n_hash_layers,
)
from megatron.bridge.models.shensi.modeling_shensi import (
    ShensiModel,
    get_shensi_decoder_block_spec,
    get_shensi_mtp_block_spec,
)


logger = logging.getLogger(__name__)


def shensi_derived_field_values(shensi_cfg: ShensiConfig, *, mtp_num_layers: int = 0) -> dict[str, object]:
    """Megatron-Core field values derived from a Shensi config."""
    num_layers = int(shensi_cfg.num_hidden_layers)
    layer_types = list(shensi_cfg.layer_types)
    mlp_layer_types = list(shensi_cfg.mlp_layer_types)
    if len(layer_types) != num_layers or len(mlp_layer_types) != num_layers:
        raise ValueError("layer_types / mlp_layer_types must have num_hidden_layers entries")
    explicit = getattr(shensi_cfg, "compress_ratios", None)
    if explicit:
        ratios = [int(r) for r in explicit]
        want = num_layers + int(mtp_num_layers)
        if len(ratios) != want:
            raise ValueError(f"compress_ratios needs {want} entries (decoder layers + MTP layers), got {len(ratios)}")
    else:
        ratios = resolve_csa_compress_ratios(layer_types)
        if mtp_num_layers:
            ratios = ratios + [128] * mtp_num_layers
    n_hash = resolve_moe_n_hash_layers(mlp_layer_types)
    hidden = int(shensi_cfg.hidden_size)
    head_dim = int(shensi_cfg.head_dim)
    rope_dim = int(getattr(shensi_cfg, "qk_rope_head_dim", 0)) or int(
        head_dim * float(shensi_cfg.partial_rotary_factor)
    )
    return {
        "num_layers": num_layers,
        "hidden_size": hidden,
        "num_attention_heads": int(shensi_cfg.num_attention_heads),
        "v_head_dim": head_dim,
        "qk_pos_emb_head_dim": rope_dim,
        "q_lora_rank": int(shensi_cfg.q_lora_rank),
        "o_groups": int(shensi_cfg.o_groups),
        "o_lora_rank": int(shensi_cfg.o_lora_rank),
        "output_projection_groups": int(shensi_cfg.o_groups),
        "output_projection_lora_rank": int(shensi_cfg.o_lora_rank),
        "csa_compress_ratios": ratios,
        "mtp_num_layers": int(mtp_num_layers) or None,
        "csa_window_size": int(shensi_cfg.sliding_window),
        "csa_compress_rotary_base": float(shensi_cfg.compress_rope_theta),
        "dsa_indexer_n_heads": int(shensi_cfg.index_n_heads),
        "dsa_indexer_head_dim": int(shensi_cfg.index_head_dim),
        "dsa_indexer_topk": int(shensi_cfg.index_topk),
        "dsa_indexer_loss_coeff": float(getattr(shensi_cfg, "indexer_loss_coeff", 0.01) or 0.0),
        "num_residual_streams": int(shensi_cfg.hc_mult),
        "hc_active_streams": int(shensi_cfg.hc_active_streams),
        "hc_fixed_streams": int(shensi_cfg.hc_fixed_streams),
        "hc_conv_kernels": _to_int_tuple(shensi_cfg.hc_conv_kernels),
        "attn_res_block_size": int(shensi_cfg.attn_res_block_size),
        "num_moe_experts": int(shensi_cfg.n_routed_experts),
        "moe_ffn_hidden_size": int(shensi_cfg.moe_intermediate_size),
        "moe_router_topk": int(shensi_cfg.num_experts_per_tok),
        "moe_router_score_function": str(shensi_cfg.scoring_func),
        "moe_router_topk_scaling_factor": float(shensi_cfg.routed_scaling_factor),
        "moe_n_hash_layers": n_hash,
        "moe_aux_loss_coeff": float(getattr(shensi_cfg, "router_aux_loss_coef", 0.0) or 0.0),
        "moe_latent_size": int(shensi_cfg.routed_expert_hidden_size),
        "routed_expert_hidden_size": int(shensi_cfg.routed_expert_hidden_size),
        "actual_vocab_size": int(shensi_cfg.vocab_size),
        "activation_func": F.silu,
        "activation_func_clamp_value": float(shensi_cfg.swiglu_limit),
        "use_te_activation_func": False,
        "erc_loss_alpha": float(shensi_cfg.erc_loss_alpha),
        "erc_loss_coef": float(shensi_cfg.erc_loss_coef),
        "rotary_base": float(shensi_cfg.rope_theta),
        "rotary_scaling_factor": float(getattr(shensi_cfg, "compress_rope_scaling_factor", 1.0)),
        "original_max_position_embeddings": int(
            getattr(
                shensi_cfg,
                "compress_rope_original_max_position_embeddings",
                shensi_cfg.max_position_embeddings,
            )
        ),
        "beta_fast": float(getattr(shensi_cfg, "compress_rope_beta_fast", 32.0)),
        "beta_slow": float(getattr(shensi_cfg, "compress_rope_beta_slow", 1.0)),
        "mscale": float(getattr(shensi_cfg, "compress_rope_mscale", 1.0)),
        "mscale_all_dim": float(getattr(shensi_cfg, "compress_rope_mscale_all_dim", 0.0)),
        "layernorm_epsilon": float(shensi_cfg.rms_norm_eps),
        "init_method_std": float(shensi_cfg.initializer_range),
        "attention_dropout": float(shensi_cfg.attention_dropout),
        "hidden_dropout": 0.0,
    }


SHENSI_HARD_DEFAULTS: dict[str, object] = {
    "experimental_attention_variant": "dsv4_hybrid",
    "multi_latent_attention": True,
    "enable_hyper_connections": True,
    "use_fused_mhc": False,
    "normalization": "RMSNorm",
    "add_bias_linear": False,
    "gated_linear_unit": True,
    "qk_layernorm": True,
    "apply_rope_fusion": True,
    "rotary_interleaved": False,
    "moe_grouped_gemm": True,
    "moe_router_pre_softmax": False,
    "moe_router_enable_expert_bias": False,
    "csa_dense_mode": False,
    "apply_dsa_kernel_fusion": False,
}


def build_shensi_transformer_config(
    shensi_cfg: ShensiConfig,
    *,
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
    seq_length: int | None = None,
    fp16: bool = False,
    bf16: bool = False,
    rope_type: str = "rope",
    mtp_num_layers: int = 0,
    **extra,
) -> ShensiTransformerConfig:
    """Shensi transformer config for a training run."""
    if pipeline_model_parallel_size > 1 and "pipeline_dtype" not in extra:
        extra["pipeline_dtype"] = torch.float16 if fp16 else (torch.bfloat16 if bf16 else torch.float32)
    tcfg = ShensiTransformerConfig(
        **shensi_derived_field_values(shensi_cfg, mtp_num_layers=mtp_num_layers),
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        expert_model_parallel_size=expert_model_parallel_size,
        fp16=fp16,
        bf16=bf16,
        rope_type=rope_type,
        **extra,
    )
    tcfg.shensi_only_fields = {
        name: (getattr(shensi_cfg, name, None), reason)
        for name, reason in SHENSI_ONLY_FIELDS.items()
        if hasattr(shensi_cfg, name)
    }
    tcfg.shensi_pending_fields = dict(SHENSI_PENDING_FIELDS)
    tcfg.max_sequence_length = int(seq_length or shensi_cfg.max_position_embeddings)
    return tcfg


def apply_shensi_overrides_from_args(config: ShensiTransformerConfig, args) -> None:
    """Copy the ``--shensi-*`` CLI arguments into the transformer config."""
    layer_types = getattr(args, "shensi_attn_layer_types", None)
    if layer_types:
        types = [t for t in str(layer_types).split(",") if t]
        if len(types) != config.num_layers:
            raise ValueError(f"--shensi-attn-layer-types needs {config.num_layers} entries, got {len(types)}")
        config.csa_compress_ratios = [
            *resolve_csa_compress_ratios(types),
            *([128] * int(getattr(config, "mtp_num_layers", 0) or 0)),
        ]
    mlp_types = getattr(args, "shensi_mlp_layer_types", None)
    if mlp_types:
        types = [t for t in str(mlp_types).split(",") if t]
        config.moe_n_hash_layers = resolve_moe_n_hash_layers(types)
    for attr, arg_name in (
        ("num_residual_streams", "shensi_hc_mult"),
        ("csa_window_size", "shensi_sliding_window"),
        ("o_groups", "shensi_o_groups"),
        ("hc_active_streams", "shensi_hc_active_streams"),
        ("hc_fixed_streams", "shensi_hc_fixed_streams"),
        ("routed_expert_hidden_size", "shensi_routed_expert_hidden_size"),
        ("attn_res_block_size", "shensi_attn_res_block_size"),
    ):
        val = getattr(args, arg_name, None)
        if val is not None:
            setattr(config, attr, int(val))
    conv_kernels = getattr(args, "shensi_hc_conv_kernels", None)
    if conv_kernels:
        config.hc_conv_kernels = _to_int_tuple(conv_kernels)
    for attr, arg_name in (
        ("erc_loss_alpha", "shensi_erc_loss_alpha"),
        ("erc_loss_coef", "shensi_erc_loss_coef"),
    ):
        val = getattr(args, arg_name, None)
        if val is not None:
            setattr(config, attr, float(val))
    keep = getattr(args, "shensi_hc_fp32_keep", None)
    if keep is not None:
        config.hc_fp32_keep = bool(keep)
    transfer = getattr(args, "shensi_attn_res_pp_state_transfer", None)
    if transfer is not None:
        config.attn_res_pp_state_transfer = bool(transfer)
    read_heads = getattr(args, "attn_res_read_heads", None)
    if read_heads is not None:
        config.attn_res_read_heads = int(read_heads)


@dataclass
class ShensiConfigFromArgs(ShensiTransformerConfig):
    """ShensiTransformerConfig that accepts being filled from training arguments."""

    def __post_init__(self):
        self.multi_latent_attention = True
        super().__post_init__()


def shensi_config_from_args(args) -> ShensiTransformerConfig:
    """Build a ShensiTransformerConfig from parsed training arguments."""
    from megatron.training import print_rank_0
    from megatron.training.arguments import core_transformer_config_from_args

    saved = args.multi_latent_attention
    args.multi_latent_attention = False
    try:
        config = core_transformer_config_from_args(args, ShensiConfigFromArgs)
    finally:
        args.multi_latent_attention = saved
    print_rank_0(
        f"[shensi] config class = {type(config).__name__} (csa_dense_mode="
        f"{getattr(config, 'csa_dense_mode', None)} moe_latent_size="
        f"{getattr(config, 'moe_latent_size', None)} moe_n_hash_layers="
        f"{getattr(config, 'moe_n_hash_layers', None)} rotary_scaling_factor="
        f"{getattr(config, 'rotary_scaling_factor', None)} csa_compress_ratios="
        f"{getattr(config, 'csa_compress_ratios', None)} moe_aux_loss_coeff="
        f"{getattr(config, 'moe_aux_loss_coeff', None)} erc_loss_coef="
        f"{getattr(config, 'erc_loss_coef', None)} dsa_indexer_loss_coeff="
        f"{getattr(config, 'dsa_indexer_loss_coeff', None)})"
    )
    return config


def inject_shensi_fields_into_args(args) -> None:
    """Publish the Shensi fields as plain Megatron-Core arguments for training scripts."""
    from megatron.bridge.models.shensi.configuration_shensi import build_shensi_config

    shensi_cfg = build_shensi_config(args)
    mtp = int(getattr(args, "mtp_num_layers", 0) or 0)
    values = shensi_derived_field_values(shensi_cfg, mtp_num_layers=mtp)
    for key, value in values.items():
        setattr(args, key, value)
    for key, value in SHENSI_HARD_DEFAULTS.items():
        setattr(args, key, value)

    args.num_experts = values["num_moe_experts"]
    args.moe_latent_size = values["moe_latent_size"]
    args.num_query_groups = int(shensi_cfg.num_attention_heads)
    print(
        "[shensi] derived Megatron-Core fields written to args: "
        f"{shensi_cfg.describe()} | csa_compress_ratios={values['csa_compress_ratios']} "
        f"moe_latent_size={values['moe_latent_size']} moe_n_hash_layers={values['moe_n_hash_layers']} "
        f"rotary_scaling_factor={values['rotary_scaling_factor']} "
        f"moe_aux_loss_coeff={values['moe_aux_loss_coeff']}"
    )


KEY_FIELDS = (
    "num_layers",
    "hidden_size",
    "num_attention_heads",
    "v_head_dim",
    "qk_pos_emb_head_dim",
    "qk_head_dim",
    "kv_lora_rank",
    "q_lora_rank",
    "o_groups",
    "o_lora_rank",
    "experimental_attention_variant",
    "csa_compress_ratios",
    "csa_window_size",
    "csa_compress_rotary_base",
    "csa_dense_mode",
    "rotary_base",
    "rotary_scaling_factor",
    "original_max_position_embeddings",
    "mscale",
    "mscale_all_dim",
    "dsa_indexer_n_heads",
    "dsa_indexer_head_dim",
    "dsa_indexer_topk",
    "dsa_indexer_loss_coeff",
    "enable_hyper_connections",
    "num_residual_streams",
    "hc_active_streams",
    "hc_fixed_streams",
    "hc_conv_kernels",
    "attn_res_block_size",
    "erc_loss_alpha",
    "erc_loss_coef",
    "num_moe_experts",
    "moe_ffn_hidden_size",
    "moe_latent_size",
    "moe_router_topk",
    "moe_router_score_function",
    "moe_router_topk_scaling_factor",
    "moe_layer_freq",
    "moe_n_hash_layers",
    "moe_aux_loss_coeff",
    "actual_vocab_size",
    "activation_func_clamp_value",
)


def describe_mapping(tcfg: ShensiTransformerConfig) -> dict[str, object]:
    """Human readable HF → Megatron field mapping (used by tests and logs)."""
    out = {}
    for k in KEY_FIELDS:
        v = getattr(tcfg, k, "<missing>")
        if callable(v):
            v = getattr(v, "__name__", str(v))
        out[k] = v
    return out


@dataclass
class ShensiModelProvider(ShensiTransformerConfig, GPTModelProvider):
    """Provider that turns the mapped Shensi config into a `ShensiModel`."""

    def finalize(self) -> None:
        """Select the PyTorch DSA path when the fused kernels are not installed.

        Megatron-Core defaults the fused backend to `cudnn` for the hybrid attention and rejects
        the config when its dependencies are missing. Shensi stays usable without them (only the
        indexer loses its fused kernels), so the unset value becomes `none` in that case.
        """
        if self.dsa_kernel_backend is None and self.experimental_attention_variant == "dsv4_hybrid":
            from megatron.core.utils import _validate_dsa_kernel_backend_dependencies

            try:
                _validate_dsa_kernel_backend_dependencies("cudnn")
            except Exception as exc:  # noqa: BLE001 - any reason means "not usable on this machine"
                logger.warning("Fused DSA kernels are unavailable (%s); using the PyTorch path.", exc)
                self.dsa_kernel_backend = "none"
        super().finalize()

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> ShensiModel:
        from megatron.core.pipeline_parallel.utils import (
            is_pp_first_stage,
            is_pp_last_stage,
            is_vp_first_stage,
            is_vp_last_stage,
        )
        from megatron.core.utils import init_method_normal, scaled_init_method_normal
        from megatron.training.vocab_utils import calculate_padded_vocab_size

        if self.init_method is None:
            self.init_method = init_method_normal(self.init_method_std)
        if self.output_layer_init_method is None:
            self.output_layer_init_method = scaled_init_method_normal(self.init_method_std, self.num_layers)
        if self.embedding_init_method is None:
            self.embedding_init_method = init_method_normal(self.embedding_init_method_std or self.init_method_std)
        transformer_layer_spec = get_shensi_decoder_block_spec(
            config=self, use_transformer_engine=True, vp_stage=vp_stage
        )
        mtp_block_spec = None
        if self.mtp_num_layers:
            mtp_block_spec = get_shensi_mtp_block_spec(self, use_transformer_engine=True, vp_stage=vp_stage)
        vp_size = self.virtual_pipeline_model_parallel_size
        pg_collection = self._pg_collection
        if pre_process is None:
            pre_process = is_vp_first_stage(vp_stage=vp_stage, vp_size=vp_size) and is_pp_first_stage(pg_collection.pp)
        if post_process is None:
            post_process = is_vp_last_stage(vp_stage=vp_stage, vp_size=vp_size) and is_pp_last_stage(pg_collection.pp)
        self._vp_stage = vp_stage
        padded_vocab_size = self.vocab_size
        if self.should_pad_vocab:
            padded_vocab_size = calculate_padded_vocab_size(
                self.vocab_size, self.make_vocab_size_divisible_by, self.tensor_model_parallel_size
            )
        return ShensiModel(
            self,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=padded_vocab_size,
            max_sequence_length=self.seq_length,
            fp16_lm_cross_entropy=self.fp16_lm_cross_entropy,
            parallel_output=self.parallel_output,
            share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
            position_embedding_type=self.position_embedding_type,
            rotary_percent=self.rotary_percent,
            rotary_base=self.rotary_base,
            rope_scaling=self.rope_scaling,
            rope_scaling_factor=self.rope_scaling_factor,
            seq_len_interpolation_factor=self.seq_len_interpolation_factor,
            pre_process=pre_process,
            post_process=post_process,
            scatter_embedding_sequence_parallel=self.scatter_embedding_sequence_parallel,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
            mtp_block_spec=mtp_block_spec,
        )
