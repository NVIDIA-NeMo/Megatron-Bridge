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

"""Shensi provider and HuggingFace-to-Megatron field mapping."""

import logging
from dataclasses import dataclass, field

from megatron.bridge.models.hybrid.hybrid_provider import HybridModelProvider
from megatron.bridge.models.shensi.configuration_shensi import (
    ShensiConfig,
    ShensiTransformerConfig,
    resolve_attn_res_block_layer_types,
    resolve_indexer_types,
    resolve_layer_types,
    resolve_mlp_layer_types,
    resolve_rosa_layer_ids,
)
from megatron.bridge.models.shensi.modeling_shensi import ShensiModel
from megatron.bridge.models.shensi.shensi_hybrid import build_shensi_model, shensi_hybrid_layer_pattern
from megatron.bridge.utils.fusions import can_enable_gradient_accumulation_fusion


logger = logging.getLogger(__name__)


def _get(config, name: str, default=None):
    value = getattr(config, name, default)
    return default if value is None else value


def shensi_derived_field_values(shensi_cfg, *, mtp_num_layers: int = 0) -> dict[str, object]:
    """Megatron-Core field values derived from a Shensi HF config."""
    num_layers = int(shensi_cfg.num_hidden_layers)
    first_k_dense = int(_get(shensi_cfg, "first_k_dense_replace", 3))
    mlp_layer_types = resolve_mlp_layer_types(_get(shensi_cfg, "mlp_layer_types", None), first_k_dense, num_layers)
    attn_res_types = _get(shensi_cfg, "attn_res_block_layer_types", None) or resolve_attn_res_block_layer_types(
        mlp_layer_types, int(_get(shensi_cfg, "attn_res_block_size", 2)), num_layers
    )
    layer_types = _get(shensi_cfg, "layer_types", None) or resolve_layer_types(attn_res_types)
    indexer_types = resolve_indexer_types(layer_types, _get(shensi_cfg, "indexer_types", None))
    rosa_ids = resolve_rosa_layer_ids(mlp_layer_types, _get(shensi_cfg, "rosa_layer_ids", None))

    hidden = int(shensi_cfg.hidden_size)
    head_dim = int(shensi_cfg.head_dim)
    local_experts = int(shensi_cfg.n_routed_experts)
    linear_heads = int(_get(shensi_cfg, "linear_num_heads", 16))
    linear_head_dim = int(_get(shensi_cfg, "linear_head_dim", 128))

    return {
        "num_layers": num_layers,
        "hidden_size": hidden,
        "num_attention_heads": int(shensi_cfg.num_attention_heads),
        "qk_head_dim": head_dim,
        "qk_pos_emb_head_dim": int(_get(shensi_cfg, "qk_rope_head_dim", 0) or 0),
        "v_head_dim": int(_get(shensi_cfg, "v_head_dim", head_dim) or head_dim),
        "q_lora_rank": int(shensi_cfg.q_lora_rank),
        "kv_lora_rank": int(shensi_cfg.deepemb_kv_rank),
        "index_n_heads": int(shensi_cfg.index_n_heads),
        "index_head_dim": int(shensi_cfg.index_head_dim),
        "index_topk_tokens": int(shensi_cfg.index_topk_tokens),
        "index_block_size": int(shensi_cfg.index_block_size),
        "index_block_topk": int(shensi_cfg.index_block_topk),
        "index_k_size": int(shensi_cfg.index_k_size),
        "indexer_loss_coeff": float(_get(shensi_cfg, "indexer_loss_coeff", 0.01) or 0.0),
        "linear_num_key_heads": linear_heads,
        "linear_num_value_heads": linear_heads,
        "linear_key_head_dim": linear_head_dim,
        "linear_value_head_dim": linear_head_dim,
        "linear_conv_kernel_dim": int(shensi_cfg.linear_conv_kernel_dim),
        "kda_two_stage_gates": True,
        "kda_safe_gate": True,
        "kda_lower_bound": float(_get(shensi_cfg, "linear_lower_bound", -5.0)),
        "kda_disable_fp8": True,
        "layer_types": layer_types,
        "mlp_layer_types": mlp_layer_types,
        "indexer_types": indexer_types,
        "attn_res_block_layer_types": attn_res_types,
        "first_k_dense_replace": first_k_dense,
        "norm_topk_prob": bool(_get(shensi_cfg, "norm_topk_prob", True)),
        "num_moe_experts": local_experts,
        "moe_ffn_hidden_size": int(shensi_cfg.moe_intermediate_size),
        "moe_router_topk": int(shensi_cfg.num_experts_per_tok),
        "moe_router_score_function": str(shensi_cfg.scoring_func),
        "moe_router_topk_scaling_factor": float(shensi_cfg.routed_scaling_factor),
        "moe_latent_size": int(shensi_cfg.latent_size),
        "moe_use_norm_before_up_proj": True,
        "moe_aux_loss_coeff": float(_get(shensi_cfg, "router_aux_loss_coef", 0.0) or 0.0),
        # Training-side deviation: the reference only *returns* router logits when
        # `output_router_logits` is set (inference-oriented), but a routed MoE needs the
        # load-balancing gradient to keep every expert trainable. Enable it whenever the
        # reference loss weight is non-zero; `router_aux_loss_coef=0` turns it off again.
        "moe_router_load_balancing_type": (
            "aux_loss"
            if bool(_get(shensi_cfg, "output_router_logits", False))
            or float(_get(shensi_cfg, "router_aux_loss_coef", 0.0) or 0.0) > 0.0
            else "none"
        ),
        "latent_size": int(shensi_cfg.latent_size),
        "hc_mult": int(shensi_cfg.hc_mult),
        "hc_active_streams": int(_get(shensi_cfg, "hc_active_streams", 4)),
        "hc_adapter_rank": int(_get(shensi_cfg, "hc_adapter_rank", 8)),
        "hc_fixed_streams": int(shensi_cfg.hc_fixed_streams),
        "hc_conv_kernels": list(shensi_cfg.hc_conv_kernels),
        "hc_stream_rank": int(shensi_cfg.hc_stream_rank),
        "hc_router_aux_loss_coef": float(_get(shensi_cfg, "hc_router_aux_loss_coef", 0.01) or 0.0),
        "hc_eps": float(shensi_cfg.hc_eps),
        "attn_res_rank": int(_get(shensi_cfg, "attn_res_rank", 32)),
        "attn_res_block_size": int(shensi_cfg.attn_res_block_size),
        "attn_res_block_summary_rank": int(shensi_cfg.attn_res_block_summary_rank),
        "deepemb_rank": int(shensi_cfg.deepemb_rank),
        "rosa_embed_dim": int(_get(shensi_cfg, "rosa_embed_dim", hidden) or hidden),
        "rosa_layer_ids": rosa_ids,
        "rosa_reset_token_id": _get(shensi_cfg, "rosa_reset_token_id", None),
        "rosa_loss_coef": float(_get(shensi_cfg, "rosa_loss_coef", 0.05) or 0.0),
        "erc_loss_alpha": float(_get(shensi_cfg, "erc_loss_alpha", 0.5)),
        "erc_loss_coef": float(_get(shensi_cfg, "erc_loss_coef", 1.0)),
        "mtp_num_layers": int(mtp_num_layers) or None,
        "actual_vocab_size": int(shensi_cfg.vocab_size),
        "activation_func_clamp_value": float(_get(shensi_cfg, "swiglu_limit", 10.0)),
        "layernorm_epsilon": float(shensi_cfg.rms_norm_eps),
        "init_method_std": float(shensi_cfg.initializer_range),
        "attention_dropout": float(shensi_cfg.attention_dropout),
        "hidden_dropout": 0.0,
    }


SHENSI_HARD_DEFAULTS: dict[str, object] = {
    "experimental_attention_variant": "dsa",
    "multi_latent_attention": True,
    "normalization": "RMSNorm",
    "add_bias_linear": False,
    "gated_linear_unit": True,
    "qk_layernorm": True,
    "apply_rope_fusion": False,
    "rotary_interleaved": False,
    "position_embedding_type": "none",
    "sequence_parallel": False,
    "moe_grouped_gemm": True,
    "moe_router_pre_softmax": False,
    "moe_router_enable_expert_bias": False,
    "moe_router_dtype": "fp32",
    "moe_token_dispatcher_type": "allgather",
}


SHENSI_HARD_DEFAULTS["transformer_impl"] = "transformer_engine"


@dataclass
class ShensiModelProvider(HybridModelProvider, ShensiTransformerConfig):
    """Provider that turns the mapped Shensi config into a `ShensiModel`."""

    gradient_accumulation_fusion: bool = field(default_factory=can_enable_gradient_accumulation_fusion)

    def finalize(self) -> None:
        """Validate the residual design and write the layer pattern before building layers."""
        if str(self.transformer_impl) != "transformer_engine":
            raise ImportError(
                "Shensi needs Transformer Engine: the HISA indexer's replicated projections use "
                '`TELinear(parallel_mode="duplicated")` and the sequence-parallel norms need '
                "`TENorm`; the Megatron-Core local backend provides neither."
            )
        if getattr(self, "enable_mhc_connections", False):
            raise ValueError(
                "shensi implements its residual path inside the decoder layers "
                "(ShensiHyperConnection / ShensiAttentionResidual); Megatron-Core's "
                "`enable_mhc_connections` is a different design and must stay disabled."
            )
        if getattr(self, "wide_residual", None) is not None:
            raise ValueError(
                "shensi carries its own attention-residual blocks (attn_res_block_size); "
                "Megatron-Core's `wide_residual` cannot be combined with them."
            )
        if self.hc_conv_kernels is None:
            self.hc_conv_kernels = [4 * (i + 1) for i in range(int(self.hc_mult) - 1)]
        elif int(self.hc_mult) != 1 + len(self.hc_conv_kernels):
            raise ValueError(
                f"hc_mult ({self.hc_mult}) must equal 1 + len(hc_conv_kernels) "
                f"({len(self.hc_conv_kernels)}): the depthwise convolutions define the auxiliary streams."
            )
        if not 0 <= int(self.hc_fixed_streams) <= int(self.hc_active_streams) <= int(self.hc_mult):
            raise ValueError(
                "Expected 0 <= hc_fixed_streams <= hc_active_streams <= hc_mult, got "
                f"{self.hc_fixed_streams}, {self.hc_active_streams}, {self.hc_mult}."
            )
        if self.context_parallel_size > 1:
            raise NotImplementedError(
                "Shensi does not support context parallelism yet: its attention-residual bank "
                "and HISA indexer operate on the full sequence."
            )
        tp_size = int(self.tensor_model_parallel_size)
        if tp_size > 1:
            self.sequence_parallel = True
            if int(self.num_attention_heads) % tp_size:
                raise ValueError(
                    f"Shensi indexed attention needs num_attention_heads ({self.num_attention_heads}) "
                    f"divisible by tensor_model_parallel_size ({tp_size})."
                )
            for name in ("linear_num_key_heads", "linear_num_value_heads"):
                linear_heads = int(getattr(self, name, 0))
                if linear_heads % tp_size:
                    raise ValueError(
                        f"Shensi KDA needs {name} ({linear_heads}) divisible by "
                        f"tensor_model_parallel_size ({tp_size})."
                    )
        if self.pipeline_model_parallel_size > 1:
            self.variable_seq_lengths = True
            if self.moe_token_dispatcher_type == "allgather":
                self.moe_token_dispatcher_type = "alltoall"
        if self.hybrid_layer_pattern is None:
            shensi_hybrid_layer_pattern(self)
        super().finalize()

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> ShensiModel:
        """Build one `ShensiModel` pipeline stage."""
        from megatron.core.utils import init_method_normal, scaled_init_method_normal
        from megatron.training.vocab_utils import calculate_padded_vocab_size

        if self.init_method is None:
            self.init_method = init_method_normal(self.init_method_std)
        if self.output_layer_init_method is None:
            self.output_layer_init_method = scaled_init_method_normal(self.init_method_std, self.num_layers)
        if self.embedding_init_method is None:
            self.embedding_init_method = init_method_normal(self.embedding_init_method_std or self.init_method_std)
        padded_vocab_size = self.vocab_size
        if self.should_pad_vocab:
            padded_vocab_size = calculate_padded_vocab_size(
                self.vocab_size, self.make_vocab_size_divisible_by, self.tensor_model_parallel_size
            )
        pg_collection = self._pg_collection
        self._pg_collection = None
        try:
            return build_shensi_model(
                self,
                pg_collection,
                vocab_size=padded_vocab_size,
                max_sequence_length=self.seq_length,
                vp_stage=vp_stage,
                pre_process=pre_process,
                post_process=post_process,
                fp16_lm_cross_entropy=self.fp16_lm_cross_entropy,
                logit_dtype=self.logit_dtype,
                parallel_output=self.parallel_output,
                share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
                position_embedding_type=self.position_embedding_type,
                rotary_percent=self.rotary_percent,
                rotary_base=self.rotary_base,
                seq_len_interpolation_factor=self.seq_len_interpolation_factor,
            )
        finally:
            self._pg_collection = pg_collection


__all__ = [
    "SHENSI_HARD_DEFAULTS",
    "ShensiConfig",
    "ShensiModelProvider",
    "shensi_derived_field_values",
]
