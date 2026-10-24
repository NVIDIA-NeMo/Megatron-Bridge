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
attention-residual blocking and ERC coefficients, so the provider carries a
few extra fields next to the MLA ones and derives the Megatron-Core values from a `ShensiConfig`
(``shensi_derived_field_values``).
"""

import logging
from dataclasses import dataclass, field

import torch.nn.functional as F

from megatron.bridge.models.hybrid.hybrid_provider import HybridModelProvider
from megatron.bridge.models.shensi.configuration_shensi import (
    ShensiConfig,
    ShensiTransformerConfig,
    resolve_csa_compress_ratios,
    resolve_moe_n_hash_layers,
)
from megatron.bridge.models.shensi.modeling_shensi import ShensiModel
from megatron.bridge.models.shensi.shensi_hybrid import build_shensi_model, shensi_hybrid_layer_pattern
from megatron.bridge.utils import fusions


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
    "apply_dsa_kernel_fusion": False,
}
# `csa_dense_mode` (dense attention early, sparse later) and the loss coefficients
# (indexer KL / aux / ERC) are per-stage training switches, not hard defaults: the
# config fields' own defaults apply unless a training config sets them.


@dataclass
class ShensiModelProvider(HybridModelProvider, ShensiTransformerConfig):
    """Provider that turns the mapped Shensi config into a `ShensiModel`.

    ``HybridModelProvider`` comes first so its `finalize()` (which derives `num_layers` from the layer
    pattern) and the hybrid construction win; the decoder is the bridge-side `ShensiDecoderStack` (see
    ``shensi_hybrid``), because MCore's ``HybridStack`` dispatches on its own layer-config classes and
    cannot build a model-defined layer type.
    """

    # The upstream providers default this from what the machine can actually run (apex/TE fused
    # kernels). Mixing `ShensiTransformerConfig` into the provider would otherwise let Megatron-Core's
    # TransformerConfig default (True) win, and model construction then fails on hosts without the
    # fused kernel. Re-declare the auto-detecting default so the provider path matches the other
    # hybrid providers.
    gradient_accumulation_fusion: bool = field(default_factory=fusions.can_enable_gradient_accumulation_fusion)

    def finalize(self) -> None:
        """Pick the DSA backend this machine can run, then write the layer pattern.

        Megatron-Core defaults the fused DSA backend to `cudnn` and rejects the config when its
        dependencies are missing; shensi stays usable without them (only the indexer loses its fused
        kernels), so an unset backend becomes `none`. The pattern must exist before
        `HybridModelProvider.finalize()` runs, which derives `num_layers` from it.
        """
        # shensi owns its residual path (`ShensiAttentionResidual`) inside the decoder layers.
        # Megatron-Core's `enable_mhc_connections` would wrap every stack layer in
        # `HyperConnectionHybridLayer` and `wide_residual` would replace the per-branch read/write
        # connections; reject both like Megatron-Core rejects an incompatible combination.
        if getattr(self, "enable_mhc_connections", False):
            raise ValueError(
                "shensi implements its residual path inside the decoder layers "
                "(ShensiAttentionResidual); Megatron-Core's `enable_mhc_connections` is a different "
                "design and must stay disabled."
            )
        if getattr(self, "wide_residual", None) is not None:
            raise ValueError(
                "shensi carries its own attention-residual blocks (attn_res_block_size); "
                "Megatron-Core's `wide_residual` cannot be combined with them."
            )
        # A block that spans a stage boundary hands its state over inside the stage payload, whose
        # shape then depends on the microbatch. Megatron-Core exchanges shapes per microbatch only
        # with `variable_seq_lengths` (it enables the flag itself for its sequence-packing path), so
        # the model declares the same requirement whenever the layout crosses; aligned splits leave
        # the flag untouched.
        if not getattr(self, "variable_seq_lengths", False) and self.pipeline_model_parallel_size > 1:
            try:
                from megatron.bridge.models.shensi.modeling_shensi import AttnResStagePlan

                plan = AttnResStagePlan.from_config(self)
                if plan.crossing_blocks() and bool(getattr(self, "attn_res_pp_state_transfer", True)):
                    self.variable_seq_lengths = True
                    logger.info(
                        "attention-residual blocks span a pipeline stage boundary: enabling "
                        "variable_seq_lengths so the schedule exchanges payload shapes per microbatch."
                    )
            except Exception:  # noqa: BLE001 - the model build reports layout problems in full
                pass
        if self.dsa_kernel_backend is None and self.experimental_attention_variant == "dsv4_hybrid":
            from megatron.core.utils import _validate_dsa_kernel_backend_dependencies

            try:
                _validate_dsa_kernel_backend_dependencies("cudnn")
            except Exception as exc:  # noqa: BLE001 - any reason means "not usable on this machine"
                logger.warning("Fused DSA kernels are unavailable (%s); using the PyTorch path.", exc)
                self.dsa_kernel_backend = "none"
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
        # MCore deep-copies the config per layer; runtime-only process groups cannot be deep-copied, so
        # they travel through the dedicated argument and are restored afterwards (the model keeps this
        # provider as its root config).
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
                # `scatter_embedding_sequence_parallel` is a GPTModelProvider field that does not
                # exist on the hybrid providers; `HybridModel` defaults it to True, which is also
                # what the other hybrid families rely on.
                seq_len_interpolation_factor=self.seq_len_interpolation_factor,
            )
        finally:
            self._pg_collection = pg_collection
