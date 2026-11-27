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

"""Bridge between Hugging Face ERNIE 5.5 (``Ernie5_5_MoeForCausalLM``) and Megatron-Core ``HybridModel``.

ERNIE 5.5 has no Hugging Face modeling code; :mod:`~megatron.bridge.models.ernie.configuration_ernie55`
registers its ``config.json`` schema with ``AutoConfig``. The bridge maps that config to an
:class:`~megatron.bridge.models.ernie.ernie55_config.Ernie55ModelConfig`, whose builder constructs
:class:`~megatron.bridge.models.ernie.modeling_ernie55.model.Ernie55HybridModel`:

* layout: every ERNIE layer becomes two hybrid layers, ``H`` (``compress_ratios`` 128) or ``D`` (-2) followed
  by ``-`` (the first ``first_k_dense_replace`` layers) or ``E``; the trailing ``compress_ratios`` entries are the
  MTP depths (``"H-DEHEDE/DE"`` for the 4-layer proxy);
* H layers: the stack geometry (``num_attention_heads``, ``q_lora_rank``, ``v_head_dim``, ``qk_rope_head_dim``,
  ``o_groups`` x ``o_lora_rank``), the ``sliding_window`` window and RoPE at ``compress_rope_theta`` (plain
  RoPE or YaRN, per ``hca_rope_type``);
* D layers: ``hybrid_mla_*`` into the ``ernie_d_*`` fields with RoPE at ``rope_theta``, and the DSA indexer
  (``index_n_heads`` x ``index_head_dim``, top-``index_topk``); the checkpoint's RoPE channel layout
  (``mqa_latent_rope_adjacent_pairing``) is kept for weight conversion and export;
* MoE: the split-feature sigmoid router with the noaux_tc expert bias, the learned per-expert scale (initialized
  to ``routed_scaling_factor``), ``n_shared_experts`` shared experts, optional latent MoE and SiTU-GLU;
* hyper-connections (``hc_mult`` streams, ``hc_sinkhorn_iters`` Sinkhorn iterations), SD-MTP
  (``mtp_shared_last_layer``), the untied output layer and the unpadded vocabulary.

Settings the Hugging Face config does not carry and the architecture fixes: the VHA rank (16 unless the config
sets ``vha_rank``), the fixed indexer settings, ``position_embedding_type="none"`` (every attention owns its
RoPE) and a frozen expert bias (``moe_router_bias_update_rate=0``; training recipes set the update rate).
``seq_length`` is the recipe's, not ``max_position_embeddings``. Generic ``CONFIG_MAPPING`` entries that would
misread ERNIE keys are dropped: ``num_key_value_heads`` (one shared latent head, not GQA groups), ``head_dim``
(not the MCore ``kv_channels``) and ``max_position_embeddings``.

Configs outside the supported envelope fail at mapping time: H-layer RoPE other than plain RoPE or YaRN
(``hca_rope_type`` other than ``"rope"`` or ``"yarn"``), D layers other than latent MQA with DSA (``hybrid_mla_attention`` other than ``"mqa_dsa"``), missing
q/kv norms (``use_qk_norm``), tied embeddings, biases, dropout, a hyper-connection epsilon other than 1e-6,
MTP depths with different layers and MoE keys that the reference implementation would read under another name
with another value (:func:`~megatron.bridge.models.ernie.configuration_ernie55.check_ernie55_reference_key_spellings`).

Weight conversion: :class:`~megatron.bridge.models.ernie.ernie55_mappings.Ernie55WeightConversionMixin` gives
the bridge its mapping registry and checks on import that a checkpoint holds exactly the tensors of its config.
"""

import json
from pathlib import Path
from typing import Any

import torch.nn.functional as F
from transformers import PretrainedConfig

from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.ernie.configuration_ernie55 import (
    ERNIE55_ARCHITECTURE,
    ERNIE55_MODEL_TYPE,
    check_ernie55_reference_key_spellings,
)
from megatron.bridge.models.ernie.ernie55_config import (
    ERNIE55_D_COMPRESS_RATIO,
    ERNIE55_H_COMPRESS_RATIO,
    Ernie55ModelConfig,
    ernie55_csa_compress_ratios,
    ernie55_hybrid_layer_pattern,
    parse_ernie55_hybrid_pattern,
)
from megatron.bridge.models.ernie.ernie55_mappings import Ernie55WeightConversionMixin
from megatron.bridge.models.ernie.modeling_ernie55 import ERNIE55_HCA_ROPE_TYPES, Ernie55HybridModel


__all__ = ["ERNIE55_DEFAULT_VHA_RANK", "ERNIE55_HC_EPS", "Ernie55Bridge"]

ERNIE55_DEFAULT_VHA_RANK = 16
"""VHA rank of ERNIE 5.5 when the Hugging Face config has no ``vha_rank`` key (the real configs have none)."""

ERNIE55_HC_EPS = 1e-6
"""Hyper-connection epsilon of the ERNIE 5.5 arithmetic (Sinkhorn normalizers, mapping norm and gates)."""

# Generic CONFIG_MAPPING results that misread ERNIE 5.5 keys; the ERNIE mapping sets or leaves them itself.
_DROPPED_GENERIC_FIELDS = (
    "num_query_groups",  # num_key_value_heads=1 is the shared latent head, not GQA groups
    "kv_channels",  # head_dim is the H-layer value/latent width, not MCore's attention head width
    "seq_length",  # max_position_embeddings is the model's limit; the recipe sets the training length
    "_mla_rope_params",
)

# transformers 5 saves fast tokenizers under its new class name, which transformers 4 does not have; the old
# name is an alias of the same class in transformers 5.
_TRANSFORMERS5_FAST_TOKENIZER_CLASS = "TokenizersBackend"
_FAST_TOKENIZER_CLASS = "PreTrainedTokenizerFast"


def _required(hf_config: PretrainedConfig, name: str) -> Any:
    value = getattr(hf_config, name, None)
    if value is None:
        raise ValueError(f"ERNIE 5.5 config.json must set {name!r}.")
    return value


def _ernie55_hca_rope_settings(hf_config: PretrainedConfig) -> dict[str, Any]:
    """H-layer RoPE fields of an ERNIE 5.5 Hugging Face config.

    ``hca_rope_type`` ``"rope"`` (the default) is plain RoPE; ``"yarn"`` is Megatron-Core's YaRN at
    ``compress_rope_theta``, as in the reference implementation. Its keys are read from the top level of the
    config, then from ``rope_scaling``, then take the reference defaults (``beta_fast`` 32, ``beta_slow`` 1). The
    YaRN concentration factor is not applied: DSv4 attention rotates with ``mscale=1.0``, as the reference does.

    Raises:
        ValueError: If ``hca_rope_type`` is neither ``"rope"`` nor ``"yarn"``.
    """
    hca_rope_type = getattr(hf_config, "hca_rope_type", None) or "rope"
    if hca_rope_type not in ERNIE55_HCA_ROPE_TYPES:
        raise ValueError(
            f"ERNIE 5.5 config.json: hca_rope_type={hca_rope_type!r}; the H layers support {ERNIE55_HCA_ROPE_TYPES}."
        )
    settings: dict[str, Any] = dict(
        ernie_hca_rope_type=hca_rope_type, rotary_scaling_factor=1.0, mscale=1.0, mscale_all_dim=0.0
    )
    if hca_rope_type == "yarn":
        rope_scaling = getattr(hf_config, "rope_scaling", None) or {}

        def yarn(name: str, scaling_name: str, default: Any) -> Any:
            value = getattr(hf_config, name, None)
            return rope_scaling.get(scaling_name, default) if value is None else value

        settings.update(
            rotary_scaling_factor=float(yarn("rotary_scaling_factor", "factor", 1.0)),
            original_max_position_embeddings=int(
                yarn("original_max_position_embeddings", "original_max_position_embeddings", 4096)
            ),
            beta_fast=float(yarn("beta_fast", "beta_fast", 32.0)),
            beta_slow=float(yarn("beta_slow", "beta_slow", 1.0)),
        )
    return settings


def _ernie55_settings(hf_config: PretrainedConfig) -> dict[str, Any]:
    """Flat Megatron config fields of an ERNIE 5.5 Hugging Face config.

    Raises:
        ValueError: If a required key is missing or the config is outside the supported envelope.
    """

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(f"ERNIE 5.5 config.json: {message}")

    # Ernie55Config checks its keys as given; a config object of another class is checked here.
    check_ernie55_reference_key_spellings(hf_config)
    hca_rope = _ernie55_hca_rope_settings(hf_config)
    d_attention = getattr(hf_config, "hybrid_mla_attention", None)
    require(
        d_attention == "mqa_dsa",
        f"hybrid_mla_attention={d_attention!r}; Bridge implements the D layer as latent MQA with DSA top-k ('mqa_dsa').",
    )
    require(not getattr(hf_config, "mla_dense_layers", None), "mla_dense_layers is not supported.")
    require(
        bool(getattr(hf_config, "use_qk_norm", None)), "use_qk_norm must be true (ERNIE 5.5 normalizes the latents)."
    )
    require(not getattr(hf_config, "tie_word_embeddings", False), "tie_word_embeddings must be false.")
    require(not getattr(hf_config, "attention_bias", False), "attention_bias must be false.")
    require(not getattr(hf_config, "mlp_bias", False), "mlp_bias must be false.")
    require(not getattr(hf_config, "attention_dropout", 0.0), "attention_dropout must be 0.")
    scoring = getattr(hf_config, "scoring_func", "sigmoid")
    require(scoring == "sigmoid", f"scoring_func={scoring!r}; the ERNIE 5.5 router scores with 'sigmoid'.")
    hidden_act = getattr(hf_config, "hidden_act", None)
    require(hidden_act in ("silu", "situ"), f"hidden_act={hidden_act!r}; ERNIE 5.5 uses 'silu' or 'situ'.")
    hc_eps = _required(hf_config, "hc_eps")
    require(
        float(hc_eps) == ERNIE55_HC_EPS,
        f"hc_eps={hc_eps}; the ERNIE 5.5 hyper-connection arithmetic uses {ERNIE55_HC_EPS}.",
    )
    require(
        int(_required(hf_config, "num_key_value_heads")) == 1,
        "num_key_value_heads must be 1 (the H layers share one latent key/value head).",
    )

    head_dim = int(_required(hf_config, "head_dim"))
    v_head_dim = int(_required(hf_config, "v_head_dim"))
    rope_dim = int(_required(hf_config, "qk_rope_head_dim"))
    nope_dim = int(_required(hf_config, "qk_nope_head_dim"))
    require(head_dim == v_head_dim, f"head_dim ({head_dim}) must equal v_head_dim ({v_head_dim}).")
    require(
        nope_dim == v_head_dim - rope_dim,
        f"qk_nope_head_dim ({nope_dim}) must be v_head_dim - qk_rope_head_dim ({v_head_dim} - {rope_dim}).",
    )

    num_hidden_layers = int(_required(hf_config, "num_hidden_layers"))
    num_mtp = int(getattr(hf_config, "num_nextn_predict_layers", 0) or 0)
    pattern = ernie55_hybrid_layer_pattern(
        _required(hf_config, "compress_ratios"),
        num_hidden_layers=num_hidden_layers,
        first_k_dense_replace=int(getattr(hf_config, "first_k_dense_replace", 0) or 0),
        num_mtp_layers=num_mtp,
    )
    shared_last_layer = bool(getattr(hf_config, "mtp_shared_last_layer", False))
    require(not (shared_last_layer and not num_mtp), "mtp_shared_last_layer needs num_nextn_predict_layers > 0.")

    moe_ffn = int(_required(hf_config, "moe_intermediate_size"))
    n_shared = int(getattr(hf_config, "n_shared_experts", 0) or 0)
    latent_size = getattr(hf_config, "moe_latent_size", None) or None
    situ = hidden_act == "situ"
    vha_rank = getattr(hf_config, "vha_rank", None)
    rope_theta = float(_required(hf_config, "rope_theta"))

    return dict(
        # Layout: two hybrid layers per ERNIE layer, MTP depths spelled out in the pattern.
        num_layers=2 * num_hidden_layers,
        hybrid_layer_pattern=pattern,
        is_hybrid_model=True,
        mtp_num_layers=num_mtp or None,
        ernie_mtp_shared_last_layer=shared_last_layer,
        experimental_attention_variant="dsv4_hybrid",
        multi_latent_attention=True,
        csa_compress_ratios=ernie55_csa_compress_ratios(pattern),
        # Shared settings.
        vocab_size=int(_required(hf_config, "vocab_size")),
        hidden_size=int(_required(hf_config, "hidden_size")),
        ffn_hidden_size=int(_required(hf_config, "intermediate_size")),
        share_embeddings_and_output_weights=False,
        position_embedding_type="none",
        normalization="RMSNorm",
        layernorm_epsilon=float(_required(hf_config, "rms_norm_eps")),
        qk_layernorm=True,
        add_bias_linear=False,
        add_qkv_bias=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        gated_linear_unit=True,
        activation_func=F.silu,
        activation_func_tanh_clamp_scale=float(getattr(hf_config, "activation_situ_beta", 1.0)) if situ else None,
        activation_func_tanh_clamp_scale_linear=(
            getattr(hf_config, "activation_situ_linear_beta", None) if situ else None
        ),
        bias_activation_fusion=False,
        apply_rope_fusion=False,
        rotary_base=rope_theta,
        ernie_vha_rank=int(vha_rank) if vha_rank is not None else ERNIE55_DEFAULT_VHA_RANK,
        ernie_output_gate_fp32=bool(getattr(hf_config, "sigmoid_gate_fusion", False)),
        # H layers (DSv4 heavily compressed attention, plain RoPE or YaRN at compress_rope_theta).
        num_attention_heads=int(_required(hf_config, "num_attention_heads")),
        q_lora_rank=int(_required(hf_config, "q_lora_rank")),
        v_head_dim=v_head_dim,
        qk_head_dim=nope_dim,
        qk_pos_emb_head_dim=rope_dim,
        output_projection_groups=int(_required(hf_config, "o_groups")),
        output_projection_lora_rank=int(_required(hf_config, "o_lora_rank")),
        csa_window_size=int(_required(hf_config, "sliding_window")),
        csa_compress_rotary_base=float(_required(hf_config, "compress_rope_theta")),
        csa_dense_mode=bool(getattr(hf_config, "csa_dense_mode", True)),
        rope_type="rope",
        **hca_rope,
        # D layers (latent MQA + DSA indexer, RoPE at rope_theta).
        ernie_d_num_attention_heads=int(_required(hf_config, "hybrid_mla_num_attention_heads")),
        ernie_d_q_lora_rank=int(_required(hf_config, "hybrid_mla_q_lora_rank")),
        ernie_d_kv_lora_rank=int(_required(hf_config, "hybrid_mla_kv_lora_rank")),
        ernie_d_qk_head_dim=int(_required(hf_config, "hybrid_mla_qk_nope_head_dim")),
        ernie_d_qk_pos_emb_head_dim=int(_required(hf_config, "hybrid_mla_qk_rope_head_dim")),
        ernie_d_v_head_dim=int(_required(hf_config, "hybrid_mla_v_head_dim")),
        ernie_d_rotary_base=rope_theta,
        # The checkpoint's RoPE channel layout, for weight conversion; the model always rotates adjacent pairs.
        ernie_mqa_latent_rope_adjacent_pairing=bool(getattr(hf_config, "mqa_latent_rope_adjacent_pairing", False)),
        dsa_indexer_n_heads=int(_required(hf_config, "index_n_heads")),
        dsa_indexer_head_dim=int(_required(hf_config, "index_head_dim")),
        dsa_indexer_topk=int(_required(hf_config, "index_topk")),
        dsa_indexer_rope_interleaved=False,
        dsa_indexer_rotate_activation=True,
        dsa_indexer_scoring_relu=True,
        dsa_indexer_k_norm_fp32=True,
        dsa_indexer_weights_proj_output_dtype="bf16",
        # Hyper-connections.
        enable_mhc_connections=True,
        mhc_num_residual_streams=int(_required(hf_config, "hc_mult")),
        mhc_sinkhorn_iterations=int(_required(hf_config, "hc_sinkhorn_iters")),
        use_fused_mhc=False,
        # MoE.
        num_moe_experts=int(_required(hf_config, "n_routed_experts")),
        moe_router_topk=int(_required(hf_config, "num_experts_per_tok")),
        moe_ffn_hidden_size=moe_ffn,
        moe_shared_expert_intermediate_size=n_shared * moe_ffn if n_shared else None,
        moe_router_score_function="sigmoid",
        moe_router_enable_expert_bias=True,
        moe_router_bias_update_rate=0.0,
        moe_router_load_balancing_type="none",
        moe_aux_loss_coeff=0.0,
        moe_router_dtype="fp32",
        # The reference's Muon excludes the router ("mlp.gate"): AdamW updates it.
        moe_router_skip_muon=True,
        moe_router_topk_scaling_factor=float(getattr(hf_config, "routed_scaling_factor", 1.0)),
        moe_router_num_groups=None,
        moe_router_group_topk=None,
        ernie_norm_topk_prob=bool(getattr(hf_config, "norm_topk_prob", True)),
        moe_latent_size=int(latent_size) if latent_size else None,
        moe_use_norm_before_up_proj=bool(getattr(hf_config, "latent_moe_use_norm", False)) if latent_size else False,
        moe_grouped_gemm=True,
        moe_token_dispatcher_type="alltoall",
    )


@MegatronModelBridge.register_bridge(
    source=ERNIE55_ARCHITECTURE,
    target=Ernie55HybridModel,
    model_type=ERNIE55_MODEL_TYPE,
)
class Ernie55Bridge(Ernie55WeightConversionMixin, MegatronModelBridge):
    """Bridge for ERNIE 5.5: Hugging Face config to :class:`Ernie55ModelConfig` (builder-backed only).

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> model_config = AutoBridge.from_hf_pretrained("/path/to/ernie55").get_model_config()

    There is no model provider for ERNIE 5.5: :meth:`provider_bridge` raises. Use the builder-backed
    :meth:`~megatron.bridge.models.conversion.auto_bridge.AutoBridge.get_model_config`.
    """

    MODEL_CONFIG_CLASS = Ernie55ModelConfig
    USE_MODEL_CONFIG_FOR_CONVERSION = True

    @classmethod
    def hf_to_megatron_activation(cls, hidden_act: str):
        """SwiGLU (``silu``) and SiTU-GLU (``situ``) both use ``F.silu``; SiTU adds Megatron-Core's clamp scales."""
        if hidden_act in ("silu", "situ"):
            return F.silu
        return super().hf_to_megatron_activation(hidden_act)

    def hf_config_to_model_config_kwargs(self, hf_config: PretrainedConfig) -> dict[str, Any]:
        """Flat :class:`Ernie55ModelConfig` kwargs of an ERNIE 5.5 Hugging Face config.

        Args:
            hf_config: The ``ernie5_5_moe`` config.

        Returns:
            Model and transformer fields; ``seq_length`` is left to the recipe.

        Raises:
            ValueError: If the config is outside the supported envelope (see the module docstring).
        """
        config_kwargs = super().hf_config_to_model_config_kwargs(hf_config)
        for name in _DROPPED_GENERIC_FIELDS:
            config_kwargs.pop(name, None)
        config_kwargs.update(_ernie55_settings(hf_config))
        return config_kwargs

    def hf_config_to_provider_kwargs(self, hf_config: PretrainedConfig) -> dict[str, Any]:
        """ERNIE 5.5 has no provider path.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "ERNIE 5.5 is builder-backed only; use AutoBridge.get_model_config() (Ernie55ModelConfig), not "
            "to_megatron_provider()."
        )

    def provider_bridge(self, hf_pretrained: Any) -> Any:
        """ERNIE 5.5 has no provider path.

        Raises:
            NotImplementedError: Always.
        """
        return self.hf_config_to_provider_kwargs(getattr(hf_pretrained, "config", hf_pretrained))

    def postprocess_hf_export_artifacts(self, path: Path) -> None:
        """Name the exported tokenizer's class so that transformers 4 can load it.

        transformers 5 writes ``"tokenizer_class": "TokenizersBackend"`` into ``tokenizer_config.json``; the
        reference implementation runs transformers 4.57, whose ``AutoTokenizer`` rejects that name. The released
        checkpoint's tokenizer names ``PreTrainedTokenizerFast``, which transformers 5 keeps as an alias of the
        same class, so the export restores it. Tokens and every other key stay as saved.

        Args:
            path: Directory containing the exported Hugging Face artifacts.
        """
        super().postprocess_hf_export_artifacts(path)
        tokenizer_config = Path(path) / "tokenizer_config.json"
        if not tokenizer_config.is_file():
            return
        settings = json.loads(tokenizer_config.read_text(encoding="utf-8"))
        if settings.get("tokenizer_class") != _TRANSFORMERS5_FAST_TOKENIZER_CLASS:
            return
        settings["tokenizer_class"] = _FAST_TOKENIZER_CLASS
        tokenizer_config.write_text(json.dumps(settings, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    @classmethod
    def megatron_to_hf_config(cls, provider: Ernie55ModelConfig) -> dict[str, Any]:
        """The ERNIE 5.5 ``config.json`` of an :class:`Ernie55ModelConfig`.

        Args:
            provider: The model config.

        Returns:
            The Hugging Face config dict (``architectures``, ``model_type`` and the ERNIE keys).

        Raises:
            ValueError: If the dense layers are not a prefix of the decoder (``first_k_dense_replace``).
        """
        layout = parse_ernie55_hybrid_pattern(provider.hybrid_layer_pattern)
        layers = [*layout.layers, *(layout.mtp_layers * layout.mtp_depths)]
        dense = [mlp == "-" for _, mlp in layers]
        first_k_dense = dense.index(False) if False in dense else len(dense)
        if any(dense[first_k_dense:]):
            raise ValueError(
                f"ERNIE 5.5 dense layers must lead the decoder (first_k_dense_replace); pattern "
                f"{provider.hybrid_layer_pattern!r} has a dense layer after an MoE layer."
            )
        ratio_of = {"H": ERNIE55_H_COMPRESS_RATIO, "D": ERNIE55_D_COMPRESS_RATIO}
        transformer = provider.transformer
        situ = transformer.activation_func_tanh_clamp_scale is not None
        hf_config: dict[str, Any] = {
            "architectures": [ERNIE55_ARCHITECTURE],
            "model_type": ERNIE55_MODEL_TYPE,
            "torch_dtype": "bfloat16" if transformer.bf16 else "float16" if transformer.fp16 else "float32",
            "vocab_size": provider.vocab_size,
            "hidden_size": transformer.hidden_size,
            "num_hidden_layers": len(layout.layers),
            "intermediate_size": transformer.ffn_hidden_size,
            "first_k_dense_replace": first_k_dense,
            "rms_norm_eps": transformer.layernorm_epsilon,
            "hidden_act": "situ" if situ else "silu",
            "tie_word_embeddings": provider.share_embeddings_and_output_weights,
            "attention_bias": transformer.add_qkv_bias,
            "mlp_bias": transformer.add_bias_linear,
            "attention_dropout": transformer.attention_dropout,
            "initializer_range": transformer.init_method_std,
            "num_attention_heads": transformer.num_attention_heads,
            "num_key_value_heads": 1,
            "head_dim": transformer.v_head_dim,
            "v_head_dim": transformer.v_head_dim,
            "qk_rope_head_dim": transformer.qk_pos_emb_head_dim,
            "qk_nope_head_dim": transformer.v_head_dim - transformer.qk_pos_emb_head_dim,
            "q_lora_rank": transformer.q_lora_rank,
            "o_groups": transformer.output_projection_groups,
            "o_lora_rank": transformer.output_projection_lora_rank,
            "use_qk_norm": True,
            "rope_theta": transformer.ernie_d_rotary_base,
            "compress_rope_theta": transformer.csa_compress_rotary_base,
            "hca_rope_type": transformer.ernie_hca_rope_type,
            "sliding_window": transformer.csa_window_size,
            "compress_ratios": [ratio_of[attention] for attention, _ in layers],
            "csa_dense_mode": transformer.csa_dense_mode,
            "hybrid_mla_attention": "mqa_dsa",
            "hybrid_mla_num_attention_heads": transformer.ernie_d_num_attention_heads,
            "hybrid_mla_q_lora_rank": transformer.ernie_d_q_lora_rank,
            "hybrid_mla_kv_lora_rank": transformer.ernie_d_kv_lora_rank,
            "hybrid_mla_qk_nope_head_dim": transformer.ernie_d_qk_head_dim,
            "hybrid_mla_qk_rope_head_dim": transformer.ernie_d_qk_pos_emb_head_dim,
            "hybrid_mla_v_head_dim": transformer.ernie_d_v_head_dim,
            # The source checkpoint's layout: weight conversion writes the RoPE rows back in it.
            "mqa_latent_rope_adjacent_pairing": transformer.ernie_mqa_latent_rope_adjacent_pairing,
            "index_n_heads": transformer.dsa_indexer_n_heads,
            "index_head_dim": transformer.dsa_indexer_head_dim,
            "index_topk": transformer.dsa_indexer_topk,
            "hc_mult": transformer.mhc_num_residual_streams,
            "hc_sinkhorn_iters": transformer.mhc_sinkhorn_iterations,
            "hc_eps": ERNIE55_HC_EPS,
            "n_routed_experts": transformer.num_moe_experts,
            "num_experts_per_tok": transformer.moe_router_topk,
            "moe_intermediate_size": transformer.moe_ffn_hidden_size,
            "n_shared_experts": (transformer.moe_shared_expert_intermediate_size or 0)
            // transformer.moe_ffn_hidden_size,
            "scoring_func": "sigmoid",
            "norm_topk_prob": transformer.ernie_norm_topk_prob,
            "routed_scaling_factor": transformer.moe_router_topk_scaling_factor,
            "latent_moe_use_norm": transformer.moe_use_norm_before_up_proj,
            "num_nextn_predict_layers": layout.mtp_depths,
            "mtp_shared_last_layer": transformer.ernie_mtp_shared_last_layer,
            "sigmoid_gate_fusion": transformer.ernie_output_gate_fp32,
        }
        if transformer.moe_latent_size:
            hf_config["moe_latent_size"] = transformer.moe_latent_size
        if situ:
            hf_config["activation_situ_beta"] = transformer.activation_func_tanh_clamp_scale
            hf_config["activation_situ_linear_beta"] = transformer.activation_func_tanh_clamp_scale_linear
        if transformer.ernie_vha_rank != ERNIE55_DEFAULT_VHA_RANK:
            hf_config["vha_rank"] = transformer.ernie_vha_rank
        if transformer.ernie_hca_rope_type == "yarn":
            yarn = {
                "factor": transformer.rotary_scaling_factor,
                "original_max_position_embeddings": transformer.original_max_position_embeddings,
                "beta_fast": transformer.beta_fast,
                "beta_slow": transformer.beta_slow,
            }
            hf_config["rotary_scaling_factor"] = yarn["factor"]
            hf_config["original_max_position_embeddings"] = yarn["original_max_position_embeddings"]
            hf_config["rope_scaling"] = {"type": "yarn", **yarn}
        return hf_config
