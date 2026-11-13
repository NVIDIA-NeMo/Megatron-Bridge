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

"""Hugging Face ``PretrainedConfig`` for ERNIE 5.5 (``model_type == "ernie5_5_moe"``, formerly ``"ernie5_5"``).

ERNIE 5.5 has no Hugging Face modeling code: the checkpoints carry a ``config.json`` whose
``architectures`` entry is ``Ernie5_5_MoeForCausalLM``. This class lets
``AutoConfig.from_pretrained`` read such a checkpoint offline and lets recipes build a config
from scratch with ``AutoConfig.for_model("ernie5_5_moe", ...)``. Importing this module registers the
class with ``AutoConfig``, together with :class:`Ernie55LegacyConfig` for checkpoints written before the
``model_type`` became ``"ernie5_5_moe"`` (``"ernie5_5"``, the reference implementation's ``--model_type``). Both read
the same keys and build the same model; exports write ``"ernie5_5_moe"``.

Architecture fields default to ``None``: an ERNIE 5.5 config must spell them out, and the Bridge
mapping (:class:`~megatron.bridge.models.ernie.ernie55_bridge.Ernie55Bridge`) rejects a config that
omits one. Optional keys default to the value the reference implementation assumes when the key is
missing, so a config written without them means the same thing to both.

The reference implementation also reads ``n_routed_experts``, ``first_k_dense_replace`` and
``n_shared_experts`` under other names (:data:`ERNIE55_REFERENCE_KEY_SPELLINGS`); Bridge reads only these
keys. A config whose other spellings give a different value is rejected
(:func:`check_ernie55_reference_key_spellings`), because the two would build different models.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig


__all__ = [
    "ERNIE55_ARCHITECTURE",
    "ERNIE55_LEGACY_MODEL_TYPE",
    "ERNIE55_MODEL_TYPE",
    "ERNIE55_REFERENCE_KEY_SPELLINGS",
    "Ernie55Config",
    "Ernie55LegacyConfig",
    "check_ernie55_reference_key_spellings",
]

ERNIE55_MODEL_TYPE = "ernie5_5_moe"
"""Hugging Face ``model_type`` of ERNIE 5.5 checkpoints; Bridge exports write it."""

ERNIE55_LEGACY_MODEL_TYPE = "ernie5_5"
"""``model_type`` of ERNIE 5.5 checkpoints written before ``"ernie5_5_moe"``; Bridge still reads them."""

ERNIE55_ARCHITECTURE = "Ernie5_5_MoeForCausalLM"
"""Hugging Face ``architectures`` entry of ERNIE 5.5 checkpoints."""

ERNIE55_REFERENCE_KEY_SPELLINGS: dict[str, tuple[str, ...]] = {
    "n_routed_experts": ("num_experts", "n_routed_experts", "moe_num_experts", "num_local_experts"),
    "first_k_dense_replace": ("first_k_dense_replace", "moe_layer_start_index"),
    "n_shared_experts": ("n_shared_experts", "num_shared_expert", "moe_num_shared_experts", "num_shared_experts"),
}
"""Config keys the reference implementation also reads under other names: the names in its lookup order, where the
first one present wins. Bridge reads only the key itself."""


def _config_value(config: Any, name: str) -> Any:
    """``name`` of a Hugging Face config object or a plain dict, ``None`` when absent."""
    if isinstance(config, Mapping):
        return config.get(name)
    return getattr(config, name, None)


def check_ernie55_reference_key_spellings(config: Any) -> None:
    """Reject a config that spells a key the way only the reference implementation reads it.

    The reference implementation also accepts other names for ``n_routed_experts``, ``first_k_dense_replace`` and
    ``n_shared_experts`` and takes the first name present (:data:`ERNIE55_REFERENCE_KEY_SPELLINGS`). Bridge reads
    only the key itself, so a config where the two readings differ would build a different model: with
    ``num_shared_experts`` alone, for example, the model would have no shared experts, and an import would skip
    their tensors.

    Args:
        config: The config's keys as given (a dict), or a config object. :class:`Ernie55Config` checks its own
            keys before it fills in defaults.

    Raises:
        ValueError: If the reference reading of a key differs from the key's own value.
    """
    for key, lookup_order in ERNIE55_REFERENCE_KEY_SPELLINGS.items():
        present = [(name, _config_value(config, name)) for name in lookup_order]
        present = [(name, value) for name, value in present if value is not None]
        others = [name for name, _ in present if name != key]
        if not others or present[0][1] == _config_value(config, key):
            continue
        raise ValueError(
            f"ERNIE 5.5 config: the config sets {', '.join(others)}, which the reference implementation reads as "
            f"{key}={present[0][1]!r}, but Bridge reads only {key} ({_config_value(config, key)!r}). Set {key} to "
            "the intended value and drop the other spellings."
        )


class Ernie55Config(PretrainedConfig):
    """Configuration of the ERNIE 5.5 hybrid MoE language model (``Ernie5_5_MoeForCausalLM``).

    Layer ``i`` of the ``num_hidden_layers`` decoder layers is an H layer (DeepSeek-V4 heavily
    compressed attention) when ``compress_ratios[i] == 128`` and a D layer (latent MQA with DSA top-k,
    window and sink) when it is ``-2``; its MLP is dense for ``i < first_k_dense_replace`` and MoE
    after. The ``num_nextn_predict_layers`` trailing ``compress_ratios`` entries describe the MTP
    depths. Each layer and each MTP depth carries ``hc_mult`` hyper-connection residual streams.

    Args:
        vocab_size: Vocabulary size (the model's, not the tokenizer's).
        hidden_size: Hidden size.
        num_hidden_layers: Decoder layers, MTP excluded.
        intermediate_size: FFN width of the dense layers.
        first_k_dense_replace: Number of leading dense (non-MoE) layers (0 when absent).
        max_position_embeddings: Longest supported sequence.
        rms_norm_eps: RMSNorm epsilon.
        hidden_act: MLP activation, ``"silu"`` (SwiGLU) or ``"situ"`` (SiTU-GLU).
        activation_situ_beta: SiTU gate clamp scale (``hidden_act="situ"``).
        activation_situ_linear_beta: SiTU up-projection clamp scale, ``None`` for no clamp.
        tie_word_embeddings: Whether the output layer shares the embedding (ERNIE 5.5: ``False``).
        attention_bias: Bias on the attention projections (ERNIE 5.5: ``False``).
        mlp_bias: Bias on the MLP projections (ERNIE 5.5: ``False``).
        attention_dropout: Attention dropout (ERNIE 5.5: ``0``).
        initializer_range: Standard deviation of the weight initialization.
        num_attention_heads: H-layer attention heads.
        num_key_value_heads: H-layer key/value heads (one shared latent head).
        head_dim: H-layer head dimension (equal to ``v_head_dim``).
        v_head_dim: H-layer value and latent dimension.
        qk_rope_head_dim: H-layer RoPE dimension.
        qk_nope_head_dim: H-layer non-RoPE dimension, ``head_dim - qk_rope_head_dim``.
        q_lora_rank: H-layer query latent rank.
        o_groups: H-layer grouped output projection groups.
        o_lora_rank: H-layer grouped output projection rank per group.
        use_qk_norm: Whether the q/kv latents are RMS-normalized (ERNIE 5.5: ``True``).
        rope_theta: RoPE base of the D layers and their indexer. A config with ``rope_scaling`` carries it in
            ``rope_parameters["rope_theta"]`` as well (transformers 5 stores ``rope_scaling`` under that name), so
            that a saved ``config.json`` has it where transformers 5 readers look for it.
        compress_rope_theta: RoPE base of the H layers.
        hca_rope_type: H-layer RoPE type; ``"rope"`` is plain RoPE.
        rotary_scaling_factor: YaRN factor of the H layers when ``hca_rope_type`` is YaRN; ``None`` (absent)
            reads ``rope_scaling["factor"]``, as the released checkpoints carry it, and is 1 when that is absent too.
        original_max_position_embeddings: YaRN original length of the H layers; ``None`` (absent) reads
            ``rope_scaling["original_max_position_embeddings"]``, and is 4096 when that is absent too.
        sliding_window: Attention window of H and D layers.
        compress_ratios: Per-layer compress ratio (``128`` H, ``-2`` D), decoder then MTP depths.
        csa_dense_mode: Whether compressed layers run without an indexer (H layers never have one).
        hybrid_mla_attention: D-layer attention kind; ``"mqa_dsa"`` is latent MQA with DSA top-k.
        hybrid_mla_num_attention_heads: D-layer attention heads.
        hybrid_mla_q_lora_rank: D-layer query latent rank.
        hybrid_mla_kv_lora_rank: D-layer key/value latent rank.
        hybrid_mla_qk_nope_head_dim: D-layer non-RoPE query/key dimension.
        hybrid_mla_qk_rope_head_dim: D-layer RoPE dimension.
        hybrid_mla_v_head_dim: D-layer value dimension.
        mqa_latent_rope_adjacent_pairing: Whether D-layer RoPE rotates adjacent channel pairs
            ``(2k, 2k + 1)`` (``True``) or the two halves (``False``). ``None`` (absent) reads
            ``rope_interleave``, the released checkpoints' name for it, and is ``False`` when that is absent too.
        index_n_heads: D-layer indexer heads.
        index_head_dim: D-layer indexer head dimension.
        index_topk: D-layer indexer top-k.
        hc_mult: Hyper-connection residual streams.
        hc_sinkhorn_iters: Sinkhorn iterations of the hyper-connection mixing matrix.
        hc_eps: Hyper-connection epsilon.
        n_routed_experts: Routed experts per MoE layer.
        num_experts_per_tok: Experts per token.
        moe_intermediate_size: Expert FFN width.
        n_shared_experts: Shared experts (their width is ``n_shared_experts * moe_intermediate_size``; 0 when
            absent).
        scoring_func: Router score function (ERNIE 5.5: ``"sigmoid"``).
        norm_topk_prob: Whether the top-k gate values are normalized to sum to one.
        routed_scaling_factor: Initial value of the learned per-expert routing scale.
        moe_latent_size: Latent width of the routed experts, ``None`` for no latent MoE.
        latent_moe_use_norm: Whether latent MoE normalizes before the up-projection.
        num_nextn_predict_layers: MTP depths.
        mtp_shared_last_layer: Whether every MTP depth's body shares the last decoder layer's
            parameters (SD-MTP).
        sigmoid_gate_fusion: Whether the attention output gate runs in fp32 and rounds once.
        vha_rank: Rank of the VHA head post-mix of H and D layers. ERNIE 5.5 configs do not carry
            this key; ``None`` means the architecture's 16.
        **kwargs: Passed to :class:`~transformers.PretrainedConfig` (token ids, dtype, ...).

    Raises:
        ValueError: If ``kwargs`` spells ``n_routed_experts``, ``first_k_dense_replace`` or ``n_shared_experts``
            the way only the reference implementation reads it, with another value
            (:func:`check_ernie55_reference_key_spellings`).
    """

    model_type = ERNIE55_MODEL_TYPE

    def __init__(
        self,
        vocab_size: int | None = None,
        hidden_size: int | None = None,
        num_hidden_layers: int | None = None,
        intermediate_size: int | None = None,
        first_k_dense_replace: int | None = None,
        max_position_embeddings: int | None = None,
        rms_norm_eps: float | None = None,
        hidden_act: str = "silu",
        activation_situ_beta: float = 1.0,
        activation_situ_linear_beta: float | None = None,
        tie_word_embeddings: bool = False,
        attention_bias: bool = False,
        mlp_bias: bool = False,
        attention_dropout: float = 0.0,
        initializer_range: float = 0.02,
        num_attention_heads: int | None = None,
        num_key_value_heads: int | None = None,
        head_dim: int | None = None,
        v_head_dim: int | None = None,
        qk_rope_head_dim: int | None = None,
        qk_nope_head_dim: int | None = None,
        q_lora_rank: int | None = None,
        o_groups: int | None = None,
        o_lora_rank: int | None = None,
        use_qk_norm: bool | None = None,
        rope_theta: float | None = None,
        compress_rope_theta: float | None = None,
        hca_rope_type: str = "rope",
        rotary_scaling_factor: float | None = None,
        original_max_position_embeddings: int | None = None,
        sliding_window: int | None = None,
        compress_ratios: list[int] | None = None,
        csa_dense_mode: bool = True,
        hybrid_mla_attention: str = "mha",
        hybrid_mla_num_attention_heads: int | None = None,
        hybrid_mla_q_lora_rank: int | None = None,
        hybrid_mla_kv_lora_rank: int | None = None,
        hybrid_mla_qk_nope_head_dim: int | None = None,
        hybrid_mla_qk_rope_head_dim: int | None = None,
        hybrid_mla_v_head_dim: int | None = None,
        mqa_latent_rope_adjacent_pairing: bool | None = None,
        index_n_heads: int = 64,
        index_head_dim: int = 128,
        index_topk: int = 2048,
        hc_mult: int | None = None,
        hc_sinkhorn_iters: int | None = None,
        hc_eps: float | None = None,
        n_routed_experts: int | None = None,
        num_experts_per_tok: int | None = None,
        moe_intermediate_size: int | None = None,
        n_shared_experts: int | None = None,
        scoring_func: str = "sigmoid",
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 1.0,
        moe_latent_size: int | None = None,
        latent_moe_use_norm: bool = False,
        num_nextn_predict_layers: int = 0,
        mtp_shared_last_layer: bool = False,
        sigmoid_gate_fusion: bool = False,
        vha_rank: int | None = None,
        **kwargs: Any,
    ) -> None:
        # Before the defaults: a key the config leaves out is absent, not 0, to the reference's other spellings.
        check_ernie55_reference_key_spellings(
            {
                "n_routed_experts": n_routed_experts,
                "first_k_dense_replace": first_k_dense_replace,
                "n_shared_experts": n_shared_experts,
                **kwargs,
            }
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.first_k_dense_replace = 0 if first_k_dense_replace is None else first_k_dense_replace
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.hidden_act = hidden_act
        self.activation_situ_beta = activation_situ_beta
        self.activation_situ_linear_beta = activation_situ_linear_beta
        self.attention_bias = attention_bias
        self.mlp_bias = mlp_bias
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.v_head_dim = v_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.q_lora_rank = q_lora_rank
        self.o_groups = o_groups
        self.o_lora_rank = o_lora_rank
        self.use_qk_norm = use_qk_norm
        self.rope_theta = rope_theta
        self.compress_rope_theta = compress_rope_theta
        self.hca_rope_type = hca_rope_type
        rope_scaling = kwargs.get("rope_scaling") or {}
        if rotary_scaling_factor is None:
            rotary_scaling_factor = rope_scaling.get("factor", 1.0)
        if original_max_position_embeddings is None:
            original_max_position_embeddings = rope_scaling.get("original_max_position_embeddings", 4096)
        self.rotary_scaling_factor = rotary_scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.sliding_window = sliding_window
        self.compress_ratios = list(compress_ratios) if compress_ratios is not None else None
        self.csa_dense_mode = csa_dense_mode
        self.hybrid_mla_attention = hybrid_mla_attention
        self.hybrid_mla_num_attention_heads = hybrid_mla_num_attention_heads
        self.hybrid_mla_q_lora_rank = hybrid_mla_q_lora_rank
        self.hybrid_mla_kv_lora_rank = hybrid_mla_kv_lora_rank
        self.hybrid_mla_qk_nope_head_dim = hybrid_mla_qk_nope_head_dim
        self.hybrid_mla_qk_rope_head_dim = hybrid_mla_qk_rope_head_dim
        self.hybrid_mla_v_head_dim = hybrid_mla_v_head_dim
        if mqa_latent_rope_adjacent_pairing is None:
            mqa_latent_rope_adjacent_pairing = bool(kwargs.get("rope_interleave", False))
        self.mqa_latent_rope_adjacent_pairing = mqa_latent_rope_adjacent_pairing
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hc_eps = hc_eps
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.n_shared_experts = 0 if n_shared_experts is None else n_shared_experts
        self.scoring_func = scoring_func
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.moe_latent_size = moe_latent_size
        self.latent_moe_use_norm = latent_moe_use_norm
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.mtp_shared_last_layer = mtp_shared_last_layer
        self.sigmoid_gate_fusion = sigmoid_gate_fusion
        self.vha_rank = vha_rank
        kwargs.setdefault("architectures", [ERNIE55_ARCHITECTURE])
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        # transformers 5 keeps ``rope_scaling`` as ``rope_parameters``, and readers written against it take the RoPE
        # base from ``rope_parameters["rope_theta"]``. PretrainedConfig moves the base there only when ``rope_theta``
        # is among its kwargs, and here it is a named argument. The top-level key, the one Bridge reads, decides.
        rope_parameters = getattr(self, "rope_parameters", None)
        if isinstance(rope_parameters, dict) and rope_theta is not None:
            self.rope_parameters = {**rope_parameters, "rope_theta": rope_theta}


class Ernie55LegacyConfig(Ernie55Config):
    """:class:`Ernie55Config` under the earlier ``model_type`` ``"ernie5_5"``, so that ``AutoConfig`` reads such
    checkpoints; the keys and the model are the same."""

    model_type = ERNIE55_LEGACY_MODEL_TYPE

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # transformers 5 generates an __init__ for every PretrainedConfig subclass that does not define one, which
        # would skip Ernie55Config.__init__ and its defaults.
        super().__init__(*args, **kwargs)


AutoConfig.register(ERNIE55_MODEL_TYPE, Ernie55Config, exist_ok=True)
AutoConfig.register(ERNIE55_LEGACY_MODEL_TYPE, Ernie55LegacyConfig, exist_ok=True)
