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

"""ERNIE 5.5 configuration: the transformer config and the builder-backed ``HybridModel`` config.

* :class:`Ernie55TransformerConfig` is Bridge's MLA config plus the ERNIE fields Megatron-Core lacks.
* :class:`Ernie55ModelConfig` is the ``HybridModel`` config whose builder
  (:class:`~megatron.bridge.models.ernie.ernie55_builder.Ernie55ModelBuilder`) constructs
  :class:`~megatron.bridge.models.ernie.modeling_ernie55.model.Ernie55HybridModel` on every pipeline stage.
* The hybrid layout helpers translate between the Hugging Face layer description
  (``compress_ratios``, ``first_k_dense_replace``) and the hybrid pattern: every ERNIE layer is two hybrid
  layers, an attention symbol (``H`` for an H layer, ``D`` for a D layer) followed by ``-`` (dense MLP) or
  ``E`` (MoE), and every MTP depth repeats the layer description of its ``compress_ratios`` entry
  (``"H-DEHEDE/DE"`` for the 4-layer proxy with one MTP depth).
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols, parse_hybrid_pattern

from megatron.bridge.models.ernie.modeling_ernie55 import (
    validate_ernie55_config,
    validate_ernie55_hca_config,
    validate_ernie55_mhc_config,
    validate_ernie55_moe_config,
    validate_ernie55_multimax_config,
)
from megatron.bridge.models.ernie.modeling_ernie55.context_parallel import require_cp_layout_api
from megatron.bridge.models.hybrid.model_config import BridgeHybridModelConfig
from megatron.bridge.models.transformer_config import MLATransformerConfig, TransformerConfig


__all__ = [
    "ERNIE55_D_COMPRESS_RATIO",
    "ERNIE55_H_COMPRESS_RATIO",
    "Ernie55HybridLayout",
    "Ernie55ModelConfig",
    "Ernie55TransformerConfig",
    "ernie55_csa_compress_ratios",
    "ernie55_hybrid_layer_pattern",
    "ernie55_pipeline_layer_pattern",
    "parse_ernie55_hybrid_pattern",
]

ERNIE55_H_COMPRESS_RATIO = 128
"""Hugging Face ``compress_ratios`` entry of an H layer (heavily compressed attention)."""

ERNIE55_D_COMPRESS_RATIO = -2
"""Hugging Face ``compress_ratios`` entry of a D layer (latent MQA with DSA top-k)."""

_ATTENTION_SYMBOLS = {ERNIE55_H_COMPRESS_RATIO: Symbols.HCA, ERNIE55_D_COMPRESS_RATIO: Symbols.DS_ATTENTION}
_RATIO_OF_SYMBOL = {symbol: ratio for ratio, symbol in _ATTENTION_SYMBOLS.items()}
_MLP_SYMBOLS = (Symbols.MLP, Symbols.MOE)
# Megatron-Core's DSv4 compress ratio of each attention symbol (D layers are not DSv4 layers: 0).
_CSA_RATIO_OF_SYMBOL = {Symbols.HCA: ERNIE55_H_COMPRESS_RATIO}


@dataclass
class Ernie55TransformerConfig(MLATransformerConfig):
    """Bridge ``MLATransformerConfig`` plus the ERNIE 5.5 fields the D-layer, H-layer and MoE modules read.

    Existing Megatron-Core fields configure the rest of the D layer: ``csa_window_size``
    (sliding window ``W``), ``dsa_indexer_topk`` / ``dsa_indexer_n_heads`` /
    ``dsa_indexer_head_dim``, ``dsa_indexer_loss_coeff`` with ``dsa_indexer_use_sparse_loss``,
    and ``dsa_kernel_backend`` (``"cudnn"`` fused, ``"none"`` PyTorch reference).

    H layers (Megatron-Core's DSv4 hybrid attention at compress ratio 128) read the stack geometry
    (``num_attention_heads``, ``q_lora_rank``, ``v_head_dim``, ``qk_pos_emb_head_dim``,
    ``output_projection_groups``, ``output_projection_lora_rank``), ``csa_window_size`` and
    ``csa_compress_rotary_base`` (Hugging Face ``compress_rope_theta``, their RoPE base), plus
    :attr:`ernie_hca_rope_type` and the shared ERNIE fields ``ernie_vha_rank``, ``ernie_vha_bf16_mix`` and
    ``ernie_output_gate_fp32``.

    The MoE layers use Megatron-Core's MoE fields: ``num_moe_experts``, ``moe_router_topk``,
    ``moe_ffn_hidden_size``, ``moe_shared_expert_intermediate_size``,
    ``moe_router_topk_scaling_factor`` (initial value of the learned per-expert scale),
    ``moe_router_bias_update_rate``, ``moe_latent_size`` and ``moe_use_norm_before_up_proj``. SiTU-GLU
    (Hugging Face ``hidden_act="situ"``) is Megatron-Core's ``activation_func_tanh_clamp_scale``
    (``activation_situ_beta``) and ``activation_func_tanh_clamp_scale_linear``
    (``activation_situ_linear_beta``); see :mod:`~megatron.bridge.models.ernie.modeling_ernie55.moe`.

    Hyper-connections use Megatron-Core's ``enable_mhc_connections``, ``mhc_num_residual_streams``
    and ``mhc_sinkhorn_iterations``; ERNIE 5.5 changes their arithmetic, not their configuration
    (:mod:`~megatron.bridge.models.ernie.modeling_ernie55.hyper_connection`). MTP uses
    Megatron-Core's fields and the hybrid pattern's ``/`` depths, plus
    :attr:`ernie_mtp_shared_last_layer`. The multimax output head has no configuration.

    :meth:`finalize` validates the supported envelope with
    :func:`~megatron.bridge.models.ernie.modeling_ernie55.validate_ernie55_config`,
    :func:`~megatron.bridge.models.ernie.modeling_ernie55.validate_ernie55_mhc_config` and
    :func:`~megatron.bridge.models.ernie.modeling_ernie55.validate_ernie55_multimax_config`; with the
    DSv4 hybrid attention variant that H layers need, also with
    :func:`~megatron.bridge.models.ernie.modeling_ernie55.validate_ernie55_hca_config`; and, when MoE
    is configured, with :func:`~megatron.bridge.models.ernie.modeling_ernie55.validate_ernie55_moe_config`.
    """

    ernie_mtp_shared_last_layer: bool = False
    """SD-MTP (Hugging Face ``mtp_shared_last_layer``): every MTP depth's inner layers share the
    Parameters of the decoder's last layers, whose layer symbols they must repeat (``DE`` for ERNIE
    5.5). :class:`~megatron.bridge.models.ernie.modeling_ernie55.model.Ernie55HybridModel` applies
    it after construction. It cannot be combined with Megatron-FSDP, torch FSDP2 or meta-device
    initialization, which re-create parameters module by module; the training setup checks this with
    :func:`~megatron.bridge.models.ernie.modeling_ernie55.validate_ernie55_mtp_sharing`."""

    ernie_norm_topk_prob: bool = True
    """Divide the top-k gate values by their sum (plus 1e-20) when ``moe_router_topk > 1``, before the
    per-expert scale (Hugging Face ``norm_topk_prob``)."""

    ernie_vha_rank: int | None = None
    """Rank ``R`` of the VHA head post-mix ``I + V U^T`` (``U``, ``V`` are ``[heads, R]``) of both the
    D and the H layers. Required when either is built; ERNIE 5.5 uses 16. Hugging Face ``vha_rank``
    (optional; the released configs omit it, which means 16)."""

    ernie_hca_rope_type: str = "rope"
    """H-layer RoPE (Hugging Face ``hca_rope_type``): ``"rope"`` is plain RoPE at ``csa_compress_rotary_base``,
    ``"yarn"`` Megatron-Core's YaRN at that base from ``rotary_scaling_factor``, ``original_max_position_embeddings``,
    ``beta_fast`` and ``beta_slow``, rotated without the concentration factor as in the reference implementation."""

    ernie_vha_bf16_mix: bool = True
    """Form ``I + V U^T`` and apply it with GEMMs in the activation dtype (bf16 in training), the
    reference implementation's default contract, in D and H layers. False accumulates the mix in fp32
    and rounds once."""

    ernie_output_gate_fp32: bool = False
    """Run the output gate's sigmoid and product in fp32 and round once, in D and H layers (Hugging
    Face ``sigmoid_gate_fusion``). False runs both in the activation dtype, the reference default."""

    ernie_native_sink_grad: bool = True
    """Let the cuDNN sparse-attention backward differentiate the attention sink on the fused path.
    True (the default) runs Megatron-Core's ``csa_sparse_attn``. ``test_sink_gradient`` checks its sink
    gradient at the 576-wide latent queries against the analytic gradient and an autograd reference; it
    has been checked on GB300 (sm_103), so run that test before relying on it on other GPUs. False uses
    the analytic sink gradient with a sink-free cuDNN backward. Has no effect on the reference path."""

    ernie_d_num_attention_heads: int | None = None
    """D-layer attention heads; None inherits ``num_attention_heads``."""

    ernie_d_q_lora_rank: int | None = None
    """D-layer q latent rank; None inherits ``q_lora_rank``."""

    ernie_d_kv_lora_rank: int | None = None
    """D-layer kv latent rank; None inherits ``kv_lora_rank``."""

    ernie_d_qk_head_dim: int | None = None
    """D-layer non-RoPE query/key head dim (Hugging Face ``qk_nope_head_dim``); None inherits ``qk_head_dim``."""

    ernie_d_qk_pos_emb_head_dim: int | None = None
    """D-layer RoPE head dim; None inherits ``qk_pos_emb_head_dim``."""

    ernie_d_v_head_dim: int | None = None
    """D-layer value head dim; None inherits ``v_head_dim``."""

    ernie_d_rotary_base: float | None = None
    """RoPE base of the D attention and its indexer (Hugging Face ``rope_theta``); None inherits
    ``rotary_base``."""

    ernie_mqa_latent_rope_adjacent_pairing: bool = True
    """Channel layout of the D-layer latent RoPE weights in the Hugging Face checkpoint
    (``mqa_latent_rope_adjacent_pairing``): ``True`` for adjacent pairs ``(2k, 2k + 1)``, ``False`` for the two
    halves. It has no effect on the model, which always rotates adjacent pairs (Megatron-Core's AbsorbedMLA);
    weight conversion permutes the RoPE rows of a half-split checkpoint on import and back on export, and the
    exported ``config.json`` keeps the source's value. The default is the model's own layout."""

    ernie_per_row_loss: bool = False
    """Train with the reference implementation's per-row LM loss
    (:func:`~megatron.bridge.models.ernie.ernie55_context_parallel.ernie55_per_row_lm_loss`): every
    micro-batch (one packed THD row) is averaged over its own supervised tokens, counted over the
    context-parallel group, and the training loss is the mean over micro-batches. Requires
    ``calculate_per_token_loss=False``, which also makes the D-layer indexer loss per row, and the
    model then normalizes every MTP depth the same way
    (:mod:`~megatron.bridge.models.ernie.modeling_ernie55.mtp_loss`). A context-parallel rank whose
    tokens are all masked contributes 0 instead of NaN. The training step must use that loss function
    (:mod:`~megatron.bridge.models.ernie.ernie55_step` does, and rejects inconsistent loss settings).
    There is no Hugging Face key."""

    ernie_fused_ops: bool = True
    """Run the reference implementation's Triton kernels (adapted in
    :mod:`~megatron.bridge.models.ernie.modeling_ernie55.fused_ops`) wherever their inputs fit: SiTU-GLU in the
    dense, routed and shared MLPs, the multimax head, and the RoPE of the H layers at context_parallel_size > 1
    (Megatron-Core's ``cp_utils.apply_thd_cp_local_rope_unfused``). CPU tensors, layouts the kernels do not take,
    and ``False`` run the eager paths. Megatron-Core's SiTU-GLU calls do not carry the config, so that kernel goes by
    the model's clamp scales: among models with the same scales in one process, the one built last decides; see
    :mod:`~megatron.bridge.models.ernie.modeling_ernie55.fused_dispatch`. The reference has no switch and always
    runs them on CUDA. There is no Hugging Face key; on the command line, ``model.ernie_fused_ops=false``."""

    ernie_reference_rmsnorm: bool = False
    """Build every RMSNorm of the model with the reference implementation's eager arithmetic
    (:class:`~megatron.bridge.models.ernie.modeling_ernie55.norm.Ernie55RMSNorm`) instead of Transformer Engine's
    ``TENorm``: each layer's input and pre-MLP norm (the MTP depths' layers included), the D and H q/kv latent norms,
    the H compressor's norm and the decoder's final norm, the norms the reference builds as ``ErnieRMSNorm``. The MTP
    block's ``enorm``, ``hnorm`` and final norm stay ``TENorm``, as in the reference. Both compute
    ``bf16(x * rsqrt(mean(x^2) + eps) * w)`` with fp32 statistics; they sum the squares in different orders, so
    about 0.05% of the token norms differ by one bf16 ulp. The reference arithmetic reproduces the reference's
    norms bitwise but is slower, so the recipe leaves it off; set it for bitwise parity runs against the
    reference. Parameter names and checkpoints are the same either way. There is no Hugging Face key; on the
    command line, ``model.ernie_reference_rmsnorm=true``."""

    def finalize(self) -> None:
        """Run the deferred Megatron-Core post-init, then validate the ERNIE 5.5 envelopes.

        The D-layer, hyper-connection and multimax envelopes are always checked; the hyper-connection
        check passes configs without ``enable_mhc_connections``.

        The H-layer envelope is checked only with ``experimental_attention_variant="dsv4_hybrid"``,
        the variant Megatron-Core's DSv4 attention requires; D-only stacks may use ``"dsa"``. The MoE
        envelope is checked only when ``num_moe_experts`` is set.

        Raises:
            ValueError: If ``ernie_fused_ops`` or ``ernie_reference_rmsnorm`` is not a bool, if
                ``ernie_per_row_loss`` is combined with ``calculate_per_token_loss`` or (with MTP)
                ``mtp_loss_early_backward`` or an MTP loss other than cross entropy, or if a layer envelope is
                violated.
            RuntimeError: If Megatron-Core lacks a piece the configured context parallelism needs (see
                :func:`~megatron.bridge.models.ernie.modeling_ernie55.validate_ernie55_context_parallel_config`).
        """
        super().finalize()
        if not isinstance(self.ernie_fused_ops, bool):
            raise ValueError(f"ernie_fused_ops must be a bool, got {self.ernie_fused_ops!r}.")
        if not isinstance(self.ernie_reference_rmsnorm, bool):
            raise ValueError(f"ernie_reference_rmsnorm must be a bool, got {self.ernie_reference_rmsnorm!r}.")
        if self.ernie_per_row_loss and self.calculate_per_token_loss:
            raise ValueError(
                "ernie_per_row_loss=True needs calculate_per_token_loss=False: the per-row loss returns one token per "
                "micro-batch and relies on the schedule's mean over micro-batches and DDP's mean over DP x CP."
            )
        if self.ernie_per_row_loss and self.mtp_num_layers and getattr(self, "mtp_loss_early_backward", False):
            raise ValueError(
                "ernie_per_row_loss=True does not support mtp_loss_early_backward: the per-row MTP loss runs every "
                "depth's backward through Megatron-Core's MTPLossAutoScaler."
            )
        mtp_loss_type = getattr(self, "mtp_loss_type", "cross_entropy")
        if self.ernie_per_row_loss and self.mtp_num_layers and mtp_loss_type != "cross_entropy":
            raise ValueError(
                f"ernie_per_row_loss=True supports only the cross-entropy MTP loss, got mtp_loss_type={mtp_loss_type!r}: "
                "the per-row MTP loss normalizes each depth's token cross entropy by the row's supervised tokens."
            )
        validate_ernie55_config(self)
        validate_ernie55_mhc_config(self)
        validate_ernie55_multimax_config(self)
        if self.experimental_attention_variant == "dsv4_hybrid":
            validate_ernie55_hca_config(self)
        if self.num_moe_experts:
            validate_ernie55_moe_config(self)


@dataclass(frozen=True)
class Ernie55HybridLayout:
    """An ERNIE 5.5 hybrid pattern, split into ERNIE layers.

    Args:
        layers: ``(attention symbol, MLP symbol)`` of every decoder layer, in order.
        mtp_layers: ``(attention symbol, MLP symbol)`` of every layer of one MTP depth.
        mtp_depths: Number of MTP depths (each repeats ``mtp_layers``).
        stage_boundaries: Indices into ``layers`` where a new pipeline stage starts (``|``).
    """

    layers: tuple[tuple[str, str], ...]
    mtp_layers: tuple[tuple[str, str], ...]
    mtp_depths: int
    stage_boundaries: tuple[int, ...]

    @property
    def num_hybrid_layers(self) -> int:
        """Hybrid layers of the decoder (two per ERNIE layer), MTP excluded."""
        return 2 * len(self.layers)


def _ernie_layers(symbols: str, what: str) -> tuple[tuple[str, str], ...]:
    """Split a pipe-free pattern section into ``(attention, MLP)`` pairs."""
    if len(symbols) % 2:
        raise ValueError(f"ERNIE 5.5 {what} {symbols!r} must pair every attention symbol with an MLP symbol.")
    layers = tuple((symbols[i], symbols[i + 1]) for i in range(0, len(symbols), 2))
    for index, (attention, mlp) in enumerate(layers):
        if attention not in _RATIO_OF_SYMBOL or mlp not in _MLP_SYMBOLS:
            raise ValueError(
                f"ERNIE 5.5 {what} {symbols!r}: layer {index} is {attention + mlp!r}; every ERNIE layer is an attention "
                f"symbol ({Symbols.HCA!r} or {Symbols.DS_ATTENTION!r}) followed by {Symbols.MLP!r} or {Symbols.MOE!r}."
            )
    return layers


def parse_ernie55_hybrid_pattern(pattern: str) -> Ernie55HybridLayout:
    """Parse and validate an ERNIE 5.5 hybrid pattern.

    Args:
        pattern: Unified hybrid pattern, e.g. ``"H-DEHEDE/DE"`` (``|`` may separate pipeline stages, at
            ERNIE-layer boundaries only).

    Returns:
        The pattern's ERNIE layers.

    Raises:
        ValueError: If the pattern is not a sequence of ERNIE layers, or a pipeline stage boundary splits one.
    """
    parsed = parse_hybrid_pattern(pattern)
    main = parsed.main_pattern or ""
    boundaries = []
    symbols = ""
    for segment_index, segment in enumerate(main.split(Symbols.PIPE)):
        if len(segment) % 2:
            raise ValueError(
                f"ERNIE 5.5 hybrid pattern {pattern!r}: pipeline segment {segment_index} ({segment!r}) splits an "
                "ERNIE layer (attention symbol plus MLP symbol)."
            )
        if segment_index:
            boundaries.append(len(symbols) // 2)
        symbols += segment
    layers = _ernie_layers(symbols, "hybrid pattern")
    if not layers:
        raise ValueError(f"ERNIE 5.5 hybrid pattern {pattern!r} has no decoder layer.")
    mtp_layers = _ernie_layers(parsed.mtp_pattern or "", "MTP pattern") if parsed.mtp_num_depths else ()
    return Ernie55HybridLayout(layers, mtp_layers, parsed.mtp_num_depths, tuple(boundaries))


def ernie55_hybrid_layer_pattern(
    compress_ratios: Sequence[int],
    *,
    num_hidden_layers: int,
    first_k_dense_replace: int,
    num_mtp_layers: int,
) -> str:
    """Build the hybrid pattern of an ERNIE 5.5 Hugging Face layer description.

    Layer ``i`` (decoder layers first, then one entry per MTP depth) is ``H`` for ``compress_ratios[i]
    == 128`` and ``D`` for ``-2``, followed by ``-`` when ``i < first_k_dense_replace`` and ``E`` otherwise.

    Args:
        compress_ratios: ``num_hidden_layers + num_mtp_layers`` compress ratios.
        num_hidden_layers: Decoder layers.
        first_k_dense_replace: Leading dense layers.
        num_mtp_layers: MTP depths.

    Returns:
        The unified hybrid pattern, e.g. ``"H-DEHEDE/DE"``.

    Raises:
        ValueError: On an unknown compress ratio, a ``compress_ratios`` length that does not cover the layers,
            or MTP depths that differ (Megatron-Core repeats one MTP pattern per depth).
    """
    ratios = [int(ratio) for ratio in compress_ratios]
    expected = num_hidden_layers + num_mtp_layers
    if len(ratios) != expected:
        raise ValueError(
            f"ERNIE 5.5 compress_ratios has {len(ratios)} entries; num_hidden_layers + num_nextn_predict_layers = "
            f"{num_hidden_layers} + {num_mtp_layers} = {expected}."
        )
    unknown = sorted({ratio for ratio in ratios if ratio not in _ATTENTION_SYMBOLS})
    if unknown:
        raise ValueError(
            f"ERNIE 5.5 compress_ratios contains {unknown}; only {ERNIE55_H_COMPRESS_RATIO} (H layer) and "
            f"{ERNIE55_D_COMPRESS_RATIO} (D layer) exist."
        )
    layers = [
        _ATTENTION_SYMBOLS[ratio] + (Symbols.MLP if index < first_k_dense_replace else Symbols.MOE)
        for index, ratio in enumerate(ratios)
    ]
    main, mtp = "".join(layers[:num_hidden_layers]), layers[num_hidden_layers:]
    if len(set(mtp)) > 1:
        raise ValueError(
            f"ERNIE 5.5 MTP depths must have the same layer, got {mtp}; Megatron-Core repeats one MTP pattern per depth."
        )
    return Symbols.MTP_SEPARATOR.join([main, *mtp])


def ernie55_csa_compress_ratios(pattern: str) -> list[int]:
    """Megatron-Core ``csa_compress_ratios`` of an ERNIE 5.5 hybrid pattern.

    One entry per hybrid layer, the decoder's (pipes skipped) and then every MTP depth's: ``128`` for an
    ``H`` layer and ``0`` for every other symbol (D layers are not DSv4 layers; Megatron-Core's DSv4 root
    accepts only 0, 4 and 128).

    Args:
        pattern: Unified hybrid pattern.

    Returns:
        The per-hybrid-layer ratios.
    """
    parsed = parse_hybrid_pattern(pattern)
    symbols = (parsed.main_pattern or "").replace(Symbols.PIPE, "") + (
        parsed.mtp_pattern or ""
    ) * parsed.mtp_num_depths
    return [_CSA_RATIO_OF_SYMBOL.get(symbol, 0) for symbol in symbols]


def ernie55_pipeline_layer_pattern(pattern: str, num_stages: int) -> str:
    """Split an ERNIE 5.5 hybrid pattern into ``num_stages`` pipeline segments at ERNIE-layer boundaries.

    Existing ``|`` separators are dropped, so a pattern saved under another pipeline size is re-split.
    Earlier stages take the remainder; the MTP depths stay after the last stage, where Megatron-Core
    runs them (SD-MTP needs the last decoder layer there too).

    Args:
        pattern: Unified hybrid pattern.
        num_stages: Pipeline stages (``pipeline_model_parallel_size * virtual_pipeline_model_parallel_size``).

    Returns:
        The pattern with ``num_stages - 1`` pipe separators.

    Raises:
        ValueError: If there are fewer ERNIE layers than stages.
    """
    layout = parse_ernie55_hybrid_pattern(pattern)
    parsed = parse_hybrid_pattern(pattern)
    layers = ["".join(layer) for layer in layout.layers]
    if num_stages > len(layers):
        raise ValueError(f"{len(layers)} ERNIE 5.5 layers cannot fill {num_stages} pipeline stages.")
    base, extra = divmod(len(layers), num_stages)
    segments, offset = [], 0
    for stage in range(num_stages):
        count = base + int(stage < extra)
        segments.append("".join(layers[offset : offset + count]))
        offset += count
    mtp = [parsed.mtp_pattern] * parsed.mtp_num_depths if parsed.mtp_num_depths else []
    return Symbols.MTP_SEPARATOR.join([Symbols.PIPE.join(segments), *mtp])


@dataclass(kw_only=True)
class Ernie55ModelConfig(BridgeHybridModelConfig):
    """ERNIE 5.5 ``HybridModel`` config, built by :class:`~megatron.bridge.models.ernie.ernie55_builder.Ernie55ModelBuilder`.

    ``hybrid_layer_pattern`` spells out the decoder and every MTP depth (``"H-DEHEDE/DE"``), so
    ``transformer.num_layers`` counts hybrid layers (two per ERNIE layer) and ``mtp_num_layers`` must equal
    the pattern's MTP depths. ``mtp_hybrid_override_pattern`` is not used.

    :meth:`finalize` validates the model-level envelope before the transformer config's own checks:

    * the pattern is a sequence of ERNIE layers (``H``/``D`` then ``-``/``E``) and ``mtp_num_layers`` matches
      its MTP depths;
    * ``csa_compress_ratios`` is the pattern's (set from the pattern when unset);
    * ``experimental_attention_variant="dsv4_hybrid"``, ``position_embedding_type="none"`` (every ERNIE
      attention owns its RoPE) and an untied output layer;
    * with H layers, Megatron-Core's ``DSv4HybridAttention._post_o_group_proj`` hook exists (Megatron-Core
      dev f78809f6d or later), and at context_parallel_size > 1 the stack is contiguous
      (``cp_partition_mode="contiguous"``), the DSv4 layout;
    * with pipeline parallelism and no ``|`` in the pattern, the pattern is split at ERNIE-layer boundaries
      (:func:`ernie55_pipeline_layer_pattern`); with pipeline parallelism ``variable_seq_lengths`` is turned on,
      because the ERNIE 5.5 SFT data path emits one packed THD row per micro-batch whose length is data-dependent
      (``ernie55_step`` pads its documents to multiples of ``2 * CP`` at CP > 1), so the stages exchange tensor
      shapes before each transfer instead of posting static receive buffers sized from ``seq_length``; the
      ``allgather`` MoE token dispatcher, which cannot take variable sequence lengths, is rejected there.

    The builder always uses :func:`~megatron.bridge.models.ernie.modeling_ernie55.ernie55_hybrid_stack_spec`
    unless ``hybrid_stack_spec`` names another ERNIE 5.5 stack spec, and never pads the vocabulary (the
    multimax head and the loss must see exactly ``vocab_size`` logits).
    """

    builder: ClassVar[str] = "megatron.bridge.models.ernie.ernie55_builder.Ernie55ModelBuilder"
    transformer_config_class: ClassVar[type[TransformerConfig]] = Ernie55TransformerConfig

    def ernie55_layout(self) -> Ernie55HybridLayout:
        """The ERNIE layers of ``hybrid_layer_pattern``.

        Raises:
            ValueError: If the pattern is unset or not an ERNIE 5.5 pattern.
        """
        if not self.hybrid_layer_pattern:
            raise ValueError("ERNIE 5.5 needs hybrid_layer_pattern (e.g. 'H-DEHEDE/DE').")
        return parse_ernie55_hybrid_pattern(self.hybrid_layer_pattern)

    def dsa_indexer_metric_num_layers(self) -> int:
        """Divisor of the logged indexer loss: ERNIE layers plus MTP depths (the reference convention).

        The D layers record their indexer losses in Megatron-Core's DSA tracker; the logged ``indexer loss`` is
        their sum divided by this number, as in the reference implementation (``(L_D + ... + L_MTP) / (num_layers
        + mtp)``, e.g. ``/ 5`` for the 4-layer proxy with one MTP depth), not by the number of D layers.
        """
        layout = self.ernie55_layout()
        return len(layout.layers) + layout.mtp_depths

    def finalize(self) -> None:
        """Validate the ERNIE 5.5 model-level envelope, then finalize the hybrid and transformer configs.

        Raises:
            ValueError: If the pattern, the MTP depths, ``csa_compress_ratios``, the attention variant, the
                position embedding, embedding tying or the context-parallel layout is outside the ERNIE 5.5
                envelope.
            RuntimeError: If the pattern has H layers and Megatron-Core lacks the ``_post_o_group_proj`` hook.
        """
        transformer = self.transformer
        if transformer.mtp_hybrid_override_pattern:
            raise ValueError(
                "ERNIE 5.5 spells its MTP depths in hybrid_layer_pattern ('<decoder>/<depth>/...'); leave "
                "mtp_hybrid_override_pattern unset."
            )
        if self.hybrid_override_pattern is not None:
            raise ValueError("ERNIE 5.5 uses hybrid_layer_pattern; hybrid_override_pattern is deprecated.")
        layout = self.ernie55_layout()
        if (transformer.mtp_num_layers or 0) != layout.mtp_depths:
            raise ValueError(
                f"ERNIE 5.5 hybrid_layer_pattern {self.hybrid_layer_pattern!r} has {layout.mtp_depths} MTP depths, "
                f"but mtp_num_layers is {transformer.mtp_num_layers}."
            )
        if transformer.experimental_attention_variant != "dsv4_hybrid":
            raise ValueError(
                "ERNIE 5.5 stacks are DeepSeek-V4 hybrid stacks; experimental_attention_variant must be 'dsv4_hybrid', "
                f"got {transformer.experimental_attention_variant!r}."
            )
        if self.position_embedding_type != "none":
            raise ValueError(
                "ERNIE 5.5 attention layers apply their own RoPE; position_embedding_type must be 'none', got "
                f"{self.position_embedding_type!r}."
            )
        if self.share_embeddings_and_output_weights:
            raise ValueError(
                "ERNIE 5.5 has an untied output layer; share_embeddings_and_output_weights must be False."
            )

        num_stages = (transformer.pipeline_model_parallel_size or 1) * (
            transformer.virtual_pipeline_model_parallel_size or 1
        )
        if (
            num_stages > 1
            and not layout.stage_boundaries
            and transformer.num_layers_in_first_pipeline_stage is None
            and transformer.num_layers_in_last_pipeline_stage is None
        ):
            self.hybrid_layer_pattern = ernie55_pipeline_layer_pattern(self.hybrid_layer_pattern, num_stages)
        if (transformer.pipeline_model_parallel_size or 1) > 1:
            # Without ``variable_seq_lengths`` the pipeline schedules post static receive buffers sized from
            # ``seq_length`` (per rank, after CP / sequence-parallel splits and the hyper-connection streams), and a
            # receive only completes when the sent activation has that shape. The ERNIE 5.5 SFT data path
            # (``collate_ernie55_rows``, consumed by ``ernie55_step``) emits one packed THD row per micro-batch whose
            # length is data-dependent (``ernie55_step`` pads its documents to multiples of ``2 * CP`` at CP > 1), so
            # the sent shape varies between micro-batches and a mismatch leaves the receiving stage waiting until
            # the NCCL watchdog fires (at CP1 with micro-batch size 1 the row goes out at its collated length, at
            # most ``seq_length``).
            # Exchanging the shapes before each transfer is what Bridge arranges for its own packed and paired
            # datasets (``_enable_in_batch_packing``, DPO); it is a model-level default here, so callers that feed
            # fixed-length rows pay the exchange too. Megatron-Core's packed fixed-shape mode is unaffected: it
            # requires ``variable_seq_lengths`` together with ``pipeline_p2p_fixed_shape``.
            if transformer.num_moe_experts and transformer.moe_token_dispatcher_type == "allgather":
                raise ValueError(
                    "ERNIE 5.5 pipeline stages exchange rows of data-dependent length (variable_seq_lengths), which "
                    "the 'allgather' MoE token dispatcher does not support; use moe_token_dispatcher_type='alltoall' "
                    "(the ERNIE 5.5 default)."
                )
            transformer.variable_seq_lengths = True

        expected_ratios = ernie55_csa_compress_ratios(self.hybrid_layer_pattern)
        if transformer.csa_compress_ratios is None:
            transformer.csa_compress_ratios = expected_ratios
        elif list(transformer.csa_compress_ratios) != expected_ratios:
            raise ValueError(
                f"ERNIE 5.5 csa_compress_ratios {list(transformer.csa_compress_ratios)} do not match hybrid_layer_pattern "
                f"{self.hybrid_layer_pattern!r} (expected {expected_ratios}: 128 for every H layer, 0 otherwise, decoder "
                "then MTP depths). Leave csa_compress_ratios unset to derive it from the pattern."
            )

        has_h_layers = any(Symbols.HCA in layer for layer in (*layout.layers, *layout.mtp_layers))
        if has_h_layers:
            from megatron.core.transformer.experimental_attention_variant.deepseek_v4_hybrid_attention import (
                DSv4HybridAttention,
            )

            if not callable(getattr(DSv4HybridAttention, "_post_o_group_proj", None)):
                raise RuntimeError(
                    "ERNIE 5.5 H layers need Megatron-Core's DSv4HybridAttention._post_o_group_proj hook. Use "
                    "Megatron-Core dev at or after f78809f6d."
                )
            if (transformer.context_parallel_size or 1) > 1:
                # Megatron-Core main has no CP partition modes (no cp_partition_mode field): say so first.
                require_cp_layout_api("ERNIE 5.5 stacks at context_parallel_size > 1")
                if transformer.cp_partition_mode != "contiguous":
                    raise ValueError(
                        "ERNIE 5.5 stacks with H layers run context parallelism in the contiguous partition "
                        "(Megatron-Core DSv4's layout); set cp_partition_mode='contiguous', got "
                        f"{transformer.cp_partition_mode!r}."
                    )
        super().finalize()
