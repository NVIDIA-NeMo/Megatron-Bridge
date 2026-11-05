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

"""Core attention of the ERNIE 5.5 D layer (``compress_ratio == -2``).

The module sits in the ``core_attention`` slot of Megatron-Core's
``AbsorbedMLASelfAttention``. It receives the absorbed query ``[q_nope W_UK | q_pe]``, the
single-head latent key ``[c_kv | k_pe]``, the q latent ``qr`` (the ``q_layernorm`` output),
the layer input ``x`` and the per-head V up-projection, and returns the gated, projected
attention output ``[..., heads * v_head_dim]``; AbsorbedMLA then applies ``linear_proj``.

Per query row ``t`` of a document starting at ``d(t)``:

1. The Megatron-Core ``DSAIndexer`` scores the remote keys ``[d(t), t - W]`` and keeps the
   top ``K`` of them (``W = csa_window_size``, ``K = dsa_indexer_topk``).
2. Latent MQA attends over ``top-k ∪ [max(d(t), t - W + 1), t]`` (disjoint sets) with a
   per-head learnable sink in the softmax denominator. The value is the first
   ``kv_lora_rank`` latent channels.
3. V up-projection, then the VHA head mix ``I + V U^T``, then ``out * sigmoid(W_g qr)``.
4. In training, the indexer learns from a KL loss against a top-k-only attention teacher
   (Megatron-Core's sparse DSA teacher). The loss only reaches the indexer parameters and is
   scaled per micro-batch by ``DSAIndexerLossAutoScaler``. Rows without remote candidates
   contribute exactly zero. The reduction follows the loss mode:

   * ``calculate_per_token_loss=False`` (reference-implementation parity):
     ``coeff * sum_t KL_t / (b * s)`` over every row of the micro-batch, padding rows of a
     pack included; the schedule then averages over micro-batches. Under context parallelism
     ``s`` is the length of the whole pack, not the rank's share (see below).
   * ``calculate_per_token_loss=True`` (required by Bridge's runtime THD packing):
     Megatron-Core's convention, the raw ``coeff * sum_t KL_t``; ``finalize_model_grads`` then
     divides every gradient by the global loss-token count. The indexer gradient differs from
     the reference by ``rows / loss tokens``, so the core warns once. Use per-row loss for
     parity checks.

Context parallelism (CP > 1, packed THD only). The attention module hands the core its rows in
the zigzag partition (rank ``r`` holds chunks ``r`` and ``2 * CP - 1 - r`` of every document; see
:mod:`~megatron.bridge.models.ernie.modeling_ernie55.context_parallel`), with AbsorbedMLA's RoPE and
the indexer's RoPE already applied at the rows' true document positions. The core then

* takes the global packed position of every local row from the zigzag geometry (built on the host
  from the micro-batch's prebuilt ``cu_seqlens``, cached per micro-batch);
* all-gathers the 576-wide latent key and the indexer key over the CP group and puts them in
  global packed order (the backward reduce-scatters the key gradients to their owners);
* evaluates the document bounds ``[d(t), t - W]``, the window and the placeholder candidate (see
  :func:`~megatron.bridge.models.ernie.modeling_ernie55.ops.ernie55_kernel_selection_ends`) in global
  coordinates, so the indices address the gathered keys;
* runs the fused scorer as Megatron-Core's zigzag ``PACKED_THD_GENERAL`` cuDNN plan: it passes the
  packed layout with one tensor as both the query and the key ``cu_seqlens`` (the plan's same-view
  requirement) and ``packed_thd_single_sequence=False``, because the single-sequence plan picks
  wrong top-k with cuDNN frontend >= 1.26 (Megatron-Core #8534). The plan scores each document
  causally from its start, and the top-k keeps only the first ``end - start`` columns, so the
  lagging ERNIE end ``t - W`` is honored. Both top-k hooks get these arguments: the loss hook in a
  training forward or a full-recompute replay, and the plain top-k hook in a forward without the
  loss (evaluation, or the no-grad forward of full recompute). :attr:`requested_indexer_scoring_plan`
  records the plan the arguments each hook received select;
* normalizes the per-row indexer loss by the rows of the whole pack, as the reference does: it
  passes ``coeff / CP``, so rank ``r``'s loss is ``coeff * sum_{t on r} KL_t / (b * s_pack)``. The
  schedule scales the indexer gradient by ``CP / micro-batches`` and DDP averages it over DP x CP,
  so the gradient is that of ``coeff * sum_t KL_t / (b * s_pack)`` over the whole pack.
  Megatron-Core DSA's per-row convention normalizes by the local rows instead, which makes its
  indexer gradient CP times too large; it is not inherited here. The tracker averages the loss
  over the CP group, the reference's logging convention (``1 / CP`` of the pack's row mean).
  Per-token loss keeps Megatron-Core's convention, which is already CP-consistent.

Two backends are selected by ``use_fused_dsa_kernels(config)``: the fused path (cuDNN DSA
indexer and loss hooks, FlashMLA sparse forward, cuDNN sparse backward; CUDA, bf16, SM90+)
and the PyTorch reference path, which is exact but materializes ``O(heads * T^2)`` fp32
tensors and is meant for CPU tests, oracles and short sequences. The reference attention
gathers the latent key in fp32, so its key gradient is summed over query rows in fp32 and
rounded once.

Indexer scores follow the reference implementation's convention on every path: the indexer
weights ``w = weights_proj(x) * H^-0.5 * d^-0.5`` (bf16) are multiplied by ``d^0.5`` in bf16
before scoring, and the score ``sum_h relu(q_h . k) * w_h`` is multiplied by ``d^-0.5`` in fp32
after the head reduction (``d`` = ``dsa_indexer_head_dim``). Megatron-Core's convention scores
with the pre-scaled weights instead; the math is the same, but the extra bf16 rounding of the
reference moves every score by about 3e-4 of the row's largest score, and near-tied keys at the
top-k boundary swap. The cuDNN hooks read their score scale from
``dsa_cudnn_kernels._INDEXER_SOFTMAX_SCALE`` (scores, the KL inputs and the indexer backward), so
the core sets it to ``d^-0.5`` around each hook call. Gradients reach ``w`` through the bf16
multiplication.

The fused attention's selection and key lists:

* At CP = 1 the attention's top-k is the reference implementation's own: its cuDNN
  ``indexer_forward_wrapper`` call on 512-row chunks (with ``q_causal_offsets``), the remote
  bounds as a ``-inf`` mask and ``torch.topk``, which leaves the keys in descending score order.
  FlashMLA gets ``[top-k | window]`` rows without their all-invalid 128-key blocks
  (:func:`~megatron.bridge.models.ernie.modeling_ernie55.ops.drop_invalid_key_blocks`) plus the
  kept lengths. FlashMLA's output depends on the key order and on which keys share a block, and
  this list reproduces the reference's output bitwise while skipping its empty blocks. In training
  the indexer loss still comes from Megatron-Core's loss hook, which selects the same keys
  (``PACKED_THD_GENERAL`` for packed THD rows, ``UNFUSED_BOUNDS`` otherwise); its top-k only feeds
  the loss. (A pack with tail rows past ``cu_seqlens[-1]``, which only hand-built inputs have, keeps
  ``UNFUSED_BOUNDS``, because the packed plan would leave the tail rows out of the last document.)
* At CP > 1 the hooks' top-k (sorted by key id) goes to FlashMLA in compacted
  ``[top-k | window]`` lists with per-row lengths, the format Megatron-Core DSA uses at
  ``d_qk = 576``.

Megatron-Core issue worked around on the fused path: for an SBHD batch
(``b > 1``), the cuDNN sparse indexer loss (``dsa_cudnn_kernels._compute_sparse_indexer_loss_and_grads``)
gives no indexer gradient to the samples after the first. For the indexer backward it offsets the
top-k indices by ``sample * s``, but it passes the indexer key per sample (``[b, s, d]``), and
cuDNN's ``indexer_backward_wrapper`` reads local per-sample indices. The KL value and the top-k are
right. The core therefore calls the loss hook once per sample when ``b > 1``; THD packs have
``b = 1`` and take one call.

Known performance cliffs on the fused path (measure them with
``examples/models/ernie55/benchmark_d_layer_indexer.py``):

* Indexer scoring of SBHD batches. Without packed metadata, ERNIE's remote bounds are neither
  causal nor reconstructible, so the loss hook scores them with Megatron-Core's generic
  ``UNFUSED_BOUNDS`` plan: fp32 per-head score chunks over the full ``s x s`` rectangle, with
  no causal skipping and no tensor cores, about ``2 * s^2 * n_heads * head_dim`` FLOPs per layer
  forward. Packed THD rows (every training step of the ERNIE recipes) use ``PACKED_THD_GENERAL``
  at every CP size, and the CP-1 attention top-k uses the causal cuDNN score kernel.
* Indexer loss. The teacher log-sum-exp comes from an extra FlashMLA pass (sink ``-inf``)
  over the top-k columns, which adds roughly one attention forward on steps with the loss.

Expected numerical differences from the reference implementation (not bugs):

* ``attn_sink`` is an fp32 parameter used as is; the reference stores a bf16 parameter and
  upcasts it at use. The forward is identical for bf16-representable sinks; training
  trajectories differ by the bf16 rounding of the sink.
* Rows past ``cu_seqlens[-1]`` (tail padding) extend the last document here; the reference
  assumes packs without such rows. Keep tail rows out of parity checks.
"""

from __future__ import annotations

import contextlib
import inspect
import logging
import math
from collections.abc import Callable, Iterator
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant import csa, dsa, dsa_kernels
from megatron.core.transformer.experimental_attention_variant.csa_utils import fused_sparse_attention
from megatron.core.transformer.experimental_attention_variant.dsa_layout import get_packed_qk_cu_seqlens
from megatron.core.transformer.module import MegatronModule, mark_keep_in_fp32
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import get_pg_size, log_single_rank
from torch import Tensor

from megatron.bridge.models.ernie.modeling_ernie55 import context_parallel
from megatron.bridge.models.ernie.modeling_ernie55.ops import (
    attention_sink_grad,
    drop_invalid_key_blocks,
    ernie55_document_starts,
    ernie55_kernel_selection_ends,
    ernie55_selection_bounds,
    ernie55_window_indices,
    lse_with_sink,
    mask_topk_beyond_end,
    pack_end,
    pad_topk,
    sigmoid_output_gate,
    vha_postmix,
)


logger = logging.getLogger(__name__)

__all__ = [
    "INDEXER_SCORING_PACKED_THD_GENERAL",
    "INDEXER_SCORING_REFERENCE_ORDER",
    "INDEXER_SCORING_UNFUSED_BOUNDS",
    "REFERENCE_FALLBACK_MAX_TOKENS",
    "Ernie55LatentMQACoreAttention",
    "Ernie55LatentMQACoreSubmodules",
    "indexer_loss_layer_number",
]

REFERENCE_FALLBACK_MAX_TOKENS = 4096
"""Most query rows (``b * s``, the whole pack under context parallelism) for which a declined fused
top-k hook may fall back to the reference path.

The reference top-k and loss materialize ``[b, s, s]`` scores and a ``[b, heads, s, s]`` teacher
in fp32, so a silent fallback at production lengths would run out of memory far from the
cause. Above this size a declined hook raises instead."""

INDEXER_SCORING_UNFUSED_BOUNDS = "unfused_bounds"
"""Megatron-Core's ``IndexerScoringPlan.UNFUSED_BOUNDS``: fp32 per-head score chunks over every key."""

INDEXER_SCORING_PACKED_THD_GENERAL = "packed_thd_general"
"""Megatron-Core's ``IndexerScoringPlan.PACKED_THD_GENERAL``: cuDNN causal scoring per packed document."""

INDEXER_SCORING_REFERENCE_ORDER = "reference_order"
"""The core's own CP-1 selection: the reference implementation's cuDNN indexer score call on 512-row chunks
and ``torch.topk``, which keeps the reference's top-k keys and their score order."""

# Megatron-Core's IndexerScoringPlan.PACKED_SINGLE_SEGMENTS, the single-sequence scorer of #8534; this core
# never requests it (single_packed_thd_sequence is always False), the record only names it if it did.
_INDEXER_SCORING_PACKED_SINGLE_SEGMENTS = "packed_single_segments"


# The cuDNN hooks ignore their block-size argument; it is part of the shared hook signature.
_FUSED_TOPK_BLOCK_SIZE = 8192
# FlashMLA's sparse kernels need d_qk = 576 (512 latent + 64 rope) and d_v = 512.
_FLASH_MLA_KV_LORA_RANK = 512
_FLASH_MLA_ROPE_DIM = 64
_FLASH_MLA_HEAD_COUNTS = (64, 128)
# cuDNN's DSA indexer kernels.
_CUDNN_INDEXER_HEAD_DIM = 128
_CUDNN_INDEXER_HEAD_COUNTS = (32, 64)
# Index-list width alignment Megatron-Core's DSA path uses for FlashMLA at d_qk = 576.
_DSA_FLASH_MLA_TOPK_ALIGNMENT = 512
# Granularity at which drop_invalid_key_blocks trims the CP-1 key lists: a multiple of FlashMLA's key block on
# SM100 (64 keys at 64 heads, 128 at 128 heads) and SM90 (128).
_FLASH_MLA_KEY_BLOCK = 128
# Query rows per cuDNN indexer score call of the CP-1 selection: the reference implementation's chunk
# size, so every score and top-k call has the reference's shapes.
_REFERENCE_SCORE_CHUNK_ROWS = 512

_WARNED: set[str] = set()


def _warn_once(key: str, message: str, level: int = logging.WARNING) -> None:
    """Log ``message`` once per process, on the default logging rank only."""
    if key not in _WARNED:
        _WARNED.add(key)
        log_single_rank(logger, level, message)


_REFERENCE_INDEXER_SCORER: Callable | None = None
_REFERENCE_INDEXER_SCORER_PROBED = False


def _reference_indexer_scorer(q: Tensor) -> Callable | None:
    """cuDNN's ``indexer_forward_wrapper`` when it can score ``q`` as the reference implementation does, else None.

    It needs a bf16 CUDA query and a cuDNN frontend whose wrapper takes ``q_causal_offsets`` (1.26+), the
    per-chunk causal offset the reference passes for every chunk after the first.
    """
    global _REFERENCE_INDEXER_SCORER, _REFERENCE_INDEXER_SCORER_PROBED
    if not q.is_cuda or q.dtype != torch.bfloat16:
        return None
    if not _REFERENCE_INDEXER_SCORER_PROBED:
        _REFERENCE_INDEXER_SCORER_PROBED = True
        try:
            from cudnn.deepseek_sparse_attention.indexer_forward import indexer_forward_wrapper
        except ImportError:
            indexer_forward_wrapper = None
        if indexer_forward_wrapper is not None:
            try:
                parameters = inspect.signature(indexer_forward_wrapper).parameters
            except (TypeError, ValueError):
                parameters = {}
            if "q_causal_offsets" in parameters:
                _REFERENCE_INDEXER_SCORER = indexer_forward_wrapper
        if _REFERENCE_INDEXER_SCORER is None:
            _warn_once(
                "reference_scorer",
                "The cuDNN frontend has no indexer_forward_wrapper with q_causal_offsets (needs 1.26+), so ERNIE 5.5 "
                "D layers at context_parallel_size=1 attend over the fused top-k hook's keys in key-id order, not "
                "the reference implementation's score order.",
            )
    return _REFERENCE_INDEXER_SCORER


def _pack_covers_rows(packed_seq_params: PackedSeqParams, rows: int) -> bool:
    """Whether a CP-1 pack's boundaries end at its last row, i.e. it has no tail rows past ``cu_seqlens[-1]``.

    Megatron-Core's ``PACKED_THD_GENERAL`` plan scores only the rows its ``cu_seqlens`` cover, while the core
    treats tail rows as part of the last document, so the plan may only be requested for packs without them
    (every ERNIE 5.5 training batch). The last boundary comes from :func:`pack_end`: the host value the ERNIE 5.5
    batch builder notes, or one device read per micro-batch for metadata built elsewhere.
    """
    return pack_end(packed_seq_params) == rows


@contextlib.contextmanager
def _cudnn_indexer_score_scale(scale: float | None) -> Iterator[None]:
    """Set Megatron-Core's cuDNN indexer score scale (``dsa_cudnn_kernels._INDEXER_SOFTMAX_SCALE``) for a block.

    The cuDNN DSA hooks read it at call time for the top-k scores, the KL inputs and the indexer backward.
    ``None`` leaves it alone.
    """
    if scale is None:
        yield
        return
    from megatron.core.transformer.experimental_attention_variant import dsa_cudnn_kernels

    saved = dsa_cudnn_kernels._INDEXER_SOFTMAX_SCALE
    dsa_cudnn_kernels._INDEXER_SOFTMAX_SCALE = scale
    try:
        yield
    finally:
        dsa_cudnn_kernels._INDEXER_SOFTMAX_SCALE = saved


def _requested_indexer_scoring_plan(hook_kwargs: dict) -> str:
    """Scoring plan the packed-layout arguments of one fused top-k hook call select in Megatron-Core's cuDNN backend.

    Without packed metadata (SBHD batches) ERNIE's lagging bounds are not causal, so the backend resolves
    ``UNFUSED_BOUNDS``. With it (packed THD rows: ``use_local_indexer_varlen=True``, identity key positions, one
    ``cu_seqlens`` tensor for queries and keys, ``sk = CP * sq``) it resolves ``PACKED_THD_GENERAL``,
    unless ``single_packed_thd_sequence=True`` selects the single-sequence segment scorer.
    Megatron-Core still falls back to ``UNFUSED_BOUNDS`` when the installed cuDNN frontend lacks the
    per-segment query offsets ``PACKED_THD_GENERAL`` needs at CP > 1, and then logs that itself; the GPU
    tests pin the executor that actually runs.

    Args:
        hook_kwargs: The keyword arguments the hook was called with.
    """
    if not hook_kwargs.get("use_local_indexer_varlen") or hook_kwargs.get("packed_seq_params") is None:
        return INDEXER_SCORING_UNFUSED_BOUNDS
    if hook_kwargs.get("single_packed_thd_sequence"):
        return _INDEXER_SCORING_PACKED_SINGLE_SEGMENTS
    return INDEXER_SCORING_PACKED_THD_GENERAL


def indexer_loss_layer_number(config: TransformerConfig, layer_number: int, is_mtp_layer: bool) -> int:
    """Return the indexer-loss tracker slot of a D layer.

    Decoder layers log at their hybrid position; MTP layers are shifted past the decoder by
    ``config.num_layers``, the Megatron-Core DSA convention. Every MTP depth's inner stack
    restarts at layer 1, so with ``mtp_num_layers >= 2`` the depths share their slots (also the
    Megatron-Core convention).

    Args:
        config: Layer config (``num_layers`` is the decoder's hybrid layer count).
        layer_number: 1-based hybrid position of the layer in its stack.
        is_mtp_layer: Whether the layer belongs to a multi-token-prediction stack.

    Returns:
        1-based tracker slot.
    """
    return layer_number + config.num_layers if is_mtp_layer else layer_number


def _param_device(config: TransformerConfig) -> torch.device | str:
    """Device for parameters this module allocates itself, chosen like Megatron-Core's TE modules."""
    if config.use_cpu_initialization:
        return "cpu"
    if config.init_model_with_meta_device:
        return "meta"
    if not torch.cuda.is_available():
        return "cpu"
    return torch.device("cuda", torch.cuda.current_device())


def _check_fused_geometry(config: TransformerConfig) -> None:
    """Raise if the D-layer geometry cannot run on FlashMLA and the cuDNN DSA kernels."""
    problems = []
    if config.kv_lora_rank != _FLASH_MLA_KV_LORA_RANK:
        problems.append(f"kv_lora_rank={config.kv_lora_rank} (needs {_FLASH_MLA_KV_LORA_RANK})")
    if config.qk_pos_emb_head_dim != _FLASH_MLA_ROPE_DIM:
        problems.append(f"qk_pos_emb_head_dim={config.qk_pos_emb_head_dim} (needs {_FLASH_MLA_ROPE_DIM})")
    if config.num_attention_heads not in _FLASH_MLA_HEAD_COUNTS:
        problems.append(f"num_attention_heads={config.num_attention_heads} (needs one of {_FLASH_MLA_HEAD_COUNTS})")
    if config.dsa_indexer_head_dim != _CUDNN_INDEXER_HEAD_DIM:
        problems.append(f"dsa_indexer_head_dim={config.dsa_indexer_head_dim} (needs {_CUDNN_INDEXER_HEAD_DIM})")
    if config.dsa_indexer_n_heads not in _CUDNN_INDEXER_HEAD_COUNTS:
        problems.append(
            f"dsa_indexer_n_heads={config.dsa_indexer_n_heads} (needs one of {_CUDNN_INDEXER_HEAD_COUNTS})"
        )
    if problems:
        raise ValueError(
            "The fused ERNIE 5.5 D-layer path (dsa_kernel_backend='cudnn') does not support this geometry: "
            + "; ".join(problems)
            + ". Use dsa_kernel_backend='none' for the PyTorch reference path."
        )


def _check_fused_attention_inputs(query: Tensor, key: Tensor) -> None:
    """Raise unless the fused sparse attention can take ``query`` and ``key`` (bf16 CUDA tensors)."""
    if not query.is_cuda or query.dtype != torch.bfloat16 or key.dtype != torch.bfloat16:
        raise RuntimeError(
            "The fused ERNIE 5.5 D-layer path needs bf16 CUDA query and key tensors, got "
            f"{query.dtype} on {query.device}. Use dsa_kernel_backend='none' for the reference path."
        )


class _FlashMLASinkGradFallback(torch.autograd.Function):
    """FlashMLA sparse forward with a sink, cuDNN backward with an analytic sink gradient.

    Megatron-Core's native path (``csa_sparse_attn``) lets the cuDNN backward differentiate the
    sink, which Megatron-Core has only run at ``d_qk = 512``; ERNIE runs ``d_qk = 576``. This
    function instead mirrors Megatron-Core DSA's ``d_qk = 576`` sparse attention: index lists
    with per-row lengths (``topk_length``), the list width padded with ``-1`` to a multiple
    of 512 for FlashMLA, the ignored suffix replaced by non-negative placeholders before the
    backward, and the cuDNN backward run with the sink disabled (``-inf``). Invalid slots inside a
    row's length (the CP-1 lists keep the ``-1`` holes of their key blocks) stay ``-1``. The only additions are
    the finite sink in the FlashMLA forward and the sink folded into the log-sum-exp the backward
    receives (``lse_full = logaddexp(lse, sink)``), plus the analytic
    ``d_sink = -sum_rows exp(sink - lse_full) * rowsum(O * dO)``. It runs with
    ``ernie_native_sink_grad=False``. The native gradient is the default: it matched this function
    and an autograd reference on GB300 (``test_sink_gradient``).

    Inputs and outputs use the flat FlashMLA layout: ``q [rows, H, 576]``, ``kv [rows_kv, 576]``,
    ``topk_idxs [rows, M]`` flat-global int32 indices (``-1`` for an invalid slot),
    ``topk_length [rows]`` int32 lengths (slots past them are ignored), output ``[rows, H, d_v]``.

    With Megatron-Core dev, a deterministic backward is requested through Megatron-Core's
    ``_deterministic_sparse_bwd_kwargs`` (``deterministic_mode`` or
    ``torch.use_deterministic_algorithms``); it raises at the first backward when the GPU, the
    cuDNN frontend (>= 1.28) or the head count cannot provide one.
    """

    @staticmethod
    def forward(
        ctx,
        q: Tensor,
        kv: Tensor,
        attn_sink: Tensor,
        topk_idxs: Tensor,
        topk_length: Tensor,
        softmax_scale: float,
        d_v: int,
        deterministic: bool,
    ) -> Tensor:
        """Run FlashMLA with the sink in the denominator; save tensors for backward."""
        pad = -topk_idxs.size(-1) % _DSA_FLASH_MLA_TOPK_ALIGNMENT
        forward_idxs = F.pad(topk_idxs, (0, pad), value=-1) if pad else topk_idxs
        out, lse, _ = fused_sparse_attention._csa_fwd_flash_mla(
            q, kv, forward_idxs.contiguous(), softmax_scale, d_v=d_v, attn_sink=attn_sink, topk_length=topk_length
        )
        # Slots past topk_length are ignored; the backward only needs non-negative placeholders there. Invalid
        # slots inside a row's length must stay -1.
        slots = torch.arange(topk_idxs.size(-1), device=topk_idxs.device)
        beyond = slots.unsqueeze(0) >= topk_length.unsqueeze(-1)
        backward_idxs = torch.where(beyond, topk_idxs.clamp_min(0), topk_idxs).contiguous()
        ctx.save_for_backward(q, kv, attn_sink, backward_idxs, topk_length, out, lse)
        ctx.softmax_scale = softmax_scale
        ctx.deterministic = deterministic
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        """cuDNN sparse backward with ``lse_full`` and a disabled sink, plus analytic ``d_sink``."""
        q, kv, attn_sink, topk_idxs, topk_length, out, lse = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        lse_full = lse_with_sink(lse, attn_sink=attn_sink).to(lse.dtype).contiguous()
        disabled_sink = torch.full_like(attn_sink, float("-inf"))
        backward_kwargs = {}
        deterministic_kwargs = getattr(fused_sparse_attention, "_deterministic_sparse_bwd_kwargs", None)
        if deterministic_kwargs is not None:
            # Returns {} unless determinism was requested here or through torch's global switch.
            backward_kwargs = deterministic_kwargs(
                q,
                kv,
                out,
                lse_full,
                disabled_sink,
                topk_idxs,
                topk_length,
                ctx.softmax_scale,
                requested=ctx.deterministic,
            )
        grad_q, grad_kv, _ = fused_sparse_attention._csa_sparse_attention_backward(
            q,
            kv,
            out,
            grad_out,
            lse_full,
            disabled_sink,
            topk_idxs,
            softmax_scale=ctx.softmax_scale,
            topk_length=topk_length,
            **backward_kwargs,
        )
        grad_sink = attention_sink_grad(attn_sink=attn_sink, lse=lse, out=out, d_out=grad_out).to(attn_sink.dtype)
        return grad_q, grad_kv, grad_sink, None, None, None, None, None


@dataclass
class Ernie55LatentMQACoreSubmodules:
    """Submodules of :class:`Ernie55LatentMQACoreAttention`.

    Args:
        indexer: Megatron-Core ``DSAIndexer`` spec (``linear_wq_b``, ``linear_wk``, ``k_norm``,
            ``linear_weights_proj``).
        linear_output_gate: Linear ``q_lora_rank -> heads * v_head_dim`` for the output gate.
            Built like the indexer linears (``parallel_mode="duplicated"``, no bias).
    """

    indexer: ModuleSpec | type | None = None
    linear_output_gate: ModuleSpec | type | None = None


class Ernie55LatentMQACoreAttention(MegatronModule):
    """ERNIE 5.5 D-layer core: indexer top-k, windowed sparse latent MQA with a sink, V-up, VHA, gate.

    The constructor mirrors Megatron-Core ``DSAttention`` because ``Attention.__init__``
    builds the core once without ``k_channels``/``v_channels``/``is_mtp_layer`` and
    ``AbsorbedMLASelfAttention`` then rebuilds it with them.

    Parameters owned by the module (names are pinned by the checkpoint mappings):

    * ``attn_sink`` ``[H]`` fp32, zero init: per-head sink logit (denominator only).
    * ``vha_postmix_u`` ``[H, R]`` fp32, ``init_method`` init: VHA input-head factor.
    * ``vha_postmix_v`` ``[H, R]`` fp32, zero init: VHA output-head factor (identity at init).
    * ``linear_output_gate.weight`` ``[H * v_head_dim, q_lora_rank]``.
    * ``indexer.{linear_wq_b, linear_wk, k_norm, linear_weights_proj}`` (Megatron-Core names).
      The ``k_norm`` weight and bias are stored in fp32, like the reference implementation;
      ``DSAIndexer`` applies them with fp32 math (``dsa_indexer_k_norm_fp32``).

    ``H`` is ``num_attention_heads`` and ``R`` is ``ernie_vha_rank``. The fp32 parameters are
    marked ``keep_in_fp32`` so ``Float16Module`` leaves them in fp32.

    ``deterministic_mode`` on the fused path: the sparse attention backward is deterministic, and
    raises at the first backward when the GPU (SM100 family), the cuDNN frontend (>= 1.28) or the
    head count cannot provide it. A Megatron-Core build without the deterministic backward only
    warns and falls back to atomics. The cuDNN indexer backward always accumulates with atomics.

    Args:
        config: D-layer config (see ``ernie55_d_layer_config``).
        submodules: Indexer and output-gate specs.
        layer_number: 1-based hybrid position of the layer.
        attn_mask_type: Must be ``AttnMaskType.causal``.
        attention_type: Must be ``"self"``.
        attention_dropout: Must be ``None`` or 0; the D layer has no attention dropout.
        softmax_scale: Attention logit scale; defaults to ``1 / sqrt(k_channels)``.
        k_channels: Query/key width (``kv_lora_rank + qk_pos_emb_head_dim``).
        v_channels: Latent value width (``kv_lora_rank``).
        cp_comm_type: Unused; context parallelism always all-gathers the keys.
        pg_collection: Process groups; ``tp`` must have size 1. A ``cp`` group of size > 1 needs
            Megatron-Core's CP partition converter (dev) and zigzag packed THD input.
        is_mtp_layer: Shift the indexer-loss tracker slot past the decoder layers.

    Attributes:
        requested_indexer_scoring_plan: How the last fused forward selected the attention's top-k:
            :data:`INDEXER_SCORING_REFERENCE_ORDER` (the core's own CP-1 scorer), or the Megatron-Core
            plan that the arguments of the last top-k hook call select
            (:data:`INDEXER_SCORING_PACKED_THD_GENERAL` for packed THD rows,
            :data:`INDEXER_SCORING_UNFUSED_BOUNDS` for SBHD batches); ``None`` before the first call.
            Megatron-Core can still run ``UNFUSED_BOUNDS`` instead of ``PACKED_THD_GENERAL`` (a cuDNN
            frontend without per-segment query offsets), which it logs itself.
        requested_indexer_loss_scoring_plan: The Megatron-Core plan that the arguments of the last
            loss-hook call select, or ``None`` before the first one.
    """

    consumes_absorbed_v_up_projection = True
    requires_dsa_inputs = True
    logs_dsa_indexer_loss = True

    def __init__(
        self,
        config: TransformerConfig,
        submodules: Ernie55LatentMQACoreSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: float | None = None,
        softmax_scale: float | None = None,
        k_channels: int | None = None,
        v_channels: int | None = None,
        cp_comm_type: str | None = "p2p",
        pg_collection: ProcessGroupCollection | None = None,
        is_mtp_layer: bool = False,
    ) -> None:
        super().__init__(config=config)
        if attention_type != "self":
            raise ValueError(f"ERNIE 5.5 D-layer core attention only supports self attention, got {attention_type!r}.")
        if attn_mask_type is not None and attn_mask_type != AttnMaskType.causal:
            raise ValueError(f"ERNIE 5.5 D-layer core attention is causal, got attn_mask_type={attn_mask_type}.")
        dropout = config.attention_dropout if attention_dropout is None else attention_dropout
        if dropout:
            raise ValueError(f"ERNIE 5.5 D layers have no attention dropout, got {dropout}.")
        if config.q_lora_rank is None:
            raise ValueError("ERNIE 5.5 D layers gate on the q latent and need q_lora_rank.")
        vha_rank = getattr(config, "ernie_vha_rank", None)
        if not isinstance(vha_rank, int) or isinstance(vha_rank, bool) or vha_rank <= 0:
            raise ValueError(f"ERNIE 5.5 D layers need a positive integer ernie_vha_rank, got {vha_rank!r}.")

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
        self.pg_collection = pg_collection
        self._check_parallel_sizes()

        self.layer_number = indexer_loss_layer_number(config, layer_number, is_mtp_layer)
        self.num_heads = config.num_attention_heads
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.window_size = config.csa_window_size
        self.index_topk = config.dsa_indexer_topk
        self.loss_coeff = config.dsa_indexer_loss_coeff or 0.0
        self.vha_bf16_mix = bool(getattr(config, "ernie_vha_bf16_mix", True))
        self.output_gate_fp32 = bool(getattr(config, "ernie_output_gate_fp32", False))
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(k_channels if k_channels is not None else config.kv_channels)
        self.softmax_scale = softmax_scale
        if config.calculate_per_token_loss and self.loss_coeff > 0:
            _warn_once(
                "per_token_indexer_loss",
                "calculate_per_token_loss=True: the ERNIE 5.5 D-layer indexer KL follows Megatron-Core's "
                "per-token convention (raw KL sum, divided by the global loss-token count in "
                "finalize_model_grads), not the reference implementation's mean over all b * s rows. Use "
                "calculate_per_token_loss=False for parity with the reference implementation.",
            )

        self.use_fused = dsa_kernels.use_fused_dsa_kernels(config)
        self.requested_indexer_scoring_plan: str | None = None
        self.requested_indexer_loss_scoring_plan: str | None = None
        # The reference implementation's score convention (module docstring): weights times d^0.5 in the
        # activation dtype, scores times d^-0.5 after the head reduction.
        self.indexer_weight_scale = config.dsa_indexer_head_dim**0.5
        self.indexer_score_scale = config.dsa_indexer_head_dim**-0.5
        # Only the cuDNN hooks take a score scale (through dsa_cudnn_kernels); other fused backends get
        # Megatron-Core's pre-scaled weights.
        self._hooks_take_score_scale = getattr(config, "dsa_kernel_backend", None) == "cudnn"
        self.native_sink_grad = bool(getattr(config, "ernie_native_sink_grad", True))
        self.deterministic = bool(getattr(config, "deterministic_mode", False))
        self._csa_sparse_attn_kwargs = {}
        if self.use_fused:
            _check_fused_geometry(config)
            if "deterministic" in inspect.signature(fused_sparse_attention.csa_sparse_attn).parameters:
                self._csa_sparse_attn_kwargs["deterministic"] = self.deterministic
            if self.deterministic and not hasattr(fused_sparse_attention, "_deterministic_sparse_bwd_kwargs"):
                _warn_once(
                    "deterministic",
                    "deterministic_mode is set, but this Megatron-Core build has no deterministic sparse-attention "
                    "backward; the ERNIE 5.5 D-layer attention backward accumulates with atomics.",
                )

        self.indexer = build_module(submodules.indexer, config=config, pg_collection=self.pg_collection)
        self._keep_indexer_k_norm_in_fp32()
        self.linear_output_gate = build_module(
            submodules.linear_output_gate,
            config.q_lora_rank,
            self.num_heads * self.v_head_dim,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )

        device = _param_device(config)
        self.attn_sink = torch.nn.Parameter(torch.zeros(self.num_heads, dtype=torch.float32, device=device))
        self.vha_postmix_u = torch.nn.Parameter(
            torch.zeros(self.num_heads, vha_rank, dtype=torch.float32, device=device)
        )
        self.vha_postmix_v = torch.nn.Parameter(
            torch.zeros(self.num_heads, vha_rank, dtype=torch.float32, device=device)
        )
        for param in (self.attn_sink, self.vha_postmix_u, self.vha_postmix_v):
            mark_keep_in_fp32(param)
        if config.perform_initialization:
            with torch.no_grad():
                config.init_method(self.vha_postmix_u)

    def _keep_indexer_k_norm_in_fp32(self) -> None:
        """Store the indexer key-norm weight and bias in fp32 (``keep_in_fp32``), as the reference does.

        ``DSAIndexer`` applies ``k_norm`` with fp32 math on ``.float()`` copies of its parameters
        (``dsa_indexer_k_norm_fp32``), so only the stored precision changes: checkpoint values and
        optimizer updates are no longer rounded to the activation dtype.
        """
        for name in ("weight", "bias"):
            param = getattr(self.indexer.k_norm, name, None)
            if param is not None:
                param.data = param.data.float()
                mark_keep_in_fp32(param)

    def _check_parallel_sizes(self) -> None:
        """Reject tensor parallelism, and context parallelism on Megatron-Core builds without the CP converter."""
        tp_size = get_pg_size(getattr(self.pg_collection, "tp", None))
        if tp_size != 1:
            raise NotImplementedError(f"ERNIE 5.5 D layers require tp size 1, got {tp_size}.")
        if get_pg_size(getattr(self.pg_collection, "cp", None)) > 1:
            context_parallel.require_cp_layout_api("ERNIE 5.5 D layers")

    def _indexer_loss_coeff(self, cp_size: int) -> float:
        """Coefficient handed to the loss kernels: ``coeff / CP`` for per-row loss under context parallelism.

        Each rank's per-row loss is then normalized by the rows of the whole pack, not by its own
        rows (see the module docstring).
        """
        if cp_size > 1 and not self.config.calculate_per_token_loss:
            return self.loss_coeff / cp_size
        return self.loss_coeff

    def backward_dw(self) -> None:
        """Run deferred weight-gradient computation of the indexer and the output gate."""
        self.indexer.backward_dw()
        gate_backward_dw = getattr(self.linear_output_gate, "backward_dw", None)
        if gate_backward_dw is not None:
            gate_backward_dw()

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor | None,
        attention_mask: Tensor | None,
        x: Tensor,
        qr: Tensor,
        position_ids: Tensor | None = None,
        attn_mask_type: AttnMaskType | None = None,
        attention_bias: Tensor | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        up_v_weight: Tensor | None = None,
    ) -> Tensor:
        """Run the D-layer core.

        ``attention_mask`` and ``position_ids`` are ignored: the layer is causal within each
        document, and the documents come from ``packed_seq_params`` (one per sample without
        THD packing).

        Under context parallelism the inputs are this rank's rows of a packed THD micro-batch in
        the zigzag partition (``packed_seq_params.cp_partition_mode == "zigzag"``); the keys of
        the whole pack are gathered over the CP group (see the module docstring).

        Args:
            query: Absorbed query ``[s, b, H, kv_lora + rope]`` or packed ``[T, H, kv_lora + rope]``.
            key: Latent key ``[s, b, 1, kv_lora + rope]`` or packed ``[T, 1, kv_lora + rope]``.
            value: Must be ``None`` (the value is the latent key's first ``kv_lora_rank`` channels).
            attention_mask: Ignored.
            x: Layer input ``[s, b, hidden]`` (``[T, 1, hidden]`` for THD), used by the indexer.
            qr: q latent ``[s, b, q_lora]`` or packed ``[T, q_lora]``.
            position_ids: Ignored.
            attn_mask_type: ``None`` or ``AttnMaskType.causal``.
            attention_bias: Must be ``None``.
            packed_seq_params: THD document boundaries, or ``None`` for SBHD.
            up_v_weight: Per-head V up-projection ``[H, v_head_dim, kv_lora_rank]``.

        Returns:
            ``[s, b, H * v_head_dim]`` (``[T, 1, H * v_head_dim]`` for THD, the local rows under
            context parallelism).
        """
        if up_v_weight is None:
            raise ValueError("ERNIE 5.5 D-layer core attention needs up_v_weight from AbsorbedMLA.")
        if value is not None:
            raise ValueError("ERNIE 5.5 D-layer core attention reads the value from the latent key; got a value.")
        if attention_bias is not None:
            raise ValueError("ERNIE 5.5 D-layer core attention does not support attention_bias.")
        if attn_mask_type is not None and attn_mask_type != AttnMaskType.causal:
            raise ValueError(f"ERNIE 5.5 D-layer core attention is causal, got attn_mask_type={attn_mask_type}.")
        self._check_parallel_sizes()

        query, key_mqa, key_heads, qr = self._normalize_layout(query, key, qr)
        seqlen, batch, num_heads, _ = query.shape
        if num_heads != self.num_heads:
            raise ValueError(f"Expected {self.num_heads} query heads, got {num_heads}.")
        is_thd = packed_seq_params is not None and packed_seq_params.qkv_format == "thd"
        if is_thd and batch != 1:
            raise ValueError(f"Packed THD inputs must have batch size 1, got {batch}.")

        cp_group = getattr(self.pg_collection, "cp", None)
        geometry = None
        if get_pg_size(cp_group) > 1:
            geometry = context_parallel.zigzag_cp_geometry(
                packed_seq_params,
                cp_group=cp_group,
                local_rows=seqlen,
                device=query.device,
                what="ERNIE 5.5 D-layer core attention",
            )
            # Every local row attends to keys anywhere before it in its document.
            key_mqa = context_parallel.gather_cp_rows(key_mqa, geometry=geometry, cp_group=cp_group)
            key_heads = key_mqa.unsqueeze(2)
        positions = geometry.query_positions if geometry is not None else None

        doc_start = ernie55_document_starts(packed_seq_params, seqlen, query.device, positions=positions)
        starts, ends = ernie55_selection_bounds(doc_start, self.window_size, positions=positions)
        kernel_ends = ernie55_kernel_selection_ends(starts=starts, ends=ends)
        window = ernie55_window_indices(doc_start, window_size=self.window_size, batch_size=batch, positions=positions)

        use_indexer_loss = self.training and torch.is_grad_enabled() and self.loss_coeff > 0
        q_indexer, k_indexer, w_indexer = self.indexer.forward_before_topk(x.detach(), qr.detach(), packed_seq_params)
        if geometry is not None:
            k_indexer = context_parallel.gather_cp_rows(k_indexer, geometry=geometry, cp_group=cp_group)
        topk, indexer_loss = self._select_topk(
            q_indexer,
            k_indexer,
            w_indexer,
            query,
            key_heads,
            starts,
            kernel_ends,
            use_indexer_loss,
            packed_seq_params=packed_seq_params,
            geometry=geometry,
        )
        topk = mask_topk_beyond_end(topk, ends=ends)

        latent_out = self._sparse_attention(
            query, key_mqa, torch.cat((topk, window), dim=-1), block_layout=geometry is None
        )
        projected = torch.einsum("sbhc,hdc->sbhd", latent_out, up_v_weight.to(latent_out.dtype))
        projected = vha_postmix(projected, u=self.vha_postmix_u, v=self.vha_postmix_v, bf16_mix=self.vha_bf16_mix)
        projected = projected.reshape(seqlen, batch, num_heads * self.v_head_dim)
        gate_logits, _ = self.linear_output_gate(qr)
        output = sigmoid_output_gate(projected, gate_logits=gate_logits, fp32=self.output_gate_fp32)

        if use_indexer_loss:
            self._save_indexer_loss(indexer_loss)
            output = dsa.DSAIndexerLossAutoScaler.apply(output, indexer_loss)
        return output

    @staticmethod
    def _normalize_layout(query: Tensor, key: Tensor, qr: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return ``query [s, b, H, D]``, ``key [s, b, D]``, ``key [s, b, 1, D]`` and ``qr [s, b, q_lora]``.

        A packed THD pack is treated as SBHD with ``b = 1``, so local key indices equal packed
        row indices.
        """
        if query.ndim == 3:
            if key.ndim != 3 or key.size(1) != 1:
                raise ValueError(f"Packed query needs a packed single-head key [T, 1, D], got {tuple(key.shape)}.")
            query = query.unsqueeze(1)
            key_heads = key.unsqueeze(1)
        elif query.ndim == 4:
            if key.ndim != 4 or key.size(2) != 1:
                raise ValueError(f"SBHD query needs a single-head key [s, b, 1, D], got {tuple(key.shape)}.")
            key_heads = key
        else:
            raise ValueError(f"query must be 3-D (THD) or 4-D (SBHD), got {query.ndim}-D.")
        if qr.ndim == 2:
            qr = qr.unsqueeze(1)
        if qr.shape[:2] != query.shape[:2]:
            raise ValueError(f"qr rows {tuple(qr.shape[:2])} do not match query rows {tuple(query.shape[:2])}.")
        return query, key_heads.squeeze(2), key_heads, qr

    def _select_topk(
        self,
        q_indexer: Tensor,
        k_indexer: Tensor,
        w_indexer: Tensor,
        query: Tensor,
        key_heads: Tensor,
        starts: Tensor,
        kernel_ends: Tensor,
        use_indexer_loss: bool,
        *,
        packed_seq_params: PackedSeqParams | None = None,
        geometry: context_parallel.ZigzagCpGeometry | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Select the top-k remote keys and, in training, compute the indexer KL loss.

        Under context parallelism (``geometry`` given) the queries are the local rows and the
        keys (``k_indexer``, ``key_heads``) and the bounds are those of the whole pack. Scores use
        the reference implementation's convention on every path (see the module docstring).

        Returns:
            ``(topk, loss)``: int32 ``[b, s, K]`` (``-1`` padded) and the scalar loss, or
            ``None`` when the loss is inactive.
        """
        # The reference's cast point: the indexer weights times d^0.5 in their own dtype (bf16 in training), kept
        # in the autograd graph so the loss gradient reaches w.
        w_scaled = w_indexer * self.indexer_weight_scale
        if self.use_fused:
            result = self._fused_topk(
                q_indexer,
                k_indexer,
                w_scaled if self._hooks_take_score_scale else w_indexer,
                query,
                key_heads,
                starts,
                kernel_ends,
                use_indexer_loss,
                packed_seq_params=packed_seq_params,
                geometry=geometry,
                w_reference=w_scaled,
            )
            if result is not None:
                return result
            hook = "run_fused_qk_topk_with_loss" if use_indexer_loss else "run_fused_qk_topk"
            seqlen, batch = query.shape[:2]
            pack_rows = batch * (geometry.num_rows if geometry is not None else seqlen)
            if pack_rows > REFERENCE_FALLBACK_MAX_TOKENS:
                raise RuntimeError(
                    f"Megatron-Core's {hook} hook declined the ERNIE 5.5 D-layer top-k for {batch} x {seqlen} "
                    f"query rows of a {pack_rows}-row pack. The PyTorch reference fallback materializes "
                    f"O(b * heads * s^2) fp32 tensors and is only allowed up to {REFERENCE_FALLBACK_MAX_TOKENS} "
                    "rows (b * s). Check the cuDNN DSA installation and the indexer geometry, or select the "
                    "reference path explicitly with dsa_kernel_backend='none'."
                )
            _warn_once(
                f"declined:{hook}",
                f"Megatron-Core's {hook} hook declined the ERNIE 5.5 D-layer top-k; using the PyTorch reference "
                f"path (allowed up to {REFERENCE_FALLBACK_MAX_TOKENS} query rows).",
            )
        return self._reference_topk(
            q_indexer,
            k_indexer,
            w_scaled,
            query,
            key_heads,
            starts,
            kernel_ends,
            use_indexer_loss,
            cp_size=geometry.cp_size if geometry is not None else 1,
        )

    @staticmethod
    def _packed_cp_hook_kwargs(
        packed_seq_params: PackedSeqParams, geometry: context_parallel.ZigzagCpGeometry | None, local_rows: int
    ) -> dict:
        """Arguments that select Megatron-Core's ``PACKED_THD_GENERAL`` scorer for a packed THD pack.

        At CP > 1 (``geometry`` given) the zigzag plan; at CP = 1 (``geometry`` is ``None``) the
        same plan with one rank holding every row. The scorer accepts the packed layout only when the
        query and key ``cu_seqlens`` are one tensor (or one view), so the hook gets metadata whose four
        ``cu_seqlens`` fields are the padded query boundaries; the layer is self attention and the
        boundaries were checked to be identical. ``single_packed_thd_sequence`` stays ``False`` even
        for one-document packs: Megatron-Core's single-sequence segment scorer picks wrong top-k with
        cuDNN frontend >= 1.26 (Megatron-Core #8534) and ignores per-row ends.
        """
        cu_seqlens, _ = get_packed_qk_cu_seqlens(packed_seq_params)
        hook_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            cu_seqlens_q_padded=cu_seqlens,
            cu_seqlens_kv_padded=cu_seqlens,
            max_seqlen_q=packed_seq_params.max_seqlen_q,
            max_seqlen_kv=packed_seq_params.max_seqlen_kv,
        )
        return dict(
            use_local_indexer_varlen=True,
            single_packed_thd_sequence=False,
            local_packed_cp_rank=geometry.cp_rank if geometry is not None else 0,
            local_packed_cp_query_start=0,
            local_packed_cp_query_len=local_rows,
            packed_seq_params=hook_params,
            cp_size=geometry.cp_size if geometry is not None else 1,
            varlen_is_plain_causal=False,
        )

    def _run_topk_hook(self, hook, *args, **hook_kwargs):
        """Call a Megatron-Core fused top-k hook and record the scoring plan its keyword arguments select.

        Both hooks go through here, so :attr:`requested_indexer_scoring_plan` describes the arguments
        the hook that ran actually received. The cuDNN hooks run with the reference's score scale
        ``d^-0.5`` (their weights are ``w * d^0.5``); see the module docstring.

        Returns:
            The hook's result, ``None`` when it declines.
        """
        with _cudnn_indexer_score_scale(self.indexer_score_scale if self._hooks_take_score_scale else None):
            result = hook(self.config, *args, **hook_kwargs)
        if result is not None:
            self._record_scoring_plan(_requested_indexer_scoring_plan(hook_kwargs))
        return result

    def _record_scoring_plan(self, plan: str) -> None:
        """Record the scoring plan of the last fused top-k call and log it once per plan."""
        self.requested_indexer_scoring_plan = plan
        cp_size = get_pg_size(getattr(self.pg_collection, "cp", None))
        if plan == INDEXER_SCORING_UNFUSED_BOUNDS:
            _warn_once(
                f"scoring_plan:{plan}",
                "ERNIE 5.5 D layers score remote indexer bounds [doc_start, t - W] with Megatron-Core's generic "
                f"UNFUSED_BOUNDS cuDNN plan at context_parallel_size={cp_size} (dense fp32 per-head chunks, "
                "~2*T^2*heads*head_dim FLOPs per layer); expect the indexer to dominate step time at long sequence "
                "lengths.",
            )
        elif plan == INDEXER_SCORING_PACKED_THD_GENERAL:
            _warn_once(
                f"scoring_plan:{plan}",
                "ERNIE 5.5 D layers request Megatron-Core's PACKED_THD_GENERAL cuDNN plan for the remote indexer "
                f"bounds at context_parallel_size={cp_size}; Megatron-Core logs it if it falls back to UNFUSED_BOUNDS "
                "(a cuDNN frontend without q_causal_offsets).",
                level=logging.INFO,
            )
        elif plan == INDEXER_SCORING_REFERENCE_ORDER:
            _warn_once(
                f"scoring_plan:{plan}",
                "ERNIE 5.5 D layers at context_parallel_size=1 select their top-k as the reference implementation "
                "does (cuDNN indexer scores per 512-row chunk, torch.topk) and keep its key order.",
                level=logging.INFO,
            )
        else:
            _warn_once(
                f"scoring_plan:{plan}",
                f"ERNIE 5.5 D layers requested Megatron-Core's {plan} indexer scoring plan at "
                f"context_parallel_size={cp_size}; only UNFUSED_BOUNDS and PACKED_THD_GENERAL are validated.",
            )

    def _fused_topk(
        self,
        q_indexer: Tensor,
        k_indexer: Tensor,
        w_indexer: Tensor,
        query: Tensor,
        key_heads: Tensor,
        starts: Tensor,
        kernel_ends: Tensor,
        use_indexer_loss: bool,
        *,
        packed_seq_params: PackedSeqParams | None = None,
        geometry: context_parallel.ZigzagCpGeometry | None = None,
        w_reference: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None] | None:
        """Top-k (and loss) on the fused path; ``None`` when a hook declines.

        At CP = 1 the attention's top-k comes from :meth:`_reference_order_topk` (when cuDNN can
        score the inputs); the loss hook still runs for the loss, and its top-k, the same keys in
        key-id order, is discarded. Otherwise the hooks' top-k is the attention's. ``w_indexer`` are
        the weights the hooks take, ``w_reference`` the reference-convention weights (``w * d^0.5``).
        """
        cp_size = geometry.cp_size if geometry is not None else 1
        is_thd = packed_seq_params is not None and packed_seq_params.qkv_format == "thd"
        packed = is_thd and (geometry is not None or _pack_covers_rows(packed_seq_params, query.size(0)))
        hook_kwargs = self._packed_cp_hook_kwargs(packed_seq_params, geometry, query.size(0)) if packed else {}
        reference_topk = None
        if geometry is None and w_reference is not None:
            reference_topk = self._reference_order_topk(q_indexer, k_indexer, w_reference, starts, kernel_ends)
        if use_indexer_loss:
            result = self._fused_topk_with_loss(
                q_indexer, k_indexer, w_indexer, query, key_heads, starts, kernel_ends, cp_size, hook_kwargs
            )
            if result is None:
                return None
            self.requested_indexer_loss_scoring_plan = self.requested_indexer_scoring_plan
            if reference_topk is None:
                return result
            self._record_scoring_plan(INDEXER_SCORING_REFERENCE_ORDER)
            return reference_topk, result[1]
        if reference_topk is not None:
            self._record_scoring_plan(INDEXER_SCORING_REFERENCE_ORDER)
            return reference_topk, None
        # No indexer loss: evaluation, or the no-grad forward of full recompute (whose replay takes the
        # loss hook above with the same packed arguments).
        result = self._run_topk_hook(
            dsa_kernels.run_fused_qk_topk,
            q_indexer,
            k_indexer,
            w_indexer,
            self.index_topk,
            starts,
            kernel_ends,
            _FUSED_TOPK_BLOCK_SIZE,
            use_relu=self.config.dsa_indexer_scoring_relu,
            **hook_kwargs,
        )
        if result is None:
            return None
        topk, _topk_length = result
        return pad_topk(topk, self.index_topk), None

    def _reference_order_topk(
        self, q_indexer: Tensor, k_indexer: Tensor, w_indexer: Tensor, starts: Tensor, ends: Tensor
    ) -> Tensor | None:
        """The reference implementation's top-k at CP = 1: its keys, in its order.

        Step for step the reference implementation's selection, so every call has the reference's shapes: for each
        chunk of 512 query rows, cuDNN's ``indexer_forward_wrapper`` scores every key (``ratio=1``,
        ``sm_scale=d^-0.5``, the chunk's first row as ``q_causal_offsets`` after the first chunk), the keys outside
        ``[starts, ends)`` are set to ``-inf``, and ``torch.topk`` keeps the ``K`` best in descending score order;
        non-finite picks become ``-1``. A row whose range holds only the placeholder candidate selects it, and the
        caller's :func:`mask_topk_beyond_end` removes it.

        Args:
            q_indexer: ``[s, b, H, d]`` indexer queries.
            k_indexer: ``[s, b, d]`` indexer keys.
            w_indexer: ``[s, b, H]`` weights in the reference convention (``w * d^0.5``).
            starts: int32 ``[s]`` candidate starts.
            ends: int32 ``[s]`` candidate ends (exclusive).

        Returns:
            int32 ``[b, s, K]`` (``-1`` padded), or ``None`` when cuDNN cannot score the inputs.
        """
        scorer = _reference_indexer_scorer(q_indexer)
        if scorer is None:
            return None
        seqlen, batch = q_indexer.shape[:2]
        num_keys = k_indexer.size(0)
        device = q_indexer.device
        budget = min(self.index_topk, num_keys)
        starts = starts.to(device=device, dtype=torch.int64)
        ends = ends.to(device=device, dtype=torch.int64)
        cols = torch.arange(num_keys, device=device)
        picks = []
        with torch.no_grad():
            keys = k_indexer.transpose(0, 1).unsqueeze(2).contiguous()
            for start in range(0, seqlen, _REFERENCE_SCORE_CHUNK_ROWS):
                end = min(start + _REFERENCE_SCORE_CHUNK_ROWS, seqlen)
                offsets = None if start == 0 else torch.full((batch,), start, dtype=torch.int32, device=device)
                scores = scorer(
                    q_indexer[start:end].transpose(0, 1).contiguous(),
                    keys,
                    w_indexer[start:end].transpose(0, 1).contiguous(),
                    ratio=1,
                    sm_scale=self.indexer_score_scale,
                    q_causal_offsets=offsets,
                )["scores"][..., :num_keys]
                remote = (cols[None, :] >= starts[start:end, None]) & (cols[None, :] < ends[start:end, None])
                logits = scores.masked_fill(~remote.unsqueeze(0), float("-inf"))
                picked_scores, picked = torch.topk(logits, budget, dim=-1)
                picks.append(torch.where(torch.isfinite(picked_scores), picked, -1))
        return pad_topk(torch.cat(picks, dim=1), self.index_topk)

    def _fused_topk_with_loss(
        self,
        q_indexer: Tensor,
        k_indexer: Tensor,
        w_indexer: Tensor,
        query: Tensor,
        key_heads: Tensor,
        starts: Tensor,
        kernel_ends: Tensor,
        cp_size: int,
        cp_kwargs: dict,
    ) -> tuple[Tensor, Tensor] | None:
        """Top-k and indexer KL through the cuDNN loss hook, one call per sample when ``b > 1``.

        Megatron-Core's batched hook drops the indexer gradient of every sample after the first
        (see the module docstring). Each per-sample call reduces over its own ``s`` rows, so the
        per-sample losses combine like the batched loss: their mean is ``coeff * sum KL / (b * s)``
        (per-row loss) and their sum is the raw ``coeff * sum KL`` (per-token loss). Autograd scales
        each call's precomputed indexer gradients by the same factor. Packed THD micro-batches
        (``b = 1``) pass the packed hook arguments ``cp_kwargs``; context parallelism also passes
        ``coeff / CP`` (per-row loss).

        Returns:
            ``(topk [b, s, K], loss)``, or ``None`` when the hook declines.
        """
        inputs = (q_indexer, k_indexer, w_indexer, query.detach(), key_heads.detach())
        batch = query.size(1)
        samples = zip(*(tensor.split(1, dim=1) for tensor in inputs)) if batch > 1 else (inputs,)
        topks, losses = [], []
        for q_sample, k_sample, w_sample, query_sample, key_sample in samples:
            result = self._run_topk_hook(
                dsa_kernels.run_fused_qk_topk_with_loss,
                q_sample,
                k_sample,
                w_sample,
                self.index_topk,
                starts,
                kernel_ends,
                _FUSED_TOPK_BLOCK_SIZE,
                query=query_sample,
                key=key_sample,
                softmax_scale=self.softmax_scale,
                loss_coeff=self._indexer_loss_coeff(cp_size),
                pg_collection=self.pg_collection,
                query_valid_rows=None,
                calculate_per_token_loss=self.config.calculate_per_token_loss,
                use_relu=self.config.dsa_indexer_scoring_relu,
                **cp_kwargs,
            )
            if result is None:
                return None
            topk, _topk_length, loss = result
            topks.append(pad_topk(topk, self.index_topk))
            losses.append(loss)
        if batch == 1:
            return topks[0], losses[0]
        losses = torch.stack(losses)
        loss = losses.sum() if self.config.calculate_per_token_loss else losses.mean()
        return torch.cat(topks, dim=0), loss

    def _reference_topk(
        self,
        q_indexer: Tensor,
        k_indexer: Tensor,
        w_indexer: Tensor,
        query: Tensor,
        key_heads: Tensor,
        starts: Tensor,
        kernel_ends: Tensor,
        use_indexer_loss: bool,
        cp_size: int = 1,
    ) -> tuple[Tensor, Tensor | None]:
        """Top-k (and loss) with Megatron-Core's PyTorch DSA reference, called by keyword.

        Under context parallelism the queries are the local rows, the keys and bounds those of the
        whole pack, and the loss coefficient ``coeff / CP`` (per-row loss). ``w_indexer`` are the
        reference-convention weights (``w * d^0.5``); the scores the loss reads are scaled by
        ``d^-0.5`` after the head reduction, as cuDNN's kernel does (the top-k does not depend on it).
        """
        with nullcontext() if use_indexer_loss else torch.no_grad():
            index_scores, topk = dsa.fused_qk_topk_naive(
                q_indexer,
                k_indexer,
                w_indexer,
                self.index_topk,
                varlen_starts=starts,
                varlen_ends=kernel_ends,
                use_relu=self.config.dsa_indexer_scoring_relu,
            )
        loss = None
        if use_indexer_loss:
            loss = dsa.compute_dsa_indexer_loss(
                index_scores=index_scores * self.indexer_score_scale,
                topk_indices=topk,
                query=query.detach(),
                key=key_heads.detach().expand(-1, -1, query.size(2), -1),
                softmax_scale=self.softmax_scale,
                loss_coeff=self._indexer_loss_coeff(cp_size),
                sparse_loss=True,
                pg_collection=self.pg_collection,
                varlen_starts=starts,
                varlen_ends=kernel_ends,
                query_valid_rows=None,
                calculate_per_token_loss=self.config.calculate_per_token_loss,
            )
        return pad_topk(topk, self.index_topk), loss

    def _sparse_attention(
        self, query: Tensor, key_mqa: Tensor, indices: Tensor, *, block_layout: bool = False
    ) -> Tensor:
        """Sparse latent MQA with the sink over per-sample ``indices [b, s, K + W]`` into ``key_mqa``.

        ``key_mqa`` is ``[s_kv, b, D]``: the sample's keys, or the gathered keys of the whole pack
        under context parallelism (``b = 1``), so ``s_kv`` may exceed the ``s`` query rows.

        On the fused path ``block_layout`` (CP = 1) hands FlashMLA the rows without their all-invalid
        key blocks (:func:`drop_invalid_key_blocks`), keeping every other block's keys and slots;
        otherwise (CP > 1) the valid keys are compacted to the front of each row.

        Returns:
            ``[s, b, H, kv_lora_rank]`` in the activation dtype.
        """
        seqlen, batch, num_heads, head_dim = query.shape
        if not self.use_fused:
            # Megatron-Core's unfused path gathers the key with index_select, whose backward adds every
            # query row's key gradient in the key's dtype, so a bf16 key loses visible precision in dKV.
            # A key of at least fp32 sums in fp32 and rounds once. The forward is unchanged (the unfused
            # path upcasts the gathered keys anyway).
            key_mqa = key_mqa.to(torch.promote_types(key_mqa.dtype, torch.float32))
            out = csa.unfused_compressed_sparse_attn(query, key_mqa, self.attn_sink, indices, self.softmax_scale)
            return out.view(seqlen, batch, num_heads, head_dim)[..., : self.kv_lora_rank]

        _check_fused_attention_inputs(query, key_mqa)
        if block_layout:
            flat_indices, _ = fused_sparse_attention.build_flat_topk_idxs(indices, batch_size=batch)
            flat_indices, topk_length = drop_invalid_key_blocks(flat_indices, block=_FLASH_MLA_KEY_BLOCK)
        else:
            flat_indices, topk_length = fused_sparse_attention.build_flat_topk_idxs(
                indices, batch_size=batch, compact=True
            )
        sink = self.attn_sink.float()
        if self.native_sink_grad:
            out = fused_sparse_attention.csa_sparse_attn(
                query,
                key_mqa,
                sink,
                flat_indices,
                self.softmax_scale,
                topk_length=topk_length,
                **self._csa_sparse_attn_kwargs,
            )
            return out.reshape(seqlen, batch, num_heads, self.kv_lora_rank)

        out = _FlashMLASinkGradFallback.apply(
            query.reshape(seqlen * batch, num_heads, head_dim),
            key_mqa.reshape(-1, head_dim),
            sink,
            flat_indices,
            topk_length,
            self.softmax_scale,
            self.kv_lora_rank,
            self.deterministic,
        )
        return out.reshape(seqlen, batch, num_heads, self.kv_lora_rank)

    def _save_indexer_loss(self, loss: Tensor) -> None:
        """Log the indexer loss with Megatron-Core's DSA tracker.

        Under context parallelism the tracker averages the per-row loss over the CP group (the
        reference's convention; each rank's value is already normalized by the whole pack, so the
        logged value is ``1 / CP`` of the pack's row mean) and sums the per-token loss.
        """
        cp_group = getattr(self.pg_collection, "cp", None)
        cp_size = get_pg_size(cp_group)
        per_token = self.config.calculate_per_token_loss
        dsa.DSAIndexerLossLoggingHelper.save_loss_to_tracker(
            loss=loss,
            layer_number=self.layer_number,
            num_layers=max(self.layer_number, self.config.num_layers + (self.config.mtp_num_layers or 0)),
            reduce_group=cp_group if cp_size > 1 and per_token else None,
            avg_group=cp_group if cp_size > 1 and not per_token else None,
            raw_loss=loss / self.loss_coeff,
        )
