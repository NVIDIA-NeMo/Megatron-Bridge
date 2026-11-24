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

"""Transformer Engine CUDA graphs of the ERNIE 5.5 attention layers, with fp32 hyper-connection streams.

ERNIE 5.5 supports one CUDA-graph configuration (:func:`validate_ernie55_cuda_graph_config`): Megatron-Core's
per-layer Transformer Engine (TE) graphs, ``cuda_graph_impl="transformer_engine"`` with
``cuda_graph_modules=["attn"]``. Megatron-Core's mHC wrapper (``HyperConnectionHybridLayer``) of a graphed attention
layer is captured whole, forward and backward: the mHC mappings and aggregate, the input norm, the attention and the
n-stream residual update. Which attention layers are graphed:

* the H layers of the decoder, always (DSv4 attention);
* the D layers of the decoder with ``ernie_graph_d_layers`` (off by default; see "D-layer graphs" below);
* the D layer of every MTP depth with ``ernie_graph_mtp_layers`` (off by default; needs ``ernie_graph_d_layers``).

Everything else runs eagerly: the MoE and dense MLP layers, and the D layers whose switch is off (their forward
raises inside a capture).

Three Megatron-Core behaviors do not fit ERNIE 5.5; this module changes them for the graphed wrappers only:

* Static inputs. Megatron-Core sizes a graph's static hidden input as ``[s, b, n*C]`` in the dtype ``config.bf16``
  selects, and builds packed THD inputs only for its sequence-packing scheduler. The ERNIE 5.5 decoder streams are
  fp32 (:class:`~megatron.bridge.models.ernie.modeling_ernie55.hybrid_stack.Ernie55HybridStack`), and TE replays a
  graph with ``static_input.copy_(input)``, which converts dtypes without a check: every graphed layer would round
  its input streams to bf16 and return bf16 streams. ERNIE 5.5 packs its rows in its own step
  (:mod:`~megatron.bridge.models.ernie.ernie55_step`). The graphed wrappers supply fp32 ``[L, 1, n*C]`` hidden
  states (``L = max_seqlen_per_dp_cp_rank``, ``n*C`` the stream width; the MTP depths' streams are in the
  activation dtype) and the four int32 ``cu_seqlens`` of one row of ``thd_max_packed_sequences`` documents.
  Megatron-Core's wrapper rebuilds the ``PackedSeqParams`` from them in capture and decomposes each micro-batch's own
  in replay, so the graph's output and input gradient keep the stream dtype, as in eager execution.
* Discovery. Megatron-Core marks every attention wrapper graphable, the D layers and the nested MTP stack's included.
  :class:`Ernie55TECudaGraphHelper` keeps the decoder's H wrappers, plus the D wrappers the switches turn on. Bridge's
  training loop builds it instead of Megatron-Core's ``TECudaGraphHelper`` because
  :class:`~megatron.bridge.models.ernie.ernie55_config.Ernie55TransformerConfig` names it in its
  ``te_cuda_graph_helper`` class attribute. A pipeline stage without graphed layers captures nothing, as
  Megatron-Core's helper does for stages without graphable layers.
* RoPE tables. The attention layers read their RoPE table from their rotary embeddings, whose ``forward``
  Megatron-Core caches in one 32-entry ``functools.lru_cache`` shared by every rotary embedding of the process. A
  graph reads the table at its capture-time address, but after capture only that cache holds it: at the full
  model depth the eager layers' own lookups evict it within one forward, and ``cache_clear`` (eval CP rebinding,
  ``YarnRotaryEmbedding``'s constructor) drops it at once, so the graph would read freed memory. A graphed wrapper
  keeps every table its rotary embeddings return during capture until Megatron-Core deletes the graphs.

An H graph replays only micro-batches of its capture shape: ``L`` rows on every CP rank, exactly
``thd_max_packed_sequences`` documents (``thd_max_packed_sequences + 1`` boundaries), the longest at most ``L * CP``
rows. Before every replay the H wrapper checks this, and every input TE would silently convert or drop, from shapes
and dtypes alone (no device synchronization). The document boundaries may change from row to row: the DSv4
attention reads them on the device.

D-layer graphs. A D layer plans part of its work on the host, from the micro-batch's host ``cu_seqlens`` (see
:mod:`~megatron.bridge.models.ernie.modeling_ernie55.context_parallel`): at CP > 1 the all-to-all split sizes and
index tensors of its zigzag conversion (Megatron-Core's prebuilt route), the zigzag geometry (the global position of
every local row and the gathered keys' permutation), and at every CP size the ``max_seqlen`` of its RoPE and its
cuDNN indexer plan and, at CP = 1, whether the pack ends at its last row. A capture bakes all of it in. A D graph
therefore assumes one fixed sequence composition, the one the 64-GPU SFT recipe's rows have: one document over the
whole row (``thd_max_packed_sequences=1``). :class:`Ernie55DHyperConnectionHybridLayer` (decoder) and
:class:`Ernie55MTPDHyperConnectionHybridLayer` (MTP depth):

* build that composition's host plan before the capture, when Megatron-Core asks for the static inputs: the route,
  the host boundaries and the zigzag geometry (with its inverse key permutation under
  ``ernie_cp_key_gather_inverse_grad``), built once and kept (pinned) until the graphs are deleted, so their
  device tensors outlive every replay;
* attach the plan to the ``PackedSeqParams`` Megatron-Core rebuilds from the static inputs for every warmup and
  capture call, so the captured D layer finds every host-planned piece and no host-to-device or device-to-host copy
  runs inside the capture (the D attention refuses a capture whose metadata lacks the plan, and the geometry and pack
  end lookups raise under a capture instead of building or reading on the device);
* before every replay, check on the host that the micro-batch has the captured composition: the same compacted host
  boundaries (Megatron-Core's ``thd_cp_host_cu_seqlens_q`` at CP > 1, the batch builder's note at CP = 1), the same
  ``max_seqlen_q`` and ``max_seqlen_kv``, the same number of boundaries in its ``cu_seqlens``. A micro-batch with
  another composition runs the D layer eagerly, with its own routes and geometry, and the first such fallback of a
  process logs a warning. Every rank of a CP group holds the same pack, so all of them take the same path;
* with fine-grained activation offloading, start a forward replay only after every graph replayed before it has
  completed, on the offload manager's graph stream (see "Offloading" below).

Offloading. A D graph has no offload boundary, so it replays on the current stream, while the H graphs, which offload
``core_attn`` and ``attn_proj`` inside the graph, replay on the offload manager's ``cuda_graph_stream``: the current
stream resumes at the H graph's ``cuda_graph_event``, recorded before the graph's tail joins the device-to-host copies
of the activations it offloaded. Those activations were freed into the graphs' shared memory pool during the H graph's
capture, so a D graph captured after it may hold the same memory, and replayed on the current stream it could
overwrite an activation that the H graph's tail is still copying to the host (the backward replays need no such order:
an H graph's tail there only fills reload buffers that stay referenced until their own graph reads them). Before
replaying its forward, a D graph therefore makes the current stream wait for the offload manager's graph stream: a
stream-order dependency only, which exposes the part of the preceding H layer's offload copies that the layers in
between did not hide. Eager D layers (the fallback) allocate outside the graphs' pool and do not wait.

With H layers in the decoder (every ERNIE 5.5 stack) every micro-batch must have the captured composition, one
document over the whole row: the H graphs replay only rows of ``thd_max_packed_sequences`` documents with
``max_seqlen`` at most the row and raise ``ValueError`` for any other micro-batch (see the H graphs above), so it fails
instead of running the D layers eagerly. There the D layers fall back only for a micro-batch the H graphs accept whose
host plan still differs, such as metadata built without the batch builder's host notes. The eager fallback for other
document counts is reached only in a decoder of D layers alone.

The D graphs also capture the indexer loss: its tracker writes go into the fixed-capacity tracker storage (Bridge's
setup initializes it; Megatron-Core's helper requires it before capture), and Megatron-Core cleans the tracker after
the capture keeping its reduction groups (``clean_loss_in_tracker(preserve_groups=True)``; Bridge's logging keeps them
too, since no eager D layer sets them again). ``DSAIndexerLossAutoScaler``'s backward reads its scale tensor in place;
the wrapper creates that tensor on the device before the capture if the schedule has not yet, in the schedule's
size-1 shape, and the schedule updates it in place.

MTP. With SD-MTP (``ernie_mtp_shared_last_layer``) the MTP depth's D layer shares every parameter with the decoder's
last D layer, so the shared parameters' gradient is the sum of the two layers' contributions, added by autograd when
both have arrived, as in eager execution. Transformer Engine reuses (in later graphs) the memory of a graph's
returned gradients once the next graph of the backward order has been captured, so a returned gradient survives only
until the replay after next. With both D layers graphed the decoder's last D graph is the one right after the MTP's
in that order, autograd adds the two contributions as soon as the decoder's arrives, and the sum is the eager one;
:meth:`Ernie55TECudaGraphHelper._discover_layers` checks that every parameter shared by graphs has exactly two
adjacent graphs as users.

The sum is the eager one, but not the number of gradient-ready calls: eagerly, each recomputed use of a shared
parameter (``layernorm`` recompute of the two D layers' input norm) runs its own nested backward, so the parameter's
Megatron-Core DDP hook fires twice per step; graphed, the two graphs return their contributions to autograd, which adds
them and fires the hook once. With ``ddp.overlap_grad_reduce`` a bucket group issues its reduce only when every
parameter's count equals the one recorded in the first (eager) batch, so after the capture that bucket group would
never reduce and ``finish_grad_sync`` would assert ("Communication call has not been issued for this bucket").
:meth:`Ernie55TECudaGraphHelper.create_cudagraphs` therefore makes the first graphed batch record the counts again
(:func:`_relearn_ddp_grad_ready_counts`): that batch reduces after its backward, as the first batch of a run does.

Megatron-Core's mHC selective recompute (``recompute_modules`` with ``"mhc"``) keeps working around the graphs: a
graphed wrapper is captured without its recompute manager, so its mHC producer is not recomputed (Megatron-Core
warns), while the MoE wrappers and the eager attention wrappers keep the saving. Keep
``mhc_recompute_attn_cuda_graph_split`` off: the attention-only split needs Megatron-Core changes for ERNIE 5.5's
recompute set, attention offloading and fp32 aggregate. ``layernorm`` recompute runs inside the graphs and snapshots
the RNG state there, which needs the graph-safe Transformer Engine RNG tracker (``use_te_rng_tracker``).

Memory. The graphs' private memory pool is reserved but not allocated memory after capture: compare a graphed run
with an eager one by ``mem-max-reserved-bytes`` or NVML (``logger.log_device_memory_used``), never by
``mem-max-allocated-bytes``. :class:`Ernie55TECudaGraphHelper` logs each rank's allocator state after capture. With
fine-grained activation offloading inside the graphs (``core_attn``, ``attn_proj``), set
``graph_capture_record_stream_reuse:True`` next to ``expandable_segments:True`` in every allocator variable
(``PYTORCH_ALLOC_CONF`` and ``PYTORCH_CUDA_ALLOC_CONF``; torch 2.9 or later): the offload manager calls
``record_stream`` on the offloaded tensors during capture, and without the option PyTorch's allocator does not reuse
their blocks while a capture is underway. The helper warns when it is missing.

Without CUDA graphs (``cuda_graph_impl="none"``) nothing here runs: no wrapper changes class, the training loop
builds no helper, and the validators return at once.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from megatron.core.models.hybrid.hybrid_block import HyperConnectionHybridLayer
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.cuda_graphs import TECudaGraphHelper
from megatron.core.transformer.module import GraphableMegatronModule
from megatron.core.utils import get_pg_size, log_single_rank
from torch import Tensor

from megatron.bridge.models.ernie.modeling_ernie55 import context_parallel, ops
from megatron.bridge.utils.cuda_graph import cuda_graph_module_names


logger = logging.getLogger(__name__)

__all__ = [
    "D_GRAPH_PLAN_ATTR",
    "Ernie55DHyperConnectionHybridLayer",
    "Ernie55HCAHyperConnectionHybridLayer",
    "Ernie55MTPDHyperConnectionHybridLayer",
    "Ernie55TECudaGraphHelper",
    "check_ernie55_d_layer_capture",
    "use_ernie55_attention_cuda_graphs",
    "use_ernie55_d_layer_cuda_graphs",
    "validate_ernie55_cuda_graph_config",
]

_GRAPH_MODULES = ["attn"]
"""The only supported ``cuda_graph_modules``: the attention wrappers, of which ERNIE 5.5 graphs the H layers."""

_PACKED_FIELDS = ("cu_seqlens_q", "cu_seqlens_kv", "cu_seqlens_q_padded", "cu_seqlens_kv_padded")
"""``PackedSeqParams`` fields an H graph takes as static inputs (Megatron-Core's wrapper decomposes them)."""

_REPLAY_INPUTS = ("hidden_states", "packed_seq_params")
"""Keyword inputs an H graph consumes at replay; every other one must be ``None`` (TE drops unknown inputs)."""

_ALLOC_CONF_VARIABLES = ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF")
"""Environment variables that configure PyTorch's caching allocator (old and new name)."""

_RECORD_STREAM_REUSE = "graph_capture_record_stream_reuse:True"
"""Allocator option that lets PyTorch reuse ``record_stream`` blocks during a graph capture (torch 2.9 or later)."""

_ROPE_TABLES = "_te_cuda_graph_rope_tables"
"""Attribute of an H wrapper holding the RoPE tables its graphs read, by ``id``; absent before the first capture."""

_ROPE_CACHE_BUFFERS = ("cos_cached", "sin_cached")
"""Cos/sin buffers a rotary embedding caches for the fused RoPE path (``apply_rope_fusion``), which graphs also read."""

_D_GRAPH_PLAN = "_te_cuda_graph_d_plan"
"""Attribute of a D wrapper holding the host plan its graphs bake in; absent before Megatron-Core asks for the static
inputs, and after the graphs are deleted."""

D_GRAPH_PLAN_ATTR = "_ernie55_d_graph_plan"
"""Attribute of a ``PackedSeqParams`` a D wrapper rebuilt for a warmup or capture call: the host plan attached to it.
The D attention accepts a capture only for metadata that carries it."""

_DTYPE_NAMES = {torch.float32: "fp32", torch.bfloat16: "bf16", torch.float16: "fp16"}

_WARNED_FALLBACKS: set[str] = set()
"""Reasons for which an eager fallback of a D graph was already logged in this process."""


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _graph_d_layers(config: Any) -> bool:
    """Whether ``config`` asks for graphs of the decoder's D layers (``ernie_graph_d_layers``)."""
    return bool(getattr(config, "ernie_graph_d_layers", False))


def _graph_mtp_layers(config: Any) -> bool:
    """Whether ``config`` asks for graphs of the MTP depths' D layers (``ernie_graph_mtp_layers``)."""
    return bool(getattr(config, "ernie_graph_mtp_layers", False))


def validate_ernie55_cuda_graph_config(config: Any, *, layer: str) -> None:
    """Raise ``ValueError`` if ``config`` enables CUDA graphs outside the ERNIE 5.5 envelope.

    Configs without CUDA graphs (``cuda_graph_impl`` ``"none"`` and no legacy ``enable_cuda_graph``) pass at once,
    unless they set ``ernie_graph_d_layers`` or ``ernie_graph_mtp_layers``, which need the graphs. Otherwise every one
    of these must hold:

    * ``cuda_graph_impl="transformer_engine"`` and not the legacy ``enable_cuda_graph`` flag: ``"local"`` and
      ``"full_iteration"`` would capture the D layers' host-planned context-parallel routes for every micro-batch;
    * ``cuda_graph_modules=["attn"]``: per-layer graphs of the attention wrappers, nothing else (the MoE dispatch
      stays eager);
    * something to graph: ``experimental_attention_variant="dsv4_hybrid"`` (D-only ``"dsa"`` stacks have no H
      layer and run the zigzag partition), an H layer among the decoder's ``csa_compress_ratios`` (checked when they
      are set and ``ernie_graph_d_layers`` is off; the MTP depths have no H layer) and
      ``enable_mhc_connections=True`` (only the mHC wrappers, which
      :class:`~megatron.bridge.models.ernie.modeling_ernie55.hybrid_stack.Ernie55HybridStack` converts, are graphed);
    * ``mhc_recompute_attn_cuda_graph_split=False``: the split needs Megatron-Core changes for ERNIE 5.5's recompute
      set, attention offloading and fp32 mHC aggregate;
    * ``use_te_rng_tracker=True``: ``layernorm`` recompute inside a graph snapshots the RNG state, which only a
      graph-safe tracker allows during capture. Set ``model.use_te_rng_tracker``: the training config syncs
      ``rng.te_rng_tracker`` into the model only after the model config is finalized;
    * at ``pipeline_model_parallel_size > 1``, ``cuda_graph_warmup_steps >= 1``: a stage without graphed layers
      captures nothing, and Bridge's training loop requires graphs on every rank when it captures before the first
      step;
    * ``delay_wgrad_compute=False`` and ``overlap_moe_expert_parallel_comm=False``: Megatron-Core's EP overlap
      schedule does not support ERNIE 5.5's hybrid mHC layers and MTP;
    * ``sequence_packing_scheduler=None``: ERNIE 5.5 packs its rows in its own step (ernie55_step) and its graphed
      layers build their own packed static inputs;
    * ``max_seqlen_per_dp_cp_rank`` and ``thd_max_packed_sequences`` positive ints: the rows of every CP rank and the
      documents of every packed row, which fix the graphs' static inputs;
    * at ``context_parallel_size > 1``, ``cp_partition_mode="contiguous"``, the DSv4 layout: the H layers' opt-in zigzag
      partition (Megatron-Core's ``dsv4_cp_zigzag``) is not validated with CUDA graphs yet.

    With ``ernie_graph_d_layers`` or ``ernie_graph_mtp_layers`` (the D-layer graphs; see the module docstring), also:

    * ``ernie_graph_d_layers=True`` whenever ``ernie_graph_mtp_layers=True``: the MTP depth's D layer shares the
      decoder's last D layer's parameters, and their summed gradient is safe from Transformer Engine's buffer reuse
      only when both are graphed;
    * ``thd_max_packed_sequences=1``: a D graph is captured for one composition, one document over the whole row;
    * with ``ernie_graph_mtp_layers``, an MTP (``mtp_num_layers >= 1``) and, with SD-MTP
      (``ernie_mtp_shared_last_layer``), one depth: more depths would give the shared parameters three graphed users.

    Args:
        config: Stack-level or per-layer transformer config.
        layer: Layer name for the error messages, ``"D"`` or ``"H"``.

    Raises:
        ValueError: On the first violated requirement. The message starts with
            ``"ERNIE 5.5 <layer> layer: CUDA graphs"``.
    """
    impl = getattr(config, "cuda_graph_impl", "none")
    legacy = bool(getattr(config, "enable_cuda_graph", False))
    graph_d, graph_mtp = _graph_d_layers(config), _graph_mtp_layers(config)

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(f"ERNIE 5.5 {layer} layer: CUDA graphs {message}")

    if impl in (None, "none") and not legacy:
        require(
            not graph_d and not graph_mtp,
            "are off (cuda_graph_impl='none'), but ernie_graph_d_layers or ernie_graph_mtp_layers asks for D-layer "
            "graphs; set cuda_graph_impl='transformer_engine' and cuda_graph_modules=['attn'] (with the other H-layer "
            "graph settings), or turn the switches off.",
        )
        return

    supported = (
        "ERNIE 5.5 supports cuda_graph_impl='transformer_engine' with cuda_graph_modules=['attn'], which graphs its "
        "H layers."
    )
    require(
        not legacy,
        f"are not supported through enable_cuda_graph (Megatron-Core's local implementation); {supported}",
    )
    require(impl == "transformer_engine", f"are not supported with cuda_graph_impl={impl!r}; {supported}")
    modules = cuda_graph_module_names(config)
    scope = (
        "only the attention layers are graphed (H, and D with ernie_graph_d_layers); the MoE dispatch runs eagerly."
        if graph_d
        else "only the H attention layers are graphed; the MoE dispatch and the D layers run eagerly."
    )
    require(modules == _GRAPH_MODULES, f"need cuda_graph_modules={_GRAPH_MODULES}, got {modules}: {scope}")
    variant = getattr(config, "experimental_attention_variant", None)
    require(
        variant == "dsv4_hybrid",
        f"need experimental_attention_variant='dsv4_hybrid', got {variant!r}: only the H layers are graphed, and only "
        "DSv4 hybrid stacks have them.",
    )
    ratios = getattr(config, "csa_compress_ratios", None)
    num_layers = getattr(config, "num_layers", None)
    if ratios is not None and _is_positive_int(num_layers) and not graph_d:
        from megatron.bridge.models.ernie.modeling_ernie55.hca_attention import ERNIE55_HCA_COMPRESS_RATIO

        require(
            ERNIE55_HCA_COMPRESS_RATIO in list(ratios)[:num_layers],
            f"need an H layer in the decoder, but none of the first num_layers={num_layers} csa_compress_ratios is "
            f"{ERNIE55_HCA_COMPRESS_RATIO}: only the decoder's H layers are graphed (the MTP depths run eagerly).",
        )
    require(
        bool(getattr(config, "enable_mhc_connections", False)),
        "need enable_mhc_connections=True: only the mHC wrappers of the H layers, which carry the fp32 streams and "
        "Ernie55HybridStack converts, are graphed.",
    )
    require(
        not getattr(config, "mhc_recompute_attn_cuda_graph_split", False),
        "need mhc_recompute_attn_cuda_graph_split=False: the attention-only split does not support ERNIE 5.5's "
        "recompute set, attention offloading or fp32 mHC aggregate; the H wrappers are captured whole.",
    )
    require(
        bool(getattr(config, "use_te_rng_tracker", False)),
        "need use_te_rng_tracker=True (model.use_te_rng_tracker=true; rng.te_rng_tracker reaches the model config "
        "only after it is finalized): layernorm recompute inside a graph snapshots the RNG state, which only a "
        "graph-safe tracker allows during capture.",
    )
    if (getattr(config, "pipeline_model_parallel_size", 1) or 1) > 1:
        warmup = getattr(config, "cuda_graph_warmup_steps", None)
        require(
            _is_positive_int(warmup),
            f"need cuda_graph_warmup_steps >= 1 at pipeline_model_parallel_size > 1, got {warmup!r}: a pipeline stage "
            "without H layers captures nothing, and with no warmup step Bridge's training loop requires graphs on "
            "every rank.",
        )
    require(
        not getattr(config, "delay_wgrad_compute", False)
        and not getattr(config, "overlap_moe_expert_parallel_comm", False),
        "need delay_wgrad_compute=False and overlap_moe_expert_parallel_comm=False: Megatron-Core's EP overlap "
        "schedule does not support ERNIE 5.5's hybrid mHC layers and MTP.",
    )
    require(
        getattr(config, "sequence_packing_scheduler", None) is None,
        "need sequence_packing_scheduler=None: ERNIE 5.5 packs its rows in its own step (ernie55_step) and its H "
        "layers build their own packed static inputs.",
    )
    for name, meaning in (
        ("max_seqlen_per_dp_cp_rank", "the rows of every context-parallel rank (seq_length / context_parallel_size)"),
        ("thd_max_packed_sequences", "the documents of every packed row (graphs replay only rows with that many)"),
    ):
        value = getattr(config, name, None)
        require(_is_positive_int(value), f"need {name}, {meaning}, as a positive int; got {value!r}.")
    if getattr(config, "context_parallel_size", 1) > 1:
        mode = getattr(config, "cp_partition_mode", None)
        require(
            not getattr(config, "dsv4_cp_zigzag", False),
            "are not validated with the H layers' zigzag partition (dsv4_cp_zigzag=True) yet: their static inputs, "
            "replay checks and host plans assume the contiguous partition. Set cuda_graph_impl='none' with "
            "dsv4_cp_zigzag, or cp_partition_mode='contiguous' with dsv4_cp_zigzag=False.",
        )
        require(
            mode == "contiguous",
            f"need cp_partition_mode='contiguous' at context_parallel_size > 1 (the DSv4 layout), got {mode!r}.",
        )
    if not graph_d and not graph_mtp:
        return
    require(
        graph_d,
        "of the MTP depths (ernie_graph_mtp_layers=True) need ernie_graph_d_layers=True: the MTP depth's D layer "
        "shares the decoder's last D layer's parameters, and their summed gradient is safe from Transformer Engine's "
        "graph-buffer reuse only when both layers are graphed.",
    )
    documents = getattr(config, "thd_max_packed_sequences", None)
    require(
        documents == 1,
        f"of the D layers need thd_max_packed_sequences=1, got {documents!r}: a D-layer graph bakes in the host plan "
        "of one composition, one document over the whole row, and replays only micro-batches of that composition.",
    )
    if graph_mtp:
        depths = getattr(config, "mtp_num_layers", None)
        require(
            _is_positive_int(depths),
            f"of the MTP depths (ernie_graph_mtp_layers=True) need an MTP, got mtp_num_layers={depths!r}.",
        )
        require(
            not (getattr(config, "ernie_mtp_shared_last_layer", False) and depths > 1),
            f"of the MTP depths (ernie_graph_mtp_layers=True) support one SD-MTP depth, got mtp_num_layers={depths}: "
            "every depth's D layer shares the decoder's last D layer's parameters, and Transformer Engine reuses a "
            "graph's returned gradients once the graph after next replays, before three graphs' contributions are "
            "summed.",
        )


def _thd_graph_shape(config: Any, kind: str = "H") -> tuple[int, int, int]:
    """``(rows per CP rank, documents per row, rows per row)`` of a graph's static input."""
    local_rows = getattr(config, "max_seqlen_per_dp_cp_rank", None)
    documents = getattr(config, "thd_max_packed_sequences", None)
    if not _is_positive_int(local_rows) or not _is_positive_int(documents):
        raise ValueError(
            f"ERNIE 5.5 {kind}-layer CUDA graphs need max_seqlen_per_dp_cp_rank and thd_max_packed_sequences as "
            f"positive ints, got {local_rows!r} and {documents!r}."
        )
    return local_rows, documents, local_rows * config.context_parallel_size


class _Ernie55GraphedHybridLayer(HyperConnectionHybridLayer):
    """Megatron-Core's mHC wrapper with the static inputs, capture and replay checks of an ERNIE 5.5 TE graph.

    The base of the graphed wrappers; Megatron-Core's wrappers are converted to its subclasses in place by switching
    their class, so it has no constructor and no parameter, buffer or checkpoint key of its own.
    """

    _GRAPH_KIND = "attention"
    """Layer kind in the messages: ``"H"``, ``"D"`` or ``"MTP D"``."""

    def _graph_stream_dtype(self) -> torch.dtype:
        """Dtype of the graph's static hidden states: the streams this wrapper receives (fp32 in the decoder)."""
        return torch.float32

    def get_layer_static_inputs(self, seq_length: int, micro_batch_size: int) -> dict[str, Tensor]:
        """Static inputs of the graph: the streams and one packed row of ``thd_max_packed_sequences`` documents.

        With ``L = max_seqlen_per_dp_cp_rank``, ``T = L * context_parallel_size`` and
        ``D = thd_max_packed_sequences``:

        * ``hidden_states``: ones ``[L, 1, n*C]`` in the stream dtype (fp32 in the decoder), ``requires_grad=True``
          (the streams' dtype and shape, so TE copies replay inputs without rounding them);
        * ``cu_seqlens_q``, ``cu_seqlens_kv``, ``cu_seqlens_q_padded``, ``cu_seqlens_kv_padded``: four int32 tensors
          ``[0, T, ..., T]`` of ``D + 1`` entries (one document of ``T`` rows, then empty ones).

        No ``attention_mask`` and no ``padding_mask``: ERNIE 5.5 packs rows without either, and TE replays only the
        inputs it captured. The tensors are on the wrapper's parameters' device.

        Args:
            seq_length: Training sequence length; must be ``T``.
            micro_batch_size: Must be 1 (one packed row per micro-batch).

        Returns:
            The static inputs, ``hidden_states`` first.

        Raises:
            ValueError: If the capacities are unset, ``seq_length != T`` or ``micro_batch_size != 1``.
        """
        kind = self._GRAPH_KIND
        local_rows, documents, total_rows = _thd_graph_shape(self.config, kind)
        if seq_length != total_rows or micro_batch_size != 1:
            raise ValueError(
                f"ERNIE 5.5 {kind}-layer CUDA graphs capture packed rows of max_seqlen_per_dp_cp_rank x "
                f"context_parallel_size = {total_rows} tokens and micro-batch size 1, got seq_length={seq_length} "
                f"and micro_batch_size={micro_batch_size}."
            )
        parameter = next(self.parameters(), None)
        if parameter is None:
            raise RuntimeError(
                f"ERNIE 5.5 {kind}-layer CUDA graphs need a wrapper with parameters to place the inputs."
            )
        device = parameter.device
        width = self.config.num_residual_streams * self.config.hidden_size
        static_inputs = {
            "hidden_states": torch.ones(
                (local_rows, 1, width), dtype=self._graph_stream_dtype(), device=device, requires_grad=True
            )
        }
        cu_seqlens = torch.full((documents + 1,), total_rows, dtype=torch.int32, device=device)
        cu_seqlens[0] = 0
        for name in _PACKED_FIELDS:
            static_inputs[name] = cu_seqlens.clone()
        return static_inputs

    def _te_cuda_graph_capture(self, *args: Any, **kwargs: Any) -> Any:
        """Megatron-Core's capture, keeping every RoPE table the graph reads alive until the graphs are deleted.

        TE calls this for its warmup iterations and its capture. The attention takes its tables from the layer's
        rotary embeddings, whose ``forward`` returns the entry of Megatron-Core's shared ``lru_cache`` (on a miss, a
        table computed now). The graph reads that tensor in place on every replay, yet after capture nothing but the
        cache holds it: the capture's autograd graph, which saved it for the RoPE backward, is released. Every tensor
        a rotary embedding of this layer returns here, and the cos/sin buffers the fused RoPE path caches on it, is
        therefore kept on the wrapper until Megatron-Core deletes the graphs
        (:meth:`clear_te_cuda_graph_static_hidden_inputs`). With the 64-GPU SFT recipe settings that is one
        ``[L * CP, 1, 1, qk_pos_emb_head_dim]`` fp32 table per H layer (64 MiB at 262144 tokens), counted in
        allocated memory, and the tables of a D layer's attention and indexer.

        Returns:
            Megatron-Core's capture outputs.
        """
        from megatron.bridge.models.ernie.modeling_ernie55.hca_attention import rotary_embedding_references

        tables = self.__dict__.setdefault(_ROPE_TABLES, {})
        embeddings = {id(embedding): embedding for _, embedding in rotary_embedding_references(self)}

        def keep(module: torch.nn.Module, inputs: Any, output: Any) -> None:
            for item in output if isinstance(output, (tuple, list)) else (output,):
                if isinstance(item, Tensor):
                    tables[id(item)] = item

        handles = [embedding.register_forward_hook(keep) for embedding in embeddings.values()]
        try:
            outputs = super()._te_cuda_graph_capture(*args, **kwargs)
        finally:
            for handle in handles:
                handle.remove()
        for embedding in embeddings.values():
            for name in _ROPE_CACHE_BUFFERS:
                buffer = getattr(embedding, name, None)
                if isinstance(buffer, Tensor):
                    tables[id(buffer)] = buffer
        return outputs

    def clear_te_cuda_graph_static_hidden_inputs(self) -> None:
        """Megatron-Core's release of the graphs' retained inputs, then of the RoPE tables the graphs read.

        Megatron-Core's ``TECudaGraphHelper.delete_cuda_graphs`` calls this for every graphed layer after it resets
        the layer's graphs.
        """
        super().clear_te_cuda_graph_static_hidden_inputs()
        self.__dict__.pop(_ROPE_TABLES, None)

    def _check_te_cuda_graph_replay_inputs(
        self, args: Sequence[Any], kwargs: dict[str, Any], *, plan_checked: bool = False
    ) -> None:
        """Raise ``ValueError`` if the graph cannot replay these inputs (see the subclasses' replay methods).

        Args:
            args: The positional inputs of the call.
            kwargs: Its keyword inputs.
            plan_checked: The caller checks the host-planned parts (the number of boundaries and ``max_seqlen``)
                against its host plan instead; here they only have to be host ints and 1-D tensors.
        """
        kind = self._GRAPH_KIND
        local_rows, documents, total_rows = _thd_graph_shape(self.config, kind)
        what = f"ERNIE 5.5 {kind}-layer CUDA graph (layer {self.layer_number})"
        if len(args) > 1 or (args and "hidden_states" in kwargs):
            raise ValueError(f"{what} takes hidden_states as its only positional input, got {len(args)} positional.")
        hidden_states = args[0] if args else kwargs.get("hidden_states")
        width = self.config.num_residual_streams * self.config.hidden_size
        dtype = self._graph_stream_dtype()
        if (
            not isinstance(hidden_states, Tensor)
            or hidden_states.dtype != dtype
            or tuple(hidden_states.shape) != (local_rows, 1, width)
        ):
            got = (
                f"{tuple(hidden_states.shape)} {hidden_states.dtype}"
                if isinstance(hidden_states, Tensor)
                else type(hidden_states).__name__
            )
            raise ValueError(
                f"{what} replays {_DTYPE_NAMES.get(dtype, dtype)} streams of shape {(local_rows, 1, width)} "
                f"(max_seqlen_per_dp_cp_rank rows), got {got}. Only packed rows of max_seqlen_per_dp_cp_rank x "
                "context_parallel_size tokens can replay."
            )
        unexpected = sorted(name for name, value in kwargs.items() if name not in _REPLAY_INPUTS and value is not None)
        if unexpected:
            raise ValueError(f"{what} has no input {unexpected}; the replay would ignore them.")
        params = kwargs.get("packed_seq_params")
        if params is None or getattr(params, "qkv_format", None) != "thd":
            raise ValueError(f"{what} replays packed THD micro-batches only (packed_seq_params, qkv_format='thd').")
        if self.config.context_parallel_size > 1 and getattr(params, "cp_partition_mode", None) != "contiguous":
            raise ValueError(
                f"{what} replays the contiguous CP partition only, got cp_partition_mode="
                f"{getattr(params, 'cp_partition_mode', None)!r}."
            )
        for name in ("max_seqlen_q", "max_seqlen_kv"):
            value = getattr(params, name, None)
            if not _is_positive_int(value) or (value > total_rows and not plan_checked):
                raise ValueError(
                    f"{what} was captured with {name}={total_rows}; the micro-batch's {name} must be a host int of at "
                    f"most that, got {value!r}."
                )
        for name in _PACKED_FIELDS:
            tensor = getattr(params, name, None)
            if (
                not isinstance(tensor, Tensor)
                or tensor.dtype != torch.int32
                or tensor.ndim != 1
                or (tensor.numel() != documents + 1 and not plan_checked)
                or tensor.device != hidden_states.device
            ):
                got = tensor
                if isinstance(tensor, Tensor):
                    got = f"{tuple(tensor.shape)} {tensor.dtype} on {tensor.device}"
                raise ValueError(
                    f"{what} replays rows of exactly thd_max_packed_sequences={documents} documents: {name} must be "
                    f"int32 of shape ({documents + 1},) on {hidden_states.device}, got {got}."
                )


class Ernie55HCAHyperConnectionHybridLayer(_Ernie55GraphedHybridLayer):
    """Megatron-Core's mHC wrapper of an ERNIE 5.5 H layer, with the static inputs and replay checks of its TE graph.

    :func:`use_ernie55_attention_cuda_graphs` converts Megatron-Core's wrappers in place by switching their class,
    so this class has no constructor and no parameter, buffer or checkpoint key of its own: those, the hooks and eager
    execution are Megatron-Core's. Only the methods TE capture, replay and Megatron-Core's graph deletion call differ:

    * :meth:`get_layer_static_inputs` builds the fp32 n-stream hidden states and the packed THD ``cu_seqlens``;
    * :meth:`_te_cuda_graph_capture` keeps the RoPE tables the graph reads alive as long as the graph;
    * :meth:`_te_cuda_graph_replay` rejects a micro-batch the graph cannot replay faithfully, then replays it;
    * :meth:`clear_te_cuda_graph_static_hidden_inputs` releases the RoPE tables with the graphs.
    """

    _GRAPH_KIND = "H"

    def _te_cuda_graph_replay(self, *args: Any, **kwargs: Any) -> Any:
        """Check that the graph can replay this micro-batch, then replay it (Megatron-Core's replay).

        The checks read shapes, dtypes and host metadata only, so they add no device synchronization:

        * ``hidden_states`` is the only positional input, fp32 ``[L, 1, n*C]``;
        * every keyword input other than ``hidden_states`` and ``packed_seq_params`` is ``None``: the graph has no
          other input, and TE would drop it;
        * ``packed_seq_params`` is THD, in the contiguous partition at CP > 1, with ``max_seqlen_q`` and
          ``max_seqlen_kv`` host ints of at most ``L * CP``, and four int32 ``cu_seqlens`` of
          ``thd_max_packed_sequences + 1`` entries on the hidden states' device.

        Returns:
            Megatron-Core's ``(hidden_states, context)``.

        Raises:
            ValueError: If a check fails; the graph is not replayed.
        """
        self._check_te_cuda_graph_replay_inputs(args, kwargs)
        return super()._te_cuda_graph_replay(*args, **kwargs)


@dataclass(frozen=True)
class _DLayerGraphPlan:
    """The host plan a D-layer graph bakes in: its composition and the device tensors built from it.

    Args:
        boundaries: Compacted host ``cu_seqlens`` of the captured composition.
        max_seqlen: ``max_seqlen_q`` and ``max_seqlen_kv`` of the captured metadata.
        num_boundaries: Entries of the captured device ``cu_seqlens`` (``thd_max_packed_sequences + 1``).
        cp_size: Context-parallel size of the D attention.
        device: Device of the plan's tensors.
        template: Metadata of the composition holding the plan's device tensors: at CP > 1 the conversion route and
            the zigzag geometry, which the graphs read in place. Kept as long as the graphs.
    """

    boundaries: tuple[int, ...]
    max_seqlen: int
    num_boundaries: int
    cp_size: int
    device: torch.device
    template: PackedSeqParams


def _prepare_capture_safe_indexer_loss(config: Any, device: torch.device, what: str) -> None:
    """Make the D layer's indexer-loss bookkeeping capturable: an in-place loss scale and fixed tracker storage.

    ``DSAIndexerLossAutoScaler``'s backward multiplies by ``main_loss_backward_scale`` and creates it with a
    host-to-device copy when it is still ``None``; a captured backward then reads the tensor in place, and the schedule
    updates it in place (``set_loss_scale`` copies into it). So it is created on the device here, before the capture,
    if the schedule has not set it yet (a capture before the first step, ``cuda_graph_warmup_steps=0``), in the
    schedule's shape: Megatron-Core's ``forward_step_calc_loss`` publishes a size-1 fp32 tensor
    (``_compute_loss_scale``), which ``copy_`` cannot write into a 0-dim one. The backward multiplies the 0-dim loss's
    gradient by it and autograd sums the size-1 product back to the loss's shape, as with the schedule's own tensor.
    The tracker writes go into the tracker's storage in place, which a graph may only do when the storage has its
    fixed capacity (Bridge's setup and Megatron-Core's training set it up; a lazily grown tracker could be rebound
    after capture).

    Args:
        config: The D layer's config.
        device: The graph's device.
        what: Subject of the error messages.

    Raises:
        RuntimeError: If the scale lives on another device, or the tracker has no fixed capacity.
    """
    if (getattr(config, "dsa_indexer_loss_coeff", None) or 0.0) <= 0:
        return
    from megatron.core.transformer.experimental_attention_variant import dsa

    if dsa.DSAIndexerLossLoggingHelper.tracker.get("agreed_size") is None:
        raise RuntimeError(
            f"{what}: the DSA indexer-loss tracker has no fixed capacity, so a captured D layer could write into "
            "storage a later step reallocates. Initialize it before the capture (megatron.core's "
            "initialize_dsa_metric_tracker; Bridge's setup does)."
        )
    scaler = dsa.DSAIndexerLossAutoScaler
    scale = scaler.main_loss_backward_scale
    if scale is None:
        scaler.main_loss_backward_scale = torch.ones(1, dtype=torch.float32, device=device)
    elif not isinstance(scale, Tensor) or scale.device != device:
        raise RuntimeError(
            f"{what}: DSAIndexerLossAutoScaler's loss scale is on {getattr(scale, 'device', type(scale).__name__)}, "
            f"but a captured backward reads it in place on {device}; set it with a {device} tensor."
        )


class Ernie55DHyperConnectionHybridLayer(_Ernie55GraphedHybridLayer):
    """Megatron-Core's mHC wrapper of an ERNIE 5.5 decoder D layer, with the host plan and replay checks of its graph.

    :func:`use_ernie55_d_layer_cuda_graphs` converts Megatron-Core's wrappers in place by switching their class (with
    ``ernie_graph_d_layers``), so this class has no constructor and no parameter, buffer or checkpoint key of its
    own. On top of the H wrapper's static inputs and RoPE tables (see the module docstring, "D-layer graphs"):

    * :meth:`get_layer_static_inputs` also builds the host plan of the captured composition (one document over the
      whole row) and makes the indexer-loss bookkeeping capturable;
    * :meth:`_reconstruct_packed_seq_params_from_kwargs` attaches the plan to the metadata Megatron-Core rebuilds for
      every warmup and capture call;
    * :meth:`_te_cuda_graph_replay` replays micro-batches of the captured composition, after the offloading graphs
      replayed before them (:meth:`_wait_for_offloading_graphs`), and runs the others eagerly;
    * :meth:`clear_te_cuda_graph_static_hidden_inputs` releases the plan with the graphs.
    """

    _GRAPH_KIND = "D"

    def _graph_what(self) -> str:
        return f"ERNIE 5.5 {self._GRAPH_KIND}-layer CUDA graph (layer {self.layer_number})"

    def _graph_cp_group(self) -> torch.distributed.ProcessGroup | None:
        """The D attention's context-parallel group."""
        attention = getattr(self.inner_layer, "self_attention", None)
        return getattr(getattr(attention, "pg_collection", None), "cp", None)

    def _compute_inner_offload_module_in_cuda_graph(self) -> bool:
        """No fine-grained offload group opens inside a D graph, so the graph has no offload boundary.

        Megatron-Core decides this from the configured attention scopes (``core_attn``, ``attn_proj``,
        ``qkv_linear``), which AbsorbedMLA, the D attention, does not offload, and it turns ``attn_norm`` offloading
        off under attention graphs; the D layers run no offload group in eager execution either. With those scopes
        configured (the ERNIE 5.5 recipes offload ``core_attn`` and ``attn_proj`` of the H layers), Megatron-Core's
        answer would make the capture end by waiting on the offload streams, which no work of this capture joined
        (CUDA rejects that wait in a capture), and replay the graph on the offload side stream with the reload joins
        of a graph that offloads. The graph therefore replays on the current stream, ordered after the offloading
        graphs before it by :meth:`_wait_for_offloading_graphs`.
        """
        return False

    def _wait_for_offloading_graphs(self) -> None:
        """With fine-grained activation offloading, make the current stream wait for the offload graph stream.

        The graphs that offload (the H graphs) replay on that stream, and the current stream resumes before their
        tails have copied the offloaded activations to the host; the memory of those activations went back to the
        graphs' shared pool during their capture, and this graph may hold it (see the module docstring,
        "Offloading"). The wait orders this graph's forward replay, its static-input copies included, after every
        graph replayed before it, tail included. No device synchronization; nothing without offloading.
        """
        if not getattr(self.config, "fine_grained_activation_offloading", False):
            return
        from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
            FineGrainedActivationOffloadingInterface as off_interface,
        )

        torch.cuda.current_stream().wait_stream(off_interface.cuda_graph_stream())

    def get_layer_static_inputs(self, seq_length: int, micro_batch_size: int) -> dict[str, Tensor]:
        """The static inputs of :class:`Ernie55HCAHyperConnectionHybridLayer`, after preparing the capture.

        Megatron-Core's helper calls this before TE's warmup and capture, outside any capture, so the host plan of the
        captured composition is built here (route, host boundaries, zigzag geometry: host-to-device and
        device-to-host copies), once per device and kept until the graphs are deleted, and the indexer-loss scale is
        created on the device if needed (:func:`_prepare_capture_safe_indexer_loss`).

        Raises:
            ValueError: As the H wrapper's, and if ``thd_max_packed_sequences`` is not 1.
            RuntimeError: If the indexer-loss bookkeeping cannot be captured.
        """
        static_inputs = super().get_layer_static_inputs(seq_length, micro_batch_size)
        device = static_inputs["hidden_states"].device
        plan = self.__dict__.get(_D_GRAPH_PLAN)
        if plan is None or plan.device != device:
            self.__dict__[_D_GRAPH_PLAN] = self._build_graph_plan(device)
        _prepare_capture_safe_indexer_loss(self.config, device, self._graph_what())
        return static_inputs

    def _build_graph_plan(self, device: torch.device) -> _DLayerGraphPlan:
        """Build the host plan of the captured composition, one document of ``L * CP`` rows (eager, before capture)."""
        what = self._graph_what()
        local_rows, documents, total_rows = _thd_graph_shape(self.config, self._GRAPH_KIND)
        if documents != 1:
            raise ValueError(
                f"{what} is captured for one document over the whole row and needs thd_max_packed_sequences=1, got "
                f"{documents}."
            )
        boundaries = [0, total_rows]
        cu_seqlens = torch.tensor(boundaries, dtype=torch.int32, device=device)
        template = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            cu_seqlens_q_padded=cu_seqlens,
            cu_seqlens_kv_padded=cu_seqlens,
            max_seqlen_q=total_rows,
            max_seqlen_kv=total_rows,
        )
        cp_group = self._graph_cp_group()
        cp_size = get_pg_size(cp_group)
        if cp_size > 1:
            template.cp_partition_mode = self.config.cp_partition_mode
            api = context_parallel.require_cp_layout_api(what)
            api.prebuild_thd_cp_partition_routes(template, cp_group, device=device)
            # With ernie_cp_key_gather_inverse_grad the captured key gathers' backward reads the inverse key
            # permutation; it is pinned with the geometry (adding it inside the capture raises).
            context_parallel.prebuild_zigzag_cp_geometry(
                template,
                cp_group=cp_group,
                local_rows=local_rows,
                device=device,
                what=what,
                with_inverse=bool(getattr(self.config, "ernie_cp_key_gather_inverse_grad", False)),
            )
            boundaries = [int(value) for value in template.thd_cp_host_cu_seqlens_q]
        else:
            ops.record_pack_boundaries(template, boundaries)
        return _DLayerGraphPlan(
            boundaries=tuple(boundaries),
            max_seqlen=total_rows,
            num_boundaries=documents + 1,
            cp_size=cp_size,
            device=torch.device(device),
            template=template,
        )

    def _reconstruct_packed_seq_params_from_kwargs(self, kwargs: dict[str, Any]) -> None:
        """Megatron-Core's metadata from the static ``cu_seqlens``, with the host plan of the captured composition.

        Megatron-Core rebuilds a ``PackedSeqParams`` from the static inputs for every TE warmup and capture call; it
        has no route, host boundaries or geometry. This attaches the plan's (see the module docstring), notes the
        boundaries for the CP = 1 pack-end lookup and marks the metadata with the plan
        (:data:`D_GRAPH_PLAN_ATTR`), so the D layer runs every host-planned step from it and the D attention accepts
        the capture.

        Raises:
            RuntimeError: If the plan was not built (Megatron-Core asks for the static inputs before it captures).
        """
        super()._reconstruct_packed_seq_params_from_kwargs(kwargs)
        params = kwargs.get("packed_seq_params")
        if params is None:
            return
        plan = self.__dict__.get(_D_GRAPH_PLAN)
        if plan is None:
            raise RuntimeError(
                f"{self._graph_what()} is being captured without its host plan, which get_layer_static_inputs builds "
                "before the capture (as Megatron-Core's TECudaGraphHelper calls it)."
            )
        params.max_seqlen_q = params.max_seqlen_kv = plan.max_seqlen
        if plan.cp_size > 1:
            params.cp_partition_route = plan.template.cp_partition_route
            params.thd_cp_host_cu_seqlens_q = list(plan.boundaries)
            params.thd_cp_host_cu_seqlens_kv = list(plan.boundaries)
            context_parallel.share_zigzag_cp_geometry(plan.template, params)
        ops.record_pack_boundaries(params, plan.boundaries)
        setattr(params, D_GRAPH_PLAN_ATTR, plan)

    def clear_te_cuda_graph_static_hidden_inputs(self) -> None:
        """The H wrapper's release, then of the host plan the graphs read."""
        super().clear_te_cuda_graph_static_hidden_inputs()
        self.__dict__.pop(_D_GRAPH_PLAN, None)

    def _replay_mismatch(self, params: PackedSeqParams) -> tuple[str, str] | None:
        """Why the graph cannot replay this micro-batch's composition, as ``(kind, message)``; ``None`` if it can.

        Host metadata only: no device synchronization.
        """
        plan = self.__dict__.get(_D_GRAPH_PLAN)
        if plan is None:
            return "plan", "the graph has no host plan"
        if plan.cp_size > 1:
            host_q = getattr(params, "thd_cp_host_cu_seqlens_q", None)
            host_kv = getattr(params, "thd_cp_host_cu_seqlens_kv", None)
            if host_q is None or host_kv is None or getattr(params, "cp_partition_route", None) is None:
                return (
                    "host-plan",
                    "the micro-batch has no prebuilt CP partition route and host cu_seqlens "
                    "(prebuild_thd_cp_partition_routes; build_ernie55_thd_cp_batch does)",
                )
            boundaries = tuple(int(value) for value in host_q)
            if tuple(int(value) for value in host_kv) != boundaries:
                return "boundaries", "its query and key document boundaries differ"
        else:
            boundaries = ops.noted_pack_boundaries(params)
            if boundaries is None:
                return (
                    "host-plan",
                    "its document boundaries were not noted on the host (record_pack_boundaries; "
                    "build_ernie55_thd_cp_batch does)",
                )
        if boundaries != plan.boundaries:
            return (
                "boundaries",
                f"its document boundaries {list(boundaries)} differ from the captured {list(plan.boundaries)}",
            )
        max_seqlen = (params.max_seqlen_q, params.max_seqlen_kv)
        if max_seqlen != (plan.max_seqlen, plan.max_seqlen):
            return (
                "max-seqlen",
                f"its (max_seqlen_q, max_seqlen_kv) = {max_seqlen} differ from the captured "
                f"{(plan.max_seqlen, plan.max_seqlen)}",
            )
        for name in _PACKED_FIELDS:
            if getattr(params, name).numel() != plan.num_boundaries:
                return (
                    "boundaries",
                    f"its {name} has {getattr(params, name).numel()} entries, the captured {plan.num_boundaries}",
                )
        return None

    def _te_cuda_graph_replay(self, *args: Any, **kwargs: Any) -> Any:
        """Replay the graph for a micro-batch of the captured composition; run any other one eagerly.

        First the H wrapper's checks of what TE copies or drops, except the host-planned parts (a ``ValueError``,
        nothing runs: the hidden states' dtype, shape and position, inputs the graph does not have, the THD format, the
        partition, ``max_seqlen`` as host ints and the ``cu_seqlens`` dtype and device). Then the host plan (see the
        module docstring): if the micro-batch's host boundaries, ``max_seqlen`` and number of boundaries are the
        captured ones, Megatron-Core replays the graph, after the offloading graphs replayed before it with
        fine-grained activation offloading (:meth:`_wait_for_offloading_graphs`); otherwise the wrapper runs eagerly
        (the eager branch of Megatron-Core's ``__call__``, its hooks and the micro-batch's own routes and geometry
        included) and the first such fallback of the process logs a warning for its reason.

        Returns:
            ``(hidden_states, context)``, from the graph or the eager forward.

        Raises:
            ValueError: If a check of the first group fails.
        """
        self._check_te_cuda_graph_replay_inputs(args, kwargs, plan_checked=True)
        mismatch = self._replay_mismatch(kwargs["packed_seq_params"])
        if mismatch is None:
            self._wait_for_offloading_graphs()
            return super()._te_cuda_graph_replay(*args, **kwargs)
        reason, message = mismatch
        if reason not in _WARNED_FALLBACKS:
            _WARNED_FALLBACKS.add(reason)
            logger.warning(
                "%s runs eagerly: %s. A D-layer graph replays only micro-batches of the composition it was captured "
                "for (one document over the whole row); further fallbacks for this reason are not logged.",
                self._graph_what(),
                message,
            )
        # The eager branch of GraphableMegatronModule.__call__ (HyperConnectionHybridLayer.__call__ already took the
        # mHC recompute manager, which the forward reads from the wrapper).
        return super(GraphableMegatronModule, self).__call__(*args, **kwargs)


class Ernie55MTPDHyperConnectionHybridLayer(Ernie55DHyperConnectionHybridLayer):
    """The wrapper of an MTP depth's D layer, graphed like the decoder's D layers (``ernie_graph_mtp_layers``).

    The MTP depths' streams are in the activation dtype (Megatron-Core's MTP projections produce them; only the
    decoder casts its streams to fp32), so the static hidden states are too: bf16 with ``bf16=True``.
    """

    _GRAPH_KIND = "MTP D"

    def _graph_stream_dtype(self) -> torch.dtype:
        if getattr(self.config, "bf16", False):
            return torch.bfloat16
        if getattr(self.config, "fp16", False):
            return torch.float16
        return torch.float32


def check_ernie55_d_layer_capture(config: Any, packed_seq_params: Any, *, is_mtp_layer: bool) -> None:
    """Raise unless a CUDA graph capture that reached a D attention is one of its graph wrapper's captures.

    The D attention calls this when its forward runs inside a capture. A D layer is captured only through its graph
    wrapper (:class:`Ernie55DHyperConnectionHybridLayer`, :class:`Ernie55MTPDHyperConnectionHybridLayer`), with its
    switch on, and with the host plan attached to the metadata (:data:`D_GRAPH_PLAN_ATTR`); any other capture would
    replay the host routes and split sizes of the capture micro-batch for every later row.

    Args:
        config: The D attention's config.
        packed_seq_params: The metadata the attention received.
        is_mtp_layer: Whether the attention belongs to an MTP depth.

    Raises:
        RuntimeError: If the capture is not a D graph wrapper's.
    """
    switch = "ernie_graph_mtp_layers" if is_mtp_layer else "ernie_graph_d_layers"
    if not bool(getattr(config, switch, False)):
        raise RuntimeError(
            "ERNIE 5.5 D layers run eagerly, but a CUDA graph capture reached one: its context-parallel "
            f"conversion plans its routes on the host. Graph only the H layers (Ernie55TECudaGraphHelper), or set "
            f"{switch}=True to graph the D layers through their graph wrapper."
        )
    if getattr(packed_seq_params, D_GRAPH_PLAN_ATTR, None) is None:
        raise RuntimeError(
            "ERNIE 5.5 D layers are captured only through their graph wrapper (Ernie55DHyperConnectionHybridLayer), "
            "which attaches the host plan of the captured composition to the metadata; this capture's metadata has "
            "none."
        )


def use_ernie55_attention_cuda_graphs(stack: torch.nn.Module) -> int:
    """Give every H-layer mHC wrapper in ``stack.layers`` the ERNIE 5.5 TE-graph inputs (in place).

    Each Megatron-Core ``HyperConnectionHybridLayer`` whose inner layer's ``self_attention`` is an
    :class:`~megatron.bridge.models.ernie.modeling_ernie55.hca_attention.Ernie55HCASelfAttention` becomes an
    :class:`Ernie55HCAHyperConnectionHybridLayer` by switching its class; no parameter, buffer, hook or checkpoint key
    changes. D, MoE and dense wrappers keep Megatron-Core's class (:func:`use_ernie55_d_layer_cuda_graphs` converts
    the D wrappers), and nested stacks (the MTP stack) are not visited. Idempotent.

    Args:
        stack: A hybrid stack (anything with a ``layers`` sequence of modules).

    Returns:
        The number of H wrappers in ``stack.layers`` after the conversion.

    Raises:
        TypeError: If an H wrapper is a ``HyperConnectionHybridLayer`` subclass this function does not know.
    """
    from megatron.bridge.models.ernie.modeling_ernie55.hca_attention import Ernie55HCASelfAttention

    count = 0
    for layer in stack.layers:
        if not isinstance(layer, HyperConnectionHybridLayer):
            continue
        if not isinstance(getattr(layer.inner_layer, "self_attention", None), Ernie55HCASelfAttention):
            continue
        if type(layer) is HyperConnectionHybridLayer:
            layer.__class__ = Ernie55HCAHyperConnectionHybridLayer
        elif type(layer) is not Ernie55HCAHyperConnectionHybridLayer:
            raise TypeError(
                f"ERNIE 5.5 cannot give the H-layer wrapper {type(layer).__name__} its CUDA-graph inputs; it converts "
                "Megatron-Core's HyperConnectionHybridLayer only."
            )
        count += 1
    return count


def use_ernie55_d_layer_cuda_graphs(stack: torch.nn.Module, *, mtp: bool = False) -> int:
    """Give every D-layer mHC wrapper in ``stack.layers`` the ERNIE 5.5 TE-graph inputs and host plan (in place).

    Each Megatron-Core ``HyperConnectionHybridLayer`` whose inner layer's ``self_attention`` is an
    :class:`~megatron.bridge.models.ernie.modeling_ernie55.latent_mqa_attention.Ernie55LatentMQASelfAttention`
    becomes an :class:`Ernie55DHyperConnectionHybridLayer` (an :class:`Ernie55MTPDHyperConnectionHybridLayer` in an MTP
    depth's stack) by switching its class; no parameter, buffer, hook or checkpoint key changes. Other wrappers keep
    their class, and nested stacks are not visited. Idempotent.
    :class:`~megatron.bridge.models.ernie.modeling_ernie55.hybrid_stack.Ernie55HybridStack` calls it for the decoder
    with ``ernie_graph_d_layers``, and
    :class:`~megatron.bridge.models.ernie.modeling_ernie55.model.Ernie55HybridModel` for every MTP depth with
    ``ernie_graph_mtp_layers``.

    Args:
        stack: A hybrid stack (anything with a ``layers`` sequence of modules).
        mtp: Whether ``stack`` is an MTP depth's inner stack.

    Returns:
        The number of D wrappers in ``stack.layers`` after the conversion.

    Raises:
        TypeError: If a D wrapper is a ``HyperConnectionHybridLayer`` subclass this function does not know.
    """
    target = Ernie55MTPDHyperConnectionHybridLayer if mtp else Ernie55DHyperConnectionHybridLayer
    count = 0
    for layer in stack.layers:
        if not _is_d_wrapper(layer):
            continue
        if type(layer) is HyperConnectionHybridLayer:
            layer.__class__ = target
            # Megatron-Core caches the offload boundary on first use; the D wrapper has none.
            layer._offload_module_in_cuda_graph_cached = None
        elif type(layer) is not target:
            raise TypeError(
                f"ERNIE 5.5 cannot give the D-layer wrapper {type(layer).__name__} the CUDA-graph inputs of "
                f"{target.__name__}; it converts Megatron-Core's HyperConnectionHybridLayer only."
            )
        count += 1
    return count


def _check_nccl_graph_register() -> None:
    """Raise ``RuntimeError`` unless ``NCCL_GRAPH_REGISTER=0`` when PyTorch's allocator uses expandable segments.

    Megatron-Core's ``TECudaGraphHelper`` asserts this for ``PYTORCH_CUDA_ALLOC_CONF`` only; the ERNIE 5.5 recipes set
    the newer ``PYTORCH_ALLOC_CONF``.
    """
    for name in _ALLOC_CONF_VARIABLES:
        if "expandable_segments:True" in os.environ.get(name, "") and os.environ.get("NCCL_GRAPH_REGISTER") != "0":
            raise RuntimeError(
                f"ERNIE 5.5 CUDA graphs: {name} enables expandable segments, so NCCL_GRAPH_REGISTER=0 must be set "
                "(NCCL's graph buffer registration fails with an illegal memory access on expandable segments)."
            )


def _check_graph_capture_record_stream_reuse(config: Any) -> None:
    """Warn when activation offloading runs inside the graphs without graph-aware ``record_stream`` reuse.

    With fine-grained activation offloading the offload manager calls ``record_stream`` on every tensor it offloads,
    also while the H graphs are captured (``core_attn`` and ``attn_proj`` are inside them). Without
    ``graph_capture_record_stream_reuse:True`` PyTorch's allocator does not reuse those blocks while a capture is
    underway, so the graphs' memory pool grows by the offloaded activations. Every allocator variable that is set
    (``PYTORCH_ALLOC_CONF``, ``PYTORCH_CUDA_ALLOC_CONF``) must enable it, as Megatron-Core's test of hybrid mHC with
    attention offloading and attention graphs and its DSv4 recipes do. A warning, not an error: it costs memory, not
    correctness.
    """
    if not getattr(config, "fine_grained_activation_offloading", False):
        return
    values = {name: os.environ[name] for name in _ALLOC_CONF_VARIABLES if name in os.environ}
    if values and all(_RECORD_STREAM_REUSE in value for value in values.values()):
        return
    log_single_rank(
        logger,
        logging.WARNING,
        "ERNIE 5.5 CUDA graphs with fine-grained activation offloading: set %s in every allocator variable "
        "(PYTORCH_ALLOC_CONF and PYTORCH_CUDA_ALLOC_CONF, e.g. expandable_segments:True,%s; torch 2.9 or later), got "
        "%s. Without it PyTorch does not reuse the offloaded activations' blocks during capture, and the graphs' "
        "memory pool grows by them.",
        _RECORD_STREAM_REUSE,
        _RECORD_STREAM_REUSE,
        values or "neither variable set",
    )


def _is_d_wrapper(layer: torch.nn.Module) -> bool:
    """Whether ``layer`` is the mHC wrapper of an ERNIE 5.5 D layer."""
    from megatron.bridge.models.ernie.modeling_ernie55.latent_mqa_attention import Ernie55LatentMQASelfAttention

    inner = getattr(layer, "inner_layer", None)
    return isinstance(layer, HyperConnectionHybridLayer) and isinstance(
        getattr(inner, "self_attention", None), Ernie55LatentMQASelfAttention
    )


def _check_not_grouped(layer: HyperConnectionHybridLayer, kind: str) -> None:
    """Raise if Megatron-Core's discovery grouped a kept wrapper with the next layer."""
    if layer._get_te_cuda_graph_group_tail() is not None:
        raise RuntimeError(
            f"ERNIE 5.5 CUDA graphs: Megatron-Core grouped {kind} layer {layer.layer_number} with the next layer; "
            f"ERNIE 5.5 graphs each {kind} layer alone."
        )


def _check_graphed_parameter_sharing(kept: Sequence[torch.nn.Module], chunk_index: int) -> None:
    """Raise unless every parameter shared by kept callables has exactly two adjacent callables as users.

    A graph returns its parameters' gradients in its static buffers, and with buffer reuse Transformer Engine lets the
    graph after next allocate over them. Autograd adds a shared parameter's contributions when the last one arrives,
    so the users must be neighbors in the backward order (SD-MTP: the MTP depth's D layer right after the decoder's
    last D layer), and at most two.
    """
    users: dict[int, list[int]] = {}
    for position, layer in enumerate(kept):
        for param in layer.parameters():
            users.setdefault(id(param), []).append(position)
    for positions in users.values():
        if len(positions) > 2 or (len(positions) == 2 and positions[1] != positions[0] + 1):
            layers = [getattr(kept[position], "layer_number", "?") for position in positions]
            raise RuntimeError(
                f"ERNIE 5.5 CUDA graphs: a parameter of model chunk {chunk_index} is shared by the graphed layers "
                f"{layers} (positions {positions} among the chunk's graphs); shared parameters need exactly two "
                "adjacent graphs as users, or Transformer Engine's buffer reuse can overwrite one contribution to "
                "their gradient before autograd adds the other."
            )


# Megatron-Core DDP bucket-group state that _relearn_ddp_grad_ready_counts resets.
_DDP_GRAD_READY_STATE = ("is_first_batch", "golden_per_param_grad_ready_counts", "per_param_grad_ready_counts")


def _relearn_ddp_grad_ready_counts(model_chunks: Sequence[torch.nn.Module]) -> int:
    """Make the next batch record Megatron-Core DDP's gradient-ready counts again; return the bucket groups reset.

    With ``overlap_grad_reduce``, a bucket group of Megatron-Core's ``DistributedDataParallel`` issues its reduce once
    every parameter's gradient-ready hook has fired as often as in the first batch
    (``golden_per_param_grad_ready_counts``, recorded by the bucket group's ``reset`` at the start of the second
    batch). Graph replays can fire a hook fewer times than the eager layers they replace (see "MTP" in the module
    docstring), so after a capture those counts no longer hold. This puts every overlapping bucket group back into its
    first-batch state: the next batch only counts, reduces in ``finish_grad_sync`` after its backward, and its counts
    become the golden ones. The counts of the eager batches before the capture are dropped, so ``reset`` does not
    record them. Model chunks without bucket groups (not wrapped in Megatron-Core's DDP) and groups without
    ``overlap_grad_reduce`` are left alone.

    These three attributes are private state of Megatron-Core's ``_ParamAndGradBucketGroup``. If a bucket group lacks
    one of them, this raises instead of setting a new attribute that DDP would never read.

    Raises:
        RuntimeError: If an overlapping bucket group lacks one of the attributes.
    """
    reset = 0
    for chunk in model_chunks:
        for groups in (getattr(chunk, "bucket_groups", None), getattr(chunk, "expert_parallel_bucket_groups", None)):
            for group in groups or ():
                if not group.ddp_config.overlap_grad_reduce:
                    continue
                missing = [name for name in _DDP_GRAD_READY_STATE if not hasattr(group, name)]
                if missing:
                    raise RuntimeError(
                        f"Cannot re-learn DDP gradient-ready counts after CUDA graph capture: {type(group).__name__} "
                        f"has no {missing}. This Megatron-Core version is not supported with ERNIE 5.5 CUDA graphs "
                        "and overlap_grad_reduce."
                    )
                group.is_first_batch = True
                group.golden_per_param_grad_ready_counts = {}
                group.per_param_grad_ready_counts = {}
                reset += 1
    return reset


class Ernie55TECudaGraphHelper(TECudaGraphHelper):
    """Megatron-Core's ``TECudaGraphHelper`` restricted to the ERNIE 5.5 attention layers that are graphed.

    Capture, replay bookkeeping, manual hooks and deletion are Megatron-Core's. These differ:

    * The constructor first requires ``NCCL_GRAPH_REGISTER=0`` when ``PYTORCH_CUDA_ALLOC_CONF`` or
      ``PYTORCH_ALLOC_CONF`` enables expandable segments (Megatron-Core checks the first name only), and warns when
      fine-grained activation offloading runs without ``graph_capture_record_stream_reuse:True``.
    * :meth:`_discover_layers` keeps only the decoder's
      :class:`Ernie55HCAHyperConnectionHybridLayer` callables of Megatron-Core's discovery, plus the decoder's
      :class:`Ernie55DHyperConnectionHybridLayer` callables with ``ernie_graph_d_layers`` and the MTP depths'
      :class:`Ernie55MTPDHyperConnectionHybridLayer` callables with ``ernie_graph_mtp_layers``; everything else stays
      eager. A pipeline stage without kept layers keeps none and captures nothing.
    * :meth:`create_cudagraphs` makes the next batch re-learn the DDP gradient-ready counts of ``overlap_grad_reduce``
      (graph replays can fire a shared parameter's hook fewer times than eager layers) and logs the rank's allocator
      state after Megatron-Core's capture.

    The constructor takes Megatron-Core's arguments.
    """

    def __init__(
        self,
        model: list[torch.nn.Module],
        config: Any,
        seq_length: int,
        micro_batch_size: int,
        optimizers: list[Any] | None = None,
        pg_collection: Any = None,
        **kwargs: Any,
    ) -> None:
        _check_nccl_graph_register()
        _check_graph_capture_record_stream_reuse(config)
        super().__init__(
            model,
            config,
            seq_length,
            micro_batch_size,
            optimizers=[] if optimizers is None else optimizers,
            pg_collection=pg_collection,
            **kwargs,
        )

    def _discover_layers(self) -> None:
        """Run Megatron-Core's discovery, then keep the attention wrappers ERNIE 5.5 graphs.

        Megatron-Core's discovery returns every attention wrapper with ``cuda_graph_modules=["attn"]``: the H and D
        wrappers of the decoder and the wrappers of the nested MTP stack. This keeps the decoder's
        :class:`Ernie55HCAHyperConnectionHybridLayer` callables; with ``ernie_graph_d_layers`` the decoder's
        :class:`Ernie55DHyperConnectionHybridLayer` callables; with ``ernie_graph_mtp_layers`` the MTP depths'
        :class:`Ernie55MTPDHyperConnectionHybridLayer` callables (marked as MTP callables, as Megatron-Core marks
        them). It drops the other D wrappers and MTP callables (they run eagerly) and rebuilds the per-chunk lists
        Megatron-Core reads (a chunk without callables gets ``None``), in Megatron-Core's order, which is the
        forward order.

        Parameters shared with the MTP block: never by an H wrapper; by a D wrapper (SD-MTP: the decoder's last D
        layer) only if, among the chunk's kept callables, each shared parameter's users are exactly two adjacent ones
        (:func:`_check_graphed_parameter_sharing`), or one when the MTP's D layer runs eagerly.

        At pipeline parallelism a stage may keep nothing (for example a light last stage with only D layers and D
        graphs off): it logs a warning, and Megatron-Core's ``create_cudagraphs`` then captures nothing on it, as for
        any stage without graphable layers.

        Raises:
            TypeError: If discovery returned a callable that is neither an H wrapper, a D wrapper nor an MTP callable
                (an H wrapper that was not converted, or a scope this envelope does not cover), or, with its switch
                on, a D wrapper that was not converted.
            RuntimeError: If a model without pipeline parallelism keeps no wrapper, or a kept wrapper was grouped
                with the next layer, or parameters are shared in a way the graphs cannot sum (see above).
        """
        super()._discover_layers()
        config = getattr(self, "config", None)
        graph_d, graph_mtp = _graph_d_layers(config), _graph_mtp_layers(config)
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        excluded = {"D": 0, "MTP": 0}
        kept_counts = {"H": 0, "D": 0, "MTP D": 0}
        chunks, callables_per_chunk, callables_per_chunk_is_mtp = [], [], []
        for chunk_index, (chunk, callables, callables_is_mtp) in enumerate(
            zip(self.chunks_with_decoder, self.callables_per_chunk, self.callables_per_chunk_is_mtp, strict=True)
        ):
            mtp = getattr(chunk, "mtp", None)
            mtp_parameters = {id(param) for param in mtp.parameters()} if mtp is not None else set()
            kept, kept_is_mtp = [], []
            for layer, is_mtp in zip(callables, callables_is_mtp, strict=True):
                if is_mtp:
                    if not (graph_mtp and _is_d_wrapper(layer)):
                        excluded["MTP"] += 1
                        continue
                    if type(layer) is not Ernie55MTPDHyperConnectionHybridLayer:
                        raise TypeError(
                            f"ERNIE 5.5 CUDA graphs: the MTP D wrapper (layer {layer.layer_number}) is a "
                            f"{type(layer).__name__}; with ernie_graph_mtp_layers, Ernie55HybridModel converts it to "
                            "Ernie55MTPDHyperConnectionHybridLayer."
                        )
                    _check_not_grouped(layer, "MTP D")
                    kept.append(layer)
                    kept_is_mtp.append(True)
                    kept_counts["MTP D"] += 1
                elif isinstance(layer, Ernie55HCAHyperConnectionHybridLayer):
                    _check_not_grouped(layer, "H")
                    if any(id(param) in mtp_parameters for param in layer.parameters()):
                        raise RuntimeError(
                            f"ERNIE 5.5 CUDA graphs: H layer {layer.layer_number} shares parameters with the MTP "
                            "block, which runs eagerly; ERNIE 5.5 graphs only H layers the MTP body does not share."
                        )
                    kept.append(layer)
                    kept_is_mtp.append(False)
                    kept_counts["H"] += 1
                elif _is_d_wrapper(layer):
                    if not graph_d:
                        excluded["D"] += 1
                        continue
                    if type(layer) is not Ernie55DHyperConnectionHybridLayer:
                        raise TypeError(
                            f"ERNIE 5.5 CUDA graphs: the D wrapper of layer {layer.layer_number} is a "
                            f"{type(layer).__name__}; with ernie_graph_d_layers, Ernie55HybridStack converts it to "
                            "Ernie55DHyperConnectionHybridLayer."
                        )
                    _check_not_grouped(layer, "D")
                    kept.append(layer)
                    kept_is_mtp.append(False)
                    kept_counts["D"] += 1
                else:
                    raise TypeError(
                        f"ERNIE 5.5 CUDA graphs: Megatron-Core's discovery returned {type(layer).__name__} (layer "
                        f"{getattr(layer, 'layer_number', '?')}), which is neither an H nor a D attention wrapper. H "
                        "wrappers become Ernie55HCAHyperConnectionHybridLayer in Ernie55HybridStack."
                    )
            if graph_d:
                _check_graphed_parameter_sharing(kept, chunk_index)
            chunks.append(chunk if kept else None)
            callables_per_chunk.append(kept)
            callables_per_chunk_is_mtp.append(kept_is_mtp)
        self.chunks_with_decoder = chunks
        self.callables_per_chunk = callables_per_chunk
        self.callables_per_chunk_is_mtp = callables_per_chunk_is_mtp
        self.num_layers_per_chunk = [len(kept) for kept in callables_per_chunk]
        self.flattened_callables = [layer for kept in callables_per_chunk for layer in kept]
        self.flattened_callables_is_mtp = [is_mtp for kept in callables_per_chunk_is_mtp for is_mtp in kept]
        if not self.flattened_callables:
            if (getattr(config, "pipeline_model_parallel_size", 1) or 1) == 1:
                if graph_d:
                    raise RuntimeError(
                        f"ERNIE 5.5 CUDA graphs (rank {rank}): no H or D attention layer to graph on this rank. Use "
                        "cuda_graph_impl='none' for stacks without attention layers."
                    )
                raise RuntimeError(
                    f"ERNIE 5.5 CUDA graphs (rank {rank}): no H attention layer to graph on this rank; the D layers "
                    "and MTP run eagerly. Use cuda_graph_impl='none' for stacks without H layers."
                )
            logger.warning(
                "ERNIE 5.5 CUDA graphs (rank %d): this pipeline stage has no H attention layer and captures nothing; "
                "eager: %d D and %d MTP attention layers",
                rank,
                excluded["D"],
                excluded["MTP"],
            )
            return
        if graph_d:
            logger.info(
                "ERNIE 5.5 CUDA graphs (rank %d): graphing %d H, %d D and %d MTP D attention layers; eager: %d D "
                "and %d MTP attention layers",
                rank,
                kept_counts["H"],
                kept_counts["D"],
                kept_counts["MTP D"],
                excluded["D"],
                excluded["MTP"],
            )
            return
        logger.info(
            "ERNIE 5.5 CUDA graphs (rank %d): graphing %d H attention layers; eager: %d D and %d MTP attention layers",
            rank,
            len(self.flattened_callables),
            excluded["D"],
            excluded["MTP"],
        )

    def create_cudagraphs(self) -> None:
        """Megatron-Core's capture, DDP's gradient-ready counts re-learned, and one log line of the allocator state.

        Graph replays can fire a parameter's DDP gradient-ready hook fewer times than the eager layers did (see "MTP"
        in the module docstring), so the first graphed batch records the counts that ``overlap_grad_reduce`` waits for
        again (:func:`_relearn_ddp_grad_ready_counts`).

        The graphs' private memory pool stays reserved by PyTorch's allocator after capture, but replays write it
        without allocating, so ``mem-max-allocated-bytes`` never counts it. Megatron-Core empties the allocator's
        cache at the end of the capture, so right after it the reserved memory that is not allocated is mostly that
        pool. Compare graphed and eager runs by ``mem-max-reserved-bytes`` or NVML instead.
        """
        super().create_cudagraphs()
        if not self.flattened_callables:
            return
        relearned = _relearn_ddp_grad_ready_counts(self.model)
        log_single_rank(
            logger,
            logging.INFO,
            f"ERNIE 5.5 CUDA graphs: the next batch re-learns the DDP gradient-ready counts of {relearned} "
            "overlap_grad_reduce bucket groups",
        )
        if not torch.cuda.is_available():
            return
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        allocated, reserved = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
        logger.info(
            "ERNIE 5.5 CUDA graphs (rank %d): after capture %.2f GiB allocated, %.2f GiB reserved; the %.2f GiB "
            "reserved but not allocated are mostly the graphs' memory pool, which allocated-memory statistics do not "
            "count",
            rank,
            allocated / 2**30,
            reserved / 2**30,
            (reserved - allocated) / 2**30,
        )
