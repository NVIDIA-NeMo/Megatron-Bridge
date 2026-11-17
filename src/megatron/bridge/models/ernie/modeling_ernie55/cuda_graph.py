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

"""Transformer Engine CUDA graphs of the ERNIE 5.5 H attention layers, with fp32 hyper-connection streams.

ERNIE 5.5 supports one CUDA-graph configuration (:func:`validate_ernie55_cuda_graph_config`): Megatron-Core's
per-layer Transformer Engine (TE) graphs, ``cuda_graph_impl="transformer_engine"`` with
``cuda_graph_modules=["attn"]``, on the H layers of the decoder. Megatron-Core's mHC wrapper
(``HyperConnectionHybridLayer``) of an H layer is captured whole, forward and backward: the mHC mappings and
aggregate, the input norm, the DSv4 attention and the n-stream residual update. Everything else runs eagerly: the D
layers (their context-parallel zigzag conversion plans its routes on the host, from host ``cu_seqlens``, which a
capture would bake in; their forward raises inside a capture), the MoE and dense MLP layers, and the MTP depth.

Three Megatron-Core behaviors do not fit ERNIE 5.5; this module changes them for the H wrappers only:

* Static inputs. Megatron-Core sizes a graph's static hidden input as ``[s, b, n*C]`` in the dtype ``config.bf16``
  selects, and builds packed THD inputs only for its sequence-packing scheduler. The ERNIE 5.5 decoder streams are
  fp32 (:class:`~megatron.bridge.models.ernie.modeling_ernie55.hybrid_stack.Ernie55HybridStack`), and TE replays a
  graph with ``static_input.copy_(input)``, which converts dtypes without a check: every graphed layer would round
  its input streams to bf16 and return bf16 streams. ERNIE 5.5 packs its rows in its own step
  (:mod:`~megatron.bridge.models.ernie.ernie55_step`). :class:`Ernie55HCAHyperConnectionHybridLayer` supplies fp32
  ``[L, 1, n*C]`` hidden states (``L = max_seqlen_per_dp_cp_rank``, ``n*C`` the stream width) and the four int32
  ``cu_seqlens`` of one row of ``thd_max_packed_sequences`` documents. Megatron-Core's wrapper rebuilds the
  ``PackedSeqParams`` from them in capture and decomposes each micro-batch's own in replay, so the graph's output
  and input gradient stay fp32, as in eager execution.
* Discovery. Megatron-Core marks every attention wrapper graphable, the D layers and the nested MTP stack's included.
  :class:`Ernie55TECudaGraphHelper` keeps the decoder's H wrappers only. Bridge's training loop builds it instead of
  Megatron-Core's ``TECudaGraphHelper`` because
  :class:`~megatron.bridge.models.ernie.ernie55_config.Ernie55TransformerConfig` names it in its
  ``te_cuda_graph_helper`` class attribute. A pipeline stage without H layers captures nothing, as Megatron-Core's
  helper does for stages without graphable layers.
* RoPE tables. The DSv4 attention, its compressor and its inverse RoPE read their table from the layer's rotary
  embedding, whose ``forward`` Megatron-Core caches in one 32-entry ``functools.lru_cache`` shared by every rotary
  embedding of the process. A graph reads the table at its capture-time address, but after capture only that cache
  holds it: at the full model depth the eager D and MTP layers' own lookups evict it within one forward, and
  ``cache_clear`` (eval CP rebinding, ``YarnRotaryEmbedding``'s constructor) drops it at once, so the graph would
  read freed memory. The H wrapper keeps every table its rotary embeddings return during capture until
  Megatron-Core deletes the graphs.

A graph replays only micro-batches of its capture shape: ``L`` rows on every CP rank, exactly
``thd_max_packed_sequences`` documents (``thd_max_packed_sequences + 1`` boundaries), the longest at most ``L * CP``
rows. Before every replay the H wrapper checks this, and every input TE would silently convert or drop, from shapes
and dtypes alone (no device synchronization). The document boundaries may change from row to row: the DSv4
attention reads them on the device.

Megatron-Core's mHC selective recompute (``recompute_modules`` with ``"mhc"``) keeps working around the graphs: a
graphed H wrapper is captured without its recompute manager, so its mHC producer is not recomputed (Megatron-Core
warns), while the MoE and D wrappers keep the saving. Keep ``mhc_recompute_attn_cuda_graph_split`` off: the
attention-only split needs Megatron-Core changes for ERNIE 5.5's recompute set, attention offloading and fp32
aggregate. ``layernorm`` recompute runs inside the H graph and snapshots the RNG state there, which needs the
graph-safe Transformer Engine RNG tracker (``use_te_rng_tracker``).

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
from typing import Any

import torch
from megatron.core.models.hybrid.hybrid_block import HyperConnectionHybridLayer
from megatron.core.transformer.cuda_graphs import TECudaGraphHelper
from megatron.core.utils import log_single_rank
from torch import Tensor

from megatron.bridge.utils.cuda_graph import cuda_graph_module_names


logger = logging.getLogger(__name__)

__all__ = [
    "Ernie55HCAHyperConnectionHybridLayer",
    "Ernie55TECudaGraphHelper",
    "use_ernie55_attention_cuda_graphs",
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


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def validate_ernie55_cuda_graph_config(config: Any, *, layer: str) -> None:
    """Raise ``ValueError`` if ``config`` enables CUDA graphs outside the ERNIE 5.5 envelope.

    Configs without CUDA graphs (``cuda_graph_impl`` ``"none"`` and no legacy ``enable_cuda_graph``) pass at once.
    Otherwise every one of these must hold:

    * ``cuda_graph_impl="transformer_engine"`` and not the legacy ``enable_cuda_graph`` flag: ``"local"`` and
      ``"full_iteration"`` would capture the D layers, whose context-parallel routes are planned on the host;
    * ``cuda_graph_modules=["attn"]``: per-layer graphs of the H attention wrappers, nothing else (MoE dispatch and
      the D layers stay eager);
    * something to graph: ``experimental_attention_variant="dsv4_hybrid"`` (D-only ``"dsa"`` stacks have no H
      layer), an H layer among the decoder's ``csa_compress_ratios`` (checked when they are set; the MTP depths are
      never graphed) and ``enable_mhc_connections=True`` (only the H layers' mHC wrappers, which
      :class:`~megatron.bridge.models.ernie.modeling_ernie55.hybrid_stack.Ernie55HybridStack` converts, are graphed);
    * ``mhc_recompute_attn_cuda_graph_split=False``: the split needs Megatron-Core changes for ERNIE 5.5's recompute
      set, attention offloading and fp32 mHC aggregate;
    * ``use_te_rng_tracker=True``: ``layernorm`` recompute inside a graph snapshots the RNG state, which only a
      graph-safe tracker allows during capture. Set ``model.use_te_rng_tracker``: the training config syncs
      ``rng.te_rng_tracker`` into the model only after the model config is finalized;
    * at ``pipeline_model_parallel_size > 1``, ``cuda_graph_warmup_steps >= 1``: a stage without H layers captures
      nothing, and Bridge's training loop requires graphs on every rank when it captures before the first step;
    * ``delay_wgrad_compute=False`` and ``overlap_moe_expert_parallel_comm=False``: Megatron-Core's EP overlap
      schedule does not support ERNIE 5.5's hybrid mHC layers and MTP;
    * ``sequence_packing_scheduler=None``: ERNIE 5.5 packs its rows in its own step, and the H wrappers build their
      own packed static inputs;
    * ``max_seqlen_per_dp_cp_rank`` and ``thd_max_packed_sequences`` positive ints: the rows of every CP rank and the
      documents of every packed row, which fix the graphs' static inputs;
    * at ``context_parallel_size > 1``, ``cp_partition_mode="contiguous"``, the DSv4 layout.

    Args:
        config: Stack-level or per-layer transformer config.
        layer: Layer name for the error messages, ``"D"`` or ``"H"``.

    Raises:
        ValueError: On the first violated requirement. The message starts with
            ``"ERNIE 5.5 <layer> layer: CUDA graphs"``.
    """
    impl = getattr(config, "cuda_graph_impl", "none")
    legacy = bool(getattr(config, "enable_cuda_graph", False))
    if impl in (None, "none") and not legacy:
        return

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(f"ERNIE 5.5 {layer} layer: CUDA graphs {message}")

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
    require(
        modules == _GRAPH_MODULES,
        f"need cuda_graph_modules={_GRAPH_MODULES}, got {modules}: only the H attention layers are graphed; the MoE "
        "dispatch and the D layers run eagerly.",
    )
    variant = getattr(config, "experimental_attention_variant", None)
    require(
        variant == "dsv4_hybrid",
        f"need experimental_attention_variant='dsv4_hybrid', got {variant!r}: only the H layers are graphed, and only "
        "DSv4 hybrid stacks have them.",
    )
    ratios = getattr(config, "csa_compress_ratios", None)
    num_layers = getattr(config, "num_layers", None)
    if ratios is not None and _is_positive_int(num_layers):
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
            mode == "contiguous",
            f"need cp_partition_mode='contiguous' at context_parallel_size > 1 (the DSv4 layout), got {mode!r}.",
        )


def _thd_graph_shape(config: Any) -> tuple[int, int, int]:
    """``(rows per CP rank, documents per row, rows per row)`` of an H graph's static input."""
    local_rows = getattr(config, "max_seqlen_per_dp_cp_rank", None)
    documents = getattr(config, "thd_max_packed_sequences", None)
    if not _is_positive_int(local_rows) or not _is_positive_int(documents):
        raise ValueError(
            "ERNIE 5.5 H-layer CUDA graphs need max_seqlen_per_dp_cp_rank and thd_max_packed_sequences as positive "
            f"ints, got {local_rows!r} and {documents!r}."
        )
    return local_rows, documents, local_rows * config.context_parallel_size


class Ernie55HCAHyperConnectionHybridLayer(HyperConnectionHybridLayer):
    """Megatron-Core's mHC wrapper of an ERNIE 5.5 H layer, with the static inputs and replay checks of its TE graph.

    :func:`use_ernie55_attention_cuda_graphs` converts Megatron-Core's wrappers in place by switching their class,
    so this class has no constructor and no parameter, buffer or checkpoint key of its own: those, the hooks and eager
    execution are Megatron-Core's. Only the methods TE capture, replay and Megatron-Core's graph deletion call differ:

    * :meth:`get_layer_static_inputs` builds the fp32 n-stream hidden states and the packed THD ``cu_seqlens``;
    * :meth:`_te_cuda_graph_capture` keeps the RoPE tables the graph reads alive as long as the graph;
    * :meth:`_te_cuda_graph_replay` rejects a micro-batch the graph cannot replay faithfully, then replays it;
    * :meth:`clear_te_cuda_graph_static_hidden_inputs` releases the RoPE tables with the graphs.
    """

    def get_layer_static_inputs(self, seq_length: int, micro_batch_size: int) -> dict[str, Tensor]:
        """Static inputs of the H graph: fp32 streams and one packed row of ``thd_max_packed_sequences`` documents.

        With ``L = max_seqlen_per_dp_cp_rank``, ``T = L * context_parallel_size`` and
        ``D = thd_max_packed_sequences``:

        * ``hidden_states``: ones ``[L, 1, n*C]``, fp32, ``requires_grad=True`` (the decoder's stream dtype and
          shape, so TE copies replay inputs without rounding them);
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
        local_rows, documents, total_rows = _thd_graph_shape(self.config)
        if seq_length != total_rows or micro_batch_size != 1:
            raise ValueError(
                f"ERNIE 5.5 H-layer CUDA graphs capture packed rows of max_seqlen_per_dp_cp_rank x "
                f"context_parallel_size = {total_rows} tokens and micro-batch size 1, got seq_length={seq_length} "
                f"and micro_batch_size={micro_batch_size}."
            )
        parameter = next(self.parameters(), None)
        if parameter is None:
            raise RuntimeError("ERNIE 5.5 H-layer CUDA graphs need a wrapper with parameters to place the inputs.")
        device = parameter.device
        width = self.config.num_residual_streams * self.config.hidden_size
        static_inputs = {
            "hidden_states": torch.ones((local_rows, 1, width), dtype=torch.float32, device=device, requires_grad=True)
        }
        cu_seqlens = torch.full((documents + 1,), total_rows, dtype=torch.int32, device=device)
        cu_seqlens[0] = 0
        for name in _PACKED_FIELDS:
            static_inputs[name] = cu_seqlens.clone()
        return static_inputs

    def _te_cuda_graph_capture(self, *args: Any, **kwargs: Any) -> Any:
        """Megatron-Core's capture, keeping every RoPE table the graph reads alive until the graphs are deleted.

        TE calls this for its warmup iterations and its capture. The DSv4 attention, its compressor and its inverse
        RoPE take their table from the layer's rotary embedding, whose ``forward`` returns the entry of Megatron-Core's
        shared ``lru_cache`` (on a miss, a table computed now). The graph reads that tensor in place on every replay,
        yet after capture nothing but the cache holds it: the capture's autograd graph, which saved it for the RoPE
        backward, is released. Every tensor a rotary embedding of this layer returns here, and the cos/sin buffers
        the fused RoPE path caches on it, is therefore kept on the wrapper until Megatron-Core deletes the graphs
        (:meth:`clear_te_cuda_graph_static_hidden_inputs`). With the 64-GPU SFT recipe settings that is one
        ``[L * CP, 1, 1, qk_pos_emb_head_dim]`` fp32 table per H layer (64 MiB at 262144 tokens), counted in
        allocated memory.

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

    def _check_te_cuda_graph_replay_inputs(self, args: Sequence[Any], kwargs: dict[str, Any]) -> None:
        """Raise ``ValueError`` if the H graph cannot replay these inputs (see :meth:`_te_cuda_graph_replay`)."""
        local_rows, documents, total_rows = _thd_graph_shape(self.config)
        what = f"ERNIE 5.5 H-layer CUDA graph (layer {self.layer_number})"
        if len(args) > 1 or (args and "hidden_states" in kwargs):
            raise ValueError(f"{what} takes hidden_states as its only positional input, got {len(args)} positional.")
        hidden_states = args[0] if args else kwargs.get("hidden_states")
        width = self.config.num_residual_streams * self.config.hidden_size
        if (
            not isinstance(hidden_states, Tensor)
            or hidden_states.dtype != torch.float32
            or tuple(hidden_states.shape) != (local_rows, 1, width)
        ):
            got = (
                f"{tuple(hidden_states.shape)} {hidden_states.dtype}"
                if isinstance(hidden_states, Tensor)
                else type(hidden_states).__name__
            )
            raise ValueError(
                f"{what} replays fp32 streams of shape {(local_rows, 1, width)} (max_seqlen_per_dp_cp_rank rows), got "
                f"{got}. Only packed rows of max_seqlen_per_dp_cp_rank x context_parallel_size tokens can replay."
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
            if not _is_positive_int(value) or value > total_rows:
                raise ValueError(
                    f"{what} was captured with {name}={total_rows}; the micro-batch's {name} must be a host int of at "
                    f"most that, got {value!r}."
                )
        for name in _PACKED_FIELDS:
            tensor = getattr(params, name, None)
            if (
                not isinstance(tensor, Tensor)
                or tensor.dtype != torch.int32
                or tuple(tensor.shape) != (documents + 1,)
                or tensor.device != hidden_states.device
            ):
                got = tensor
                if isinstance(tensor, Tensor):
                    got = f"{tuple(tensor.shape)} {tensor.dtype} on {tensor.device}"
                raise ValueError(
                    f"{what} replays rows of exactly thd_max_packed_sequences={documents} documents: {name} must be "
                    f"int32 of shape ({documents + 1},) on {hidden_states.device}, got {got}."
                )


def use_ernie55_attention_cuda_graphs(stack: torch.nn.Module) -> int:
    """Give every H-layer mHC wrapper in ``stack.layers`` the ERNIE 5.5 TE-graph inputs (in place).

    Each Megatron-Core ``HyperConnectionHybridLayer`` whose inner layer's ``self_attention`` is an
    :class:`~megatron.bridge.models.ernie.modeling_ernie55.hca_attention.Ernie55HCASelfAttention` becomes an
    :class:`Ernie55HCAHyperConnectionHybridLayer` by switching its class; no parameter, buffer, hook or checkpoint key
    changes. D, MoE and dense wrappers keep Megatron-Core's class, and nested stacks (the MTP stack) are not visited.
    Idempotent.

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


class Ernie55TECudaGraphHelper(TECudaGraphHelper):
    """Megatron-Core's ``TECudaGraphHelper`` restricted to the ERNIE 5.5 decoder's H attention layers.

    Capture, replay bookkeeping, manual hooks and deletion are Megatron-Core's. These differ:

    * The constructor first requires ``NCCL_GRAPH_REGISTER=0`` when ``PYTORCH_CUDA_ALLOC_CONF`` or
      ``PYTORCH_ALLOC_CONF`` enables expandable segments (Megatron-Core checks the first name only), and warns when
      fine-grained activation offloading runs without ``graph_capture_record_stream_reuse:True``.
    * :meth:`_discover_layers` keeps only the decoder's
      :class:`Ernie55HCAHyperConnectionHybridLayer` callables of Megatron-Core's discovery; the D wrappers and every
      MTP callable stay eager. A pipeline stage without H layers keeps none and captures nothing.
    * :meth:`create_cudagraphs` logs the rank's allocator state after Megatron-Core's capture.

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
        """Run Megatron-Core's discovery, then keep the decoder's H attention wrappers only.

        Megatron-Core's discovery returns every attention wrapper with ``cuda_graph_modules=["attn"]``: the H and D
        wrappers of the decoder and the wrappers of the nested MTP stack. This keeps the decoder's
        :class:`Ernie55HCAHyperConnectionHybridLayer` callables, drops the D wrappers and every MTP callable (both run
        eagerly) and rebuilds the per-chunk lists Megatron-Core reads (a chunk without callables gets ``None``).

        At pipeline parallelism a stage may hold only D layers (for example a light last stage with the MTP depth and
        the head): it keeps no callable and logs a warning, and Megatron-Core's ``create_cudagraphs`` then captures
        nothing on it, as for any stage without graphable layers. The other stages graph their H layers.

        Raises:
            TypeError: If discovery returned a callable that is neither an H wrapper, a D wrapper nor an MTP callable
                (an H wrapper that was not converted, or a scope this envelope does not cover).
            RuntimeError: If a model without pipeline parallelism has no H wrapper left, or a kept wrapper was grouped
                with the next layer or shares a parameter with the chunk's MTP block.
        """
        super()._discover_layers()
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        excluded = {"D": 0, "MTP": 0}
        chunks, callables_per_chunk = [], []
        for chunk, callables, callables_is_mtp in zip(
            self.chunks_with_decoder, self.callables_per_chunk, self.callables_per_chunk_is_mtp, strict=True
        ):
            mtp = getattr(chunk, "mtp", None)
            mtp_parameters = {id(param) for param in mtp.parameters()} if mtp is not None else set()
            kept = []
            for layer, is_mtp in zip(callables, callables_is_mtp, strict=True):
                if is_mtp:
                    excluded["MTP"] += 1
                elif isinstance(layer, Ernie55HCAHyperConnectionHybridLayer):
                    if layer._get_te_cuda_graph_group_tail() is not None:
                        raise RuntimeError(
                            f"ERNIE 5.5 CUDA graphs: Megatron-Core grouped H layer {layer.layer_number} with the next "
                            "layer; ERNIE 5.5 graphs each H layer alone."
                        )
                    if any(id(param) in mtp_parameters for param in layer.parameters()):
                        raise RuntimeError(
                            f"ERNIE 5.5 CUDA graphs: H layer {layer.layer_number} shares parameters with the MTP "
                            "block, which runs eagerly; ERNIE 5.5 graphs only H layers the MTP body does not share."
                        )
                    kept.append(layer)
                elif _is_d_wrapper(layer):
                    excluded["D"] += 1
                else:
                    raise TypeError(
                        f"ERNIE 5.5 CUDA graphs: Megatron-Core's discovery returned {type(layer).__name__} (layer "
                        f"{getattr(layer, 'layer_number', '?')}), which is neither an H nor a D attention wrapper. H "
                        "wrappers become Ernie55HCAHyperConnectionHybridLayer in Ernie55HybridStack."
                    )
            chunks.append(chunk if kept else None)
            callables_per_chunk.append(kept)
        self.chunks_with_decoder = chunks
        self.callables_per_chunk = callables_per_chunk
        self.callables_per_chunk_is_mtp = [[False] * len(kept) for kept in callables_per_chunk]
        self.num_layers_per_chunk = [len(kept) for kept in callables_per_chunk]
        self.flattened_callables = [layer for kept in callables_per_chunk for layer in kept]
        self.flattened_callables_is_mtp = [False] * len(self.flattened_callables)
        if not self.flattened_callables:
            if (getattr(getattr(self, "config", None), "pipeline_model_parallel_size", 1) or 1) == 1:
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
        logger.info(
            "ERNIE 5.5 CUDA graphs (rank %d): graphing %d H attention layers; eager: %d D and %d MTP attention layers",
            rank,
            len(self.flattened_callables),
            excluded["D"],
            excluded["MTP"],
        )

    def create_cudagraphs(self) -> None:
        """Megatron-Core's capture, then one log line of this rank's allocator state, which shows the graphs' memory.

        The graphs' private memory pool stays reserved by PyTorch's allocator after capture, but replays write it
        without allocating, so ``mem-max-allocated-bytes`` never counts it. Megatron-Core empties the allocator's
        cache at the end of the capture, so right after it the reserved memory that is not allocated is mostly that
        pool. Compare graphed and eager runs by ``mem-max-reserved-bytes`` or NVML instead.
        """
        super().create_cudagraphs()
        if not self.flattened_callables or not torch.cuda.is_available():
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
