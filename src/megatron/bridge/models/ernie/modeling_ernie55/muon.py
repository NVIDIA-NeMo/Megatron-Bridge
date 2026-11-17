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

"""The reference implementation's Muon recipe as parameter attributes Megatron-Core's Muon reads.

The reference's Muon (``muon_qkv_update_mode="split_head"``, ``muon_ffn_split``) orthogonalizes some
weights block by block: every row block below goes through Newton-Schulz alone, with Muon's scale of
its own shape. :func:`tag_ernie55_muon_params` describes the blocks (:func:`ernie55_muon_split_sizes`)
to Megatron-Core as a per-parameter :class:`~megatron.core.transformer.attention.QKVLayout` with one
group whose per-head slices are the blocks (``is_mla=False``), on

* H layers: ``linear_q_up_proj`` per head, ``linear_o_group_proj`` and the output gate per output group;
* D layers: ``linear_q_up_proj`` per head and, inside a head, the no-RoPE and RoPE rows apart;
  ``linear_kv_down_proj`` as the latent and the RoPE key; ``k_b_proj``, ``v_b_proj`` and the output
  gate per head; the indexer's ``linear_wq_b`` per indexer head;
* every gated fc1 (dense MLP, routed and shared experts) as its gate and up halves.

Megatron-Core's emerging-optimizer setup tags every 2-D weight that carries a ``qkv_layout`` (``is_qkv``
and its split shapes). ``muon_split_qkv`` (Megatron-Core's default, which the ERNIE 5.5 recipe keeps) is
what turns the blocks on: with it ``TensorParallelMuon.orthogonalize`` orthogonalizes each block
separately. With ``muon_split_qkv=False`` Muon silently orthogonalizes these weights whole, dropping the
reference's blocks. Since the layout has a single group, ``muon_split_qkv_per_head`` does not change which
rows form a block; it only picks Megatron-Core's per-head implementation. The recipe leaves it off: for
blocks of one size that implementation runs one batched Newton-Schulz and reshapes its result with
``view_as`` (``TensorParallelMuon._orthogonalize_split_qkv``), which fails when the blocks have more rows
than columns, because Newton-Schulz then returns a transposed, non-contiguous tensor. ``k_b_proj`` and the
dense fc1 halves have such blocks. Without it Megatron-Core orthogonalizes the blocks one at a time.
These layouts replace Megatron-Core's own on the H and D query
up projections, which are MLA layouts (``is_mla=True``) that Muon splits only under
``muon_split_mla_qkv``; the recipe leaves that off, so no other weight is split.

The compressor's APE (``muon_exclude_patterns`` ``"ape"``) is tagged ``use_muon=False``, which sends it
to the scalar optimizer; the router matrices (``"mlp.gate"``) follow ``moe_router_skip_muon``, which
the ERNIE 5.5 bridge turns on.

The block sizes describe a weight's local rows, so they are set only when no weight is sharded along
dim 0 (tensor and expert tensor parallel size 1); otherwise Muon orthogonalizes whole matrices, with
a warning.

Per-parameter QKV layouts exist only in Megatron-Core dev: with a Megatron-Core that has no
``QKVLayout``, :func:`tag_ernie55_muon_params` still excludes the APE but leaves every weight unsplit,
with one warning per process.

Batched Newton-Schulz (``ernie_muon_ns_batch_size``). With these blocks a rank of a large expert-parallel
run orthogonalizes about a thousand matrices per step, most of them its local experts' 2048 x 2048 gate, up and
fc2 blocks, one Newton-Schulz (15 small GEMMs plus elementwise kernels) at a time, so the optimizer step is bound by kernel
launches. :class:`Ernie55BatchedMuon` is ``TensorParallelMuon`` with the same per-parameter steps whose
Newton-Schulz runs on stacks of up to ``ernie_muon_ns_batch_size`` same-shape blocks: the 3-D input selects
emerging-optimizers' batched iteration, which normalizes, iterates and scales every block on its own. Only the
GEMM kernels change, so the updates equal the unbatched ones up to floating-point summation order.
:func:`install_ernie55_batched_muon` (called when the model is built) points Megatron-Core's ``muon`` registry
entry at it; ``ernie_muon_ns_batch_size=1`` restores Megatron-Core's entry.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import re
from collections.abc import Callable, Iterator
from typing import Any

import torch
from megatron.core.optimizer import emerging_optimizers as mcore_eopt
from megatron.core.optimizer.emerging_optimizers import TensorParallelMuon
from megatron.core.process_groups_config import resolve_gtp_remat_group
from megatron.core.utils import get_pg_size, log_single_rank


try:
    from megatron.core.transformer.attention import QKVLayout
except ImportError:  # Megatron-Core main has no per-parameter QKV layouts (Megatron-Core dev only).
    QKVLayout = None

from megatron.bridge.models.ernie.modeling_ernie55.hca_attention import Ernie55HCASelfAttention
from megatron.bridge.models.ernie.modeling_ernie55.latent_mqa_attention import Ernie55LatentMQASelfAttention
from megatron.bridge.models.ernie.modeling_ernie55.latent_mqa_core import Ernie55LatentMQACoreAttention


logger = logging.getLogger(__name__)

__all__ = [
    "Ernie55BatchedMuon",
    "ernie55_muon_layout",
    "ernie55_muon_split_sizes",
    "install_ernie55_batched_muon",
    "tag_ernie55_muon_params",
]

_GATED_FC1 = re.compile(r"(^|\.)linear_fc1\.weight\d*$")
_warned_no_qkv_layout = False


def ernie55_muon_split_sizes(module: torch.nn.Module) -> dict[str, list[int]]:
    """The Muon row blocks of the weights ``module`` owns directly, by name relative to ``module``."""
    if isinstance(module, Ernie55HCASelfAttention):
        heads, groups = module.num_attention_heads_per_partition, module.o_local_groups
        q_rows = module.linear_q_up_proj.weight.shape[0]
        group_rows = module.linear_o_group_proj.shape[0] // groups
        return {
            "linear_q_up_proj.weight": [q_rows // heads] * heads,
            "linear_o_group_proj": [group_rows] * groups,
            "linear_output_gate.weight": [module.linear_output_gate.weight.shape[0] // groups] * groups,
        }
    if isinstance(module, Ernie55LatentMQASelfAttention):
        config = module.config
        heads = module.num_attention_heads_per_partition
        return {
            "linear_q_up_proj.weight": [config.qk_head_dim, config.qk_pos_emb_head_dim] * heads,
            "linear_kv_down_proj.weight": [config.kv_lora_rank, config.qk_pos_emb_head_dim],
            "k_b_proj": [config.kv_lora_rank] * heads,
            "v_b_proj": [config.v_head_dim] * heads,
        }
    if isinstance(module, Ernie55LatentMQACoreAttention):
        sizes = {"linear_output_gate.weight": [module.v_head_dim] * module.num_heads}
        indexer = module.indexer
        if indexer is not None and getattr(indexer, "linear_wq_b", None) is not None:
            sizes["indexer.linear_wq_b.weight"] = [indexer.index_head_dim] * indexer.index_n_heads
        return sizes
    return {}


def ernie55_muon_layout(sizes: list[int]) -> QKVLayout:
    """The layout that has Megatron-Core's split-QKV Muon orthogonalize every row block alone.

    One group whose projections and per-head slices are both ``sizes``, so Megatron-Core's split is
    ``sizes`` with or without ``muon_split_qkv_per_head`` (``per_head_split_shapes * num_groups`` or
    ``projection_split_shapes`` over one group); only ``muon_split_qkv`` turns it on. ``is_mla=False``,
    so the split does not depend on ``muon_split_mla_qkv``.
    """
    if QKVLayout is None:
        raise RuntimeError(
            "ERNIE 5.5 Muon blocks need Megatron-Core's per-parameter QKVLayout "
            "(megatron.core.transformer.attention.QKVLayout), which this Megatron-Core build lacks; "
            "use Megatron-Core dev."
        )
    blocks = tuple(int(size) for size in sizes)
    return QKVLayout(num_groups=1, projection_split_shapes=blocks, per_head_split_shapes=blocks, is_mla=False)


def _has_glu_interleave(config, name: str) -> bool:
    """Whether the fc1 ``name`` keeps Megatron-Core's block-interleaved ``[g, u, g, u, ...]`` rows."""
    if ".experts." in name and ".shared_experts." not in name:
        return config.moe_mlp_glu_interleave_size is not None
    if ".shared_experts." in name:
        return bool(config.use_grouped_gemm_for_shared_expert) and (
            config.moe_shared_expert_glu_interleave_size is not None
        )
    return False


def tag_ernie55_muon_params(model: torch.nn.Module) -> None:
    """Set the reference's Muon recipe on ``model``'s parameters (see the module docstring).

    Shared SD-MTP parameters are the decoder's Parameters, so they are tagged once.
    """
    global _warned_no_qkv_layout
    config = model.config
    for module in model.modules():
        compressor = getattr(module, "compressor", None)
        ape = getattr(compressor, "ape", None)
        if isinstance(ape, torch.nn.Parameter):
            ape.use_muon = False
    if QKVLayout is None:
        if not _warned_no_qkv_layout:
            _warned_no_qkv_layout = True
            logger.warning(
                "ERNIE 5.5: this Megatron-Core build has no per-parameter QKV layouts, so the reference's "
                "per-head and gate/up Muon blocks are not applied and Muon orthogonalizes those weights "
                "whole (requires Megatron-Core dev)."
            )
        return
    if config.tensor_model_parallel_size > 1 or (config.expert_tensor_parallel_size or 1) > 1:
        logger.warning(
            "ERNIE 5.5: the reference's Muon row blocks assume unsharded weights; with tensor parallelism "
            "Muon orthogonalizes whole matrices."
        )
        return
    for prefix, module in model.named_modules():
        for name, sizes in ernie55_muon_split_sizes(module).items():
            param = module.get_parameter(name)
            if sum(sizes) != param.shape[0]:
                raise RuntimeError(
                    f"ERNIE 5.5 Muon split of {prefix}.{name}: blocks {sizes} do not cover its "
                    f"{tuple(param.shape)} rows."
                )
            param.qkv_layout = ernie55_muon_layout(sizes)
    if not config.gated_linear_unit:
        return
    skipped = []
    for name, param in model.named_parameters():
        if not _GATED_FC1.search(name):
            continue
        if _has_glu_interleave(config, name):
            skipped.append(name)
            continue
        param.qkv_layout = ernie55_muon_layout([param.shape[0] // 2] * 2)
    if skipped:
        logger.warning(
            f"ERNIE 5.5: {len(skipped)} fc1 weights keep GLU-interleaved rows (e.g. {skipped[0]}); Muon "
            "orthogonalizes them whole instead of as gate and up halves."
        )


# ---------------------------------------------------------------------------------------------
# Batched Newton-Schulz (ernie_muon_ns_batch_size)
# ---------------------------------------------------------------------------------------------


@contextlib.contextmanager
def _fp32_matmul_precision(precision: str) -> Iterator[None]:
    """emerging-optimizers' ``fp32_matmul_precision``, under which the parent's step runs Newton-Schulz."""
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision(precision)
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


@dataclasses.dataclass(frozen=True)
class _NSBlock:
    """One Newton-Schulz input of a parameter: rows ``offset:offset + rows`` of each of its ``groups`` row groups.

    ``TensorParallelMuon._orthogonalize_split_qkv`` views a split weight as ``groups`` groups of ``width`` rows and
    orthogonalizes each projection (the same rows of every group) on its own; a whole matrix is one group, one block.
    """

    param: int
    position: int
    groups: int
    width: int
    offset: int
    rows: int

    def take(self, ns_input: torch.Tensor) -> torch.Tensor:
        """This block of the parameter's full Newton-Schulz input, cut as Megatron-Core cuts it."""
        if self.groups == 1 and self.rows == self.width:
            return ns_input
        cols = ns_input.shape[-1]
        grouped = ns_input.view(self.groups, self.width, cols)
        return grouped[:, self.offset : self.offset + self.rows, :].reshape(-1, cols)


class _PendingBlocks:
    """The step's bookkeeping for one parameter group: inputs being cut and outputs being collected per parameter."""

    def __init__(self, params: list[torch.Tensor], plans: dict[int, list[_NSBlock] | None]) -> None:
        self.params = params
        self.plans = plans
        self._to_take = {index: len(plan) for index, plan in plans.items() if plan is not None}
        self._to_put = dict(self._to_take)
        self._inputs: dict[int, torch.Tensor] = {}
        self._outputs: dict[int, list[torch.Tensor | None]] = {index: [None] * n for index, n in self._to_take.items()}

    def take(self, block: _NSBlock, ns_input: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """``block``'s input; the parameter's full input is built on its first block and dropped after its last."""
        if block.param not in self._inputs:
            self._inputs[block.param] = ns_input(self.params[block.param])
        piece = block.take(self._inputs[block.param])
        self._to_take[block.param] -= 1
        if self._to_take[block.param] == 0:
            del self._inputs[block.param]
        return piece

    def put(self, block: _NSBlock, output: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]] | None:
        """Record ``block``'s output; returns the parameter and all its outputs once its last block is in."""
        self._outputs[block.param][block.position] = output
        self._to_put[block.param] -= 1
        if self._to_put[block.param]:
            return None
        return self.params[block.param], self._outputs.pop(block.param)

    def detach(self, chunk: list[_NSBlock]) -> None:
        """Replace the outputs of ``chunk`` that are still waiting by copies."""
        for block in chunk:
            outputs = self._outputs.get(block.param)
            if outputs is not None:
                outputs[block.position] = outputs[block.position].clone()


class Ernie55BatchedMuon(TensorParallelMuon):
    """Megatron-Core's ``TensorParallelMuon`` with Newton-Schulz batched over same-shape blocks.

    The step applies the parent's per-parameter steps in the parent's order (lazy state, weight decay, momentum,
    Nesterov input), cuts every input into exactly the blocks ``TensorParallelMuon.orthogonalize`` orthogonalizes
    (the ``qkv_split_shapes`` projections of a split weight, otherwise the whole matrix), and stacks up to
    ``ns_batch_size`` blocks of one shape into one call of the parent's ``scaled_orthogonalize_fn``. A 3-D input
    selects emerging-optimizers' batched Newton-Schulz, which normalizes, iterates and transposes every block on its
    own, and the Muon scale and ``extra_scale_factor`` depend only on the block shape, so each block's arithmetic is
    the unbatched one with batched GEMM kernels: updates match ``TensorParallelMuon``'s up to summation order. A chunk
    of one block stays 2-D (the parent's exact call). Each parameter's update is reassembled from its blocks and
    applied through the parent's hooks, as soon as its last block is done; inputs are built chunk by chunk, so the
    transient memory is about ``ns_batch_size`` blocks.

    The parent's step runs unchanged with ``ns_batch_size=1``, with ``use_syrk``, and whenever a tensor, expert tensor
    or GTP group has more than one rank. Parameters the batched path does not handle (per-head split, a split whose
    groups are not complete on this rank, the ``distributed`` mode on a partitioned weight) are orthogonalized by the
    parent one at a time inside the same step. The optimizer state is the parent's.

    Args:
        params: Parameters or parameter groups, as for ``TensorParallelMuon``.
        ns_batch_size: Most blocks per batched Newton-Schulz; ``1`` runs the parent's step.
        **kwargs: ``TensorParallelMuon``'s keyword arguments.
    """

    def __init__(self, params: Any, *, ns_batch_size: int = 1, **kwargs: Any) -> None:
        if isinstance(ns_batch_size, bool) or not isinstance(ns_batch_size, int) or ns_batch_size < 1:
            raise ValueError(f"ns_batch_size must be an int of at least 1, got {ns_batch_size!r}")
        super().__init__(params, **kwargs)
        self.ns_batch_size = ns_batch_size
        self._unsharded: bool | None = None
        self._logged = False

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """One Muon step; batched Newton-Schulz unless the parent's step must run (see the class docstring)."""
        if closure is not None:
            raise ValueError("closure is not supported")
        if self.ns_batch_size == 1 or self.use_syrk or not self._is_unsharded():
            return super().step()
        self._batched_step(self.ns_batch_size)
        return None

    def _is_unsharded(self) -> bool:
        """No tensor, expert tensor or GTP group of more than one rank (cached: the groups never change)."""
        if self._unsharded is None:
            collection = self.pg_collection
            if not collection:
                self._unsharded = True
            else:
                groups = [getattr(collection, "tp", None), getattr(collection, "expt_tp", None)]
                groups += [resolve_gtp_remat_group(collection, is_expert) for is_expert in (False, True)]
                self._unsharded = all(get_pg_size(group) == 1 for group in groups)
        return self._unsharded

    def _ns_blocks(self, index: int, param: torch.Tensor) -> list[_NSBlock] | None:
        """The blocks ``TensorParallelMuon.orthogonalize`` would orthogonalize, or None to leave ``param`` to it.

        Called only without sharding, where the parent's path reduces to a plain Newton-Schulz of each block: no GTP
        gather, and a partitioned weight's ``duplicated`` all-gather over its one-rank group is a copy.
        """
        if param.dim() != 2:
            return None
        partition_dim = None if self.tp_mode == "blockwise" else getattr(param, "partition_dim", None)
        if partition_dim == -1:
            partition_dim = None
        if partition_dim is not None and (self.tp_mode not in ("duplicated", "auto") or not self.pg_collection):
            return None
        rows, _ = param.shape
        if not (self.split_qkv and self.is_qkv_fn(param)):
            return [_NSBlock(index, 0, 1, rows, 0, rows)]
        if self.split_qkv_per_head or getattr(param, "qkv_split_groups_are_complete", None) is False:
            return None
        shapes = getattr(param, "qkv_split_shapes", None)
        if shapes is None:
            shapes = self.qkv_split_shapes
        if not shapes or any(int(size) <= 0 for size in shapes) or rows % sum(shapes):
            return None  # the parent raises its own error for these
        width = sum(int(size) for size in shapes)
        blocks, offset = [], 0
        for position, size in enumerate(shapes):
            blocks.append(_NSBlock(index, position, rows // width, width, offset, int(size)))
            offset += int(size)
        return blocks

    def _ns_input(self, param: torch.Tensor, momentum: float) -> torch.Tensor:
        """The parent's Newton-Schulz input: the Nesterov interpolation of gradient and momentum, or the momentum."""
        buffer = self.state[param]["momentum_buffer"]
        return param.grad.lerp(buffer, momentum) if self.nesterov else buffer

    def _apply_update(self, param: torch.Tensor, update: torch.Tensor, lr: float) -> None:
        """The parent's weight update, with its hooks."""
        self.pre_weight_update_fn_inplace(param, update)
        param.add_(update, alpha=-lr)
        self.post_weight_update_fn_inplace(param)

    @staticmethod
    def _assemble(param: torch.Tensor, blocks: list[_NSBlock], outputs: list[torch.Tensor]) -> torch.Tensor:
        """A parameter's update from its orthogonalized blocks, reassembled as the parent's split path does."""
        if len(blocks) == 1 and blocks[0].groups == 1 and blocks[0].rows == blocks[0].width:
            return outputs[0]
        groups, cols = blocks[0].groups, param.shape[-1]
        return torch.cat([output.view(groups, -1, cols) for output in outputs], dim=1).view(param.shape)

    def _run_chunk(self, chunk: list[_NSBlock], pending: _PendingBlocks, momentum: float, lr: float) -> None:
        """Orthogonalize one chunk of same-shape blocks and apply the updates of the parameters it completes.

        A method of its own, so the chunk's batched tensors are released when it returns.
        """
        pieces = [pending.take(block, lambda param: self._ns_input(param, momentum)) for block in chunk]
        batch = pieces[0] if len(pieces) == 1 else torch.stack(pieces)
        del pieces
        with _fp32_matmul_precision(self.fp32_matmul_prec):
            orthogonalized = self.scaled_orthogonalize_fn(batch, None, None, tp_mode_this_group="duplicated")
        del batch
        results = [orthogonalized] if len(chunk) == 1 else list(orthogonalized.unbind(0))
        for block, result in zip(chunk, results):
            done = pending.put(block, result)
            if done is not None:
                param, outputs = done
                self._apply_update(param, self._assemble(param, pending.plans[block.param], outputs), lr)
        if len(chunk) > 1:
            # A parameter still waiting for blocks of another shape keeps copies of this chunk's blocks, not views
            # that would hold the whole batch.
            pending.detach(chunk)

    def _batched_step(self, batch_size: int) -> None:
        """The parent's step with Newton-Schulz stacked ``batch_size`` same-shape blocks at a time."""
        counts = {"blocks": 0, "shapes": 0, "batched": 0, "single": 0, "unbatched": 0}
        for group in self.param_groups:
            self._init_group(group)
            lr, momentum = group["lr"], group["momentum"]
            params = [param for param in group["params"] if param.grad is not None]
            if not params:
                continue
            # The parent's per-parameter steps before orthogonalization (decoupled weight decay changes the weight,
            # L2 the gradient), then the momentum; every parameter's input depends only on its own state.
            for param in params:
                self._apply_weight_decay_inplace(param, param.grad, lr, group["weight_decay"])
                self.state[param]["momentum_buffer"].lerp_(param.grad, 1 - momentum)
            plans = {index: self._ns_blocks(index, param) for index, param in enumerate(params)}
            group_kwargs = {key: value for key, value in group.items() if key != "params"}
            for index, plan in plans.items():
                if plan is None:
                    counts["unbatched"] += 1
                    param = params[index]
                    with _fp32_matmul_precision(self.fp32_matmul_prec):
                        update = self.orthogonalize(param, self._ns_input(param, momentum), **group_kwargs)
                    self._apply_update(param, update, lr)
            by_shape: dict[tuple, list[_NSBlock]] = {}
            for index, plan in plans.items():
                for block in plan or ():
                    param = params[index]
                    key = (block.groups * block.rows, param.shape[-1], param.dtype, param.device)
                    by_shape.setdefault(key, []).append(block)
            pending = _PendingBlocks(params, plans)
            counts["shapes"] += len(by_shape)
            for blocks in by_shape.values():
                counts["blocks"] += len(blocks)
                for start in range(0, len(blocks), batch_size):
                    chunk = blocks[start : start + batch_size]
                    counts["batched" if len(chunk) > 1 else "single"] += 1
                    self._run_chunk(chunk, pending, momentum, lr)
        if not self._logged:
            self._logged = True
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            logger.info(
                "ERNIE 5.5 batched Muon (rank %d): ns_batch_size=%d; %d Newton-Schulz blocks of %d shapes in %d "
                "batched and %d single chunks; %d parameters orthogonalized one at a time",
                rank,
                batch_size,
                counts["blocks"],
                counts["shapes"],
                counts["batched"],
                counts["single"],
                counts["unbatched"],
            )


# Megatron-Core's own ``muon`` registry entry, saved before the first install replaces it.
_MCORE_MUON_ENTRY: Any = None
_LOGGED: set[str] = set()


def _batched_muon_config_to_cls(config: Any) -> type:
    """Megatron-Core's class choice, with :class:`Ernie55BatchedMuon` in place of ``TensorParallelMuon``.

    ``LayerShardedMuon`` (``muon_tp_mode='layer_sharded'``) stays: it batches on its own. The layer-wise chunked
    optimizer-state offload accepts exactly ``TensorParallelMuon``, so it keeps it.
    """
    cls = mcore_eopt._muon_config_to_cls(config)
    if cls is not TensorParallelMuon:
        return cls
    if (
        getattr(config, "chunked_optimizer_state_offload", False)
        and (getattr(config, "optimizer_state_offload_fraction", 0.0) or 0.0) > 0.0
    ):
        return cls
    return Ernie55BatchedMuon


def _batched_muon_config_to_kwargs(config: Any, model_chunks: list, pg_collection: Any) -> dict[str, Any]:
    """Megatron-Core's ``muon`` kwargs, plus ``ns_batch_size`` from the model's ``ernie_muon_ns_batch_size``."""
    kwargs = mcore_eopt._muon_registry_config_to_kwargs(config, model_chunks, pg_collection)
    if _batched_muon_config_to_cls(config) is Ernie55BatchedMuon:
        kwargs["ns_batch_size"] = int(getattr(model_chunks[0].config, "ernie_muon_ns_batch_size", 1))
    return kwargs


def install_ernie55_batched_muon(config: object) -> None:
    """Apply ``config.ernie_muon_ns_batch_size`` to Megatron-Core's ``muon`` optimizer for this process.

    :class:`~megatron.bridge.models.ernie.modeling_ernie55.model.Ernie55HybridModel` calls this when it is built, so
    the optimizer built after it sees the setting. With a value above 1 the ``muon`` registry entry builds
    :class:`Ernie55BatchedMuon` (with ``ns_batch_size`` from the optimizer's model) wherever Megatron-Core would build
    ``TensorParallelMuon``; with 1 Megatron-Core's own entry is restored. Among models built in one process, the one
    built last decides.

    Args:
        config: The model's transformer config.
    """
    global _MCORE_MUON_ENTRY
    batch_size = int(getattr(config, "ernie_muon_ns_batch_size", 1))
    registry = mcore_eopt._EMERGING_OPTIMIZERS
    current = registry.get("muon")
    if current is None:
        if batch_size > 1:
            logger.warning(
                "ERNIE 5.5: Megatron-Core registers no 'muon' optimizer; ernie_muon_ns_batch_size is unused."
            )
        return
    installed = current.config_to_cls is _batched_muon_config_to_cls
    if not installed:
        _MCORE_MUON_ENTRY = current
    if batch_size == 1:
        if installed:
            registry["muon"] = _MCORE_MUON_ENTRY
            _log_install_once("off", "ERNIE 5.5 batched Muon: off (ernie_muon_ns_batch_size=1); Megatron-Core's Muon.")
        return
    if not installed:
        registry["muon"] = dataclasses.replace(
            current,
            optimizer_cls=Ernie55BatchedMuon,
            config_to_cls=_batched_muon_config_to_cls,
            config_to_kwargs=_batched_muon_config_to_kwargs,
        )
    _log_install_once(
        f"on:{batch_size}",
        f"ERNIE 5.5 batched Muon: on (ernie_muon_ns_batch_size={batch_size}); Newton-Schulz runs on stacks of up to "
        f"{batch_size} same-shape blocks.",
    )


def _log_install_once(key: str, message: str) -> None:
    if key not in _LOGGED:
        _LOGGED.add(key)
        log_single_rank(logger, logging.INFO, message)
