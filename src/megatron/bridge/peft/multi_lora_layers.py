# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Multi-adapter LoRA layer for Megatron parallel linears.

:class:`MultiLoRALinear` wraps a single Megatron parallel linear module with
*N* concurrent LoRA adapters.  The active adapter is selected at forward time
via per-layer ``tokens_per_adapter`` set by :func:`set_tokens_per_adapter_slot`.

Forward stacks the raw weights of all adapters and uses ``torch._grouped_mm``
for a single fused kernel; TP/SP collectives are issued once around the two
GEMMs to match the layout of the wrapped base linear.

:class:`MultiLoRAGroupedExpertLinear` is the MoE counterpart, wrapping a grouped
expert linear (``mlp.experts.linear_fc{1,2}`` of a ``TEGroupedMLP``) with one
low-rank pair per (adapter slot, local expert). Inside the experts the token
order is the dispatcher's expert-major permutation rather than the micro-batch's
adapter-major order, so ``tokens_per_adapter`` alone cannot segment it;
:func:`install_moe_slot_routing` co-permutes a per-token slot id through the
dispatcher to recover the per-(slot, expert) segmentation.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from megatron.core import parallel_state
from megatron.core.tensor_parallel.mappings import (
    all_to_all,
    gather_from_sequence_parallel_region,
    gather_from_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
    scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.moe.moe_utils import sort_chunks_by_idxs

from megatron.bridge.peft.adapter_wrapper import AdapterWrapper
from megatron.bridge.peft.utils import (
    GroupedExpertLinearAdapter,
    ParallelLinearAdapter,
    all2all_hp2sp,
    get_adapter_attributes_from_linear,
)


class MultiLoRALinear(AdapterWrapper):
    """Megatron parallel linear wrapped with *N* concurrent LoRA adapters.

    Each adapter slot is a :class:`ParallelLinearAdapter` stored in an
    ``nn.ModuleList``. Forward uses grouped GEMM with a single set of
    TP/SP comms for efficiency.

    For bridge export compatibility, use :func:`expose_adapter_slot` to
    temporarily expose one slot as ``.adapter``.
    """

    def __init__(
        self,
        to_wrap: nn.Module,
        n_adapters: int,
        dim: int,
        alpha: float,
        full_name: str,
        column_init_method: str = "xavier",
        row_init_method: str = "zero",
        dropout: float = 0.0,
        dropout_position: str = "pre",
        a2a_experimental: bool = False,
    ) -> None:
        nn.Module.__init__(self)
        # The grouped-GEMM forward below never runs each adapter's own
        # ParallelLinearAdapter.forward, so adapter dropout would be silently
        # dropped. Reject dropout>0 loudly instead of pretending to apply it.
        assert dropout == 0.0, (
            f"MultiLoRALinear grouped-GEMM path does not apply adapter dropout "
            f"(got dropout={dropout}); set dropout/--lora-dropout to 0."
        )
        self.to_wrap = to_wrap
        self._adapter_enabled = True
        self.n_adapters = n_adapters
        self.max_rank = dim
        # Kept so a slot re-init (reset_adapter) mirrors the construction-time
        # init methods instead of hardcoding xavier/zero.
        self._column_init_method = column_init_method
        self._row_init_method = row_init_method

        attrs = get_adapter_attributes_from_linear(to_wrap)

        # input_is_parallel distinguishes column-parallel base (False, e.g. linear_qkv,
        # linear_fc1) from row-parallel base (True, e.g. linear_proj, linear_fc2).
        # It controls which TP collective runs between the two grouped GEMMs and
        # whether the second GEMM's output needs to be all-gathered to match the
        # wrapped base linear's output layout.
        self.input_is_parallel = attrs.input_is_parallel
        self.disable_sequence_parallel_comm = attrs.disable_sequence_parallel_comm
        self.use_a2a = a2a_experimental
        self._gather_output = attrs.input_is_parallel

        # ModuleList of ParallelLinearAdapters gives per-adapter optimizer state
        # isolation, clean checkpoint serialization, and bridge export compatibility.
        # Adapter kwargs mirror the single-LoRA path (LoRA.transform).
        self.adapters = nn.ModuleList(
            [
                ParallelLinearAdapter(
                    in_features=attrs.in_features,
                    out_features=attrs.out_features,
                    dim=dim,
                    base_linear_name=full_name,
                    activation="identity",
                    alpha=alpha,
                    input_is_parallel=attrs.input_is_parallel,
                    column_init_method=column_init_method,
                    row_init_method=row_init_method,
                    model_parallel_config=getattr(to_wrap, "config", None),
                    disable_tensor_parallel_comm=attrs.disable_tensor_parallel_comm,
                    disable_sequence_parallel_comm=attrs.disable_sequence_parallel_comm,
                    base_linear_is_parallel=attrs.base_linear_is_parallel,
                    a2a_experimental=a2a_experimental,
                    dropout=dropout,
                    dropout_position=dropout_position,
                )
                for _ in range(n_adapters)
            ]
        )

        self.tokens_per_adapter: Optional[torch.Tensor] = None
        device = next(to_wrap.parameters()).device
        dtype = next(to_wrap.parameters()).dtype
        # Non-persistent: slot lifecycle is externally managed, not checkpointed.
        self.register_buffer("alpha_values", torch.ones(n_adapters, dtype=dtype, device=device), persistent=False)
        self.register_buffer(
            "rank_values", torch.full((n_adapters,), dim, dtype=dtype, device=device), persistent=False
        )

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        linear_output, bias, layernorm_output = self.base_linear_forward(x, *args, **kwargs)

        if not self._adapter_enabled:
            return linear_output, bias

        tokens_per_adapter = self.tokens_per_adapter
        x = layernorm_output.contiguous()

        # SP gather (once) — for column-parallel base layers without an LN-fused
        # gather, the layernorm output is still SP-sharded and must be gathered
        # to full sequence before the adapter matmul.
        if not self.disable_sequence_parallel_comm and not self.input_is_parallel:
            x = gather_from_sequence_parallel_region(x)

        x_flat = x.reshape(-1, x.shape[-1])
        offsets = tokens_per_adapter.cumsum(dim=0, dtype=torch.int32)

        stacked_A = torch.stack([a.linear_in.weight for a in self.adapters])
        stacked_B = torch.stack([a.linear_out.weight for a in self.adapters])

        mid = torch._grouped_mm(x_flat, stacked_A.transpose(-2, -1), offsets)

        # TP collective between A and B: row-parallel base needs an all-reduce
        # of the partial sums; column-parallel base needs an all-gather of the
        # rank-sharded output to a full [tokens, dim] for the second GEMM.
        if self.input_is_parallel:
            mid = reduce_from_tensor_model_parallel_region(mid)
        else:
            mid = gather_from_tensor_model_parallel_region(mid)

        out = torch._grouped_mm(mid, stacked_B.transpose(-2, -1), offsets)

        # Per-token scaling is applied *before* the output-side TP/SP comms.
        # ``per_token_scaling`` is indexed by the full token count
        # (``tokens_per_adapter`` sums to it); doing it after a sequence-parallel
        # scatter would leave ``out`` with ``tokens/tp`` rows and crash here.
        # The ratio is computed and applied in the activation dtype: a ratio not
        # exactly representable there (e.g. alpha/rank = 32/24 in bf16) is
        # rounded, while the rollout engine (sglang) multiplies the exact fp32
        # ratio into an fp32 accumulator. Where train/rollout parity at that
        # level matters, keep alpha/rank ratios exactly representable; closing
        # the gap entirely would require applying the scaling in fp32.
        scaling = self.alpha_values / self.rank_values
        per_token_scaling = torch.repeat_interleave(scaling, tokens_per_adapter).unsqueeze(-1)
        out = out * per_token_scaling

        # Match the wrapped base linear's output layout: row-parallel base
        # produces a fully-summed [tokens, h_out] tensor (which we then SP
        # scatter); column-parallel base keeps the [tokens, h_out/tp] shard.
        if self._gather_output:
            out = gather_from_tensor_model_parallel_region(out)

        if not self.disable_sequence_parallel_comm and self.input_is_parallel:
            if self.use_a2a:
                out = all2all_hp2sp(out)
            else:
                out = scatter_to_sequence_parallel_region(out)

        return linear_output + out.reshape(linear_output.shape), bias

    def reset_adapter(self, idx: int) -> None:
        # Re-init through the model-parallel RNG tracker so every DP replica
        # produces identical weights regardless of how far the global RNG has
        # advanced since model build. A bare nn.init here diverges replicas on
        # slot reuse (breaking the DP-equal invariant the weight checker relies
        # on). Mirror the construction-time init methods rather than hardcoding.
        from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

        from megatron.bridge.peft.utils import ParallelLinearAdapter

        adapter = self.adapters[idx]
        col_fn = ParallelLinearAdapter._get_init_fn(None, self._column_init_method)
        row_fn = ParallelLinearAdapter._get_init_fn(None, self._row_init_method)
        with get_cuda_rng_tracker().fork():
            col_fn(adapter.linear_in.weight.data)
            row_fn(adapter.linear_out.weight.data)

    def init_adapter_slot(self, idx: int, rank: int, alpha: float) -> None:
        """Claim slot ``idx`` for an adapter: bind ``rank``/``alpha`` and apply the rank mask."""
        assert 0 < rank <= self.max_rank, f"Adapter rank {rank} must be in (0, {self.max_rank}]"
        self.alpha_values[idx] = alpha
        self.rank_values[idx] = rank
        self._apply_rank_mask(idx)

    def clear_adapter_slot(self, idx: int) -> None:
        """Free slot ``idx``: zero alpha, restore max rank, re-init weights."""
        self.alpha_values[idx] = 0
        self.rank_values[idx] = self.max_rank
        self.reset_adapter(idx)

    def _apply_rank_mask(self, idx: int) -> None:
        """Zero padded rows of A and padded cols of B for slot ``idx``.

        For column-parallel base layers (``linear_qkv``, ``linear_fc1``)
        ``linear_in.weight`` is sharded across TP — rank ``r`` owns global
        rows ``[r*L : (r+1)*L]`` where ``L = max_rank/tp``. For row-parallel
        base it is replicated. Map the global cutoff ``actual_rank`` into
        the local shard before zeroing.

        With both sides zero in the padded region, the autograd chain through
        the two GEMMs keeps the gradient zero there too — no periodic
        re-masking needed during training.
        """
        actual_rank = int(self.rank_values[idx].item())
        if actual_rank >= self.max_rank:
            return
        adapter = self.adapters[idx]
        local_rank_dim = adapter.linear_in.weight.shape[0]
        if local_rank_dim < self.max_rank:
            tp_rank = parallel_state.get_tensor_model_parallel_rank()
            shard_start = tp_rank * local_rank_dim
            local_start = max(0, actual_rank - shard_start)
        else:
            local_start = actual_rank
        with torch.no_grad():
            if local_start < local_rank_dim:
                adapter.linear_in.weight.data[local_start:].zero_()
            adapter.linear_out.weight.data[:, actual_rank:].zero_()

    def state_dict(
        self,
        destination: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> Dict[str, Any]:
        if destination is None:
            destination = {}
        self.to_wrap.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        self.adapters.state_dict(destination=destination, prefix=f"{prefix}adapters.", keep_vars=keep_vars)
        return destination

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: Tuple[Tuple[int, int, int], ...] = (),
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        sharded_sd: Dict[str, Any] = {}
        sharded_sd.update(self.to_wrap.sharded_state_dict(prefix, sharded_offsets, metadata))
        for i, adapter in enumerate(self.adapters):
            sharded_sd.update(adapter.sharded_state_dict(f"{prefix}adapters.{i}.", sharded_offsets, metadata))
        return sharded_sd


@dataclass
class ExpertSlotRouting:
    """Per-forward mapping from dispatched expert tokens to adapter slots.

    Built once per MoE layer forward by :func:`install_moe_slot_routing`'s hook
    and shared by that layer's ``linear_fc1``/``linear_fc2`` adapters, which see
    the same rows in the same order.

    Attributes:
        sort_idx: Permutation ordering the dispatched tokens by
            ``(slot, local_expert)``, so one grouped GEMM covers every
            (slot, expert) pair.
        inverse_idx: Inverse of ``sort_idx``, restoring the base layer's order.
        group_offsets: Inclusive cumsum of the ``n_adapters * num_local_experts``
            group sizes, in ``slot``-major order (group ``s * E + e``).
        slot_token_counts: Tokens per slot, for per-token alpha/rank scaling.
        num_tokens: Row count the routing was built for; guards against a base
            layer that pads its input (e.g. fp8 quantization padding).
    """

    sort_idx: torch.Tensor
    inverse_idx: torch.Tensor
    group_offsets: torch.Tensor
    slot_token_counts: torch.Tensor
    num_tokens: int


class MultiLoRAGroupedExpertLinear(MultiLoRALinear):
    """Grouped MoE expert linear wrapped with *N* concurrent LoRA adapters.

    One :class:`GroupedExpertLinearAdapter` per slot, i.e. an independent
    low-rank pair per (slot, local expert). Reusing the single-LoRA adapter
    class keeps the packed ``[num_local_experts, ...]`` weight layout that the
    bridge's grouped-expert export and distributed checkpointing already
    understand; this class only owns the multi-slot forward.

    Subclassing :class:`MultiLoRALinear` is deliberate: the slot lifecycle
    helpers here and the ``isinstance``-based multi-LoRA discovery in downstream
    consumers (per-slot optimizers, adapter-state zeroing) then pick expert
    layers up with no changes.

    Unlike the dense layer, the wrapped base sits *inside* the experts, where
    rows are the dispatcher's expert-major permutation of tokens from every EP
    rank — ``tokens_per_adapter`` does not segment it. The per-forward
    :class:`ExpertSlotRouting` supplies that segmentation instead.
    """

    def __init__(
        self,
        to_wrap: nn.Module,
        n_adapters: int,
        dim: int,
        alpha: float,
        full_name: str,
        num_local_experts: int,
        column_init_method: str = "xavier",
        row_init_method: str = "zero",
        dropout: float = 0.0,
        dropout_position: str = "pre",
    ) -> None:
        nn.Module.__init__(self)
        # Same reason as the dense layer: the grouped-GEMM forward never runs an
        # adapter's own forward, so dropout would be silently dropped.
        assert dropout == 0.0, (
            f"MultiLoRAGroupedExpertLinear grouped-GEMM path does not apply adapter dropout "
            f"(got dropout={dropout}); set dropout/--lora-dropout to 0."
        )
        self.to_wrap = to_wrap
        self._adapter_enabled = True
        self.n_adapters = n_adapters
        self.max_rank = dim
        self.base_linear_name = full_name
        self.num_local_experts = num_local_experts
        self._column_init_method = column_init_method
        self._row_init_method = row_init_method

        # Not defensive: the dispatch requirements below are the only thing standing
        # between an unsupported MoE config and silently mis-routed expert tokens, so
        # a missing config must fail rather than let every check read its default.
        config = to_wrap.config
        expert_tp_size = parallel_state.get_expert_tensor_parallel_world_size() or getattr(
            config, "expert_tensor_parallel_size", 1
        )
        # Expert TP would shard the adapter's rank axis and require ETP
        # collectives between the two GEMMs. Refuse rather than inherit the
        # single-LoRA grouped adapter's per-shard factorization, which is not
        # equivalent to the full LoRA product for a row-parallel base.
        if expert_tp_size and expert_tp_size > 1:
            raise NotImplementedError(
                f"Multi-LoRA on MoE experts requires expert_tensor_parallel_size=1 "
                f"(got {expert_tp_size}) for {full_name}."
            )
        # TEGroupedMLP pads its input to the quantization alignment before
        # calling the grouped linears, which would desynchronize the row order
        # the slot routing was built for.
        if getattr(config, "fp8", None) or getattr(config, "fp4", None):
            raise NotImplementedError(
                f"Multi-LoRA on MoE experts does not support fp8/fp4 expert quantization "
                f"(quantization padding changes the dispatched token order) for {full_name}."
            )
        # The remaining dispatch requirements are checked here, at model build
        # time, rather than in the routing hook: a hook that raises on only some
        # ranks would leave its peers waiting in the companion all-to-all.
        dispatcher_type = getattr(config, "moe_token_dispatcher_type", None)
        if dispatcher_type != "alltoall":
            raise NotImplementedError(
                f"Multi-LoRA on MoE experts requires moe_token_dispatcher_type='alltoall' "
                f"(got {dispatcher_type!r}) for {full_name}: the slot routing replays that "
                f"dispatcher's permutation stages."
            )
        # With fusion on, the dispatcher's recorded permutation is TE's row_id_map
        # — a per-(expert, token) destination table with -1 for unrouted pairs —
        # which is only meaningful to the matching fused unpermute, not as a
        # gather index over local tokens.
        if getattr(config, "moe_permute_fusion", False):
            raise NotImplementedError(
                f"Multi-LoRA on MoE experts requires moe_permute_fusion=False (got True) for "
                f"{full_name}: the fused permute records a row_id_map instead of a token gather "
                f"index, so the adapter cannot follow the dispatcher's permutation."
            )
        if getattr(config, "moe_pad_expert_input_to_capacity", False):
            raise NotImplementedError(
                f"Multi-LoRA on MoE experts does not support moe_pad_expert_input_to_capacity "
                f"for {full_name} (the drop-and-pad dispatch has no slot-routing implementation)."
            )

        attrs = get_adapter_attributes_from_linear(to_wrap, is_expert=True)
        self.input_is_parallel = attrs.input_is_parallel
        # With expert TP disabled the adapter needs no TP/SP collectives at all:
        # the base grouped linear issues none, and the dispatched rows it is
        # handed are already gathered to full sequence.
        self.disable_sequence_parallel_comm = True
        self.use_a2a = False
        self._gather_output = False

        first_param = next(to_wrap.parameters())
        self.adapters = nn.ModuleList(
            [
                GroupedExpertLinearAdapter(
                    attrs.in_features,
                    attrs.out_features,
                    dim,
                    num_local_experts=num_local_experts,
                    base_linear_name=full_name,
                    activation="identity",
                    column_init_method=column_init_method,
                    row_init_method=row_init_method,
                    input_is_parallel=attrs.input_is_parallel,
                    dropout=dropout,
                    dropout_position=dropout_position,
                    model_parallel_config=config,
                    alpha=alpha,
                    base_linear_is_parallel=attrs.base_linear_is_parallel,
                    params_device=first_param.device,
                    params_dtype=first_param.dtype,
                )
                for _ in range(n_adapters)
            ]
        )

        self.tokens_per_adapter: Optional[torch.Tensor] = None
        # Republished by the MoE-layer forward pre-hook on every experts forward
        # (including recompute replays), so it is never stale when read; the None
        # here only guards a forward that runs before install_moe_slot_routing.
        self.expert_slot_routing: Optional[ExpertSlotRouting] = None
        device = first_param.device
        dtype = first_param.dtype
        self.register_buffer("alpha_values", torch.ones(n_adapters, dtype=dtype, device=device), persistent=False)
        self.register_buffer(
            "rank_values", torch.full((n_adapters,), dim, dtype=dtype, device=device), persistent=False
        )

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        linear_output, bias, layernorm_output = self.base_linear_forward(x, *args, **kwargs)

        if not self._adapter_enabled:
            return linear_output, bias

        routing = self.expert_slot_routing
        if routing is None:
            raise RuntimeError(
                f"{self.base_linear_name}: no expert slot routing for this forward. "
                f"install_moe_slot_routing(model) must run after MultiLoRA is applied (it is "
                f"called by MultiLoRA.__call__), and set_tokens_per_adapter_slot(model, counts) "
                f"before every forward."
            )

        x_flat = layernorm_output.reshape(-1, layernorm_output.shape[-1])

        # Every slot's weights must stay in the autograd graph even when this
        # rank's experts received no tokens: Megatron's grad buffers expect a
        # grad hook per trainable parameter, and a slot missing from the graph
        # would leave its bucket incomplete. torch.stack below provides that in
        # the normal path; the empty path needs an explicit zero term.
        if routing.num_tokens == 0:
            zero_term = sum((a.linear_in.weight.sum() + a.linear_out.weight.sum()) for a in self.adapters) * 0.0
            return linear_output + zero_term, bias

        if x_flat.shape[0] != routing.num_tokens:
            raise RuntimeError(
                f"{self.base_linear_name}: base layer received {x_flat.shape[0]} rows but the "
                f"slot routing was built for {routing.num_tokens}. The base grouped linear must "
                f"not pad or reorder its input between dispatch and the expert GEMMs."
            )

        # [n_adapters, num_local_experts, ...] -> one group per (slot, expert),
        # slot-major so group index s*E+e matches the sort key below. The stack
        # is a copy, but it is what keeps every slot in the graph (see above).
        stacked_A = torch.stack([a.linear_in.weight for a in self.adapters])
        stacked_B = torch.stack([a.linear_out.weight for a in self.adapters])
        num_groups = stacked_A.shape[0] * stacked_A.shape[1]
        grouped_A = stacked_A.reshape(num_groups, *stacked_A.shape[2:])
        grouped_B = stacked_B.reshape(num_groups, *stacked_B.shape[2:])

        x_sorted = x_flat.index_select(0, routing.sort_idx)
        mid = torch._grouped_mm(x_sorted, grouped_A.transpose(-2, -1), routing.group_offsets)
        out = torch._grouped_mm(mid, grouped_B.transpose(-2, -1), routing.group_offsets)

        # Scaling is applied in sorted (slot-major) order, before unsorting.
        # Same dtype caveat as the dense layer: the ratio is rounded to the
        # activation dtype while sglang multiplies an exact fp32 ratio.
        scaling = self.alpha_values / self.rank_values
        out = out * torch.repeat_interleave(scaling, routing.slot_token_counts).unsqueeze(-1)
        out = out.index_select(0, routing.inverse_idx)

        return linear_output + out.reshape(linear_output.shape), bias

    def reset_adapter(self, idx: int) -> None:
        # Same model-parallel RNG discipline as the dense layer, but forking the
        # *expert* tracker: its seed is constant across expert-data-parallel peers
        # and varies across ep/etp ranks, matching how Megatron seeds the base
        # expert weights. Forking the dense tracker instead would make EDP
        # replicas of a reused slot diverge, since the dense tracker's seed
        # varies along the dimension EDP peers differ in.
        from megatron.core.tensor_parallel.random import (
            get_cuda_rng_tracker,
            get_expert_parallel_rng_tracker_name,
        )

        adapter = self.adapters[idx]
        col_fn = ParallelLinearAdapter._get_init_fn(None, self._column_init_method)
        row_fn = ParallelLinearAdapter._get_init_fn(None, self._row_init_method)
        with get_cuda_rng_tracker().fork(get_expert_parallel_rng_tracker_name()):
            col_fn(adapter.linear_in.weight.data)
            row_fn(adapter.linear_out.weight.data)

    def _apply_rank_mask(self, idx: int) -> None:
        """Zero the padded rank rows of A and rank columns of B for slot ``idx``.

        Packed grouped-expert weights are ``[num_local_experts, rank, in]`` and
        ``[num_local_experts, out, rank]``, so the rank axis is 1 and -1
        respectively, for every local expert at once. ``__init__`` rejects
        expert TP, so the rank axis is never sharded and needs no local
        remapping (unlike the dense layer).
        """
        actual_rank = int(self.rank_values[idx].item())
        if actual_rank >= self.max_rank:
            return
        adapter = self.adapters[idx]
        with torch.no_grad():
            adapter.linear_in.weight.data[:, actual_rank:, :].zero_()
            adapter.linear_out.weight.data[..., actual_rank:].zero_()


# ==================================================================
# Standalone functions
# ==================================================================

_MULTI_LORA_TYPES = (MultiLoRALinear,)


def _iter_multi_lora_modules(model):
    models = model if isinstance(model, list) else [model]
    for model_chunk in models:
        for module in model_chunk.modules():
            if isinstance(module, _MULTI_LORA_TYPES):
                yield module


def set_tokens_per_adapter_slot(model, tokens_per_adapter: torch.Tensor) -> None:
    """Route a packed micro-batch to its per-slot token spans.

    ``tokens_per_adapter[i]`` is the number of contiguous tokens in the
    upcoming forward that belong to adapter slot ``i``. Must sum to the total
    token count of the micro-batch.
    """
    for module in _iter_multi_lora_modules(model):
        module.tokens_per_adapter = tokens_per_adapter


def _split_sizes_to_list(splits) -> Optional[List[int]]:
    """Normalize a dispatcher split spec to the list form ``all_to_all`` expects."""
    if splits is None:
        return None
    if isinstance(splits, torch.Tensor):
        return splits.tolist()
    return [int(s) for s in splits]


def _co_permute_slot_ids(dispatcher, slot_ids: torch.Tensor, num_local_experts: int) -> torch.Tensor:
    """Apply the dispatcher's token permutation to a per-token adapter-slot vector.

    Mirrors :class:`MoEAlltoAllTokenDispatcher`'s dispatch stages in order —
    local expert-major permute, EP all-to-all, then the local-expert sort — using
    the dispatcher's own recorded metadata, so the result is aligned row-for-row
    with the ``permuted_local_hidden_states`` the experts receive.

    The companion all-to-all carries one int32 per dispatched token, i.e.
    ``2/hidden_size`` of the hidden-state exchange it shadows.
    """
    from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher

    # Checked by class, not by attribute: the all-gather dispatcher records a
    # permutation mapping too, but its stages (TP*EP all-gather, no EP
    # all-to-all, no local-expert chunk sort) are not the ones replayed here.
    if not isinstance(dispatcher, MoEAlltoAllTokenDispatcher):
        raise NotImplementedError(
            f"Multi-LoRA on MoE experts replays MoEAlltoAllTokenDispatcher's permutation "
            f"stages; got {type(dispatcher).__name__}. Use moe_token_dispatcher_type='alltoall'."
        )
    slot_ids = slot_ids.index_select(0, dispatcher.reversed_local_input_permutation_mapping)

    if getattr(dispatcher, "ep_size", 1) > 1:
        slot_ids = all_to_all(
            dispatcher.ep_group,
            slot_ids,
            _split_sizes_to_list(dispatcher.output_splits),
            _split_sizes_to_list(dispatcher.input_splits),
        )

    # Expert TP is rejected at layer construction; assert here too because the
    # dispatcher would otherwise have gathered rows this vector does not cover.
    if getattr(dispatcher, "tp_size", 1) > 1:
        raise NotImplementedError(
            "Multi-LoRA on MoE experts requires expert_tensor_parallel_size=1, but the token "
            f"dispatcher reports expert TP size {dispatcher.tp_size}."
        )

    if num_local_experts > 1:
        slot_ids, _ = sort_chunks_by_idxs(
            slot_ids,
            dispatcher.num_global_tokens_per_local_expert.ravel(),
            dispatcher.sort_input_by_local_experts,
            fused=False,
        )
    return slot_ids


def _build_expert_slot_routing(
    dispatcher,
    tokens_per_expert: Union[torch.Tensor, Sequence[int]],
    tokens_per_adapter: torch.Tensor,
    n_adapters: int,
    num_local_experts: int,
    device: torch.device,
) -> ExpertSlotRouting:
    """Derive the per-(slot, expert) segmentation of one MoE layer's dispatched tokens.

    The checks below run before the companion all-to-all in
    :func:`_co_permute_slot_ids`, so they can only fail together on all ranks —
    they depend on the micro-batch and on group sizes, which are uniform within a
    tensor-parallel group. A rank-local data corruption that tripped one of them
    on a single rank would hang its expert-parallel peers in that all-to-all
    rather than surfacing the error; every configuration-level requirement is
    therefore checked at layer construction instead, where failure is uniform.
    """
    slot_ids = torch.repeat_interleave(
        torch.arange(n_adapters, device=device, dtype=torch.int32),
        tokens_per_adapter.to(device=device),
    )

    # ``tokens_per_adapter`` counts the whole micro-batch, while a MoE layer's
    # input is sequence-parallel sharded. The shard is contiguous in the same
    # flattened order the dense layer's spans assume, so slicing by TP rank
    # recovers this rank's slot ids.
    local_tokens = int(dispatcher.hidden_shape_before_permute[0])
    if slot_ids.shape[0] != local_tokens:
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        if slot_ids.shape[0] != local_tokens * tp_size:
            raise RuntimeError(
                f"Cannot map {slot_ids.shape[0]} adapter-slot token ids onto {local_tokens} "
                f"local MoE tokens with tensor_model_parallel_size={tp_size}. Check that "
                f"set_tokens_per_adapter_slot() was given this micro-batch's token counts."
            )
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        slot_ids = slot_ids.narrow(0, tp_rank * local_tokens, local_tokens)

    slot_ids = _co_permute_slot_ids(dispatcher, slot_ids, num_local_experts)

    if isinstance(tokens_per_expert, torch.Tensor):
        per_expert = tokens_per_expert.to(device=device, dtype=torch.long)
    else:
        per_expert = torch.tensor(list(tokens_per_expert), device=device, dtype=torch.long)
    expert_ids = torch.repeat_interleave(torch.arange(num_local_experts, device=device, dtype=torch.long), per_expert)
    if expert_ids.shape[0] != slot_ids.shape[0]:
        raise RuntimeError(
            f"Dispatched token count mismatch: tokens_per_expert sums to {expert_ids.shape[0]} "
            f"but the co-permuted slot ids cover {slot_ids.shape[0]} tokens."
        )

    # Slot-major key so a stable sort yields contiguous (slot, expert) groups
    # while preserving the dispatcher's expert order inside each slot.
    keys = slot_ids.long() * num_local_experts + expert_ids
    sort_idx = torch.argsort(keys, stable=True)
    inverse_idx = torch.empty_like(sort_idx)
    inverse_idx.scatter_(0, sort_idx, torch.arange(sort_idx.shape[0], device=device))

    counts = torch.bincount(keys, minlength=n_adapters * num_local_experts)
    return ExpertSlotRouting(
        sort_idx=sort_idx,
        inverse_idx=inverse_idx,
        group_offsets=counts.cumsum(dim=0, dtype=torch.int32),
        slot_token_counts=counts.view(n_adapters, num_local_experts).sum(dim=1),
        num_tokens=int(sort_idx.shape[0]),
    )


def _make_slot_routing_hook(moe_layer: nn.Module, expert_layers: List[MultiLoRAGroupedExpertLinear]):
    """Build the forward pre-hook that publishes slot routing to one MoE layer's adapters.

    The routing is rebuilt from the layer's current ``tokens_per_adapter``, so an
    activation recompute must happen while that still describes the micro-batch
    being recomputed. That holds without pipelining (each micro-batch's backward
    immediately follows its forward) — which is the only supported multi-LoRA
    configuration, since weight sync also requires
    ``pipeline_model_parallel_size == 1``. A pipelined schedule would interleave a
    later micro-batch's forward before the earlier one's recompute and would need
    the counts carried on the graph instead.
    """

    def hook(module: nn.Module, args: Tuple[Any, ...]) -> None:
        if not any(layer._adapter_enabled for layer in expert_layers):
            return None
        if len(args) < 2:
            raise RuntimeError(
                f"Expected the experts module to be called as (hidden_states, tokens_per_expert, "
                f"...); got {len(args)} positional argument(s)."
            )
        hidden_states, tokens_per_expert = args[0], args[1]
        reference = expert_layers[0]
        tokens_per_adapter = reference.tokens_per_adapter
        if tokens_per_adapter is None:
            raise RuntimeError(
                "set_tokens_per_adapter_slot(model, adapter_token_counts) must run before every "
                "forward when MoE experts carry multi-LoRA adapters."
            )
        routing = _build_expert_slot_routing(
            moe_layer.token_dispatcher,
            tokens_per_expert,
            tokens_per_adapter,
            reference.n_adapters,
            reference.num_local_experts,
            hidden_states.device,
        )
        for layer in expert_layers:
            layer.expert_slot_routing = routing
        return None

    return hook


def install_moe_slot_routing(model) -> int:
    """Install per-MoE-layer slot routing for wrapped grouped expert linears.

    A forward pre-hook on each MoE layer's ``experts`` module runs after the
    token dispatcher has permuted and exchanged tokens but before the expert
    GEMMs, which is the only point where both the dispatcher's permutation
    metadata and the final row order are available.

    Idempotent, and a no-op on models whose expert linears carry no adapters.
    Returns the number of MoE layers hooked.
    """
    from megatron.core.transformer.moe.moe_layer import BaseMoELayer

    installed = 0
    for model_chunk in model if isinstance(model, list) else [model]:
        for module in model_chunk.modules():
            if not isinstance(module, BaseMoELayer):
                continue
            experts = getattr(module, "experts", None)
            if experts is None:
                continue
            expert_layers = [m for m in experts.modules() if isinstance(m, MultiLoRAGroupedExpertLinear)]
            if not expert_layers:
                continue
            if getattr(experts, "_multi_lora_slot_routing_handle", None) is not None:
                continue
            experts._multi_lora_slot_routing_handle = experts.register_forward_pre_hook(
                _make_slot_routing_hook(module, expert_layers)
            )
            installed += 1
    return installed


def init_adapter_slot(model, idx: int, rank: int, alpha: float) -> None:
    """Claim slot ``idx`` across every multi-LoRA layer for an adapter.

    A model-wide adapter is the set of slot-``idx`` chunks across all layers;
    this initialises that set with the given ``rank``/``alpha``. Thin iterator
    over the model — per-slot setup (rank/alpha bookkeeping + rank-mask
    invariant) lives on the layer itself in
    :meth:`MultiLoRALinear.init_adapter_slot` /
    """
    for module in _iter_multi_lora_modules(model):
        module.init_adapter_slot(idx, rank, alpha)


def clear_adapter_slot(model, idx: int) -> None:
    """Release slot ``idx`` across every multi-LoRA layer (zero alpha, re-init weights)."""
    for module in _iter_multi_lora_modules(model):
        module.clear_adapter_slot(idx)


def load_adapter(model, idx: int, state_dict: Dict[str, torch.Tensor]) -> int:
    """Load Megatron-shard format adapter weights into slot ``idx``.

    ``state_dict`` must use the *Megatron-native* names produced by saving
    while ``expose_adapter_slot(model, idx)`` is active — i.e. the same
    layout this function constructs to look them up. Each tensor is the
    local TP/PP shard, copied straight into the slot parameter with no
    gather, scatter, or rank-padding logic.

    Saving from slot A and loading into slot B is fine because the slot
    index is stripped from the name (``...adapter.linear_in.weight``)
    while ``expose_adapter_slot`` is active.

    Returns the number of tensors loaded (for logging / sanity checks).
    Raises ``KeyError`` when the checkpoint and the model's adapter params do
    not match exactly in either direction (missing or unconsumed tensors).
    """
    loaded = 0
    missing = []
    seen = set()
    with expose_adapter_slot(model, idx):
        models = model if isinstance(model, list) else [model]
        for chunk in models:
            for name, param in chunk.named_parameters():
                if ".adapter." not in name:
                    continue
                seen.add(name)
                if name not in state_dict:
                    missing.append(name)
                    continue
                src = state_dict[name].to(device=param.device, dtype=param.dtype)
                param.data.copy_(src)
                loaded += 1
    # A partial load silently leaves the unmatched slots at random init (e.g.
    # resuming after target_modules changed) — a zero-delta / wrong adapter with
    # no error. Fail loud instead.
    if missing:
        raise KeyError(
            f"load_adapter(slot={idx}): {len(missing)} adapter param(s) absent from the "
            f"checkpoint (e.g. {missing[0]}); they would stay at random init. "
            f"Loaded {loaded}. Did target_modules change since the checkpoint was saved?"
        )
    # The reverse mismatch is just as silent: checkpoint tensors no module
    # consumed (e.g. target_modules shrank since the save) would drop part of
    # the trained adapter without an error.
    unused = [key for key in state_dict if ".adapter." in key and key not in seen]
    if unused:
        raise KeyError(
            f"load_adapter(slot={idx}): {len(unused)} checkpoint tensor(s) matched no "
            f"adapter param (e.g. {unused[0]}); part of the saved adapter would be "
            f"silently dropped. Did target_modules change since the checkpoint was saved?"
        )
    return loaded


def expose_adapter_slot(model, idx: int):
    """Context manager that temporarily exposes one adapter slot as ``.adapter``.

    Used by two consumers:

    * The bridge's ``export_adapter_weights`` looks for ``.adapter.linear_in.weight``
      (single-LoRA layout) on each wrapped module.
    * Megatron-native save/load walk ``model.named_parameters()`` and want names
      that don't contain the slot index, so saving from slot ``A`` and loading into
      slot ``B`` produces matching keys.

    Export contract: tensors are exported max-rank padded with ``.dim == max_rank``,
    so the exposed ``.alpha`` is set to ``alpha * max_rank / rank`` — consumers
    computing ``alpha / dim`` apply the slot's runtime scaling. Restored on exit.

    ``MultiLoRALinear`` is handled via the
    common ``.adapters`` ModuleList — duck-typed rather than ``isinstance``-checked
    so future multi-LoRA module types are picked up automatically.
    """
    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        modules = list(_iter_multi_lora_modules(model))
        saved = {}
        saved_alphas = {}
        for m in modules:
            if "adapters" in m._modules:
                saved[id(m)] = m._modules.pop("adapters")
                adapter = saved[id(m)][idx]
                saved_alphas[id(m)] = adapter.alpha
                adapter.alpha = float(m.alpha_values[idx]) * m.max_rank / float(m.rank_values[idx])
                m.adapter = adapter

        # try/finally: an exception in the body (e.g. an export/save error, which
        # happens on the weight-push path) must still restore the ModuleList,
        # otherwise `adapters` stays detached and every later forward/save/
        # named_parameters is silently corrupted.
        try:
            yield
        finally:
            for m in modules:
                if id(m) in saved:
                    if "adapter" in m._modules:
                        del m._modules["adapter"]
                    m._modules["adapters"] = saved[id(m)]
                    saved[id(m)][idx].alpha = saved_alphas[id(m)]

    return _ctx()


def hide_adapters(model):
    """Context manager that temporarily hides all adapter params from the model.

    Used during base checkpoint loading so the bridge doesn't try to map
    adapter parameters to HF weights.
    """
    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        modules = list(_iter_multi_lora_modules(model))
        saved = {}
        for m in modules:
            if isinstance(m, MultiLoRALinear) and "adapters" in m._modules:
                saved[id(m)] = m._modules.pop("adapters")
        # try/finally: restore even if base-checkpoint loading raises, else the
        # adapters stay hidden from the model permanently.
        try:
            yield
        finally:
            for m in modules:
                if id(m) in saved:
                    m._modules["adapters"] = saved[id(m)]

    return _ctx()
