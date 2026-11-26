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

"""Looped Inkling2 (Inkling 2 Mini): a core of decoder blocks run several times with shared weights.

The decoder runs the prelude blocks ``[0, core_start)`` once, the core blocks ``[core_start, core_end)`` once per
pass, and the coda blocks after the core once. Every pass starts again from the AttnRes state the prelude left.
From the second pass on, each core sublayer also reads a message from the previous pass's block outputs (the
"bank": the AttnRes blocks the core produced, completed blocks and the final partial block) through Block Attention
Residuals with its own pseudo-queries, and mixes it into its input through a low-rank controller
(:class:`Inkling2LoopController`).

Routing replay (``moe_enable_routing_replay``): a core MoE layer routes once per pass, so a looped model's replay
targets are ``[runs, tokens, topk]`` for every MoE layer (one run outside the core, ``inkling2_loop_num_passes`` in
it, in pass order). Each layer run replays its own slice, fixed when the forward builds the run, so an
activation-recompute rerun replays the same routing in whatever order backward reruns the layer runs; Megatron's
per-call recompute FIFO is not used.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Callable

import torch
import torch.nn.functional as F
from megatron.core import tensor_parallel
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.models.hybrid.hybrid_block import HybridStack
from megatron.core.transformer.module import MegatronModule, mark_keep_in_fp32
from megatron.core.transformer.utils import precision_aware_checkpoint
from megatron.core.utils import WrappedTensor, make_viewless_tensor
from torch import Tensor, nn

from megatron.bridge.models.inkling.modeling_inkling2.ops import _compiled


# Order of a core block's controllers, of its message queries and of the messages a layer reads.
LOOP_MODULES = ("attn", "k", "v", "mlp")


def is_looped(config) -> bool:
    """Whether ``config`` describes a looped Inkling2."""
    return getattr(config, "inkling2_loop_num_passes", 1) > 1


def loop_core_blocks(config) -> range:
    """The looped decoder blocks (empty for an unlooped model)."""
    if not is_looped(config):
        return range(0)
    return range(config.inkling2_loop_core_start, config.inkling2_loop_core_end)


def decoder_block_runs(config, num_blocks: int) -> list[tuple[int, int]]:
    """``(block, pass)`` of every decoder-block run, in execution order: prelude, core x passes, coda.

    This is also the order of an inference engine's per-visit routing record.
    """
    core = loop_core_blocks(config)
    if not core:
        return [(block, 0) for block in range(num_blocks)]
    runs = [(block, 0) for block in range(core.start)]
    for pass_index in range(config.inkling2_loop_num_passes):
        runs += [(block, pass_index) for block in core]
    return runs + [(block, 0) for block in range(core.stop, num_blocks)]


def _controller_norm(dest: Tensor, message: Tensor, dtype: torch.dtype) -> Tensor:
    pair = torch.stack((dest.float(), message.float()), dim=-2)
    pair = pair * torch.rsqrt(pair.square().mean(-1, keepdim=True) + 1e-6)
    return pair.to(dtype).flatten(-2)


def _controller_mix(dest: Tensor, message: Tensor, gates: Tensor) -> Tensor:
    gain, forget = gates.float().chunk(2, dim=-1)
    return (1.0 - torch.tanh(forget)) * dest.float() + torch.tanh(gain) * message.float()


class Inkling2LoopController(MegatronModule):
    """Mix a cross-pass message into a core sublayer input.

    ``(1 - tanh(f)) * dest + tanh(g) * message`` with ``[g, f] = h @ [gain; forget]^T`` in FP32, and
    ``h = [rmsnorm(dest), rmsnorm(message)] @ down^T`` (weightless RMSNorms, eps 1e-6) in the activation dtype.
    The rank-``inkling2_loop_gate_rank`` weights are replicated over tensor parallelism; under sequence parallelism
    each rank sees different tokens, so their gradients add across the TP group.
    """

    def __init__(self, config) -> None:
        super().__init__(config=config)
        hidden, rank = config.hidden_size, config.inkling2_loop_gate_rank
        device, dtype = torch.cuda.current_device(), config.params_dtype
        self.down_weight = nn.Parameter(torch.empty(rank, 2 * hidden, device=device, dtype=dtype))
        self.gain_weight = nn.Parameter(torch.empty(hidden, rank, device=device, dtype=dtype))
        self.forget_weight = nn.Parameter(torch.empty(hidden, rank, device=device, dtype=dtype))
        for parameter in (self.down_weight, self.gain_weight, self.forget_weight):
            if config.perform_initialization:
                config.init_method(parameter)
            parameter.sequence_parallel = True

    def forward(self, dest: Tensor, message: Tensor) -> Tensor:
        """Return the FP32 mixture of ``dest`` and ``message``, both ``[S(/TP), B, H]``."""
        compiled = self.config.inkling2_compile_ops
        norm = _compiled(_controller_norm) if compiled else _controller_norm
        mix = _compiled(_controller_mix) if compiled else _controller_mix
        hidden = F.linear(norm(dest, message, self.down_weight.dtype), self.down_weight)
        gates = F.linear(hidden.float(), torch.cat((self.gain_weight, self.forget_weight)).float())
        return mix(dest, message, gates)


class Inkling2LoopQueries(MegatronModule):
    """The FP32 cross-pass message queries of the core blocks.

    ``middle_cycle_cross_pass_queries`` ``[2 * n_core, H]``: rows ``2l`` and ``2l + 1`` read core block ``l``'s
    attention-input and MoE-input messages; ``middle_cycle_cross_pass_kv_queries``: its key- and value-input messages.
    """

    def __init__(self, config) -> None:
        super().__init__(config=config)
        shape = (2 * len(loop_core_blocks(config)), config.hidden_size)
        device = torch.cuda.current_device()
        self._keep_in_float32_parameter_names = ("middle_cycle_cross_pass_queries", "middle_cycle_cross_pass_kv_queries")
        for name in self._keep_in_float32_parameter_names:
            parameter = nn.Parameter(torch.zeros(*shape, device=device, dtype=torch.float32))
            parameter.sequence_parallel = True
            self.register_parameter(name, mark_keep_in_fp32(parameter))

    def for_block(self, block: int) -> Tensor:
        """``[4, H]``: the attn, k, v and mlp message queries of core block ``block`` (:data:`LOOP_MODULES`)."""
        index = block - self.config.inkling2_loop_core_start
        queries, kv_queries = self.middle_cycle_cross_pass_queries, self.middle_cycle_cross_pass_kv_queries
        return torch.stack((queries[2 * index], kv_queries[2 * index], kv_queries[2 * index + 1], queries[2 * index + 1]))


def _router_replay(layer: nn.Module):
    return getattr(getattr(getattr(layer, "mlp", None), "router", None), "router_replay", None)


def _replay_slice(layer: nn.Module, pass_index: int) -> Tensor | None:
    """The ``[tokens, topk]`` routing a core MoE layer replays in pass ``pass_index``, or None without replay."""
    from megatron.core.transformer.moe.router_replay import RouterReplayAction

    replay = _router_replay(layer)
    if replay is None or replay.router_replay_action != RouterReplayAction.REPLAY_FORWARD:
        return None
    target = replay.target_topk_idx
    if target is None:
        return None
    if target.ndim != 3:
        raise ValueError(f"Looped routing replay expects [runs, tokens, topk] targets, got {tuple(target.shape)}")
    return target[pass_index]


def _drop_replay_backward_entries(layers) -> None:
    """Drop the entry ``set_target_indices`` queued for Megatron's recompute FIFO: the loop's recomputes replay their
    own slices, so nothing would pop it."""
    from megatron.core.transformer.moe.router_replay import RouterReplayAction

    for layer in layers:
        replay = _router_replay(layer)
        if replay is None or replay.router_replay_action != RouterReplayAction.REPLAY_FORWARD:
            continue
        backward = replay.replay_backward_list
        if backward and backward[-1] is replay.target_topk_idx:
            backward.pop()


class _Replaying:
    """Route ``layer``'s MoE through ``indices`` (``[tokens, topk]``) for one call, if set."""

    def __init__(self, layer: nn.Module, indices: Tensor | None) -> None:
        self.replay = _router_replay(layer) if indices is not None else None
        self.indices = indices

    def __enter__(self):
        if self.replay is not None:
            from megatron.core.transformer.moe.router_replay import RouterReplayAction

            self.saved = self.replay.target_topk_idx, self.replay.router_replay_action
            self.replay.target_topk_idx = self.indices
            self.replay.router_replay_action = RouterReplayAction.REPLAY_FORWARD
        return self

    def __exit__(self, *exc) -> bool:
        if self.replay is not None:
            self.replay.target_topk_idx, self.replay.router_replay_action = self.saved
        return False


class Inkling2LoopStack(HybridStack):
    """``HybridStack`` that runs a looped Inkling2's core blocks once per pass.

    Hybrid layers ``2 * b`` and ``2 * b + 1`` hold decoder block ``b``. An unlooped config runs ``HybridStack``'s own
    forward. Activation recompute (``recompute_granularity="full"``) checkpoints layer runs, not physical layers:
    ``uniform`` groups ``recompute_num_layers`` consecutive layer runs of one segment (prelude, one pass, coda);
    ``block`` checkpoints the first ``recompute_num_layers`` layer runs one by one.
    """

    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)
        if is_looped(config):
            # A child module, so the stack's sharded state dict saves it with the layers.
            self.loop_queries = Inkling2LoopQueries(config)

    def forward(self, hidden_states, attention_mask, inference_context=None, rotary_pos_emb=None, **kwargs):
        if not is_looped(self.config):
            return super().forward(hidden_states, attention_mask, inference_context, rotary_pos_emb, **kwargs)
        if inference_context is not None or kwargs.get("inference_params") is not None:
            raise ValueError("Looped Inkling2 does not support Megatron's KV-cache inference")
        config = self.config
        if config.fp4 or (config.fp8 and config.fp8_recipe == "delayed"):
            raise ValueError("Looped Inkling2 supports FP8 only with per-layer recipes")
        packed_seq_params = kwargs.get("packed_seq_params")
        padding_mask = kwargs.get("padding_mask")
        if not self.pre_process:
            hidden_states = self.input_tensor
        if isinstance(hidden_states, WrappedTensor):
            hidden_states = hidden_states.unwrap()
        if len(self.layers) != config.num_layers:
            raise ValueError("Looped Inkling2 builds every decoder layer on one pipeline stage")

        core = loop_core_blocks(config)
        checkpointing = config.recompute_granularity == "full" and self.training

        def layer_run(layer_index: int, pass_index: int) -> Callable:
            layer = self.layers[layer_index]
            replay = _replay_slice(layer, pass_index)

            def run(state: Tensor, bank: Tensor | None, queries: Tensor | None) -> Tensor:
                quant = get_fp8_context(config, layer.layer_number - 1) if config.fp8 else nullcontext()
                with quant, _Replaying(layer, replay):
                    output, _ = layer(
                        state,
                        attention_mask=attention_mask,
                        packed_seq_params=packed_seq_params,
                        padding_mask=padding_mask,
                        loop_bank=bank,
                        loop_queries=queries,
                    )
                return output

            return run

        done = 0

        def segment(runs: list[tuple[int, int]], state: Tensor, bank: Tensor | None) -> Tensor:
            nonlocal done
            if not runs:
                return state
            steps = [
                (layer_run(index, pass_index), self.loop_queries.for_block(index // 2) if pass_index else None)
                for index, pass_index in runs
            ]
            if not checkpointing:
                for run, queries in steps:
                    state = run(state, bank if queries is not None else None, queries)
                return state
            count = config.recompute_num_layers or 1
            if config.recompute_method == "block":
                groups = [[step] for step in steps]
                checkpointed = [done + k < count for k in range(len(steps))]
            else:
                groups = [steps[k : k + count] for k in range(0, len(steps), count)]
                checkpointed = [True] * len(groups)
            done += len(steps)
            for group, use_checkpoint in zip(groups, checkpointed):

                def chunk(state, bank, *queries, group=group):
                    for (run, _), query in zip(group, queries):
                        state = run(state, bank if query is not None else None, query)
                    return state

                queries = [query for _, query in group]
                if not use_checkpoint:
                    state = chunk(state, bank, *queries)
                    continue
                # Checkpointed functions take tensors only; an absent bank or query is an empty placeholder.
                empty = state.new_empty(0)

                def checkpointed_chunk(state, bank, *queries, chunk=chunk):
                    return chunk(
                        state,
                        bank if bank.numel() else None,
                        *[query if query.numel() else None for query in queries],
                    )

                state = precision_aware_checkpoint(
                    checkpointed_chunk,
                    config,
                    self.pg_collection.tp,
                    state,
                    bank if bank is not None else empty,
                    *[query if query is not None else empty for query in queries],
                    distribute_saved_activations=config.distribute_saved_activations,
                )
            return state

        _drop_replay_backward_entries(self.layers)
        hidden = config.hidden_size
        rng = tensor_parallel.get_cuda_rng_tracker().fork() if config.sequence_parallel else nullcontext()
        with rng:
            state = segment([(index, 0) for index in range(2 * core.start)], hidden_states, None)
            prelude = state
            # The prelude state is [partial, *completed]; the core's first block closes the prelude's partial block.
            prelude_completed = prelude.shape[-1] // hidden - 1
            bank = None
            for pass_index in range(config.inkling2_loop_num_passes):
                if pass_index:
                    partial, completed = state[..., :hidden], state[..., hidden:]
                    produced = completed[..., (prelude_completed + 1) * hidden :]
                    bank = torch.cat((produced, partial), dim=-1)
                    state = prelude
                state = segment([(index, pass_index) for index in range(2 * core.start, 2 * core.stop)], state, bank)
            state = segment([(index, 0) for index in range(2 * core.stop, config.num_layers)], state, None)

        if self.post_process and self.post_layer_norm:
            state = self.final_norm(state)
        return make_viewless_tensor(inp=state, requires_grad=state.requires_grad, keep_graph=True)

