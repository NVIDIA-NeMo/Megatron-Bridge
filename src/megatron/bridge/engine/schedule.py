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
"""Model adapter and schedule driver: strict ``PreparedMicrobatch`` in, ``ModelOutput`` out.

The model is always called with ``labels=target_ids``, so the recipe's configured cross-entropy kernel
returns per-token ``-log p(target)`` with the TP reductions done inside; scoring and training share
this call. The call uses the keyword contract shared by Megatron's GPT and hybrid language models.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from typing import Any

import torch
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import MegatronModule
from megatron.core.utils import get_model_config

from megatron.bridge.data.iterator_utils import make_data_iterator_list
from megatron.bridge.engine.datum import ModelOutput, PreparedMicrobatch
from megatron.bridge.engine.router_replay import RouterReplayState
from megatron.bridge.training.gpt_step import _prepare_packed_padding_mask
from megatron.bridge.training.utils.pg_utils import pipeline_stage_roles


def model_loss_mask(microbatch: PreparedMicrobatch, mtp_target_scope: str) -> torch.Tensor:
    """The ``loss_mask`` handed to the model, which only the MTP block reads (the main NLL is unmasked).

    ``"primary"`` selects the primary objective's positions, ``"all_valid"`` every non-padded position.
    Padding is never a target, even when a custom batch carries ``loss_mask=1`` there.
    """
    if mtp_target_scope == "primary":
        selected = microbatch.active_mask
    elif mtp_target_scope == "all_valid":
        selected = ~microbatch.padding_mask
    else:
        raise ValueError(f"Unknown mtp_target_scope {mtp_target_scope!r}.")
    return selected.to(microbatch.loss_mask.dtype)


def uses_mrope(model_config: object) -> bool:
    """Whether the model built from ``model_config`` takes M-RoPE position ids ``[3, B, T]``.

    Pass the recipe's model config, as Bridge's own step does: a model built from a builder-backed
    config carries only the nested ``TransformerConfig``, which has no ``position_embedding_type``.
    """
    return getattr(model_config, "position_embedding_type", None) == "mrope"


def call_model(
    model: MegatronModule,
    microbatch: PreparedMicrobatch,
    pg_collection: ProcessGroupCollection,
    *,
    mtp_target_scope: str,
    mrope: bool,
    output_head: str = "lm",
) -> torch.Tensor:
    """Run one microbatch through the model: per-token NLL ``[B, T_local]`` on the labels path
    (``output_head="lm"``), or FP32 per-token values ``[B, T_local]`` from the value head. ``mrope``
    (``uses_mrope`` of the recipe's model config) hands the model ``[3, B, T_local]`` position ids."""
    config = get_model_config(model)
    padding_mask = _prepare_packed_padding_mask(
        microbatch.padding_mask, config=config, model=model, pg_collection=pg_collection
    )
    position_ids = microbatch.position_ids
    if mrope:
        # Text tokens have the same position on the temporal, height and width axes: [3, B, T_local].
        # TODO(multimodal): image and video tokens need grid positions and pixel inputs from the Datum.
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
    common = dict(
        input_ids=microbatch.input_ids,
        position_ids=position_ids,
        attention_mask=None,
        packed_seq_params=microbatch.packed_seq_params,
        padding_mask=padding_mask,
    )
    if output_head == "lm":
        return model(labels=microbatch.target_ids, loss_mask=model_loss_mask(microbatch, mtp_target_scope), **common)
    if output_head == "value":
        # No labels: the replaced output layer returns the per-token values as "logits" [B, T, 1].
        # Only the model chunk that ends the pipeline has that layer; the others return hidden states for
        # the next chunk.
        output = model(labels=None, **common)
        _, is_last = pipeline_stage_roles(model)
        return output[..., 0] if is_last else output
    raise ValueError(f"Unknown output_head {output_head!r}.")


def to_model_output(raw: torch.Tensor, output_head: str = "lm") -> ModelOutput:
    """Adapt the raw model return into the ``ModelOutput`` contract: NLL -> logprobs, or values."""
    if output_head == "lm":
        return ModelOutput(logprobs=-raw)
    if output_head == "value":
        return ModelOutput(logprobs=None, values=raw)
    raise ValueError(f"Unknown output_head {output_head!r}.")


def scored_tensor(output: ModelOutput) -> torch.Tensor:
    """The per-token score a scoring call returns: logprobs on an LM engine, values on a value-head engine."""
    return output.values if output.logprobs is None else output.logprobs


def _make_scoring_step(
    pg_collection: ProcessGroupCollection,
    mtp_target_scope: str,
    mrope: bool,
    output_head: str,
    router_replay: RouterReplayState | None,
) -> Callable:
    def forward_step_func(
        data_iterator: Iterator[PreparedMicrobatch], model: MegatronModule
    ) -> tuple[torch.Tensor, Callable]:
        microbatch = next(data_iterator)
        if router_replay is not None:
            router_replay.arm_forward(microbatch, model)
        raw = call_model(
            model, microbatch, pg_collection, mtp_target_scope=mtp_target_scope, mrope=mrope, output_head=output_head
        )

        def collect(output_tensor: torch.Tensor, non_loss_data: bool = False) -> torch.Tensor:
            return scored_tensor(to_model_output(output_tensor, output_head))[0].detach().float()

        return raw, collect

    return forward_step_func


def run_forward_only(
    model: list[MegatronModule],
    microbatches: list[PreparedMicrobatch],
    *,
    pg_collection: ProcessGroupCollection,
    mrope: bool,
    mtp_target_scope: str = "primary",
    output_head: str = "lm",
    router_replay: RouterReplayState | None = None,
) -> list[torch.Tensor]:
    """Score microbatches with the schedule in forward-only mode; one ``[T_local]`` logprob tensor each.
    With pipeline parallelism only the last stage has scores; the other stages return an empty list.

    Runs in eval mode under ``no_grad`` so dropout, router expert-bias updates and auxiliary-loss
    bookkeeping are untouched; an open accumulation window is preserved. ``router_replay`` (the
    engine's ``RouterReplayState``) arms the MoE routers per microbatch when it carries routing.
    ``mrope`` is ``uses_mrope`` of the recipe's model config.
    """
    if not microbatches:
        return []
    config = get_model_config(model[0])
    schedule = get_forward_backward_func(
        pp_size=pg_collection.pp.size(), vp_size=config.virtual_pipeline_model_parallel_size
    )
    was_training = [chunk.training for chunk in model]
    for chunk in model:
        chunk.eval()
    try:
        with torch.no_grad():
            store: list[Any] = schedule(
                forward_step_func=_make_scoring_step(
                    pg_collection, mtp_target_scope, mrope, output_head, router_replay
                ),
                # With virtual pipeline stages: one iterator per model chunk, each over the same microbatches.
                data_iterator=make_data_iterator_list(model, iter(microbatches)),
                model=model,
                num_microbatches=len(microbatches),
                # The pack length before the CP split: the pipelined schedule divides by the CP size itself.
                seq_length=microbatches[0].input_ids.shape[1] * pg_collection.cp.size(),
                micro_batch_size=microbatches[0].input_ids.shape[0],
                forward_only=True,
                collect_non_loss_data=True,
                p2p_communicator=P2PCommunicator(pp_group=pg_collection.pp, config=config),
                pg_collection=pg_collection,
            )
    finally:
        if router_replay is not None:
            router_replay.clear()
        for chunk, flag in zip(model, was_training):
            chunk.train(flag)
    return store


def run_forward_backward(
    model: list[MegatronModule],
    microbatches: list[PreparedMicrobatch],
    loss_method: Callable[[PreparedMicrobatch, ModelOutput], Any],
    *,
    pg_collection: ProcessGroupCollection,
    mrope: bool,
    mtp_target_scope: str = "primary",
    output_head: str = "lm",
    router_replay: RouterReplayState | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Forward/backward over the microbatches, accumulating raw FP32 gradient sums in the DDP buffers.

    Gradient finalization is suppressed for the call: ``finalize_model_grads_func``, ``grad_sync_func``
    and ``no_sync_func`` are nulled on the model config and the DDP wrappers run under ``no_sync()``,
    so no DP reduce-scatter and no ``1/num_tokens`` scaling happen here (the engine does both once at
    ``optim_step``). The loss method returns an additive scalar per microbatch, optionally paired with
    a dictionary of additive scalar metrics; with ``calculate_per_token_loss=True`` the schedule leaves
    the scalar unscaled. ``router_replay`` (the engine's ``RouterReplayState``) arms the MoE routers
    per microbatch when it carries routing: replay mode for the forward, queued replay for the
    recompute in the backward. ``mrope`` is ``uses_mrope`` of the recipe's model config.

    Returns:
        ``(loss_sum, num_loss_tokens, metrics)`` for this rank's microbatches, un-reduced, detached;
        ``metrics`` sums each key over the microbatches.
    """
    if not microbatches:
        raise ValueError("forward_backward needs at least one microbatch on every DP replica.")
    config = get_model_config(model[0])
    schedule = get_forward_backward_func(
        pp_size=pg_collection.pp.size(), vp_size=config.virtual_pipeline_model_parallel_size
    )
    saved = (config.finalize_model_grads_func, config.grad_sync_func, config.no_sync_func)
    config.finalize_model_grads_func = None
    config.grad_sync_func = None
    config.no_sync_func = None
    device = microbatches[0].input_ids.device
    loss_sum = torch.zeros((), dtype=torch.float32, device=device)
    num_loss_tokens = torch.zeros((), dtype=torch.int64, device=device)
    metrics: dict[str, torch.Tensor] = {}

    def forward_step_func(data_iterator: Iterator[PreparedMicrobatch], chunk: MegatronModule) -> tuple:
        microbatch = next(data_iterator)
        if router_replay is not None:
            router_replay.arm_forward(microbatch, chunk)
        raw = call_model(
            chunk, microbatch, pg_collection, mtp_target_scope=mtp_target_scope, mrope=mrope, output_head=output_head
        )
        if router_replay is not None:
            router_replay.arm_backward(microbatch, chunk)

        def loss_func(output_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
            loss, extra = split_loss_return(loss_method(microbatch, to_model_output(output_tensor, output_head)))
            tokens = microbatch.active_mask.sum()
            loss_sum.add_(loss.detach().float())
            num_loss_tokens.add_(tokens)
            for key, value in extra.items():
                metrics[key] = metrics.get(key, 0.0) + value.detach().float()
            # Three-tuple: with calculate_per_token_loss=True the schedule keeps the sum unscaled.
            return loss, tokens.to(torch.int32), {}

        return raw, loss_func

    for chunk in model:
        chunk.train()
    try:
        with contextlib.ExitStack() as stack:
            for chunk in model:
                if hasattr(chunk, "no_sync"):
                    stack.enter_context(chunk.no_sync())
            schedule(
                forward_step_func=forward_step_func,
                # With virtual pipeline stages: one iterator per model chunk, each over the same microbatches.
                data_iterator=make_data_iterator_list(model, iter(microbatches)),
                model=model,
                num_microbatches=len(microbatches),
                # The pack length before the CP split: the pipelined schedule divides by the CP size itself.
                seq_length=microbatches[0].input_ids.shape[1] * pg_collection.cp.size(),
                micro_batch_size=microbatches[0].input_ids.shape[0],
                forward_only=False,
                p2p_communicator=P2PCommunicator(pp_group=pg_collection.pp, config=config),
                pg_collection=pg_collection,
            )
    finally:
        if router_replay is not None:
            router_replay.clear()
        config.finalize_model_grads_func, config.grad_sync_func, config.no_sync_func = saved
    return loss_sum, num_loss_tokens, metrics


def split_loss_return(result: Any) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Validate a loss method's return: a scalar tensor, or ``(scalar, {name: scalar tensor})``."""
    loss, extra = (result, {}) if not isinstance(result, tuple) else result
    if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
        raise ValueError("A loss method must return a scalar tensor (the additive loss sum), optionally with a dict.")
    if not isinstance(extra, dict) or any(
        not isinstance(key, str) or not isinstance(value, torch.Tensor) or value.ndim != 0
        for key, value in extra.items()
    ):
        raise ValueError("Loss metrics must be a dict of str -> scalar tensor (additive sums or counts).")
    return loss, extra
