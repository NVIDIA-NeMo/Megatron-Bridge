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
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import MegatronModule
from megatron.core.utils import get_model_config

from megatron.bridge.engine.datum import ModelOutput, PreparedMicrobatch
from megatron.bridge.training.gpt_step import _prepare_packed_padding_mask


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


def call_model(
    model: MegatronModule,
    microbatch: PreparedMicrobatch,
    pg_collection: ProcessGroupCollection,
    *,
    mtp_target_scope: str,
) -> torch.Tensor:
    """Run one microbatch through the model; returns per-token NLL ``[B, T_local]`` (labels path)."""
    config = get_model_config(model)
    padding_mask = _prepare_packed_padding_mask(
        microbatch.padding_mask, config=config, model=model, pg_collection=pg_collection
    )
    return model(
        input_ids=microbatch.input_ids,
        position_ids=microbatch.position_ids,
        attention_mask=None,
        labels=microbatch.target_ids,
        loss_mask=model_loss_mask(microbatch, mtp_target_scope),
        packed_seq_params=microbatch.packed_seq_params,
        padding_mask=padding_mask,
    )


def to_model_output(nll: torch.Tensor) -> ModelOutput:
    """Adapt the raw model return (per-token NLL) into the ``ModelOutput`` contract (no logits on this path)."""
    return ModelOutput(logprobs=-nll)


def _make_scoring_step(pg_collection: ProcessGroupCollection, mtp_target_scope: str) -> Callable:
    def forward_step_func(
        data_iterator: Iterator[PreparedMicrobatch], model: MegatronModule
    ) -> tuple[torch.Tensor, Callable]:
        microbatch = next(data_iterator)
        nll = call_model(model, microbatch, pg_collection, mtp_target_scope=mtp_target_scope)

        def collect(output_tensor: torch.Tensor, non_loss_data: bool = False) -> torch.Tensor:
            return to_model_output(output_tensor).logprobs[0].detach().float()

        return nll, collect

    return forward_step_func


def run_forward_only(
    model: list[MegatronModule],
    microbatches: list[PreparedMicrobatch],
    *,
    pg_collection: ProcessGroupCollection,
    mtp_target_scope: str = "primary",
) -> list[torch.Tensor]:
    """Score microbatches with the schedule in forward-only mode; one ``[T_local]`` logprob tensor each.

    Runs in eval mode under ``no_grad`` so dropout, router expert-bias updates and auxiliary-loss
    bookkeeping are untouched; an open accumulation window is preserved.
    """
    if not microbatches:
        return []
    was_training = [chunk.training for chunk in model]
    for chunk in model:
        chunk.eval()
    try:
        with torch.no_grad():
            store: list[Any] = get_forward_backward_func()(
                forward_step_func=_make_scoring_step(pg_collection, mtp_target_scope),
                data_iterator=iter(microbatches),
                model=model,
                num_microbatches=len(microbatches),
                seq_length=microbatches[0].input_ids.shape[1],
                micro_batch_size=microbatches[0].input_ids.shape[0],
                forward_only=True,
                collect_non_loss_data=True,
                pg_collection=pg_collection,
            )
    finally:
        for chunk, flag in zip(model, was_training):
            chunk.train(flag)
    return store


def run_forward_backward(
    model: list[MegatronModule],
    microbatches: list[PreparedMicrobatch],
    loss_method: Callable[[PreparedMicrobatch, ModelOutput], torch.Tensor],
    *,
    pg_collection: ProcessGroupCollection,
    mtp_target_scope: str = "primary",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward/backward over the microbatches, accumulating raw FP32 gradient sums in the DDP buffers.

    Gradient finalization is suppressed for the call: ``finalize_model_grads_func``, ``grad_sync_func``
    and ``no_sync_func`` are nulled on the model config and the DDP wrappers run under ``no_sync()``,
    so no DP reduce-scatter and no ``1/num_tokens`` scaling happen here (the engine does both once at
    ``optim_step``). The loss method returns an additive scalar per microbatch; with
    ``calculate_per_token_loss=True`` the schedule leaves it unscaled.

    Returns:
        ``(loss_sum, num_loss_tokens)`` for this rank's microbatches, un-reduced, detached.
    """
    if not microbatches:
        raise ValueError("forward_backward needs at least one microbatch on every DP replica.")
    config = get_model_config(model[0])
    saved = (config.finalize_model_grads_func, config.grad_sync_func, config.no_sync_func)
    config.finalize_model_grads_func = None
    config.grad_sync_func = None
    config.no_sync_func = None
    device = microbatches[0].input_ids.device
    loss_sum = torch.zeros((), dtype=torch.float32, device=device)
    num_loss_tokens = torch.zeros((), dtype=torch.int64, device=device)

    def forward_step_func(data_iterator: Iterator[PreparedMicrobatch], chunk: MegatronModule) -> tuple:
        microbatch = next(data_iterator)
        nll = call_model(chunk, microbatch, pg_collection, mtp_target_scope=mtp_target_scope)

        def loss_func(output_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
            loss = loss_method(microbatch, to_model_output(output_tensor))
            if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
                raise ValueError("A loss method must return a scalar tensor (the additive loss sum).")
            tokens = microbatch.active_mask.sum()
            loss_sum.add_(loss.detach().float())
            num_loss_tokens.add_(tokens)
            # Three-tuple: with calculate_per_token_loss=True the schedule keeps the sum unscaled.
            return loss, tokens.to(torch.int32), {}

        return nll, loss_func

    for chunk in model:
        chunk.train()
    try:
        with contextlib.ExitStack() as stack:
            for chunk in model:
                if hasattr(chunk, "no_sync"):
                    stack.enter_context(chunk.no_sync())
            get_forward_backward_func()(
                forward_step_func=forward_step_func,
                data_iterator=iter(microbatches),
                model=model,
                num_microbatches=len(microbatches),
                seq_length=microbatches[0].input_ids.shape[1],
                micro_batch_size=microbatches[0].input_ids.shape[0],
                forward_only=False,
                pg_collection=pg_collection,
            )
    finally:
        config.finalize_model_grads_func, config.grad_sync_func, config.no_sync_func = saved
    return loss_sum, num_loss_tokens
