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

from collections.abc import Callable, Iterator
from typing import Any

import torch
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import MegatronModule
from megatron.core.utils import get_model_config

from megatron.bridge.engine.datum import ModelOutput, PreparedMicrobatch
from megatron.bridge.training.gpt_step import _prepare_packed_padding_mask


def call_model(
    model: MegatronModule, microbatch: PreparedMicrobatch, pg_collection: ProcessGroupCollection
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
        loss_mask=microbatch.loss_mask,
        packed_seq_params=microbatch.packed_seq_params,
        padding_mask=padding_mask,
    )


def to_model_output(nll: torch.Tensor) -> ModelOutput:
    """Adapt the raw model return (per-token NLL) into the ``ModelOutput`` contract (no logits on this path)."""
    return ModelOutput(logprobs=-nll)


def _make_scoring_step(pg_collection: ProcessGroupCollection) -> Callable:
    def forward_step_func(
        data_iterator: Iterator[PreparedMicrobatch], model: MegatronModule
    ) -> tuple[torch.Tensor, Callable]:
        microbatch = next(data_iterator)
        nll = call_model(model, microbatch, pg_collection)

        def collect(output_tensor: torch.Tensor, non_loss_data: bool = False) -> torch.Tensor:
            return to_model_output(output_tensor).logprobs[0].detach().float()

        return nll, collect

    return forward_step_func


def run_forward_only(
    model: list[MegatronModule], microbatches: list[PreparedMicrobatch], *, pg_collection: ProcessGroupCollection
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
                forward_step_func=_make_scoring_step(pg_collection),
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


def run_forward_backward(engine: Any, microbatches: list[PreparedMicrobatch], loss_method: Any) -> Any:
    """Drive the schedule with gradient finalization suppressed (accumulation across calls)."""
    raise NotImplementedError("run_forward_backward is implemented in phase B2.")
