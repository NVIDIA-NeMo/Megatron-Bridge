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

"""
Audio-language model training step, modeled after vlm_step.py.
"""

import logging
from functools import partial
from typing import Iterable

import torch
from megatron.core.models.gpt import GPTModel
from megatron.core.utils import get_model_config

from megatron.bridge.training.losses import (
    create_masked_next_token_loss_function as _create_loss_function,
)
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.utils.packed_seq_utils import get_packed_seq_params
from megatron.bridge.training.utils.pg_utils import get_pg_collection
from megatron.bridge.training.vlm_step import get_batch


logger = logging.getLogger(__name__)


def forward_step(
    state: GlobalState, data_iterator: Iterable, model: GPTModel, return_schedule_plan: bool = False
) -> tuple[torch.Tensor, partial]:
    """Forward training step for audio-language models.

    Reuses the VLM get_batch pipeline (which handles visual_inputs generically)
    and passes audio-specific kwargs to the model.

    Args:
        state: Global state for the run
        data_iterator: Input data iterator
        model: The audio-language model
        return_schedule_plan (bool): Whether to return the schedule plan instead of the output tensor

    Returns:
        tuple containing the output tensor and the loss function
    """
    timers = state.timers
    straggler_timer = state.straggler_timer

    config = get_model_config(model)
    use_mtp = (getattr(config, "mtp_num_layers", None) or 0) > 0

    timers("batch-generator", log_level=2).start()
    pg_collection = get_pg_collection(model)
    with straggler_timer(bdata=True):
        (
            tokens,
            labels,
            loss_mask,
            attention_mask,
            position_ids,
            cu_seqlens,
            max_seqlen,
            visual_inputs,
        ) = get_batch(data_iterator, state.cfg, use_mtp, pg_collection=pg_collection)
    timers("batch-generator").stop()

    forward_args = {
        "input_ids": tokens,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "loss_mask": loss_mask,
    }

    if visual_inputs is not None:
        forward_args.update(visual_inputs.normalized_for_model())

    # Add packed sequence support
    if cu_seqlens is not None:
        cu_seqlens_argmin = torch.tensor(len(cu_seqlens))  # no padding in cu_seqlens since packing is done in-batch
        packed_seq_params = {
            "cu_seqlens": cu_seqlens,
            "max_seqlen": max_seqlen,
            "cu_seqlens_argmin": cu_seqlens_argmin,
        }
        forward_args["packed_seq_params"] = get_packed_seq_params(packed_seq_params)

    check_for_nan_in_loss = state.cfg.rerun_state_machine.check_for_nan_in_loss
    check_for_spiky_loss = state.cfg.rerun_state_machine.check_for_spiky_loss
    with straggler_timer:
        model_output = model(**forward_args)
        if isinstance(model_output, tuple):
            output_tensor, loss_mask = model_output
        else:
            output_tensor = model_output

    loss_function = _create_loss_function(loss_mask, check_for_nan_in_loss, check_for_spiky_loss)

    return output_tensor, loss_function
