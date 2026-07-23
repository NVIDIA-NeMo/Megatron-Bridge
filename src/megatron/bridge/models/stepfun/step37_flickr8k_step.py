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

"""Step3.7 Flickr8k forward step — consumes the packed batch dict produced
by :class:`Step37Flickr8kSFTDataProvider`.

``Step37Model.forward`` takes ``(input_ids, images: list[ImageForInsert],
cu_seqlens, position_ids, attention_mask, labels, loss_mask,
packed_seq_params, max_seq_len)``. This file performs no
``list[ImageForInsert] → pixel_values`` translation; the packed batch
flows straight from preprocess to model forward kwargs.

Responsibilities:
  1. ``next(data_iterator)`` → packed dict.
  2. ``preprocess_packed_batch`` → CUDA move + PIL load +
     ``list[ImageForInsert]`` with raw pixels (PP rank 0 only).
  3. Pad ``tokens / labels / loss_mask / position_id`` to TP×16 multiple
     for TE/FP8; tail padding becomes its own sub-seq via cu_seqlens.
  4. Build ``PackedSeqParams`` from ``cu_seqlens`` for FlashAttn varlen.
  5. Call ``model(**forward_args)``.
"""

from __future__ import annotations

import math
from functools import partial
from typing import Any, Iterable

import torch
from megatron.core.models.gpt import GPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.utils import is_pp_last_stage
from megatron.core.utils import get_model_config

from megatron.bridge.models.stepfun.data.flickr8k.preprocess import preprocess_packed_batch
from megatron.bridge.training.losses import (
    create_masked_next_token_loss_function as _create_loss_function,
)
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.utils.pg_utils import get_pg_collection


def _build_packed_seq_params(cu_seqlens: torch.Tensor) -> PackedSeqParams:
    """Build ``PackedSeqParams`` from a 1-D ``cu_seqlens`` (the sub-seq
    boundary array inside one packed row).
    """
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    max_seqlen = int(seqlens.max().item())
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens.to(torch.int32),
        max_seqlen_q=max_seqlen,
        cu_seqlens_kv=cu_seqlens.to(torch.int32),
        max_seqlen_kv=max_seqlen,
        cu_seqlens_q_padded=cu_seqlens.to(torch.int32),
        cu_seqlens_kv_padded=cu_seqlens.to(torch.int32),
    )


def forward_step(
    state: GlobalState,
    data_iterator: Iterable,
    model: GPTModel,
    return_schedule_plan: bool = False,
) -> tuple[torch.Tensor, partial]:
    """Forward step for the Flickr8k packed pipeline."""
    timers = state.timers
    straggler_timer = state.straggler_timer

    this_pg_collection = get_pg_collection(model)
    is_last = is_pp_last_stage(this_pg_collection.pp)

    config = get_model_config(model)
    cfg = state.cfg

    timers("batch-generator", log_level=2).start()
    with straggler_timer(bdata=True):
        batch = next(data_iterator)

        # Run the per-step preprocess. PP rank 0 loads images + builds
        # ``list[ImageForInsert]``; other PP ranks only receive
        # cu_seqlens / position_id.
        ds_cfg = cfg.dataset
        model_input = preprocess_packed_batch(
            batch,
            img_start_token_id=ds_cfg.img_start_token_id,
            patch_start_token_id=ds_cfg.patch_start_token_id,
            image_size=ds_cfg.image_size,
            patch_image_size=ds_cfg.patch_image_size,
            encoder_patch_size=ds_cfg.encoder_patch_size,
            only_pp_first_stage=True,
        )
    timers("batch-generator").stop()

    # Translate the packed dict into mbridge model kwargs.
    forward_args: dict[str, Any] = {}
    if "input_ids" in model_input:
        # PP rank 0 path: full inputs available.
        # Pad ``input_ids`` / ``labels`` / ``loss_masks`` to a TP×16
        # multiple so TE / FP8 kernels are happy. Sub-seq boundaries
        # inside the pack live in ``cu_seqlens`` and are NOT extended.
        tokens = model_input["input_ids"]
        labels = model_input["labels"]
        loss_mask = model_input["loss_masks"]
        position_id = model_input["position_id"]
        cu_seqlens = model_input["cu_seqlens"]

        tp_size = this_pg_collection.tp.size()
        divisible_by = math.lcm(tp_size, 16)
        cur_len = tokens.shape[-1]
        target_len = math.ceil(cur_len / divisible_by) * divisible_by

        if target_len > cur_len:
            pad = target_len - cur_len
            tokens = torch.nn.functional.pad(tokens, (0, pad), value=0)
            labels = torch.nn.functional.pad(labels, (0, pad), value=-100)
            loss_mask = torch.nn.functional.pad(loss_mask, (0, pad), value=0.0)
            position_id = torch.nn.functional.pad(position_id, (0, pad), value=0)
            # Tail padding becomes its own sub-seq inside cu_seqlens.
            cu_seqlens = torch.cat(
                [cu_seqlens, torch.tensor([target_len], dtype=cu_seqlens.dtype, device=cu_seqlens.device)]
            )

        forward_args["input_ids"] = tokens
        if is_last:
            forward_args["labels"] = labels.reshape(1, -1)
            forward_args["loss_mask"] = loss_mask.reshape(1, -1)
        forward_args["position_ids"] = None
        forward_args["attention_mask"] = torch.ones(tokens.shape, dtype=torch.bool, device=tokens.device)
        forward_args["packed_seq_params"] = _build_packed_seq_params(cu_seqlens)
        forward_args["cu_seqlens"] = cu_seqlens
        # Pass list[ImageForInsert] straight through — Step37Model.forward
        # consumes it natively.
        forward_args["images"] = model_input.get("images") or []
    else:
        # Non-first PP rank: only carries cu_seqlens + position_id.
        forward_args["packed_seq_params"] = _build_packed_seq_params(model_input["cu_seqlens"])
        forward_args["cu_seqlens"] = model_input["cu_seqlens"]

    check_for_nan_in_loss = cfg.rerun_state_machine.check_for_nan_in_loss
    check_for_spiky_loss = cfg.rerun_state_machine.check_for_spiky_loss
    with straggler_timer:
        if return_schedule_plan:
            assert config.overlap_moe_expert_parallel_comm, (
                "overlap_moe_expert_parallel_comm must be enabled to return the schedule plan"
            )
            schedule_plan = model.build_schedule_plan(
                forward_args.get("input_ids"),
                None,
                forward_args.get("attention_mask"),
                labels=forward_args.get("labels"),
                loss_mask=forward_args.get("loss_mask"),
            )
            loss_function = _create_loss_function(
                forward_args.get("loss_mask"), check_for_nan_in_loss, check_for_spiky_loss
            )
            return schedule_plan, loss_function
        output_tensor = model(**forward_args)

    loss_function = _create_loss_function(forward_args.get("loss_mask"), check_for_nan_in_loss, check_for_spiky_loss)
    return output_tensor, loss_function


__all__ = ["forward_step"]
