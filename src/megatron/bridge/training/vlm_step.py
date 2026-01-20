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
import math
import logging
from functools import partial
from typing import Any, Iterable

import torch
from megatron.core.models.gpt import GPTModel
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from megatron.core.utils import get_batch_on_this_cp_rank, get_model_config

from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.losses import (
    create_masked_next_token_loss_function as _create_loss_function,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.utils.padding_utils import (
    pad_or_truncate_2d_to_len,
    pad_or_truncate_attn_to_len,
    pad_or_truncate_pos_to_len,
)
from megatron.bridge.training.utils.pg_utils import get_pg_collection


logger = logging.getLogger(__name__)


def get_batch_from_iterator(
    data_iterator: Iterable,
    use_mtp: bool = False,
    skip_getting_attention_mask_from_dataset: bool = True,
    *,
    is_first_pp_stage: bool,
    is_last_pp_stage: bool,
) -> dict[str, Any]:
    """Get a batch of data from the iterator.

    Args:
        data_iterator: The data iterator to get the batch from.
        use_mtp: Whether Multi-Token Prediction layers are enabled.
        skip_getting_attention_mask_from_dataset: If set, the dataset will pass a None attention mask.

    Returns:
        dict[str, torch.Tensor]: A dictionary containing the batch data.
    """
    batch = next(data_iterator)

    required_device_keys = set()
    required_host_keys = set()

    if not skip_getting_attention_mask_from_dataset:
        required_device_keys.add("attention_mask")

    # Instead of raw tensors, expect a single 'visual_inputs' object in batch
    required_device_keys.add("visual_inputs")

    if "cu_seqlens" in batch:
        required_device_keys.add("cu_seqlens")
        required_host_keys.add("cu_seqlens_argmin")
        required_host_keys.add("max_seqlen")

    required_device_keys.update(("tokens", "input_ids", "position_ids"))
    if is_last_pp_stage:
        required_device_keys.update(("labels", "loss_mask"))

    _batch_required_keys = {}
    for key, val in batch.items():
        if key in required_device_keys:
            if key == "visual_inputs":
                if val is None:
                    _batch_required_keys[key] = None
                else:
                    _batch_required_keys[key] = val
                    # Move all visual inputs contained tensors to CUDA
                    for k, v in val.__dict__.items():
                        _batch_required_keys[key].__dict__[k] = v.cuda(non_blocking=True) if v is not None else None
            else:
                _batch_required_keys[key] = val.cuda(non_blocking=True) if val is not None else None
        elif key in required_host_keys:
            _batch_required_keys[key] = val.cpu() if val is not None else None
        else:
            _batch_required_keys[key] = None

    return _batch_required_keys


def get_batch(
    data_iterator: Iterable,
    cfg: ConfigContainer,
    use_mtp: bool = False,
    *,
    is_first_pp_stage: bool,
    is_last_pp_stage: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Any,
]:
    """Generate a batch.

    Args:
        data_iterator: Input data iterator
        cfg: Configuration container
        use_mtp: Whether Multi-Token Prediction layers are enabled
        is_first_pp_stage: Whether the current stage is the first stage
        is_last_pp_stage: Whether the current stage is the last stage
    Returns:
        TODO: add description
    """
    # All PP stages load from iterator to get input_ids and visual grid info
    # This allows each stage to compute MRoPE position_ids locally without broadcasting
    batch = get_batch_from_iterator(
        data_iterator,
        use_mtp,
        getattr(cfg.dataset, "skip_getting_attention_mask_from_dataset", True),
        is_first_pp_stage=is_first_pp_stage,
        is_last_pp_stage=is_last_pp_stage,
    )

    if "visual_inputs" in batch:
        # convert visual_inputs to multi_modal_inputs which is a dict contains "pixel_values" and "image_grid_thw"
        # TODO(jinliangl): add video support
        multi_modal_inputs = batch.get("visual_inputs").normalized_for_model()
    else:
        multi_modal_inputs = {}

    # return naive batch and don't do any padding or cp slicing
    return (
        batch.get("tokens") if batch.get("tokens") is not None else batch.get("input_ids"),
        batch.get("labels"),
        batch.get("loss_mask"),
        batch.get("attention_mask"),
        batch.get("position_ids"),
        multi_modal_inputs,
    )


def forward_step(
    state: GlobalState,
    data_iterator: Iterable,
    model: GPTModel,
    return_schedule_plan: bool = False,
    vision_model: bool = False,
    data_format: str = "bshd",
) -> tuple[torch.Tensor, partial]:
    """Forward training step.

    Args:
        state: Global state for the run
        data_iterator: Input data iterator
        model: The GPT Model
        return_schedule_plan (bool): Whether to return the schedule plan instead of the output tensor

    Returns:
        tuple containing the output tensor and the loss function
    """
    timers = state.timers
    straggler_timer = state.straggler_timer
    assert data_format in ["thd", "bshd"], "data_format must be 'thd' or 'bshd'"

    this_pg_collection = get_pg_collection(model)
    is_first = (
        is_pp_first_stage(this_pg_collection.pp) if not vision_model else False
    )  # vision model does not need pre_process, because we pack the input_ids to thd in the forward function
    is_last = is_pp_last_stage(this_pg_collection.pp)

    config = get_model_config(model)
    use_mtp = (getattr(config, "mtp_num_layers", None) or 0) > 0

    timers("batch-generator", log_level=2).start()
    with straggler_timer(bdata=True):
        (
            tokens,
            labels,
            loss_mask,
            attention_mask,
            position_ids,
            multi_modal_inputs,
        ) = get_batch(data_iterator, state.cfg, use_mtp, is_first_pp_stage=is_first, is_last_pp_stage=is_last)
    timers("batch-generator").stop()

    # padded and get packed seq params
    seq_len = state.cfg.model.seq_length
    # assert input token must be bshd format, tokens's shape must be [batch_size, seq_len].

    assert tokens.shape == labels.shape == loss_mask.shape
    assert tokens.shape[0] == state.cfg.train.micro_batch_size, (
        f"We only support bshd format, and tokens's shape must be [batch_size, seq_len], but got {tokens.shape} and batch_size is {state.cfg.train.micro_batch_size}"
    )

    # pad tokens, labels, loss_mask, attention_mask, position_ids to target_len.
    cur_len = tokens.size(1)
    tp_size = this_pg_collection.tp.size()
    cp_size = this_pg_collection.cp.size()
    divisible_by = tp_size * cp_size * 2 if cp_size > 1 else tp_size

    use_fp8_padding = True
    divisible_by = math.lcm(divisible_by, 16) if use_fp8_padding else divisible_by
    target_len = math.ceil(cur_len / divisible_by) * divisible_by

    tokens = pad_or_truncate_2d_to_len(tokens, target_len=target_len, max_cap=target_len, pad_value=0)
    labels = pad_or_truncate_2d_to_len(labels, target_len=target_len, max_cap=target_len, pad_value=-100)
    loss_mask = pad_or_truncate_2d_to_len(loss_mask, target_len=target_len, max_cap=target_len, pad_value=0)
    attention_mask = pad_or_truncate_attn_to_len(attention_mask, target_len=target_len, max_cap=target_len)
    position_ids = pad_or_truncate_pos_to_len(position_ids, target_len=target_len, max_cap=target_len)

    forward_args = {
        "input_ids": tokens,
        "labels": labels,
        "loss_mask": loss_mask,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }
    # tokens, labels, loss_mask, attention_mask, position_ids variable is not split by cp.
    # but these variables in forward_args dict are split by cp.
    forward_args = get_batch_on_this_cp_rank(forward_args)

    # We need original input_ids, not split by cp, because it will be split by cp in the model forward function.
    forward_args["input_ids"] = tokens

    # We need loss_mask_split_by_cp to calculate loss.
    loss_mask = forward_args["loss_mask"]  # which is split by cp

    # follow the design of verl, we put the multi-modal inputs in the forward args
    if "pixel_values" in multi_modal_inputs:
        forward_args["pixel_values"] = multi_modal_inputs["pixel_values"]
    if "image_grid_thw" in multi_modal_inputs:
        forward_args["image_grid_thw"] = multi_modal_inputs["image_grid_thw"]
    if "pixel_values_videos" in multi_modal_inputs:
        forward_args["pixel_values_videos"] = multi_modal_inputs["pixel_values_videos"]
    if "video_grid_thw" in multi_modal_inputs:
        forward_args["video_grid_thw"] = multi_modal_inputs["video_grid_thw"]

    if data_format == "thd":
        batch_size = tokens.shape[0]
        # construct packed_seq_params for thd data_format.
        seqlens_in_batch = torch.ones(batch_size, dtype=torch.int32, device=tokens.device) * cur_len
        seqlens_in_batch_padded = torch.ones(batch_size, dtype=torch.int32, device=tokens.device) * target_len

        cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=tokens.device)
        cu_seqlens[1:] = torch.cumsum(seqlens_in_batch, dim=0)
        cu_seqlens_padded = torch.zeros(batch_size + 1, dtype=torch.int32, device=tokens.device)
        cu_seqlens_padded[1:] = torch.cumsum(seqlens_in_batch_padded, dim=0)
        max_seqlen_in_batch = seqlens_in_batch.max().item()
        max_seqlen_in_batch_padded = seqlens_in_batch_padded.max().item()

        packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens,
            max_seqlen_q=max_seqlen_in_batch_padded,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_kv=max_seqlen_in_batch_padded,
            cu_seqlens_q_padded=cu_seqlens_padded,
            cu_seqlens_kv_padded=cu_seqlens_padded,
        )
        # after preprocess, input_ids_rmpad is sliced by cp.
        # when processing vision model, the pre_process is False, and input_ids is not sliced by cp.
        forward_args["packed_seq_params"] = packed_seq_params

        # We need labels with THD data_format. With [1, total_seq_len] shape.
        forward_args["labels"] = forward_args["labels"].reshape(1, -1)

        if vision_model and attention_mask is None:
            # We will split input_ids by cp on the fly using attention_mask.
            attention_mask = torch.ones(tokens.shape[0], tokens.shape[1], dtype=torch.bool)
            forward_args["attention_mask"] = attention_mask
    elif data_format == "bshd":
        pass
    else:
        raise ValueError(f"Invalid data format: {data_format}")

    check_for_nan_in_loss = state.cfg.rerun_state_machine.check_for_nan_in_loss
    check_for_spiky_loss = state.cfg.rerun_state_machine.check_for_spiky_loss
    with straggler_timer:
        if return_schedule_plan:
            assert config.overlap_moe_expert_parallel_comm, (
                "overlap_moe_expert_parallel_comm must be enabled to return the schedule plan"
            )
            schedule_plan = model.build_schedule_plan(
                tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask
            )
            loss_function = _create_loss_function(loss_mask, check_for_nan_in_loss, check_for_spiky_loss)
            return schedule_plan, loss_function
        else:
            output_tensor = model(**forward_args)

    loss_function = _create_loss_function(loss_mask, check_for_nan_in_loss, check_for_spiky_loss)

    return output_tensor, loss_function
