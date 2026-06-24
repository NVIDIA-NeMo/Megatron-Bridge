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

from collections.abc import Mapping
from functools import partial
from inspect import Parameter, signature
from typing import Any, Iterable

import torch
from megatron.core.models.gpt import GPTModel
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from megatron.core.utils import get_model_config

from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.losses import (
    create_masked_next_token_loss_function as _create_loss_function,
)
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.utils.packed_seq_utils import get_packed_seq_params
from megatron.bridge.training.utils.pg_utils import get_pg_collection


def _unwrap_forward_module(model: Any) -> Any:
    """Return the innermost wrapped module used for forward signature checks."""
    module = model
    seen_ids = set()
    while hasattr(module, "module") and id(module) not in seen_ids:
        seen_ids.add(id(module))
        wrapped = getattr(module, "module")
        if wrapped is None or wrapped is module:
            break
        module = wrapped
    return module


def _filter_visual_kwargs_for_model(model: Any, visual_kwargs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Drop visual kwargs that the target model forward cannot consume.

    Shared VLM processors may return model-specific fields such as
    ``mm_token_type_ids``.  Keep those fields for models that accept them, but
    avoid passing them through wrappers into models with stricter signatures.
    """
    if not visual_kwargs:
        return {}

    forward_module = _unwrap_forward_module(model)
    forward = getattr(forward_module, "forward", getattr(forward_module, "__call__", None))
    if forward is None:
        return dict(visual_kwargs)

    try:
        forward_signature = signature(forward)
    except (TypeError, ValueError):
        return dict(visual_kwargs)

    params = forward_signature.parameters.values()
    if any(param.kind == Parameter.VAR_KEYWORD for param in params):
        return dict(visual_kwargs)

    supported_kwargs = {
        name
        for name, param in forward_signature.parameters.items()
        if param.kind in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY)
    }
    return {key: value for key, value in visual_kwargs.items() if key in supported_kwargs}


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
        if "cu_seqlens_unpadded" in batch:
            required_device_keys.add("cu_seqlens_unpadded")
        required_host_keys.add("cu_seqlens_argmin")
        if "cu_seqlens_unpadded_argmin" in batch:
            required_host_keys.add("cu_seqlens_unpadded_argmin")
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


def get_batch(data_iterator: Iterable, cfg: ConfigContainer, use_mtp: bool = False, *, pg_collection) -> tuple[...]:
    """Generate a batch.

    Args:
        data_iterator: Input data iterator
        cfg: Configuration container
        use_mtp: Whether Multi-Token Prediction layers are enabled

    Returns:
        tuple of tensors containing tokens, labels, loss_mask, attention_mask, position_ids,
        cu_seqlens, cu_seqlens_argmin, max_seqlen, cu_seqlens_unpadded,
        cu_seqlens_unpadded_argmin, and visual_inputs.
    """
    is_first = is_pp_first_stage(pg_collection.pp)
    is_last = is_pp_last_stage(pg_collection.pp)

    batch = get_batch_from_iterator(
        data_iterator,
        use_mtp,
        getattr(cfg.dataset, "skip_getting_attention_mask_from_dataset", True),
        is_first_pp_stage=is_first,
        is_last_pp_stage=is_last,
    )

    visual_inputs = batch.get("visual_inputs")

    return (
        (batch.get("tokens") if batch.get("tokens") is not None else batch.get("input_ids")),
        batch.get("labels"),
        batch.get("loss_mask"),  # Full packed loss_mask, will be CP-sliced by model
        batch.get("attention_mask"),
        batch.get("position_ids"),
        batch.get("cu_seqlens"),
        batch.get("cu_seqlens_argmin"),
        batch.get("max_seqlen"),
        batch.get("cu_seqlens_unpadded"),
        batch.get("cu_seqlens_unpadded_argmin"),
        visual_inputs,
    )


def forward_step(
    state: GlobalState, data_iterator: Iterable, model: GPTModel, return_schedule_plan: bool = False
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
            cu_seqlens_argmin,
            max_seqlen,
            cu_seqlens_unpadded,
            cu_seqlens_unpadded_argmin,
            visual_inputs,
        ) = get_batch(data_iterator, state.cfg, use_mtp, pg_collection=pg_collection)
    timers("batch-generator").stop()

    # Accumulate FLOPS metadata across micro-batches.
    # Each micro-batch contributes its actual padded seq_length (not cfg.model.seq_length).
    # train.py resets these before each step and reads accumulated values afterwards.
    if tokens is not None:
        mbs = tokens.shape[0]
        seq_len = tokens.shape[1]
        state._flops_seqlen_sum = getattr(state, "_flops_seqlen_sum", 0) + mbs * seq_len
        state._flops_seqlen_sq_sum = getattr(state, "_flops_seqlen_sq_sum", 0) + mbs * seq_len**2
    if visual_inputs is not None:
        for attr in ("image_grid_thw", "video_grid_thw"):
            grid = getattr(visual_inputs, attr, None)
            if grid is not None and grid.numel() > 0:
                state._flops_vision_patches = getattr(state, "_flops_vision_patches", 0) + int(
                    grid.prod(dim=-1).sum().item()
                )

    forward_args = {
        "input_ids": tokens,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "loss_mask": loss_mask,  # Pass full loss_mask so model can slice it consistently with labels
    }

    if visual_inputs is not None:
        visual_kwargs = visual_inputs.normalized_for_model()
        forward_args.update(_filter_visual_kwargs_for_model(model, visual_kwargs))

    # Add packed sequence support
    if cu_seqlens is not None:
        packed_seq_params = {
            "cu_seqlens": cu_seqlens,
            "cu_seqlens_argmin": cu_seqlens_argmin,
        }
        if max_seqlen is not None:
            packed_seq_params["max_seqlen"] = max_seqlen
        if cu_seqlens_unpadded is not None:
            packed_seq_params["cu_seqlens_unpadded"] = cu_seqlens_unpadded
        if cu_seqlens_unpadded_argmin is not None:
            packed_seq_params["cu_seqlens_unpadded_argmin"] = cu_seqlens_unpadded_argmin
        # total_tokens drives seq_idx computation in PackedSeqParams.__post_init__,
        # which is only needed for Mamba/hybrid SSM layers. Skip it for pure
        # transformer models to avoid per-step CUDA overhead.
        if getattr(config, "is_hybrid_model", False):
            packed_seq_params["total_tokens"] = tokens.size(1) if tokens is not None else labels.size(1)
        forward_args["packed_seq_params"] = get_packed_seq_params(packed_seq_params)

    if loss_mask is not None:
        loss_mask = loss_mask.contiguous()

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
            model_output = model(**forward_args)
            # Handle tuple return: (output_tensor, sliced_loss_mask) from VLM models with CP
            if isinstance(model_output, tuple):
                output_tensor, loss_mask = model_output
            else:
                output_tensor = model_output

    loss_function = _create_loss_function(loss_mask, check_for_nan_in_loss, check_for_spiky_loss)

    return output_tensor, loss_function
