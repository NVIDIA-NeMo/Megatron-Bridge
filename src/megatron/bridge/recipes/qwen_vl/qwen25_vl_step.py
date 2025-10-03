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

import logging
from functools import partial
from typing import Iterable

import torch
import torch.nn.functional as F
from megatron.core import parallel_state
from megatron.core.models.gpt import GPTModel
from megatron.core.utils import get_batch_on_this_cp_rank, get_model_config

from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.utils.loss_utils import create_loss_function as _create_loss_function
from megatron.bridge.training.utils.packed_seq_utils import get_packed_seq_params


logger = logging.getLogger(__name__)


def get_batch_from_iterator(
    data_iterator: Iterable,
    use_mtp: bool = False,
    skip_getting_attention_mask_from_dataset: bool = True,
) -> dict[str, torch.Tensor]:
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
    # Optionally include vision inputs if present in the batch
    if "pixel_values" in batch:
        required_device_keys.add("pixel_values")
    if "image_grid_thw" in batch:
        required_device_keys.add("image_grid_thw")

    if "cu_seqlens" in batch:
        required_device_keys.add("cu_seqlens")
        required_host_keys.add("cu_seqlens_argmin")
        required_host_keys.add("max_seqlen")

    required_device_keys.update(("tokens", "input_ids", "position_ids"))
    if parallel_state.is_pipeline_last_stage():
        required_device_keys.update(("labels", "loss_mask"))

    _batch_required_keys = {}
    for key, val in batch.items():
        if key in required_device_keys:
            _batch_required_keys[key] = val.cuda(non_blocking=True) if val is not None else None
        elif key in required_host_keys:
            _batch_required_keys[key] = val.cpu() if val is not None else None
        else:
            _batch_required_keys[key] = None

    return _batch_required_keys


def get_batch(
    data_iterator: Iterable, cfg: ConfigContainer, use_mtp: bool = False
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Generate a batch.

    Args:
        data_iterator: Input data iterator
        cfg: Configuration container
        use_mtp: Whether Multi-Token Prediction layers are enabled

    Returns:
        tuple of tensors containing tokens, labels, loss_mask, attention_mask, position_ids,
        cu_seqlens, cu_seqlens_argmin, max_seqlen, pixel_values (optional), image_grid_thw (optional)
    """
    if (not parallel_state.is_pipeline_first_stage()) and (not parallel_state.is_pipeline_last_stage()):
        return None, None, None, None, None, None, None, None

    batch = get_batch_from_iterator(
        data_iterator,
        use_mtp,
        getattr(cfg.dataset, "skip_getting_attention_mask_from_dataset", True),
    )

    # Keep optional vision tensors aside to avoid being dropped by CP slicing util
    pixel_values = batch.get("pixel_values")
    image_grid_thw = batch.get("image_grid_thw")

    # slice batch along sequence dimension for context parallelism
    batch = get_batch_on_this_cp_rank(batch)
    if pixel_values is not None:
        batch["pixel_values"] = pixel_values
    if image_grid_thw is not None:
        batch["image_grid_thw"] = image_grid_thw

    # Helpers to pad/truncate to a desired target length with an upper cap
    def _pad_or_truncate_2d_to_len(
        x: torch.Tensor | None, target_len: int, max_cap: int, pad_value: int | float
    ) -> torch.Tensor | None:
        if x is None:
            return None
        cur = x.size(1)
        if cur < target_len:
            return F.pad(x, (0, target_len - cur), value=pad_value)
        if cur > max_cap:
            return x[:, :max_cap]
        return x

    def _pad_or_truncate_pos_to_len(
        pos: torch.Tensor | None, target_len: int, max_cap: int
    ) -> torch.Tensor | None:
        if pos is None:
            return None
        cur = pos.size(1)
        if cur < target_len:
            add = (
                torch.arange(cur, target_len, device=pos.device, dtype=pos.dtype)
                .unsqueeze(0)
                .expand(pos.size(0), -1)
            )
            return torch.cat([pos, add], dim=1)
        if cur > max_cap:
            return pos[:, :max_cap]
        return pos

    def _pad_or_truncate_attn_to_len(
        mask: torch.Tensor | None, target_len: int, max_cap: int
    ) -> torch.Tensor | None:
        if mask is None:
            return None
        # Expected shape (B, 1, S, S)
        b, h, s1, s2 = mask.shape
        pad_val = False if mask.dtype == torch.bool else 0
        if s1 < target_len:
            return F.pad(mask, (0, target_len - s2, 0, target_len - s1), value=pad_val)
        if s1 > max_cap:
            return mask[:, :, :max_cap, :max_cap]
        return mask

    # When using pipeline parallelism, ensure fixed shapes equal to cfg.model.seq_length
    if getattr(cfg.model, "pipeline_model_parallel_size", 1) > 1:
        seq_len = cfg.model.seq_length

        tokens_or_input = batch.get("tokens") if batch.get("tokens") is not None else batch.get("input_ids")
        tokens_or_input = _pad_or_truncate_2d_to_len(tokens_or_input, seq_len, seq_len, pad_value=0)
        if batch.get("tokens") is not None:
            batch["tokens"] = tokens_or_input  # type: ignore[assignment]
        else:
            batch["input_ids"] = tokens_or_input  # type: ignore[assignment]
        batch["labels"] = _pad_or_truncate_2d_to_len(batch.get("labels"), seq_len, seq_len, pad_value=-100)  # type: ignore[assignment]
        batch["loss_mask"] = _pad_or_truncate_2d_to_len(batch.get("loss_mask"), seq_len, seq_len, pad_value=0)  # type: ignore[assignment]
        batch["position_ids"] = _pad_or_truncate_pos_to_len(batch.get("position_ids"), seq_len, seq_len)  # type: ignore[assignment]
        if batch.get("attention_mask") is not None:
            batch["attention_mask"] = _pad_or_truncate_attn_to_len(batch.get("attention_mask"), seq_len, seq_len)  # type: ignore[assignment]
    else:
        # No PP: pad sequence length to nearest multiple of 64 for efficiency (capped at model seq_length)
        seq_cap = cfg.model.seq_length

        def _ceil_to_mult(n: int, mult: int) -> int:
            return ((n + mult - 1) // mult) * mult

        tokens_or_input = batch.get("tokens") if batch.get("tokens") is not None else batch.get("input_ids")
        if tokens_or_input is not None:
            cur_len = tokens_or_input.size(1)
            target_len = min(seq_cap, _ceil_to_mult(cur_len, 64))

            # tokens/input_ids
            padded_tokens = _pad_or_truncate_2d_to_len(tokens_or_input, target_len, seq_cap, pad_value=0)
            if batch.get("tokens") is not None:
                batch["tokens"] = padded_tokens  # type: ignore[assignment]
            else:
                batch["input_ids"] = padded_tokens  # type: ignore[assignment]

            # labels and loss mask
            batch["labels"] = _pad_or_truncate_2d_to_len(batch.get("labels"), target_len, seq_cap, pad_value=-100)  # type: ignore[assignment]
            batch["loss_mask"] = _pad_or_truncate_2d_to_len(batch.get("loss_mask"), target_len, seq_cap, pad_value=0)  # type: ignore[assignment]

            # position_ids: extend with increasing positions
            pos = batch.get("position_ids")
            pos = _pad_or_truncate_pos_to_len(pos, target_len, seq_cap)
            if pos is not None:
                batch["position_ids"] = pos  # type: ignore[assignment]

            # attention_mask if present
            attn = batch.get("attention_mask")
            if attn is not None:
                attn = _pad_or_truncate_attn_to_len(attn, target_len, seq_cap)
                batch["attention_mask"] = attn  # type: ignore[assignment]

    return (
        (batch.get("tokens") if batch.get("tokens") is not None else batch.get("input_ids")),
        batch["labels"],
        batch["loss_mask"],
        batch["attention_mask"],
        batch["position_ids"],
        batch.get("cu_seqlens"),
        batch.get("cu_seqlens_argmin"),
        batch.get("max_seqlen"),
        batch.get("pixel_values"),
        batch.get("image_grid_thw"),
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
            pixel_values,
            image_grid_thw,
        ) = get_batch(data_iterator, state.cfg, use_mtp)
    timers("batch-generator").stop()

    forward_args = {
        "input_ids": tokens,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
    # Add optional vision inputs if available
    if pixel_values is not None:
        # Flatten possible [batch, num_images, C, H, W] -> [num_images_total, C, H, W]
        if pixel_values.dim() == 5:
            b, n, c, h, w = pixel_values.shape
            pixel_values = pixel_values.view(b * n, c, h, w)
        forward_args["pixel_values"] = pixel_values
    if image_grid_thw is not None:
        # Flatten possible [batch, num_images, 3] -> [num_images_total, 3]
        if image_grid_thw.dim() == 3:
            image_grid_thw = image_grid_thw.view(-1, image_grid_thw.size(-1))
        forward_args["image_grid_thw"] = image_grid_thw

    # Add packed sequence support
    if cu_seqlens is not None:
        packed_seq_params = {
            "cu_seqlens": cu_seqlens,
            "cu_seqlens_argmin": cu_seqlens_argmin,
            "max_seqlen": max_seqlen,
        }
        forward_args["packed_seq_params"] = get_packed_seq_params(packed_seq_params)

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
