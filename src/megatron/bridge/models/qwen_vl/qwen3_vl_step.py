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
import math
from functools import partial
from typing import Any, Iterable

import torch
from megatron.core.models.gpt import GPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from megatron.core.utils import get_batch_on_this_cp_rank, get_model_config

from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.losses import (
    create_masked_next_token_loss_function as _create_loss_function,
)
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

    if "visual_inputs" in batch and batch.get("visual_inputs") is not None:
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


def pack_or_pad_batch_sequences(
    tokens: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    this_pg_collection,
    use_fp8_padding: bool = False,
    force_to_pad_to_seq_len: bool = False,
    seq_length: int = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, PackedSeqParams]:
    """
    Right-pad BSHD sequences to a common length and build a bookkeeping PackedSeqParams.

    The returned ``packed_seq_params`` carries uniform `target_len` per segment and is only used
    as a fallback. Real-length-aware packing (the path that delivers pad-skip FLOP savings) is
    handled separately by ``_pack_bshd_to_thd`` in ``forward_step``.
    """

    batch_size, cur_len = tokens.shape
    device = tokens.device

    tp_size = this_pg_collection.tp.size()
    cp_size = this_pg_collection.cp.size()
    divisible_by = tp_size * cp_size * 2 if cp_size > 1 else tp_size
    divisible_by = math.lcm(divisible_by, 16) if use_fp8_padding else divisible_by

    # build bshd sequences with tiny padding to be compatible with qwen3vl model
    target_len = math.ceil(cur_len / divisible_by) * divisible_by
    if force_to_pad_to_seq_len:
        target_len = seq_length
    tokens = pad_or_truncate_2d_to_len(tokens, target_len=target_len, max_cap=target_len, pad_value=0)
    labels = pad_or_truncate_2d_to_len(labels, target_len=target_len, max_cap=target_len, pad_value=-100)
    loss_mask = pad_or_truncate_2d_to_len(loss_mask, target_len=target_len, max_cap=target_len, pad_value=0)
    attention_mask = pad_or_truncate_attn_to_len(attention_mask, target_len=target_len, max_cap=target_len)
    position_ids = pad_or_truncate_pos_to_len(position_ids, target_len=target_len, max_cap=target_len)

    seqlens_in_batch = torch.ones(batch_size, dtype=torch.int32, device=device) * target_len
    seqlens_in_batch_padded = torch.ones(batch_size, dtype=torch.int32, device=device) * target_len
    cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    cu_seqlens[1:] = torch.cumsum(seqlens_in_batch, dim=0)
    cu_seqlens_padded = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    cu_seqlens_padded[1:] = torch.cumsum(seqlens_in_batch_padded, dim=0)
    max_seqlen_in_batch = seqlens_in_batch.max().item()

    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        max_seqlen_q=max_seqlen_in_batch,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_kv=max_seqlen_in_batch,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
    )

    return tokens, labels, loss_mask, attention_mask, position_ids, packed_seq_params


def _per_sample_real_lengths(attention_mask: torch.Tensor) -> torch.Tensor:
    """Reduce a BSHD attention mask to (B,) int32 real-length per sample.

    Accepts a 2D `(B, T)` keep mask (1=real, 0=pad) or a 4D Megatron-style bool mask
    `(B|1, 1, T, T)` where True=masked.
    """
    if attention_mask.dim() == 2:
        return attention_mask.to(torch.int32).sum(dim=-1)
    if attention_mask.dim() == 4:
        # Megatron causal masks are (B|1, 1, T, T) bool with True=masked. The diagonal row tells
        # us which key positions are "real": True on the diagonal means a fully masked-out row.
        keep_row = ~attention_mask[:, 0, :, 0].bool()
        if keep_row.size(0) == 1:
            keep_row = keep_row.expand(attention_mask.size(0) if attention_mask.size(0) > 1 else 1, -1)
        return keep_row.to(torch.int32).sum(dim=-1)
    raise ValueError(f"Unsupported attention_mask rank: {attention_mask.dim()}")


def _pack_bshd_to_thd(
    *,
    tokens: torch.Tensor,
    labels: torch.Tensor | None,
    loss_mask: torch.Tensor | None,
    position_ids: torch.Tensor | None,
    real_lengths: torch.Tensor,
    align_size: int,
    pad_token_id: int = 0,
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
]:
    """Pack a BSHD micro-batch into a single THD row by concatenating per-sample real content.

    Per segment i in the packed row: `[real_content_i, align_pad_i]` where `align_pad_i` brings
    the segment length up to a multiple of `align_size` (filled with `pad_token_id` / `-100` /
    `0` for tokens / labels / loss_mask respectively). Real content is taken from the leftmost
    `real_lengths[i]` positions of row i.

    Returns:
        packed_tokens, packed_labels, packed_loss_mask, packed_position_ids,
        cu_seqlens_unpadded, cu_seqlens_padded, moe_padding_mask, max_real_seqlen
    """
    batch_size, _ = tokens.shape
    device = tokens.device

    real_lens_cpu = real_lengths.tolist()
    padded_lens_cpu = [((rl + align_size - 1) // align_size) * align_size for rl in real_lens_cpu]
    total_padded = int(sum(padded_lens_cpu))
    max_real_seqlen = int(max(real_lens_cpu)) if real_lens_cpu else 0

    cu_unpadded_cpu = [0]
    cu_padded_cpu = [0]
    for rl, pl in zip(real_lens_cpu, padded_lens_cpu):
        cu_unpadded_cpu.append(cu_unpadded_cpu[-1] + int(rl))
        cu_padded_cpu.append(cu_padded_cpu[-1] + int(pl))

    cu_seqlens_unpadded = torch.tensor(cu_unpadded_cpu, dtype=torch.int32, device=device)
    cu_seqlens_padded = torch.tensor(cu_padded_cpu, dtype=torch.int32, device=device)

    packed_tokens = torch.full((1, total_padded), pad_token_id, dtype=tokens.dtype, device=device)
    packed_labels = (
        torch.full((1, total_padded), -100, dtype=labels.dtype, device=device) if labels is not None else None
    )
    packed_loss_mask = (
        torch.zeros((1, total_padded), dtype=loss_mask.dtype, device=device) if loss_mask is not None else None
    )
    packed_position_ids = (
        torch.zeros((1, total_padded), dtype=position_ids.dtype, device=device) if position_ids is not None else None
    )
    moe_padding_mask = torch.zeros((1, total_padded), dtype=torch.bool, device=device)

    for i in range(batch_size):
        real_len = int(real_lens_cpu[i])
        padded_len = int(padded_lens_cpu[i])
        start = int(cu_padded_cpu[i])
        if real_len > 0:
            packed_tokens[0, start : start + real_len] = tokens[i, :real_len]
            if packed_labels is not None:
                packed_labels[0, start : start + real_len] = labels[i, :real_len]
            if packed_loss_mask is not None:
                packed_loss_mask[0, start : start + real_len] = loss_mask[i, :real_len]
            if packed_position_ids is not None:
                packed_position_ids[0, start : start + real_len] = position_ids[i, :real_len]
        if padded_len > real_len:
            # Align-pad slots are excluded from MoE accounting.
            moe_padding_mask[0, start + real_len : start + padded_len] = True

    return (
        packed_tokens,
        packed_labels,
        packed_loss_mask,
        packed_position_ids,
        cu_seqlens_unpadded,
        cu_seqlens_padded,
        moe_padding_mask,
        max_real_seqlen,
    )


def forward_step(
    state: GlobalState,
    data_iterator: Iterable,
    model: GPTModel,
    return_schedule_plan: bool = False,
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

    this_pg_collection = get_pg_collection(model)
    is_first = is_pp_first_stage(this_pg_collection.pp)
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

    # To be compatible with qwen3vl, we move the sequence padding and packing to forward_step function.
    # Qwen3VL model need the original input and do cp and sp split in model.forward.
    pack_sequences_in_batch = getattr(state.cfg.dataset, "pack_sequences_in_batch", False)

    # Capture per-sample real lengths from a dataset-supplied 2D `(B, T)` attention_mask before
    # `pack_or_pad_batch_sequences` pads rows. `pad_or_truncate_attn_to_len` only accepts 4D
    # Megatron-style masks, so strip the 2D mask here and let downstream packing rebuild what
    # it needs from the captured lengths.
    dataset_real_lengths: torch.Tensor | None = None
    if attention_mask is not None and attention_mask.dim() == 2:
        dataset_real_lengths = attention_mask.to(torch.int32).sum(dim=-1)
        attention_mask = None

    tokens, labels, loss_mask, attention_mask, position_ids, _bookkeeping_packed_seq_params = (
        pack_or_pad_batch_sequences(
            tokens,
            labels,
            loss_mask,
            attention_mask,
            position_ids,
            this_pg_collection,
            use_fp8_padding=True,
            force_to_pad_to_seq_len=this_pg_collection.pp.size() > 1 or this_pg_collection.ep.size() > 1,
            seq_length=config.seq_length,
        )
    )
    forward_args = {
        "input_ids": tokens,
        "labels": labels,
        "loss_mask": loss_mask,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }

    original_tokens = tokens.clone()
    forward_args = get_batch_on_this_cp_rank(forward_args, cp_group=this_pg_collection.cp)
    forward_args["packed_seq_params"] = None
    forward_args["input_ids"] = original_tokens
    # calculate position_ids in model forward
    forward_args["position_ids"] = None
    if pack_sequences_in_batch:
        # Derive real per-sample lengths. Preference order:
        # 1. lengths captured from the 2D attention_mask the dataset supplied (most reliable);
        # 2. an explicit 4D mask still attached to forward_args (rare for qwen3-vl);
        # 3. "position of last non-zero token" in the BSHD padded tokens — only the trailing-pad
        #    that pad_or_truncate_2d_to_len introduces is reliably zero, so this misses a
        #    collator-pad token (e.g. tokenizer.pad_token_id) between real content and the
        #    trailing zeros.  Option 1 is much preferred; enable it via
        #    `dataset.skip_getting_attention_mask_from_dataset=False`.
        if dataset_real_lengths is not None:
            real_lengths = dataset_real_lengths.to(torch.int32)
        elif forward_args["attention_mask"] is not None:
            real_lengths = _per_sample_real_lengths(forward_args["attention_mask"])
        else:
            indices = torch.arange(original_tokens.shape[1], device=original_tokens.device)
            nonzero = original_tokens != 0
            masked_indices = torch.where(
                nonzero, indices.unsqueeze(0).expand_as(nonzero), torch.full_like(nonzero, -1, dtype=indices.dtype)
            )
            real_lengths = (masked_indices.max(dim=-1).values + 1).clamp(min=0).to(torch.int32)

        tp_size = this_pg_collection.tp.size()
        cp_size = this_pg_collection.cp.size()
        align_size = tp_size * cp_size * 2 if cp_size > 1 else tp_size
        align_size = math.lcm(align_size, 16)  # match fp8/transformer-engine alignment used above

        (
            packed_tokens,
            packed_labels,
            packed_loss_mask,
            _packed_position_ids,  # model recomputes MRoPE; dataset position_ids are dropped
            cu_seqlens_unpadded,
            cu_seqlens_padded,
            moe_padding_mask,
            max_real_seqlen,
        ) = _pack_bshd_to_thd(
            tokens=original_tokens,
            labels=forward_args["labels"],
            loss_mask=forward_args["loss_mask"],
            position_ids=None,
            real_lengths=real_lengths,
            align_size=align_size,
            pad_token_id=0,
        )

        packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens_unpadded,
            max_seqlen_q=max_real_seqlen,
            cu_seqlens_kv=cu_seqlens_unpadded,
            max_seqlen_kv=max_real_seqlen,
            cu_seqlens_q_padded=cu_seqlens_padded,
            cu_seqlens_kv_padded=cu_seqlens_padded,
        )

        forward_args["input_ids"] = packed_tokens
        forward_args["labels"] = packed_labels
        forward_args["loss_mask"] = packed_loss_mask
        # Attention isolation now comes from cu_seqlens; the mask is unused on the THD path.
        forward_args["attention_mask"] = None
        forward_args["position_ids"] = None
        forward_args["packed_seq_params"] = packed_seq_params
        # Tell the model's per-subseq MRoPE path which content belongs to which sample
        # (consumed by model.forward when running on a packed (1, total) input).
        forward_args["rope_cu_seqlens"] = cu_seqlens_unpadded
        # Exclude align-pad slots from MoE router accounting.
        forward_args["moe_padding_mask"] = moe_padding_mask

    # use cp split loss mask for calculate loss
    loss_mask = forward_args["loss_mask"]
    # follow the design of verl, we put the multi-modal inputs in the forward args
    if "pixel_values" in multi_modal_inputs:
        forward_args["pixel_values"] = multi_modal_inputs["pixel_values"]
    if "image_grid_thw" in multi_modal_inputs:
        forward_args["image_grid_thw"] = multi_modal_inputs["image_grid_thw"]
    if "pixel_values_videos" in multi_modal_inputs:
        forward_args["pixel_values_videos"] = multi_modal_inputs["pixel_values_videos"]
    if "video_grid_thw" in multi_modal_inputs:
        forward_args["video_grid_thw"] = multi_modal_inputs["video_grid_thw"]

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
