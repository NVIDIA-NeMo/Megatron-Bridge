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
import os
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
from megatron.bridge.training.utils.flop_utils import accumulate_flops_metadata, accumulate_token_throughput_metadata
from megatron.bridge.training.utils.padding_utils import (
    pad_or_truncate_2d_to_len,
    pad_or_truncate_attn_to_len,
    pad_or_truncate_pos_to_len,
)
from megatron.bridge.training.utils.pg_utils import get_pg_collection


logger = logging.getLogger(__name__)

try:
    import transformer_engine_torch as tex
except ImportError:
    tex = None


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

    raw_attn = batch.get("attention_mask")
    if isinstance(raw_attn, torch.Tensor) and raw_attn.dim() == 2:
        _batch_required_keys["_padding_mask"] = raw_attn.cuda(non_blocking=True)

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
        batch.get("_padding_mask"),
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
    Pad or truncate the batch sequences to the target length, and build packed sequences.
    If is_qwen3vl, return bshd tokens for be compatible with qwen3vl model.
    Otherwise, return thd tokens and packed sequences.
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


def _get_real_sequence_lengths(
    tokens: torch.Tensor,
    padding_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Return per-sample non-pad lengths. The 2D mask follows HF convention: 1/True means keep."""
    if padding_mask is None:
        return torch.full((tokens.size(0),), tokens.size(1), dtype=torch.int32, device=tokens.device)
    if padding_mask.dim() != 2:
        raise ValueError(f"QWEN3VL_THD_COMPACT_PACKING expects a 2D padding mask, got {tuple(padding_mask.shape)}")
    real_lengths = padding_mask.to(dtype=torch.int32, device=tokens.device).sum(dim=1)
    return real_lengths.clamp(min=1, max=tokens.size(1))


def _build_compact_packed_seq_params(
    real_lengths: torch.Tensor,
    pad_to_multiple_of: int,
) -> tuple[PackedSeqParams, torch.Tensor]:
    """Build THD metadata where real and physical padded boundaries are both preserved.

    ``cu_seqlens_q`` describes the true sample lengths used for attention/FLOPS
    semantics. ``cu_seqlens_q_padded`` describes the physical packed layout used
    by TransformerEngine's THD CP partitioner.
    """
    padded_lengths = torch.div(
        real_lengths + pad_to_multiple_of - 1,
        pad_to_multiple_of,
        rounding_mode="floor",
    ) * pad_to_multiple_of
    cu_seqlens = torch.zeros(real_lengths.numel() + 1, dtype=torch.int32, device=real_lengths.device)
    cu_seqlens[1:] = torch.cumsum(real_lengths.to(torch.int32), dim=0)
    cu_seqlens_padded = torch.zeros_like(cu_seqlens)
    cu_seqlens_padded[1:] = torch.cumsum(padded_lengths.to(torch.int32), dim=0)
    max_seqlen = int(padded_lengths.max().item())
    return (
        PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens,
            max_seqlen_q=max_seqlen,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_kv=max_seqlen,
            cu_seqlens_q_padded=cu_seqlens_padded,
            cu_seqlens_kv_padded=cu_seqlens_padded,
        ),
        padded_lengths,
    )


def _compact_pack_batch_sequences(
    tokens: torch.Tensor,
    labels: torch.Tensor | None,
    loss_mask: torch.Tensor | None,
    real_lengths: torch.Tensor,
    padded_lengths: torch.Tensor,
    pad_token_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
    """Pack BSHD rows directly into one THD stream without first padding to seq_length.

    The returned ``moe_padding_mask`` marks only the alignment padding inserted
    between packed samples. It is not an attention mask; MCore MoE uses it to
    exclude fake padding from router losses and expert-bias token statistics.
    """
    total_padded_len = int(padded_lengths.sum().item())
    packed_tokens = torch.full((1, total_padded_len), pad_token_id, dtype=tokens.dtype, device=tokens.device)
    packed_labels = (
        torch.full((1, total_padded_len), -100, dtype=labels.dtype, device=labels.device) if labels is not None else None
    )
    packed_loss_mask = (
        torch.zeros((1, total_padded_len), dtype=loss_mask.dtype, device=loss_mask.device)
        if loss_mask is not None
        else None
    )
    moe_padding_mask = torch.zeros((1, total_padded_len), dtype=torch.bool, device=tokens.device)

    offset = 0
    for batch_idx in range(tokens.size(0)):
        real_len = int(real_lengths[batch_idx].item())
        padded_len = int(padded_lengths[batch_idx].item())
        packed_tokens[0, offset : offset + real_len] = tokens[batch_idx, :real_len]
        if packed_labels is not None:
            packed_labels[0, offset : offset + real_len] = labels[batch_idx, :real_len]
        if packed_loss_mask is not None:
            packed_loss_mask[0, offset : offset + real_len] = loss_mask[batch_idx, :real_len]
        if padded_len > real_len:
            moe_padding_mask[0, offset + real_len : offset + padded_len] = True
        offset += padded_len

    return packed_tokens, packed_labels, packed_loss_mask, moe_padding_mask


def _get_compact_thd_cp_index(
    packed_seq_params: PackedSeqParams,
    total_tokens: int,
    cp_size: int,
    cp_rank: int,
) -> torch.Tensor | None:
    if cp_size <= 1:
        return None
    if tex is None:
        raise RuntimeError("QWEN3VL_THD_COMPACT_PACKING with CP>1 requires transformer_engine_torch")
    # THD CP must split each packed segment independently. TE's helper uses
    # cu_seqlens_q_padded to apply the standard zigzag CP split per segment.
    return tex.thd_get_partitioned_indices(
        packed_seq_params.cu_seqlens_q_padded,
        total_tokens,
        cp_size,
        cp_rank,
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
            padding_mask,
            multi_modal_inputs,
        ) = get_batch(data_iterator, state.cfg, use_mtp, is_first_pp_stage=is_first, is_last_pp_stage=is_last)
    timers("batch-generator").stop()

    # To be compatible with qwen3vl, we move the sequence padding and packing to forward_step function.
    # Qwen3VL model need the original input and do cp and sp split in model.forward.
    pack_sequences_in_batch = getattr(state.cfg.dataset, "pack_sequences_in_batch", False)

    compact_thd_packing = pack_sequences_in_batch and os.getenv("QWEN3VL_THD_COMPACT_PACKING", "0") == "1"

    if compact_thd_packing:
        # Contract with Qwen3VLModel.forward:
        # - ``input_ids`` below becomes compact THD [1, total_padded_tokens].
        # - Qwen3-VL still needs the original BSHD layout to compute MRoPE
        #   and to interpret image/video placeholder order per sample.
        compact_input_ids_bshd = tokens
        compact_attention_mask_bshd = (
            padding_mask
            if padding_mask is not None
            else torch.ones_like(compact_input_ids_bshd, dtype=torch.bool, device=compact_input_ids_bshd.device)
        )
        real_lengths = _get_real_sequence_lengths(tokens, compact_attention_mask_bshd)
        cp_size = this_pg_collection.cp.size()
        tp_size = this_pg_collection.tp.size()
        # CP requires each packed segment to be divisible by 2*CP for zigzag
        # splitting. Sequence parallelism further requires the CP-local chunk to
        # be divisible across TP ranks. Keep 16 for FP8/TE-friendly alignment.
        cp_multiple = 2 * cp_size if cp_size > 1 else 1
        sp_multiple = cp_size * tp_size if getattr(config, "sequence_parallel", False) and tp_size > 1 else 1
        pad_to_multiple_of = math.lcm(16, cp_multiple, sp_multiple)
        packed_seq_params, padded_lengths = _build_compact_packed_seq_params(real_lengths, pad_to_multiple_of)
        tokens, labels, loss_mask, moe_padding_mask = _compact_pack_batch_sequences(
            tokens,
            labels,
            loss_mask,
            real_lengths,
            padded_lengths,
        )
        # Compact THD uses packed_seq_params for sequence boundaries, so there
        # is no dense attention mask or precomputed BSHD position_ids to pass.
        attention_mask = None
        position_ids = None

        accumulate_flops_metadata(
            state,
            tokens,
            cu_seqlens_unpadded=packed_seq_params.cu_seqlens_q,
            image_grid_thw=multi_modal_inputs.get("image_grid_thw") if isinstance(multi_modal_inputs, dict) else None,
            video_grid_thw=multi_modal_inputs.get("video_grid_thw") if isinstance(multi_modal_inputs, dict) else None,
        )
        accumulate_token_throughput_metadata(
            state,
            real_tokens=int(real_lengths.sum().item()),
            packed_tokens=int(padded_lengths.sum().item()),
        )

        cp_index = _get_compact_thd_cp_index(
            packed_seq_params,
            tokens.size(1),
            this_pg_collection.cp.size(),
            this_pg_collection.cp.rank(),
        )
        if cp_index is not None:
            # The model CP-splits embeddings after vision/text combine. The
            # loss tensors live in the step closure, so they must be split here
            # with exactly the same THD index.
            labels = labels.index_select(1, cp_index) if labels is not None else None
            loss_mask = loss_mask.index_select(1, cp_index) if loss_mask is not None else None
            moe_padding_mask = moe_padding_mask.index_select(1, cp_index)

        forward_args = {
            "input_ids": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "packed_seq_params": packed_seq_params,
            "moe_padding_mask": moe_padding_mask,
            "qwen3vl_thd_compact_packing": True,
            # Extra internal metadata needed because compact THD removes the
            # original batch dimension before model.forward runs MRoPE.
            "qwen3vl_compact_input_ids_bshd": compact_input_ids_bshd,
            "qwen3vl_compact_attention_mask_bshd": compact_attention_mask_bshd,
        }
    else:
        tokens, labels, loss_mask, attention_mask, position_ids, packed_seq_params = pack_or_pad_batch_sequences(
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

        # Accumulate FLOPS metadata across micro-batches. When in-batch packing is
        # active, cu_seqlens_q describes the sub-seq boundaries used by THD attention.
        accumulate_flops_metadata(
            state,
            tokens,
            cu_seqlens=getattr(packed_seq_params, "cu_seqlens_q", None) if packed_seq_params is not None else None,
            image_grid_thw=multi_modal_inputs.get("image_grid_thw") if isinstance(multi_modal_inputs, dict) else None,
            video_grid_thw=multi_modal_inputs.get("video_grid_thw") if isinstance(multi_modal_inputs, dict) else None,
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
            if forward_args["labels"] is not None:
                # When using pp, labels could be None
                forward_args["labels"] = forward_args["labels"].reshape(1, -1)
            attention_mask = torch.ones(
                original_tokens.shape[0], original_tokens.shape[1], dtype=torch.bool, device=original_tokens.device
            )
            forward_args["attention_mask"] = attention_mask
            if forward_args["loss_mask"] is not None:
                forward_args["loss_mask"] = forward_args["loss_mask"].reshape(1, -1)
            # qwen3vl need the original input_ids and position_ids
            # use split attention mask for calculate loss
            forward_args["packed_seq_params"] = packed_seq_params

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
