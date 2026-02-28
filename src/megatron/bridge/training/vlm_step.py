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


def pack_batch_sequences(
    tokens: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    pad_token_id: int = 0,
    pad_to_multiple_of: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pack sequences in a batch by concatenating them and removing padding.

    Args:
        tokens: [batch_size, seq_len]
        labels: [batch_size, seq_len]
        loss_mask: [batch_size, seq_len]
        attention_mask: [batch_size, 1, seq_len, seq_len] or None
        position_ids: [batch_size, seq_len]
        pad_token_id: Token ID used for padding

    Returns:
        Tuple of:
        - packed_tokens: [1, total_len] - concatenated sequences
        - packed_labels: [1, total_len]
        - packed_loss_mask: [1, total_len]
        - packed_attention_mask: None (not used with packing)
        - packed_position_ids: [1, total_len]
        - cu_seqlens: [num_sequences + 1] - cumulative sequence lengths
        - max_seqlen: tensor - max sequence length in packed batch
    """
    batch_size, seq_len = tokens.shape
    device = tokens.device

    # Find actual sequence lengths (excluding padding)
    # Assuming padding is at the end and uses pad_token_id (0)
    seq_lengths = []
    valid_sequences = []

    for i in range(batch_size):
        # Find first padding token or use full length
        non_pad_mask = tokens[i] != pad_token_id
        if non_pad_mask.any():
            # Find the last non-padding token
            last_valid_idx = non_pad_mask.nonzero(as_tuple=True)[0][-1].item() + 1
        else:
            # Empty sequence, skip
            continue

        seq_lengths.append(last_valid_idx)
        valid_sequences.append(i)

    if len(valid_sequences) == 0:
        # No valid sequences, return empty packed batch
        logger.warning("No valid sequences found in batch, skipping packing")
        return (
            tokens[:, :0],  # Empty tensor
            labels[:, :0],
            loss_mask[:, :0],
            attention_mask,
            position_ids[:, :0],
            torch.tensor([0], dtype=torch.int32, device=device),  # Empty cu_seqlens
            torch.tensor(0, dtype=torch.int32, device=device),
        )

    # Build cumulative sequence lengths
    cu_seqlens = [0]
    padded_seq_lengths = []
    for length in seq_lengths:
        if pad_to_multiple_of > 1:
            padded_len = ((length + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
        else:
            padded_len = length
        padded_seq_lengths.append(padded_len)
        # Use padded lengths for cu_seqlens so THD RoPE splits sum correctly under CP.
        cu_seqlens.append(cu_seqlens[-1] + padded_len)

    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
    max_seqlen = torch.tensor(max(padded_seq_lengths), dtype=torch.int32, device=device)
    total_len = cu_seqlens[-1].item()

    # Concatenate sequences (remove padding)
    packed_tokens = torch.zeros(1, total_len, dtype=tokens.dtype, device=device)
    packed_labels = torch.zeros(1, total_len, dtype=labels.dtype, device=device)
    packed_loss_mask = torch.zeros(1, total_len, dtype=loss_mask.dtype, device=device)
    packed_position_ids = torch.zeros(1, total_len, dtype=position_ids.dtype, device=device)

    offset = 0
    for i, seq_idx in enumerate(valid_sequences):
        length = seq_lengths[i]
        padded_len = padded_seq_lengths[i]
        pad_len = padded_len - length
        packed_tokens[0, offset : offset + length] = tokens[seq_idx, :length]
        packed_labels[0, offset : offset + length] = labels[seq_idx, :length]
        packed_loss_mask[0, offset : offset + length] = loss_mask[seq_idx, :length]
        packed_position_ids[0, offset : offset + length] = position_ids[seq_idx, :length]
        if pad_len > 0:
            packed_tokens[0, offset + length : offset + padded_len] = pad_token_id
            packed_labels[0, offset + length : offset + padded_len] = -100
            packed_loss_mask[0, offset + length : offset + padded_len] = 0
            start_pos = position_ids[seq_idx, length - 1] + 1
            packed_position_ids[0, offset + length : offset + padded_len] = torch.arange(
                start_pos,
                start_pos + pad_len,
                device=device,
                dtype=position_ids.dtype,
            )
        offset += padded_len

    logger.debug(
        f"Packed {len(valid_sequences)} sequences: lengths={seq_lengths}, total_len={total_len}, max_len={max_seqlen}"
    )

    # Attention mask is not used with packed sequences (handled by cu_seqlens)
    packed_attention_mask = None

    return (
        packed_tokens,
        packed_labels,
        packed_loss_mask,
        packed_attention_mask,
        packed_position_ids,
        cu_seqlens,
        max_seqlen,
    )


def get_batch(data_iterator: Iterable, cfg: ConfigContainer, use_mtp: bool = False, *, pg_collection) -> tuple[...]:
    """Generate a batch.

    Args:
        data_iterator: Input data iterator
        cfg: Configuration container
        use_mtp: Whether Multi-Token Prediction layers are enabled

    Returns:
        tuple of tensors containing tokens, labels, loss_mask, attention_mask, position_ids,
        cu_seqlens, cu_seqlens_argmin, max_seqlen, visual_inputs (container of optional modalities)
    """
    is_first = is_pp_first_stage(pg_collection.pp)
    is_last = is_pp_last_stage(pg_collection.pp)

    # All PP stages load from iterator to get input_ids and visual grid info
    # This allows each stage to compute MRoPE position_ids locally without broadcasting
    batch = get_batch_from_iterator(
        data_iterator,
        use_mtp,
        getattr(cfg.dataset, "skip_getting_attention_mask_from_dataset", True),
        is_first_pp_stage=is_first,
        is_last_pp_stage=is_last,
    )
    enable_packing = getattr(cfg.dataset, "pack_sequences_in_batch", False)

    if not enable_packing:
        # When using pipeline parallelism, ensure fixed shapes equal to cfg.model.seq_length
        if getattr(cfg.model, "pipeline_model_parallel_size", 1) > 1:
            seq_len = cfg.model.seq_length

            tokens_or_input = batch.get("tokens") if batch.get("tokens") is not None else batch.get("input_ids")
            tokens_or_input = pad_or_truncate_2d_to_len(tokens_or_input, seq_len, seq_len, pad_value=0)
            if batch.get("tokens") is not None:
                batch["tokens"] = tokens_or_input  # type: ignore[assignment]
            else:
                batch["input_ids"] = tokens_or_input  # type: ignore[assignment]
            batch["labels"] = pad_or_truncate_2d_to_len(batch.get("labels"), seq_len, seq_len, pad_value=-100)  # type: ignore[assignment]
            batch["loss_mask"] = pad_or_truncate_2d_to_len(batch.get("loss_mask"), seq_len, seq_len, pad_value=0)  # type: ignore[assignment]
            batch["position_ids"] = pad_or_truncate_pos_to_len(batch.get("position_ids"), seq_len, seq_len)  # type: ignore[assignment]
            if batch.get("attention_mask") is not None:
                batch["attention_mask"] = pad_or_truncate_attn_to_len(batch.get("attention_mask"), seq_len, seq_len)  # type: ignore[assignment]
        else:
            # No PP: pad sequence length to nearest multiple of 128 for efficiency (capped at model seq_length)
            seq_cap = cfg.model.seq_length

            def _ceil_to_mult(n: int, mult: int) -> int:
                return ((n + mult - 1) // mult) * mult

            tokens_or_input = batch.get("tokens") if batch.get("tokens") is not None else batch.get("input_ids")
            if tokens_or_input is not None:
                cur_len = tokens_or_input.size(1)
                target_len = min(seq_cap, _ceil_to_mult(cur_len, 128))

                # tokens/input_ids
                padded_tokens = pad_or_truncate_2d_to_len(tokens_or_input, target_len, seq_cap, pad_value=0)
                if batch.get("tokens") is not None:
                    batch["tokens"] = padded_tokens  # type: ignore[assignment]
                else:
                    batch["input_ids"] = padded_tokens  # type: ignore[assignment]

                # labels and loss mask
                batch["labels"] = pad_or_truncate_2d_to_len(batch.get("labels"), target_len, seq_cap, pad_value=-100)  # type: ignore[assignment]
                batch["loss_mask"] = pad_or_truncate_2d_to_len(
                    batch.get("loss_mask"), target_len, seq_cap, pad_value=0
                )  # type: ignore[assignment]

                # position_ids: extend with increasing positions
                pos = batch.get("position_ids")
                pos = pad_or_truncate_pos_to_len(pos, target_len, seq_cap)
                if pos is not None:
                    batch["position_ids"] = pos  # type: ignore[assignment]

                # attention_mask if present
                attn = batch.get("attention_mask")
                if attn is not None:
                    attn = pad_or_truncate_attn_to_len(attn, target_len, seq_cap)
                    batch["attention_mask"] = attn  # type: ignore[assignment]

    visual_inputs = batch.get("visual_inputs")
    cp_size = pg_collection.cp.size() if pg_collection is not None and pg_collection.cp is not None else 1

    if enable_packing:
        # Pack sequences
        tokens_or_input = batch.get("tokens") if batch.get("tokens") is not None else batch.get("input_ids")
        (
            packed_tokens,
            packed_labels,
            packed_loss_mask,
            packed_attention_mask,
            packed_position_ids,
            cu_seqlens,
            max_seqlen,
        ) = pack_batch_sequences(
            tokens=tokens_or_input,
            labels=batch.get("labels"),
            loss_mask=batch.get("loss_mask"),
            attention_mask=batch.get("attention_mask"),
            position_ids=batch.get("position_ids"),
            pad_token_id=0,
            pad_to_multiple_of=cp_size * 2 if cp_size > 1 else 1,
        )

        # Update batch dict with packed tensors
        if batch.get("tokens") is not None:
            batch["tokens"] = packed_tokens
        else:
            batch["input_ids"] = packed_tokens
        batch["labels"] = packed_labels
        batch["loss_mask"] = packed_loss_mask
        batch["attention_mask"] = packed_attention_mask
        batch["position_ids"] = packed_position_ids

        # # Add packing metadata
        logger.debug(f"Packed batch: cu_seqlens={cu_seqlens.tolist()}, max_seqlen={max_seqlen}")
    else:
        # No packing, use dummy values
        cu_seqlens = None
        max_seqlen = None

    return (
        (batch.get("tokens") if batch.get("tokens") is not None else batch.get("input_ids")),
        batch.get("labels"),
        batch.get("loss_mask"),  # Full packed loss_mask, will be CP-sliced by model
        batch.get("attention_mask"),
        batch.get("position_ids"),
        cu_seqlens,
        max_seqlen,
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
            max_seqlen,
            visual_inputs,
        ) = get_batch(data_iterator, state.cfg, use_mtp, pg_collection=pg_collection)
    timers("batch-generator").stop()

    forward_args = {
        "input_ids": tokens,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "loss_mask": loss_mask,  # Pass full loss_mask so model can slice it consistently with labels
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

    # -------------------------------------------------------------------------
    # Runtime FLOPs support for packed / padded language tokens
    # -------------------------------------------------------------------------
    # `flop_utils.num_floating_point_operations(cfg, batch_size)` only receives `(cfg, batch_size)`.
    # Attention compute depends on the actual per-sample lengths in the microbatch:
    #   total tokens: sum_i L_i
    #   attention matmul: sum_i L_i^2 (causal uses ~half, handled in flop_utils)
    #
    # We compute per-sample averages here and stash them on the config for more accurate FLOPs.
    microbatch_size_for_stats = None
    try:
        packed = forward_args.get("packed_seq_params", None)
        cu = None
        if packed is not None:
            cu = getattr(packed, "cu_seqlens_q_padded", None)
            if cu is None:
                cu = getattr(packed, "cu_seqlens_q", None)

        if cu is not None:
            lengths = (cu[1:] - cu[:-1]).to(torch.int64)
            denom = max(1, int(lengths.numel()))
            total_tokens_padded = int(cu[-1].item())
            sum_seqlen_sq_padded = int((lengths * lengths).sum().item())

            state.cfg._runtime_lm_total_tokens_padded_per_sample = float(total_tokens_padded) / float(denom)
            state.cfg._runtime_lm_sum_seqlen_sq_padded_per_sample = float(sum_seqlen_sq_padded) / float(denom)
            microbatch_size_for_stats = denom
        else:
            # Non-packed path: use the padded sequence length directly.
            seq_len_padded = int(tokens.shape[1])
            state.cfg._runtime_lm_total_tokens_padded_per_sample = float(seq_len_padded)
            state.cfg._runtime_lm_sum_seqlen_sq_padded_per_sample = float(seq_len_padded * seq_len_padded)
            microbatch_size_for_stats = int(tokens.shape[0]) if tokens is not None and tokens.dim() >= 2 else 1
    except Exception:
        # FLOPs logging must never break training.
        microbatch_size_for_stats = int(tokens.shape[0]) if tokens is not None and tokens.dim() >= 2 else 1

    microbatch_size_for_stats = max(1, int(microbatch_size_for_stats or 1))

    # -------------------------------------------------------------------------
    # Runtime FLOPs support for vision-language models
    # -------------------------------------------------------------------------
    # Vision FLOPs depend on the *actual* number of visual tokens (image/video size and frames).
    # Stash per-sample averages on the config so flop_utils can use them later.
    vision_cfg = getattr(state.cfg.model, "vision_config", None)
    if vision_cfg is not None:
        image_grid_thw = forward_args.get("image_grid_thw", None)
        video_grid_thw = forward_args.get("video_grid_thw", None)

        vision_tokens_pre_total = 0
        vision_sum_seqlen_sq_pre_total = 0

        if image_grid_thw is not None:
            vision_tokens_pre_total += int(image_grid_thw.prod(dim=-1).sum().item())
            hw = image_grid_thw[:, 1].to(torch.int64) * image_grid_thw[:, 2].to(torch.int64)
            t = image_grid_thw[:, 0].to(torch.int64)
            vision_sum_seqlen_sq_pre_total += int((t * hw * hw).sum().item())

        if video_grid_thw is not None:
            vision_tokens_pre_total += int(video_grid_thw.prod(dim=-1).sum().item())
            hw = video_grid_thw[:, 1].to(torch.int64) * video_grid_thw[:, 2].to(torch.int64)
            t = video_grid_thw[:, 0].to(torch.int64)
            vision_sum_seqlen_sq_pre_total += int((t * hw * hw).sum().item())

        vision_tokens_pre_per_sample = float(vision_tokens_pre_total) / float(microbatch_size_for_stats)
        vision_sum_seqlen_sq_pre_per_sample = float(vision_sum_seqlen_sq_pre_total) / float(microbatch_size_for_stats)

        spatial_merge_size = getattr(vision_cfg, "spatial_merge_size", None)
        merge_sq = int(spatial_merge_size) ** 2 if spatial_merge_size is not None else 1
        vision_tokens_post_per_sample = vision_tokens_pre_per_sample / float(max(1, merge_sq))

        state.cfg._runtime_vision_tokens_pre_per_sample = vision_tokens_pre_per_sample
        state.cfg._runtime_vision_tokens_post_per_sample = vision_tokens_post_per_sample
        state.cfg._runtime_vision_sum_seqlen_sq_pre_per_sample = vision_sum_seqlen_sq_pre_per_sample

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
