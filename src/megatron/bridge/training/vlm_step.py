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
        if "cu_seqlens_unpadded" in batch:
            required_device_keys.add("cu_seqlens_unpadded")
        required_host_keys.add("cu_seqlens_argmin")
        required_host_keys.add("max_seqlen")
        if "cu_seqlens_unpadded_argmin" in batch:
            required_host_keys.add("cu_seqlens_unpadded_argmin")

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

    # Preserve collator's 2D padding mask for sequence packing length detection.
    # skip_getting_attention_mask_from_dataset may discard the attention_mask for model
    # forward, but packing still needs it to identify real vs padding positions.
    raw_attn = batch.get("attention_mask")
    if isinstance(raw_attn, torch.Tensor) and raw_attn.dim() == 2:
        _batch_required_keys["_padding_mask"] = raw_attn.cuda(non_blocking=True)

    return _batch_required_keys


def pack_batch_sequences(
    tokens: torch.Tensor,
    labels: torch.Tensor | None,
    loss_mask: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor,
    pad_token_id: int = 0,
    pad_to_multiple_of: int = 1,
    padding_mask: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
]:
    """
    Pack sequences in a batch by concatenating them and removing padding.

    Args:
        tokens: [batch_size, seq_len]
        labels: [batch_size, seq_len] or None (non-last PP stages)
        loss_mask: [batch_size, seq_len] or None (non-last PP stages)
        attention_mask: [batch_size, 1, seq_len, seq_len] or None
        position_ids: [batch_size, seq_len]
        pad_token_id: Token ID used for padding (fallback when padding_mask is unavailable)
        pad_to_multiple_of: Pad each sequence length to a multiple of this value
        padding_mask: [batch_size, seq_len] explicit mask from collator (1=real, 0=padding).
            When provided, this is used instead of pad_token_id for robust length detection.

    Returns:
        Tuple of:
        - packed_tokens: [1, total_len] - concatenated sequences
        - packed_labels: [1, total_len] or None
        - packed_loss_mask: [1, total_len] or None
        - packed_attention_mask: None (not used with packing)
        - packed_position_ids: [1, total_len]
        - cu_seqlens: [num_sequences + 1] - cumulative sequence lengths
        - max_seqlen: tensor - max sequence length in packed batch
    """
    batch_size, seq_len = tokens.shape
    device = tokens.device

    seq_lengths = []
    valid_sequences = []

    for i in range(batch_size):
        if padding_mask is not None:
            length = int(padding_mask[i].sum().item())
        else:
            non_pad_mask = tokens[i] != pad_token_id
            if non_pad_mask.all():
                length = seq_len
            elif non_pad_mask.any():
                length = non_pad_mask.nonzero(as_tuple=True)[0][-1].item() + 1
            else:
                length = 0

        if length > 0:
            seq_lengths.append(length)
            valid_sequences.append(i)

    if len(valid_sequences) == 0:
        # No valid sequences, return empty packed batch
        logger.warning("No valid sequences found in batch, skipping packing")
        return (
            tokens[:, :0],
            labels[:, :0] if labels is not None else None,
            loss_mask[:, :0] if loss_mask is not None else None,
            attention_mask,
            position_ids[:, :0],
            torch.tensor([0], dtype=torch.int32, device=device),
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
    packed_labels = torch.zeros(1, total_len, dtype=labels.dtype, device=device) if labels is not None else None
    packed_loss_mask = (
        torch.zeros(1, total_len, dtype=loss_mask.dtype, device=device) if loss_mask is not None else None
    )
    packed_position_ids = torch.zeros(1, total_len, dtype=position_ids.dtype, device=device)

    offset = 0
    for i, seq_idx in enumerate(valid_sequences):
        length = seq_lengths[i]
        padded_len = padded_seq_lengths[i]
        pad_len = padded_len - length
        packed_tokens[0, offset : offset + length] = tokens[seq_idx, :length]
        if packed_labels is not None:
            packed_labels[0, offset : offset + length] = labels[seq_idx, :length]
        if packed_loss_mask is not None:
            packed_loss_mask[0, offset : offset + length] = loss_mask[seq_idx, :length]
        packed_position_ids[0, offset : offset + length] = position_ids[seq_idx, :length]
        if pad_len > 0:
            packed_tokens[0, offset + length : offset + padded_len] = pad_token_id
            if packed_labels is not None:
                packed_labels[0, offset + length : offset + padded_len] = -100
            if packed_loss_mask is not None:
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
    tp_size = pg_collection.tp.size() if pg_collection is not None and pg_collection.tp is not None else 1
    has_sp = getattr(cfg.model, "sequence_parallel", False)

    # Energon pre-packed path: cu_seqlens already present from TaskEncoder packing
    energon_cu_seqlens = batch.get("cu_seqlens")
    if energon_cu_seqlens is not None:
        tokens_or_input = batch.get("tokens") if batch.get("tokens") is not None else batch.get("input_ids")
        energon_max_seqlen = batch.get("max_seqlen")
        energon_cu_argmin = batch.get("cu_seqlens_argmin")

        # Log detailed iteration stats for FLOPs analysis
        if tokens_or_input is not None:
            _seq_dim = tokens_or_input.shape[-1]
            if energon_cu_seqlens.dim() == 1:
                _content_len = energon_cu_seqlens[-1].item()
            elif energon_cu_argmin is not None:
                _argmin_val = energon_cu_argmin.item() if energon_cu_argmin.dim() == 0 else energon_cu_argmin[0].item()
                _content_len = energon_cu_seqlens[0, _argmin_val - 1].item()
            else:
                _content_len = energon_cu_seqlens[0, -1].item()
            _pad_len = _seq_dim - _content_len
            _n_img_toks = int((tokens_or_input == 151655).sum().item())
            _vit_shape = (
                visual_inputs.pixel_values.shape
                if visual_inputs is not None and visual_inputs.pixel_values is not None
                else "None"
            )
            logger.info(
                f"[IterStats] decoder_input={list(tokens_or_input.shape)}, "
                f"content={_content_len}, pad={_pad_len}, "
                f"image_tokens_in_decoder={_n_img_toks}, "
                f"vit_input={_vit_shape}"
            )

        return (
            tokens_or_input,
            batch.get("labels"),
            batch.get("loss_mask"),
            batch.get("attention_mask"),
            batch.get("position_ids"),
            energon_cu_seqlens,
            energon_max_seqlen,
            visual_inputs,
            energon_cu_argmin,
        )

    if enable_packing:
        # In-batch packing (concatenate all micro-batch sequences)
        tokens_or_input = batch.get("tokens") if batch.get("tokens") is not None else batch.get("input_ids")

        # Compute pad_to_multiple_of as lcm of CP and SP constraints.
        # CP zigzag requires divisibility by 2*cp_size; SP reduce_scatter requires
        # the per-CP-rank length to be divisible by tp_size (i.e. total divisible by
        # cp_size*tp_size). Reference: megatron/core/models/multimodal/context_parallel.py
        cp_multiple = 2 * cp_size if cp_size > 1 else 1
        sp_multiple = cp_size * tp_size if has_sp and tp_size > 1 else 1
        pad_multiple = math.lcm(cp_multiple, sp_multiple)

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
            pad_to_multiple_of=pad_multiple,
            padding_mask=batch.get("_padding_mask"),
        )

        if batch.get("tokens") is not None:
            batch["tokens"] = packed_tokens
        else:
            batch["input_ids"] = packed_tokens
        batch["labels"] = packed_labels
        batch["loss_mask"] = packed_loss_mask
        batch["attention_mask"] = packed_attention_mask
        batch["position_ids"] = packed_position_ids

        logger.debug(f"In-batch packed: cu_seqlens={cu_seqlens.tolist()}, max_seqlen={max_seqlen}")
    else:
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
        None,  # cu_seqlens_argmin (only set for Energon pre-packed path)
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
            cu_seqlens_argmin,
        ) = get_batch(data_iterator, state.cfg, use_mtp, pg_collection=pg_collection)
    timers("batch-generator").stop()

    # ---- Unified debug dump for first 2 iterations (both BSHD and Energon) ----
    _iter = getattr(state, "_debug_iter_count", 0)
    if _iter < 2 and tokens is not None:
        _is_energon = cu_seqlens is not None
        _mode = "Energon-THD" if _is_energon else "HF-BSHD"
        _t = tokens.squeeze()
        _lm = loss_mask.squeeze() if loss_mask is not None else None
        _lab = labels.squeeze() if labels is not None else None
        _sup_count = int((_lab != -100).sum().item()) if _lab is not None else 0
        _lm_sum = int(_lm.sum().item()) if _lm is not None else 0
        logger.info(
            f"\n{'='*70}\n"
            f"[DEBUG {_mode}] iteration {_iter}\n"
            f"  tokens.shape        = {list(tokens.shape)}\n"
            f"  tokens[:20]         = {_t[:20].tolist()}\n"
            f"  tokens[-10:]        = {_t[-10:].tolist()}\n"
            f"  labels[:20]         = {_lab[:20].tolist() if _lab is not None else None}\n"
            f"  loss_mask sum/total = {_lm_sum} / {_t.shape[0]}\n"
            f"  supervised labels   = {_sup_count}\n"
            f"  cu_seqlens (raw)    = {cu_seqlens}\n"
            f"  position_ids[:10]   = {position_ids.squeeze()[:10].tolist() if position_ids is not None else None}\n"
            f"  visual_inputs       = {'present' if visual_inputs is not None else 'None'}\n"
            f"{'='*70}"
        )
        # Label alignment check: labels[i] should equal tokens[i+1] at supervised positions
        if _lab is not None:
            _sup_idx = (_lab != -100).nonzero(as_tuple=True)[0]
            if len(_sup_idx) > 0:
                _valid = _sup_idx[_sup_idx < len(_t) - 1]
                _match = int((_lab[_valid] == _t[_valid + 1]).sum().item())
                logger.info(
                    f"  label alignment: labels[i]==tokens[i+1] at {_match}/{len(_valid)} supervised positions "
                    f"(first_sup_pos={_sup_idx[0].item()}, last_sup_pos={_sup_idx[-1].item()})"
                )
                if _match < len(_valid):
                    _mismatch_idx = _valid[_lab[_valid] != _t[_valid + 1]][:5]
                    for _mi in _mismatch_idx:
                        _mi = _mi.item()
                        logger.info(
                            f"    MISMATCH pos {_mi}: label={_lab[_mi].item()} vs tokens[{_mi+1}]={_t[_mi+1].item()}"
                        )
        state._debug_iter_count = _iter + 1

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
        physical_seq_len = tokens.shape[-1]

        # Determine the number of valid entries in cu_seqlens.
        cu = cu_seqlens.squeeze()
        if cu_seqlens_argmin is not None:
            n_valid = int(cu_seqlens_argmin.item())
        elif cu.dim() == 1:
            n_valid = len(cu)
        else:
            n_valid = int(torch.argmin(cu).item())

        cu_clean = cu[:n_valid]
        last_boundary = int(cu_clean[-1].item())

        if last_boundary < physical_seq_len:
            # Trailing padding exists: extend the last real segment's padded
            # boundary to cover the tail so that all layers (TE attention, GDN,
            # etc.) see the same sequence boundaries.  We intentionally do NOT
            # set cu_seqlens_unpadded here so that cu_seqlens_q_padded stays
            # None — this forces every layer to treat pad tokens identically to
            # real tokens, avoiding undefined attention output at pad positions
            # that would otherwise propagate NaN through subsequent layers and
            # into the backward pass.  loss_mask already zeros pad positions.
            cu_padded = cu_clean.clone()
            cu_padded[-1] = physical_seq_len

            padded_diffs = cu_padded[1:] - cu_padded[:-1]
            max_seqlen_padded = padded_diffs.max()

            packed_seq_dict = {
                "cu_seqlens": cu_padded,
                "cu_seqlens_argmin": torch.tensor(n_valid),
                "max_seqlen": max_seqlen_padded,
            }
        else:
            # No trailing padding (content fills the full sequence length).
            packed_seq_dict = {
                "cu_seqlens": cu_seqlens,
                "max_seqlen": max_seqlen,
            }
            if cu_seqlens_argmin is not None:
                packed_seq_dict["cu_seqlens_argmin"] = cu_seqlens_argmin
            elif cu_seqlens.dim() == 1:
                packed_seq_dict["cu_seqlens_argmin"] = torch.tensor(len(cu_seqlens))

        forward_args["packed_seq_params"] = get_packed_seq_params(packed_seq_dict)

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
