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

"""DSv4-specific training step with contiguous CP partition support.

DSv4 hybrid attention uses a CSA (Compressed Sparse Attention) compressor that
exchanges boundary hidden states between adjacent CP ranks. This requires
contiguous token assignment (each rank gets a consecutive slice), unlike the
default zigzag interleaved assignment used by standard causal models.

MCore enforces cp_partition_mode='contiguous' is only valid with dsv4_hybrid
attention (see TransformerConfig validation). Use --step_func dsv4_step for
DSv4 SFT/pretrain with CP > 1.
"""

import logging
from typing import Iterable

import torch
from megatron.core import parallel_state
from megatron.core.models.gpt import GPTModel
from megatron.core.pipeline_parallel.utils import (
    is_pp_first_stage,
    is_pp_last_stage,
    is_vp_first_stage,
    is_vp_last_stage,
)
from megatron.core.utils import get_batch_on_this_cp_rank

from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.gpt_step import (
    _create_loss_function,
    _cu_seqlens_for_cp_partition,
    _current_stage_needs_mtp_inputs_from_layout,
    _forward_step_common,
    _has_packed_sequence_metadata,
    _middle_pp_stage_needs_batch,
    _packed_metadata_for_forward,
    _partition_packed_batch_for_cp,
    get_batch_from_iterator,
)
from megatron.bridge.training.state import GlobalState


logger = logging.getLogger(__name__)

# Sequence-length metadata keys — excluded from token-dimension slicing.
_SEQLEN_KEYS = frozenset(
    {
        "cu_seqlens",
        "cu_seqlens_unpadded",
        "cu_seqlens_argmin",
        "cu_seqlens_unpadded_argmin",
        "max_seqlen",
        "cu_seqlens_q",
        "cu_seqlens_kv",
        "cu_seqlens_q_padded",
        "cu_seqlens_kv_padded",
        "max_seqlen_q",
        "max_seqlen_kv",
        "token_count",
        "attention_mask",
    }
)


def _partition_packed_batch_contiguous(
    batch: dict[str, torch.Tensor],
    cp_size: int,
) -> dict[str, torch.Tensor]:
    """Slice a consecutive [start, end) token window for this CP rank.

    Required for DSv4 hybrid attention: the CSA compressor exchanges boundary
    hidden states between adjacent CP ranks, which requires contiguous token
    assignments. The packed sequence length must be divisible by cp_size —
    ensure packed_sequence_size = N * cp_size when running pack_sft_data.

    cu_seqlens are clipped to the local window; -1 padding sentinels in the
    legacy cu_seqlens path are stripped via _cu_seqlens_for_cp_partition to
    avoid invalid decreasing entries after clamping.
    """
    cp_rank = parallel_state.get_context_parallel_rank()
    cu_seqlens = _cu_seqlens_for_cp_partition(batch)

    _data_val = next((v for k, v in batch.items() if v is not None and k not in _SEQLEN_KEYS), None)
    if _data_val is None:
        return batch  # middle PP stage with no data tensors — nothing to slice

    total_tokens = _data_val.size(1)
    if total_tokens % cp_size != 0:
        raise RuntimeError(
            f"Contiguous CP partitioning requires packed sequence length ({total_tokens}) "
            f"to be divisible by cp_size ({cp_size}). "
            "Set packed_sequence_size to a multiple of cp_size when running pack_sft_data."
        )
    local_len = total_tokens // cp_size
    start = cp_rank * local_len
    end = start + local_len

    for key, val in batch.items():
        if val is None or key in _SEQLEN_KEYS:
            continue
        batch[key] = val[:, start:end].contiguous()

    # Clip cu_seqlens to local window; use trimmed source for legacy path.
    _seqlen_src = {
        "cu_seqlens": cu_seqlens,
        "cu_seqlens_q": batch.get("cu_seqlens_q"),
        "cu_seqlens_kv": batch.get("cu_seqlens_kv"),
        "cu_seqlens_unpadded": batch.get("cu_seqlens_unpadded"),
    }
    for key in _SEQLEN_KEYS:
        val = batch.get(key)
        if (
            val is None
            or "argmin" in key
            or key in {"max_seqlen", "max_seqlen_q", "max_seqlen_kv", "token_count", "attention_mask"}
        ):
            continue
        src_val = _seqlen_src.get(key)
        src = src_val if src_val is not None else val
        batch[key] = (src.clamp(min=start, max=end) - start).to(val.dtype)

    for argmin_key, cs_key in [
        ("cu_seqlens_argmin", "cu_seqlens"),
        ("cu_seqlens_unpadded_argmin", "cu_seqlens_unpadded"),
    ]:
        if batch.get(argmin_key) is not None and batch.get(cs_key) is not None:
            batch[argmin_key] = batch[argmin_key].new_tensor([[batch[cs_key].squeeze().numel()]])

    for max_key, cs_key in [
        ("max_seqlen", "cu_seqlens"),
        ("max_seqlen_q", "cu_seqlens_q"),
        ("max_seqlen_kv", "cu_seqlens_kv"),
    ]:
        cs = batch.get(cs_key)
        if cs is None or batch.get(max_key) is None:
            continue
        cs_flat = cs.squeeze()
        diffs = (cs_flat[1:] - cs_flat[:-1]).clamp(min=0)
        batch[max_key] = batch[max_key].new_tensor([[int(diffs.max().item()) if diffs.numel() > 0 else 0]])

    return batch


def get_batch(
    data_iterator: Iterable,
    cfg: ConfigContainer,
    use_mtp: bool = False,
    *,
    pg_collection,
    vp_stage: int | None = None,
):
    """get_batch with DSv4 contiguous CP partition support.

    Identical to gpt_step.get_batch but dispatches to contiguous partitioning
    when cfg.model.cp_partition_mode == 'contiguous', and injects cp_partition_mode
    into the batch so get_packed_seq_params can forward it to PackedSeqParams.
    """
    model_cfg = getattr(cfg, "model", None)
    vp_size = getattr(model_cfg, "virtual_pipeline_model_parallel_size", None)
    is_first = is_pp_first_stage(pg_collection.pp) and (
        vp_stage is None or is_vp_first_stage(vp_stage=vp_stage, vp_size=vp_size)
    )
    is_last = is_pp_last_stage(pg_collection.pp) and (
        vp_stage is None or is_vp_last_stage(vp_stage=vp_stage, vp_size=vp_size)
    )
    is_middle = (not is_first) and (not is_last)
    include_full_batch_fields = is_middle and _middle_pp_stage_needs_batch(cfg)
    include_mtp_inputs = use_mtp and _current_stage_needs_mtp_inputs_from_layout(
        cfg, pg_collection=pg_collection, is_last=is_last, vp_stage=vp_stage
    )
    if is_middle and not include_full_batch_fields and not include_mtp_inputs:
        return None, None, None, None, None, None

    batch = get_batch_from_iterator(
        data_iterator,
        include_mtp_inputs=include_mtp_inputs,
        skip_getting_attention_mask_from_dataset=getattr(
            cfg.dataset, "skip_getting_attention_mask_from_dataset", True
        ),
        is_first_pp_stage=is_first,
        is_last_pp_stage=is_last,
        include_full_batch_fields=include_full_batch_fields,
    )

    cp_size = pg_collection.cp.size()
    has_packed = _has_packed_sequence_metadata(batch)
    if has_packed and cp_size > 1:
        cp_mode = getattr(cfg.model, "cp_partition_mode", "zigzag")
        if cp_mode == "contiguous":
            batch = _partition_packed_batch_contiguous(batch, cp_size)
        else:
            batch = _partition_packed_batch_for_cp(batch, pg_collection.cp)
        # Inject cp_partition_mode so get_packed_seq_params forwards it to PackedSeqParams.
        batch["cp_partition_mode"] = cp_mode
    else:
        batch = get_batch_on_this_cp_rank(batch, is_hybrid_cp=False, cp_group=pg_collection.cp)

    return (
        batch["tokens"],
        batch["labels"],
        batch["loss_mask"],
        batch.get("attention_mask"),
        batch["position_ids"],
        _packed_metadata_for_forward(batch),
    )


def forward_step(
    state: GlobalState,
    data_iterator: Iterable,
    model: GPTModel,
    return_schedule_plan: bool = False,
):
    """Forward training step for DSv4 with contiguous CP partition support."""
    output, loss_mask = _forward_step_common(
        state, data_iterator, model, return_schedule_plan, _get_batch_fn=get_batch
    )
    return output, _create_loss_function(
        loss_mask,
        check_for_nan_in_loss=state.cfg.rerun_state_machine.check_for_nan_in_loss,
        check_for_spiky_loss=state.cfg.rerun_state_machine.check_for_spiky_loss,
    )
