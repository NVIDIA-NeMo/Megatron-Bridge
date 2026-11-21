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

At CP>1, this step requires packed THD metadata and both native model fields
attention_cp_layout=linear_cp_layout='contiguous'. Non-packed batches and
zigzag layouts are rejected; there is no fallback to the standard zigzag CP
partitioner. A legacy dev-only cp_partition_mode setting is not sufficient.
Use --step-func dsv4_step with a packed dataset. MCore's native DSv4 packed-CP
support validates the supported model topology.
"""

import logging
from dataclasses import fields
from typing import Iterable

import torch
from megatron.core import parallel_state
from megatron.core.models.gpt import GPTModel
from megatron.core.packed_seq_params import PackedSeqParams
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
    _current_stage_needs_mtp_inputs_from_layout,
    _forward_step_common,
    _has_packed_sequence_metadata,
    _middle_pp_stage_needs_batch,
    get_batch_from_iterator,
)
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.utils.packed_seq_utils import get_packed_seq_params


try:
    from megatron.core.context_parallel_layout import finalize_packed_seq_params
except ImportError:
    # Older MCore releases do not expose precomputed THD CP layout routes.
    finalize_packed_seq_params = None


logger = logging.getLogger(__name__)

# Keep global packed boundaries for every pipeline stage. CP layout is a model
# configuration field on main, not a PackedSeqParams field.
_DSV4_CURRENT_PACKED_SEQ_PARAM_KEYS = (
    "cu_seqlens_q",
    "cu_seqlens_kv",
    "cu_seqlens_q_padded",
    "cu_seqlens_kv_padded",
    "max_seqlen_q",
    "max_seqlen_kv",
    "pad_between_seqs",
    "total_tokens",
)
_DSV4_LEGACY_PACKED_SEQ_PARAM_KEYS = (
    "cu_seqlens",
    "cu_seqlens_unpadded",
    "cu_seqlens_argmin",
    "max_seqlen",
    "cu_seqlens_unpadded_argmin",
    "total_tokens",
)


def _packed_metadata_for_forward(batch: dict, *, include_padding_mask: bool = False) -> dict | None:
    """Extract global boundaries, preserving native DSv4 router-padding semantics.

    Only the fixed THD benchmark requests padding_mask forwarding. Normal native
    packed batches continue counting alignment padding in router statistics.
    """
    if batch.get("cu_seqlens_q") is not None:
        keys = _DSV4_CURRENT_PACKED_SEQ_PARAM_KEYS
    elif batch.get("cu_seqlens") is not None:
        keys = _DSV4_LEGACY_PACKED_SEQ_PARAM_KEYS
    else:
        return None
    metadata = {key: batch[key] for key in keys if batch.get(key) is not None}
    if include_padding_mask and batch.get("padding_mask") is not None:
        metadata["padding_mask"] = batch["padding_mask"]
    return metadata


def _get_dsv4_packed_seq_params(batch: dict) -> PackedSeqParams:
    """Prepare THD CP routes before the attention forward or graph capture."""
    # DSv4 has no SSM layers, so it does not need the seq_idx allocation that
    # PackedSeqParams derives from total_tokens for generic hybrid models.
    metadata = {key: value for key, value in batch.items() if key != "total_tokens"}
    params = get_packed_seq_params(metadata)
    if finalize_packed_seq_params is not None:
        params = finalize_packed_seq_params(params)
    return params


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
        "total_tokens",
        "pad_between_seqs",
        "cp_partition_mode",
    }
)

# These are the sequence-shaped fields selected by get_batch_from_iterator.
# padding_mask follows tokens; attention_mask is retained as global metadata.
_TOKEN_KEYS = ("tokens", "labels", "loss_mask", "position_ids", "padding_mask")


def _partition_packed_batch_contiguous(
    batch: dict[str, torch.Tensor], cp_size: int, cp_rank: int | None = None
) -> dict[str, torch.Tensor]:
    """Slice a consecutive [start, end) token window for this CP rank.

    Only tokens, labels, loss_mask, position_ids, and padding_mask are sliced.
    Sequence-length metadata (cu_seqlens, max_seqlen, ...) is intentionally kept
    at global values — the DSv4 CSA compressor needs global sequence boundaries to
    correctly exchange boundary hidden states between adjacent CP ranks.
    This mirrors how zigzag mode leaves cu_seqlens untouched.

    The packed sequence length must be divisible by cp_size — ensure
    packed_sequence_size = N * cp_size when running pack_sft_data.
    """
    cp_rank = parallel_state.get_context_parallel_rank() if cp_rank is None else cp_rank
    if not 0 <= cp_rank < cp_size:
        raise ValueError("CP rank must be within its explicit process group.")

    unexpected = [k for k, v in batch.items() if v is not None and k not in _SEQLEN_KEYS and k not in _TOKEN_KEYS]
    if unexpected:
        raise ValueError(f"Unsupported populated packed batch fields: {unexpected}.")
    _data_val = next((batch[k] for k in _TOKEN_KEYS if batch.get(k) is not None), None)
    if _data_val is None:
        return batch  # middle PP stage with no data tensors — nothing to slice

    if not isinstance(_data_val, torch.Tensor) or _data_val.ndim != 2 or _data_val.size(0) != 1:
        raise ValueError("DSv4 packed CP requires micro-batch size 1 and [1, tokens] data tensors.")
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

    for key in _TOKEN_KEYS:
        val = batch.get(key)
        if val is None:
            continue
        if not isinstance(val, torch.Tensor) or val.ndim != 2 or val.shape != _data_val.shape:
            raise ValueError(f"Packed data field {key!r} must match the global [1, tokens] shape.")
        batch[key] = val[:, start:end].contiguous()

    return batch


def get_batch(  # pragma: no cover
    data_iterator: Iterable, cfg: ConfigContainer, use_mtp: bool = False, *, pg_collection, vp_stage: int | None = None
):
    """get_batch with DSv4 contiguous CP partition support.

    Native attention_cp_layout selects contiguous partitioning. Every CP
    pipeline stage consumes the global packed metadata before token slicing.
    """
    model_cfg = getattr(cfg, "model", None)
    cp_size = pg_collection.cp.size()
    if cp_size > 1 and getattr(model_cfg, "attention_cp_layout", None) != "contiguous":
        raise ValueError("DSv4 packed context parallelism requires attention_cp_layout='contiguous'.")
    if cp_size > 1 and getattr(model_cfg, "linear_cp_layout", None) != "contiguous":
        raise ValueError("DSv4 packed context parallelism requires linear_cp_layout='contiguous'.")
    vp_size = getattr(model_cfg, "virtual_pipeline_model_parallel_size", None)
    is_first = is_pp_first_stage(pg_collection.pp) and (
        vp_stage is None or is_vp_first_stage(vp_stage=vp_stage, vp_size=vp_size)
    )
    is_last = is_pp_last_stage(pg_collection.pp) and (
        vp_stage is None or is_vp_last_stage(vp_stage=vp_stage, vp_size=vp_size)
    )
    is_middle = (not is_first) and (not is_last)
    # Packed datasets already request full fields. With CP>1, also read a
    # non-packed batch on middle stages so every stage rejects it consistently.
    include_full_batch_fields = is_middle and (cp_size > 1 or _middle_pp_stage_needs_batch(cfg))
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

    has_packed = _has_packed_sequence_metadata(batch)
    if cp_size > 1:
        if not has_packed:
            raise ValueError("DSv4 context parallelism requires packed THD metadata; use a packed dataset.")
        batch = _partition_packed_batch_contiguous(batch, cp_size, pg_collection.cp.rank())
    elif not has_packed:
        batch = get_batch_on_this_cp_rank(batch, dynamic_cp=False, cp_group=pg_collection.cp)

    from megatron.bridge.perf_recipes.deepseek.gb300._deepseek_v4_compat import FixedTHDMockDatasetConfig

    metadata = _packed_metadata_for_forward(
        batch, include_padding_mask=isinstance(cfg.dataset, FixedTHDMockDatasetConfig)
    )
    if cp_size > 1 and "cp_partition_mode" in {field.name for field in fields(PackedSeqParams)}:
        # Development MCore still consumes this field; never let a contiguous
        # native batch silently use its legacy zigzag default.
        metadata["cp_partition_mode"] = "contiguous"

    return (
        batch["tokens"],
        batch["labels"],
        batch["loss_mask"],
        batch.get("attention_mask"),
        batch["position_ids"],
        metadata,
    )


def forward_step(  # pragma: no cover
    state: GlobalState, data_iterator: Iterable, model: GPTModel, return_schedule_plan: bool = False
):
    """Forward training step for DSv4 with contiguous CP partition support."""
    output, loss_mask = _forward_step_common(
        state,
        data_iterator,
        model,
        return_schedule_plan,
        _get_batch_fn=get_batch,
        _get_packed_seq_params_fn=_get_dsv4_packed_seq_params,
    )
    return output, _create_loss_function(
        loss_mask,
        check_for_nan_in_loss=state.cfg.rerun_state_machine.check_for_nan_in_loss,
        check_for_spiky_loss=state.cfg.rerun_state_machine.check_for_spiky_loss,
    )
