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

from __future__ import annotations

from typing import Iterable

import torch
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage

from megatron.bridge.training.config import ConfigContainer, FinetuningDatasetConfig


def _to_cuda(val):
    """Move *val* to CUDA, recursing into objects that hold tensor attributes.

    This generalises the inline ``visual_inputs`` handling found in
    ``vlm_step.get_batch_from_iterator`` and ``qwen3_vl_step.get_batch_from_iterator``,
    which iterate over ``val.__dict__`` and call ``.cuda()`` on each tensor.
    Unlike those, this helper recurses to arbitrary depth and is not
    tied to a specific key name.
    """
    if isinstance(val, torch.Tensor):
        return val.cuda(non_blocking=True)
    if val is None:
        return val
    for attr, v in getattr(val, "__dict__", {}).items():
        val.__dict__[attr] = _to_cuda(v)
    return val


def get_batch_on_this_tp_rank(
    data_iterator: Iterable,
    cfg: ConfigContainer,
    use_mtp: bool = False,
    *,
    pg_collection,
    broadcast_all_keys: bool = False,
) -> dict[str, torch.Tensor]:
    """Load a batch on TP-rank 0 and broadcast to other TP ranks.

    When ``dataset.broadcast_data_across_tp`` is enabled, only TP-rank 0
    reads from the data iterator.  The batch is then broadcast to the
    remaining TP ranks, eliminating redundant I/O -- critical for
    storage backends with high contention.

    Broadcasting is split into two tiers for efficiency:

    1. **Standard keys** (``tokens``, ``labels``, ``loss_mask``,
       ``attention_mask``, ``position_ids``) have fixed, pre-known shapes.
       They are broadcast via ``torch.distributed.broadcast`` (fast,
       zero-copy on the receiver).
    2. **Extra keys** (everything else -- e.g. ``cu_seqlens``,
       ``visual_inputs``) have shapes or types unknown at allocation time.
       They are broadcast via ``torch.distributed.broadcast_object_list``
       (slower, involves pickling).  A boolean flag is broadcast first so
       the heavy path is skipped entirely when there are no extras.

    Args:
        data_iterator: Yields ``dict[str, Tensor]`` batches.  Only
            consumed on TP-rank 0.
        cfg: Run configuration (provides shapes, PP size, etc.).
        use_mtp: Whether Multi-Token Prediction layers are enabled.
            When ``True``, tokens and position_ids are also broadcast
            on the last PP stage.
        pg_collection: Process-group collection with ``.tp`` and ``.pp``
            groups.
        broadcast_all_keys: When ``True``, broadcast every standard key
            regardless of PP stage.  Required by VLM recipes where all
            PP stages need the full batch (e.g. for MRoPE).

    Returns:
        Batch dict with all tensors on CUDA.  Keys that are not relevant
        to the current PP stage may be ``None``.

    Note:
        If the dataset supplies ``input_ids`` instead of ``tokens``
        (HuggingFace convention), it is aliased to ``tokens`` for the
        broadcast.  The original ``input_ids`` key is preserved as an
        extra key so downstream code that looks it up still works.
    """

    def _broadcast(item):
        if item is not None:
            # Broadcast from TP group's rank-0 within that group
            tp_group = pg_collection.tp
            src_global_rank = torch.distributed.get_process_group_ranks(tp_group)[0]
            torch.distributed.broadcast(item, src_global_rank, group=tp_group)

    # Determine if this rank is TP rank 0 using pg_collection.tp
    tp_group = pg_collection.tp
    tp_ranks = torch.distributed.get_process_group_ranks(tp_group)
    is_tp_rank0 = torch.distributed.get_rank() == tp_ranks[0]

    if is_tp_rank0:
        if data_iterator is not None:
            data = next(data_iterator)
        else:
            data = None

        # VLM/LLaVA datasets may supply ``input_ids`` instead of ``tokens``.
        # Alias to ``tokens`` so the fixed-shape broadcast path below works
        # for every recipe.  The original key is kept so callers that
        # look up ``input_ids`` still find it.
        if "tokens" not in data and "input_ids" in data:
            data["tokens"] = data["input_ids"]

        batch = {
            "tokens": data["tokens"].cuda(non_blocking=True),
            "labels": data["labels"].cuda(non_blocking=True),
            "loss_mask": data["loss_mask"].cuda(non_blocking=True),
            "attention_mask": None if "attention_mask" not in data else data["attention_mask"].cuda(non_blocking=True),
            "position_ids": data["position_ids"].cuda(non_blocking=True),
        }

        if cfg.model.pipeline_model_parallel_size == 1 or broadcast_all_keys:
            _broadcast(batch["tokens"])
            _broadcast(batch["labels"])
            _broadcast(batch["loss_mask"])
            _broadcast(batch["attention_mask"])
            _broadcast(batch["position_ids"])

        elif is_pp_first_stage(pg_collection.pp):
            _broadcast(batch["tokens"])
            _broadcast(batch["attention_mask"])
            _broadcast(batch["position_ids"])

        elif is_pp_last_stage(pg_collection.pp):
            if use_mtp:
                _broadcast(batch["tokens"])
                _broadcast(batch["position_ids"])
            _broadcast(batch["labels"])
            _broadcast(batch["loss_mask"])
            _broadcast(batch["attention_mask"])

    else:
        mbs = cfg.train.micro_batch_size
        seq_length = cfg.model.seq_length
        tokens = torch.empty(
            (mbs, seq_length),
            dtype=torch.int64,
            device=torch.cuda.current_device(),
        )
        labels = torch.empty(
            (mbs, seq_length),
            dtype=torch.int64,
            device=torch.cuda.current_device(),
        )
        loss_mask = torch.empty(
            (mbs, seq_length),
            dtype=torch.float32,
            device=torch.cuda.current_device(),
        )
        if isinstance(cfg.dataset, FinetuningDatasetConfig) or cfg.dataset.create_attention_mask:
            attention_mask = torch.empty(
                (
                    mbs,
                    1,
                    seq_length,
                    seq_length,
                ),
                dtype=torch.bool,
                device=torch.cuda.current_device(),
            )
        else:
            attention_mask = None
        position_ids = torch.empty(
            (mbs, seq_length),
            dtype=torch.int64,
            device=torch.cuda.current_device(),
        )

        if cfg.model.pipeline_model_parallel_size == 1 or broadcast_all_keys:
            _broadcast(tokens)
            _broadcast(labels)
            _broadcast(loss_mask)
            _broadcast(attention_mask)
            _broadcast(position_ids)

        elif is_pp_first_stage(pg_collection.pp):
            labels = None
            loss_mask = None

            _broadcast(tokens)
            _broadcast(attention_mask)
            _broadcast(position_ids)

        elif is_pp_last_stage(pg_collection.pp):
            if use_mtp:
                _broadcast(tokens)
                _broadcast(position_ids)
            else:
                tokens = None
                position_ids = None

            _broadcast(labels)
            _broadcast(loss_mask)
            _broadcast(attention_mask)

        batch = {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }

    # Broadcast any extra keys (e.g. packed-sequence metadata) that are not
    # covered by the fixed-shape direct broadcasts above.  A lightweight
    # boolean flag is broadcast first so the heavy broadcast_object_list call
    # is skipped entirely when there are no extra keys (common case).
    has_extra = [bool({k for k in data if k not in batch}) if is_tp_rank0 else False]
    torch.distributed.broadcast_object_list(has_extra, src=tp_ranks[0], group=tp_group)
    if has_extra[0]:
        extra = [{k: v for k, v in data.items() if k not in batch} if is_tp_rank0 else None]
        torch.distributed.broadcast_object_list(extra, src=tp_ranks[0], group=tp_group)
        for key, val in extra[0].items():
            batch[key] = _to_cuda(val)

    return batch
