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

from collections.abc import Iterable

import torch
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage

from megatron.bridge.training.config import ConfigContainer, FinetuningDatasetConfig


def get_batch_on_this_tp_rank(
    data_iterator: Iterable, cfg: ConfigContainer, use_mtp: bool = False, *, pg_collection
) -> dict[str, torch.Tensor]:
    """Get a batch from the data iterator, handling TP broadcasting.

    This is a generic helper used by multiple recipes. The implementation is
    identical to the prior one in `gpt_step.py`.
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

        batch = {
            "tokens": data["tokens"].cuda(non_blocking=True),
            "labels": data["labels"].cuda(non_blocking=True),
            "loss_mask": data["loss_mask"].cuda(non_blocking=True),
            "attention_mask": None if "attention_mask" not in data else data["attention_mask"].cuda(non_blocking=True),
            "position_ids": data["position_ids"].cuda(non_blocking=True),
        }

        if cfg.model.pipeline_model_parallel_size == 1:
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

        if cfg.model.pipeline_model_parallel_size == 1:
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

    return batch
