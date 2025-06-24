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
from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.utils import get_batch_on_this_cp_rank

from megatron.hub.training.config import ConfigContainer, FinetuningDatasetConfig
from megatron.hub.training.losses import masked_next_token_loss
from megatron.hub.training.state import GlobalState


logger = logging.getLogger(__name__)


def get_batch_from_iterator(data_iterator: Iterable) -> dict[str, torch.Tensor]:
    """Get a batch of data from the iterator.

    Args:
        data_iterator: The data iterator to get the batch from.

    Returns:
        dict[str, torch.Tensor]: A dictionary containing the batch data.
    """
    assert data_iterator is not None, "data_iterator must not be None"

    data = next(data_iterator)

    batch = {
        "tokens": data["tokens"].cuda(non_blocking=True),
        "labels": data["labels"].cuda(non_blocking=True),
        "loss_mask": data["loss_mask"].cuda(non_blocking=True),
        "attention_mask": None if "attention_mask" not in data else data["attention_mask"].cuda(non_blocking=True),
        "position_ids": data["position_ids"].cuda(non_blocking=True),
    }

    return batch


def get_batch_on_this_tp_rank(data_iterator: Iterable, cfg: ConfigContainer) -> dict[str, torch.Tensor]:
    """Get a batch from the data iterator, handling TP broadcasting.

    On TP rank 0, it fetches the next batch from the iterator and broadcasts
    the necessary tensors to other TP ranks based on the pipeline stage.
    On other TP ranks, it allocates tensors and receives the broadcasted data.

    Args:
        data_iterator: The data iterator.
        cfg: The configuration container.

    Returns:
        A dictionary containing the batch data for the current rank.
    """

    def _broadcast(item):
        if item is not None:
            torch.distributed.broadcast(
                item,
                parallel_state.get_tensor_model_parallel_src_rank(),
                group=parallel_state.get_tensor_model_parallel_group(),
            )

    if parallel_state.get_tensor_model_parallel_rank() == 0:
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

        elif parallel_state.is_pipeline_first_stage():
            _broadcast(batch["tokens"])
            _broadcast(batch["attention_mask"])
            _broadcast(batch["position_ids"])

        elif parallel_state.is_pipeline_last_stage():
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

        elif parallel_state.is_pipeline_first_stage():
            labels = None
            loss_mask = None

            _broadcast(tokens)
            _broadcast(attention_mask)
            _broadcast(position_ids)

        elif parallel_state.is_pipeline_last_stage():
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


def get_batch(
    data_iterator: Iterable, cfg: ConfigContainer
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Generate a batch.

    Args:
        data_iterator: Input data iterator
        cfg: Configuration container

    Returns:
        tuple of tensors containing tokens, labels, loss_mask, attention_mask, and position_ids
    """
    if (not parallel_state.is_pipeline_first_stage()) and (not parallel_state.is_pipeline_last_stage()):
        return None, None, None, None, None

    if isinstance(cfg.dataset, FinetuningDatasetConfig):
        batch = get_batch_from_iterator(data_iterator)
    else:
        # get batches based on the TP rank you are on
        batch = get_batch_on_this_tp_rank(data_iterator, cfg)

    # slice batch along sequence dimension for context parallelism
    batch = get_batch_on_this_cp_rank(batch)

    return batch.values()


def forward_step(state: GlobalState, data_iterator: Iterable, model: GPTModel) -> tuple[torch.Tensor, partial]:
    """Forward training step.

    Args:
        state: Global state for the run
        data_iterator: Input data iterator
        model: The GPT Model

    Returns:
        tuple containing the output tensor and the loss function
    """
    timers = state.timers
    straggler_timer = state.straggler_timer

    timers("batch-generator", log_level=2).start()
    with straggler_timer(bdata=True):
        tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator, state.cfg)
    timers("batch-generator").stop()

    forward_args = {
        "input_ids": tokens,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

    with straggler_timer:
        output_tensor = model(**forward_args)

    return output_tensor, partial(masked_next_token_loss, loss_mask)
