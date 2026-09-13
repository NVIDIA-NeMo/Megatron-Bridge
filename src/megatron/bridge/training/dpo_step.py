# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""DPO forward step: pair-batch fetch, the model call, and the checked loss closure."""

from collections.abc import Callable, Iterator
from functools import partial
from typing import TYPE_CHECKING

import torch

from megatron.bridge.data.collators.sequence_padding import _ceil_to_multiple
from megatron.bridge.training.dpo import DPOLossConfig, dpo_loss
from megatron.bridge.training.losses import validate_local_loss
from megatron.bridge.training.utils.pg_utils import pipeline_stage_roles


if TYPE_CHECKING:
    from megatron.core.models.gpt import GPTModel

    from megatron.bridge.training.state import GlobalState


# The generic SFT batch fetch keeps only the first four; the loss needs the rest too.
DPO_BATCH_KEYS = (
    "tokens",
    "labels",
    "loss_mask",
    "position_ids",
    "pair_id",
    "loss_multiplier",
    "ref_logprob_sum",
    "ref_num_tokens",
)


def validate_dpo_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Fail loudly if a batch is missing DPO keys (e.g. built without ref logprobs)."""
    missing = [key for key in DPO_BATCH_KEYS if batch.get(key) is None]
    if missing:
        raise ValueError(
            f"DPO batch is missing {missing}. The loader must be built by "
            "build_preference_data_loader over a dataset with ref_logprobs attached "
            "(load_ref_logprobs over the offline scorer's artifact)."
        )
    return batch


_SEQ_DIM_KEYS = ("tokens", "labels", "loss_mask", "position_ids")


def trim_dpo_batch_padding(batch: dict[str, torch.Tensor], pad_to_multiple: int = 1) -> dict[str, torch.Tensor]:
    """Drop all-pad tail columns so the microbatch width matches a standalone collate of its own pairs.

    ``pad_to_multiple`` must equal the collate's ``pad_seq_length_to_mult``.
    """
    live = batch["loss_mask"].any(dim=0)
    if not bool(live.any()):
        return batch  # all-stub microbatch: nothing anchors the width, keep as-is
    width = _ceil_to_multiple(max(int(live.nonzero().max().item()) + 1, 2), pad_to_multiple)
    if width < batch["tokens"].size(1):
        for key in _SEQ_DIM_KEYS:
            batch[key] = batch[key][:, :width].contiguous()
    return batch


def get_dpo_batch(
    data_iterator: Iterator[dict[str, torch.Tensor]], pad_to_multiple: int = 1
) -> dict[str, torch.Tensor]:
    """Fetch one row-interleaved pair batch, move it to device, and trim pad tails."""
    batch = validate_dpo_batch(next(data_iterator))
    return trim_dpo_batch_padding({key: batch[key].cuda(non_blocking=True) for key in DPO_BATCH_KEYS}, pad_to_multiple)


def _loss_not_on_this_stage(output_tensor: torch.Tensor) -> None:
    raise RuntimeError(
        "The DPO loss was requested on a pipeline stage that holds no batch; only the last stage computes it."
    )


def dpo_loss_with_checks(
    loss_config: DPOLossConfig,
    batch: dict[str, torch.Tensor],
    output_tensor: torch.Tensor,
    *,
    check_for_nan_in_loss: bool,
    check_for_spiky_loss: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """``dpo_loss`` followed by the rerun-state-machine loss checks."""
    loss, num_live_pairs, metrics = dpo_loss(loss_config, batch, output_tensor)
    validate_local_loss(loss, check_for_nan_in_loss=check_for_nan_in_loss, check_for_spiky_loss=check_for_spiky_loss)
    return loss, num_live_pairs, metrics


def dpo_forward_step(
    state: "GlobalState",
    data_iterator: Iterator[dict[str, torch.Tensor]],
    model: "GPTModel",
) -> tuple[torch.Tensor, Callable]:
    """DPO forward training step.

    Args:
        state: Global state for the run; ``state.cfg.dpo`` holds the ``DPOLossConfig``.
        data_iterator: Yields preference_collate_fn batches (rows: chosen@even/rejected@odd).
        model: The GPT model chunk.

    Returns:
        The ``[rows, seq]`` per-token NLL tensor (hidden states on a non-final pipeline stage) and the
        loss closure, which the schedule only invokes on the last stage.
    """
    loss_config: DPOLossConfig = state.cfg.dpo
    timers = state.timers
    straggler_timer = state.straggler_timer

    is_first, is_last = pipeline_stage_roles(model)
    if not (is_first or is_last):
        with straggler_timer:
            output_tensor = model(input_ids=None, position_ids=None, attention_mask=None, labels=None)
        return output_tensor, _loss_not_on_this_stage

    timers("batch-generator", log_level=2).start()
    with straggler_timer(bdata=True):
        batch = get_dpo_batch(data_iterator, state.cfg.dataset.pad_seq_length_to_mult)
    timers("batch-generator").stop()

    with straggler_timer:
        output_tensor = model(
            input_ids=batch["tokens"],
            position_ids=batch["position_ids"],
            attention_mask=None,
            labels=batch["labels"],
        )

    return output_tensor, partial(
        dpo_loss_with_checks,
        loss_config,
        batch,
        check_for_nan_in_loss=state.cfg.rerun_state_machine.check_for_nan_in_loss,
        check_for_spiky_loss=state.cfg.rerun_state_machine.check_for_spiky_loss,
    )
