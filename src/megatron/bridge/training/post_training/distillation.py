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

from dataclasses import dataclass, field
from typing import Callable

import modelopt.torch.distill as mtd
import torch
from megatron.core import parallel_state
from megatron.core.transformer import MegatronModule


@dataclass
class ModelOptDistillConfig:
    """Configuration settings for Model Optimizer distillation."""

    logit_layers: tuple[str, str] = ("output_layer", "output_layer")
    """Layer names to use for logit distillation."""

    intermediate_layer_pairs: list[tuple[str, ...]] = field(default_factory=list)
    """Layer names to use for intermediate distillation."""

    skip_lm_loss: bool = True
    """Whether to skip the original LM loss computation."""

    kd_loss_scale: float = 1.0
    """Scale for weighing the KD loss, if original LM loss is not skipped."""

    logit_kl_temperature: float = 1.0
    """Temperature for the logit KL divergence."""


def loss_func_kd(
    output_tensor: torch.Tensor, loss_mask: torch.Tensor, original_loss_fn: Callable, model: MegatronModule
):
    """Loss function (with KD Loss support).

    Args:
        output_tensor (Tensor): The tensor with the losses
        loss_mask (Tensor): Used to mask out some portions of the loss
        original_loss_fn (Callable): The original loss function
        model (GPTModel): The model (can be wrapped)
    """
    assert isinstance(model, mtd.DistillationModel), "Model must be a ModelOpt DistillationModel"

    # Standard lm loss
    loss_lm, num_tokens, report = original_loss_fn(output_tensor)

    # Handle knowledge distillation
    losses_kd = model.compute_kd_loss(
        student_loss=loss_lm,
        loss_reduction_fn=lambda x: _mask_loss(x, loss_mask),
    )

    report["total loss"] = torch.cat([losses_kd["kd_loss"].clone().detach().view(1), num_tokens.view(1)])
    report["logits distillation loss"] = torch.cat(
        [losses_kd["logits_loss"].clone().detach().view(1), num_tokens.view(1)]
    )
    report["intermediate distillation loss"] = torch.cat(
        [losses_kd["intermediate_loss"].clone().detach().view(1), num_tokens.view(1)]
    )

    # Validation loss remains unchanged
    if model.training:
        loss = losses_kd["kd_loss"]
    else:
        loss = loss_lm

    return loss, num_tokens, report


def _mask_loss(output_tensor: torch.Tensor, loss_mask: torch.Tensor):
    if isinstance(output_tensor, tuple):
        # Special distillation flags indicating whether to perform additional tensor-parallel adjustments.
        output_tensor, tp_reduce, is_sequence_parallel = output_tensor
    else:
        tp_reduce, is_sequence_parallel = False, False
    tp_group = parallel_state.get_tensor_model_parallel_group()

    if is_sequence_parallel:
        # Sequence-parallel tensor derived from intermediate activation - need to split loss mask.
        idx = tp_group.rank()
        loss_mask = torch.tensor_split(loss_mask, tp_group.size(), dim=1)[idx]

    losses = output_tensor.view(-1).float()
    loss_mask = loss_mask.reshape(-1).float()
    loss = torch.sum(losses * loss_mask)

    if tp_reduce or is_sequence_parallel:
        # Losses on parallel tensors require extra all-reduce to sync across MP ranks.
        torch.distributed.all_reduce(loss, group=tp_group)

    return loss
