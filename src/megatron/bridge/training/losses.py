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

from functools import partial
from typing import Tuple

import torch
from megatron.core.rerun_state_machine import get_rerun_state_machine


_DEFAULT_SPIKY_LOSS_FACTOR: float = 10.0


def create_masked_next_token_loss_function(
    loss_mask: torch.Tensor, check_for_nan_in_loss: bool, check_for_spiky_loss: bool
) -> partial:
    """Create a partial loss function configured for masked next-token loss.

    This replaces the generic helper previously in utils/loss_utils.py.
    """

    return partial(
        masked_next_token_loss,
        loss_mask,
        check_for_nan_in_loss=check_for_nan_in_loss,
        check_for_spiky_loss=check_for_spiky_loss,
    )


def masked_next_token_loss(
    loss_mask: torch.Tensor,
    output_tensor: torch.Tensor | Tuple[torch.Tensor],
    check_for_nan_in_loss: bool = True,
    check_for_spiky_loss: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    """Loss function.

    Args:
        loss_mask: Used to mask out some portions of the loss
        output_tensor: The tensor with the losses. For LLaVAModel, this is a tuple of (losses, new_loss_mask)
        check_for_nan_in_loss: Whether to check for NaN values in the loss
        check_for_spiky_loss: Whether to check for spiky loss values

    Returns:
        tuple containing:
        - The loss scalar for this micro-batch
        - The number of non-padded tokens in this microbatch
        - A dict containing reporting metrics on the loss and number of tokens across
          the data parallel ranks
    """
    if isinstance(output_tensor, tuple):
        losses = output_tensor[0].view(-1).float()
        loss_mask = output_tensor[1].view(-1).float()
    else:
        losses = output_tensor.view(-1).float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses * loss_mask)

    # Check individual rank losses are not NaN prior to DP all-reduce.
    rerun_state_machine = get_rerun_state_machine()
    if check_for_nan_in_loss:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=True,
        )
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isinf,
            message="found Inf in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=True,
        )
    # Check for spiky loss
    if check_for_spiky_loss:
        spiky_loss_factor = getattr(rerun_state_machine, "spiky_loss_factor", _DEFAULT_SPIKY_LOSS_FACTOR)
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=partial(
                rerun_state_machine.is_unexpectedly_large,
                threshold=spiky_loss_factor,
                context="loss",
            ),
            message="Spiky loss",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=False,
        )

    num_tokens = loss_mask.sum().clone().detach().to(torch.int)
    reporting_loss = torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])

    return (loss, num_tokens, {"lm loss": reporting_loss})
