# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch


__all__ = [
    "calculate_adam_update",
]


@torch.compile  # type: ignore[misc]
@torch.no_grad()  # type: ignore[misc]
def calculate_adam_update(
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    betas: tuple[float, float],
    correct_bias: bool,
    nesterov: bool,
    step: int,
    eps: float,
) -> torch.Tensor:
    """Performs the Adam update.

    This function performs the computation of 1 step of Adam.

    The update rule is as follows:

    .. math::
        m_t = \\beta_1 m_{t-1} + (1 - \\beta_1) g_t \\\\
        v_t = \\beta_2 v_{t-1} + (1 - \\beta_2) g_t^2 \\\\
        \\hat{m}_t = \\frac{m_t}{1 - \\beta_1^t} \\\\
        \\hat{v}_t = \\frac{v_t}{1 - \\beta_2^t} \\\\
        \\text{update} = \\frac{\\hat{m}_t}{\\sqrt{\\hat{v}_t} + \\epsilon} \\\\

    Args:
        grad: The gradient tensor.
        exp_avg: The accumulated first moment of the gradient.
        exp_avg_sq: The accumulated second moment of the gradient.
        betas: The EMA beta coefficients for the Adam update.
        correct_bias: Whether to correct the bias of the Adam update.
        nesterov: Whether to use nesterov momentum.
        step: The current step of the optimizer, used to compute the bias correction terms.
        eps: The epsilon for the Adam second moment update.

    Returns:
        The Adam-update.
    """

    beta1, beta2 = betas

    # Decay the first and second moment running average coefficient
    exp_avg.lerp_(grad, 1 - beta1)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2)

    # step size correction for optimizer states EMA
    bias_correction1 = 1.0
    bias_correction2 = 1.0
    if correct_bias:
        # step size correction for ADAM moments EMA
        bias_correction1 = 1.0 - beta1 ** (step)
        bias_correction2 = 1.0 - beta2 ** (step)

    if nesterov:
        # Apply nesterov momentum correction, optionally with bias correction
        bias_correction_nesterov = (1 - beta1 ** (step + 1)) if correct_bias else 1.0
        momentum = beta1 * exp_avg / bias_correction_nesterov + (1 - beta1) * grad / bias_correction1
    else:
        # Use standard momentum, optionally with bias correction
        momentum = exp_avg / bias_correction1

    # construct the denominator of the inner ADAM optimizer
    adam_second_moment = exp_avg_sq / bias_correction2
    adam_second_moment = adam_second_moment.sqrt() + eps
    return momentum / adam_second_moment
