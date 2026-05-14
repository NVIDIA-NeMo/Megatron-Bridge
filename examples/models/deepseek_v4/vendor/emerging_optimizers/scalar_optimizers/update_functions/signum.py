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
    "calculate_signum_update",
]


@torch.compile  # type: ignore[misc]
@torch.no_grad()  # type: ignore[misc]
def calculate_signum_update(
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    momentum: float,
    correct_bias: bool,
    nesterov: bool,
    step: int,
    use_shape_scaling: bool = False,
) -> torch.Tensor:
    """Performs the sign-SGD or Signum update.

    This function performs the computation of 1 step of sign-SGD or Signum.
    Based on https://arxiv.org/abs/1802.04434.
    When using signSGD with shape scaling, general recommendation is to
    scale :math:`lr = \\text{adam lr} \\cdot \\text{network width} \\cdot \\frac{2}{\\text{rows} + \\text{cols}}`.
    This is for learning rate transfer with width scaling (https://arxiv.org/abs/2506.07254v1).

    The update rule is as follows:

    .. math::
        m_t = \\beta m_{t-1} + (1 - \\beta) g_t \\\\
        \\hat{m}_t = \\frac{m_t}{1 - \\beta^t} \\\\
        \\text{update} = \\text{sign}(\\hat{m}_t)

    Args:
        grad: The gradient tensor.
        exp_avg: The accumulated first moment of the gradient.
        momentum: The EMA beta coefficients for the momentum update.
        correct_bias: Whether to correct the bias of the momentum update.
        nesterov: Whether to use nesterov momentum.
        step: The current step of the optimizer, used to compute the bias correction terms.
        use_shape_scaling: Whether to scale the update by the shape of the tensor.

    Returns:
        The sign-SGD/Signum update.
    """

    # Standard SignSGD: update momentum first, then compute signed update
    # Decay the momentum with exponential moving average
    exp_avg.lerp_(grad, 1 - momentum)

    if correct_bias:
        bias_correction1 = 1 - momentum**step
    else:
        bias_correction1 = 1

    if nesterov:
        # Apply nesterov momentum correction, optionally with bias correction
        bias_correction_nesterov = (1 - momentum ** (step + 1)) if correct_bias else 1.0
        new_momentum = momentum * exp_avg / bias_correction_nesterov + (1 - momentum) * grad / bias_correction1
    else:
        # Use standard momentum, optionally with bias correction
        new_momentum = exp_avg / bias_correction1

    # scale update by shape of tensor to ensure consistent update size: https://arxiv.org/abs/2506.07254
    if use_shape_scaling:
        m, n = grad.shape
        return torch.sign(new_momentum) * (2 / (m + n))
    else:
        return torch.sign(new_momentum)
