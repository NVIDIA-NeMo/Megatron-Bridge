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
    "calculate_lion_update",
]


@torch.compile  # type: ignore[misc]
@torch.no_grad()  # type: ignore[misc]
def calculate_lion_update(
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    betas: tuple[float, float],
) -> torch.Tensor:
    """Performs the Lion update.

    This function performs the computation of 1 step of Lion update.

    The update rule is as follows:

    .. math::
        \\text{update} = \\text{sign}(\\beta_1 m_{t-1} + (1 - \\beta_1) g_t) \\\\
        m_t = \\beta_2 m_{t-1} + (1 - \\beta_2) g_t

    Args:
        grad: The gradient tensor.
        exp_avg: The accumulated first moment of the gradient.
        betas: The EMA beta coefficients (beta1, beta2) for the Lion update.

    Returns:
        The Lion update.
    """

    beta1, beta2 = betas

    # Compute update using interpolation (Lion's beta1)
    update_momentum = exp_avg.lerp(grad, 1 - beta1)

    # Update the momentum state (Lion's beta2)
    exp_avg.lerp_(grad, 1 - beta2)

    # Return signed update (no shape scaling for Lion)
    return torch.sign(update_momentum)
