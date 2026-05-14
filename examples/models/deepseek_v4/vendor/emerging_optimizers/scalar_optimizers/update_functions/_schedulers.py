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
import math


def _half_life_steps(beta: float, eps: float = 1e-8) -> float:
    """Function that maps beta to the number of steps to reach 0.5.

    Equation:
        f(beta) = log(0.5) / log(beta + eps) - 1

    Args:
        beta: The beta parameter.
        eps: A small constant to avoid division by zero.

    Returns:
        The number of steps to reach 0.5.
    """
    return math.log(0.5) / math.log(beta + eps) - 1


def _inverse_half_life_beta(t: float) -> float:
    """Maps number of steps to reach 0.5 to beta.

    Equation:
        f_inv(t) = 0.5^(1 / (t + 1))

    Args:
        t: The number of steps to reach 0.5.

    Returns:
        The beta parameter.
    """
    return math.pow(0.5, 1 / (t + 1))


def _linear_half_life_warmup_scheduler(
    step: int, beta_end: float, beta_start: float = 0, num_warmup_steps: int = 1
) -> float:
    """Half-life linear warmup scheduler for the beta parameter.

    Equation:
        beta = f_inv((1 - step / num_warmup_steps) * f(beta_start) + (step / num_warmup_steps) * f(beta_end))


    Args:
        step: The current step of the optimizer.
        beta_end: The final value of the beta parameter.
        beta_start: The initial value of the beta parameter.
        num_warmup_steps: The number of warmup steps.

    Returns:
        The value of the beta parameter at the current step.
    """

    if step < num_warmup_steps:
        a = step / float(num_warmup_steps)
        return _inverse_half_life_beta((1.0 - a) * _half_life_steps(beta_start) + a * _half_life_steps(beta_end))
    return beta_end


def _linear_warmup_scheduler(step: int, alpha_end: float, alpha_start: float = 0, num_warmup_steps: int = 1) -> float:
    """Linear warmup scheduler for the alpha parameter.

    Equation:
        alpha = (1 - step / num_warmup_steps) * alpha_start + (step / num_warmup_steps) * alpha_end

    Args:
        step: The current step of the optimizer.
        alpha_end: The final value of the alpha parameter.
        alpha_start: The initial value of the alpha parameter.
        num_warmup_steps: The number of warmup steps.

    Returns:
        The value of the alpha parameter at the current step.
    """
    if step < num_warmup_steps:
        a = step / float(num_warmup_steps)
        return (1.0 - a) * alpha_start + a * alpha_end
    return alpha_end
