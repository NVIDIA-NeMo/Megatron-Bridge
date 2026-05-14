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
from collections.abc import Callable
from typing import TYPE_CHECKING, override


if TYPE_CHECKING:
    from typing import overload

import torch
from torch.optim.optimizer import ParamsT

from emerging_optimizers import registry
from emerging_optimizers.mixin import WeightDecayMixin, WeightDecayT
from emerging_optimizers.scalar_optimizers.update_functions import calculate_lion_update


__all__ = [
    "Lion",
]


@registry.register_optimizer("lion")
class Lion(WeightDecayMixin, torch.optim.Optimizer):
    """Lion optimizer (Chen et al., 2023).

    A memory-efficient optimizer that uses only sign updates and tracks a single
    exponential moving average (no second moment), resulting in lower memory usage
    than Adam.

    The update rule below assumes ``weight_decay_method="decoupled"`` (the default).
    See :class:`~emerging_optimizers.mixin.WeightDecayMixin` for the other modes.

    .. math::
        p = p \\cdot (1 - \\text{lr} \\cdot \\lambda) \\\\
        \\text{update} = \\text{sign}(\\beta_1 m_{t-1} + (1 - \\beta_1) g_t) \\\\
        m_t = \\beta_2 m_{t-1} + (1 - \\beta_2) g_t \\\\
        p = p - \\text{lr} \\cdot \\text{update}

    References:
        - Chen, X., Liang, C., Huang, D., Real, E., Wang, K., Liu, Y., Pham, H., Dong, X.,
          Luber, T., Cho, T., Le, Q. V., & Henaff, O. J. *Symbolic Discovery of Optimization Algorithms.*
          arXiv:2302.06675 (2023).
          [`arXiv:2302.06675 <https://arxiv.org/abs/2302.06675>`_]

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups.
        lr: Learning rate.
        betas: Coefficients (beta1, beta2) for computing the update and running average.
            beta1 is used for the sign update interpolation, beta2 for the momentum EMA update.
        weight_decay: Weight decay coefficient.
        weight_decay_method: Method to apply weight decay, see
            :class:`~emerging_optimizers.mixin.WeightDecayMixin` for more details.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.01,
        weight_decay_method: WeightDecayT = "decoupled",
    ) -> None:
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        self.weight_decay_method = weight_decay_method
        super().__init__(params, defaults)

    @torch.no_grad()
    def _init_group(
        self,
        group: dict,
        skip_non_grad_params: bool = True,
    ) -> None:
        """Performs lazy state initialization for parameters.

        Args:
            group: Parameter group dictionary.
            skip_non_grad_params: If True, skip parameters without gradients.
        """
        for p in group["params"]:
            if skip_non_grad_params and p.grad is None:
                continue
            state = self.state[p]

            if len(state) == 0:
                state["exp_avg"] = torch.zeros_like(p.data)

    if TYPE_CHECKING:

        @overload
        def step(self, closure: None = ...) -> None: ...

        @overload
        def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()  # type: ignore[misc]
    @override
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Perform a single optimization step.

        Note:
            When ``weight_decay_method="l2"``, ``p.grad`` is modified in-place
            (the L2 penalty ``weight_decay * p`` is added to the gradient).
            If you need the original gradient after this call, clone it beforehand.

        Args:
            closure: A closure that reevaluates the model and returns the loss (optional).

        Returns:
            The loss from the closure, if provided.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self._init_group(group)

            lr = group["lr"]
            betas = group["betas"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                exp_avg = self.state[p]["exp_avg"]

                # Weight decay
                self._apply_weight_decay_inplace(p.data, grad, lr, weight_decay)

                # Lion update: sign(beta1 * m + (1-beta1) * g)
                # Note: different betas per param-group will each trigger a one-time
                # torch.compile recompilation of calculate_lion_update.
                update = calculate_lion_update(grad, exp_avg, betas)
                p.data.add_(update, alpha=-lr)

        return loss
