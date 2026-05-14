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
from typing import TYPE_CHECKING, Callable, override


if TYPE_CHECKING:
    from typing import overload

import torch
from torch.optim.optimizer import Optimizer

from emerging_optimizers import registry


__all__ = ["ObliqueSGD", "ObliqueAdam"]


@registry.register_optimizer("oblique_sgd")
class ObliqueSGD(Optimizer):
    """SGD optimizer for row- or column-normalized 2D parameters on oblique manifolds.

    This optimizer performs SGD on oblique manifolds, where parameters are constrained
    to have unit-norm rows or columns. It implements Riemannian SGD with manifold-aware
    gradient updates and retraction operations.

    References:
        - An Introduction to Optimization on Smooth Manifolds (Nicolas Boumal)
        - EDM2: https://arxiv.org/abs/2312.02696
        - Jianlin Su: https://kexue.fm/archives/11196
        - Raman et al.: https://arxiv.org/abs/1909.06463
        - Franz Cesista: https://leloykun.github.io/ponder/steepest-descent-stiefel/#6-bonus-a-muon-like-optimizer-for-the-embedding-and-unembedding-layers

    Args:
        lr: learning rate
        momentum: momentum coefficient
        weight_decay: weight decay coefficient
        dim: The dimension to normalize over
        eps: epsilon for numerical stability
    """

    def __init__(
        self,
        params: list[torch.nn.Parameter],
        lr: float = 1e-3,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        dim: int = 0,
        eps: float = 1e-8,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            dim=dim,
            eps=eps,
        )
        super().__init__(params, defaults)

    if TYPE_CHECKING:

        @overload
        def step(self, closure: None = ...) -> None: ...

        @overload
        def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()  # type: ignore[misc]
    @override
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Performs a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            lr = group["lr"]
            mom = group["momentum"]
            wd = group["weight_decay"]
            dim = group["dim"]
            eps = group["eps"]

            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.ndim != 2:
                    raise ValueError("ObliqueSGD only supports 2D parameters")
                grad = param.grad

                # Initialize momentum buffer if needed
                state = self.state[param]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(param)

                buf = state["momentum_buffer"]

                # theory style momentum
                buf = torch.add(grad, buf, alpha=mom)

                # Apply Riemannian gradient update
                _compute_riemannian_grad_and_update(param, buf, dim, lr, wd)

                # Retraction back to the manifold, the hyper-sphere
                torch.nn.functional.normalize(param, p=2.0, dim=dim, eps=eps, out=param)

        return loss


@registry.register_optimizer("oblique_adam")
class ObliqueAdam(Optimizer):
    """Adam optimizer for row- or column-normalized 2D parameters on oblique manifolds.

    This optimizer adapts an Adam-like algorithm to work on oblique manifolds, where
    parameters are constrained to have unit-norm rows or columns. It combines
    adaptive momentum estimation with Riemannian gradient computation and manifold retraction.
    """

    def __init__(
        self,
        params: list[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        dim: int = 0,
        eps: float = 1e-8,
        correct_bias: bool = True,
    ) -> None:
        """An Adam-like optimizer for Normalized 2d Parameters

        Args:
            lr: The learning rate.
            betas: The coefficients used for computing running averages of gradient and its square.
            weight_decay: The weight decay coefficient.
            dim: The dimension to normalize over.
            eps: The epsilon for numerical stability.
            correct_bias: Whether to correct bias in Adam-like computation.
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if betas[0] < 0.0 or betas[0] >= 1.0:
            raise ValueError(f"Invalid beta1 value: {betas[0]}")
        if betas[1] < 0.0 or betas[1] >= 1.0:
            raise ValueError(f"Invalid beta2 value: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            dim=dim,
            eps=eps,
            correct_bias=correct_bias,
        )
        super().__init__(params, defaults)

    if TYPE_CHECKING:

        @overload
        def step(self, closure: None = ...) -> None: ...

        @overload
        def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()  # type: ignore[misc]
    @override
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Performs a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            wd = group["weight_decay"]
            dim = group["dim"]
            eps = group["eps"]
            correct_bias = group["correct_bias"]

            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.ndim != 2:
                    raise ValueError("ObliqueAdam only supports 2D parameters")

                state = self.state[param]
                if "step" not in state:
                    state["step"] = 0

                grad = param.grad

                # Initialize momentum buffer if needed
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(param)
                if "exp_avg_sq" not in state:
                    state["exp_avg_sq"] = torch.zeros_like(param)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                # Increment step counter
                state["step"] += 1
                step = state["step"]

                # Update biased first and second moment estimates
                exp_avg.mul_(betas[0]).add_(grad, alpha=1 - betas[0])
                exp_avg_sq.mul_(betas[1]).addcmul_(grad, grad, value=1 - betas[1])

                if correct_bias:
                    # step size correction for ADAM moments EMA
                    bias_correction1 = 1.0 - betas[0] ** step
                    bias_correction2 = 1.0 - betas[1] ** step
                else:
                    bias_correction1 = 1.0
                    bias_correction2 = 1.0

                norm_grad = (exp_avg / bias_correction1) / (exp_avg_sq.sqrt() / bias_correction2 + eps)

                # Apply Riemannian gradient update
                _compute_riemannian_grad_and_update(param, norm_grad, dim, lr, wd)

                # Retraction back to the manifold, i.e. the hyper-sphere
                torch.nn.functional.normalize(param, p=2.0, dim=dim, eps=eps, out=param)

        return loss


def _compute_riemannian_grad_and_update(
    param: torch.Tensor, grad_like: torch.Tensor, dim: int, lr: float, wd: float
) -> None:
    """Compute Riemannian gradient for oblique manifold and update parameter in-place.

    Args:
        param: Parameter tensor (2D)
        grad_like: Gradient-like tensor (momentum buffer or normalized gradient)
        dim: The dimension to normalize over
        lr: Learning rate
        wd: Weight decay coefficient
    """

    inner = (param * grad_like).sum(dim=dim, keepdim=True)
    riem_grad = torch.add(grad_like, param * inner, alpha=-1)

    # Add decoupled weight decay
    param.mul_(1 - lr * wd)

    # Apply update in-place
    param.add_(riem_grad, alpha=-lr)
