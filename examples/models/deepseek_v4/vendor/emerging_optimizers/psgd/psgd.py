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
from typing import TYPE_CHECKING, Callable, override


if TYPE_CHECKING:
    from typing import overload

import torch
from torch.optim.optimizer import ParamsT

from emerging_optimizers import mixin as opt_mixin
from emerging_optimizers import registry
from emerging_optimizers.psgd import psgd_kron_contractions, psgd_utils
from emerging_optimizers.psgd.procrustes_step import procrustes_step
from emerging_optimizers.soap.soap import _clip_update_rms_in_place


__all__ = [
    "PSGDPro",
]


@registry.register_optimizer("psgd_pro")
class PSGDPro(opt_mixin.WeightDecayMixin, torch.optim.Optimizer):
    """Implements a variant of the PSGD optimization algorithm (PSGD-Kron-Whiten with Procrustes step for preconditioner update).

    Preconditioned Stochastic Gradient Descent (PSGD) (https://arxiv.org/abs/1512.04202) is a preconditioned optimization algorithm
    that fits amplitudes of perturbations of preconditioned stochastic gradient to match that of the perturbations of parameters.
    PSGD with Kronecker-factored Preconditioner (PSGD-Kron-Whiten) is a variant of PSGD that reduces memory and computational complexity.
    Procrustes step is an algorithm to update the preconditioner which respects a particular geometry: Q^0.5 * E * Q^1.5, see Stochastic Hessian
    Fittings with Lie Groups (https://arxiv.org/abs/2402.11858) for more details.

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        lr: The learning rate to use
        weight_decay: Weight decay coefficient
        weight_decay_method: Method to apply weight decay, see :class:`~emerging_optimizers.mixin.WeightDecayMixin`
            for more details.
        momentum: Momentum coefficient for exponential moving average of gradient.
        beta_lip: EMA beta for the Lipschitz constants.
        precond_lr: Inner learning rate for the preconditioner.
        precond_init_scale: scale of initial preconditioner values.
        min_precond_lr: Minimum learning rate for preconditioner learning rate schedule.
        warmup_steps: Warmup steps for preconditioner learning rate schedule.
        damping_noise_scale: scale of dampening noise added to gradients.
        max_update_rms: Clip the update RMS to this value (0 means no clipping).
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 3e-3,
        weight_decay: float = 0.01,
        momentum: float = 0.9,
        *,
        weight_decay_method: opt_mixin.WeightDecayT = "decoupled",
        beta_lip: float = 0.9,
        precond_lr: float = 0.1,
        precond_init_scale: float = 1.0,
        damping_noise_scale: float = 0.1,
        min_precond_lr: float = 0.01,
        warmup_steps: int = 10000,
        max_update_rms: float = 0.0,
    ) -> None:
        self.weight_decay_method = weight_decay_method
        self.max_update_rms = max_update_rms
        self.precond_init_scale = precond_init_scale
        self.damping_noise_scale = damping_noise_scale
        self.warmup_steps = warmup_steps
        defaults = {
            "lr": lr,
            "beta_lip": beta_lip,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "precond_lr": precond_lr,
            "min_precond_lr": min_precond_lr,
        }
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
        if closure is None:
            loss = None
        else:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                # Optimizer state initialization
                if "step" not in state:
                    state["step"] = 0
                # Momentum buffer
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(grad)
                # PSGD kronecker factor matrices and Lipschitz constants initialization
                if "Q" not in state or "L" not in state:
                    state["Q"], state["L"] = _init_psgd_kron_states(
                        grad,
                        precond_init_scale=self.precond_init_scale,
                    )

                self._apply_weight_decay_inplace(
                    p,
                    grad,
                    group["lr"],
                    group["weight_decay"],
                )

                # update momentum buffer with EMA of gradient
                exp_avg = state["exp_avg"]
                exp_avg.lerp_(grad, 1 - group["momentum"])

                # Get hyperparameters for preconditioner update
                precond_lr = _get_precond_lr(
                    group["precond_lr"], state["step"], group["min_precond_lr"], self.warmup_steps
                )

                beta_lip = group["beta_lip"]
                # Preconditioner update
                state["Q"], state["L"] = _update_precond_procrustes(
                    state["Q"], state["L"], exp_avg, self.damping_noise_scale, precond_lr, beta_lip
                )
                psgd_utils.uniformize_q_in_place(state["Q"])

                # Get weight update by preconditioning the momentum
                update = psgd_kron_contractions.apply_preconditioner(state["Q"], exp_avg)
                _clip_update_rms_in_place(update, self.max_update_rms)

                # Apply weight update
                p.add_(update, alpha=-group["lr"])

        return loss


def _init_psgd_kron_states(
    grad: torch.Tensor,
    precond_init_scale: float = 1.0,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Initialize the Kronecker factor matrices and Lipschitz constants.

    Args:
        grad: Gradient tensor.
        precond_init_scale: Scale of preconditioner initialization.

    Returns:
        q_list: List of Kronecker factors.
        lip_const_list: List of Lipschitz constants for the Kronecker factors.
    """
    q_list: list[torch.Tensor] = []
    lip_const_list: list[torch.Tensor] = []

    # Create identity matrices scaled by precond_init_scale for each dimension
    for size in grad.shape:
        q_list.append(torch.eye(size, device=grad.device) * precond_init_scale)
        lip_const_list.append(torch.ones((), device=grad.device))

    return q_list, lip_const_list


def _update_precond_procrustes(
    q_list: list[torch.Tensor],
    lip_const_list: list[torch.Tensor],
    exp_avg: torch.Tensor,
    damping_noise_scale: float = 1e-9,
    precond_lr: float = 0.1,
    beta_lip: float = 0.9,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    r"""Update the Kron preconditioner Q using procrustes step and uniformization.

    Args:
        q_list: List of Kronecker factors.
        lip_const_list: List of Lipschitz constants for the Kronecker factors.
        exp_avg: Exponential moving average of gradient.
        damping_noise_scale: Scale of noise added to gradient.
        precond_lr: Learning rate.
        beta_lip: EMA beta for the Lipschitz constant.

    Returns:
        q_list: List of Kronecker factors.
        lip_const_list: List of Lipschitz constants for the Kronecker factors.
    """
    dampened_momentum = exp_avg + (damping_noise_scale + 1e-7 * exp_avg.abs()) * torch.randn_like(exp_avg)
    pg = psgd_kron_contractions.apply_preconditioner(q_list, dampened_momentum)
    total_numel = pg.numel()
    updated_q_list: list[torch.Tensor] = []
    updated_lip_const_list: list[torch.Tensor] = []
    for dim, q in enumerate(q_list):
        # compute gradient covariance
        precond_grad_cov = psgd_kron_contractions.partial_contraction(pg, pg, dim)
        if q.dim() < 2:
            # diagonal or scalar-structured preconditioner
            q, updated_lip_const = _update_1d_preconditioner(
                q, lip_const_list[dim], precond_grad_cov, total_numel, precond_lr, beta_lip
            )
        else:
            # matrix-structured preconditioner
            q, updated_lip_const = _update_matrix_preconditioner(
                q, lip_const_list[dim], precond_grad_cov, total_numel, precond_lr, beta_lip
            )
        updated_q_list.append(q)
        updated_lip_const_list.append(updated_lip_const)

    return updated_q_list, updated_lip_const_list


def _update_matrix_preconditioner(
    q: torch.Tensor,
    lip_const: torch.Tensor,
    precond_grad_cov: torch.Tensor,
    total_numel: int,
    precond_lr: float,
    beta_lip: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Update matrix-structured preconditioner with adaptive Lipschitz constant.

    Args:
        q: Kronecker factor matrix for this dimension to update.
        lip_const: Lipschitz constant for this dimension.
        precond_grad_cov: Gradient covariance.
        total_numel: Total number of elements in the gradient.
        precond_lr: Learning rate.
        beta_lip: EMA beta for the Lipschitz constant.

    Returns:
        q: Updated Kronecker factor matrix for this dimension.
        lip_const: Updated Lipschitz constant for this dimension.
    """
    normalization = total_numel / q.shape[0]
    ell = psgd_utils.norm_lower_bound_spd(precond_grad_cov) + normalization
    lip_const = torch.max(beta_lip * lip_const + (1 - beta_lip) * ell, ell)
    q = q - precond_lr / lip_const * (precond_grad_cov @ q - normalization * q)
    q = procrustes_step(q)
    return q, lip_const


def _update_1d_preconditioner(
    q: torch.Tensor,
    lip_const: torch.Tensor,
    precond_grad_cov: torch.Tensor,
    total_numel: int,
    precond_lr: float,
    beta_lip: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Update 1D preconditioner with adaptive Lipschitz constant.

    Args:
        q: Kronecker factor 1D tensor for this dimension to update.
        lip_const: Lipschitz constant for this dimension.
        precond_grad_cov: Gradient covariance.
        total_numel: Total number of elements in the gradient.
        precond_lr: Learning rate.
        beta_lip: EMA beta for the Lipschitz constant.

    Returns:
        q: Updated Kronecker factor 1D tensor for this dimension.
        lip_const: Updated Lipschitz constant for this dimension.
    """
    normalization = total_numel / q.numel()
    ell = torch.max(precond_grad_cov) + normalization
    lip_const = torch.max(beta_lip * lip_const + (1 - beta_lip) * ell, ell)
    q = q * (1 - precond_lr / lip_const * (precond_grad_cov - normalization))
    return q, lip_const


def _get_precond_lr(precond_lr: float, step: int, min_precond_lr: float = 0.01, warmup_steps: int = 10000) -> float:
    r"""Helper function to get preconditioner learning rate for this optimization step based on a square root schedule.

    Decaying from a higher lr down to min_precond_lr improves accuracy.

    Args:
        precond_lr: Learning rate.
        step: Current step.
        min_precond_lr: Minimum learning rate.
        warmup_steps: Warmup steps.

    Returns:
        The preconditioner learning rate.
    """

    scheduled_lr = precond_lr / math.sqrt(1.0 + step / warmup_steps)
    return max(scheduled_lr, min_precond_lr)
