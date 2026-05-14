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
from functools import partial
from typing import TYPE_CHECKING, Callable, override


if TYPE_CHECKING:
    from typing import overload

import torch
from absl import logging
from torch import optim
from torch.optim.optimizer import ParamsT

from emerging_optimizers import mixin as opt_mixin
from emerging_optimizers import registry, utils
from emerging_optimizers.scalar_optimizers import update_functions
from emerging_optimizers.soap import soap_utils
from emerging_optimizers.utils import FP32MatmulPrecT


__all__ = [
    "SOAP",
    "precondition",
    "init_kronecker_factors",
    "update_kronecker_factors",
    "update_eigenbasis_and_exp_avgs",
]


@registry.register_optimizer("soap")
class SOAP(opt_mixin.WeightDecayMixin, optim.Optimizer):
    """Implements a variant of SOAP (ShampoO with Adam in the Preconditioner eigenbasis) algorithm.

    SOAP (https://arxiv.org/abs/2409.11321) is a preconditioned optimizer that combines the benefits of Shampoo's
    non-diagonal preconditioning with Adam's adaptive learning rates. It uses
    gradient correlation matrix eigenbasis-based preconditioning to adapt to the local geometry of the
    optimization landscape.

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        lr: The learning rate to use
        betas: Inner Adam's betas parameters (b1, b2)
        shampoo_beta: Beta for the kronecker factor matrices (L and R in paper) moving average
            instead of betas[1] if >= 0
        eps: Inner Adam's epsilon for numerical stability
        weight_decay: Weight decay coefficient
        weight_decay_method: Method to apply weight decay, see :class:`~emerging_optimizers.mixin.WeightDecayMixin`
            for more details.
        nesterov: uses Nesterov momentum in Adam (https://cs229.stanford.edu/proj2015/054_report.pdf,
            https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ)
        precondition_frequency: How often to update the preconditioner. Can be an integer for fixed frequency
            or a callable function that takes the current step as input and returns the frequency.
        adam_warmup_steps: How many steps to skip preconditioning in the beginning (i.e. use standard AdamW updates)
        correct_bias: Whether to use bias correction in Inner Adam and Kronecker factor matrices EMA
        fp32_matmul_prec: Precision of the matmul operations in optimizer states GEMM operations
        use_eigh: Whether to use full symmetric eigendecomposition (eigh) to compute the eigenbasis.
            If False, use orthogonal iteration to compute the eigenbasis.
        qr_fp32_matmul_prec: Precision of the matmul operations in QR decomposition.
        use_adaptive_criteria: Whether to use criteria to determine if eigenbasis update is needed
        adaptive_update_tolerance: Tolerance threshold for the update criteria.
            Only used if use_adaptive_criteria is True.
        power_iter_steps: Number of power iteration steps to perform before QR decomposition.
            More steps can lead to better convergence but increased computation time.
        max_update_rms: Clip the update RMS to this value (0 means no clipping).
        use_kl_shampoo: Whether to use KL-Shampoo correction.
        correct_shampoo_beta_bias: Whether to correct shampoo beta bias. Decoupled it from correct_bias for
            testability because reference implementation of Soap doesn't bias correct shampoo beta.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.95),
        shampoo_beta: float = 0.95,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        *,
        weight_decay_method: opt_mixin.WeightDecayT = "decoupled",
        nesterov: bool = False,
        precondition_frequency: int | Callable[[int], int] = 1,
        adam_warmup_steps: int = 0,
        correct_bias: bool = True,
        fp32_matmul_prec: FP32MatmulPrecT = "high",
        use_eigh: bool = False,
        qr_fp32_matmul_prec: FP32MatmulPrecT = "high",
        use_adaptive_criteria: bool = False,
        adaptive_update_tolerance: float = 1e-7,
        power_iter_steps: int = 1,
        max_update_rms: float = 0.0,
        use_kl_shampoo: bool = False,
        correct_shampoo_beta_bias: bool | None = None,
    ) -> None:
        self.precondition_frequency = precondition_frequency
        self.adam_warmup_steps = adam_warmup_steps
        self.nesterov = nesterov
        self.correct_bias = correct_bias
        self.weight_decay_method = weight_decay_method
        self.fp32_matmul_prec = fp32_matmul_prec
        self.use_eigh = use_eigh
        self.qr_fp32_matmul_prec = qr_fp32_matmul_prec
        self.use_adaptive_criteria = use_adaptive_criteria
        self.adaptive_update_tolerance = adaptive_update_tolerance
        self.power_iter_steps = power_iter_steps
        self.max_update_rms = max_update_rms
        self.use_kl_shampoo = use_kl_shampoo
        if correct_shampoo_beta_bias is not None:
            self.correct_shampoo_beta_bias = correct_shampoo_beta_bias
        else:
            self.correct_shampoo_beta_bias = correct_bias

        defaults = {
            "lr": lr,
            "betas": betas,
            "shampoo_beta": shampoo_beta,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()  # type: ignore[misc]
    def _init_group(
        self,
        group: dict,
        skip_non_grad_params: bool = True,
    ) -> None:
        """Performs lazy state initialization for parameters with gradients.

        Args:
            group: Parameter group dictionary.
            skip_non_grad_params: Whether to skip parameters with no gradients.

        Raises:
            TypeError: If the parameter is not a 2D tensor.
        """
        for p in group["params"]:
            if skip_non_grad_params and p.grad is None:
                continue

            if p.dim() != 2:
                raise TypeError("SOAP is only supported for 2D tensors")

            state = self.state[p]

            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p.data, dtype=torch.float32)
                state["exp_avg_sq"] = torch.zeros_like(p.data, dtype=torch.float32)

                # Use shape of p instead of grad for initialization because of the introduction of skip_non_grad_params
                # for megatron-lm distributed checkpointing use. _init_group can be called without grad.
                state["L"], state["R"] = init_kronecker_factors(p.shape, device=p.device)
                state["Q_L"] = torch.eye(p.shape[0], device=p.device)
                state["Q_R"] = torch.eye(p.shape[1], device=p.device)

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
            self._init_group(group)

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.to(torch.float32)
                state = self.state[p]

                # NOTE: The upstream PyTorch implementations increment the step counter in the middle of the loop
                # to be used in bias correction. But this is confusing and error prone if anything else needs to use
                # the step counter.
                # We decided to follow Python and C convention to increment the step counter at the end of the loop.
                # An explicitly named 1-based iteration/step counter is created for bias correction and other terms
                # in the math equation that needs 1-based iteration count.
                curr_iter_1_based = state["step"] + 1

                # Define kronecker_factor_update_fn based on whether to use KL-Shampoo here
                # because it needs access to eigenbasis_list and group
                kronecker_factor_list = [state["L"], state["R"]]
                eigenbasis_list = [state["Q_L"], state["Q_R"]]

                if not self.use_kl_shampoo:
                    kronecker_factor_update_fn = update_kronecker_factors
                else:
                    kronecker_factor_update_fn = partial(
                        update_kronecker_factors_kl_shampoo,
                        eigenbasis_list=eigenbasis_list,
                        eps=group["eps"],
                    )

                shampoo_beta = group["shampoo_beta"]
                if self.correct_shampoo_beta_bias:
                    shampoo_beta = 1 - (1 - shampoo_beta) / (1 - shampoo_beta**curr_iter_1_based)

                torch.cuda.nvtx.range_push("update_kronecker_factors")
                with utils.fp32_matmul_precision(self.fp32_matmul_prec):
                    kronecker_factor_update_fn(
                        kronecker_factor_list=kronecker_factor_list, grad=grad, shampoo_beta=shampoo_beta
                    )
                torch.cuda.nvtx.range_pop()

                # After the adam_warmup_steps are completed , update eigenbases at precondition_frequency steps
                torch.cuda.nvtx.range_push("Update eigen basis")
                if _is_eigenbasis_update_step(
                    state["step"],
                    self.adam_warmup_steps,
                    self.precondition_frequency,
                ):
                    # Always use eigh for the first eigenbasis update
                    use_eigh = self.use_eigh if state["step"] != self.adam_warmup_steps else True

                    # Skip eigenbasis update if use_adaptive_criteria is True and all eigenbases meet the criteria.
                    # Never skip the first eigenbasis update (step == adam_warmup_steps) since QL/QR are still identity.
                    skip_update = (
                        self.use_adaptive_criteria
                        and state["step"] > self.adam_warmup_steps
                        and soap_utils.all_eigenbases_met_criteria(
                            kronecker_factor_list, eigenbasis_list, self.adaptive_update_tolerance
                        )
                    )
                    if not skip_update:
                        with utils.fp32_matmul_precision(self.qr_fp32_matmul_prec):
                            updated_eigenbasis_list, exp_avg, exp_avg_sq = update_eigenbasis_and_exp_avgs(
                                kronecker_factor_list=kronecker_factor_list,
                                eigenbasis_list=eigenbasis_list,
                                exp_avg_sq=state["exp_avg_sq"],
                                exp_avg=state["exp_avg"],
                                use_eigh=use_eigh,
                                power_iter_steps=self.power_iter_steps,
                            )
                            state["Q_L"], state["Q_R"] = updated_eigenbasis_list

                            # rebind local ref so precondition() below uses the updated Q
                            eigenbasis_list = updated_eigenbasis_list

                            state["exp_avg"] = exp_avg
                            state["exp_avg_sq"] = exp_avg_sq
                torch.cuda.nvtx.range_pop()

                self._apply_weight_decay_inplace(
                    p,
                    grad,
                    group["lr"],
                    group["weight_decay"],
                )

                grad_projected = grad
                # Project gradients to the eigenbases of Shampoo's preconditioner
                torch.cuda.nvtx.range_push("precondition")
                if state["step"] >= self.adam_warmup_steps:
                    with utils.fp32_matmul_precision(self.fp32_matmul_prec):
                        grad_projected = precondition(
                            grad,
                            eigenbasis_list=eigenbasis_list,
                            dims=[[0], [0]],
                        )
                torch.cuda.nvtx.range_pop()

                # Calculate the Adam update for the projected gradient tensor
                adam_update = update_functions.calculate_adam_update(
                    grad_projected,
                    state["exp_avg"],
                    state["exp_avg_sq"],
                    group["betas"],
                    self.correct_bias,
                    self.nesterov,
                    curr_iter_1_based,  # 1-based iteration index is used for bias correction
                    group["eps"],
                )

                # Projecting back the preconditioned (by ADAM) exponential moving average of gradients
                torch.cuda.nvtx.range_push("precondition")
                if state["step"] >= self.adam_warmup_steps:
                    with utils.fp32_matmul_precision(self.fp32_matmul_prec):
                        precond_update = precondition(
                            adam_update,
                            eigenbasis_list=eigenbasis_list,
                            dims=[[0], [1]],
                        )
                else:
                    precond_update = adam_update
                torch.cuda.nvtx.range_pop()

                _clip_update_rms_in_place(precond_update, self.max_update_rms)
                p.add_(precond_update, alpha=-group["lr"])

                state["step"] += 1

        return loss


@torch.no_grad()  # type: ignore[misc]
def init_kronecker_factors(
    grad_shape: torch.Size,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Initializes the kronecker factor matrices for the SOAP optimizer.

    This function creates the initial Kronecker factor matrices (L and R) used for
    preconditioning. It creates a square kronecker factor matrix for each dimension
    of the 2D gradient shape.

    Note:
        The Kronecker factors are always initialized to float32 (unless default precision is set otherwise) as its
        accumulation and decomposition are not safe in lower precisions.

    Args:
        grad_shape: Shape of the gradient tensor. Must be 2D.
            Determines the size of the kronecker factor matrices.
        device: Device on which to create the kronecker factor matrices.

    Returns:
        Tuple of kronecker factor matrices (L and R in paper).

    Example:
        >>> # For a 2D tensor (weight matrix)
        >>> grad_shape = torch.Size([10, 20])
        >>> precond_2d = init_kronecker_factors(grad_shape)
        >>> print(len(precond_2d))  # 2
        >>> print(precond_2d[0].shape)  # (10, 10)
        >>> print(precond_2d[1].shape)  # (20, 20)

    """
    if len(grad_shape) != 2:
        raise TypeError("init_kronecker_factors is only supported for 2D tensors")

    # Create a square kronecker factor matrix for each dimension
    L = torch.zeros(grad_shape[0], grad_shape[0], device=device)
    R = torch.zeros(grad_shape[1], grad_shape[1], device=device)
    return L, R


@torch.no_grad()  # type: ignore[misc]
def update_kronecker_factors(
    kronecker_factor_list: list[torch.Tensor],
    grad: torch.Tensor,
    shampoo_beta: float,
) -> None:
    """Updates the preconditioner matrices using gradient outer products.

    This function updates the Kronecker factor matrices (L and R) used for preconditioning
    by computing and accumulating gradient outer products. kronecker_factor_list is updated in place.

    Args:
        kronecker_factor_list: List of preconditioner matrices (L and R) to update.
            Each matrix should be square and match the corresponding dimension of grad.
        grad: Gradient tensor of the parameter being optimized
        shampoo_beta: Momentum coefficient for updating preconditioners.
            Controls how much weight to give to new vs old gradient statistics.

    Example:
        >>> grad = torch.randn(10, 20)
        >>> L = torch.zeros(10, 10)
        >>> R = torch.zeros(20, 20)
        >>> update_kronecker_factors([L, R], grad, shampoo_beta=0.95)

    """
    # L = G @ G.T, R = G.T @ G
    kronecker_factor_list[0].lerp_(grad @ grad.T, 1 - shampoo_beta)
    kronecker_factor_list[1].lerp_(grad.T @ grad, 1 - shampoo_beta)


@torch.no_grad()  # type: ignore[misc]
def update_kronecker_factors_kl_shampoo(
    kronecker_factor_list: list[torch.Tensor],
    grad: torch.Tensor,
    shampoo_beta: float,
    eigenbasis_list: list[torch.Tensor],
    eps: float,
    eigval_exp: float = -1.0,
) -> None:
    """Updates the kronecker factor matrices in place using KL-Shampoo correction.

    Implement Kullback–Leibler Minimization from https://arxiv.org/pdf/2509.03378

    Args:
        kronecker_factor_list: List of preconditioner matrices (L and R) to update.
        grad: Gradient tensor of the parameter being optimized
        shampoo_beta: Momentum coefficient for updating preconditioners.
        eigenbasis_list: List of orthonormal eigenbases of the kronecker factor matrices
        eps: Small offset for numerical stability.
        eigenval_exp: Exponent of the eigenvalues.
    """
    if grad.dim() != 2:
        raise TypeError("KL-Shampoo mathematical correction is only supported for 2D tensors")

    # Scale the gradient matrix by the approximate eigenvalues and the eigenbasis
    # G@Q_R@λ_R^(−1)@Q_R.T@G.T/dim(GG.T) and G.T@Q_L@λ_L^(−1)@Q_L.T@G/dim(G.TG)
    updates = []
    for idx, (kronecker_factor, eigenbasis) in enumerate(zip(kronecker_factor_list, eigenbasis_list, strict=True)):
        approx_eigvals = utils.eig.conjugate(kronecker_factor, eigenbasis, diag=True)
        scale_factor = 1 / grad.shape[idx] * approx_eigvals.clamp_min(eps) ** eigval_exp

        logging.debug(f"scale_factor[{idx}]: {scale_factor}")

        correction = (eigenbasis * scale_factor[None, :]) @ eigenbasis.T

        maybe_transpose_grad = grad.T if idx == 1 else grad
        updates.append(utils.eig.conjugate(correction, maybe_transpose_grad))

    # Note that updates caculated in previous loop are in reverse order of the kronecker factor list they apply to
    for kronecker_factor, update in zip(kronecker_factor_list, updates[::-1], strict=True):
        kronecker_factor.lerp_(update, 1 - shampoo_beta)


@torch.no_grad()  # type: ignore[misc]
def update_eigenbasis_and_exp_avgs(
    kronecker_factor_list: list[torch.Tensor],
    eigenbasis_list: list[torch.Tensor],
    exp_avg_sq: torch.Tensor,
    exp_avg: torch.Tensor,
    use_eigh: bool = False,
    power_iter_steps: int = 1,
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Updates the eigenbases and moving averages.

    This function performs an update of the eigenbases (QL and QR)
    used for preconditioning. It follows these steps:

    1. Projects exp_avg back to the original basis
    2. Updates the eigenbases using QR decomposition and power iteration (orthogonal iteration)
    3. Projects exp_avg back to the new eigenbasis

    Args:
        kronecker_factor_list: List of preconditioner matrices (L and R) that define
            the optimization landscape. These are updated with gradient statistics.
        eigenbasis_list: List of current eigenbases (QL and QR)
            used for preconditioning. These will be updated by this function.
        exp_avg_sq: Inner Adam's second moment tensor, used for scaling the preconditioner updates.
            This tensor is modified in-place.
        exp_avg: Inner Adam's first moment tensor, used for tracking gradient momentum.
            This tensor is modified in-place.
        use_eigh: Whether to use full symmetric eigendecomposition (eigh) to compute the eigenbasis.
            If False, use orthogonal iteration to compute the eigenbasis.
        power_iter_steps: Number of power iteration steps to perform before QR decomposition.
            More steps can lead to better convergence but increased computation time.

    Returns:
        A tuple containing:
            - Updated list of eigenbases (QL and QR)
            - Updated exp_avg tensor projected to the new eigenbasis
            - Updated exp_avg_sq tensor

    Example:
        >>> L = torch.randn(10, 10)
        >>> R = torch.randn(20, 20)
        >>> QL = torch.randn(10, 10)
        >>> QR = torch.randn(20, 20)
        >>> exp_avg_sq = torch.randn(10, 20)
        >>> exp_avg = torch.randn(10, 20)
        >>> updated_eigenbasis_list, updated_exp_avg, updated_exp_avg_sq = update_eigenbasis_and_exp_avgs(
        ...     [L, R], [QL, QR], exp_avg_sq, exp_avg)

    """
    # Step 1: Project exp_avg back to the original basis
    torch.cuda.nvtx.range_push("eigenbasis update step 1: precondition")
    exp_avg = precondition(
        exp_avg,
        eigenbasis_list,
        dims=[[0], [1]],
    )
    torch.cuda.nvtx.range_pop()

    # Step 2: Update eigenbases
    torch.cuda.nvtx.range_push("eigenbasis update step 2: update Q")
    if use_eigh:
        updated_eigenbasis_list = soap_utils.get_eigenbasis_eigh(
            kronecker_factor_list,
        )
    else:
        # Use QR decomposition and power iteration (orthogonal iteration)
        updated_eigenbasis_list, exp_avg_sq = soap_utils.get_eigenbasis_qr(
            kronecker_factor_list,
            eigenbasis_list,
            exp_avg_sq,
            power_iter_steps,
        )
    torch.cuda.nvtx.range_pop()

    # Step 3: Project exp_avg to the new eigenbasis using the updated eigenbases
    torch.cuda.nvtx.range_push("eigenbasis update step 3: project exp_avg")
    exp_avg = precondition(
        exp_avg,
        updated_eigenbasis_list,
        dims=[[0], [0]],
    )
    torch.cuda.nvtx.range_pop()

    return updated_eigenbasis_list, exp_avg, exp_avg_sq


@torch.no_grad()  # type: ignore[misc]
def precondition(
    x: torch.Tensor,
    eigenbasis_list: list[torch.Tensor] | None = None,
    dims: list[list[int]] | None = None,
) -> torch.Tensor:
    """Projects the gradient to and from the eigenbases of the kronecker factor matrices.

    This function performs tensor contractions between the input gradient
    and kronecker factor eigenbases.

    Note:
        For 2D tensors, we can use matmul instead of tensordot for code legibility. However, the code has
        been using tensordot historically, so does the reference implementation. It is difficult to match
        matmul and tensordot outputs exactly because of underlying floating point arithmetic differences.
        Therefore, we decided to keep using tensordot for consistency.


    Args:
        x: Input tensor to be preconditioned
        eigenbasis_list: List of eigenbases for preconditioning.
            Each matrix should be a square matrix of eigenvectors.
        dims: Dimensions for tensor contraction. Default is [[0], [0]] which contracts
            the first dimension of grad with the first dimension of each eigenbasis matrix,
            for projecting into the eigenbasis. Use [[0], [1]] for projecting back to original space.

    Example:
        >>> x = torch.randn(10, 20)
        >>> Q = torch.randn(10, 10)
        >>> precondition(x, [Q], dims=[[0], [0]])
    """
    if dims is None:
        # Pick contraction dims to project to the eigenbasis
        dims = [[0], [0]]

    if eigenbasis_list is None:
        # If eigenbases are not provided, return the gradient without any preconditioning
        return x

    for Q in eigenbasis_list:
        x = torch.tensordot(x, Q, dims=dims)

    return x


def _is_eigenbasis_update_step(
    step: int,
    adam_warmup_steps: int,
    precondition_frequency: int | Callable[[int], int],
) -> bool:
    """Checks if amortized computation of the eigenbasis should be recomputed.

    Args:
        step: Current step of the optimizer
        adam_warmup_steps: Number of steps to skip preconditioning in the beginning (i.e. use standard AdamW updates)
        precondition_frequency: How often to update the preconditioner. Can be an integer for fixed frequency
            or a callable function that takes the current step as input and returns the frequency.
    """
    if step < adam_warmup_steps:
        return False

    current_frequency = (
        precondition_frequency if not callable(precondition_frequency) else precondition_frequency(step)
    )

    return step % current_frequency == 0


@torch.compile  # type: ignore[misc]
def _clip_update_rms_in_place(u: torch.Tensor, max_rms: float, eps: float = 1e-7) -> None:
    """Clip the update root mean square (RMS) to a maximum value, in place.

    Do not clip if max_rms is 0.
    Inspired by Adafactor (https://arxiv.org/abs/1804.04235) and RMS_t (https://arxiv.org/abs/2304.13013)

    Args:
        u: The update tensor.
        max_rms: The maximum RMS value.
        eps: The epsilon value to prevent division by zero.
    """
    if max_rms == 0:
        return
    # compute current update RMS
    rms = u.square().mean().sqrt()
    # compute scale factor = min(1.0, max_rms/(rms + eps))
    scale = (max_rms / (rms + eps)).clamp(max=1.0)
    # in‐place scale
    u.mul_(scale)
