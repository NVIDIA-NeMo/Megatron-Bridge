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
from absl import logging
from torch.optim.optimizer import ParamsT

from emerging_optimizers import registry
from emerging_optimizers.mixin import WeightDecayT
from emerging_optimizers.orthogonalized_optimizers import muon_utils
from emerging_optimizers.orthogonalized_optimizers.muon_utils import NSCoeffT
from emerging_optimizers.orthogonalized_optimizers.orthogonalized_optimizer import OrthogonalizedOptimizer, _args_doc
from emerging_optimizers.utils import FP32MatmulPrecT


__all__ = ["PolarGrad"]


@registry.register_optimizer("polargrad")
class PolarGrad(OrthogonalizedOptimizer):
    """PolarGrad: Polar Gradient Methods.

    PolarGrad runs standard SGD-momentum with Nesterov momentum, and then performs an orthogonalization
    post-processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. Note that the update is also scaled by the nuclear norm of the momentum term. This is
    equivalent to solving the steepest descent w.r.t. the spectral norm, as opposed to the LMO formulation
    of Scion and Muon. To efficiently orthogonalize each update, Newton-Schulz iteration is used, which has the
    advantage that it may be stably run on tensor cores on GPUs.

    This implementation incorporates decoupled weight decay.

    References:
        - *PolarGrad: A Class of Matrix-Gradient Optimizers from a Unifying Preconditioning Perspective.* arXiv:2505.21799 (2025).
          [`arXiv:2505.21799 <https://arxiv.org/abs/2505.21799>`_]
        - Lau, T. T.-K. *PolarGrad Optimizer Implementation.*
          [`polar_grad.py <https://github.com/timlautk/polargrad/blob/main/polar_grad.py>`_]

    Warning:
        - This optimizer requires that all parameters passed in are 2D.
        - It should not be used for the embedding layer, the final fully connected layer (with the Newton-Schulz iteration), or any 1-D
          parameters; those can all be optimized by a standard method (e.g., AdamW).

    Args:
        {_args_doc}
        coefficient_type: The type of coefficient set to use for the Newton-Schulz iteration. Can be one of
            ["simple", "quintic", "polar_express"].
        num_ns_steps: The number of iteration steps to use in the Newton-Schulz iteration.
        extra_scale_factor: The additional scale factor to use for the update. Setting it to 0.2 can closely match
            the update RMS norm of AdamW as suggested by https://arxiv.org/abs/2502.16982.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 3e-4,
        momentum: float = 0.95,
        weight_decay: float = 0.01,
        *,
        nesterov: bool = False,
        weight_decay_method: WeightDecayT = "decoupled",
        fp32_matmul_prec: FP32MatmulPrecT = "highest",
        coefficient_type: NSCoeffT = "quintic",
        num_ns_steps: int = 5,
        extra_scale_factor: float = 1.0,
    ) -> None:
        if num_ns_steps < 0:
            # 0 NS steps is allowed for some tests to bypass Newton-Schulz iterations and have exact match.
            raise ValueError(f"num_ns_steps must be positive, got {num_ns_steps}")

        def scaled_orthogonalize_fn(grad: torch.Tensor) -> torch.Tensor:
            logging.debug(
                f"Orthogonalizing grad with {num_ns_steps} steps, {coefficient_type} coefficient, "
                f"multiplied with the nuclear norm of grad, extra_scale_factor={extra_scale_factor}"
            )
            orth_grad = muon_utils.newton_schulz(
                grad,
                steps=num_ns_steps,
                coefficient_type=coefficient_type,
            )
            scale_factor = (orth_grad * grad).sum()
            return orth_grad * scale_factor * extra_scale_factor

        super().__init__(
            params,
            lr,
            momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            weight_decay_method=weight_decay_method,
            fp32_matmul_prec=fp32_matmul_prec,
            scaled_orthogonalize_fn=scaled_orthogonalize_fn,
        )


PolarGrad.__doc__ = PolarGrad.__doc__.format(_args_doc=_args_doc)  # type: ignore[union-attr]
