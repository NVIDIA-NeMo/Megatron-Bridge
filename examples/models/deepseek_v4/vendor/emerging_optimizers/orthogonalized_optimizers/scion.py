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
from emerging_optimizers.orthogonalized_optimizers import muon_utils
from emerging_optimizers.orthogonalized_optimizers.muon import get_muon_scale_factor
from emerging_optimizers.orthogonalized_optimizers.muon_utils import NSCoeffT
from emerging_optimizers.orthogonalized_optimizers.orthogonalized_optimizer import OrthogonalizedOptimizer
from emerging_optimizers.utils import FP32MatmulPrecT


__all__ = ["Scion"]


@registry.register_optimizer("scion")
class Scion(OrthogonalizedOptimizer):
    """Scion: Stochastic CondItional descent with Operator Norms

    Scion runs standard SGD-momentum and then performs an orthogonalization
    post-processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, Newton-Schulz iteration is used, which has the
    advantage that it may be stably run on tensor cores on GPUs.

    This implementation incorporates `step_size` and `spectral_radius`, refer to Scion which views weight decay as constrained
    optimization via Frank-Wolfe.

    References:
        - *Training Deep Learning Models with Norm-Constrained LMOs.* arXiv:2502.07529 (2025).
          [`arXiv:2502.07529 <https://arxiv.org/abs/2502.07529>`_]

    Warning:
        - This optimizer requires that all parameters passed in are 2D.
        - It should not be used for the embedding layer, the final fully connected layer, or any 1-D
          parameters; those should all be optimized by the appropriate LMO for that layer. For example,
          for 1d params, it is scaled by the `ell_inf` radius.


    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        fp32_matmul_prec: Precision of the matmul operations in optimizer states GEMM operations.
        coefficient_type: The type of coefficient set to use for the Newton-Schulz iteration. Can be one of
            ["simple", "quintic", "polar_express"].
        num_ns_steps: The number of iteration steps to use in the Newton-Schulz iteration.
        spectral_radius: The spectral radius to use for the update, we are scaling the LMO by this spectral radius.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 3e-4,
        momentum: float = 0.95,
        *,
        fp32_matmul_prec: FP32MatmulPrecT = "medium",
        coefficient_type: NSCoeffT = "quintic",
        num_ns_steps: int = 5,
        spectral_radius: float = 1.0,
    ) -> None:
        if num_ns_steps < 1:
            raise ValueError(f"num_ns_steps must be at least 1, got {num_ns_steps}")

        # Add checks for weight decay arguments to enable Franke-Wolfe step.
        logging.info(
            "Scion does not use weight decay. Setting weight_decay to 1 and weight_decay_method to decoupled."
        )
        weight_decay = 1
        weight_decay_method = "decoupled"

        logging.info("Scion does not use Nesterov momentum. Setting nesterov to False.")
        nesterov = False

        def scaled_orthogonalize_fn(grad: torch.Tensor) -> torch.Tensor:
            logging.debug(
                f"Orthogonalizing grad with {num_ns_steps} steps, {coefficient_type} coefficient, spectral_radius={spectral_radius}"
            )
            orth_grad = muon_utils.newton_schulz(
                grad, steps=num_ns_steps, coefficient_type=coefficient_type, use_syrk=False
            )
            width_factor = get_muon_scale_factor(grad.size(-2), grad.size(-1), mode="unit_rms_norm")
            return orth_grad * width_factor * spectral_radius

        super().__init__(
            params,
            lr,
            momentum,
            weight_decay,
            nesterov=nesterov,
            weight_decay_method=weight_decay_method,  # type: ignore[arg-type]
            fp32_matmul_prec=fp32_matmul_prec,
            scaled_orthogonalize_fn=scaled_orthogonalize_fn,
        )
