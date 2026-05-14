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
from typing import TYPE_CHECKING, Any, Callable, override


if TYPE_CHECKING:
    from typing import overload

import torch
import torch.optim as optim
from absl import logging
from torch.optim.optimizer import ParamsT

from emerging_optimizers import mixin as opt_mixin
from emerging_optimizers import utils
from emerging_optimizers.utils import FP32MatmulPrecT


_args_doc = """params: Iterable of parameters to optimize or dicts defining parameter groups
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        weight_decay: The weight decay used by the optimizer, default to be decoupled weight decay.
            See Decoupled Weight Decay Regularization: https://arxiv.org/abs/1711.05101
        nesterov: Whether to use Nesterov-style momentum in the internal SGD.
        weight_decay_method: Method to apply weight decay, see :class:`~emerging_optimizers.mixin.WeightDecayMixin`
            for more details.
        fp32_matmul_prec: Precision of the matmul operations in optimizer states GEMM operations.
"""


class OrthogonalizedOptimizer(opt_mixin.WeightDecayMixin, optim.Optimizer):
    """Base class for orthogonalized optimizers.

    This class is a wrapper around a base optimizer that performs orthogonalization on the updates.
    The theoretical foundation of orthogonalization for stochastic gradient descent was developed by the
    following papers:

    - Carlson, D., Cevher, V., and Carin, L. *Stochastic spectral descent for Restricted Boltzmann Machines.*
      In International Conference on Artificial Intelligence and Statistics (2015a).
    - Carlson, D., Hsieh, Y.-P., Collins, E., Carin, L., and Cevher, V.
      *Stochastic Spectral Descent for Discrete Graphical Models.*
      In IEEE Journal of Selected Topics in Signal Processing, vol. 10, no. 2, pp. 296-311 (2016).
    - Carlson, D., Collins, E., Hsieh, Y.-P., Carin, L., and Cevher, V.
      *Preconditioned spectral descent for deep learning.*
      In Neural Information Processing Systems (2015b).
    - Flynn, T. *The duality structure gradient descent algorithm: analysis and applications to neural networks.*
      arXiv preprint arXiv:1708.00523 (2017). [`arXiv:1708.00523 <https://arxiv.org/abs/1708.00523>`_]

    Note:
        OrthogonalizedOptimizer as base class doesn't directly support orthogonalizing fused parameters separately.
        Subclass can override the orthogonalize function to support this, see example below.

    .. code-block:: python
       :caption: Split QKV example

       class SplitQkvOrthogonalizedOptimizer(OrthogonalizedOptimizer):
           def __init__(..., split_qkv_shapes):
               super().__init__(...)
               self.qkv_split_shapes = split_qkv_shapes

           def orthogonalize(self, p: torch.Tensor, grad: torch.Tensor, **kwargs: Any) -> torch.Tensor:

               # Alternative is passing "is_qkv" to scaled_orthogonalize_fn and split inside the
               # scaled_orthogonalize_fn.
               if getattr(p, "is_qkv", False) or kwargs.get("is_qkv", False):
                   qkv_grads = torch.split(grad, self.qkv_split_shapes, dim=0)
                   qkv_orthogonalized = [self.scaled_orthogonalize_fn(g) for g in qkv_grads]
                   grad = torch.cat([orthogonalized for orthogonalized in qkv_orthogonalized])
               else:
                   grad = self.scaled_orthogonalize_fn(grad)

               return grad

    Args:
        {_args_doc}
        scaled_orthogonalize_fn: Function to orthogonalize and scale the updates.
        **kwargs: Arguments passed through to the base optimizer.

    Note:
        Keyword arguments passed through are not checked here. Optimizer inherited from this class should check them.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float,
        momentum: float,
        weight_decay: float,
        *,
        nesterov: bool,
        weight_decay_method: opt_mixin.WeightDecayT,
        fp32_matmul_prec: FP32MatmulPrecT,
        scaled_orthogonalize_fn: Callable | None = None,
        **kwargs: Any,
    ):
        if scaled_orthogonalize_fn is None:
            logging.warning("scaled_orthogonalize_fn not provided. Using noop")
            scaled_orthogonalize_fn = torch.nn.Identity()

        self.fp32_matmul_prec = fp32_matmul_prec
        self.nesterov = nesterov
        self.weight_decay_method = weight_decay_method

        default_args_dict = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            **kwargs,
        )

        super().__init__(params, default_args_dict)
        self.scaled_orthogonalize_fn = scaled_orthogonalize_fn

    @torch.no_grad()  # type: ignore[misc]
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

            # initialize momentum buffer
            if len(state) == 0:
                state["momentum_buffer"] = torch.zeros_like(p.data)

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

                grad = p.grad
                state = self.state[p]

                self._apply_weight_decay_inplace(
                    p,
                    grad,
                    group["lr"],
                    group["weight_decay"],
                )

                # update momentum buffer with EMA of gradient
                state["momentum_buffer"].lerp_(grad, 1 - group["momentum"])

                # include nesterov momentum
                if self.nesterov:
                    grad = grad.lerp(state["momentum_buffer"], group["momentum"])
                else:
                    grad = state["momentum_buffer"]

                with utils.fp32_matmul_precision(self.fp32_matmul_prec):
                    group_kwargs = {k: v for k, v in group.items() if k != "params"}
                    orth_grad = self.orthogonalize(p, grad, **group_kwargs)

                # perform weight update with pre and post weight update functions for subclass customization
                self.pre_weight_update_fn_inplace(p, orth_grad)
                p.add_(orth_grad, alpha=-group["lr"])
                self.post_weight_update_fn_inplace(p)

        return loss

    def orthogonalize(self, p: torch.Tensor, grad: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Orthogonalize the momentum.

        The default orthogonalize function calls the scaled_orthogonalize_fn with the gradient. Subclass can
        override this function to implement different orthogonalization logic as well as split fused parameters.
        For example, a scaled_orthogonalize_fn function can get attributes from p or from kwargs to determine if
        the parameter is a fused parameter and should be split for preconditioning.

        Note:
            N-D parameters can be supported by overriding this function. For example, convolution weight can be
            supported by reshaping to [output_channels, input_channels * kernel_height * kernel_width], i.e. treating
            convolution as matrix multiplication with im2col.

        Args:
            p: The parameter tensor. It is necessary to pass param tensor in addition to momentum because a lot of
                information is only available in the param tensor, attributes for example. Although not used in
                this default orthogonalize function.
            grad: The momentum tensor.
            **kwargs: keyword arguments of the param_group that p was belonged to.

        Returns:
            The orthogonalized gradient tensor.
        """
        if grad.ndim != 2:
            raise ValueError("Only 2D parameters are supported.")
        grad = self.scaled_orthogonalize_fn(grad)
        return grad

    def pre_weight_update_fn_inplace(self, p: torch.Tensor, update: torch.Tensor) -> None:
        """Function called before the final weight update.

        Subclasses can override this to implement custom behavior before the weight update.
        For example, to implement hyperball-style updates that preserve weight norms.

        Warning:
            This function is experimental and may change in future versions.

        Args:
            p: The parameter tensor.
            update: The orthogonalized gradient tensor (will be applied as p -= lr * update).
        """
        pass

    def post_weight_update_fn_inplace(self, p: torch.Tensor) -> None:
        """Function called after the final weight update.

        Subclasses can override this to implement custom behavior after the weight update.
        For example, to implement hyperball-style updates that preserve weight norms.

        Warning:
            This function is experimental and may change in future versions.

        Args:
            p: The parameter tensor (already updated).
        """
        pass


OrthogonalizedOptimizer.__doc__ = OrthogonalizedOptimizer.__doc__.format(_args_doc=_args_doc)  # type: ignore[union-attr]
