# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import torch.nn.functional as F


__all__ = [
    "SinkhornMapper",
]


class SinkhornMapper:
    """Applies the Sinkhorn-Knopp mapping to the input tensor.

    The Sinkhorn-Knopp mapping is an iterative technique for normalizing the rows and columns of a matrix:
    Input -> [Exp] -> [Iterative Row/Col Normalization]

    Supports batched inputs (3D+). The mapping operates on the last two dimensions.

    For an M×N matrix, the normalization targets are:
        - Square (M=N): row sums = 1.0, col sums = 1.0 (standard doubly-stochastic)
        - Wide (N>M): row sums = N/M, col sums = 1.0
        - Tall (M>N): row sums = 1.0, col sums = M/N

    Based on Deepseek's Manifold-Constrained Hyperconnections (https://arxiv.org/abs/2512.24880)

    Args:
        num_iters: The number of iterations to run the Sinkhorn-Knopp mapping.
        eps: The epsilon value to use for the Sinkhorn-Knopp mapping for numerical stability.
    """

    def __init__(self, num_iters: int = 20, eps: float = 1e-8):
        self.num_iters = num_iters
        self.eps = eps

    @torch.no_grad()
    def _sinkhorn_map(self, x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
        """Apply Sinkhorn-Knopp mapping to the input tensor.

        Args:
            x: Input tensor to apply the mapping to. Must be at least 2D. Batched inputs (3D+) are supported.
            inplace: If True, modify x in place. If False, work on a copy.

        Returns:
            The tensor with the Sinkhorn-Knopp mapping applied.
        """
        if x.dim() < 2:
            raise ValueError(
                f"{self.__class__.__name__} requires at least a 2D tensor, got {x.dim()}D with shape {x.shape}"
            )
        result = x if inplace else x.clone()

        # Enforce positivity via exp with numerical stability.
        # Subtract global max before exp to prevent overflow (log-sum-exp trick).
        # The normalization step will scale the result, so subtracting any max (global, row, or column)
        # is sufficient for numerical stability.
        result.sub_(result.max()).exp_()

        # Determine normalization targets based on aspect ratio.
        # For non-square matrices (M x N), we scale the shorter dimension so that
        # rows sum to N/M and cols sum to 1.0 (if N > M), or
        # rows sum to 1.0 and cols sum to M/N (if M > N).
        # See chapter 4 of https://arxiv.org/abs/1803.00567.
        # For square matrices, both targets are 1.0 (standard doubly-stochastic).
        M, N = result.shape[-2], result.shape[-1]
        if N > M:
            row_target = N / M
            col_target = 1.0
        else:
            row_target = 1.0
            col_target = M / N

        # Iterative normalization of rows and columns
        for _ in range(self.num_iters):
            # Normalize columns (along row dimension)
            F.normalize(result, p=1, dim=-2, eps=self.eps, out=result)
            if col_target != 1.0:
                result.mul_(col_target)

            # Normalize rows (along column dimension)
            F.normalize(result, p=1, dim=-1, eps=self.eps, out=result)
            if row_target != 1.0:
                result.mul_(row_target)

        return result

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
        """Apply Sinkhorn-Knopp mapping to the input tensor.

        Args:
            x: Input tensor to apply the mapping to.
            inplace: If True, modify x in place. If False, work on a copy.

        Returns:
            The tensor with the Sinkhorn-Knopp mapping applied (modified in place if inplace=True, otherwise a new tensor).
        """
        return self._sinkhorn_map(x, inplace=inplace)
