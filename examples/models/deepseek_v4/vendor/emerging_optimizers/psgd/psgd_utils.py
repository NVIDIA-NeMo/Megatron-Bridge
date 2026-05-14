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
    "uniformize_q_in_place",
    "norm_lower_bound_spd",
    "norm_lower_bound_skew",
]


@torch.compile  # type: ignore[misc]
def uniformize_q_in_place(Q_list: list[torch.Tensor]) -> None:
    """Balance the dynamic ranges of kronecker factors in place to prevent numerical underflow or overflow.

    Each tensor in `Q_list` is rescaled so that its maximum absolute entry
    becomes the geometric mean of all factors original maxima. This preserves
    the overall product of norms (and thus the scale of the Kronecker product)
    while avoiding numerical underflow or overflow when factors have widely
    differing magnitudes.

    Given tensors :math:`Q_1, Q_2, \\ldots, Q_n`:

    1. Compute max-absolute norms: :math:`\\|Q_i\\|_\\infty = \\max(|Q_i|)` for :math:`i = 1, \\ldots, n`
    2. Compute geometric mean: :math:`g = \\left(\\prod_{i=1}^{n} \\|Q_i\\|_\\infty \\right)^{1/n}`
    3. Rescale each tensor: :math:`Q_i \\leftarrow Q_i \\cdot \\frac{g}{\\|Q_i\\|_\\infty}`

    This ensures :math:`\\|Q_i\\|_\\infty = g` for all :math:`i`, while preserving the norm of
    the Kronecker product :math:`Q_1 \\otimes Q_2 \\otimes \\cdots \\otimes Q_n`.

    Args:
        Q_list: List of Q (e.g. the Kronecker factors), each tensor will be modified in place.

    Returns:
        None

    """
    if not Q_list:
        raise TypeError("Q_list cannot be empty.")

    order = len(Q_list)
    if order == 1:
        # with a single factor, no balancing is needed
        return

    # Compute max-abs norm of each factor
    norms = [torch.max(torch.abs(Q)) for Q in Q_list]

    # Compute geometric mean of those norms
    gmean = torch.prod(torch.stack(norms)) ** (1.0 / order)

    # Rescale each factor so its max‐abs entry == geometric mean
    for Q, norm in zip(Q_list, norms, strict=True):
        Q.mul_(gmean / norm)


@torch.compile  # type: ignore[misc]
def norm_lower_bound_spd(A: torch.Tensor, k: int = 32, half_iters: int = 2, eps: float = 1e-8) -> torch.Tensor:
    r"""A cheap lower bound for the spectral norm of a symmetric positive definite matrix.


    Args:
        A: Tensor of shape :math:`(n, n)`, symmetric positive definite.
        k: Dimension of the subspace.
        half_iters: Half of the number of subspace iterations.
        eps: Small number for numerical stability.

    Returns:
        A scalar giving a lower bound on :math:`\\|A\\|_2`.
    """

    # Compute scaling factor from the largest diagonal entry to prevent overflow/underflow
    scale = torch.clamp(A.diagonal().amax(), min=eps)
    A = A / scale

    bound_unnormalized = _subspace_iteration_bound(A, k=k, half_iters=half_iters, eps=eps)

    return scale * bound_unnormalized


@torch.compile  # type: ignore[misc]
def norm_lower_bound_skew(A: torch.Tensor, k: int = 32, half_iters: int = 2, eps: float = 1e-8) -> torch.Tensor:
    """A cheap lower bound on the spectral norm (largest eigenvalue) of skew-symmetric matrix.


    Note: For skew-symmetric matrices, all diagonal entries are zero and :math:`A^T = -A`.
    From Xi-Lin Li.

    Args:
        A: Tensor of shape :math:`(n, n)`, skew-symmetric.
        k: Dimension of the subspace. Suggested values: 128 for bfloat16, 32 for float32, 4 for float64.
        half_iters: Half of the number of subspace iterations.
        eps: Small number for numerical stability.

    Returns:
        A scalar Tensor giving a lower bound on :math:`\\|A\\|_2`.

    """

    # Compute scaling factor from the max absolute value to prevent overflow/underflow
    scale = torch.clamp(A.abs().amax(), min=eps)
    A = A / scale

    bound_unnormalized = _subspace_iteration_bound(A, k=k, half_iters=half_iters, eps=eps)

    return scale * bound_unnormalized


@torch.compile  # type: ignore[misc]
def _subspace_iteration_bound(
    A: torch.Tensor,
    k: int = 32,
    half_iters: int = 2,
    eps: float = 1e-8,
) -> torch.Tensor:
    """A helper function for subspace iteration to estimate spectral norm bounds.

    Uses numerically stable subspace iteration with a random initialization that aligns with the
    largest row of A to approximate the dominant eigenspace. This is more robust than simple
    power iteration, especially for large matrices with very low rank. From Xi-Lin Li.

    The algorithm:
    1. Normalize :math:`A` by its largest absolute entry to avoid overflow.
    2. Find the row :math:`j` of :math:`A_{\\text{scaled}}` with the largest 2-norm.
    3. Initialize a :math:`k \\times n` subspace matrix :math:`V` with random vectors aligned to :math:`A[j]`.
    4. Perform subspace iteration for `half_iters` steps: :math:`V \\leftarrow V \\cdot A_{\\text{scaled}}`.
    5. Estimate the norm as the maximum 2-norm among the k vectors, then rescale.

    Args:
        A: Input matrix, already normalized by caller.
        k: Dimension of the subspace (number of random vectors).
        half_iters: Number of half-iterations (each applies A twice).
        eps: Smallest number for numerical stability.

    Returns:
        Maximum vector norm from the final subspace iteration (unnormalized).
    """

    # Initialize random subspace matrix V of shape (k, n)
    V = torch.randn(k, A.shape[1], dtype=A.dtype, device=A.device)

    # Find the row index with the largest 2-norm to initialize our subspace
    # This helps the algorithm converge faster to the dominant eigenspace
    dominant_row_idx = torch.argmax(torch.linalg.vector_norm(A, dim=1))
    # Rotate the random vectors to align with the dominant row A[dominant_row_idx]
    # This initialization trick makes the subspace iteration more robust for low-rank matrices
    dominant_row = A[dominant_row_idx]
    alignment = torch.sign(torch.sum(dominant_row * V, dim=1, keepdim=True))

    V = dominant_row + alignment * V

    # Perform subspace iteration
    for _ in range(half_iters):
        V = V @ A
        # Normalize each row of V to prevent exponential growth/decay
        V /= torch.linalg.vector_norm(V, dim=1, keepdim=True) + eps
        # Apply A again (V approximates the dominant eigenspace of A^2)
        V = V @ A

    # Return the maximum 2-norm among the k vectors
    return torch.amax(torch.linalg.vector_norm(V, dim=1))
