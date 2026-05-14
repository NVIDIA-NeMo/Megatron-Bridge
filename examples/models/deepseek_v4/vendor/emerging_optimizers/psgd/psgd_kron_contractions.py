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
    "partial_contraction",
    "apply_kronecker_factors",
    "apply_preconditioner",
]


def partial_contraction(G1: torch.Tensor, G2: torch.Tensor, axis: int) -> torch.Tensor:
    """Compute the partial contraction of G1 and G2 along axis `axis`.
    This is the contraction of the two tensors, but with all axes except `axis` contracted.

    Args:
        G1: Tensor of shape (d_0, d_1, ..., d_{axis-1}, d_{axis}, d_{axis+1}, ..., d_N)
        G2: Tensor of shape (d_0, d_1, ..., d_{axis-1}, d_{axis}, d_{axis+1}, ..., d_N)
        axis: int, the axis to contract along

    Returns:
        Tensor of shape (d_{axis}, d_{axis})
    """
    # dims_to_contract = all dims except `axis`
    dims_to_contract = [i for i in range(G1.dim()) if i != axis]
    # contraction is symmetric and has shape (d_{axis}, d_{axis})
    return torch.tensordot(G1, G2, dims=(dims_to_contract, dims_to_contract))


@torch.compile  # type: ignore[misc]
def apply_kronecker_factors(Q_list: list[torch.Tensor], X: torch.Tensor) -> torch.Tensor:
    """Apply all Kronecker factors once to tensor :math:`X`, each to its corresponding dimension.

    This applies each :math:`Q` factor once, for example in 2D case: :math:`Q_1 X Q_2^T`.

    Args:
        Q_list: List of :math:`Q` (the upper-triangular Kronecker factors), each of shape `(d_i, d_i)` or `(d_i,)`.
        X: Tensor of shape `(d_0, d_1, ..., d_N)`.

    Returns:
        Tensor of shape `(d_0, d_1, ..., d_N)`.
    """
    if len(Q_list) != X.dim():
        raise ValueError(
            f"Number of Kronecker factors {len(Q_list)} must match the number of dimensions of X {X.dim()}"
        )

    Y = X
    for i in range(len(Q_list)):
        Y = _apply_single_kronecker_factor(Q_list, Y, i)
    return Y


@torch.compile  # type: ignore[misc]
def apply_preconditioner(Q_list: list[torch.Tensor], X: torch.Tensor) -> torch.Tensor:
    """Apply the full PSGD preconditioner to X.

    This is the full Kronecker product of PSGD's kronecker factors Q^T Q, applied to X.

    :math:`P X = (Q_1^T Q_1) X (Q_2^T Q_2)`

    This applies each factor followed by its transpose for the full preconditioner effect.

    Args:
        Q_list: List of :math:`Q` (the Kronecker factors), each of shape `(d_i, d_i)` or `(d_i,)`.
        X: Tensor of shape `(d_0, d_1, ..., d_N)`.

    Returns:
        Tensor of shape `(d_0, d_1, ..., d_N)`.
    """
    # Apply Q first, then Q.T to get Q^T @ Q
    Px = apply_kronecker_factors(Q_list, X)
    Px = apply_kronecker_factors([q if q.dim() == 1 else q.T for q in Q_list], Px)
    return Px


def _dim_n_mul_and_permute(X: torch.Tensor, M: torch.Tensor, contract_dim: int) -> torch.Tensor:
    """Multiply tensor X along axis `contract_dim` by 2D matrix M.

    Helper function for `_apply_single_kronecker_factor`.
    If M is (d_out, d_in) we contract M’s second index with X’s `contract_dim` index.
    `torch.tensordot` is used to contract the two tensors, and then the result is permuted to move the new axis 0 to position `contract_dim`.
    Returns a new tensor of the same rank, but with size[contract_dim] replaced by d_out.
    Note that d_{contract_dim} == d_in.

    Args:
        X: Tensor of shape (d_0, d_1, ..., d_{contract_dim-1}, d_{contract_dim}, d_{contract_dim+1}, ..., d_N)
        M: Tensor of shape (d_out, d_in)
        contract_dim: int, the dimension to contract with M, with d_{contract_dim} == d_in

    Returns:
        Tensor of shape (d_0, d_1, ..., d_{contract_dim-1}, d_out, d_{contract_dim+1}, ..., d_N)

    Examples
    --------
    >>> X = torch.randn(2, 3, 6)
    >>> M = torch.randn(5, 6)
    >>> contract_dim = 2
    >>> result = _dim_n_mul_and_permute(X, M, contract_dim)
    >>> print(result.shape)
    torch.Size([2, 3, 5])

    """
    if X.shape[contract_dim] != M.shape[1]:
        raise ValueError(
            f"Shape mismatch: X.shape[{contract_dim}] = {X.shape[contract_dim]}, M.shape[1] = {M.shape[1]}"
        )
    # Contract M's 2nd dim (idx=1) with X's `contract_dim` dim
    Y = torch.tensordot(M, X, dims=([1], [contract_dim]))
    # Y now has shape (d_out, d_0, …, d_{contract_dim-1}, d_{contract_dim+1}, …).
    # We want to move that new axis 0 back to position `contract_dim`, due to `torch.tensordot`.
    nd = X.dim()
    perm = list(range(1, contract_dim + 1)) + [0] + list(range(contract_dim + 1, nd))
    return Y.permute(perm)


@torch.compile  # type: ignore[misc]
def _apply_single_kronecker_factor(Q_list: list[torch.Tensor], X: torch.Tensor, axis: int) -> torch.Tensor:
    """Apply a single Kronecker factor Q to X at dimension `axis`. Helper function for apply_kronecker_factors.

    If Q is a vector, we multiply X by Q.
    If Q is a matrix, we contract Q's second index with X's `axis` index.

    Args:
        Q_list: List of Q (e.g. the Kronecker factors).
        X: Tensor of shape (d_0, d_1, ..., d_{axis-1}, d_{axis+1}, ..., d_N)
    """
    Q = Q_list[axis]
    if Q.dim() == 1:
        shape = [1] * X.dim()
        shape[axis] = Q.size(0)
        return X * Q.view(shape)

    return _dim_n_mul_and_permute(X, Q, contract_dim=axis)
