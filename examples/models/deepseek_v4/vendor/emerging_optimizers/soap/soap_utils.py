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
from typing import TypeAlias

import torch

from emerging_optimizers.utils import eig as eig_utils


TensorList: TypeAlias = list[torch.Tensor]


__all__ = [
    "all_eigenbases_met_criteria",
    "get_eigenbasis_eigh",
    "get_eigenbasis_qr",
]


def all_eigenbases_met_criteria(
    kronecker_factor_list: TensorList,
    eigenbasis_list: TensorList,
    adaptive_update_tolerance: float = 1e-7,
) -> bool:
    """Checks if every eigenbasis in the list meets the adaptive update tolerance criteria.

    Args:
        kronecker_factor_list: List of Kronecker factor matrices
        eigenbasis_list: List of orthonormal eigenbases of the kronecker factor matrices
        adaptive_update_tolerance: Tolerance threshold for the normalized diagonal component of approximated
            eigenvalue matrix.

    Returns:
        True if all eigenbases meet the criteria (no update needed), False otherwise.
    """
    for kronecker_factor, eigenbasis in zip(kronecker_factor_list, eigenbasis_list, strict=True):
        approx_eigvals = eig_utils.conjugate(kronecker_factor, eigenbasis, diag=True)
        if not eig_utils.met_approx_eigvals_criteria(kronecker_factor, approx_eigvals, adaptive_update_tolerance):
            return False

    return True


def get_eigenbasis_eigh(
    kronecker_factor_list: TensorList,
    eps: float | None = None,
) -> TensorList:
    """Computes the eigenbases of the preconditioner using torch.linalg.eigh decomposition.

    Args:
        kronecker_factor_list: Matrix List to compute eigenbases of
        eps: Small offset for numerical stability.

    Returns:
        List of orthonormal kronecker factor eigenbases matrices

    Example:
        .. code-block:: python

            # Create sample Kronecker factors (symmetric positive definite matrices)
            k_factor1 = torch.randn(4, 4)
            k_factor1 = k_factor1 @ k_factor1.T  # Make symmetric positive definite
            k_factor2 = torch.randn(5, 5)
            k_factor2 = k_factor2 @ k_factor2.T  # Make symmetric positive definite

            # Get orthogonal matrices for these factors
            ortho_matrices = get_eigenbasis_eigh([k_factor1, k_factor2])
            # ortho_matrices[0] has shape [4, 4] and ortho_matrices[1] has shape [5, 5]
    """
    updated_eigenbasis_list: TensorList = []

    for kronecker_factor in kronecker_factor_list:
        _, Q = eig_utils.eigh_with_fallback(kronecker_factor, force_double=False, eps=eps)
        updated_eigenbasis_list.append(Q)

    return updated_eigenbasis_list


def get_eigenbasis_qr(
    kronecker_factor_list: TensorList,
    eigenbasis_list: TensorList,
    exp_avg_sq: torch.Tensor,
    power_iter_steps: int = 1,
) -> tuple[TensorList, torch.Tensor]:
    """Updates the eigenbases of the preconditioner using power iteration and QR.

    Computes using multiple rounds of power iteration followed by QR decomposition (orthogonal iteration).

    Args:
        kronecker_factor_list: List containing preconditioner (:math:`GG^T` and :math:`G^TG`)
        eigenbasis_list: List containing eigenbases (:math:`Q_L` and :math:`Q_R`)
        exp_avg_sq: inner adam second moment (exp_avg_sq).
        power_iter_steps: Number of power iteration steps to perform before QR decomposition.
            More steps can lead to better convergence but increased computation time.

    Returns:
        Tuple of updated list of orthonormal kronecker factor eigenbases matrices and updated (sorted) inner
            Adam's second moment.

    Example:
        .. code-block:: python

            # Create sample Kronecker factors (symmetric positive definite matrices)
            n, m = 10, 20
            k_factor1 = torch.randn(n, n)
            k_factor1 = k_factor1 @ k_factor1.T  # Make symmetric positive definite
            k_factor2 = torch.randn(m, m)
            k_factor2 = k_factor2 @ k_factor2.T  # Make symmetric positive definite

            # Get orthogonal matrices for these kronecker factors
            kronecker_factor_list = [k_factor1, k_factor2]
            eigenbasis_list = get_eigenbasis_eigh(kronecker_factor_list)

            # Perturb the kronecker factor matrices, simulating the effect of gradient updates
            perturbation = 1e-2*torch.randn(n, m)
            perturbed_kronecker_factor_list = [None, None]
            perturbed_kronecker_factor_list[0] = k_factor1 + perturbation@perturbation.T
            perturbed_kronecker_factor_list[1] = k_factor2 + perturbation.T@perturbation

            # Initialize exp_avg_sq tensor
            exp_avg_sq = torch.randn(n, m).abs()

            # Refine the orthogonal matrices using QR
            updated_ortho_matrices, updated_exp_avg_sq = get_eigenbasis_qr(
                perturbed_kronecker_factor_list,
                eigenbasis_list,
                exp_avg_sq
            )
    """
    updated_eigenbasis_list: TensorList = []
    for ind, (kronecker_factor, eigenbasis) in enumerate(zip(kronecker_factor_list, eigenbasis_list, strict=True)):
        approx_eigvals = eig_utils.conjugate(kronecker_factor, eigenbasis, diag=True)
        Q, exp_avg_sq = eig_utils.orthogonal_iteration(
            approx_eigvals=approx_eigvals,
            kronecker_factor=kronecker_factor,
            eigenbasis=eigenbasis,
            ind=ind,
            exp_avg_sq=exp_avg_sq,
            power_iter_steps=power_iter_steps,
        )
        updated_eigenbasis_list.append(Q)

    return updated_eigenbasis_list, exp_avg_sq
