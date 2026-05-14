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

from emerging_optimizers.orthogonalized_optimizers.muon_utils import newton_schulz


__all__ = ["spectral_hardcap", "spectral_clip"]


def spectral_clip(X: torch.Tensor, sigma_min: float = -1.0, sigma_max: float = 1.0) -> torch.Tensor:
    r"""Applies spectral clipping to the input tensor.

    From the idea that clipping can be written using the sign function. This idea can be extended to singular values of matrices
    using the matrix sign function, computed using Newton-Schulz iteration for efficiency.

    Based on https://leloykun.github.io/ponder/spectral-clipping/.

    Args:
        X: The input tensor.
        sigma_min: The minimum singular value.
        sigma_max: The maximum singular value.

    Returns:
        The spectral clipped tensor.
    """
    if needs_transpose := X.shape[0] > X.shape[1]:
        X = X.T
    OX = newton_schulz(X, steps=8, coefficient_type="polar_express")
    result = (sigma_min + sigma_max) * OX
    identity_matrix = torch.eye(X.shape[0], device=X.device, dtype=X.dtype)
    for s, sign in zip([sigma_min, sigma_max], [1, -1]):
        A = torch.addmm(s * identity_matrix, OX, X.T, beta=1.0, alpha=-1.0)
        B = torch.add(s * OX, X, alpha=-1)
        result = torch.addmm(result, newton_schulz(A, steps=8, coefficient_type="polar_express"), B, alpha=sign)
    result = result * 0.5

    if needs_transpose:
        result = result.T
    return result


def spectral_hardcap(X: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    r"""Spectral hardcap function clips singular values from above to be less than beta.

    Simplifies the spectral clipping function to just an upper bound, resulting in a hardcap.
    Based on https://leloykun.github.io/ponder/spectral-clipping/.

    Args:
        X: The input tensor.
        beta: The upper bound on the singular values.

    Returns:
        The spectral hardcapped tensor.

    """
    if needs_transpose := X.shape[0] > X.shape[1]:
        X = X.T
    OX = newton_schulz(X, steps=8, coefficient_type="polar_express")
    aX = torch.add(beta * OX, X, alpha=-1)
    result = torch.add(beta * OX, X)
    result = torch.addmm(
        result, aX, torch.mm(newton_schulz(aX, steps=8, coefficient_type="polar_express").T, OX), alpha=-1
    )
    result = result * 0.5
    if needs_transpose:
        result = result.T
    return result


def spectral_clipped_weight_decay(X: torch.Tensor, beta: float = 1.0, c: float = 0.5) -> torch.Tensor:
    r"""Applies weight decay to the input tensor while applying spectral hardcapping.

    This is the spectral version of Euclidean decoupled weight decay (Hanson & Pratt, 1988).

    Based on https://leloykun.github.io/ponder/spectral-clipping/.

    Args:
        X: The input tensor.
        beta: The upper bound on the singular values.
        c: The coefficient parameter.

    Returns:
        The spectral clipped weight decay tensor.
    """
    return torch.add((1 - c) * X, spectral_hardcap(X, beta), alpha=c)
