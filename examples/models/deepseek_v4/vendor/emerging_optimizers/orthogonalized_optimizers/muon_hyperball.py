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

from typing import Any, override

import torch

from emerging_optimizers import registry
from emerging_optimizers.orthogonalized_optimizers import muon


__all__ = ["MuonHyperball"]


@registry.register_optimizer("muon_hyperball")
class MuonHyperball(muon.Muon):
    """Muon optimizer with hyperball-style norm-preserving weight updates.

    This optimizer extends Muon by performing gradient descent on the sphere manifold
    while preserving the weight norm. The update rule is:

    .. math::

        W_{t+1} = R \\cdot \\text{normalize}(W_t - \\text{lr} \\cdot R \\cdot \\text{normalize}(\\text{update}))

    where :math:`R` is the Frobenius norm of :math:`W_t` (or a user-specified radius). This keeps
    the weight matrix at constant scale while updating.

    Warning:
        This optimizer is experimental and may change in future versions.

    See :class:`~emerging_optimizers.orthogonalized_optimizers.muon.Muon` for full documentation
    of the base Muon optimizer.


    Args:
        *args: Arguments passed to Muon.
        hyperball_eps: Epsilon for numerical stability in normalization.
            Default: ``1e-8``.
        hyperball_radius: Fixed radius for the hyperball. If ``None`` (default),
            uses each parameter's initial Frobenius norm as its radius. If specified, all
            parameters will be rescaled to have this radius at initialization.
        **kwargs: Keyword arguments passed to Muon.

    """

    def __init__(
        self,
        *args: Any,
        hyperball_eps: float = 1e-8,
        hyperball_radius: float | None = None,
        **kwargs: Any,
    ) -> None:
        self.hyperball_eps = hyperball_eps
        self.hyperball_radius = hyperball_radius
        super().__init__(*args, **kwargs)

        # Validate and optionally rescale parameters based on hyperball_radius.
        with torch.no_grad():
            for group in self.param_groups:
                for p in group["params"]:
                    p_norm = p.norm()
                    # Validate that parameter has non-zero norm.
                    if p_norm.item() == 0:
                        raise ValueError(
                            "MuonHyperball requires all parameters to have non-zero norm. Found parameter with zero norm."
                        )
                    # Rescale parameter to have the specified radius if provided.
                    if self.hyperball_radius is not None:
                        p.mul_(self.hyperball_radius / p_norm.clamp_min(self.hyperball_eps))

    @override
    def pre_weight_update_fn_inplace(self, p: torch.Tensor, update: torch.Tensor) -> None:
        """Store the original weight norm and normalize the update using Frobenius norm.

        Args:
            p: The parameter tensor.
            update: The orthogonalized gradient tensor.
        """
        # Use user-specified radius or compute R = ||W_t||_F (Frobenius norm)
        R = self.hyperball_radius if self.hyperball_radius is not None else p.norm().item()
        self.state[p]["hyperball_R"] = R

        # Normalize the update in-place and scale by R
        # This modifies update to be: R * normalize(update) using Frobenius norm.
        update_norm = update.norm().clamp_min(self.hyperball_eps)
        update.mul_(R / update_norm)

    @override
    def post_weight_update_fn_inplace(self, p: torch.Tensor) -> None:
        """Normalize the updated weights and scale back to original norm using Frobenius norm.

        Args:
            p: The parameter tensor (already updated).
        """
        # Retrieve R from per-parameter state
        R = self.state[p]["hyperball_R"]

        # Normalize the result and scale back by R: p = R * (p / ||p||_F) using Frobenius norm.
        p_norm = p.norm().clamp_min(self.hyperball_eps)
        p.mul_(R / p_norm)
