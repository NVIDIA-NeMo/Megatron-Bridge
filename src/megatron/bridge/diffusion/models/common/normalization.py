# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Common normalization modules for diffusion models."""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    A normalization technique that normalizes the input by its root mean square,
    then scales by a learnable weight parameter.

    Args:
        hidden_size: Size of the hidden dimension.
        config: Transformer configuration (unused, for compatibility with megatron build_module).
        eps: Small epsilon for numerical stability.
    """

    def __init__(self, hidden_size: int, config=None, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # Compute normalization and weight scaling in float32 for numerical stability,
        # then convert back to input dtype to preserve dtype throughout the model
        output = self._norm(x.float()) * self.weight
        return output.type_as(x)
