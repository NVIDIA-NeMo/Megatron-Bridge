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

"""FLUX embedding layers for diffusion models."""

from typing import List

import torch
from diffusers.models.embeddings import Timesteps
from torch import Tensor, nn


def rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    """
    Compute rotary position embeddings.

    Different from the original ROPE used for flux.
    Megatron attention takes the outer product and calculates sin/cos inside,
    so we only need to get the freqs here in the shape of [seq, ..., dim].

    Args:
        pos: Position tensor.
        dim: Embedding dimension (must be even).
        theta: Base frequency.

    Returns:
        Rotary position embeddings of shape [..., dim//2].
    """
    assert dim % 2 == 0, "The dimension must be even."

    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)

    out = torch.einsum("...n,d->...nd", pos, omega)

    return out.float()


class EmbedND(nn.Module):
    """
    N-Dimensional Rotary Position Embedding generator.

    Generate Rope matrix with preset axes dimensions.

    Args:
        dim: Total embedding dimension.
        theta: Base frequency for rotary embeddings.
        axes_dim: List of dimensions for each axis.
    """

    def __init__(self, dim: int, theta: int, axes_dim: List[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """
        Compute N-dimensional rotary position embeddings.

        Args:
            ids: Position IDs tensor of shape [batch, seq, n_axes].

        Returns:
            Rotary embeddings tensor.
        """
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-1,
        )
        emb = emb.unsqueeze(1).permute(2, 0, 1, 3)
        return torch.stack([emb, emb], dim=-1).reshape(*emb.shape[:-1], -1)


class MLPEmbedder(nn.Module):
    """
    MLP embedder with two projection layers and SiLU activation.

    Args:
        in_dim: Input dimension.
        hidden_dim: Hidden/output dimension.
    """

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the MLP embedder."""
        return self.out_layer(self.silu(self.in_layer(x)))


class TimeStepEmbedder(nn.Module):
    """
    A neural network module that embeds timesteps for use in diffusion models.

    It projects the input timesteps to a higher-dimensional space and then embeds
    them using an MLP (Multilayer Perceptron). The projection and embedding provide
    a learned representation of the timestep that can be used in further computations.

    Args:
        embedding_dim: The dimensionality of the timestep embedding space.
        hidden_dim: The dimensionality of the hidden layer in the MLPEmbedder.
        flip_sin_to_cos: Whether to flip the sine and cosine components.
        downscale_freq_shift: A scaling factor for the frequency shift.
        scale: A scaling factor applied to the timestep projections.
        max_period: The maximum period for the sine and cosine functions.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        flip_sin_to_cos: bool = True,
        downscale_freq_shift: float = 0,
        scale: float = 1,
        max_period: int = 10000,
    ):
        super().__init__()

        self.time_proj = Timesteps(
            num_channels=embedding_dim,
            flip_sin_to_cos=flip_sin_to_cos,
            downscale_freq_shift=downscale_freq_shift,
            scale=scale,
        )
        self.time_embedder = MLPEmbedder(in_dim=embedding_dim, hidden_dim=hidden_dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Compute timestep embeddings.

        Args:
            timesteps: Input timestep tensor.

        Returns:
            Embedded timesteps tensor.
        """
        timesteps_proj = self.time_proj(timesteps)
        timesteps_emb = self.time_embedder(timesteps_proj)

        return timesteps_emb
