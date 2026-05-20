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

# pylint: disable=C0115,C0116,C0301


import logging

import torch
from diffusers.models.embeddings import TimestepEmbedding
from megatron.core import parallel_state


log = logging.getLogger(__name__)


# To be used from Common
class ParallelTimestepEmbedding(TimestepEmbedding):
    """
    ParallelTimestepEmbedding is a subclass of TimestepEmbedding that initializes
    the embedding layers with an optional random seed for syncronization.

    Args:
        in_channels (int): Number of input channels.
        time_embed_dim (int): Dimension of the time embedding.
        seed (int, optional): Random seed for initializing the embedding layers.
                              If None, no specific seed is set.

    Attributes:
        linear_1 (nn.Module): First linear layer for the embedding.
        linear_2 (nn.Module): Second linear layer for the embedding.

    Methods:
        __init__(in_channels, time_embed_dim, seed=None): Initializes the embedding layers.
    """

    def __init__(self, in_channels: int, time_embed_dim: int, seed=None):
        super().__init__(in_channels=in_channels, time_embed_dim=time_embed_dim)
        if seed is not None:
            with torch.random.fork_rng():
                torch.manual_seed(seed)
                self.linear_1.reset_parameters()
                self.linear_2.reset_parameters()

        if parallel_state.get_pipeline_model_parallel_world_size() > 1:
            setattr(self.linear_1.weight, "pipeline_parallel", True)
            setattr(self.linear_1.bias, "pipeline_parallel", True)
            setattr(self.linear_2.weight, "pipeline_parallel", True)
            setattr(self.linear_2.bias, "pipeline_parallel", True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the positional embeddings for the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, H, W, C).

        Returns:
            torch.Tensor: Positional embeddings of shape (B, T, H, W, C).
        """
        return super().forward(x.to(torch.bfloat16, non_blocking=True))
