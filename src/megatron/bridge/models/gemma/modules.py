# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import torch


def extend_instance(obj, mixin):
    """Apply mixins to a class instance after creation"""
    base_cls = obj.__class__
    base_cls_name = obj.__class__.__name__
    obj.__class__ = type(
        base_cls_name, (mixin, base_cls), {}
    )  # mixin needs to go first for our forward() logic to work


class EmbeddingScalingMixin(torch.nn.Module):
    """
    A mixin class for scaling embeddings in Megatron GPT.
    The scaling is applied only if the configuration (accessible via `self.config`)
    includes `apply_embedding_scaling` set to True.
    """

    def forward(self, **kwargs):
        """
        Forward pass that scales the output embeddings from the `forward` method of
        the superclass by the square root of the hidden size specified in the configuration.
        """
        embeddings = super().forward(**kwargs)
        return embeddings * torch.tensor(self.config.hidden_size**0.5, dtype=embeddings.dtype)
