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

import math
from typing import Any, Self, override

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1dFlatWeights(nn.Conv1d):
    """Conv1d with weights+bias stored in a single 2D tensor

    There are conv1d used in some LLM, in mamba mixer for example. Because the weight is not 2d, we cannot apply
    many of the emerging optimizers originally introduced for 2d weights of Linear layers without bias. Since
    convolution can be viewed as a matrix multiplication with im2col (either implicit or explicit), we can flatten
    the weight into a single 2D tensor and then apply the emerging optimizers to it.

    Bias is not commonly used in most LLM's anymore, but they are often included in this type of conv1d.
    Since bias is mathematically the 0 order term of the polynomial, we can combine weight and bias into a
    single 2D tensor.

    Arguments are the same as ::class:`torch.nn.Conv1d`.

    Note:
        This implementation potentially introduces a small overhead because of split weights can combining gradients
        of it. This should be trivial compared to computational cost of LLM training. If it becomes a concern, a
        kernel can be developed to eliminate the overhead.

    Note:
        Similar flattening logic can be applied to N-D convolution. But since we don't have use cases of them in LLM
        yet, they are not supported despite the __init__() function is generalized enough to support N-D convolution.

    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        assert self.padding_mode == "zeros", "Only zeros padding is supported"

        self.weight: nn.Parameter
        self.bias: torch.Tensor | None

        flat_weight_shape = [self.out_channels, math.prod(self.weight.shape[1:])]
        if self.bias is not None:
            flat_weight_shape[1] += 1
        flat_weight_buffer = torch.empty(flat_weight_shape, device=self.weight.device, dtype=self.weight.dtype)
        if self.bias is not None:
            flat_weight_buffer[..., :-1].copy_(self.weight.view(self.out_channels, -1))
            flat_weight_buffer[..., -1].copy_(self.bias)
            del self.bias
            self.has_bias = True
        else:
            flat_weight_buffer.copy_(self.weight.view(self.out_channels, -1))
            self.has_bias = False
        del self.weight

        self.weight = nn.Parameter(flat_weight_buffer)

    @classmethod
    def from_conv1d(cls, conv1d: nn.Conv1d) -> Self:
        conv1d_flat = cls(
            in_channels=conv1d.in_channels,
            out_channels=conv1d.out_channels,
            kernel_size=conv1d.kernel_size,
            bias=conv1d.bias is not None,
            stride=conv1d.stride,
            padding=conv1d.padding,
            dilation=conv1d.dilation,
            groups=conv1d.groups,
            padding_mode=conv1d.padding_mode,
            device=conv1d.weight.device,
            dtype=conv1d.weight.dtype,
        )

        if conv1d.bias is not None:
            conv1d_flat.weight.data[..., :-1].copy_(conv1d.weight.data.view(conv1d.out_channels, -1))
            conv1d_flat.weight.data[..., -1].copy_(conv1d.bias.data)
        else:
            conv1d_flat.weight.data.copy_(conv1d.weight.data.view(conv1d.out_channels, -1))
        return conv1d_flat

    @property
    def weight_shape(self) -> tuple[int, int, int]:
        return (self.out_channels, self.in_channels // self.groups, self.kernel_size[0])

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.has_bias:
            weight = self.weight[..., :-1].view(self.weight_shape)
            bias = self.weight[..., -1]
        else:
            weight = self.weight.view(self.weight_shape)
            bias = None

        return F.conv1d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)

    @override
    def extra_repr(self) -> str:
        s = f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}"
        if self.padding != (0,) * len(self.padding):
            s += f", padding={self.padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += f", dilation={self.dilation}"
        if self.output_padding != (0,) * len(self.output_padding):
            s += f", output_padding={self.output_padding}"
        if self.groups != 1:
            s += f", groups={self.groups}"
        if not self.has_bias:
            s += ", bias=False"
        if self.padding_mode != "zeros":
            s += f", padding_mode={self.padding_mode}"
        s += f", flattened_param_shape={tuple(self.weight.shape)}"
        return s
