# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) ModelScope Contributors. All rights reserved.
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

"""Preserve Qwen4-Exp activation rounding at gated residual operation boundaries."""

import torch
import torch.nn.functional as F
from torch import Tensor


def _activate_reference(down: Tensor, streams: int) -> Tensor:
    return F.silu(down / streams)


def _mix_reference(up: Tensor, normalized: Tensor, streams: int) -> Tensor:
    width = up.shape[-1] // streams
    gates = up.sigmoid().unflatten(-1, (streams, width))
    return (gates * normalized.unflatten(-1, (streams, width))).mean(-2)


_activate_eager = torch.compiler.disable(_activate_reference)
_mix_eager = torch.compiler.disable(_mix_reference)


def activate_residual_gate(down: Tensor, streams: int) -> Tensor:
    """Keep the low-precision division store before SiLU, including under compile."""
    if down.dtype in (torch.float16, torch.bfloat16):
        return _activate_eager(down, streams)
    return _activate_reference(down, streams)


def mix_residual_streams(up: Tensor, normalized: Tensor, streams: int) -> Tensor:
    """Keep sigmoid and multiplication stores before the stream mean."""
    if up.dtype in (torch.float16, torch.bfloat16):
        return _mix_eager(up, normalized, streams)
    return _mix_reference(up, normalized, streams)
