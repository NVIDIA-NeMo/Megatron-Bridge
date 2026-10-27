# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""Predefined losses and TP/CP-aware, token-chunked helpers (cross_entropy: phase B2; RL losses: phase C1)."""

from __future__ import annotations

from collections.abc import Callable

import torch


def chunked_vocab_apply(
    logits: torch.Tensor, fn: Callable[[torch.Tensor], torch.Tensor], *, chunk_tokens: int
) -> torch.Tensor:
    """Apply a full-vocabulary op per token chunk, preserving gradient flow, to bound peak memory."""
    raise NotImplementedError("chunked_vocab_apply is implemented in phase B2.")
