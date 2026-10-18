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

"""D1 readouts with explicit option order and frozen vocabulary semantics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class D1TokenReadoutConfig:
    """Select answer-token rows without untying the released vocabulary head.

    Args:
        hidden_size: Width of the normalized final backbone states.
        token_ids: Unique answer-alias vocabulary IDs in adapter row order.
    """

    hidden_size: int
    token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.hidden_size <= 0 or not self.token_ids:
            raise ValueError("hidden_size and token_ids must be non-empty and positive")
        if len(set(self.token_ids)) != len(self.token_ids) or min(self.token_ids) < 0:
            raise ValueError("token_ids must be unique non-negative vocabulary IDs")

    def build(self, vocabulary_weight: torch.Tensor) -> D1TokenReadout:
        """Build an adapter from the frozen released vocabulary weight.

        Args:
            vocabulary_weight: Tensor of shape [vocabulary, hidden_size].

        Returns:
            Zero-initialized row adapter, with detached base rows.
        """
        return D1TokenReadout(self, vocabulary_weight)


class D1TokenReadout(nn.Module):
    """Train additive answer-row updates while preserving input/output tying.

    This is an adaptation of the released readout, not an independent classifier.
    The zero-initialized adapter preserves the max-over-alias option contract.
    """

    def __init__(self, config: D1TokenReadoutConfig, vocabulary_weight: torch.Tensor) -> None:
        super().__init__()
        self.config = config
        if vocabulary_weight.ndim != 2 or vocabulary_weight.shape[1] != config.hidden_size:
            raise ValueError("vocabulary_weight must have shape [vocabulary, hidden_size]")
        if max(config.token_ids) >= vocabulary_weight.shape[0]:
            raise ValueError("answer token ID exceeds vocabulary size")
        ids = torch.tensor(config.token_ids, device=vocabulary_weight.device, dtype=torch.long)
        self.register_buffer("token_ids", ids)
        self.register_buffer("base_weight", vocabulary_weight.detach().index_select(0, ids).clone())
        self.delta_weight = nn.Parameter(torch.zeros(self.base_weight.shape, device=ids.device, dtype=torch.float32))

    def forward(self, hidden_states: torch.Tensor, alias_groups: tuple[tuple[int, ...], ...]) -> torch.Tensor:
        """Score declared options by their maximum answer-alias logit.

        Args:
            hidden_states: Tensor [..., hidden_size] at the answer position.
            alias_groups: Adapter row indices for each option, in target order.

        Returns:
            Float32 logits with shape [..., options], before option softmax.
        """
        if hidden_states.shape[-1] != self.config.hidden_size:
            raise ValueError("hidden_states must end in hidden_size")
        if not alias_groups or any(not group for group in alias_groups):
            raise ValueError("every option needs at least one answer alias")
        if any(min(group) < 0 or max(group) >= len(self.config.token_ids) for group in alias_groups):
            raise ValueError("alias_groups must index adapter rows")
        logits = F.linear(hidden_states.to(self.base_weight.dtype), self.base_weight).float()
        logits = logits + F.linear(hidden_states.float(), self.delta_weight.float())
        return torch.stack([logits[..., list(group)].amax(dim=-1) for group in alias_groups], dim=-1)
