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
"""Accumulation window: update-wide counts for the gradients accumulated across ``forward_backward`` calls.

Gradients themselves live in the DDP FP32 buffers; this object only tracks what is needed to normalize
them once at ``optim_step``. Counts are global (already reduced over DP x CP).
"""

from __future__ import annotations

from dataclasses import dataclass

from megatron.bridge.engine.datum import NormalizationPolicy


@dataclass
class AccumulationWindow:
    """Open from the first ``forward_backward`` after a step until ``optim_step`` / ``zero_grad``."""

    num_calls: int = 0
    num_loss_tokens: int = 0  # D: active target positions over the update
    num_samples: int = 0  # N: Datums over the update
    loss_sum: float = 0.0  # sum of the loss methods' returns over the update (reporting only)
    # TODO(detached-mtp): with mtp_detach_heads=True and mtp_target_scope="all_valid", the MTP-only
    # parameters get the MTP loss summed over all non-padded positions but are still divided by
    # num_loss_tokens, so their effective step grows with the non-padded / active-token ratio. Track
    # the non-padded count over the update and divide those parameters by it instead. Under
    # mtp_target_scope="primary" MCore's own rescale already makes 1/num_loss_tokens a per-target mean.

    @property
    def is_open(self) -> bool:
        """Whether gradients from at least one call are pending."""
        return self.num_calls > 0

    def add(self, *, num_loss_tokens: int, num_samples: int, loss_sum: float) -> None:
        """Record one ``forward_backward`` call."""
        self.num_calls += 1
        self.num_loss_tokens += int(num_loss_tokens)
        self.num_samples += int(num_samples)
        self.loss_sum += float(loss_sum)

    def denominator(self, policy: NormalizationPolicy) -> float | None:
        """Update-wide divisor for the primary gradients, or ``None`` when normalization is bypassed."""
        if policy == NormalizationPolicy.TOKEN_MEAN:
            return float(self.num_loss_tokens)
        if policy == NormalizationPolicy.SAMPLE_MEAN:
            return float(self.num_samples)
        if policy == NormalizationPolicy.NONE:
            return None
        raise ValueError(f"Unknown normalization policy {policy!r}.")

    def reset(self) -> None:
        """Close the window."""
        self.num_calls = 0
        self.num_loss_tokens = 0
        self.num_samples = 0
        self.loss_sum = 0.0
