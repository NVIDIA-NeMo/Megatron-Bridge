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
"""Accumulation window: update-wide counts, auxiliary-loss policy, normalization and single finalization (phase B2)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AccumulationWindow:
    """FP32 gradient accumulation state spanning several ``forward_backward`` calls."""

    num_calls: int = 0
    primary_count: float = 0.0  # D: valid tokens or Datums, per NormalizationPolicy
    mtp_count: float = 0.0  # M: valid MTP targets (detached-MTP mode only)

    def is_open(self) -> bool:
        """Whether gradients from at least one call are pending."""
        return self.num_calls > 0
