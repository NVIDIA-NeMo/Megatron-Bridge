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

"""Identity collate for datasets consumed by Megatron-Core's online packing scheduler."""

from __future__ import annotations

from typing import Any


def identity_collate(samples: list[Any]) -> list[Any]:
    """Return the samples as a list instead of stacking them.

    Megatron-Core's sequence-packing scheduler pulls ``micro_batch_size`` samples
    per ``next()`` and packs them itself; the default collate would stack the
    variable-length tensors and destroy the per-sample lengths it needs.
    """
    return list(samples)
