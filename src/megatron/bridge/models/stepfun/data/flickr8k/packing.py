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

"""Model-owned greedy non-truncation packing.

Walks the sample sizes in order and greedily fills each pack up to
``max_len`` without ever truncating a sample. Do NOT modify the loop
arithmetic or the ``flush()`` semantics — the exact sequence of "drop" vs
"extend" decisions determines the contents of every pack, and any
reordering would shift the entire downstream packing layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class PackingResult:
    """Result metadata from greedy sample packing."""

    num_packed_samples: int
    num_droped: int
    packed_sample_ranges: list[tuple[int, int]]
    """Offset & Num of samples packed for each packed-sample, e.g. [(0, 2), (3, 2), ...]"""


def pack(sizes: list[int], max_len: int, oversize_policy: Literal["drop", "extend"]) -> PackingResult:
    """Pack ordered sample lengths into contiguous groups without truncation.

    Args:
        sizes: Token lengths for the samples to pack.
        max_len: Maximum packed sequence length.
        oversize_policy: Whether to drop oversize samples or keep them in extended packs.

    Returns:
        Metadata describing the packed sample ranges and dropped sample count.
    """
    total = len(sizes)
    packed_sample_ranges: list[tuple[int, int]] = []
    packed_ids, packed_size, consumed, droped = [], 0, 0, 0

    def flush() -> None:
        nonlocal packed_ids, packed_size, consumed
        if packed_ids:
            packed_sample_ranges.append((packed_ids[0], len(packed_ids)))
            packed_ids = []
            packed_size = 0

    while consumed < total:
        sample_len = sizes[consumed]

        if packed_size + sample_len <= max_len:
            packed_ids.append(consumed)
            packed_size += sample_len
            if packed_size == max_len:
                flush()
        else:  # too long with new sample
            flush()
            if sample_len <= max_len:
                consumed -= 1  # putback
            else:  # oversize
                if oversize_policy == "extend":
                    packed_ids.append(consumed)
                    packed_size = sample_len
                    flush()
                else:
                    droped += 1  # drop

        consumed += 1

    flush()
    return PackingResult(
        num_packed_samples=len(packed_sample_ranges),
        num_droped=droped,
        packed_sample_ranges=packed_sample_ranges,
    )
