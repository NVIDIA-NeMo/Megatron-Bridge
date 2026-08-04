# Copyright (c) 2024-2026, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Any

from megatron.bridge.diffusion.data.common.sequence_packing_utils import packing_length


class _Sample:
    """Stand-in for DiffusionSample's bin-packing contract.

    The diffusion task encoder packs live sample objects, keying capacity on the padded
    query sequence length when one is set and the unpadded length otherwise (this mirrors
    DiffusionSample.__radd__). This reproduces just that contract, without pulling in
    torch or energon.
    """

    def __init__(self, seq_len_q: int, *, seq_len_q_padded: int | None = None):
        self.seq_len_q = seq_len_q
        self.seq_len_q_padded = seq_len_q_padded

    def __radd__(self, other: Any) -> int:
        if isinstance(other, int):
            return self.length + other
        raise NotImplementedError

    @property
    def length(self) -> int:
        return self.seq_len_q_padded if self.seq_len_q_padded is not None else self.seq_len_q


def test_packing_length_handles_plain_ints():
    assert packing_length(42) == 42
    assert packing_length(0) == 0


def test_packing_length_prefers_padded_query_length():
    """packing_length must match DiffusionSample: the padded length wins when it is set."""
    assert packing_length(_Sample(100)) == 100
    assert packing_length(_Sample(100, seq_len_q_padded=128)) == 128
