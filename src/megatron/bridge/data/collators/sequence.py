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

"""Explicit collator policy for padded or in-batch-packed sequence output."""

from collections.abc import Mapping, MutableMapping
from typing import Any

from megatron.bridge.data.collators.sequence_padding import pad_or_truncate_sequence_batch
from megatron.bridge.data.datasets.utils import IGNORE_INDEX
from megatron.bridge.data.packing.in_batch import pack_right_padded_sequence_batch_to_mcore_thd


def prepare_sequence_batch(
    batch: MutableMapping[str, Any],
    *,
    sequence_length: int | None,
    pad_to_max_length: bool = False,
    pad_to_multiple_of: int = 128,
    enable_in_batch_packing: bool = False,
    in_batch_packing_pad_to_multiple_of: int = 1,
    pad_token_id: int = 0,
    ignore_index: int = IGNORE_INDEX,
    sequence_tensor_pad_values: Mapping[str, int | float] | None = None,
) -> None:
    """Apply the collator's explicit padded or in-batch-packed output policy."""
    if enable_in_batch_packing:
        pack_right_padded_sequence_batch_to_mcore_thd(
            batch,
            sequence_length=sequence_length,
            pad_token_id=pad_token_id,
            ignore_index=ignore_index,
            pad_to_multiple_of=in_batch_packing_pad_to_multiple_of,
            sequence_tensor_pad_values=sequence_tensor_pad_values,
        )
        return
    pad_or_truncate_sequence_batch(
        batch,
        sequence_length=sequence_length,
        pad_to_max_length=pad_to_max_length,
        pad_to_multiple_of=pad_to_multiple_of,
        pad_token_id=pad_token_id,
        ignore_index=ignore_index,
        sequence_tensor_pad_values=sequence_tensor_pad_values,
    )
