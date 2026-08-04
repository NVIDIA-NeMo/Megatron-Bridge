# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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


def packing_length(item: Any) -> int:
    """
    Returns the bin-packing length of an item.

    This is the only diffusion-specific piece of sequence packing: the bin-packing
    algorithms themselves live in `megatron.bridge.data.packing.algorithms` and are
    shared with LLM offline SFT packing. Diffusion packs live `DiffusionSample`
    objects rather than integer lengths, so it computes each sample's length with
    this adapter and passes the results as `item_lengths` to the shared packer.

    Items are either plain sequence lengths or objects that report their length by
    adding with an int -- `DiffusionSample` does this, returning its padded query
    sequence length when one is set and its unpadded length otherwise. Going through
    `0 + item` keeps this module free of a hard dependency on the sample type, and
    matches the length the previous `sum(bin) + s` capacity check used.

    Args:
      item: A sequence length, or an object implementing `__radd__` against an int.

    Returns:
      The integer length used for bin-packing capacity accounting.
    """
    return 0 + item
