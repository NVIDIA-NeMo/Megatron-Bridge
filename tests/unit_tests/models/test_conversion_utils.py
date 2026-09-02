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

import pytest

from megatron.bridge.models.conversion.utils import mcore_to_hf_window_size, remove_non_pickleables


@pytest.mark.parametrize(
    ("window_size", "expected"),
    [
        (None, None),
        (2048, 2048),
        ((2047, 0), 2048),
        ([2047, 0], 2048),
    ],
)
def test_mcore_to_hf_window_size(window_size, expected):
    assert mcore_to_hf_window_size(window_size) == expected


def test_mcore_to_hf_window_size_rejects_malformed_pair():
    with pytest.raises(ValueError, match="two-element MCore window"):
        mcore_to_hf_window_size([2047])


def test_remove_non_pickleables_reads_raw_config_attributes():
    class HeterogeneousConfig:
        def __init__(self):
            self.num_key_value_heads = 8
            self.callback = lambda: None

        def __getattribute__(self, name):
            if name == "num_key_value_heads":
                raise RuntimeError("attribute must be read from the per-layer config")
            return super().__getattribute__(name)

    original = HeterogeneousConfig()

    cleaned = remove_non_pickleables(original)

    assert vars(cleaned)["num_key_value_heads"] == 8
    assert cleaned.callback is None
    assert vars(original)["callback"] is not None
