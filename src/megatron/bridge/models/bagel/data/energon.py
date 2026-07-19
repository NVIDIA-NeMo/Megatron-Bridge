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

import json
from dataclasses import dataclass

from megatron.energon import Cooker, Sample, TaskEncoder, basic_sample_keys


@dataclass
class BagelT2IRawSample(Sample):
    """Raw BAGEL text-to-image sample."""

    image: bytes
    metadata: dict[str, object]


def cook_bagel_t2i(crude_sample: dict[str, object]) -> BagelT2IRawSample:
    """Cook raw WebDataset fields without decoding image or caption data."""
    missing = {"image", "json"}.difference(crude_sample)
    if missing:
        raise ValueError(f"missing required WebDataset fields: {sorted(missing)}")

    image = crude_sample["image"]
    if not isinstance(image, bytes):
        raise TypeError("WebDataset image field must contain bytes")
    json_value = crude_sample["json"]
    if not isinstance(json_value, (bytes, bytearray, str)):
        raise TypeError("WebDataset json field must contain bytes or text")
    metadata = json.loads(json_value)
    if not isinstance(metadata, dict):
        raise TypeError("WebDataset json field must contain an object")

    return BagelT2IRawSample(
        **basic_sample_keys(crude_sample),
        image=image,
        metadata=metadata,
    )


class BagelT2IRawTaskEncoder(TaskEncoder):
    """Register the BAGEL T2I raw sample cooker."""

    cookers = [Cooker(cook_bagel_t2i)]
