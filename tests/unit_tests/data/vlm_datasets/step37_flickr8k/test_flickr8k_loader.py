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

from megatron.bridge.data.vlm_datasets.step37_flickr8k.flickr8k_loader import (
    Flickr8kSample,
    Step37Flickr8kDataset,
)
from megatron.bridge.data.vlm_datasets.step37_flickr8k.template import IMAGE_PLACEHOLDER


pytestmark = pytest.mark.unit


def test_sample_is_frozen():
    sample = Flickr8kSample(image_path="/x.jpg", caption="a cat")
    assert sample.image_path == "/x.jpg"
    assert sample.caption == "a cat"
    with pytest.raises(Exception):
        sample.caption = "changed"  # frozen dataclass


def test_to_dialog_structure():
    sample = Flickr8kSample(image_path="/x.jpg", caption="a cat")
    dialog = Step37Flickr8kDataset._to_dialog(sample, prompt="Describe.")

    assert dialog["images"] == ["/x.jpg"]
    user, assistant = dialog["conversations"]
    assert user["role"] == "user"
    assert user["content"] == f"{IMAGE_PLACEHOLDER}\nDescribe."
    assert assistant["role"] == "assistant"
    assert assistant["content"] == "a cat"


def test_dataset_rejects_empty_samples():
    with pytest.raises(ValueError):
        Step37Flickr8kDataset([], template=lambda d: d, prompt="p")


def test_dataset_len_and_getitem_invoke_template():
    samples = [
        Flickr8kSample(image_path="/0.jpg", caption="c0"),
        Flickr8kSample(image_path="/1.jpg", caption="c1"),
    ]
    # Stub template: return the dialog it is handed so we can inspect it.
    dataset = Step37Flickr8kDataset(samples, template=lambda d: d, prompt="P")

    assert len(dataset) == 2
    item = dataset[1]
    assert item["images"] == ["/1.jpg"]
    assert item["conversations"][1]["content"] == "c1"
