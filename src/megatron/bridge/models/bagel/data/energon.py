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

import io
import json
import random
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import torch
from megatron.energon import Cooker, Sample, TaskEncoder, basic_sample_keys
from PIL import Image
from transformers import PreTrainedTokenizerBase


@dataclass
class BagelT2IRawSample(Sample):
    """Raw BAGEL text-to-image sample."""

    image: bytes
    metadata: dict[str, object]


@dataclass
class BagelT2ISample(Sample):
    """BAGEL text-to-image sample ready for packing."""

    image_tensor_list: list[torch.Tensor]
    text_ids_list: list[list[int]]
    num_tokens: int
    sequence_plan: list[dict[str, object]]
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


def cook_bagel_t2i_sample(
    crude_sample: dict[str, object],
    *,
    tokenizer: PreTrainedTokenizerBase,
    transform: Callable[[Image.Image], torch.Tensor],
    image_stride: int,
) -> BagelT2ISample:
    """Apply BAGEL's T2I image, caption, and sequence-plan processing."""
    raw_sample = cook_bagel_t2i(crude_sample)
    with Image.open(io.BytesIO(raw_sample.image)) as image:
        if image.mode == "RGBA" or image.info.get("transparency") is not None:
            image = image.convert("RGBA")
            rgb_image = Image.new("RGB", image.size, (255, 255, 255))
            rgb_image.paste(image, mask=image.split()[3])
        else:
            rgb_image = image.convert("RGB")
    image_tensor = transform(rgb_image)

    captions = raw_sample.metadata["captions"]
    if not isinstance(captions, str):
        raise TypeError("BAGEL T2I captions metadata must contain text")
    caption_dict = json.loads(captions)
    caption_tokens = [tokenizer.encode(value) for value in caption_dict.values()]
    text_ids = random.choice(caption_tokens) if caption_tokens else tokenizer.encode(" ")
    sequence_plan = [
        {"type": "text", "enable_cfg": 1, "loss": 0, "special_token_loss": 0, "special_token_label": None},
        {"type": "vae_image", "enable_cfg": 0, "loss": 1, "special_token_loss": 0, "special_token_label": None},
    ]
    return BagelT2ISample(
        **basic_sample_keys(crude_sample),
        image_tensor_list=[image_tensor],
        text_ids_list=[text_ids],
        num_tokens=image_tensor.shape[1] * image_tensor.shape[2] // image_stride**2 + len(text_ids),
        sequence_plan=sequence_plan,
        metadata=raw_sample.metadata,
    )


class BagelT2IRawTaskEncoder(TaskEncoder):
    """Register the BAGEL T2I raw sample cooker."""

    cookers = [Cooker(cook_bagel_t2i)]


class BagelT2ITaskEncoder(TaskEncoder):
    """Register configured BAGEL T2I sample processing."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        transform: Callable[[Image.Image], torch.Tensor],
        image_stride: int,
    ) -> None:
        """Configure the official tokenizer and image transform."""
        self.cookers = [
            Cooker(
                partial(
                    cook_bagel_t2i_sample,
                    tokenizer=tokenizer,
                    transform=transform,
                    image_stride=image_stride,
                )
            )
        ]
