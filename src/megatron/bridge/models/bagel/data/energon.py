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


@dataclass
class BagelEditingSample(Sample):
    """BAGEL Editing sample ready for packing."""

    image_tensor_list: list[torch.Tensor]
    text_ids_list: list[list[int]]
    num_tokens: int
    sequence_plan: list[dict[str, object]]
    metadata: dict[str, object]


@dataclass
class BagelVLMSample(Sample):
    """BAGEL vision-language sample ready for packing."""

    image_tensor_list: list[torch.Tensor]
    text_ids_list: list[list[int]]
    num_tokens: int
    sequence_plan: list[dict[str, object]]
    metadata: dict[str, object]


def _load_metadata(crude_sample: dict[str, object]) -> dict[str, object]:
    """Load one WebDataset JSON member."""
    json_value = crude_sample["json"]
    if not isinstance(json_value, (bytes, bytearray, str)):
        raise TypeError("WebDataset json field must contain bytes or text")
    metadata = json.loads(json_value)
    if not isinstance(metadata, dict):
        raise TypeError("WebDataset json field must contain an object")
    return metadata


def _decode_rgb(image_bytes: bytes) -> Image.Image:
    """Decode image bytes with BAGEL's transparent-image handling."""
    with Image.open(io.BytesIO(image_bytes)) as image:
        if image.mode == "RGBA" or image.info.get("transparency") is not None:
            image = image.convert("RGBA")
            rgb_image = Image.new("RGB", image.size, (255, 255, 255))
            rgb_image.paste(image, mask=image.split()[3])
            return rgb_image
        return image.convert("RGB")


def cook_bagel_t2i(crude_sample: dict[str, object]) -> BagelT2IRawSample:
    """Cook raw WebDataset fields without decoding image or caption data."""
    missing = {"image", "json"}.difference(crude_sample)
    if missing:
        raise ValueError(f"missing required WebDataset fields: {sorted(missing)}")

    image = crude_sample["image"]
    if not isinstance(image, bytes):
        raise TypeError("WebDataset image field must contain bytes")

    return BagelT2IRawSample(
        **basic_sample_keys(crude_sample),
        image=image,
        metadata=_load_metadata(crude_sample),
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
    image_tensor = transform(_decode_rgb(raw_sample.image))

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


def cook_bagel_editing_sample(
    crude_sample: dict[str, object],
    *,
    tokenizer: PreTrainedTokenizerBase,
    transform: Callable[[Image.Image], torch.Tensor],
    vit_transform: Callable[[Image.Image], torch.Tensor],
    image_stride: int,
    vit_image_stride: int,
) -> BagelEditingSample:
    """Apply BAGEL's Editing path choice, transforms, tokens, and sequence plan."""
    metadata = _load_metadata(crude_sample)
    image_count = metadata["image_count"]
    instruction_list = metadata["instruction_list"]
    if not isinstance(image_count, int) or not isinstance(instruction_list, list):
        raise TypeError("BAGEL Editing metadata has invalid image or instruction fields")

    images = []
    for image_index in range(image_count):
        image_bytes = crude_sample[f"image{image_index}"]
        if not isinstance(image_bytes, bytes):
            raise TypeError("BAGEL Editing image fields must contain bytes")
        images.append(_decode_rgb(image_bytes))

    image_tensor_list: list[torch.Tensor] = []
    text_ids_list: list[list[int]] = []
    sequence_plan: list[dict[str, object]] = []
    num_tokens = 0

    def add_text(text: str) -> None:
        nonlocal num_tokens
        text_ids = tokenizer.encode(text)
        text_ids_list.append(text_ids)
        num_tokens += len(text_ids)
        sequence_plan.append(
            {"type": "text", "enable_cfg": 1, "loss": 0, "special_token_loss": 0, "special_token_label": None}
        )

    def add_image(image: Image.Image, *, need_loss: bool, need_vae: bool, need_vit: bool) -> None:
        nonlocal num_tokens
        if need_loss:
            sequence_plan.append(
                {
                    "type": "vae_image",
                    "enable_cfg": 0,
                    "loss": 1,
                    "special_token_loss": 0,
                    "special_token_label": None,
                }
            )
            image_tensor = transform(image)
            num_tokens += image_tensor.shape[1] * image_tensor.shape[2] // image_stride**2
            image_tensor_list.append(image_tensor)
        if need_vae:
            sequence_plan.append(
                {
                    "type": "vae_image",
                    "enable_cfg": 1,
                    "loss": 0,
                    "special_token_loss": 0,
                    "special_token_label": None,
                }
            )
            image_tensor = transform(image)
            num_tokens += image_tensor.shape[1] * image_tensor.shape[2] // image_stride**2
            image_tensor_list.append(image_tensor.clone())
        if need_vit:
            sequence_plan.append(
                {
                    "type": "vit_image",
                    "enable_cfg": 1,
                    "loss": 0,
                    "special_token_loss": 0,
                    "special_token_label": None,
                }
            )
            image_tensor = vit_transform(image)
            num_tokens += image_tensor.shape[1] * image_tensor.shape[2] // vit_image_stride**2
            image_tensor_list.append(image_tensor)

    start_idx = random.choice(range(image_count - 1))
    end_idx = random.choice(range(start_idx + 1, min(start_idx + 3, image_count)))
    add_image(images[start_idx], need_loss=False, need_vae=True, need_vit=True)
    if end_idx - start_idx > 1 and random.random() < 0.5:
        if end_idx == image_count - 1:
            end_idx -= 1
        instruction = ""
        for index in range(start_idx + 1, end_idx + 1):
            instruction += random.choice(instruction_list[index - 1]) + ". "
        add_text(instruction.rstrip())
        add_image(images[end_idx], need_loss=True, need_vae=False, need_vit=False)
    else:
        for index in range(start_idx + 1, end_idx + 1):
            add_text(random.choice(instruction_list[index - 1]))
            add_image(
                images[index],
                need_loss=True,
                need_vae=index != end_idx,
                need_vit=index != end_idx,
            )

    return BagelEditingSample(
        **basic_sample_keys(crude_sample),
        image_tensor_list=image_tensor_list,
        text_ids_list=text_ids_list,
        num_tokens=num_tokens,
        sequence_plan=sequence_plan,
        metadata=metadata,
    )


def cook_bagel_vlm_sample(
    crude_sample: dict[str, object],
    *,
    tokenizer: PreTrainedTokenizerBase,
    transform: Callable[[Image.Image, int], torch.Tensor],
    image_stride: int,
) -> BagelVLMSample:
    """Apply BAGEL's VLM image, conversation, token, and sequence-plan processing."""
    metadata = _load_metadata(crude_sample)
    image_count = metadata["image_count"]
    conversations = metadata["conversations"]
    if not isinstance(image_count, int) or not isinstance(conversations, list):
        raise TypeError("BAGEL VLM metadata has invalid image or conversation fields")

    image_tensor_list = []
    num_tokens = 0
    for image_index in range(image_count):
        image_bytes = crude_sample[f"image{image_index}"]
        if not isinstance(image_bytes, bytes):
            raise TypeError("BAGEL VLM image fields must contain bytes")
        image_tensor = transform(_decode_rgb(image_bytes), image_count)
        image_tensor_list.append(image_tensor)
        num_tokens += image_tensor.shape[1] * image_tensor.shape[2] // image_stride**2

    elements = []
    for conversation in conversations:
        if conversation["from"] == "human":
            if "<image>" not in conversation["value"]:
                elements.append({"type": "text", "has_loss": 0, "text": conversation["value"]})
            else:
                text_list = conversation["value"].split("<image>")
                for index, text in enumerate(text_list):
                    if text.strip():
                        elements.append({"type": "text", "has_loss": 0, "text": text.strip()})
                    if index != len(text_list) - 1 and index < image_count:
                        elements.append({"type": "image"})
        elif conversation["from"] == "gpt":
            elements.append({"type": "text", "has_loss": 1, "text": conversation["value"]})

    text_ids_list = []
    sequence_plan = []
    for element in elements:
        if element["type"] == "text":
            text_ids = tokenizer.encode(element["text"])
            if text_ids:
                text_ids_list.append(text_ids)
                num_tokens += len(text_ids)
                sequence_plan.append(
                    {
                        "type": "text",
                        "enable_cfg": 0,
                        "loss": element["has_loss"],
                        "special_token_loss": 0,
                        "special_token_label": None,
                    }
                )
        elif element["type"] == "image":
            sequence_plan.append(
                {
                    "type": "vit_image",
                    "enable_cfg": 0,
                    "loss": 0,
                    "special_token_loss": 0,
                    "special_token_label": None,
                }
            )
    if not any(item["loss"] for item in sequence_plan):
        raise ValueError("BAGEL VLM sample has no loss-bearing text")

    return BagelVLMSample(
        **basic_sample_keys(crude_sample),
        image_tensor_list=image_tensor_list,
        text_ids_list=text_ids_list,
        num_tokens=num_tokens,
        sequence_plan=sequence_plan,
        metadata=metadata,
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


class BagelEditingTaskEncoder(TaskEncoder):
    """Register configured BAGEL Editing sample processing."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        transform: Callable[[Image.Image], torch.Tensor],
        vit_transform: Callable[[Image.Image], torch.Tensor],
        image_stride: int,
        vit_image_stride: int,
    ) -> None:
        """Configure the official tokenizer and image transforms."""
        self.cookers = [
            Cooker(
                partial(
                    cook_bagel_editing_sample,
                    tokenizer=tokenizer,
                    transform=transform,
                    vit_transform=vit_transform,
                    image_stride=image_stride,
                    vit_image_stride=vit_image_stride,
                )
            )
        ]


class BagelVLMTaskEncoder(TaskEncoder):
    """Register configured BAGEL VLM sample processing."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        transform: Callable[[Image.Image, int], torch.Tensor],
        image_stride: int,
    ) -> None:
        """Configure the official tokenizer and image transform."""
        self.cookers = [
            Cooker(
                partial(
                    cook_bagel_vlm_sample,
                    tokenizer=tokenizer,
                    transform=transform,
                    image_stride=image_stride,
                )
            )
        ]
