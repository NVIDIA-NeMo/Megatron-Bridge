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

"""Shared utilities for Energon-based VLM task encoders.

Contains helpers extracted from the Qwen-VL task encoder so they can be
reused by the generic ``HFEncoderVLMTaskEncoder`` and any future
model-specific encoders.
"""

import json
import logging
import pickle
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from megatron.energon.epathlib.epath import EPath
from megatron.energon.flavors.base_dataset import Sample
from megatron.energon.flavors.webdataset import DefaultDecoderWebdatasetFactory
from webdataset.autodecode import Decoder, imagehandler


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IGNORE_INDEX = -100


# ---------------------------------------------------------------------------
# Mask / position-id helpers
# ---------------------------------------------------------------------------
def get_ltor_masks_and_position_ids(
    data: torch.Tensor,
    eod_token: int,
    eod_mask_loss: bool,
    reset_attention_mask: bool,
    reset_position_ids: bool,
    compute_attention_mask: bool = True,
):
    """Build masks and position ids for a left-to-right model.

    Returns:
        attention_mask: [att_mask_batch, 1, s, s] boolean mask (True means masked)
        loss_mask: [b, s] float mask (1.0 to keep loss, 0.0 to drop)
        position_ids: [b, s] positions
    """
    micro_batch_size, seq_length = data.size()

    att_mask_batch = micro_batch_size if reset_attention_mask else 1
    attention_mask = None
    if compute_attention_mask:
        attention_mask = torch.tril(torch.ones((att_mask_batch, seq_length, seq_length), device=data.device)).view(
            att_mask_batch, 1, seq_length, seq_length
        )

    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0

    position_ids = torch.arange(seq_length, dtype=torch.long, device=data.device)
    position_ids = position_ids.unsqueeze(0).repeat(micro_batch_size, 1)
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        for b in range(micro_batch_size):
            eod_index = position_ids[b, data[b] == eod_token]
            if reset_position_ids:
                eod_index = eod_index.clone()
            prev_index = 0
            for j in range(eod_index.size(0)):
                i = eod_index[j]
                if reset_attention_mask and attention_mask is not None:
                    attention_mask[b, 0, (i + 1) :, : (i + 1)] = 0
                if reset_position_ids:
                    position_ids[b, (i + 1) :] -= i + 1 - prev_index
                    prev_index = i + 1

    if compute_attention_mask and attention_mask is not None:
        attention_mask = attention_mask < 0.5

    return attention_mask, loss_mask, position_ids


# ---------------------------------------------------------------------------
# Pattern matching
# ---------------------------------------------------------------------------
def find_pattern_indices(sequence: np.ndarray, pattern, start: int = 0):
    """Find the [start, end) indices of the first occurrence of pattern in sequence from start."""
    if not isinstance(sequence, np.ndarray):
        sequence = np.array(sequence)
    pattern = np.array(pattern, dtype=sequence.dtype)
    n, m = sequence.shape[0], pattern.shape[0]
    if m == 0 or start >= n:
        return -1, -1
    end_limit = n - m + 1
    for i in range(start, max(end_limit, start)):
        if np.array_equal(sequence[i : i + m], pattern):
            return i, i + m
    return -1, -1


# ---------------------------------------------------------------------------
# PIL conversion helpers
# ---------------------------------------------------------------------------
def _tensor_to_pil(t):
    """Convert a [C,H,W] float tensor in [0,1] to a PIL Image (uint8 [0,255])."""
    from PIL import Image

    img_np = (t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img_np)


def _images_to_pil(imgs):
    """Convert WDS tensor images to PIL to match HF flow input format.

    WDS imagehandler decodes JPEG to float tensors in [0,1]. The HF flow passes
    PIL images (uint8 [0,255]) to the processor. Converting to PIL here ensures
    the processor applies identical rescaling and normalization in both flows.
    """
    if isinstance(imgs, torch.Tensor):
        if imgs.dim() == 3:
            return [_tensor_to_pil(imgs)]
        elif imgs.dim() == 4:
            return [_tensor_to_pil(img) for img in imgs]
    elif isinstance(imgs, list):
        return [_tensor_to_pil(img) if isinstance(img, torch.Tensor) else img for img in imgs]
    return imgs


def _videos_to_pil(videos):
    """Convert WDS video frame tensors to PIL to match HF flow input format."""
    if videos is None:
        return None
    result = []
    for video in videos:
        if isinstance(video, list):
            result.append([_tensor_to_pil(f) if isinstance(f, torch.Tensor) else f for f in video])
        elif isinstance(video, torch.Tensor):
            if video.dim() == 4:
                result.append([_tensor_to_pil(f) for f in video])
            elif video.dim() == 3:
                result.append([_tensor_to_pil(video)])
            else:
                result.append([video])
        else:
            result.append(video)
    return result


# ---------------------------------------------------------------------------
# Sample / dataset types
# ---------------------------------------------------------------------------
@dataclass
class ChatMLSample(Sample):
    """Multi-turn complex samples with images and videos."""

    conversation: str  # JSON string of GPT-format conversations
    imgs: Optional[List[torch.Tensor]] = None
    videos: Optional[List[List[torch.Tensor]]] = None


class videohandler:
    """Create a video handler."""

    def __init__(self, imagespec):
        self.extensions = ["jpgs", "mp4s", "videos"]
        self.extensions_mapping = {"jpgs": "jpg", "mp4s": "jpg", "videos": "jpg"}
        self.image_handler = imagehandler(imagespec)

    def __call__(self, key, data):
        """Perform nested image decoding."""
        extension = re.sub(r".*[.]", "", key)
        if extension.lower() not in self.extensions:
            return None
        data = pickle.loads(data)
        key = self.extensions_mapping[extension]
        if extension.lower() == "jpgs":
            data = [self.image_handler(key, d) for d in data]
        else:
            data = [[self.image_handler(key, d) for d in video] for video in data]
        return data


class ChatMLWebdataset(DefaultDecoderWebdatasetFactory[ChatMLSample]):
    """Webdataset factory for multi-turn ChatML samples with multimodal support.

    Extends DefaultDecoderWebdatasetFactory to decode webdataset shards into
    ChatMLSample instances, using custom handlers for image and video fields.
    """

    __sample_type__ = ChatMLSample

    def __init__(self, path: EPath, *, auto_decode: bool = True, image_decode_spec: Optional[str] = None, **kwargs):
        kwargs.pop("decoder", None)
        super().__init__(path, auto_decode=auto_decode, **kwargs)
        if auto_decode:
            spec = image_decode_spec if image_decode_spec is not None else getattr(self, "image_decode", "torchrgb")
            self._decoder = Decoder(
                [
                    imagehandler(spec),
                    videohandler(spec),
                ]
            )


# ---------------------------------------------------------------------------
# Conversation parsing
# ---------------------------------------------------------------------------
def cook_chatml_sample(conversation) -> List[Dict]:
    """Normalize a ChatML conversation to ``[{"role": ..., "content": ...}, ...]``.

    Accepts both ``from``/``value`` (GPT-style) and ``role``/``content``
    (OpenAI-style) formats, with an optional leading system turn when the
    total number of turns is odd.

    Returns a cleaned list of dicts with ``role`` in
    ``{"system", "user", "assistant"}`` and ``content`` as a plain string.
    """
    if isinstance(conversation, (str, bytes)):
        conversation = json.loads(conversation)

    conversation = conversation if not isinstance(conversation, dict) else conversation.get("conversations", [])

    _from_system_ = "from" in conversation[0]
    role_key = "from" if _from_system_ else "role"
    content_key = "value" if _from_system_ else "content"

    converted: List[Dict] = []

    # Handle optional system turn (odd number of messages)
    if len(conversation) % 2 != 0:
        converted.append({"role": "system", "content": conversation[0][content_key]})
        conversation = conversation[1:]

    if _from_system_:
        EXPECTED_ROLE = ["human", "gpt"]
        for turn_idx, turn in enumerate(conversation):
            role = turn[role_key]
            if role != EXPECTED_ROLE[turn_idx % len(EXPECTED_ROLE)]:
                logging.warning(
                    f"Expect conversation organized in order: [sys] human gpt human gpt...,"
                    f"but got role '{role}' in turn {turn_idx}"
                )
            content = turn[content_key]
            if role == "human":
                role = "user"
            elif role == "gpt":
                role = "assistant"
            converted.append({"role": role, "content": content})
    else:
        EXPECTED_ROLE = ["user", "assistant"]
        for turn_idx, turn in enumerate(conversation):
            role = turn[role_key]
            if role != EXPECTED_ROLE[turn_idx % len(EXPECTED_ROLE)]:
                logging.warning(
                    f"Expect conversation organized in order: [sys] user assistant user assistant...,"
                    f" but got role '{role}' in turn {turn_idx}"
                )
            content = turn[content_key]
            converted.append({"role": role, "content": content})

    return converted
