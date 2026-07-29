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

"""Step3.7 multimodal SFT template.

Loads the tokenizer with ``transformers.AutoTokenizer.from_pretrained``
and ``trust_remote_code=False`` (so no custom HF Python code is executed).
This is the **only** transformers-library use in this package; everything
else is pure torch / huggingface_hub.

The tokenize path (``apply_chat_template``) uses the local
``chat_template.jinja`` shipped with ``step3p7_flash_bf16``, which
determines the token sequence produced for a given input dialog.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable
from typing import Any, Optional

import numpy as np
import torch

from megatron.bridge.models.stepfun.data.flickr8k.multimodal_utils import (
    IMAGE_ITEM_TYPE,
)


IMAGE_PLACEHOLDER = "<image>"
MULTICROP_IMAGE_PLACEHOLDER = "<@image@>"
MULTICROP_PATCH_PLACEHOLDER = "<#image#>"
IMAGE_TOKEN = "<im_patch>"
IMAGE_START_TOKEN = "<im_start>"
IMAGE_END_TOKEN = "<im_end>"
PATCH_START_TOKEN = "<patch_start>"
PATCH_END_TOKEN = "<patch_end>"
IMAGE_TOKEN_COUNT = 169
PATCH_TOKEN_COUNT = 81
logger = logging.getLogger(__name__)


def _identity_path(path: str) -> str:
    return path


def _expand_step37_image_placeholders(
    text: str,
    *,
    image_token_count: int = IMAGE_TOKEN_COUNT,
    patch_token_count: int = PATCH_TOKEN_COUNT,
    image_token: str = IMAGE_TOKEN,
    image_start_token: str = IMAGE_START_TOKEN,
    image_end_token: str = IMAGE_END_TOKEN,
) -> str:
    """Expand a single ``<image>`` placeholder into
    ``<im_start>`` + ``<im_patch>`` × 169 + ``<im_end>`` (or the multicrop
    variant for patches).
    """
    image_tokens = image_token * int(image_token_count)
    patch_tokens = image_token * int(patch_token_count)
    if MULTICROP_IMAGE_PLACEHOLDER in text or MULTICROP_PATCH_PLACEHOLDER in text:
        return text.replace(MULTICROP_IMAGE_PLACEHOLDER, image_tokens).replace(
            MULTICROP_PATCH_PLACEHOLDER, patch_tokens
        )
    return text.replace(IMAGE_PLACEHOLDER, f"{image_start_token}{image_tokens}{image_end_token}")


class MultimodalSFTSample(dict):
    """Tokenized SFT sample whose length is the shifted LM training length.

    ``len(sample) = tokens.numel() - 1`` because the pack step uses
    ``tokens[:-1]`` / ``tokens[1:]`` shift-by-one.
    """

    def __len__(self) -> int:
        return max(0, int(self["tokens"].numel()) - 1)


def _load_hf_tokenizer(tokenizer_path: str):
    """Load a tokenizer from a local HF snapshot. ``trust_remote_code=False``
    is hard-coded — we never execute custom HF Python code.
    """
    if not tokenizer_path:
        raise ValueError("Step3.7 multimodal data requires `tokenizer_path` to be set")
    # Local-only import so the package import is cheap when this branch
    # isn't exercised (e.g. when other VLM data paths are used).
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=False)


class Step37MultimodalTemplate:
    """Step3.7 SFT tokenize template.

    Expands ``<image>`` placeholders → ``<im_start><im_patch>×169<im_end>``
    inside every user / tool turn, then runs
    ``tokenizer.apply_chat_template(messages, tokenize=True)`` to produce
    ``tokens`` (LongTensor). The ``loss_mask`` is set to 1 only on the
    assistant turn span(s), found by re-tokenizing the prefix up to and
    including each assistant turn.
    """

    image_placeholder = IMAGE_PLACEHOLDER
    multicrop_image_placeholder = MULTICROP_IMAGE_PLACEHOLDER
    multicrop_patch_placeholder = MULTICROP_PATCH_PLACEHOLDER

    def __init__(
        self,
        *,
        tokenizer_path: str,
        image_token_count: int,
        patch_token_count: int,
        image_token: str,
        image_start_token: str,
        image_end_token: str,
        patch_start_token: str,
        patch_end_token: str,
        max_sequence_length: int,
        path_rewrite_fn: Optional[Callable[[str], str]] = None,
    ):
        self.tokenizer = _load_hf_tokenizer(tokenizer_path)
        self.image_token_count = int(image_token_count)
        self.patch_token_count = int(patch_token_count)
        self.image_token = image_token
        self.image_start_token = image_start_token
        self.image_end_token = image_end_token
        self.patch_start_token = patch_start_token
        self.patch_end_token = patch_end_token
        self.max_sequence_length = int(max_sequence_length)
        self.path_rewrite_fn = path_rewrite_fn or _identity_path

    def _expand_image_placeholders(self, text: str) -> str:
        return _expand_step37_image_placeholders(
            text,
            image_token_count=self.image_token_count,
            patch_token_count=self.patch_token_count,
            image_token=self.image_token,
            image_start_token=self.image_start_token,
            image_end_token=self.image_end_token,
        )

    def _normalize_messages(self, data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        messages = copy.deepcopy(data)
        for message in messages:
            content = message.get("content")
            if isinstance(content, str):
                message["content"] = self._expand_image_placeholders(content)
            elif isinstance(content, list):
                for part_idx, part in enumerate(content):
                    if isinstance(part, str):
                        content[part_idx] = self._expand_image_placeholders(part)
                        continue
                    if part.get("type") == "text" and isinstance(part.get("value"), str):
                        part["value"] = self._expand_image_placeholders(part["value"])
                    if part.get("type") == "text" and isinstance(part.get("text"), str):
                        part["text"] = self._expand_image_placeholders(part["text"])
            if isinstance(message.get("reasoning_content"), str):
                message["reasoning_content"] = self._expand_image_placeholders(message["reasoning_content"])
        return messages

    def _apply_chat_template(self, messages: list[dict[str, Any]]) -> list[int]:
        add_generation_prompt = messages[-1]["role"] != "assistant"
        kwargs: dict[str, Any] = {"tokenize": True, "add_generation_prompt": add_generation_prompt}
        tool_schemas = messages[0].get("tool_schemas")
        if tool_schemas:
            kwargs["tools"] = tool_schemas
        reasoning_effort = messages[0].get("reasoning_effort")
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort

        tokenized = self.tokenizer.apply_chat_template(messages, **kwargs)
        if isinstance(tokenized, list):
            return tokenized
        return tokenized["input_ids"]

    def _normalize_images(self, raw_images: Optional[list[Any]]) -> list[tuple[str, int]]:
        normalized = []
        for image in raw_images or []:
            if isinstance(image, (tuple, list)):
                path = str(image[0])
                image_type = int(image[1]) if len(image) > 1 else IMAGE_ITEM_TYPE
            else:
                path = str(image)
                image_type = IMAGE_ITEM_TYPE
            normalized.append((self.path_rewrite_fn(path), image_type))
        return normalized

    def __call__(self, data: dict) -> MultimodalSFTSample:
        conversations = self._normalize_messages(data["conversations"])
        if conversations[-1]["role"] != "assistant":
            raise ValueError("Step3.7 multimodal SFT sample must end with an assistant message")

        all_tokens = self._apply_chat_template(conversations)
        loss_mask: np.ndarray = np.zeros(len(all_tokens), dtype=np.float32)

        last_user_idx = -1
        for idx in range(len(conversations) - 1, -1, -1):
            if conversations[idx]["role"] == "user":
                last_user_idx = idx
                break
        if last_user_idx < 0:
            raise ValueError("No user turn found in Step3.7 multimodal SFT sample")

        last_user_end = len(self._apply_chat_template(conversations[: last_user_idx + 1]))
        current_pos = last_user_end
        for idx in range(last_user_idx + 1, len(conversations)):
            tokens_up_to_current = self._apply_chat_template(conversations[: idx + 1])
            if conversations[idx]["role"] == "assistant" and conversations[idx].get("loss_mask", 1) == 1:
                loss_mask[current_pos : len(tokens_up_to_current)] = 1.0
            current_pos = len(tokens_up_to_current)

        if len(all_tokens) > self.max_sequence_length + 1:
            logger.warning(
                "Tokenized Step3.7 multimodal sample length %s exceeds max_sequence_length+1=%s; "
                "the packed dataloader oversize policy will decide whether to drop it.",
                len(all_tokens),
                self.max_sequence_length + 1,
            )

        return MultimodalSFTSample(
            tokens=torch.tensor(all_tokens, dtype=torch.long),
            loss_mask=torch.tensor(loss_mask, dtype=torch.float32),
            image_paths=self._normalize_images(data.get("images")),
        )
