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

"""Inkling2 processor for image-text SFT.

The ``thinkingmachines/Inkling-2-Small`` release names an ``Inkling2Processor`` in its
``processor_config.json`` but ships no implementation. This one renders a conversation straight to token
IDs exactly as the release's ``chat_template.jinja`` does, then gives each image its patches: every
content part is its own block, ``<|message_ROLE|>[name]<|content_KIND|>payload<|end_message|>``, a
``Thinking effort level`` system line precedes the first non-system message, and every assistant turn ends
with ``<|content_model_end_sampling|>``. Where the template writes one ``<|unused_200054|>`` image
placeholder, the processor writes one per 40x40-pixel patch, which the model replaces with the image's
features. The whole assistant turn, headers and end-of-sampling token included, is the supervised span.

Two inputs go beyond the template: an assistant message may carry its reasoning in ``thinking`` (the field
Bridge's dataset adapters write) or ``reasoning`` instead of ``reasoning_content``, the first one set being used,
as Transformers does from version 5.22; and an image part carries its pixels in ``image``.

Registering it with ``AutoProcessor`` for :class:`Inkling2Config` lets
``DirectHFSFTDatasetConfig.hf_processor_path`` load it from a checkpoint directory or the Hub.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from PIL import Image
from transformers import AutoConfig, AutoProcessor, AutoTokenizer, BatchFeature

from megatron.bridge.models.inkling.configuration_inkling2 import Inkling2Config
from megatron.bridge.models.inkling.data.image_processing import (
    Inkling2EncodedImage,
    Inkling2ImageProcessor,
    load_processor_config,
)


_MESSAGE_TOKENS = {
    "system": "<|message_system|>",
    "user": "<|message_user|>",
    "assistant": "<|message_model|>",
    "tool": "<|message_tool|>",
}
_FRAMING_TOKENS = (
    *_MESSAGE_TOKENS.values(),
    "<|content_text|>",
    "<|content_image|>",
    "<|content_thinking|>",
    "<|content_invoke_tool_json|>",
    "<|end_message|>",
    "<|content_model_end_sampling|>",
)
# The chat template's named thinking-effort levels and its default.
REASONING_EFFORTS = {"none": 0.0, "minimal": 0.1, "low": 0.2, "medium": 0.7, "high": 0.9, "xhigh": 0.99, "max": 0.99}
DEFAULT_REASONING_EFFORT = 0.9
_TEXT_PARTS = ("text", "input_text")
_REASONING_FIELDS = ("reasoning_content", "thinking", "reasoning")
_IMAGE_PARTS = ("image", "input_image", "image_url")
# Tokenizer arguments that apply to loading the text tokenizer from the same checkpoint.
_TOKENIZER_KWARGS = ("revision", "trust_remote_code", "cache_dir", "local_files_only", "token")


def reasoning_effort_level(effort: str | float | None) -> str:
    """Return the chat template's ``Thinking effort level`` text for a named or numeric effort.

    Args:
        effort: A named level such as ``"none"`` or ``"high"``, a number in ``[0, 0.99]``, or ``None`` for the
            template's default.

    Returns:
        The level as the template prints it.

    Raises:
        ValueError: For an unknown name or a number outside ``[0, 0.99]``.
    """
    if effort is None:
        effort = DEFAULT_REASONING_EFFORT
    if isinstance(effort, str):
        if effort.strip() not in REASONING_EFFORTS:
            raise ValueError(f"Unknown reasoning_effort: {effort}")
        level = REASONING_EFFORTS[effort.strip()]
    else:
        level = float(effort)
    if not 0.0 <= level <= 0.99:
        raise ValueError("reasoning_effort must be in [0.0, 0.99]")
    return "0" if level == 0.0 else str(level)


class Inkling2Processor:
    """Turns conversations with images into Inkling2 token IDs, assistant masks and vision patches.

    Args:
        tokenizer: The checkpoint's Hugging Face tokenizer.
        image_processor: Image preprocessing for the checkpoint.
        image_token_id: Placeholder token the model replaces with image features.
        reasoning_effort: Thinking effort written before the first non-system message, as the chat template's
            ``reasoning_effort`` argument; ``None`` writes the template's default.
    """

    def __init__(
        self,
        tokenizer: Any,
        image_processor: Inkling2ImageProcessor,
        *,
        image_token_id: int,
        reasoning_effort: str | float | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.image_token_id = image_token_id
        self.reasoning_effort = reasoning_effort
        self._effort_level = reasoning_effort_level(reasoning_effort)
        ids = {token: tokenizer.convert_tokens_to_ids(token) for token in _FRAMING_TOKENS}
        unknown = [token for token, token_id in ids.items() if token_id is None or token_id == tokenizer.unk_token_id]
        if unknown:
            raise ValueError(f"The tokenizer lacks Inkling2 framing tokens: {unknown}")
        self._ids = ids

    @classmethod
    def from_pretrained(
        cls,
        path: str | os.PathLike,
        *,
        max_image_tokens: int | None = None,
        reasoning_effort: str | float | None = None,
        **kwargs: Any,
    ) -> Inkling2Processor:
        """Load the tokenizer, image processing and image placeholder of a checkpoint.

        Args:
            path: Checkpoint directory or Hugging Face repository ID.
            max_image_tokens: Token budget per image; larger images are encoded at the largest size that fits.
                ``None`` keeps native sizes.
            reasoning_effort: Thinking effort for every rendered conversation; ``None`` writes the template's
                default.
            **kwargs: ``AutoProcessor`` arguments; the loading-related ones are passed on.

        Returns:
            The processor.

        Raises:
            ValueError: If the checkpoint is not an Inkling2 checkpoint with the release's processor settings,
                or its image token is not the one the model replaces.
        """
        tokenizer_kwargs = {key: kwargs[key] for key in _TOKENIZER_KWARGS if key in kwargs}
        config = AutoConfig.from_pretrained(path, **tokenizer_kwargs)
        # Bridge's configuration class, or Transformers' own where it implements Inkling2.
        if config.model_type != Inkling2Config.model_type or config.vision_config is None:
            raise ValueError(f"{path} is not an Inkling2 checkpoint with an image encoder")
        download_kwargs = {key: value for key, value in tokenizer_kwargs.items() if key != "trust_remote_code"}
        processor_config = load_processor_config(path, **download_kwargs)
        if processor_config.get("processor_class") != "Inkling2Processor":
            raise ValueError(f"Unsupported Inkling2 processor configuration in {path}: {processor_config}")
        tokenizer = AutoTokenizer.from_pretrained(path, **tokenizer_kwargs)
        image_token_id = tokenizer.convert_tokens_to_ids(processor_config.get("image_token"))
        if image_token_id != config.image_token_id:
            raise ValueError(
                f"The image token {processor_config.get('image_token')!r} is token {image_token_id}, but the model "
                f"replaces token {config.image_token_id}"
            )
        return cls(
            tokenizer,
            Inkling2ImageProcessor.from_dict(processor_config, max_image_tokens=max_image_tokens),
            image_token_id=image_token_id,
            reasoning_effort=reasoning_effort,
        )

    def _text(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _block(self, header: list[int], kind: str, payload: list[int]) -> list[int]:
        return [*header, self._ids[kind], *payload, self._ids["<|end_message|>"]]

    @staticmethod
    def _tool_header_name(message: Mapping[str, Any], conversation: Sequence[Mapping[str, Any]]) -> str:
        """Name a tool result by its ``name``, or by every assistant tool call its ``tool_call_id`` matches."""
        if message.get("name"):
            return str(message["name"])
        call_id = message.get("tool_call_id")
        if not call_id:
            return ""
        names = []
        for previous in conversation:
            if previous.get("role") != "assistant":
                continue
            for call in previous.get("tool_calls") or []:
                function = call.get("function") or {}
                if call.get("id") == call_id and function.get("name") is not None:
                    names.append(str(function["name"]))
        return "".join(names)

    def _tool_calls(self, message: Mapping[str, Any]) -> list[int]:
        tokens: list[int] = []
        for call in message.get("tool_calls") or []:
            function = call["function"]
            name = function.get("name")
            if not isinstance(name, str):
                raise ValueError("tool call function name must be a string")
            if message.get("name") and message["name"] != name:
                raise ValueError(f"assistant name {message['name']} does not match its tool call {name}")
            arguments = function.get("arguments") or {}
            if isinstance(arguments, str):
                raise ValueError("tool call arguments must be a parsed object, not a JSON string")
            if not isinstance(arguments, Mapping):
                raise ValueError("tool call arguments must be an object")
            payload = (
                '{"name":'
                + json.dumps(name, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
                + ',"args":'
                + json.dumps(arguments, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
                + "}"
            )
            tokens += [
                self._ids["<|message_model|>"],
                *self._text(name),
                self._ids["<|content_invoke_tool_json|>"],
                *self._text(payload),
                self._ids["<|end_message|>"],
            ]
        return tokens

    def render(
        self, conversation: Sequence[Mapping[str, Any]]
    ) -> tuple[list[int], list[int], list[Inkling2EncodedImage]]:
        """Render one conversation.

        Args:
            conversation: Messages with ``role`` and ``content``; content is a string or a list of strings,
                ``{"type": "text", "text": ...}`` and ``{"type": "image", "image": ...}`` parts. An assistant
                message may carry its reasoning in ``reasoning_content``, ``thinking`` or ``reasoning`` and its calls in
                ``tool_calls``.

        Returns:
            Token IDs, a 0/1 mask of the supervised (assistant) tokens, and the images in order.

        Raises:
            ValueError: On an unsupported role or content part, an image outside a user or tool message, or
                malformed tool calls.
        """
        input_ids: list[int] = []
        assistant_mask: list[int] = []
        images: list[Inkling2EncodedImage] = []
        effort = self._block(
            [self._ids["<|message_system|>"]],
            "<|content_text|>",
            self._text(f"Thinking effort level: {self._effort_level}"),
        )
        effort_written = False
        for message in conversation:
            role = message["role"]
            if role not in _MESSAGE_TOKENS:
                raise ValueError(f"Unknown message role: {role}")
            if not effort_written and role != "system":
                input_ids += effort
                assistant_mask += [0] * len(effort)
                effort_written = True
            header = [self._ids[_MESSAGE_TOKENS[role]]]
            if role == "tool":
                header += self._text(self._tool_header_name(message, conversation))
            elif role == "assistant" and message.get("name"):
                header += self._text(str(message["name"]))
            # The first reasoning field that is set, in the order Transformers aliases them.
            reasoning = next((message[key] for key in _REASONING_FIELDS if message.get(key) is not None), None)
            if role != "assistant":
                reasoning = None
            tokens: list[int] = []
            if reasoning:
                tokens += self._block(header, "<|content_thinking|>", self._text(reasoning))
            content = message.get("content")
            if isinstance(content, str):
                if content or not (role == "assistant" and (message.get("tool_calls") or reasoning)):
                    tokens += self._block(header, "<|content_text|>", self._text(content))
            elif content:
                for part in content:
                    if isinstance(part, str):
                        tokens += self._block(header, "<|content_text|>", self._text(part))
                        continue
                    kind = part.get("type")
                    if kind is None or kind in _TEXT_PARTS:
                        text = part.get("text")
                        tokens += self._block(
                            header, "<|content_text|>", self._text(text if isinstance(text, str) else "")
                        )
                    elif kind in _IMAGE_PARTS:
                        if role not in ("user", "tool"):
                            raise ValueError("Image content is only supported for user and tool messages")
                        if "image" not in part:
                            raise ValueError(f"An Inkling2 {kind} part must carry its pixels under 'image'")
                        image = self.image_processor.encode(part["image"])
                        images.append(image)
                        tokens += self._block(header, "<|content_image|>", [self.image_token_id] * image.num_tokens)
                    else:
                        raise ValueError(f"Unsupported content part type: {kind}")
            if role == "assistant":
                tokens += self._tool_calls(message)
                tokens.append(self._ids["<|content_model_end_sampling|>"])
            input_ids += tokens
            assistant_mask += [int(role == "assistant")] * len(tokens)
        if not effort_written:
            input_ids += effort
            assistant_mask += [0] * len(effort)
        return input_ids, assistant_mask, images

    @property
    def chat_template(self) -> str | None:
        """The checkpoint's chat template, which Hugging Face-style callers render prompts with."""
        return getattr(self.tokenizer, "chat_template", None)

    def apply_chat_template(
        self,
        conversation: Sequence[Mapping[str, Any]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Render a conversation with the checkpoint's chat template, as a Transformers processor does.

        The template writes one image placeholder per image; ``processor(text=..., images=...)`` expands each into
        one placeholder per patch. Inkling 2's template adds no generation prompt, since the model writes its own
        message headers.

        Args:
            conversation: The messages.
            tokenize: Return token IDs instead of text.
            add_generation_prompt: Passed to the template, which ignores it.
            **kwargs: Template arguments such as ``reasoning_effort`` or ``tools``.

        Returns:
            The rendered text, or token IDs with ``tokenize=True``.
        """
        if self.reasoning_effort is not None:
            kwargs.setdefault("reasoning_effort", self.reasoning_effort)
        return self.tokenizer.apply_chat_template(
            conversation, tokenize=tokenize, add_generation_prompt=add_generation_prompt, **kwargs
        )

    def _encode_text(self, text: str | Sequence[str], images: Any) -> BatchFeature:
        """Tokenize rendered prompts and give each image placeholder one position per patch."""
        texts = [text] if isinstance(text, str) else list(text)
        if images is None:
            images = []
        elif isinstance(images, Image.Image) or not isinstance(images, Sequence):
            images = [images]
        flat = [image for item in images for image in (item if isinstance(item, (list, tuple)) else [item])]
        encoded = [self.image_processor.encode(image) for image in flat]
        remaining = iter(encoded)
        rows = []
        for prompt in texts:
            row: list[int] = []
            for token in self._text(prompt):
                if token != self.image_token_id:
                    row.append(token)
                    continue
                image = next(remaining, None)
                if image is None:
                    raise ValueError("The text has more image placeholders than images")
                row += [self.image_token_id] * image.num_tokens
            rows.append(row)
        if next(remaining, None) is not None:
            raise ValueError("More images than image placeholders in the text")
        length = max(len(row) for row in rows)
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        input_ids = torch.full((len(rows), length), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(rows), length), dtype=torch.long)
        for index, row in enumerate(rows):
            input_ids[index, : len(row)] = torch.tensor(row, dtype=torch.long)
            attention_mask[index, : len(row)] = 1
        return BatchFeature(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": torch.cat([image.patches for image in encoded]) if encoded else None,
                "image_grid_thw": torch.cat([image.grid_thw for image in encoded]) if encoded else None,
            }
        )

    def __call__(
        self,
        conversations: Sequence[Sequence[Mapping[str, Any]]] | None = None,
        *,
        max_length: int | None = None,
        text: str | Sequence[str] | None = None,
        images: Any = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor | None] | BatchFeature:
        """Render and right-pad a batch of conversations, or encode rendered text as a Transformers processor.

        Args:
            conversations: One message list per sample.
            max_length: Longest allowed sample. Text after the last image is cut to fit; a sample whose
                image placeholders would be cut is rejected, since its image features could not be placed.
            text: Prompts already rendered, for example by :meth:`apply_chat_template`, with one image placeholder
                per image. Given instead of ``conversations``.
            images: The images for ``text``'s placeholders, in order.
            **kwargs: Transformers processor arguments such as ``return_tensors`` and ``padding``; the outputs are
                always right-padded PyTorch tensors.

        Returns:
            ``input_ids``, ``attention_mask`` and, for conversations, ``assistant_masks`` ``[B, L]``;
            ``pixel_values`` ``[patches, 2, 40, 40, 3]`` BF16 and ``image_grid_thw`` ``[images, 3]`` for the
            batch's images in order, or ``None`` when the batch has none.

        Raises:
            ValueError: If an image does not fit in ``max_length``, or the images do not match the text's image
                placeholders.
        """
        if text is not None:
            if conversations is not None:
                raise ValueError("Pass conversations or text, not both")
            return self._encode_text(text, images)
        if conversations is None:
            raise ValueError("Pass conversations or text")
        rows, masks, images = [], [], []
        for conversation in conversations:
            input_ids, assistant_mask, conversation_images = self.render(conversation)
            if max_length is not None and len(input_ids) > max_length:
                cut = input_ids[max_length:]
                if self.image_token_id in cut:
                    raise ValueError(
                        f"A sample needs {len(input_ids)} tokens, and its image placeholders do not fit in "
                        f"{max_length}; raise the sequence length."
                    )
                input_ids, assistant_mask = input_ids[:max_length], assistant_mask[:max_length]
            rows.append(input_ids)
            masks.append(assistant_mask)
            images += conversation_images
        length = max(len(row) for row in rows)
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        input_ids = torch.full((len(rows), length), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(rows), length), dtype=torch.long)
        assistant_masks = torch.zeros((len(rows), length), dtype=torch.long)
        for index, (row, mask) in enumerate(zip(rows, masks, strict=True)):
            input_ids[index, : len(row)] = torch.tensor(row, dtype=torch.long)
            attention_mask[index, : len(row)] = 1
            assistant_masks[index, : len(row)] = torch.tensor(mask, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "assistant_masks": assistant_masks,
            "pixel_values": torch.cat([image.patches for image in images]) if images else None,
            "image_grid_thw": torch.cat([image.grid_thw for image in images]) if images else None,
        }


AutoProcessor.register(Inkling2Config, Inkling2Processor, exist_ok=True)


__all__ = ["DEFAULT_REASONING_EFFORT", "REASONING_EFFORTS", "Inkling2Processor", "reasoning_effort_level"]
