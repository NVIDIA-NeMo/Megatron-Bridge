# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""
Provider for datasets preloaded from JSON/JSONL files into conversation schema.
"""

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Literal

from transformers import AutoProcessor

from megatron.bridge.data.vlm_datasets.conversation_dataset import VLMConversationDataset
from megatron.bridge.training.config import DatasetBuildContext, DatasetProvider


def _split_text_by_placeholders(
    text: str, image_paths: list[str], video_paths: list[str] | None = None
) -> list[dict[str, Any]]:
    """
    Split legacy text containing "<image>"/"<video>" markers into an alternating
    sequence of text and media parts, preserving the original order and spacing.
    """
    parts: list[dict[str, Any]] = []
    img_idx = 0
    vid_idx = 0

    last_end = 0
    for match in re.finditer(r"<image>|<video>", text):
        # Preceding text (if any)
        if match.start() > last_end:
            seg = text[last_end : match.start()]
            if seg:
                parts.append({"type": "text", "text": seg})

        token = match.group(0)
        if token == "<image>":
            if img_idx >= len(image_paths):
                logging.warning("Encountered <image> without corresponding entry in images list.")
            else:
                parts.append({"type": "image", "image": image_paths[img_idx]})
            img_idx += 1
        else:  # <video>
            if video_paths is None or vid_idx >= len(video_paths):
                logging.warning("Encountered <video> without corresponding entry in videos list.")
            else:
                parts.append({"type": "video", "video": video_paths[vid_idx]})
            vid_idx += 1
        last_end = match.end()

    # Trailing text (if any)
    if last_end < len(text):
        tail = text[last_end:]
        if tail:
            parts.append({"type": "text", "text": tail})
    return parts


def _normalize_paths(paths: list[Any] | None, base_folder: str | None) -> list[Any] | None:
    if not paths or base_folder is None:
        return paths
    normalized: list[Any] = []
    for p in paths:
        if not isinstance(p, str):
            normalized.append(p)
            continue
        if any(prefix in p for prefix in ["http:", "https:", "file:"]) or os.path.isabs(p):
            normalized.append(p)
        else:
            normalized.append(os.path.normpath(os.path.join(base_folder, p)))
    return normalized


def _record_to_conversation(record: dict[str, Any], image_folder: str | None) -> list[dict[str, Any]] | None:
    """
    Transform a single legacy record into an AutoProcessor-friendly conversation schema.
    Supports two input styles:
      - {"conversation": [...]} already in HF schema -> passthrough
      - {"messages": [...], "images": [...], "videos": [...]} with <image>/<video> markers
    """
    if "conversation" in record:
        return record["conversation"]

    # Accept legacy "messages" or LLaVA-style "conversations"
    messages = record.get("messages")
    llava_conversations = record.get("conversations")
    if not messages and not llava_conversations:
        return None

    # Build images/videos list from several possible fields
    images: list[Any] = []
    if "images" in record and isinstance(record["images"], list):
        images = record["images"]
    elif "image" in record and record["image"] is not None:
        # Single image string -> list
        if isinstance(record["image"], list):
            images = record["image"]
        else:
            images = [record["image"]]
    videos: list[Any] = record.get("videos", []) or []
    images = _normalize_paths(images, image_folder) or []
    videos = _normalize_paths(videos, image_folder) or []

    conversation: list[dict[str, Any]] = []
    source_msgs = messages if messages is not None else llava_conversations
    for msg in source_msgs:
        # LLaVA uses {'from': 'human'|'gpt', 'value': '...'}
        role = msg.get("role")
        if role is None:
            from_role = msg.get("from", "human")
            role = "user" if from_role.lower() in ("human", "user") else "assistant"
            content_str = msg.get("value", "")
        else:
            content_str = msg.get("content", "")

        content_list = _split_text_by_placeholders(content_str, images, videos)
        if content_list:
            # Reorder to media-first followed by a single combined text segment to
            # match typical VLM chat templates (media before text)
            media_parts = [p for p in content_list if p.get("type") in ("image", "video")]
            text_parts = [p.get("text", "") for p in content_list if p.get("type") == "text" and p.get("text")]
            if text_parts:
                media_parts.append({"type": "text", "text": "".join(text_parts)})
            content_list = media_parts
        if not content_list:
            content_list = [{"type": "text", "text": content_str}]
        conversation.append({"role": role, "content": content_list})
    return conversation


def _load_preloaded_examples(path: str) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    if path.endswith(".jsonl"):
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                examples.append(json.loads(line))
    else:
        with open(path, "r") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            examples = payload
        elif isinstance(payload, dict):
            # Some datasets wrap under a key, try common ones
            for key in ["data", "examples", "records"]:
                if key in payload and isinstance(payload[key], list):
                    examples = payload[key]
                    break
            if not examples:
                examples = [payload]
        else:
            raise ValueError(f"Unsupported JSON structure in {path}")
    return examples


@dataclass(kw_only=True)
class PreloadedVLMConversationProvider(DatasetProvider):
    """DatasetProvider that builds VLM conversation datasets from preloaded JSON/JSONL files.

    The provider converts legacy Qwen2/VL style records with '<image>'/'<video>' markers
    into a conversation schema consumable by HuggingFace AutoProcessor for Qwen2.5-VL.
    """

    # Required to match model.seq_length
    sequence_length: int

    # HF processor/model identifier (e.g., "Qwen/Qwen2.5-VL-3B-Instruct")
    hf_processor_path: str = "Qwen/Qwen2.5-VL-3B-Instruct"

    # Paths to preloaded datasets (JSON/JSONL). Any can be None.
    train_data_path: str | None = None
    valid_data_path: str | None = None
    test_data_path: str | None = None

    # Optional image/video root to resolve relative paths
    image_folder: str | None = None

    # Keep parity with GPTDatasetConfig usage in batching utilities
    skip_getting_attention_mask_from_dataset: bool = True

    # Default dataloader type for VLM providers
    dataloader_type: Literal["single", "cyclic", "external"] | None = "single"

    def _build_split_dataset(
        self,
        split_path: str | None,
        target_length: int,
        processor: Any,
    ) -> VLMConversationDataset | None:
        if not split_path or target_length <= 0:
            return None
        raw_examples = _load_preloaded_examples(split_path)
        base_examples: list[dict[str, Any]] = []
        for rec in raw_examples:
            conv = _record_to_conversation(rec, self.image_folder)
            if conv is None:
                continue
            base_examples.append({"conversation": conv})
        if not base_examples:
            logging.warning(f"No usable examples parsed from {split_path}")
            return None
        return VLMConversationDataset(
            base_examples=base_examples,
            target_length=target_length,
            processor=processor,
        )

    def build_datasets(self, context: DatasetBuildContext) -> tuple[Any | None, Any | None, Any | None]:
        processor = AutoProcessor.from_pretrained(self.hf_processor_path, trust_remote_code=True)
        train_ds = self._build_split_dataset(self.train_data_path, context.train_samples, processor)
        valid_ds = self._build_split_dataset(self.valid_data_path, context.valid_samples, processor)
        test_ds = self._build_split_dataset(self.test_data_path, context.test_samples, processor)
        return train_ds, valid_ds, test_ds
