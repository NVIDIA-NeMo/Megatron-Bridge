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

"""Declarative local conversation sources and JSON/JSONL loading helpers."""

from __future__ import annotations

import copy
import json
import logging
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


logger = logging.getLogger(__name__)

_MEDIA_KEYS = {
    "image": ("image", "path"),
    "video": ("video", "path"),
    "audio": ("audio", "path", "audio_url"),
}
_PLACEHOLDER_PATTERN = re.compile(r"<image>|<video>|<audio>")
_WRAPPED_RECORD_KEYS = ("data", "examples", "records")


@dataclass(kw_only=True)
class LocalConversationDatasetSourceConfig:
    """Serializable source selection for one local JSON or JSONL conversation file.

    Args:
        path: Local JSON or JSONL file. Recipes may set this to ``None`` only
            when a CLI or YAML override supplies the path before finalization.
        media_root: Optional directory used to resolve relative image, video,
            and audio paths.
        records_key: Optional top-level JSON key containing the record list.
            Without it, common wrappers named ``data``, ``examples``, or
            ``records`` are detected automatically.
    """

    path: str | None
    media_root: str | None = None
    records_key: str | None = None

    def validate(self) -> None:
        """Validate the declarative local source settings."""
        if not isinstance(self.path, str) or not self.path.strip():
            raise ValueError("Local conversation source path must be a non-empty string.")
        if Path(self.path).suffix.lower() not in {".json", ".jsonl"}:
            raise ValueError("Local conversation source path must end in .json or .jsonl.")
        if self.media_root is not None and (not isinstance(self.media_root, str) or not self.media_root.strip()):
            raise ValueError("media_root must be a non-empty string when set.")
        if self.records_key is not None and (not isinstance(self.records_key, str) or not self.records_key.strip()):
            raise ValueError("records_key must be a non-empty string when set.")


def _is_remote_or_absolute_media_path(path: str) -> bool:
    parsed = urlparse(path)
    return bool(parsed.scheme) or Path(path).is_absolute()


def _resolve_media_path(value: Any, media_root: str | None) -> Any:
    if media_root is None or not isinstance(value, str) or _is_remote_or_absolute_media_path(value):
        return value
    return os.path.normpath(os.path.join(media_root, value))


def _resolve_media_values(values: Any, media_root: str | None) -> list[Any]:
    if values is None:
        return []
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        values = [values]
    return [_resolve_media_path(value, media_root) for value in values]


def _resolve_inline_media(conversation: list[dict[str, Any]], media_root: str | None) -> list[dict[str, Any]]:
    resolved = copy.deepcopy(conversation)
    for turn in resolved:
        content = turn.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            media_type = part.get("type")
            for key in _MEDIA_KEYS.get(media_type, ()):
                if part.get(key) is not None:
                    part[key] = _resolve_media_path(part[key], media_root)
    return resolved


def _split_text_by_placeholders(
    text: str,
    media: Mapping[str, Sequence[Any]],
    offsets: dict[str, int],
) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    last_end = 0
    for match in _PLACEHOLDER_PATTERN.finditer(text):
        if match.start() > last_end:
            parts.append({"type": "text", "text": text[last_end : match.start()]})

        media_type = match.group(0)[1:-1]
        values = media[media_type]
        index = offsets[media_type]
        if index >= len(values):
            logger.warning("Encountered <%s> without a corresponding local media entry.", media_type)
        else:
            parts.append({"type": media_type, media_type: values[index]})
        offsets[media_type] += 1
        last_end = match.end()

    if last_end < len(text):
        parts.append({"type": "text", "text": text[last_end:]})
    return [part for part in parts if part.get("type") != "text" or part.get("text")]


def _legacy_record_to_conversation(
    record: Mapping[str, Any],
    media_root: str | None,
) -> list[dict[str, Any]]:
    messages = record.get("messages")
    if messages is None:
        messages = record.get("conversations")
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        raise ValueError("Local conversation records must contain messages, conversation, or conversations.")

    media = {
        "image": _resolve_media_values(record.get("images", record.get("image")), media_root),
        "video": _resolve_media_values(record.get("videos", record.get("video")), media_root),
        "audio": _resolve_media_values(
            record.get("audios", record.get("audio_paths", record.get("audio_path", record.get("audio")))),
            media_root,
        ),
    }
    offsets = {media_type: 0 for media_type in media}
    conversation = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise ValueError("Local conversation turns must be dictionaries.")
        if message.get("role") is not None:
            role = str(message["role"])
            content = message.get("content", "")
        else:
            source_role = str(message.get("from", "human")).lower()
            role = "user" if source_role in {"human", "user"} else "assistant"
            content = message.get("value", "")

        if isinstance(content, str):
            content_parts = _split_text_by_placeholders(content, media, offsets)
            if content_parts:
                media_parts = [part for part in content_parts if part["type"] != "text"]
                text = "".join(part["text"] for part in content_parts if part["type"] == "text")
                if text:
                    media_parts.append({"type": "text", "text": text})
                content = media_parts
            if not content:
                content = [{"type": "text", "text": ""}]
        conversation.append({"role": role, "content": content})
    return _resolve_inline_media(conversation, media_root)


def normalize_local_conversation_record(
    record: Mapping[str, Any],
    *,
    media_root: str | None,
) -> dict[str, Any]:
    """Normalize one local record to the canonical conversation row schema."""
    if record.get("conversation") is not None:
        conversation = record["conversation"]
        if not isinstance(conversation, list):
            raise ValueError("Local conversation records require a list-valued conversation field.")
        normalized_conversation = _resolve_inline_media(conversation, media_root)
    else:
        normalized_conversation = _legacy_record_to_conversation(record, media_root)

    source_keys = {
        "messages",
        "conversation",
        "conversations",
        "image",
        "images",
        "video",
        "videos",
        "audio",
        "audios",
        "audio_path",
        "audio_paths",
    }
    metadata = {key: copy.deepcopy(value) for key, value in record.items() if key not in source_keys}
    return {"conversation": normalized_conversation, **metadata}


def _validate_records(records: Any, *, source_path: Path) -> list[dict[str, Any]]:
    if not isinstance(records, list):
        records = [records]
    if not all(isinstance(record, Mapping) for record in records):
        raise ValueError(f"Local conversation source {source_path} must contain JSON objects.")
    return [dict(record) for record in records]


def load_local_json_records(source: LocalConversationDatasetSourceConfig) -> list[dict[str, Any]]:
    """Load records from one validated local JSON or JSONL source."""
    source.validate()
    assert source.path is not None
    source_path = Path(source.path)
    if source_path.suffix.lower() == ".jsonl":
        records = []
        with source_path.open(encoding="utf-8") as source_file:
            for line_number, line in enumerate(source_file, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON on line {line_number} of {source_path}.") from error
                records.extend(_validate_records(record, source_path=source_path))
        return records

    with source_path.open(encoding="utf-8") as source_file:
        payload = json.load(source_file)
    if isinstance(payload, Mapping):
        if source.records_key is not None:
            if source.records_key not in payload:
                raise ValueError(f"Local conversation source {source_path} has no '{source.records_key}' key.")
            payload = payload[source.records_key]
        elif (wrapper_key := next((key for key in _WRAPPED_RECORD_KEYS if key in payload), None)) is not None:
            payload = payload[wrapper_key]
    return _validate_records(payload, source_path=source_path)


def load_local_conversation_examples(
    source: LocalConversationDatasetSourceConfig,
) -> list[dict[str, Any]]:
    """Load and normalize one local conversation source for direct SFT."""
    examples = [
        normalize_local_conversation_record(record, media_root=source.media_root)
        for record in load_local_json_records(source)
    ]
    if not examples:
        raise ValueError("Local conversation source produced no examples after normalization.")
    return examples
