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

"""Shared data helpers for the agentic SFT preparation scripts.

Adapted from the Megatron-LM agentic SFT example by Zhongbo Zhu.
"""

from __future__ import annotations

import hashlib
import json
import operator
import os
import tempfile
from array import array
from collections.abc import Mapping, Sequence
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Return the SHA-256 digest of a file, read in 1 MiB blocks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: str | os.PathLike[str], value: Any) -> None:
    """Write JSON to ``path`` through a temporary file in the same directory."""
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2)
        stream.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class JsonlRows:
    """Map-style JSONL access that indexes nonblank rows by byte offset.

    Files must remain immutable for the lifetime of the reader. Construction scans every physical
    line and stores offsets and line numbers, but does not materialize trajectories or cache
    tokenization. The reader keeps one file handle per process and omits it from pickle state, so
    it can be passed to worker processes. ``column_names`` describes only the first record and must
    not be used to infer optional row fields.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        """Index ``path``.

        Raises:
            ValueError: If the file is empty or a row is not a UTF-8 JSON object.
        """
        self.path = os.fspath(path)
        self.offsets = array("Q")
        self.line_numbers = array("Q")
        self.column_names: list[str] = []
        self._file = None
        self._pid: int | None = None
        with open(self.path, "rb") as stream:
            self._signature = self._file_signature(stream)
            line_number = 0
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                line_number += 1
                if line.strip():
                    if not self.offsets:
                        self.column_names = list(self._decode(line, line_number))
                    self.offsets.append(offset)
                    self.line_numbers.append(line_number)
        if not self.offsets:
            raise ValueError(f"Empty JSONL: {self.path}")

    @staticmethod
    def _file_signature(stream) -> tuple[int, int, int, int]:
        stat = os.fstat(stream.fileno())
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns

    def _decode(self, line: bytes, line_number: int) -> dict[str, Any]:
        location = f"{self.path}:{line_number}"
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise ValueError(f"{location}: invalid UTF-8 JSON object") from None
        if not isinstance(row, dict):
            raise ValueError(f"{location}: JSONL row must be an object")
        return row

    def __len__(self) -> int:
        """Return the number of nonblank rows."""
        return len(self.offsets)

    def location(self, index: int) -> str:
        """Return a safe error location: the path and the one-based physical line."""
        return f"{self.path}:{self.line_numbers[index]}"

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Decode one row, reopening the file in a new process."""
        index = operator.index(index)
        offset = self.offsets[index]  # Bounds checking precedes opening a handle.
        if self._file is None or self._pid != os.getpid():
            self.close()
            self._file = open(self.path, "rb")
            self._pid = os.getpid()
        if self._file_signature(self._file) != self._signature:
            raise ValueError(f"{self.path}: JSONL changed after indexing; immutable input is required")
        self._file.seek(offset)
        return self._decode(self._file.readline(), self.line_numbers[index])

    def close(self) -> None:
        """Release this process's file handle; a later access reopens it."""
        if self._file is not None:
            self._file.close()
        self._file = None
        self._pid = None

    def __getstate__(self) -> dict[str, Any]:
        """Drop the file handle so the reader can be pickled."""
        return {**self.__dict__, "_file": None, "_pid": None}

    def __del__(self) -> None:
        """Close any open handle."""
        if getattr(self, "_file", None) is not None:
            self._file.close()


# ---------------------------------------------------------------------------
# Chat encoding: one complete rendering per conversation, with unshifted targets.
# ---------------------------------------------------------------------------

IGNORE_INDEX = -100
_CHAT_ROLES = frozenset({"system", "developer", "user", "assistant", "tool"})
_PIPELINE_TEMPLATE_KWARGS = frozenset(
    {
        "add_generation_prompt",
        "add_special_tokens",
        "chat_template",
        "continue_final_message",
        "conversation",
        "max_length",
        "messages",
        "padding",
        "return_assistant_tokens_mask",
        "return_assistant_token_mask",
        "return_dict",
        "return_offsets_mapping",
        "return_tensors",
        "tokenize",
        "tokenizer_kwargs",
        "tools",
        "truncation",
    }
)


def _parse_json(value: Any, field: str) -> Any:
    if not isinstance(value, str):
        return deepcopy(value)
    try:
        return json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"{field} contains invalid JSON at character {error.pos}") from None


def normalize_chat(messages: Any, tools: Any = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    """Validate text records while retaining fields and caller-owned values.

    Native and JSON-serialized arrays are accepted. Tool arguments normalize to objects and the
    template controls their serialization. Unknown roles and multimodal content are errors, not
    silent remapping or loss of information.

    Raises:
        ValueError: If the conversation or tools are malformed.
    """
    messages = _parse_json(messages, "messages")
    tools = _parse_json(tools, "tools")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a nonempty list")
    if tools is not None and (not isinstance(tools, list) or any(not isinstance(tool, dict) for tool in tools)):
        raise ValueError("tools must be a list of objects")
    for index, message in enumerate(messages):
        field = f"messages[{index}]"
        if not isinstance(message, dict):
            raise ValueError(f"{field} must be an object")
        if not isinstance(message.get("role"), str) or message["role"] not in _CHAT_ROLES:
            raise ValueError(f"{field}.role must be one of {sorted(_CHAT_ROLES)}")
        if message.get("content") is None:
            message["content"] = ""
        if not isinstance(message["content"], str):
            raise ValueError(f"{field}.content must be text; multimodal content is unsupported")
        calls = message.get("tool_calls")
        if calls is not None and not isinstance(calls, list):
            raise ValueError(f"{field}.tool_calls must be a list")
        for call_index, call in enumerate(calls or []):
            call_field = f"{field}.tool_calls[{call_index}]"
            if not isinstance(call, dict):
                raise ValueError(f"{call_field} must be an object")
            function = call.get("function", call)
            if not isinstance(function, dict):
                raise ValueError(f"{call_field}.function must be an object")
            arguments = _parse_json(function.get("arguments"), f"{call_field}.arguments")
            if not isinstance(arguments, dict):
                raise ValueError(f"{call_field}.arguments must be a JSON object")
            function["arguments"] = arguments
    return messages, tools


def _template_controls(chat_template_kwargs: Mapping[str, Any] | None) -> dict[str, Any]:
    if chat_template_kwargs is None:
        controls: dict[str, Any] = {}
    elif isinstance(chat_template_kwargs, Mapping) and all(isinstance(key, str) for key in chat_template_kwargs):
        controls = deepcopy(dict(chat_template_kwargs))
    else:
        raise ValueError("chat_template_kwargs must be a string-keyed mapping")
    forbidden = _PIPELINE_TEMPLATE_KWARGS.intersection(controls)
    if forbidden:
        raise ValueError(f"chat_template_kwargs cannot override {sorted(forbidden)}")
    return controls


def _single_sequence(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 1:
        raise ValueError(f"Expected a single {name} sequence, got shape {array.shape}")
    return array


def _input_ids(encoded: Any) -> np.ndarray:
    if isinstance(encoded, Mapping):
        encoded = encoded["input_ids"]
    elif hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    ids = _single_sequence(encoded, "input_ids")
    if ids.size and ids.dtype.kind not in "iu":
        raise ValueError("Chat input_ids must be integers")
    return ids.astype(np.int64, copy=False)


@lru_cache(maxsize=32)
def _validate_generation_template(template: str) -> None:
    """Distinguish missing training-template support from an unsupervised row."""
    from jinja2 import Environment

    in_block = False
    for _, kind, value in Environment().lex(template):
        if kind == "block_begin":
            in_block = True
        elif in_block and kind != "whitespace":
            if kind == "name" and value == "generation":
                return
            in_block = False
    raise ValueError("Assistant loss requires a generation-tagged HF training template")


def tokenize_chat(
    hf_tokenizer: Any,
    messages: Any,
    *,
    tools: Any = None,
    loss_mode: str = "assistant",
    chat_template_kwargs: Mapping[str, Any] | None = None,
    add_generation_prompt: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return complete-conversation IDs and equal-length, unshifted targets.

    Assistant loss follows the Hugging Face generation mask exactly. The chat template must mark
    every desired target, including reasoning, tool calls and any end tokens, with
    ``{% generation %}`` blocks. Missing, malformed or empty masks fail explicitly; no boundaries
    are inferred from rendered text. Full loss supervises every rendered token without adding an
    EOS token.

    Args:
        hf_tokenizer: A Hugging Face tokenizer that carries the training chat template.
        messages: The conversation, as a list or a JSON string.
        tools: Optional tool schemas, as a list or a JSON string.
        loss_mode: ``assistant`` or ``full``.
        chat_template_kwargs: Per-row template controls such as ``enable_thinking``.
        add_generation_prompt: Append the assistant prefix; not allowed with assistant loss.

    Returns:
        ``(ids, targets)`` where ``targets`` holds the token ID where supervised and ``-100`` elsewhere.

    Raises:
        ValueError: If the inputs, template or mask are invalid, or no assistant target remains.
    """
    if loss_mode not in ("assistant", "full"):
        raise ValueError(f"Unknown chat loss mode: {loss_mode}")
    if loss_mode == "assistant" and add_generation_prompt:
        raise ValueError("Assistant supervision expects completed turns, not a generation prompt")
    messages, tools = normalize_chat(messages, tools)
    controls = _template_controls(chat_template_kwargs)
    kwargs = dict(tools=tools, add_generation_prompt=add_generation_prompt, **controls)
    if loss_mode == "full":
        ids = _input_ids(hf_tokenizer.apply_chat_template(messages, tokenize=True, **kwargs))
        return ids, ids.copy()
    template = hf_tokenizer.get_chat_template(tools=tools)
    _validate_generation_template(template)
    encoded = hf_tokenizer.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_assistant_tokens_mask=True, chat_template=template, **kwargs
    )
    ids = _input_ids(encoded)
    if not isinstance(encoded, Mapping) or "assistant_masks" not in encoded:
        raise ValueError(
            "Chat template returned no assistant mask; assistant loss requires a generation-tagged HF chat template"
        )
    raw_mask = _single_sequence(encoded["assistant_masks"], "assistant_masks")
    if raw_mask.shape != ids.shape or not np.isin(raw_mask, [0, 1]).all():
        raise ValueError("Chat template returned an invalid assistant mask")
    mask = raw_mask.astype(bool)
    if not mask[1:].any():
        raise ValueError("No next-token assistant targets in this conversation")
    return ids, np.where(mask, ids, IGNORE_INDEX)


def truncate_chat(
    tokens: Sequence[int] | np.ndarray, targets: Sequence[int] | np.ndarray, sequence_length: int, loss_mode: str
) -> tuple[np.ndarray, np.ndarray]:
    """Right-truncate unshifted chat targets without adding or replacing EOS.

    The caller pads and applies the single next-token shift after this function returns, so a
    sample keeps at most ``sequence_length`` runtime positions.

    Raises:
        ValueError: If the sequences disagree, fewer than two tokens remain, or assistant loss has no
            target left after truncation.
    """
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    tokens = _input_ids(tokens)
    targets = _single_sequence(targets, "targets")
    if tokens.shape != targets.shape or (targets.size and targets.dtype.kind not in "iu"):
        raise ValueError("Chat tokens and targets must be equal-length integer sequences")
    if np.any((targets != IGNORE_INDEX) & (targets != tokens)):
        raise ValueError("Each chat target must equal its token ID or IGNORE_INDEX")
    tokens = tokens[: sequence_length + 1]
    targets = targets[: sequence_length + 1].astype(np.int64, copy=False)
    if len(tokens) < 2:
        raise ValueError("Explicit chat requires at least two tokens for next-token prediction")
    if loss_mode == "assistant" and not np.any(targets[1:] != IGNORE_INDEX):
        raise ValueError(
            f"No assistant targets remain after truncation to {sequence_length}; "
            "increase --seq-length or filter the source trajectory"
        )
    return tokens, targets


class ChatEncoder:
    """Encode one trajectory with the prepared tokenizer and its generation-tagged chat template."""

    def __init__(self, tokenizer_path: str | os.PathLike[str], loss_mode: str) -> None:
        """Load the tokenizer offline from the directory prepared by script 01."""
        from transformers import AutoTokenizer

        self.loss_mode = loss_mode
        self._hf = AutoTokenizer.from_pretrained(os.fspath(tokenizer_path), local_files_only=True)

    def encode(self, row: Mapping[str, Any], messages: Any = None) -> tuple[np.ndarray, np.ndarray]:
        """Return unshifted ``(ids, targets)`` for a trajectory row.

        Args:
            row: A prepared JSONL row; its ``tools`` and ``chat_template_kwargs`` are honored.
            messages: Replacement messages, for measuring a prefix of the same trajectory.
        """
        return tokenize_chat(
            self._hf,
            row["messages"] if messages is None else messages,
            tools=row.get("tools"),
            loss_mode=self.loss_mode,
            chat_template_kwargs=row.get("chat_template_kwargs"),
        )
