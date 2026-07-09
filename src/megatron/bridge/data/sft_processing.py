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

"""Shared declarative preprocessing and canonical examples for SFT backends."""

from __future__ import annotations

import string
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

import torch

from megatron.bridge.data.conversation_processing import get_processor_tokenizer, normalize_chat_conversation


class ChatSFTExample(TypedDict):
    """Canonical structured chat example."""

    conversation: list[dict[str, Any]]


class PromptCompletionSFTExample(TypedDict):
    """Canonical unformatted prompt-completion example."""

    prompt: str
    completion: str


CanonicalSFTExample = ChatSFTExample | PromptCompletionSFTExample


@dataclass(kw_only=True)
class ChatSFTPreprocessingConfig:
    """Declarative preprocessing for structured conversation examples.

    Args:
        loss_mode: Tokens that contribute to the loss. ``assistant`` trains all
            assistant turns, ``last_turn`` trains only the final assistant span,
            and ``full`` trains the complete rendered conversation.
    """

    loss_mode: Literal["assistant", "last_turn", "full"] = "assistant"

    def validate(self) -> None:
        """Validate chat preprocessing settings."""
        if self.loss_mode not in {"assistant", "last_turn", "full"}:
            raise ValueError("Chat SFT loss_mode must be assistant, last_turn, or full.")


@dataclass(kw_only=True)
class PromptCompletionSFTPreprocessingConfig:
    """Declarative preprocessing for unformatted prompt-completion examples.

    The prompt and completion are tokenized separately and concatenated without
    calling ``apply_chat_template``. ``prompt_template`` may contain only the
    ``{prompt}`` placeholder; raw column selection remains explicit through
    ``prompt_column`` and ``completion_column``.

    Args:
        prompt_column: Source column containing the prompt text.
        completion_column: Source column containing the completion text.
        prompt_template: Formatting applied to the prompt before tokenization.
        separator: Text inserted between the rendered prompt and completion.
        loss_mode: ``completion`` masks the prompt; ``full`` trains all tokens.
        strip_whitespace: Strip leading and trailing ASCII spaces from both fields.
        add_bos: Add the tokenizer BOS token before the prompt.
        add_sep: Add the tokenizer SEP token between prompt and completion.
        add_eos: Add the tokenizer EOS token after the completion.
        truncation_method: Side retained when a prompt or completion must be truncated.
    """

    prompt_column: str = "prompt"
    completion_column: str = "completion"
    prompt_template: str = "{prompt}"
    separator: str = ""
    loss_mode: Literal["completion", "full"] = "completion"
    strip_whitespace: bool = True
    add_bos: bool = False
    add_sep: bool = False
    add_eos: bool = True
    truncation_method: Literal["left", "right"] = "right"

    def validate(self) -> None:
        """Validate prompt-completion preprocessing settings."""
        if not isinstance(self.prompt_column, str) or not self.prompt_column.strip():
            raise ValueError("prompt_column must be a non-empty string.")
        if not isinstance(self.completion_column, str) or not self.completion_column.strip():
            raise ValueError("completion_column must be a non-empty string.")
        if self.prompt_column == self.completion_column:
            raise ValueError("prompt_column and completion_column must be different.")
        if not isinstance(self.prompt_template, str) or not self.prompt_template:
            raise ValueError("prompt_template must be a non-empty string.")
        try:
            fields = [
                field_name
                for _, field_name, _, _ in string.Formatter().parse(self.prompt_template)
                if field_name is not None
            ]
            self.prompt_template.format(prompt="")
        except (IndexError, KeyError, ValueError) as error:
            raise ValueError("prompt_template may contain only the {prompt} placeholder.") from error
        if not fields or any(field_name != "prompt" for field_name in fields):
            raise ValueError("prompt_template must contain the {prompt} placeholder.")
        if not isinstance(self.separator, str):
            raise TypeError("separator must be a string.")
        if self.loss_mode not in {"completion", "full"}:
            raise ValueError("Prompt-completion SFT loss_mode must be completion or full.")
        if not all(
            isinstance(value, bool) for value in (self.strip_whitespace, self.add_bos, self.add_sep, self.add_eos)
        ):
            raise TypeError("strip_whitespace, add_bos, add_sep, and add_eos must be booleans.")
        if self.truncation_method not in {"left", "right"}:
            raise ValueError("truncation_method must be left or right.")


SFTPreprocessingConfig = ChatSFTPreprocessingConfig | PromptCompletionSFTPreprocessingConfig


@dataclass(frozen=True, kw_only=True)
class TokenizedPromptCompletion:
    """Tokenized prompt-completion row with an unshifted loss mask."""

    input_ids: torch.Tensor
    loss_mask: torch.Tensor
    prompt_ids: torch.Tensor
    completion_ids: torch.Tensor


_CONVERSATION_KEYS = ("messages", "conversation", "conversations")
_CANONICAL_PROMPT_KEY = "prompt"
_CANONICAL_COMPLETION_KEY = "completion"
_MEDIA_KEYS = (
    "image",
    "images",
    "image_path",
    "image_paths",
    "video",
    "videos",
    "video_path",
    "video_paths",
    "audio",
    "audio_path",
    "audio_paths",
)


def validate_sft_preprocessing_config(config: SFTPreprocessingConfig) -> None:
    """Validate one supported SFT preprocessing variant."""
    if not isinstance(config, (ChatSFTPreprocessingConfig, PromptCompletionSFTPreprocessingConfig)):
        raise TypeError("preprocessing must be ChatSFTPreprocessingConfig or PromptCompletionSFTPreprocessingConfig.")
    config.validate()


def _canonical_prompt_completion_values(example: Mapping[str, Any]) -> tuple[str, str] | None:
    if _CANONICAL_PROMPT_KEY not in example or _CANONICAL_COMPLETION_KEY not in example:
        return None
    prompt = example[_CANONICAL_PROMPT_KEY]
    completion = example[_CANONICAL_COMPLETION_KEY]
    if not isinstance(prompt, str) or not isinstance(completion, str):
        raise ValueError("Canonical prompt and completion values must be strings.")
    return prompt, completion


def normalize_sft_example(
    example: Mapping[str, Any],
    preprocessing: SFTPreprocessingConfig,
) -> dict[str, Any]:
    """Normalize one adapted row for the selected preprocessing semantics.

    Prompt-completion rows may be promoted to a two-turn chat. Structured
    conversations are deliberately not flattened into prompt-completion text,
    because doing so without a chat template would silently change their model
    semantics.
    """
    validate_sft_preprocessing_config(preprocessing)
    row = dict(example)
    canonical_pair = _canonical_prompt_completion_values(row)
    has_conversation = any(row.get(key) is not None for key in _CONVERSATION_KEYS)

    if canonical_pair is not None and has_conversation:
        raise ValueError("SFT rows must select exactly one schema: structured chat or prompt-completion.")

    if isinstance(preprocessing, ChatSFTPreprocessingConfig):
        if canonical_pair is not None:
            prompt, completion = canonical_pair
            metadata = {
                key: value
                for key, value in row.items()
                if key not in {_CANONICAL_PROMPT_KEY, _CANONICAL_COMPLETION_KEY}
            }
            return {
                "conversation": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion},
                ],
                **metadata,
            }
        conversation = normalize_chat_conversation(row)
        metadata = {key: value for key, value in row.items() if key not in _CONVERSATION_KEYS}
        # Keep the canonical singular key expected by registered VLM/audio
        # collators; shared text preprocessing accepts it as well.
        return {"conversation": conversation, **metadata}

    if has_conversation:
        raise ValueError(
            "Prompt-completion preprocessing requires paired text columns; structured conversations require "
            "ChatSFTPreprocessingConfig."
        )

    if preprocessing.prompt_column in row and preprocessing.completion_column in row:
        prompt = row[preprocessing.prompt_column]
        completion = row[preprocessing.completion_column]
    elif canonical_pair is not None:
        prompt, completion = canonical_pair
    else:
        raise ValueError(
            "Prompt-completion rows must contain the configured prompt/completion columns "
            f"({preprocessing.prompt_column}, {preprocessing.completion_column})."
        )
    if not isinstance(prompt, str) or not isinstance(completion, str):
        raise ValueError("Prompt-completion columns must contain strings.")
    metadata = {
        key: value
        for key, value in row.items()
        if key
        not in {
            preprocessing.prompt_column,
            preprocessing.completion_column,
            _CANONICAL_PROMPT_KEY,
            _CANONICAL_COMPLETION_KEY,
        }
    }
    return {
        preprocessing.prompt_column: prompt,
        preprocessing.completion_column: completion,
        **metadata,
    }


def normalize_sft_examples(
    examples: Sequence[Mapping[str, Any]],
    preprocessing: SFTPreprocessingConfig,
) -> list[dict[str, Any]]:
    """Normalize adapted rows for one explicit SFT preprocessing variant."""
    normalized = [normalize_sft_example(example, preprocessing) for example in examples]
    if not normalized:
        raise ValueError("SFT source produced no examples after preprocessing normalization.")
    return normalized


def is_text_only_prompt_completion_example(
    example: Mapping[str, Any],
    preprocessing: PromptCompletionSFTPreprocessingConfig,
) -> bool:
    """Return whether a row is a text-only prompt-completion example."""
    if any(example.get(key) is not None for key in _MEDIA_KEYS):
        return False
    try:
        normalize_sft_example(example, preprocessing)
    except (TypeError, ValueError):
        return False
    return True


def sft_example_metadata(
    example: Mapping[str, Any],
    preprocessing: SFTPreprocessingConfig,
) -> dict[str, Any]:
    """Return non-training columns retained as example metadata."""
    if isinstance(preprocessing, ChatSFTPreprocessingConfig):
        excluded = set(_CONVERSATION_KEYS)
    else:
        excluded = {
            preprocessing.prompt_column,
            preprocessing.completion_column,
            _CANONICAL_PROMPT_KEY,
            _CANONICAL_COMPLETION_KEY,
        }
    return {key: value for key, value in example.items() if key not in excluded}


def _tokenize_text(tokenizer_or_processor: Any, text: str) -> list[int]:
    tokenizer = get_processor_tokenizer(tokenizer_or_processor)
    try:
        if hasattr(tokenizer, "encode"):
            token_ids = tokenizer.encode(text, add_special_tokens=False)
        elif callable(tokenizer):
            token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        elif hasattr(tokenizer, "text_to_ids"):
            token_ids = tokenizer.text_to_ids(text)
        elif hasattr(tokenizer, "tokenize"):
            token_ids = tokenizer.tokenize(text)
        else:
            raise TypeError(f"Unsupported tokenizer type: {type(tokenizer_or_processor).__name__}")
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise ValueError("Unable to tokenize prompt-completion text.") from error
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.detach().cpu().tolist()
    if isinstance(token_ids, Sequence) and token_ids and isinstance(token_ids[0], Sequence):
        if len(token_ids) != 1:
            raise ValueError("Expected one tokenized prompt-completion sequence.")
        token_ids = token_ids[0]
    return [int(token_id) for token_id in token_ids]


def _is_space_sensitive(tokenizer_or_processor: Any) -> bool:
    """Return whether a leading word-boundary space changes tokenization."""
    tokenizer = get_processor_tokenizer(tokenizer_or_processor)
    try:
        declared = getattr(tokenizer, "space_sensitive", None)
    except (AttributeError, NotImplementedError):
        declared = None
    if declared is not None:
        return bool(declared)
    return _tokenize_text(tokenizer, "x y") != [*_tokenize_text(tokenizer, "x"), *_tokenize_text(tokenizer, "y")]


def _token_id(tokenizer_or_processor: Any, *names: str) -> int | None:
    tokenizer = get_processor_tokenizer(tokenizer_or_processor)
    for owner in (tokenizer_or_processor, tokenizer):
        for name in names:
            try:
                value = getattr(owner, name)
            except AttributeError:
                continue
            if callable(value):
                value = value()
            if value is not None:
                return int(value)
    return None


def _truncate(ids: list[int], target_length: int, method: Literal["left", "right"]) -> list[int]:
    if target_length <= 0:
        return []
    if len(ids) <= target_length:
        return ids
    return ids[-target_length:] if method == "left" else ids[:target_length]


def tokenize_prompt_completion_example(
    example: Mapping[str, Any],
    tokenizer_or_processor: Any,
    preprocessing: PromptCompletionSFTPreprocessingConfig,
    *,
    max_length: int | None = None,
    skipped_tokens: torch.Tensor | None = None,
    allow_missing_completion: bool = False,
    prefix_token_ids: Sequence[int] = (),
    minimum_completion_length: int = 0,
    sep_token_id: int | None = None,
) -> TokenizedPromptCompletion:
    """Tokenize a prompt-completion row without applying a chat template."""
    preprocessing.validate()
    working_example = dict(example)
    if (
        allow_missing_completion
        and preprocessing.prompt_column in working_example
        and preprocessing.completion_column not in working_example
    ):
        working_example[preprocessing.completion_column] = ""
    normalized = normalize_sft_example(working_example, preprocessing)
    prompt_value = normalized[preprocessing.prompt_column]
    completion_value = normalized[preprocessing.completion_column]
    if preprocessing.strip_whitespace:
        prompt_value = prompt_value.strip(" ")
        completion_value = completion_value.strip(" ")
    prompt_text = preprocessing.prompt_template.format(prompt=prompt_value)
    separator = preprocessing.separator
    if separator.startswith(" ") and not _is_space_sensitive(tokenizer_or_processor):
        # Preserve GPTSFTDataset's established split-template behavior: a
        # non-space-sensitive tokenizer drops one boundary space, while a
        # space-sensitive tokenizer keeps it with the completion segment.
        separator = separator[1:]
    completion_text = separator + completion_value
    prompt_ids = _tokenize_text(tokenizer_or_processor, prompt_text)
    completion_ids = _tokenize_text(tokenizer_or_processor, completion_text)

    bos_id = None
    if preprocessing.add_bos:
        bos_id = _token_id(tokenizer_or_processor, "bos_id", "bos_token_id")
        if bos_id is None:
            raise ValueError("Prompt-completion preprocessing requested BOS, but the tokenizer has no BOS token.")
    sep_id = None
    if preprocessing.add_sep:
        sep_id = sep_token_id
        if sep_id is None:
            sep_id = _token_id(tokenizer_or_processor, "sep_id", "sep_token_id")
        if sep_id is None:
            raise ValueError("Prompt-completion preprocessing requested SEP, but the tokenizer has no SEP token.")
        sep_id = int(sep_id)
    eos_id = None
    if preprocessing.add_eos:
        eos_id = _token_id(tokenizer_or_processor, "eos_id", "eos_token_id", "eod")
        if eos_id is None:
            raise ValueError("Prompt-completion preprocessing requested EOS, but the tokenizer has no EOS token.")

    if max_length is not None:
        if max_length <= 0:
            raise ValueError("max_length must be greater than 0 when set.")
        special_token_count = (
            len(prefix_token_ids) + int(bos_id is not None) + int(sep_id is not None) + int(eos_id is not None)
        )
        if special_token_count > max_length:
            raise ValueError("max_length is too short for the requested prompt-completion special tokens.")
        body_budget = max_length - special_token_count
        overflow = len(prompt_ids) + max(len(completion_ids), minimum_completion_length) - body_budget
        if overflow > 0:
            prompt_truncation = min(len(prompt_ids), overflow)
            prompt_ids = _truncate(
                prompt_ids,
                len(prompt_ids) - prompt_truncation,
                preprocessing.truncation_method,
            )
            overflow -= prompt_truncation
        if overflow > 0:
            if preprocessing.truncation_method == "right":
                # Preserve GPTSFTDataset's generation reserve when prompt
                # truncation alone cannot fit an overlength right-truncated row.
                overflow += min(len(completion_ids), minimum_completion_length)
            completion_ids = _truncate(
                completion_ids,
                max(0, len(completion_ids) - overflow),
                preprocessing.truncation_method,
            )

    context_ids = [bos_id] if bos_id is not None else []
    context_ids += [int(token_id) for token_id in prefix_token_ids]
    context_ids += prompt_ids
    if sep_id is not None:
        context_ids.append(sep_id)
    target_ids = completion_ids + ([eos_id] if eos_id is not None else [])
    input_ids = torch.tensor([*context_ids, *target_ids], dtype=torch.long)
    if input_ids.numel() < 2:
        raise ValueError("Prompt-completion preprocessing must produce at least two tokens for next-token loss.")
    if preprocessing.loss_mode == "completion":
        loss_mask = torch.tensor([0] * len(context_ids) + [1] * len(target_ids), dtype=torch.bool)
    else:
        loss_mask = torch.ones(input_ids.numel(), dtype=torch.bool)
    if skipped_tokens is not None and skipped_tokens.numel() > 0:
        loss_mask &= ~torch.isin(input_ids, skipped_tokens.to(device=input_ids.device, dtype=torch.long))
    return TokenizedPromptCompletion(
        input_ids=input_ids,
        loss_mask=loss_mask,
        prompt_ids=torch.tensor(context_ids, dtype=torch.long),
        completion_ids=torch.tensor(completion_ids, dtype=torch.long),
    )
