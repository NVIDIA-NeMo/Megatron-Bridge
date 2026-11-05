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

"""Text SFT rows to ERNIE 5.5 training samples, encoded as the reference implementation does.

Accepted rows, one JSON object per line (the text formats of Megatron-Bridge's text-only SFT tutorial):

* chat: ``{"messages": [{"role": ..., "content": ...}, ...]}``, optionally with ``"tools"``;
* prompt/completion: ``{"input": "...", "output": "..."}`` (column names configurable), encoded as one user turn and
  one assistant turn.

:func:`encode_ernie55_sft_row` returns ``{"input_ids", "labels"}`` with Hugging Face labels (``labels[t] ==
input_ids[t]`` where supervised, ``-100`` elsewhere), the pre-tokenized format of
:class:`~megatron.bridge.models.ernie.ernie55_data.Ernie55SFTDatasetConfig`.

The encoding mirrors the reference's ``Ernie5_5Template._encode``: it renders ``ernie55_chat_template.jinja`` (the
reference's ``ernie5_5.jinja``: the checkpoint's chat template with ``{% generation %}`` spans around what SFT
supervises) through ``tokenizer.apply_chat_template(..., return_assistant_tokens_mask=True)``, supervises exactly the
generation spans and never the first token. String tool-call arguments are decoded to JSON objects, ``tool_call``
messages are folded into the preceding assistant turn, and a trailing assistant message without content is dropped. The
template is the reference's file with its special tags in the released tokenizer's names (sha256 b23eda7a...):
``<|ernie:im_start|>``, ``<|ernie:im_end|>``, ``<ernie:system_setting>``, ``<ernie:global_setting>``,
``<ernie:tool_list>``, ``<ernie:tool_output>`` and ``<ernie:think>`` (and their closing tags) for the reference's
``<|im_start|>``, ..., ``<think>``, with the same token ids; nothing else differs. Per the reference's documentation,
``<ernie:think>`` belongs to the prompt, ``"loss": false`` on a history assistant message keeps it out of the loss, and
``enable_thinking`` switches thinking; reasoning inlined in message content is read between ``<ernie:think>`` and
``</ernie:think>``.

The tokenizer must be the released ERNIE 5.5 checkpoint's (``tokenizer.json`` and ``tokenizer_config.json``): with any
other tokenizer the token ids, and the splitting of ERNIE's special tokens, differ from the reference. A tokenizer
with the reference's tag names needs the reference's template (``chat_template``).
"""

from __future__ import annotations

import json
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal


__all__ = [
    "ERNIE55_CHAT_TEMPLATE_PATH",
    "encode_ernie55_sft_row",
    "ernie55_sft_messages",
    "load_ernie55_chat_template",
]

ERNIE55_CHAT_TEMPLATE_PATH = Path(__file__).with_name("ernie55_chat_template.jinja")
"""ERNIE 5.5's SFT chat template: the reference implementation's ``ernie5_5.jinja`` in the
released tokenizer's tag names."""

_IGNORE = -100


@lru_cache(maxsize=8)
def load_ernie55_chat_template(path: str | None = None) -> str:
    """The chat template text: ``path``, or the reference's template when ``None``."""
    return Path(path or ERNIE55_CHAT_TEMPLATE_PATH).read_text(encoding="utf-8")


def ernie55_sft_messages(
    row: dict[str, Any], *, prompt_column: str = "input", completion_column: str = "output"
) -> list[dict[str, Any]]:
    """The row's conversation as messages, normalized as the reference does.

    Args:
        row: A chat row (``messages``) or a prompt/completion row.
        prompt_column: Prompt column of prompt/completion rows.
        completion_column: Completion column of prompt/completion rows.

    Returns:
        The messages to render.

    Raises:
        ValueError: If the row is neither format, or the conversation is empty.
    """
    if "messages" in row:
        messages = deepcopy(row["messages"])
    elif prompt_column in row and completion_column in row:
        messages = [
            {"role": "user", "content": row[prompt_column]},
            {"role": "assistant", "content": row[completion_column]},
        ]
    else:
        raise ValueError(
            f"An ERNIE 5.5 SFT row needs 'messages' or '{prompt_column}'/'{completion_column}'; got keys {sorted(row)}."
        )
    restored: list[dict[str, Any]] = []
    for message in messages:
        if message["role"] == "tool_call":
            call = message["content"]
            if isinstance(call, str):
                call = json.loads(call)
            if not restored or restored[-1]["role"] != "assistant":
                restored.append({"role": "assistant", "content": ""})
            restored[-1].setdefault("tool_calls", []).append({"type": "function", "function": call})
        else:
            restored.append(message)
    for message in restored:
        for call in message.get("tool_calls") or []:
            call = call.get("function") or call
            # The template reads arguments as a mapping; Jinja in transformers has no fromjson filter.
            if isinstance(call.get("arguments"), str):
                call["arguments"] = json.loads(call["arguments"])
    while restored and restored[-1]["role"] == "assistant" and restored[-1].get("content") is None:
        restored.pop()
    if not restored:
        raise ValueError("An ERNIE 5.5 SFT row has an empty conversation.")
    return restored


def encode_ernie55_sft_row(
    tokenizer: Any,
    row: dict[str, Any],
    *,
    chat_template: str | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
    prompt_column: str = "input",
    completion_column: str = "output",
    loss: Literal["assistant", "all"] = "assistant",
) -> dict[str, list[int]]:
    """One text row as an ERNIE 5.5 training sample.

    Args:
        tokenizer: A Hugging Face tokenizer (ERNIE 5.5's own for real runs).
        row: A chat row or a prompt/completion row (see the module docstring).
        chat_template: Template text; ``None`` uses the reference's template.
        chat_template_kwargs: Extra template variables, such as ``enable_thinking``.
        prompt_column: Prompt column of prompt/completion rows.
        completion_column: Completion column of prompt/completion rows.
        loss: ``"assistant"`` supervises the template's generation spans (the reference's default); ``"all"`` every
            token but the first.

    Returns:
        ``{"input_ids": [...], "labels": [...]}`` with Hugging Face labels.
    """
    messages = ernie55_sft_messages(row, prompt_column=prompt_column, completion_column=completion_column)
    kwargs = dict(chat_template_kwargs or {})
    if row.get("tools"):
        kwargs["tools"] = row["tools"]
    out = tokenizer.apply_chat_template(
        messages,
        chat_template=chat_template if chat_template is not None else load_ernie55_chat_template(),
        tokenize=True,
        add_generation_prompt=messages[-1]["role"] != "assistant",
        return_dict=True,
        return_assistant_tokens_mask=True,
        **kwargs,
    )
    input_ids = [int(token) for token in out["input_ids"]]
    if loss == "all":
        labels = list(input_ids)
    else:
        labels = [token if supervised else _IGNORE for token, supervised in zip(input_ids, out["assistant_masks"])]
    if labels:
        labels[0] = _IGNORE
    return {"input_ids": input_ids, "labels": labels}
