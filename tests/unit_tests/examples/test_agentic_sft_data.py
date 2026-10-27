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

"""Chat encoding, truncation and JSONL access contracts shared by the agentic SFT scripts."""

import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest


pytestmark = pytest.mark.unit

_SOURCE = Path(__file__).resolve().parents[3] / "examples/agentic_sft/agentic_sft_data.py"
_SPEC = importlib.util.spec_from_file_location("agentic_sft_data_under_test", _SOURCE)
data = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = data
_SPEC.loader.exec_module(data)

GENERATION_TEMPLATE = "{% for m in messages %}{% generation %}{{ m.content }}{% endgeneration %}{% endfor %}"


class _Tokenizer:
    """Minimal Hugging Face-style tokenizer: one token per message, assistant messages in the mask."""

    def __init__(self, template=GENERATION_TEMPLATE, mask=None):
        self.template = template
        self.mask = mask
        self.calls = []

    def get_chat_template(self, tools=None):
        return self.template

    def apply_chat_template(self, messages, tokenize=True, return_dict=False, **kwargs):
        self.calls.append(kwargs)
        ids = list(range(10, 10 + len(messages)))
        if not return_dict:
            return ids
        encoded = {"input_ids": ids}
        if kwargs.get("return_assistant_tokens_mask"):
            encoded["assistant_masks"] = (
                self.mask if self.mask is not None else [int(m["role"] == "assistant") for m in messages]
            )
        return encoded


def _messages():
    return [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "follow-up"},
        {"role": "assistant", "content": "done"},
    ]


def test_assistant_targets_follow_the_generation_mask_without_shifting():
    ids, targets = data.tokenize_chat(_Tokenizer(), _messages())

    assert ids.tolist() == [10, 11, 12, 13]
    assert targets.tolist() == [-100, 11, -100, 13]


def test_full_loss_supervises_every_token_and_adds_no_eos():
    ids, targets = data.tokenize_chat(_Tokenizer(), _messages(), loss_mode="full")

    assert ids.tolist() == targets.tolist() == [10, 11, 12, 13]


def test_tools_and_template_controls_reach_the_template_and_pipeline_keys_are_protected():
    tokenizer = _Tokenizer()
    tools = [{"type": "function", "function": {"name": "check"}}]

    data.tokenize_chat(tokenizer, _messages(), tools=json.dumps(tools), chat_template_kwargs={"enable_thinking": True})

    assert tokenizer.calls[-1]["tools"] == tools
    assert tokenizer.calls[-1]["enable_thinking"] is True
    with pytest.raises(ValueError, match="cannot override"):
        data.tokenize_chat(tokenizer, _messages(), chat_template_kwargs={"tokenize": False})


def test_tool_call_arguments_are_normalized_to_objects():
    messages = _messages()
    messages[1]["tool_calls"] = [{"function": {"name": "check", "arguments": '{"x": 1}'}}]

    normalized, _ = data.normalize_chat(messages)

    assert normalized[1]["tool_calls"][0]["function"]["arguments"] == {"x": 1}
    assert isinstance(messages[1]["tool_calls"][0]["function"]["arguments"], str)  # caller's rows are not mutated


@pytest.mark.parametrize(
    ("messages", "match"),
    [
        ([], "nonempty list"),
        ([{"role": "narrator", "content": "x"}], "role must be one of"),
        ([{"role": "user", "content": [{"type": "image"}]}], "multimodal content is unsupported"),
        ([{"role": "assistant", "content": "x", "tool_calls": [{"function": {"arguments": "[1]"}}]}], "JSON object"),
        ("not json", "invalid JSON"),
    ],
)
def test_malformed_conversations_fail_with_a_reason(messages, match):
    with pytest.raises(ValueError, match=match):
        data.normalize_chat(messages)


def test_template_without_generation_block_is_a_template_error_not_an_empty_row():
    with pytest.raises(ValueError, match="generation-tagged HF training template"):
        data.tokenize_chat(_Tokenizer(template="{{ messages }}"), _messages())


@pytest.mark.parametrize("mask", [[0, 0, 0, 0], [1, 0, 0, 0]])
def test_rows_without_next_token_assistant_targets_fail_loudly(mask):
    with pytest.raises(ValueError, match="No next-token assistant targets in this conversation"):
        data.tokenize_chat(_Tokenizer(mask=mask), _messages())


def test_invalid_mask_shape_is_rejected():
    with pytest.raises(ValueError, match="invalid assistant mask"):
        data.tokenize_chat(_Tokenizer(mask=[0, 1, 0]), _messages())


def test_truncation_keeps_sequence_length_plus_one_tokens_and_adds_no_eos():
    tokens = np.arange(10)
    targets = np.where(np.arange(10) % 2, np.arange(10), -100)

    kept_tokens, kept_targets = data.truncate_chat(tokens, targets, 5, "assistant")

    assert kept_tokens.tolist() == [0, 1, 2, 3, 4, 5]
    assert kept_targets.tolist() == [-100, 1, -100, 3, -100, 5]


def test_truncation_that_removes_every_assistant_target_names_the_remedy():
    tokens = np.arange(10)
    targets = np.full(10, -100)
    targets[8] = 8

    with pytest.raises(ValueError, match="No assistant targets remain after truncation to 4; increase --seq-length"):
        data.truncate_chat(tokens, targets, 4, "assistant")


def test_truncation_rejects_targets_that_do_not_match_their_tokens():
    with pytest.raises(ValueError, match="equal its token ID or IGNORE_INDEX"):
        data.truncate_chat(np.arange(4), np.array([-100, 9, -100, 3]), 3, "assistant")


def _write_rows(path, lines):
    path.write_bytes(b"".join(lines))


def test_jsonl_rows_index_nonblank_lines_and_report_physical_locations(tmp_path):
    path = tmp_path / "rows.jsonl"
    _write_rows(path, [b"\n", b'{"a": 1}\n', b"\n", b'{"a": 2, "b": "\xe4\xb8\x96"}\n'])

    rows = data.JsonlRows(path)

    assert len(rows) == 2
    assert rows[1] == {"a": 2, "b": "\u4e16"}
    assert rows.location(1) == f"{path}:4"
    assert rows.column_names == ["a"]
    with pytest.raises(IndexError):
        rows[2]


def test_jsonl_rows_survive_pickling_and_reject_modified_files(tmp_path):
    path = tmp_path / "rows.jsonl"
    _write_rows(path, [b'{"a": 1}\n', b'{"a": 2}\n'])
    rows = data.JsonlRows(path)
    assert rows[0] == {"a": 1}

    restored = pickle.loads(pickle.dumps(rows))
    assert restored[1] == {"a": 2}

    path.write_bytes(path.read_bytes() + b'{"a": 3}\n')
    with pytest.raises(ValueError, match="changed after indexing"):
        restored[0]


@pytest.mark.parametrize("content", [b"", b"\n\n", b"[1]\n", b"not json\n"])
def test_jsonl_rows_reject_empty_and_non_object_files(tmp_path, content):
    path = tmp_path / "rows.jsonl"
    path.write_bytes(content)

    with pytest.raises(ValueError):
        data.JsonlRows(path)
