# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Unit tests for the unified diffusion inference CLI."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
from types import SimpleNamespace

import pytest


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_CLI_PATH = _REPO_ROOT / "examples" / "models" / "diffusion" / "inference_dllm.py"
_PAD_TOKEN_ID = 0
_PROMPT_TOKEN_IDS = {
    "short": [11, 12],
    "long": [21, 22, 23, 24],
}


@pytest.fixture(scope="module")
def cli():
    """Load the example CLI as a module under a stable test name."""
    spec = importlib.util.spec_from_file_location("inference_dllm_under_test", _CLI_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)


class _FakeTokenizer:
    chat_template = None

    def __call__(self, texts, *, return_tensors, padding, padding_side):
        assert return_tensors == "pt"
        assert padding is True
        width = max(len(_PROMPT_TOKEN_IDS[text]) for text in texts)
        rows = []
        for text in texts:
            token_ids = _PROMPT_TOKEN_IDS[text]
            pads = [_PAD_TOKEN_ID] * (width - len(token_ids))
            rows.append(pads + token_ids if padding_side == "left" else token_ids + pads)
        return SimpleNamespace(input_ids=rows)


def _distance_from_last_prompt_token_to_generation(row):
    last_prompt_index = max(index for index, token_id in enumerate(row) if token_id != _PAD_TOKEN_ID)
    generation_start = len(row)
    return generation_start - last_prompt_index


def test_mixed_length_batch_preserves_prompt_to_generation_distance(cli):
    tokenizer = _FakeTokenizer()
    short_alone = cli._encode_prompts(tokenizer, ["short"], use_chat_template=False, model="llada15").input_ids[0]
    batched = cli._encode_prompts(
        tokenizer,
        ["short", "long"],
        use_chat_template=False,
        model="llada15",
    ).input_ids

    standalone_distance = _distance_from_last_prompt_token_to_generation(short_alone)
    assert standalone_distance == 1
    assert _distance_from_last_prompt_token_to_generation(batched[0]) == standalone_distance
    assert _distance_from_last_prompt_token_to_generation(batched[1]) == standalone_distance


def test_nemotron_retains_right_padding(cli):
    batched = cli._encode_prompts(
        _FakeTokenizer(),
        ["short", "long"],
        use_chat_template=False,
        model="nemotron_labs_diffusion",
    ).input_ids

    assert batched[0] == [11, 12, _PAD_TOKEN_ID, _PAD_TOKEN_ID]
    assert batched[1] == _PROMPT_TOKEN_IDS["long"]
