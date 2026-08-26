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

"""Contract tests for the OpenAI-compatible chat-completions endpoint."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints.chat_completions import (
    bp,
)
from quart import Quart


class _Tokenizer:
    """Minimal tokenizer for the endpoint's plain-text fallback."""

    chat_template = None
    bos = None

    def tokenize(self, text: str) -> list[int]:
        return [len(text)]


class _RecordingClient:
    """Record submitted sampling parameters without running inference."""

    def __init__(self) -> None:
        self.sampling_params: Any = None

    async def add_request(self, prompt_tokens: list[int], sampling_params: Any, **kwargs: Any) -> dict[str, Any]:
        self.sampling_params = sampling_params
        return {
            "status": "FAILED",
            "events": [{"type": "ERROR_TRANSIENT", "payload": "test sentinel"}],
        }


@pytest.mark.unit
@pytest.mark.parametrize(
    ("stop", "expected_stop_words"),
    [
        pytest.param("END", ["END"], id="string"),
        pytest.param(["END", "DONE"], ["END", "DONE"], id="list"),
        pytest.param(None, None, id="omitted"),
    ],
)
def test_chat_completions_forwards_stop_words(stop: object, expected_stop_words: object) -> None:
    """The OpenAI stop field reaches the engine sampling parameters."""
    app = Quart(__name__)
    app.register_blueprint(bp)
    recording_client = _RecordingClient()
    app.config.update(
        client=recording_client,
        tokenizer=_Tokenizer(),
        parsers=[],
        verbose=False,
    )
    payload = {
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 4,
        "prevent_retokenization": False,
    }
    if stop is not None:
        payload["stop"] = stop

    response = asyncio.run(app.test_client().post("/v1/chat/completions", json=payload))

    assert response.status_code == 500
    assert recording_client.sampling_params.stop_words == expected_stop_words
