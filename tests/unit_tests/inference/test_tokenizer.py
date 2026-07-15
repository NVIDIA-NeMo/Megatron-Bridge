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

"""Unit tests for the shared HFTokenizerAdapter."""

from __future__ import annotations

from megatron.bridge.inference._tokenizer import HFTokenizerAdapter


class _FakeTokenizer:
    """Minimal stand-in for a HuggingFace tokenizer."""

    def __init__(self):
        self.eos_token = "</s>"
        self.eos_token_id = 2
        self.bos_token_id = 1
        self.pad_token = None
        self.decode_calls = []

    def __len__(self):
        return 32000

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(c) for c in text]

    def decode(self, tokens, skip_special_tokens=True):
        self.decode_calls.append(skip_special_tokens)
        return f"decoded({list(tokens)},sst={skip_special_tokens})"


class _FakeChatTokenizer(_FakeTokenizer):
    """HF tokenizer with the chat-template contract used by MCore's server."""

    def __init__(self):
        super().__init__()
        self.chat_template = "{{ messages }}"
        self.chat_template_calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.chat_template_calls.append((messages, kwargs))
        return [101, 102]


def test_text_generation_defaults():
    tok = _FakeTokenizer()
    adapter = HFTokenizerAdapter(tok)

    assert adapter.eod == 2
    assert adapter.bos == 1
    assert adapter.vocab_size == 32000
    # pad token defaulted to eos when unset
    assert tok.pad_token == "</s>"
    assert adapter.tokenize("AB") == [65, 66]
    # default honors caller-supplied skip_special_tokens
    adapter.detokenize([65], skip_special_tokens=True)
    adapter.detokenize([65], skip_special_tokens=False)
    assert tok.decode_calls == [True, False]


def test_vlm_preserving_flags():
    """VLM wrapper semantics: no pad mutation, vocab_size None, always keep special tokens."""
    tok = _FakeTokenizer()
    adapter = HFTokenizerAdapter(
        tok,
        set_pad_token=False,
        expose_vocab_size=False,
        force_skip_special_tokens=False,
    )

    assert adapter.vocab_size is None
    assert tok.pad_token is None  # not defaulted
    # force flag overrides whatever the controller passes
    adapter.detokenize([65], skip_special_tokens=True)
    assert tok.decode_calls == [False]


def test_force_skip_special_tokens_none_honors_caller():
    tok = _FakeTokenizer()
    adapter = HFTokenizerAdapter(tok, force_skip_special_tokens=None)
    adapter.detokenize([65], skip_special_tokens=False)
    assert tok.decode_calls == [False]


def test_openai_server_preserves_hf_chat_template_contract():
    """The server adapter must expose an available HF chat template to MCore."""
    tok = _FakeChatTokenizer()
    adapter = HFTokenizerAdapter(tok)
    messages = [{"role": "system", "content": "Be concise."}, {"role": "user", "content": "Hello"}]

    assert adapter.chat_template == tok.chat_template
    assert adapter.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        tools=[{"type": "function"}],
    ) == [101, 102]
    assert tok.chat_template_calls == [
        (
            messages,
            {
                "tokenize": True,
                "add_generation_prompt": True,
                "tools": [{"type": "function"}],
            },
        )
    ]
