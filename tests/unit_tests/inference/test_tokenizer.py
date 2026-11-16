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

import asyncio

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

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

    assert adapter.chat_template is None
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
    assert adapter.eos_id == tok.eos_token_id
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


def test_openai_streaming_accepts_hf_fast_tokenizer_adapter():
    incremental_detokenizer = pytest.importorskip(
        "megatron.core.inference.text_generation_server.dynamic_text_gen_server.incremental_detokenizer",
        reason="MCore does not provide the incremental detokenizer yet",
    )
    backend = Tokenizer(models.WordLevel(vocab={"[UNK]": 0, "hello": 1}, unk_token="[UNK]"))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", eos_token="[UNK]")
    adapter = HFTokenizerAdapter(tokenizer)

    detokenizer = incremental_detokenizer.HuggingFaceFastIncrementalDetokenizer(adapter, [])

    assert detokenizer.text == ""
    assert detokenizer.update([tokenizer.eos_token_id]) == ""


def _bos_tokenizer():
    """A fast tokenizer whose post-processor adds BOS, as Llama's, Gemma's and Mistral's do."""
    backend = Tokenizer(models.WordLevel(vocab={"[UNK]": 0, "<s>": 1, "</s>": 2, "hello": 3}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(single="<s> $A", special_tokens=[("<s>", 1)])
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", bos_token="<s>", eos_token="</s>")


def test_tokenize_adds_special_tokens_only_when_asked():
    tokenizer = _bos_tokenizer()
    assert tokenizer("hello hello").input_ids == [1, 3, 3]
    assert HFTokenizerAdapter(tokenizer).tokenize("hello hello") == [3, 3]
    assert HFTokenizerAdapter(tokenizer, add_special_tokens=True).tokenize("hello hello") == [1, 3, 3]


def test_build_tokenizer_tokenizes_text_like_hugging_face(monkeypatch):
    from megatron.bridge.inference import text_generation

    tokenizer = _bos_tokenizer()
    monkeypatch.setattr(text_generation.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)

    adapter = text_generation.build_tokenizer("org/model", trust_remote_code=False)

    assert adapter.tokenize("hello hello") == tokenizer("hello hello").input_ids


class _RecordingClient:
    """Records the prompt ids an endpoint submits and answers with one token."""

    def __init__(self):
        self.prompt_tokens = []

    def add_request_with_id(self, prompt_tokens, sampling_params, *, multi_modal_data=None, offload_params=None):
        self.prompt_tokens.append(list(prompt_tokens))
        future = asyncio.get_running_loop().create_future()
        future.set_result(
            {
                "uid": "req-1",
                "status": "COMPLETED",
                "events": [],
                "prompt_tokens": list(prompt_tokens),
                "prompt_length": len(prompt_tokens),
                "generated_tokens": [3],
                "generated_log_probs": None,
                "routing_indices": None,
                "num_cached_tokens": 0,
                "sampling_params": {"num_tokens_to_generate": 1},
            }
        )
        return len(self.prompt_tokens), future

    def abort_request(self, request_id):
        pass


def test_openai_completions_prompt_starts_with_bos(monkeypatch):
    """MCore's /v1/completions submits the adapter's ids as they are, so they need the BOS."""
    quart = pytest.importorskip("quart")
    completions = pytest.importorskip(
        "megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints.completions"
    )
    from megatron.core.inference.config import MultimodalPromptConfig

    from megatron.bridge.inference import text_generation

    tokenizer = _bos_tokenizer()
    monkeypatch.setattr(text_generation.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)
    client = _RecordingClient()
    app = quart.Quart(__name__)
    app.config.update(
        client=client,
        tokenizer=text_generation.build_tokenizer("org/model", trust_remote_code=False),
        parsers=[],
        verbose=False,
        multimodal_prompt_config=MultimodalPromptConfig(),
        eval_mode=True,
    )
    app.register_blueprint(completions.bp)

    async def post():
        return await app.test_client().post("/v1/completions", json={"prompt": "hello hello", "max_tokens": 1})

    response = asyncio.run(post())

    assert response.status_code == 200
    assert client.prompt_tokens == [tokenizer("hello hello").input_ids]
