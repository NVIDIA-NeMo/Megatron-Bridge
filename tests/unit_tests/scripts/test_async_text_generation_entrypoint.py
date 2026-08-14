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

"""Unit tests for the asynchronous text-generation entrypoint."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import sys
import types
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[3]
_MODULE_PATH = _REPO_ROOT / "scripts" / "inference" / "async_text_generation.py"


def _module(name: str, **attrs: object) -> types.ModuleType:
    module = types.ModuleType(name)
    for attr_name, value in attrs.items():
        setattr(module, attr_name, value)
    return module


def _add_no_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    return parser


class _AsyncLLM:
    is_primary_rank = True

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs

    async def __aenter__(self) -> _AsyncLLM:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def generate(self, prompt: str, sampling_params: object) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            generated_text=f"generated from {prompt}",
            prompt_log_probs=[-0.1, -0.2],
            generated_log_probs=[-0.3],
            prompt_top_n_logprobs=[{"prompt-token": -0.1}],
            generated_top_n_logprobs=[{"generated-token": -0.3}],
            failed=lambda: False,
        )


@pytest.fixture
def async_text_generation_entrypoint(monkeypatch: pytest.MonkeyPatch):
    shared_helpers = {
        "HFTokenizerAdapter": object,
        "add_distributed_args": _add_no_args,
        "add_engine_args": _add_no_args,
        "add_model_loading_args": _add_no_args,
        "add_parallelism_args": _add_no_args,
        "add_prompt_args": _add_no_args,
        "add_sampling_args": _add_no_args,
        "build_inference_config": lambda **kwargs: kwargs,
        "build_sampling_params": lambda **kwargs: kwargs,
        "build_tokenizer": lambda *args: object(),
        "load_bridge_model": lambda **kwargs: kwargs,
        "load_prompts": lambda *args: list(args),
        "resolve_hf_model_path": lambda *args: args[0],
    }
    stubs = {
        "megatron.core.inference.apis": _module(
            "megatron.core.inference.apis",
            MegatronAsyncLLM=_AsyncLLM,
            SamplingParams=object,
        ),
        "megatron.bridge.inference.text_generation": _module(
            "megatron.bridge.inference.text_generation",
            **shared_helpers,
        ),
        "megatron.bridge.utils.activation_map": _module(
            "megatron.bridge.utils.activation_map",
            str_to_dtype=lambda value: value,
        ),
        "megatron.bridge.utils.common_utils": _module(
            "megatron.bridge.utils.common_utils",
            maybe_initialize_distributed=lambda timeout: timeout,
            print_rank_0=lambda message: message,
        ),
    }
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location("async_text_generation_entrypoint_under_test", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)


@pytest.mark.unit
def test_generate_prints_requested_log_probabilities(
    async_text_generation_entrypoint: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages: list[str] = []
    monkeypatch.setattr(async_text_generation_entrypoint, "print_rank_0", messages.append)
    args = types.SimpleNamespace(
        max_seq_length=32,
        max_new_tokens=2,
        max_batch_size=None,
        tp=1,
        block_size_tokens=8,
        kv_cache_buffer_size_gb=1.0,
        max_tokens=None,
        return_log_probs=True,
        enable_chunked_prefill=False,
        coordinator_host=None,
        coordinator_port=None,
    )
    tokenizer = types.SimpleNamespace(tokenize=lambda prompt: [1, 2])

    asyncio.run(
        async_text_generation_entrypoint._generate(
            args,
            model=object(),
            tokenizer=tokenizer,
            prompts=["prompt"],
            sampling_params=object(),
        )
    )

    rendered = "\n".join(messages)
    assert "Prompt log probs: [-0.1, -0.2]" in rendered
    assert "Generated log probs: [-0.3]" in rendered
    assert "Prompt top-n logprobs: [{'prompt-token': -0.1}]" in rendered
    assert "Generated top-n logprobs: [{'generated-token': -0.3}]" in rendered


@pytest.mark.unit
def test_generate_rejects_failed_inference_requests(
    async_text_generation_entrypoint: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _failed_generate(_self, _prompt, _sampling_params):
        return types.SimpleNamespace(
            request_id=7,
            status=types.SimpleNamespace(name="FAILED"),
            generated_text="",
            failed=lambda: True,
        )

    monkeypatch.setattr(_AsyncLLM, "generate", _failed_generate)
    args = types.SimpleNamespace(
        max_seq_length=32,
        max_new_tokens=2,
        max_batch_size=None,
        tp=1,
        block_size_tokens=8,
        kv_cache_buffer_size_gb=1.0,
        max_tokens=None,
        return_log_probs=False,
        enable_chunked_prefill=False,
        coordinator_host=None,
        coordinator_port=None,
    )
    tokenizer = types.SimpleNamespace(tokenize=lambda prompt: [1, 2])

    with pytest.raises(RuntimeError, match="request 7.*FAILED"):
        asyncio.run(
            async_text_generation_entrypoint._generate(
                args,
                model=object(),
                tokenizer=tokenizer,
                prompts=["prompt"],
                sampling_params=object(),
            )
        )
