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

"""Unit tests for the synchronous text-generation entrypoint."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[3]
_MODULE_PATH = _REPO_ROOT / "scripts" / "inference" / "text_generation.py"


class _PassthroughInit:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.args = args
        self.kwargs = kwargs


def _module(name: str, **attrs: object) -> types.ModuleType:
    module = types.ModuleType(name)
    for attr_name, value in attrs.items():
        setattr(module, attr_name, value)
    return module


def _add_no_args(parser: object) -> object:
    return parser


@pytest.fixture
def text_generation_entrypoint(monkeypatch: pytest.MonkeyPatch):
    shared_helpers = {
        "HFTokenizerAdapter": _PassthroughInit,
        "add_distributed_args": _add_no_args,
        "add_engine_args": _add_no_args,
        "add_model_loading_args": _add_no_args,
        "add_parallelism_args": _add_no_args,
        "add_prompt_args": _add_no_args,
        "add_sampling_args": _add_no_args,
        "build_inference_config": lambda **kwargs: kwargs,
        "build_sampling_params": lambda **kwargs: kwargs,
        "build_tokenizer": _PassthroughInit,
        "load_bridge_model": lambda **kwargs: kwargs,
        "load_prompts": lambda *args: list(args),
        "resolve_hf_model_path": lambda *args: args[0],
        "validate_sequence_length": lambda **kwargs: None,
    }
    stubs = {
        "megatron.core.inference.apis": _module(
            "megatron.core.inference.apis",
            MegatronLLM=_PassthroughInit,
            SamplingParams=_PassthroughInit,
        ),
        "megatron.core.inference.contexts": _module(
            "megatron.core.inference.contexts",
            StaticInferenceContext=_PassthroughInit,
        ),
        "megatron.core.inference.engines.static_engine": _module(
            "megatron.core.inference.engines.static_engine",
            StaticInferenceEngine=_PassthroughInit,
        ),
        "megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper": _module(
            "megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper",
            GPTInferenceWrapper=_PassthroughInit,
        ),
        "megatron.core.inference.text_generation_controllers.text_generation_controller": _module(
            "megatron.core.inference.text_generation_controllers.text_generation_controller",
            TextGenerationController=_PassthroughInit,
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

    spec = importlib.util.spec_from_file_location("text_generation_entrypoint_under_test", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)


def _args(**overrides: object) -> types.SimpleNamespace:
    values = {
        "use_legacy_generation": True,
        "use_coordinator": False,
        "ep": 1,
        "coordinator_host": None,
        "coordinator_port": None,
        "top_n_logprobs": 0,
        "return_log_probs": False,
        "distributed_timeout_minutes": 10,
        "termination_id": None,
        "stop_words": None,
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


@pytest.mark.unit
@pytest.mark.parametrize(
    "stopping_override",
    [
        {"termination_id": 42},
        {"stop_words": ["<END>"]},
    ],
)
def test_legacy_static_rejects_unsupported_stopping_controls(
    text_generation_entrypoint: types.ModuleType,
    stopping_override: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="legacy static generation does not support"):
        text_generation_entrypoint._validate_args(_args(**stopping_override))


@pytest.mark.unit
def test_dynamic_generation_accepts_stopping_controls(text_generation_entrypoint: types.ModuleType) -> None:
    text_generation_entrypoint._validate_args(
        _args(
            use_legacy_generation=False,
            termination_id=42,
            stop_words=["<END>"],
        )
    )


@pytest.mark.unit
def test_print_results_includes_returned_log_probabilities(
    text_generation_entrypoint: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages: list[str] = []
    monkeypatch.setattr(text_generation_entrypoint, "print_rank_0", messages.append)
    output = types.SimpleNamespace(
        generated_text="generated",
        prompt_log_probs=[-0.1, -0.2],
        generated_log_probs=[-0.3],
        prompt_top_n_logprobs=[{"prompt-token": -0.1}],
        generated_top_n_logprobs=[{"generated-token": -0.3}],
    )

    text_generation_entrypoint._print_results(["prompt"], [output])

    rendered = "\n".join(messages)
    assert "Prompt log probs: [-0.1, -0.2]" in rendered
    assert "Generated log probs: [-0.3]" in rendered
    assert "Prompt top-n logprobs: [{'prompt-token': -0.1}]" in rendered
    assert "Generated top-n logprobs: [{'generated-token': -0.3}]" in rendered


@pytest.mark.unit
def test_dynamic_generation_rejects_failed_inference_requests(
    text_generation_entrypoint: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FailedLLM:
        is_primary_rank = True

        def __init__(self, **_kwargs: object) -> None:
            pass

        def __enter__(self) -> _FailedLLM:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def generate(self, _prompts: list[str], _sampling_params: object) -> list[types.SimpleNamespace]:
            return [
                types.SimpleNamespace(
                    request_id=7,
                    status=types.SimpleNamespace(name="FAILED"),
                    generated_text="",
                    failed=lambda: True,
                )
            ]

    monkeypatch.setattr(text_generation_entrypoint, "MegatronLLM", _FailedLLM)
    args = types.SimpleNamespace(
        max_new_tokens=1,
        max_seq_length=32,
        max_batch_size=None,
        tp=1,
        block_size_tokens=8,
        kv_cache_buffer_size_gb=1.0,
        max_tokens=1,
        return_log_probs=False,
        enable_chunked_prefill=False,
        use_coordinator=False,
        coordinator_host=None,
        coordinator_port=None,
    )
    tokenizer = types.SimpleNamespace(tokenize=lambda _prompt: [1, 2])

    with pytest.raises(RuntimeError, match="request 7.*FAILED"):
        text_generation_entrypoint._generate_with_dynamic_engine(
            args,
            model=object(),
            tokenizer=tokenizer,
            prompts=["Hello world"],
            sampling_params=object(),
        )
