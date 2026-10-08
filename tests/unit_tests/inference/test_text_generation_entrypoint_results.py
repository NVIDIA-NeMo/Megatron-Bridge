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

"""Unit tests for result completeness in the synchronous text-generation entrypoint."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[3]
_MODULE_PATH = _REPO_ROOT / "scripts" / "inference" / "text_generation.py"


@pytest.fixture
def entrypoint() -> types.ModuleType:
    """Load scripts/inference/text_generation.py as a standalone module."""
    spec = importlib.util.spec_from_file_location("text_generation_entrypoint_results", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)


def _dynamic_args() -> types.SimpleNamespace:
    return types.SimpleNamespace(
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


def _llm_returning(records: list[types.SimpleNamespace]) -> type:
    class _StubLLM:
        is_primary_rank = True

        def __init__(self, **_kwargs: object) -> None:
            pass

        def __enter__(self) -> "_StubLLM":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def generate(self, _prompts: list[str], _sampling_params: object) -> list[types.SimpleNamespace]:
            return records

    return _StubLLM


def _run(entrypoint: types.ModuleType, prompts: list[str]) -> None:
    entrypoint._generate_with_dynamic_engine(
        _dynamic_args(),
        model=object(),
        tokenizer=types.SimpleNamespace(tokenize=lambda _prompt: [1, 2]),
        prompts=prompts,
        sampling_params=object(),
    )


@pytest.mark.unit
def test_dynamic_generation_rejects_missing_inference_results(
    entrypoint: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A prompt the engine returns no record for fails the run instead of printing an empty banner."""
    printed: list[str] = []
    monkeypatch.setattr(entrypoint, "print_rank_0", printed.append)
    monkeypatch.setattr(entrypoint, "MegatronLLM", _llm_returning([]))
    monkeypatch.setattr(entrypoint, "build_inference_config", lambda **kwargs: kwargs)
    monkeypatch.setattr(entrypoint, "validate_sequence_length", lambda **kwargs: None)

    with pytest.raises(RuntimeError, match=r"Inference failed: engine returned 0 result\(s\) for 2 prompt\(s\)"):
        _run(entrypoint, ["Hello world", "Second prompt"])

    assert not any("GENERATED TEXT OUTPUT" in message for message in printed)


@pytest.mark.unit
def test_dynamic_generation_prints_results_when_every_prompt_returns(
    entrypoint: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: a complete result set is still printed and does not raise."""
    printed: list[str] = []
    records = [
        types.SimpleNamespace(
            request_id=idx,
            status=types.SimpleNamespace(name="COMPLETED"),
            generated_text=f"generated-{idx}",
            failed=lambda: False,
        )
        for idx in range(2)
    ]
    monkeypatch.setattr(entrypoint, "print_rank_0", printed.append)
    monkeypatch.setattr(entrypoint, "MegatronLLM", _llm_returning(records))
    monkeypatch.setattr(entrypoint, "build_inference_config", lambda **kwargs: kwargs)
    monkeypatch.setattr(entrypoint, "validate_sequence_length", lambda **kwargs: None)

    _run(entrypoint, ["Hello world", "Second prompt"])

    rendered = "\n".join(printed)
    assert "GENERATED TEXT OUTPUT" in rendered
    assert "Generated: generated-0" in rendered
    assert "Generated: generated-1" in rendered
