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

"""Unit tests for the OpenAI-compatible server entrypoint."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import sys
import types
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[3]
_MODULE_PATH = _REPO_ROOT / "scripts" / "inference" / "openai_server.py"


def _module(name: str, **attrs: object) -> types.ModuleType:
    module = types.ModuleType(name)
    for attr_name, value in attrs.items():
        setattr(module, attr_name, value)
    return module


def _add_no_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    return parser


class _AsyncLLM:
    instances: list[_AsyncLLM] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.instances.append(self)

    async def __aenter__(self) -> _AsyncLLM:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def serve(self, serve_config: object, *, blocking: bool) -> None:
        self.serve_config = serve_config
        self.blocking = blocking


@pytest.fixture
def openai_server_entrypoint(monkeypatch: pytest.MonkeyPatch):
    _AsyncLLM.instances.clear()
    shared_helpers = {
        "add_distributed_args": _add_no_args,
        "add_engine_args": _add_no_args,
        "add_model_loading_args": _add_no_args,
        "add_parallelism_args": _add_no_args,
        "build_inference_config": lambda **kwargs: kwargs,
        "build_tokenizer": lambda *args: object(),
        "load_bridge_model": lambda **kwargs: kwargs,
        "resolve_hf_model_path": lambda *args: args[0],
    }
    stubs = {
        "megatron.core.inference.apis": _module(
            "megatron.core.inference.apis",
            MegatronAsyncLLM=_AsyncLLM,
            ServeConfig=types.SimpleNamespace,
        ),
        "megatron.core.utils": _module(
            "megatron.core.utils",
            configure_nvtx_profiling=lambda enabled: enabled,
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
        ),
    }
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location("openai_server_entrypoint_under_test", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("cli_args", "expected_return_log_probs"),
    [
        pytest.param([], True, id="default"),
        pytest.param(["--return-log-probs"], True, id="explicit-enable"),
        pytest.param(["--no-return-log-probs"], False, id="explicit-disable"),
    ],
)
def test_server_configures_log_prob_materialization(
    openai_server_entrypoint: types.ModuleType,
    cli_args: list[str],
    expected_return_log_probs: bool,
) -> None:
    args = openai_server_entrypoint.add_server_args(argparse.ArgumentParser()).parse_args(cli_args)
    args.max_seq_length = 32
    args.max_batch_size = None
    args.tp = 1
    args.block_size_tokens = 8
    args.kv_cache_buffer_size_gb = 1.0
    args.max_tokens = None
    args.enable_chunked_prefill = False

    asyncio.run(openai_server_entrypoint._serve(args, model=object(), tokenizer=object()))

    inference_config = _AsyncLLM.instances[-1].kwargs["inference_config"]
    assert inference_config["return_log_probs"] is expected_return_log_probs
