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

"""Focused tests for the audio-language generation CLI."""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest


_SCRIPT = Path(__file__).resolve().parents[4] / "examples" / "conversion" / "hf_to_megatron_generate_audio_lm.py"
_DISTRIBUTED_ENV = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT")


def _module(name: str, **attrs: object) -> types.ModuleType:
    module = types.ModuleType(name)
    for attr_name, value in attrs.items():
        setattr(module, attr_name, value)
    return module


@pytest.mark.unit
def test_main_initializes_single_process_distributed_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    events = []

    def initialize_distributed() -> None:
        events.append("distributed")
        defaults = {
            "RANK": "0",
            "WORLD_SIZE": "1",
            "LOCAL_RANK": "0",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "29500",
        }
        for name, value in defaults.items():
            os.environ.setdefault(name, value)

    stubs = {
        "librosa": _module("librosa"),
        "megatron.core": _module("megatron.core", parallel_state=types.SimpleNamespace()),
        "megatron.core.pipeline_parallel.schedules": _module(
            "megatron.core.pipeline_parallel.schedules", get_forward_backward_func=lambda: None
        ),
        "transformers": _module("transformers", AutoProcessor=object, AutoTokenizer=object),
        "megatron.bridge": _module("megatron.bridge", AutoBridge=object),
        "megatron.bridge.models.hf_pretrained.utils": _module(
            "megatron.bridge.models.hf_pretrained.utils", is_safe_repo=lambda **kwargs: True
        ),
        "megatron.bridge.utils.common_utils": _module(
            "megatron.bridge.utils.common_utils",
            get_last_rank=lambda: 0,
            maybe_initialize_distributed=initialize_distributed,
            print_rank_0=lambda message: None,
        ),
        "megatron.bridge.utils.safe_url": _module(
            "megatron.bridge.utils.safe_url",
            is_safe_public_http_url=lambda url: (True, ""),
            safe_url_open=lambda url: None,
        ),
    }
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)
    for name in _DISTRIBUTED_ENV:
        monkeypatch.delenv(name, raising=False)

    spec = importlib.util.spec_from_file_location("hf_to_megatron_generate_audio_lm_cli_under_test", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def initialize_model_parallel(*, seed: int) -> None:
        assert seed == 0
        assert events == ["distributed"], "model-parallel initialization ran before distributed bootstrap"
        assert {name: os.environ[name] for name in _DISTRIBUTED_ENV} == {
            "RANK": "0",
            "WORLD_SIZE": "1",
            "LOCAL_RANK": "0",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "29500",
        }
        raise RuntimeError("stop after model-parallel initialization")

    provider = types.SimpleNamespace(finalize=lambda: None, initialize_model_parallel=initialize_model_parallel)
    bridge = types.SimpleNamespace(to_megatron_provider=lambda load_weights: provider)
    module.AutoBridge = types.SimpleNamespace(from_hf_pretrained=lambda *args, **kwargs: bridge)
    args = types.SimpleNamespace(
        tp=1,
        pp=1,
        ep=1,
        etp=1,
        megatron_model_path=None,
        hf_model_path="org/model",
        trust_remote_code=False,
    )

    with pytest.raises(RuntimeError, match="stop after model-parallel initialization"):
        module.main(args)
