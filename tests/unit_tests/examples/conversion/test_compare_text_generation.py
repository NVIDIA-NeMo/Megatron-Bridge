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

"""Focused tests for loading a Megatron checkpoint in the text generation comparison example."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


_SCRIPT = Path(__file__).resolve().parents[4] / "examples" / "conversion" / "compare_text_generation.py"


class _FakeModelConfig:
    """Builder-backed model config without a provider-style initialize_model_parallel."""

    def __init__(self) -> None:
        self.transformer = types.SimpleNamespace()
        self.finalized = False

    def __setattr__(self, name: str, value: object) -> None:
        if name.endswith("_parallel_size"):
            setattr(self.transformer, name, value)
        else:
            object.__setattr__(self, name, value)

    def finalize(self) -> None:
        self.finalized = True


class _FakeProvider:
    """Model provider that records model-parallel initialization."""

    def __init__(self) -> None:
        self.finalized = False
        self.init_seeds = []

    def finalize(self) -> None:
        self.finalized = True

    def initialize_model_parallel(self, *, seed: int) -> None:
        self.init_seeds.append(seed)


def _module(name: str, **attrs: object) -> types.ModuleType:
    module = types.ModuleType(name)
    for attr_name, value in attrs.items():
        setattr(module, attr_name, value)
    return module


def _run_from_checkpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, model_cfg: object) -> dict:
    calls = {"pg_collection": [], "build": [], "destroyed": 0}

    def destroy_model_parallel() -> None:
        calls["destroyed"] += 1

    def build_and_load_model(checkpoint_path: str, cfg: object) -> str:
        calls["build"].append((checkpoint_path, cfg))
        return "model"

    auto_bridge = types.SimpleNamespace(_get_or_initialize_pg_collection=calls["pg_collection"].append)
    stubs = {
        "megatron.core": _module(
            "megatron.core", parallel_state=types.SimpleNamespace(destroy_model_parallel=destroy_model_parallel)
        ),
        "megatron.core.pipeline_parallel.schedules": _module(
            "megatron.core.pipeline_parallel.schedules", get_forward_backward_func=lambda: None
        ),
        "megatron.core.transformer": _module("megatron.core.transformer", MegatronModule=object),
        "transformers": _module("transformers", AutoModelForCausalLM=object, AutoTokenizer=object),
        "megatron.bridge.models.conversion.auto_bridge": _module(
            "megatron.bridge.models.conversion.auto_bridge", AutoBridge=auto_bridge
        ),
        "megatron.bridge.training.config": _module("megatron.bridge.training.config", TokenizerConfig=object),
        "megatron.bridge.training.tokenizers.tokenizer": _module(
            "megatron.bridge.training.tokenizers.tokenizer", MegatronTokenizer=object, build_tokenizer=lambda *a: None
        ),
        "megatron.bridge.utils.common_utils": _module(
            "megatron.bridge.utils.common_utils",
            get_last_rank=lambda: 0,
            get_local_rank_preinit=lambda: 0,
            print_rank_0=lambda message: None,
        ),
        "megatron.bridge.training.model_load_save": _module(
            "megatron.bridge.training.model_load_save",
            build_and_load_model=build_and_load_model,
            load_model_config=lambda checkpoint_path: (model_cfg, None),
            load_tokenizer=lambda checkpoint_path, trust_remote_code=False: "tokenizer",
        ),
        "megatron.training.models.base": _module("megatron.training.models.base", ModelConfig=_FakeModelConfig),
    }
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location("compare_text_generation_under_test", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "megatron_generate", lambda model, tokenizer, prompt, max_new_tokens: "generated")

    (tmp_path / "iter_0000000").mkdir()
    result = module.megatron_generate_from_checkpoint(str(tmp_path), "hello", tp=2, pp=1, ep=4, etp=1)

    assert result == "generated"
    assert calls["build"] == [(str(tmp_path / "iter_0000000"), model_cfg)]
    assert calls["destroyed"] == 1
    return calls


@pytest.mark.unit
def test_from_checkpoint_initializes_parallel_state_for_builder_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A builder-backed ModelConfig initializes model parallelism through its transformer config."""
    model_cfg = _FakeModelConfig()

    calls = _run_from_checkpoint(monkeypatch, tmp_path, model_cfg)

    assert model_cfg.finalized
    assert vars(model_cfg.transformer) == {
        "tensor_model_parallel_size": 2,
        "pipeline_model_parallel_size": 1,
        "expert_model_parallel_size": 4,
        "expert_tensor_parallel_size": 1,
    }
    assert calls["pg_collection"] == [model_cfg.transformer]


@pytest.mark.unit
def test_from_checkpoint_keeps_provider_model_parallel_initialization(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A model provider still initializes model parallelism itself."""
    provider = _FakeProvider()

    calls = _run_from_checkpoint(monkeypatch, tmp_path, provider)

    assert provider.finalized
    assert provider.tensor_model_parallel_size == 2
    assert provider.expert_model_parallel_size == 4
    assert provider.init_seeds == [0]
    assert calls["pg_collection"] == []
