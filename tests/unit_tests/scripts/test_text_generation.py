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

"""Unit tests for ``scripts/inference/text_generation.py``."""

from __future__ import annotations

import importlib.util
import sys
import types
from datetime import timedelta
from enum import Enum
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "inference" / "text_generation.py"


class _AttnBackend(Enum):
    auto = "auto"
    flash = "flash"
    fused = "fused"
    unfused = "unfused"
    local = "local"


class _MambaInferenceStateConfig:
    @classmethod
    def from_model(cls, model):
        return cls()


class _SamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _PassthroughInit:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class _MegatronLLM(_PassthroughInit):
    is_primary_rank = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return None

    def generate(self, prompts, sampling_params):
        return []


def _module(name: str, **attrs: object) -> types.ModuleType:
    module = types.ModuleType(name)
    for attr_name, value in attrs.items():
        setattr(module, attr_name, value)
    return module


def _install_text_generation_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    stubs = {
        "megatron.core.inference.apis": _module(
            "megatron.core.inference.apis",
            MegatronLLM=_MegatronLLM,
            SamplingParams=_SamplingParams,
        ),
        "megatron.core.inference.config": _module(
            "megatron.core.inference.config",
            InferenceConfig=_PassthroughInit,
            MambaInferenceStateConfig=_MambaInferenceStateConfig,
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
        "megatron.core.transformer.enums": _module(
            "megatron.core.transformer.enums",
            AttnBackend=_AttnBackend,
        ),
        "transformers": _module(
            "transformers",
            AutoConfig=_PassthroughInit,
            AutoTokenizer=_PassthroughInit,
            PreTrainedTokenizerBase=object,
        ),
        "megatron.bridge": _module("megatron.bridge", AutoBridge=_PassthroughInit),
        "megatron.bridge.models.hf_pretrained.utils": _module(
            "megatron.bridge.models.hf_pretrained.utils",
            is_safe_repo=lambda *, hf_path, trust_remote_code: bool(trust_remote_code),
        ),
        "megatron.bridge.training.utils.checkpoint_utils": _module(
            "megatron.bridge.training.utils.checkpoint_utils",
            get_hf_model_id_from_checkpoint=lambda path: None,
        ),
        "megatron.bridge.utils.common_utils": _module(
            "megatron.bridge.utils.common_utils",
            disable_mtp_for_inference=lambda model: None,
            get_local_rank_preinit=lambda: 0,
            get_master_addr_safe=lambda: "localhost",
            get_master_port_safe=lambda: 29500,
            get_rank_safe=lambda: 0,
            get_world_size_safe=lambda: 1,
            print_rank_0=lambda message: None,
        ),
    }
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)


@pytest.fixture
def text_generation(monkeypatch):
    _install_text_generation_stubs(monkeypatch)
    spec = importlib.util.spec_from_file_location("text_generation_under_test", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)


def test_megatron_checkpoint_overrides_preserve_attention_backend(text_generation):
    provider = types.SimpleNamespace(cache_mla_latents=True)
    args = types.SimpleNamespace(
        tp=2,
        pp=2,
        ep=4,
        etp=1,
        sequence_parallel=True,
        attention_backend="local",
        inference_moe_token_dispatcher_type="nvls",
    )

    overrides = text_generation._build_megatron_checkpoint_overrides(
        provider,
        args,
        text_generation.torch.bfloat16,
    )

    assert overrides["attention_backend"] is text_generation.AttnBackend.local
    assert overrides["tensor_model_parallel_size"] == 2
    assert overrides["pipeline_model_parallel_size"] == 2
    assert overrides["expert_model_parallel_size"] == 4
    assert overrides["expert_tensor_parallel_size"] == 1
    assert overrides["sequence_parallel"] is True
    assert overrides["params_dtype"] is text_generation.torch.bfloat16
    assert overrides["pipeline_dtype"] is text_generation.torch.bfloat16
    assert overrides["bf16"] is True
    assert overrides["fp16"] is False
    assert overrides["cache_mla_latents"] is True
    assert overrides["inference_moe_token_dispatcher_type"] == "nvls"


def test_maybe_initialize_distributed_populates_env_from_safe_helpers(monkeypatch, text_generation):
    init_calls = []
    set_device_calls = []

    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        monkeypatch.delenv(key, raising=False)

    monkeypatch.setattr(text_generation.dist, "is_available", lambda: True)
    monkeypatch.setattr(text_generation.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(text_generation, "get_rank_safe", lambda: 7)
    monkeypatch.setattr(text_generation, "get_world_size_safe", lambda: 16)
    monkeypatch.setattr(text_generation, "get_local_rank_preinit", lambda: 3)
    monkeypatch.setattr(text_generation, "get_master_addr_safe", lambda: "node-0")
    monkeypatch.setattr(text_generation, "get_master_port_safe", lambda: 23456)
    monkeypatch.setattr(text_generation.torch.cuda, "set_device", lambda device: set_device_calls.append(device))
    monkeypatch.setattr(
        text_generation.dist,
        "init_process_group",
        lambda backend, timeout: init_calls.append({"backend": backend, "timeout": timeout}),
    )

    text_generation._maybe_initialize_distributed(timeout_minutes=11)

    assert (
        dict(
            RANK="7",
            WORLD_SIZE="16",
            LOCAL_RANK="3",
            MASTER_ADDR="node-0",
            MASTER_PORT="23456",
        ).items()
        <= dict(text_generation.os.environ).items()
    )
    assert set_device_calls == [3]
    assert init_calls == [{"backend": "nccl", "timeout": timedelta(minutes=11)}]


def test_maybe_initialize_distributed_is_noop_when_dist_unavailable(monkeypatch, text_generation):
    monkeypatch.setattr(text_generation.dist, "is_available", lambda: False)
    monkeypatch.setattr(text_generation.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(
        text_generation.dist,
        "init_process_group",
        lambda *args, **kwargs: pytest.fail("init_process_group should not be called"),
    )
    monkeypatch.setattr(
        text_generation.torch.cuda,
        "set_device",
        lambda device: pytest.fail("set_device should not be called"),
    )

    text_generation._maybe_initialize_distributed(timeout_minutes=1)
