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

"""Focused tests for the text generation CLI."""

from __future__ import annotations

import contextlib
import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest
import torch


_SCRIPT = Path(__file__).resolve().parents[4] / "examples" / "conversion" / "hf_to_megatron_generate_text.py"
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
        "megatron.core": _module("megatron.core", parallel_state=types.SimpleNamespace()),
        "megatron.core.inference.contexts": _module("megatron.core.inference.contexts", StaticInferenceContext=object),
        "megatron.core.inference.utils": _module("megatron.core.inference.utils", InferenceMode=object),
        "megatron.core.pipeline_parallel.schedules": _module(
            "megatron.core.pipeline_parallel.schedules", get_forward_backward_func=lambda: None
        ),
        "transformers": _module("transformers", AutoTokenizer=object),
        "megatron.bridge": _module("megatron.bridge", AutoBridge=object),
        "megatron.bridge.models.hf_pretrained.utils": _module(
            "megatron.bridge.models.hf_pretrained.utils", is_safe_repo=lambda **kwargs: True
        ),
        "megatron.bridge.utils.common_utils": _module(
            "megatron.bridge.utils.common_utils",
            disable_mtp_for_inference=lambda model: None,
            get_last_rank=lambda: 0,
            maybe_initialize_distributed=initialize_distributed,
            print_rank_0=lambda message: None,
        ),
    }
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)
    for name in _DISTRIBUTED_ENV:
        monkeypatch.delenv(name, raising=False)

    spec = importlib.util.spec_from_file_location("hf_to_megatron_generate_text_cli_under_test", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def initialize_model_parallel(*, seed: int) -> None:
        assert seed == 0
        assert events == ["distributed"]
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
        hf_revision=None,
        pipeline_model_parallel_layout=None,
    )

    with pytest.raises(RuntimeError, match="stop after model-parallel initialization"):
        module.main(args)


@pytest.mark.unit
def test_main_provides_an_iterator_for_each_virtual_pipeline_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    class InferenceContext:
        def __init__(self, **kwargs) -> None:
            pass

    class ModelChunk:
        def cuda(self):
            return self

        def eval(self) -> None:
            pass

    def interleaved_forward(**kwargs):
        assert isinstance(kwargs["data_iterator"], list), (
            "interleaved pipeline parallelism expected each model chunk to have a data iterator"
        )
        assert len(kwargs["data_iterator"]) == len(kwargs["model"]) == 2
        assert len({id(iterator) for iterator in kwargs["data_iterator"]}) == 2
        raise RuntimeError("stop after interleaved schedule accepts iterators")

    tokenizer = types.SimpleNamespace(
        pad_token="<pad>",
        eos_token_id=2,
        encode=lambda prompt, return_tensors: torch.tensor([[1, 2]]),
    )
    parallel_state = types.SimpleNamespace()
    stubs = {
        "megatron.core": _module("megatron.core", parallel_state=parallel_state),
        "megatron.core.inference.contexts": _module(
            "megatron.core.inference.contexts", StaticInferenceContext=InferenceContext
        ),
        "megatron.core.inference.utils": _module(
            "megatron.core.inference.utils",
            InferenceMode=types.SimpleNamespace(active=contextlib.nullcontext),
        ),
        "megatron.core.pipeline_parallel.schedules": _module(
            "megatron.core.pipeline_parallel.schedules", get_forward_backward_func=lambda: interleaved_forward
        ),
        "transformers": _module(
            "transformers", AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: tokenizer)
        ),
        "megatron.bridge": _module("megatron.bridge", AutoBridge=object),
        "megatron.bridge.models.hf_pretrained.utils": _module(
            "megatron.bridge.models.hf_pretrained.utils", is_safe_repo=lambda **kwargs: True
        ),
        "megatron.bridge.utils.common_utils": _module(
            "megatron.bridge.utils.common_utils",
            disable_mtp_for_inference=lambda model: None,
            get_last_rank=lambda: 1,
            maybe_initialize_distributed=lambda: None,
            print_rank_0=lambda message: None,
        ),
    }
    for name, stub in stubs.items():
        monkeypatch.setitem(sys.modules, name, stub)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)

    spec = importlib.util.spec_from_file_location("hf_to_megatron_generate_text_vpp_under_test", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    provider = types.SimpleNamespace(
        finalize=lambda: None,
        initialize_model_parallel=lambda *, seed: None,
        provide_distributed_model=lambda *, wrap_with_ddp: [ModelChunk(), ModelChunk()],
        vocab_size=32,
    )
    bridge = types.SimpleNamespace(to_megatron_provider=lambda load_weights: provider)
    module.AutoBridge = types.SimpleNamespace(from_hf_pretrained=lambda *args, **kwargs: bridge)
    args = types.SimpleNamespace(
        tp=1,
        pp=2,
        ep=1,
        etp=1,
        megatron_model_path=None,
        hf_model_path="org/four-layer-model",
        trust_remote_code=False,
        hf_revision=None,
        pipeline_model_parallel_layout="Et|t|t|tL",
        prompt="hello",
        apply_chat_template=False,
        thinking_mode="adaptive",
        max_new_tokens=1,
        legacy_full_prefix=False,
    )

    with pytest.raises(RuntimeError, match="stop after interleaved schedule accepts iterators"):
        module.main(args)
