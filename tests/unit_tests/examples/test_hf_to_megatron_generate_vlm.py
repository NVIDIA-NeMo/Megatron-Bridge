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

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


_SCRIPT = Path(__file__).parents[3] / "examples" / "conversion" / "hf_to_megatron_generate_vlm.py"


def _module(name: str, **attrs: object) -> types.ModuleType:
    module = types.ModuleType(name)
    for attr_name, value in attrs.items():
        setattr(module, attr_name, value)
    return module


@pytest.fixture
def vlm_generation(monkeypatch: pytest.MonkeyPatch):
    stubs = {
        "megatron": _module("megatron"),
        "megatron.core": _module("megatron.core", parallel_state=SimpleNamespace()),
        "megatron.core.pipeline_parallel": _module("megatron.core.pipeline_parallel"),
        "megatron.core.pipeline_parallel.schedules": _module(
            "megatron.core.pipeline_parallel.schedules", get_forward_backward_func=lambda: None
        ),
        "transformers": _module(
            "transformers",
            AutoConfig=object,
            AutoProcessor=object,
            AutoTokenizer=object,
        ),
        "vlm_generate_utils": _module(
            "vlm_generate_utils",
            pad_input_ids_to_tp_multiple=lambda *args: None,
            patch_kimi_vision_processor=lambda *args: None,
            process_image_inputs=lambda *args: None,
            process_multi_image_inputs=lambda *args: None,
            process_video_inputs=lambda *args: None,
            to_cuda=lambda value: value,
        ),
        "megatron.bridge": _module("megatron.bridge", AutoBridge=object),
        "megatron.bridge.models": _module("megatron.bridge.models"),
        "megatron.bridge.models.hf_pretrained": _module("megatron.bridge.models.hf_pretrained"),
        "megatron.bridge.models.hf_pretrained.utils": _module(
            "megatron.bridge.models.hf_pretrained.utils", is_safe_repo=lambda **kwargs: True
        ),
        "megatron.bridge.utils": _module("megatron.bridge.utils"),
        "megatron.bridge.utils.common_utils": _module(
            "megatron.bridge.utils.common_utils",
            get_last_rank=lambda: 0,
            print_rank_0=lambda *args: None,
            print_rank_last=lambda *args: None,
        ),
    }
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location("vlm_generation_under_test", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)


@pytest.mark.unit
@pytest.mark.parametrize("params_dtype", [torch.float32, torch.bfloat16])
def test_pipeline_dtype_matches_provider_parameters(vlm_generation, params_dtype: torch.dtype) -> None:
    provider = SimpleNamespace(params_dtype=params_dtype)

    assert vlm_generation._resolve_pipeline_dtype(provider) is params_dtype


@pytest.mark.unit
def test_pipeline_dtype_defaults_to_bfloat16(vlm_generation) -> None:
    assert vlm_generation._resolve_pipeline_dtype(SimpleNamespace()) is torch.bfloat16


@pytest.mark.unit
def test_hf_revision_kwargs(vlm_generation) -> None:
    revision = "4d7ae4984b7db7de8f8457170b3f1a419ee76d52"  # pragma: allowlist secret

    assert vlm_generation._hf_revision_kwargs(revision) == {"revision": revision}
    assert vlm_generation._hf_revision_kwargs(None) == {}
