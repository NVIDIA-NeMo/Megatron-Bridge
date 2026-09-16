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

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[3]
_MODULE_PATH = _REPO_ROOT / "examples" / "conversion" / "hf_to_megatron_generate_text.py"


class _PassthroughInit:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.args = args
        self.kwargs = kwargs


def _module(name: str, **attrs: object) -> types.ModuleType:
    module = types.ModuleType(name)
    for attr_name, value in attrs.items():
        setattr(module, attr_name, value)
    return module


@pytest.fixture
def generation_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    stubs = {
        "megatron.core": _module("megatron.core", parallel_state=_PassthroughInit()),
        "megatron.core.inference.contexts": _module(
            "megatron.core.inference.contexts", StaticInferenceContext=_PassthroughInit
        ),
        "megatron.core.inference.utils": _module("megatron.core.inference.utils", InferenceMode=_PassthroughInit),
        "megatron.core.pipeline_parallel.schedules": _module(
            "megatron.core.pipeline_parallel.schedules", get_forward_backward_func=lambda: None
        ),
        "transformers": _module("transformers", AutoTokenizer=_PassthroughInit),
        "megatron.bridge": _module("megatron.bridge", AutoBridge=_PassthroughInit),
        "megatron.bridge.models.hf_pretrained.utils": _module(
            "megatron.bridge.models.hf_pretrained.utils", is_safe_repo=lambda **kwargs: True
        ),
        "megatron.bridge.utils.common_utils": _module(
            "megatron.bridge.utils.common_utils",
            disable_mtp_for_inference=lambda model: None,
            get_last_rank=lambda: 0,
            print_rank_0=lambda message: None,
        ),
    }
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location("hf_to_megatron_generate_text_under_test", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)


@pytest.mark.unit
def test_checkpoint_load_forwards_provider_derived_pipeline_layout(
    generation_module: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    layout = [
        ["embedding", *(["decoder"] * 11)],
        ["decoder"] * 11,
        ["decoder"] * 11,
        [*(["decoder"] * 10), "mtp", "loss"],
    ]
    checkpoint = tmp_path / "checkpoint"
    iteration = checkpoint / "iter_0000000"
    iteration.mkdir(parents=True)
    (iteration / "run_config.yaml").write_text(
        "model:\n"
        "  pipeline_model_parallel_size: 1\n"
        "  pipeline_model_parallel_layout:\n"
        "    _target_: megatron.core.transformer.pipeline_parallel_layer_layout.PipelineParallelLayerLayout\n",
        encoding="utf-8",
    )

    provider = types.SimpleNamespace(pipeline_model_parallel_layout=None)

    def apply_overrides_and_finalize() -> None:
        provider.pipeline_model_parallel_layout = layout

    provider.apply_overrides_and_finalize = apply_overrides_and_finalize
    provider.finalize = lambda: None
    provider.initialize_model_parallel = lambda seed: None
    bridge = types.SimpleNamespace(
        to_megatron_provider=lambda load_weights: provider,
        load_megatron_model=lambda *args, **kwargs: [],
    )

    class _Bridge:
        @staticmethod
        def from_hf_pretrained(*args: object, **kwargs: object) -> object:
            return bridge

    monkeypatch.setattr(generation_module, "AutoBridge", _Bridge)
    monkeypatch.setattr(generation_module, "print_rank_0", lambda message: None)

    recorded_overrides: dict[str, object] = {}

    def load_model(*args: object, **kwargs: object) -> list[object]:
        recorded_overrides.update(kwargs["mp_overrides"])
        raise RuntimeError("stop after checkpoint loading contract")

    bridge.load_megatron_model = load_model
    args = types.SimpleNamespace(
        tp=1,
        pp=4,
        ep=1,
        etp=1,
        megatron_model_path=str(checkpoint),
        hf_model_path="deepseek-ai/DeepSeek-V4-Flash",
        trust_remote_code=True,
        hf_revision=None,
        pipeline_model_parallel_layout=None,
    )

    with pytest.raises(RuntimeError, match="stop after checkpoint loading contract"):
        generation_module.main(args)

    assert recorded_overrides["pipeline_model_parallel_layout"] == layout
