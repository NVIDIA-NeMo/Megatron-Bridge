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

"""Focused tests for the omni-language generation CLI input processing."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


_SCRIPT = Path(__file__).resolve().parents[4] / "examples" / "conversion" / "hf_to_megatron_generate_omni_lm.py"


def _module(name: str, **attrs: object) -> types.ModuleType:
    module = types.ModuleType(name)
    for attr_name, value in attrs.items():
        setattr(module, attr_name, value)
    return module


class _RecordingProcessor:
    def __init__(self) -> None:
        self.call_kwargs = None

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return "prompt"

    def __call__(self, **kwargs):
        self.call_kwargs = kwargs
        return types.SimpleNamespace(input_ids="ids", attention_mask="mask")


def _load_script(monkeypatch: pytest.MonkeyPatch, audios: list) -> types.ModuleType:
    stubs = {
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
            "megatron.bridge.utils.common_utils", get_last_rank=lambda: 0, print_rank_0=lambda message: None
        ),
        "qwen_omni_utils": _module(
            "qwen_omni_utils", process_mm_info=lambda messages, use_audio_in_video: (audios, None, ["video"])
        ),
    }
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location("hf_to_megatron_generate_omni_lm_under_test", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_audio_from_video_is_marked_as_video_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_script(monkeypatch, audios=["video_audio"])
    processor = _RecordingProcessor()

    module.process_omni_inputs(processor, "clip.mp4", "question", use_audio_in_video=True)

    assert processor.call_kwargs["audio"] == ["video_audio"]
    assert processor.call_kwargs["audio_from_video_indices"] == [0]


@pytest.mark.unit
@pytest.mark.parametrize(("use_audio_in_video", "audios"), [(False, None), (True, None), (True, [])])
def test_no_video_audio_passes_no_audio_indices(
    monkeypatch: pytest.MonkeyPatch, use_audio_in_video: bool, audios: list | None
) -> None:
    module = _load_script(monkeypatch, audios=audios)
    processor = _RecordingProcessor()

    module.process_omni_inputs(processor, "clip.mp4", "question", use_audio_in_video=use_audio_in_video)

    assert "audio_from_video_indices" not in processor.call_kwargs
    assert processor.call_kwargs["use_audio_in_video"] is use_audio_in_video
