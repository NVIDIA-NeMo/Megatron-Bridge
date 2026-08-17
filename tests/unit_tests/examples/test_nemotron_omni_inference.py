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

import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


_EXAMPLE_ROOT = Path(__file__).parents[3] / "examples" / "models" / "nemotron" / "nemotron_3_omni"


@pytest.mark.unit
def test_hf_revision_kwargs():
    script_globals = runpy.run_path(_EXAMPLE_ROOT / "hf_to_megatron_generate_nemotron_omni.py")
    revision_kwargs = script_globals["_hf_revision_kwargs"]

    assert revision_kwargs(None) == {}
    assert revision_kwargs("immutable-revision") == {"revision": "immutable-revision"}


@pytest.mark.unit
@pytest.mark.parametrize(
    "script_name",
    [
        "cord_v2_inference.py",
        "hf_to_megatron_generate_nemotron_omni.py",
        "valor32k_avqa_inference.py",
    ],
)
def test_inference_forward_step_uses_canonical_expanded_sequence_contract(script_name):
    script_globals = runpy.run_path(_EXAMPLE_ROOT / script_name)
    iterator_cls = script_globals["SingleBatchIterator"]
    forward_step = script_globals["vlm_forward_step"]
    input_ids = torch.tensor([[10, 11, 12]])
    num_image_tiles = torch.tensor([256, 128], dtype=torch.int)
    seen = {}

    class _Model:
        def __call__(self, **kwargs):
            seen.update(kwargs)
            return torch.zeros(1, 3, 8)

    iterator = iterator_cls(
        input_ids,
        torch.arange(3).unsqueeze(0),
        torch.ones_like(input_ids, dtype=torch.bool),
        images=torch.zeros(1, 2, 3),
        num_image_tiles=num_image_tiles,
    )

    output, _ = forward_step(iterator, _Model())

    assert output.shape == (1, 3, 8)
    assert "num_image_tiles" not in seen


@pytest.mark.unit
def test_generic_inference_processes_heterogeneous_source_images(monkeypatch):
    script_globals = runpy.run_path(_EXAMPLE_ROOT / "hf_to_megatron_generate_nemotron_omni.py")
    process_inputs = script_globals["process_image_inputs"]
    pixel_values = [
        torch.arange(3 * 32 * 16, dtype=torch.float32).reshape(3, 32, 16),
        torch.arange(3 * 16 * 32, dtype=torch.float32).reshape(3, 16, 32),
    ]

    class _Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages[-1]["content"].count("<image>") == 2
            return "rendered prompt"

    class _Inputs:
        input_ids = torch.tensor([[1, 2, 3]])
        num_patches = torch.tensor([1, 1])

        def __init__(self):
            self.pixel_values = pixel_values

    class _Processor:
        def __call__(self, *, text, images, return_tensors):
            assert text == ["rendered prompt"]
            assert images == ["first.png", "second.png"]
            assert return_tensors == "pt"
            return _Inputs()

    monkeypatch.setitem(process_inputs.__globals__, "load_image", lambda path: path)
    input_ids, packed, num_patches, imgs_sizes = process_inputs(
        _Tokenizer(), _Processor(), "first.png,second.png", "describe"
    )

    assert torch.equal(input_ids, torch.tensor([[1, 2, 3]]))
    assert packed.shape == (1, 4, 3 * 16 * 16)
    assert torch.equal(num_patches, torch.tensor([1, 1]))
    assert torch.equal(imgs_sizes, torch.tensor([[32, 16], [16, 32]]))


@pytest.mark.unit
def test_generic_audio_inference_uses_parakeet_feature_mask_length(monkeypatch):
    script_globals = runpy.run_path(_EXAMPLE_ROOT / "hf_to_megatron_generate_nemotron_omni.py")
    process_inputs = script_globals["process_audio_inputs"]

    class _Tokenizer:
        audio_token = "<so_embedding>"

        def apply_chat_template(self, messages, **kwargs):
            del kwargs
            assert messages[-1]["content"].startswith("<so_embedding>")
            return "rendered prompt"

        def convert_tokens_to_ids(self, token):
            assert token == "<so_embedding>"
            return 90

    class _Inputs(dict):
        @property
        def input_ids(self):
            return self["input_ids"]

    class _Processor:
        def __call__(self, *, text, audio, return_tensors):
            assert text == ["rendered prompt"]
            assert audio == ["clip.wav"]
            assert return_tensors == "pt"
            return _Inputs(
                input_ids=torch.tensor([[5, 90, 90, 6]]),
                sound_clips=torch.zeros(1, 1280),
            )

    class _FeatureExtractor:
        def __init__(self, *, sampling_rate, feature_size):
            assert sampling_rate == 16000
            assert feature_size == 128

        def __call__(self, raw_sound_clips, **kwargs):
            assert raw_sound_clips.shape == (1, 1280)
            assert kwargs["return_attention_mask"] is True
            return SimpleNamespace(
                input_features=torch.ones(1, 9, 128),
                attention_mask=torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0]]),
            )

    monkeypatch.setattr("transformers.ParakeetFeatureExtractor", _FeatureExtractor)

    input_ids, sound_clips, sound_length = process_inputs(
        _Tokenizer(),
        _Processor(),
        "clip.wav",
        "transcribe",
    )

    assert input_ids.tolist() == [[5, 90, 6]]
    assert sound_clips.shape == (1, 9, 128)
    assert sound_length.tolist() == [8]
