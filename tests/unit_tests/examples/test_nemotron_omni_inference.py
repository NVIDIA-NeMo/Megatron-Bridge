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

import megatron.bridge.models.nemotron_vl.nemotron_vl_utils as vl_utils


_EXAMPLE_ROOT = Path(__file__).parents[3] / "examples" / "models" / "nemotron" / "nemotron_3_omni"


class _VideoImageProcessor:
    _is_video_mode = False

    def __call__(self, *, images, return_tensors):
        assert self._is_video_mode is True
        assert return_tensors is None
        frame_count = len(images)
        return {
            "pixel_values": torch.arange(frame_count * 3 * 32 * 64, dtype=torch.float32).reshape(
                frame_count, 3, 32, 64
            ),
            "imgs_sizes": [(32, 64)] * frame_count,
            "num_tokens": [2] * frame_count,
        }


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
def test_generic_video_inference_uses_processor_grid_and_source_timestamps(monkeypatch):
    script_globals = runpy.run_path(_EXAMPLE_ROOT / "hf_to_megatron_generate_nemotron_omni.py")
    process_inputs = script_globals["process_video_inputs"]
    rendered_messages = []

    class _Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            del kwargs
            rendered_messages.extend(messages)
            return "rendered prompt"

        def __call__(self, text, *, return_tensors):
            assert text == ["rendered prompt"]
            assert return_tensors == "pt"
            return SimpleNamespace(input_ids=torch.tensor([[1, 2, 3]]))

    processor = SimpleNamespace(image_processor=_VideoImageProcessor())
    metadata = SimpleNamespace(fps=30.0, frames_indices=[0, 16, 31])
    monkeypatch.setattr(
        vl_utils,
        "maybe_path_or_url_to_data_urls",
        lambda *args, **kwargs: (["frame-0", "frame-1", "frame-2"], metadata),
    )
    monkeypatch.setattr(vl_utils, "pil_image_from_base64", lambda frame: frame)

    input_ids, packed, num_patches, imgs_sizes, num_frames = process_inputs(
        _Tokenizer(),
        processor,
        "clip.mp4",
        "describe",
    )

    content = rendered_messages[-1]["content"]
    assert "This is a video:" not in content
    for frame_number, timestamp in enumerate(("0.00", "0.53", "1.02"), start=1):
        assert f"{frame_number} sampled at {timestamp} seconds" in content
    assert content.count("<img><image></img>") == 2
    assert input_ids.tolist() == [[1, 2, 3]]
    assert packed.shape == (1, 24, 768)
    assert num_patches.tolist() == [1, 1]
    assert imgs_sizes.tolist() == [[32, 64], [32, 64], [32, 64]]
    assert num_frames.tolist() == [3]
    assert processor.image_processor._is_video_mode is False


@pytest.mark.unit
def test_generic_video_audio_inference_uses_processor_grid_and_source_timestamps(monkeypatch):
    script_globals = runpy.run_path(_EXAMPLE_ROOT / "hf_to_megatron_generate_nemotron_omni.py")
    process_inputs = script_globals["process_video_audio_inputs"]
    rendered_messages = []

    class _Tokenizer:
        audio_token = "<so_embedding>"

        def apply_chat_template(self, messages, **kwargs):
            del kwargs
            rendered_messages.extend(messages)
            return "rendered prompt"

        def convert_tokens_to_ids(self, token):
            assert token == "<so_embedding>"
            return 90

    class _Inputs(dict):
        @property
        def input_ids(self):
            return self["input_ids"]

    class _Processor:
        image_processor = _VideoImageProcessor()

        def __call__(self, *, text, audio, return_tensors):
            assert text == ["rendered prompt"]
            assert audio == ["clip.wav"]
            assert return_tensors == "pt"
            return _Inputs(input_ids=torch.tensor([[5, 90, 90, 6]]), sound_clips=torch.zeros(1, 1280))

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

    metadata = SimpleNamespace(fps=30.0, frames_indices=[0, 16, 31])
    monkeypatch.setattr(
        vl_utils,
        "maybe_path_or_url_to_data_urls",
        lambda *args, **kwargs: (["frame-0", "frame-1", "frame-2"], metadata),
    )
    monkeypatch.setattr(vl_utils, "pil_image_from_base64", lambda frame: frame)
    monkeypatch.setattr("transformers.ParakeetFeatureExtractor", _FeatureExtractor)

    output = process_inputs(_Tokenizer(), _Processor(), "clip.mp4", "clip.wav", "describe")
    input_ids, packed, num_patches, imgs_sizes, num_frames, sound_clips, sound_length = output

    content = rendered_messages[-1]["content"]
    assert "This is a video:" not in content
    assert "Frame 1 sampled at 0.00 seconds and frame 2 sampled at 0.53 seconds" in content
    assert "Frame 3 sampled at 1.02 seconds" in content
    assert input_ids.tolist() == [[5, 90, 6]]
    assert packed.shape == (1, 24, 768)
    assert num_patches.tolist() == [1, 1]
    assert imgs_sizes.tolist() == [[32, 64], [32, 64], [32, 64]]
    assert num_frames.tolist() == [3]
    assert sound_clips.shape == (1, 9, 128)
    assert sound_length.tolist() == [8]


@pytest.mark.unit
def test_valor_inference_uses_processor_grid_counts_and_source_timestamps(monkeypatch, tmp_path):
    script_globals = runpy.run_path(_EXAMPLE_ROOT / "valor32k_avqa_inference.py")
    process_sample = script_globals["process_sample"]
    rendered_messages = []

    class _Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            del kwargs
            rendered_messages.extend(messages)
            return "rendered prompt"

        def __call__(self, text, *, return_tensors):
            assert text == ["rendered prompt"]
            assert return_tensors == "pt"
            return SimpleNamespace(input_ids=torch.tensor([[1, 10, 11, 12, 10, 11, 12, 2]]))

        def convert_tokens_to_ids(self, token):
            return {"<img>": 10, "<image>": 11, "</img>": 12, "<so_embedding>": 90}[token]

    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    (videos_dir / "sample.mp4").touch()
    (tmp_path / "audio").mkdir()
    metadata = SimpleNamespace(fps=30.0, frames_indices=[0, 16, 31])
    monkeypatch.setitem(
        process_sample.__globals__,
        "maybe_path_or_url_to_data_urls",
        lambda *args, **kwargs: (["frame-0", "frame-1", "frame-2"], metadata),
    )
    monkeypatch.setitem(process_sample.__globals__, "pil_image_from_base64", lambda frame: frame)

    sample = process_sample(
        {"video_id": "video-1", "question": "What happens?", "options": ["A", "B"], "correct_answer_idx": 1},
        {"video-1": "sample"},
        tmp_path,
        _Tokenizer(),
        _VideoImageProcessor(),
        object(),
    )

    content = rendered_messages[-1]["content"]
    assert "This is a video:" not in content
    assert "Frame 1 sampled at 0.00 seconds and frame 2 sampled at 0.53 seconds" in content
    assert "Frame 3 sampled at 1.02 seconds" in content
    assert sample["images"].shape == (1, 24, 768)
    assert sample["images"].dtype == torch.bfloat16
    assert sample["imgs_sizes"].tolist() == [[32, 64], [32, 64], [32, 64]]
    assert sample["num_frames"].tolist() == [3]
    assert int((sample["input_ids"] == 11).sum().item()) == 4


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
