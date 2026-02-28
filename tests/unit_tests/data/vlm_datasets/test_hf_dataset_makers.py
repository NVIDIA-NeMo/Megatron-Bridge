# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import json
from types import SimpleNamespace

import megatron.bridge.data.vlm_datasets.hf_dataset_makers as makers


class _DummyDataset(list):
    def remove_columns(self, cols):  # match datasets API used
        return self


def _monkeypatch_load_dataset(monkeypatch, rows):
    def _fake_load_dataset(path_or_dataset, name=None, split="train", **kwargs):  # noqa: ARG001 - interface
        return _DummyDataset(rows)

    def _fake_concatenate_datasets(datasets):  # noqa: ARG001 - interface
        # Combine all datasets into one _DummyDataset
        combined = _DummyDataset()
        for ds in datasets:
            combined.extend(ds)
        return combined

    monkeypatch.setattr(makers, "load_dataset", _fake_load_dataset)
    monkeypatch.setattr(makers, "concatenate_datasets", _fake_concatenate_datasets)


def test_make_rdr_dataset(monkeypatch):
    rows = [
        {"image": SimpleNamespace(), "text": "a cat"},
        {"image": SimpleNamespace(), "text": "a dog"},
    ]
    _monkeypatch_load_dataset(monkeypatch, rows)
    out = makers.make_rdr_dataset()
    assert isinstance(out, list) and len(out) == 2
    assert out[0]["conversation"][0]["content"][0]["type"] == "image"


def test_make_cord_v2_dataset_variants(monkeypatch):
    gt = {"gt_parses": [{"x": 1}, {"y": 2}]}
    rows = [{"image": SimpleNamespace(), "ground_truth": json.dumps(gt)}]
    _monkeypatch_load_dataset(monkeypatch, rows)
    out = makers.make_cord_v2_dataset()
    assert out and out[0]["conversation"][1]["role"] == "assistant"

    # alt structure with single gt_parse
    gt2 = {"gt_parse": {"a": 1}}
    rows2 = [{"image": SimpleNamespace(), "ground_truth": json.dumps(gt2)}]
    _monkeypatch_load_dataset(monkeypatch, rows2)
    out2 = makers.make_cord_v2_dataset()
    assert out2 and "<s_a>" in makers.json2token({"a": 1}, sort_json_key=True)


def test_make_medpix_dataset(monkeypatch):
    rows = [{"image_id": SimpleNamespace(), "question": "q?", "answer": "a"}]
    _monkeypatch_load_dataset(monkeypatch, rows)
    out = makers.make_medpix_dataset()
    assert out and out[0]["conversation"][1]["content"][0]["type"] == "text"


def test_make_cv17_dataset(monkeypatch):
    rows = [{"audio": {"array": [0.1, 0.2], "sampling_rate": 16000}, "transcription": "hello"}]
    _monkeypatch_load_dataset(monkeypatch, rows)
    out = makers.make_cv17_dataset()
    assert out and isinstance(out[0]["audio"], tuple)


def test_make_raven_dataset(monkeypatch):
    # Simulate a row with images and the expected texts structure
    rows = [
        {"images": [SimpleNamespace(), SimpleNamespace()], "texts": [{"user": "What?", "assistant": "Answer."}]},
        # No images or malformed rows
        {"images": [], "texts": [{"user": "?", "assistant": "A"}]},
        {"images": [SimpleNamespace()], "texts": []},
        {"images": [SimpleNamespace()], "texts": [{}]},
        {"images": [SimpleNamespace()], "texts": [{"assistant": "A"}]},
    ]
    _monkeypatch_load_dataset(monkeypatch, rows)
    out = makers.make_raven_dataset()
    # Only the first example should produce a valid output
    assert isinstance(out, list)
    assert len(out) == 1
    assert out[0]["conversation"][0]["role"] == "user"
    assert out[0]["conversation"][1]["role"] == "assistant"
    assert out[0]["conversation"][0]["content"][0]["type"] == "image"


def test_make_llava_pretrain_dataset_hf(monkeypatch, tmp_path):
    """Fallback to load_dataset when local JSON is absent."""
    image_root = tmp_path
    convs = [{"from": "human", "value": "Describe this image.\n<image>"}, {"from": "gpt", "value": "A cat."}]
    valid = {"image": "00453/004539375.jpg", "conversations": convs}
    no_image = {"image": "", "conversations": convs}
    no_convs = {"image": "00453/004539375.jpg", "conversations": []}
    human_contentless = {
        "image": "00453/004539375.jpg",
        "conversations": [{"from": "human", "value": ""}, {"from": "gpt", "value": "A."}],
    }
    rows = [valid, no_image, no_convs, human_contentless]
    _monkeypatch_load_dataset(monkeypatch, rows)
    out = makers.make_llava_pretrain_dataset(str(image_root))
    assert isinstance(out, list)
    # valid and human_contentless both produce output
    assert len(out) == 2

    # Check the valid conversation
    valid_conv = out[0]["conversation"]
    assert valid_conv[0]["role"] == "user"
    assert valid_conv[0]["content"][0]["type"] == "image"
    assert "path" in valid_conv[0]["content"][0]
    # <image> placeholder should be stripped from the prompt
    assert "<image>" not in valid_conv[0]["content"][1]["text"]
    assert "Describe this image." in valid_conv[0]["content"][1]["text"]
    assert valid_conv[1]["role"] == "assistant"
    assert valid_conv[1]["content"][0]["text"] == "A cat."


def test_make_llava_pretrain_dataset_local_json(tmp_path):
    """Load from a local directory with blip_laion_cc_sbu_558k.json."""
    convs = [{"from": "human", "value": "Summarize.\n<image>"}, {"from": "gpt", "value": "A dog."}]
    entries = [
        {"id": "001", "image": "00001/001.jpg", "conversations": convs},
        {"id": "002", "image": "", "conversations": convs},  # filtered out
    ]
    json_path = tmp_path / "blip_laion_cc_sbu_558k.json"
    json_path.write_text(json.dumps(entries))

    out = makers.make_llava_pretrain_dataset(
        image_root_path=str(tmp_path), path_or_dataset=str(tmp_path)
    )
    assert isinstance(out, list)
    assert len(out) == 1
    valid_conv = out[0]["conversation"]
    assert valid_conv[0]["content"][0]["type"] == "image"
    assert "<image>" not in valid_conv[0]["content"][1]["text"]
    assert "Summarize." in valid_conv[0]["content"][1]["text"]
    assert valid_conv[1]["content"][0]["text"] == "A dog."


def test_make_llava_video_178k_dataset(monkeypatch, tmp_path):
    # Happy path: valid video and conversation
    video_file = "the_vid.mp4"
    video_root = tmp_path
    convs = [{"from": "human", "value": "<video>\nQ?"}, {"from": "gpt", "value": "A."}]
    valid = {"video": video_file, "conversations": convs}
    # Invalid variants
    no_video = {"video": "", "conversations": convs}
    no_convs = {"video": video_file, "conversations": []}
    # Note: empty human value gets skipped but gpt turn is kept (results in assistant-only conversation)
    human_contentless = {
        "video": video_file,
        "conversations": [{"from": "human", "value": ""}, {"from": "gpt", "value": "A."}],
    }
    rows = [valid, no_video, no_convs, human_contentless]
    _monkeypatch_load_dataset(monkeypatch, rows)
    out = makers.make_llava_video_178k_dataset(str(video_root), subsets="sub1")
    assert isinstance(out, list)
    # valid and human_contentless both produce output (though human_contentless is malformed)
    assert len(out) == 2

    # Check the valid conversation (first one)
    valid_conv = out[0]["conversation"]
    assert valid_conv[0]["role"] == "user" and any(d["type"] == "video" for d in valid_conv[0]["content"])
    # Clean prompt is stripped
    assert "Q?" in valid_conv[0]["content"][-1]["text"]
    assert valid_conv[1]["role"] == "assistant"

    # The human_contentless case produces an assistant-only conversation (edge case)
    contentless_conv = out[1]["conversation"]
    assert len(contentless_conv) == 1
    assert contentless_conv[0]["role"] == "assistant"
