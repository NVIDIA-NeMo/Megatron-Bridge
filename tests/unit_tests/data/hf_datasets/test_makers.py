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
import sys
from types import SimpleNamespace

import pytest

import megatron.bridge.data.hf_datasets.makers as makers


pytestmark = pytest.mark.unit


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


def test_make_text_chat_dataset_accepts_messages_conversation_and_conversations(monkeypatch):
    rows = [
        {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
            "extra": "kept",
        },
        {
            "conversation": [
                {"role": "user", "content": "bye"},
                {"role": "assistant", "content": "later"},
            ]
        },
        {
            "conversations": [
                {"from": "human", "value": "question"},
                {"from": "gpt", "value": "answer"},
            ]
        },
    ]
    _monkeypatch_load_dataset(monkeypatch, rows)

    out = makers.make_text_chat_dataset(path_or_dataset="dummy/text", split="train")

    assert out[0]["messages"][1]["content"] == "hello"
    assert out[0]["extra"] == "kept"
    assert out[1]["conversation"][1]["content"] == "later"
    assert out[2]["conversations"][1]["value"] == "answer"


def test_make_text_chat_dataset_requires_chat_columns(monkeypatch):
    _monkeypatch_load_dataset(monkeypatch, [{"prompt": "hi", "response": "hello"}])

    with pytest.raises(ValueError, match="messages.*conversation.*conversations"):
        makers.make_text_chat_dataset(path_or_dataset="dummy/text", split="train")


def test_make_squad_dataset_formats_messages(monkeypatch):
    rows = [
        {
            "context": "The Amazon rainforest is a moist broadleaf forest.",
            "question": "What type of forest is the Amazon rainforest?",
            "answers": {"text": ["moist broadleaf forest", "broadleaf forest"]},
        }
    ]
    _monkeypatch_load_dataset(monkeypatch, rows)

    out = makers.make_squad_dataset(split="train")

    assert out == [
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Context: The Amazon rainforest is a moist broadleaf forest. "
                        "Question: What type of forest is the Amazon rainforest? Answer:"
                    ),
                },
                {"role": "assistant", "content": "moist broadleaf forest"},
            ],
            "original_answers": ["moist broadleaf forest", "broadleaf forest"],
        }
    ]


def test_get_hf_dataset_maker_accepts_aliases_and_function_names():
    assert makers.get_hf_dataset_maker("squad") is makers.make_squad_dataset
    assert makers.get_hf_dataset_maker("make_squad_dataset") is makers.make_squad_dataset

    with pytest.raises(ValueError, match="Unknown maker_name"):
        makers.get_hf_dataset_maker("missing")


def test_make_gsm8k_dataset_formats_messages_and_final_answer(monkeypatch):
    rows = [
        {
            "question": "Janet has 3 apples. She buys 2 more. How many does she have?",
            "answer": "Janet starts with 3 apples. 3 + 2 = <<3+2=5>>5.\n#### 5",
        }
    ]
    _monkeypatch_load_dataset(monkeypatch, rows)

    out = makers.make_gsm8k_dataset(split="train")

    assert out[0]["messages"] == [
        {
            "role": "user",
            "content": "Question: Janet has 3 apples. She buys 2 more. How many does she have? Answer:",
        },
        {"role": "assistant", "content": "Janet starts with 3 apples. 3 + 2 = <<3+2=5>>5.\n#### 5"},
    ]
    assert out[0]["original_answers"] == ["5"]


def test_make_openmathinstruct2_thinking_dataset_formats_messages(monkeypatch):
    rows = [
        {
            "problem": "What is 2 + 3?",
            "generated_solution": r"We add 2 and 3 to get \boxed{5}.",
            "expected_answer": "5",
        }
    ]
    _monkeypatch_load_dataset(monkeypatch, rows)

    out = makers.make_openmathinstruct2_thinking_dataset(split="train_1M")

    assert out[0]["messages"] == [
        {"role": "user", "content": "What is 2 + 3?"},
        {"role": "assistant", "thinking": "We add 2 and 3 to get", "content": "#### 5"},
    ]
    assert out[0]["original_answers"] == ["5"]


def test_make_cv17_dataset(monkeypatch):
    rows = [{"audio": {"array": [0.1, 0.2], "sampling_rate": 16000}, "transcription": "hello"}]
    _monkeypatch_load_dataset(monkeypatch, rows)
    monkeypatch.setitem(sys.modules, "soundfile", SimpleNamespace(read=lambda *_args, **_kwargs: None))
    out = makers.make_cv17_dataset()
    assert out and isinstance(out[0]["audio"], tuple)
    assert out[0]["conversation"][0]["content"][0]["type"] == "audio"
    assert out[0]["conversation"][1]["content"][0]["text"] == "hello"


def test_make_default_audio_dataset_custom_text_column_keeps_spaces(monkeypatch):
    rows = [{"audio": {"array": [0.1, 0.2], "sampling_rate": 16000}, "transcription": "hello world"}]
    _monkeypatch_load_dataset(monkeypatch, rows)

    out = makers.make_default_audio_dataset(
        path_or_dataset="ysdede/commonvoice_17_tr_fixed",
        split="train",
        text_column="transcription",
        remove_text_spaces=False,
    )

    assert out[0]["conversation"][1]["content"][0]["text"] == "hello world"
    assert out[0]["audio"] == ([0.1, 0.2], 16000)


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


def test_make_valor32k_avqa_dataset_formats_modalities(tmp_path):
    root = tmp_path
    (root / "videos").mkdir()
    (root / "audio").mkdir()
    (root / "videos" / "av.mp4").touch()
    (root / "audio" / "av.wav").touch()
    (root / "audio" / "audio_only.wav").touch()
    (root / "videos" / "visual_only.mp4").touch()

    qa_rows = [
        {
            "video_id": "av",
            "modality": "audio-visual",
            "question": "Which sound matches the clip?",
            "options": ["music", "speech", "rain"],
            "correct_answer_idx": 1,
        },
        {
            "video_id": "audio_only",
            "modality": "audio",
            "question": "What is heard?",
            "rephrased_answers": ["applause"],
        },
        {
            "video_id": "visual_only",
            "modality": "visual",
            "question": "What is visible?",
            "options": ["dog", "car"],
            "correct_answer_idx": 0,
        },
        {
            "video_id": "missing_visual",
            "modality": "visual",
            "question": "This row should be skipped.",
            "rephrased_answers": ["missing"],
        },
    ]
    (root / "combined_dataset_train_flattened.json").write_text(json.dumps(qa_rows))

    out = makers.make_valor32k_avqa_dataset(str(root), max_audio_duration=7.5)

    assert len(out) == 3
    av = out[0]
    assert av["conversation"][0]["content"][0] == {"type": "video", "path": str(root / "videos" / "av.mp4")}
    assert "A. music\nB. speech\nC. rain" in av["conversation"][0]["content"][1]["text"]
    assert av["conversation"][1]["content"][0]["text"] == "speech"
    assert av["audio_path"] == str(root / "audio" / "av.wav")
    assert av["max_audio_duration"] == 7.5

    audio_only = out[1]
    assert [item["type"] for item in audio_only["conversation"][0]["content"]] == ["text"]
    assert audio_only["audio_path"] == str(root / "audio" / "audio_only.wav")
    assert audio_only["conversation"][1]["content"][0]["text"] == "applause"

    visual_only = out[2]
    assert visual_only["conversation"][0]["content"][0]["type"] == "video"
    assert "audio_path" not in visual_only


def test_make_valor32k_avqa_dataset_validation_split_and_empty_error(tmp_path):
    root = tmp_path
    (root / "videos").mkdir()
    (root / "audio").mkdir()
    (root / "videos" / "sample.mp4").touch()
    (root / "combined_dataset_val_flattened.json").write_text(
        json.dumps(
            [
                {
                    "video_id": "sample",
                    "modality": "visual",
                    "question": "What is shown?",
                    "rephrased_answers": ["a scene"],
                }
            ]
        )
    )

    out = makers.make_valor32k_avqa_dataset(str(root), split="validation", modality_filter="visual")
    assert len(out) == 1
    assert out[0]["conversation"][0]["content"][0]["path"] == str(root / "videos" / "sample.mp4")

    with pytest.raises(ValueError, match="No valid examples found"):
        makers.make_valor32k_avqa_dataset(str(root), split="validation", modality_filter="audio")
