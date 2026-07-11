# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import json
import random
from types import SimpleNamespace

import pytest

from megatron.bridge.data.sources import hf_adapters as adapter_module
from megatron.bridge.data.sources.hf_adapters import adapt_hf_dataset, prepare_hf_dataset_for_adapter


pytestmark = pytest.mark.unit


@pytest.mark.parametrize("column", ["messages", "conversation", "conversations"])
def test_native_adapter_preserves_supported_conversation_columns(column):
    row = {column: [{"role": "assistant", "content": "answer"}], "id": 7}

    assert adapt_hf_dataset([row], adapter_name=None) == [row]


def test_native_adapter_preserves_schema_for_preprocessing_validation():
    row = {"prompt": "question", "answer": "answer"}

    assert adapt_hf_dataset([row], adapter_name=None) == [row]


def test_text_adapters_normalize_squad_and_gsm8k():
    squad = adapt_hf_dataset(
        [{"context": "ctx", "question": "q", "answers": {"text": ["a", "also a"]}}],
        adapter_name="squad",
    )
    gsm8k = adapt_hf_dataset(
        [{"question": "1 + 1?", "answer": "work\n#### 2"}],
        adapter_name="gsm8k",
    )

    assert squad[0]["original_answers"] == ["a", "also a"]
    assert squad[0]["prompt"] == "Context: ctx Question: q Answer:"
    assert squad[0]["completion"] == "a"
    assert gsm8k[0]["completion"] == "work\n#### 2"
    assert gsm8k[0]["original_answers"] == ["2"]


def test_openmath_thinking_adapter_separates_reasoning_and_answer():
    adapted = adapt_hf_dataset(
        [
            {
                "problem": "What is 2 + 3?",
                "generated_solution": r"We add 2 and 3 to get \boxed{5}.",
                "expected_answer": "5",
            }
        ],
        adapter_name="openmathinstruct2_thinking",
    )

    assert adapted[0]["messages"] == [
        {"role": "user", "content": "What is 2 + 3?"},
        {"role": "assistant", "thinking": "We add 2 and 3 to get", "content": "#### 5"},
    ]


@pytest.mark.parametrize(
    "ground_truth",
    [
        {"gt_parses": [{"name": "receipt"}]},
        {"gt_parse": {"name": "receipt"}},
    ],
)
def test_cord_adapter_supports_plural_and_singular_ground_truth(ground_truth):
    adapted = adapt_hf_dataset(
        [{"image": SimpleNamespace(), "ground_truth": json.dumps(ground_truth)}],
        adapter_name="cord_v2",
    )

    assert adapted[0]["conversation"][0]["content"][0]["type"] == "image"
    assert adapted[0]["conversation"][1]["role"] == "assistant"


def test_cord_adapter_multiple_ground_truths_is_independent_of_global_rng():
    row = {
        "image": SimpleNamespace(),
        "ground_truth": json.dumps({"gt_parses": [{"name": "first"}, {"name": "second"}]}),
    }

    random.seed(42)
    first_adaptation = adapt_hf_dataset([row], adapter_name="cord_v2")
    random.seed(142)
    second_adaptation = adapt_hf_dataset([row], adapter_name="cord_v2")

    assert first_adaptation == second_adaptation


@pytest.mark.parametrize(
    ("adapter_name", "row"),
    [
        ("rdr", {"image": object(), "text": "A cat."}),
        ("medpix", {"image_id": object(), "question": "What?", "answer": "A cat."}),
    ],
)
def test_image_adapters_normalize_single_image_rows(adapter_name, row):
    adapted = adapt_hf_dataset([row], adapter_name=adapter_name)

    assert adapted[0]["conversation"][0]["content"][0]["type"] == "image"
    assert adapted[0]["conversation"][1]["content"][0]["text"] == "A cat."


def test_raven_adapter_filters_malformed_rows():
    rows = [
        {"images": [object()], "texts": [{"user": "What?", "assistant": "Answer."}]},
        {"images": [], "texts": [{"user": "?", "assistant": "A"}]},
        {"images": [object()], "texts": []},
    ]

    adapted = adapt_hf_dataset(rows, adapter_name="raven")

    assert len(adapted) == 1
    assert adapted[0]["conversation"][0]["content"][0]["type"] == "image"


def test_llava_video_adapter_normalizes_turns(tmp_path):
    rows = [
        {
            "video": "clip.mp4",
            "conversations": [
                {"from": "human", "value": "<video>\nWhat happens?"},
                {"from": "gpt", "value": "A person waves."},
            ],
        },
        {"video": "", "conversations": []},
    ]

    adapted = adapt_hf_dataset(
        rows,
        adapter_name="llava_video_178k",
        adapter_kwargs={"video_root_path": str(tmp_path)},
    )

    assert len(adapted) == 1
    assert adapted[0]["conversation"][0]["content"][0] == {
        "type": "video",
        "path": str(tmp_path / "clip.mp4"),
    }
    assert adapted[0]["conversation"][0]["content"][1]["text"] == "What happens?"


def test_audio_adapters_preserve_schema_defaults(monkeypatch):
    monkeypatch.setattr(adapter_module, "_decode_audio", lambda audio: ([0.0], 16_000))

    cv17 = adapt_hf_dataset([{"audio": {}, "transcription": "iki kelime"}], adapter_name="cv17")
    generic = adapt_hf_dataset([{"audio": {}, "text": "two words"}], adapter_name="default_audio")

    assert cv17[0]["conversation"][-1]["content"][0]["text"] == "iki kelime"
    assert generic[0]["conversation"][-1]["content"][0]["text"] == "twowords"


def test_audio_adapter_preparation_disables_automatic_decoding():
    class _Dataset:
        cast = None

        def cast_column(self, column, feature):
            self.cast = (column, feature.decode)
            return self

    dataset = _Dataset()

    assert prepare_hf_dataset_for_adapter(dataset, adapter_name="cv17") is dataset
    assert dataset.cast == ("audio", False)


def test_valor32k_adapter_formats_modalities_and_filters_missing_media(tmp_path):
    (tmp_path / "videos").mkdir()
    (tmp_path / "audio").mkdir()
    (tmp_path / "videos" / "av.mp4").touch()
    (tmp_path / "audio" / "av.wav").touch()
    (tmp_path / "audio" / "audio_only.wav").touch()
    rows = [
        {
            "video_id": "av",
            "modality": "audio-visual",
            "question": "Which sound?",
            "options": ["music", "speech"],
            "correct_answer_idx": 1,
        },
        {
            "video_id": "audio_only",
            "modality": "audio",
            "question": "What is heard?",
            "rephrased_answers": ["applause"],
        },
        {
            "video_id": "missing",
            "modality": "visual",
            "question": "Missing",
            "rephrased_answers": ["nothing"],
        },
    ]

    adapted = adapt_hf_dataset(
        rows,
        adapter_name="valor32k_avqa",
        adapter_kwargs={"data_root": str(tmp_path), "max_audio_duration": 7.5},
    )

    assert len(adapted) == 2
    assert adapted[0]["conversation"][1]["content"][0]["text"] == "speech"
    assert adapted[0]["audio_path"] == str(tmp_path / "audio" / "av.wav")
    assert adapted[0]["max_audio_duration"] == 7.5
    assert [item["type"] for item in adapted[1]["conversation"][0]["content"]] == ["text"]


def test_adapter_registry_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown Hugging Face schema adapter"):
        adapt_hf_dataset([{"messages": []}], adapter_name="missing")
