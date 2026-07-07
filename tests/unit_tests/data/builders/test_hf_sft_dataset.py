# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import pytest

from megatron.bridge.data.base import DatasetBuildContext
from megatron.bridge.data.builders import HFDatasetSourceConfig, HFSFTDatasetBuilder, HFSFTDatasetConfig
from megatron.bridge.data.builders import hf_sft_dataset as builder_module
from megatron.bridge.data.builders.hf_sft_dataset import select_hf_sft_collate


pytestmark = pytest.mark.unit


class _Tokenizer:
    added_tokens_decoder = {}
    pad_token_id = 0
    eos_token_id = 3
    chat_template = "{% generation %}{{ messages }}{% endgeneration %}"

    def apply_chat_template(self, conversation, tokenize=True, **kwargs):
        assert tokenize is True
        assert conversation[-1]["role"] == "assistant"
        return {"input_ids": [1, 2, 3], "assistant_masks": [0, 1, 1]}


@pytest.mark.parametrize("column", ["messages", "conversation", "conversations"])
def test_builder_auto_selects_shared_text_collate_for_all_chat_columns(monkeypatch, column):
    turns = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ]
    row = (
        {column: [{"from": turn["role"], "value": turn["content"]} for turn in turns]}
        if column == "conversations"
        else {column: turns}
    )
    monkeypatch.setattr(builder_module, "load_and_adapt_hf_dataset", lambda source: [row])
    config = HFSFTDatasetConfig(
        seq_length=16,
        source=HFDatasetSourceConfig(path_or_dataset="org/chat"),
        pad_to_multiple_of=1,
        do_validation=False,
        do_test=False,
    )

    train, validation, test = HFSFTDatasetBuilder(config).build(DatasetBuildContext(1, 0, 0, tokenizer=_Tokenizer()))

    assert train is not None
    assert train.collate_fn([train[0]])["tokens"].tolist() == [[1, 2, 3]]
    assert (validation, test) == (None, None)


def test_config_validates_source_and_padding():
    config = HFSFTDatasetConfig(
        seq_length=0,
        source=HFDatasetSourceConfig(path_or_dataset=""),
        pad_to_multiple_of=0,
    )

    with pytest.raises(ValueError, match="seq_length"):
        config.validate()


def test_config_rejects_disabled_explicit_split_sources():
    config = HFSFTDatasetConfig(
        seq_length=16,
        source=HFDatasetSourceConfig(path_or_dataset="org/chat"),
        validation_source=HFDatasetSourceConfig(path_or_dataset="org/validation"),
        do_validation=False,
    )

    with pytest.raises(ValueError, match="validation_source requires do_validation"):
        config.validate()


@pytest.mark.parametrize("media_type", ["image", "video", "audio"])
def test_builder_leaves_multimodal_conversations_to_processor_collators(media_type):
    row = {
        "conversation": [
            {
                "role": "user",
                "content": [{"type": media_type, media_type: object()}, {"type": "text", "text": "describe"}],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "answer"}]},
        ]
    }

    assert select_hf_sft_collate([row]) is None


@pytest.mark.parametrize(
    "row",
    [
        {"conversation": [{"role": "user", "content": "describe <image>"}], "image": object()},
        {"conversation": [{"role": "user", "content": "describe <video>"}], "videos": [object()]},
        {"conversation": [{"role": "user", "content": "transcribe"}], "audio": object()},
        {"conversation": [{"role": "user", "content": "transcribe"}], "audio_path": "audio.wav"},
        {"conversation": [{"role": "user", "content": "describe"}], "image_paths": ["image.png"]},
        {"conversation": [{"role": "user", "content": "describe"}], "video_path": "video.mp4"},
    ],
)
def test_builder_preserves_top_level_media_for_processor_collators(row):
    assert select_hf_sft_collate([row]) is None


def test_builder_does_not_classify_mixed_rows_from_first_text_example():
    text_row = {"messages": [{"role": "assistant", "content": "text"}]}
    image_row = {
        "conversation": [
            {"role": "user", "content": [{"type": "image", "image": object()}]},
            {"role": "assistant", "content": [{"type": "text", "text": "image"}]},
        ]
    }

    assert select_hf_sft_collate([text_row, image_row]) is None
