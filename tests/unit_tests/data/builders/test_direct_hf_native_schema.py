# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import json
from itertools import permutations

import pytest

from megatron.bridge.data.base import DatasetBuildContext
from megatron.bridge.data.builders import (
    ChatSFTPreprocessingConfig,
    DirectHFSFTDatasetBuilder,
    DirectHFSFTDatasetConfig,
    HFDatasetSourceConfig,
    PromptCompletionSFTPreprocessingConfig,
)


pytestmark = pytest.mark.unit

_CHAT_COLUMNS = ("messages", "conversation", "conversations")
_TURNS = [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]


class _Tokenizer:
    added_tokens_decoder = {}
    pad_token_id = 0
    eos_token_id = 3

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return [{"Q": 1, "A": 2}[text]]


def _build(tmp_path, row, preprocessing, adapter_kwargs=None):
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    config = DirectHFSFTDatasetConfig(
        seq_length=16,
        source=HFDatasetSourceConfig(
            path_or_dataset="json",
            split="train",
            load_kwargs={"data_files": str(path)},
            adapter_kwargs=adapter_kwargs,
        ),
        preprocessing=preprocessing,
        pad_to_multiple_of=1,
        do_validation=False,
        do_test=False,
    )
    train, validation, test = DirectHFSFTDatasetBuilder(config).build(
        DatasetBuildContext(1, 0, 0, tokenizer=_Tokenizer())
    )
    assert train is not None
    assert len(train) == 1
    assert validation is test is None
    return train


@pytest.mark.parametrize(("prompt_column", "completion_column"), tuple(permutations(_CHAT_COLUMNS, 2)))
def test_native_builder_preserves_both_selected_text_columns(tmp_path, prompt_column, completion_column):
    row = {prompt_column: "Q", completion_column: "A", "row_id": 7}
    preprocessing = PromptCompletionSFTPreprocessingConfig(
        prompt_column=prompt_column, completion_column=completion_column
    )

    dataset = _build(tmp_path, row, preprocessing)
    batch = dataset.collate_fn([dataset[0]])

    assert dataset[0] == row
    assert batch["tokens"].tolist() == [[1, 2, 3]]
    assert batch["labels"].tolist() == [[2, 3, -100]]
    assert batch["loss_mask"].tolist() == [[1, 1, 0]]
    assert batch["metadata"] == [{"row_id": 7}]


@pytest.mark.parametrize(("text_column", "chat_column"), tuple(permutations(_CHAT_COLUMNS, 2)))
@pytest.mark.parametrize("selected_role", ["prompt", "completion"])
def test_native_builder_rejects_other_chat_column(tmp_path, text_column, chat_column, selected_role):
    prompt_column, completion_column = (
        (text_column, "answer") if selected_role == "prompt" else ("question", text_column)
    )
    row = {prompt_column: "Q", completion_column: "A", chat_column: _TURNS}
    preprocessing = PromptCompletionSFTPreprocessingConfig(
        prompt_column=prompt_column, completion_column=completion_column
    )

    with pytest.raises(ValueError, match="structured conversations require ChatSFTPreprocessingConfig"):
        _build(tmp_path, row, preprocessing)


@pytest.mark.parametrize(("first", "second"), tuple(permutations(_CHAT_COLUMNS, 2)))
def test_native_builder_rejects_multiple_chat_columns(tmp_path, first, second):
    with pytest.raises(ValueError, match="multiple populated conversation columns"):
        _build(tmp_path, {first: _TURNS, second: _TURNS}, ChatSFTPreprocessingConfig())


def test_native_builder_rejects_multiple_custom_chat_columns(tmp_path):
    with pytest.raises(ValueError, match="multiple populated conversation columns"):
        _build(
            tmp_path,
            {"dialogue": _TURNS, "history": _TURNS},
            ChatSFTPreprocessingConfig(),
            {"messages_column": "dialogue", "conversation_column": "history"},
        )


@pytest.mark.parametrize("column", _CHAT_COLUMNS)
def test_native_builder_accepts_single_chat_with_null_alternatives(tmp_path, column):
    row = {key: _TURNS if key == column else None for key in _CHAT_COLUMNS}
    dataset = _build(tmp_path, row, ChatSFTPreprocessingConfig())

    assert dataset[0] == {"conversation": _TURNS}


@pytest.mark.parametrize(
    ("source_column", "adapter_kwargs"),
    [
        ("conversation", {"messages_column": "conversation"}),
        ("dialogue", {"messages_column": "dialogue", "conversation_column": "dialogue"}),
    ],
)
def test_native_builder_accepts_one_source_with_overlapping_aliases(tmp_path, source_column, adapter_kwargs):
    dataset = _build(tmp_path, {source_column: _TURNS}, ChatSFTPreprocessingConfig(), adapter_kwargs)

    assert dataset[0] == {"conversation": _TURNS}


def test_native_builder_preserves_selected_column_priority(tmp_path):
    preprocessing = PromptCompletionSFTPreprocessingConfig(prompt_column="messages", completion_column="answer")
    row = {"messages": "Q", "answer": "A", "prompt": "decoy", "completion": "decoy"}

    dataset = _build(tmp_path, row, preprocessing)
    assert dataset[0] == {"messages": "Q", "answer": "A"}
    assert dataset.collate_fn([dataset[0]])["tokens"].tolist() == [[1, 2, 3]]

    with pytest.raises(ValueError, match="exactly one schema"):
        _build(tmp_path, {**row, "messages": _TURNS}, preprocessing)
