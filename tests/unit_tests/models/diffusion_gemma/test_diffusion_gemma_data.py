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

"""Exercise the JSONL completion adapter with local synthetic data."""

import json
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from megatron.bridge.models.diffusion_gemma.data import (
    DiffusionGemmaDatasetConfig,
    DiffusionGemmaJSONLDataset,
)


pytestmark = pytest.mark.unit


class _Processor:
    def __init__(self):
        self.tokenizer = SimpleNamespace(eos_token_id=99, pad_token_id=0, encode=self.encode)
        self.calls = []

    def encode(self, completion, *, add_special_tokens):
        assert not add_special_tokens
        return [int(word) for word in completion.split()]

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["add_generation_prompt"] and kwargs["tokenize"]
        self.calls.append(messages)
        result = {"input_ids": torch.tensor([[2, 5]]), "mm_token_type_ids": torch.tensor([[0, 0]])}
        if any(item.get("type") == "image" for message in messages for item in message.get("content", [])):
            result.update(pixel_values=torch.zeros(1, 4, 48), image_position_ids=torch.zeros(1, 4, 2).long())
        return result


def _write_rows(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def _row(completion="10 11 12 13 14"):
    return {
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Answer in JSON."}]}],
        "completion": completion,
    }


def test_current_or_future_completion_blocks_never_enter_prompt(tmp_path):
    path = tmp_path / "train.jsonl"
    _write_rows(path, [_row()])
    config = DiffusionGemmaDatasetConfig(train_path=str(path), hf_processor_path="test", canvas_length=2)
    dataset = DiffusionGemmaJSONLDataset(str(path), processor=_Processor(), config=config)
    assert len(dataset) == 3
    assert dataset[0]["input_ids"].tolist() == [2, 5]
    assert dataset[0]["canvas_ids"].tolist() == [10, 11]
    assert dataset[1]["input_ids"].tolist() == [2, 5, 10, 11]
    assert dataset[1]["canvas_ids"].tolist() == [12, 13]
    assert dataset[2]["input_ids"].tolist() == [2, 5, 10, 11, 12, 13]
    assert dataset[2]["canvas_ids"].tolist() == [14, 99]
    batch = dataset.collate_fn([dataset[2]])
    assert batch["canvas_mask"].tolist() == [[1.0, 1.0]]
    assert len(dataset.processor.calls) == 7


def test_relative_image_is_loaded_on_cpu_and_collated(tmp_path):
    path = tmp_path / "train.jsonl"
    Image.new("RGB", (8, 8), "white").save(tmp_path / "image.png")
    row = _row("10")
    row["messages"][0]["content"].insert(0, {"type": "image", "image": "image.png"})
    _write_rows(path, [row])
    processor = _Processor()
    config = DiffusionGemmaDatasetConfig(train_path=str(path), hf_processor_path="test", canvas_length=4)
    dataset = DiffusionGemmaJSONLDataset(str(path), processor=processor, config=config)
    batch = dataset.collate_fn([dataset[0]])
    assert processor.calls[0][0]["content"][0]["image"].size == (8, 8)
    assert batch["pixel_values"].device.type == "cpu"
    assert batch["pixel_values"].shape == (1, 4, 48)
    assert batch["image_position_ids"].shape == (1, 4, 2)
    assert batch["canvas_ids"].tolist() == [[10, 99, 0, 0]]
    assert batch["canvas_mask"].tolist() == [[1.0, 1.0, 0.0, 0.0]]


def test_context_overflow_is_an_error_not_silent_target_truncation(tmp_path):
    path = tmp_path / "train.jsonl"
    _write_rows(path, [_row()])
    config = DiffusionGemmaDatasetConfig(
        train_path=str(path), hf_processor_path="test", canvas_length=2, max_encoder_length=3
    )
    dataset = DiffusionGemmaJSONLDataset(str(path), processor=_Processor(), config=config)
    with pytest.raises(ValueError, match="never silently truncated"):
        dataset[1]


def test_dataset_never_repartitions_and_rejects_same_split_file(tmp_path):
    path = tmp_path / "train.jsonl"
    config = DiffusionGemmaDatasetConfig(train_path=str(path), validation_path=str(path), hf_processor_path="test")
    with pytest.raises(ValueError, match="distinct split files"):
        config.finalize()


def test_remote_image_urls_are_rejected(tmp_path):
    path = tmp_path / "train.jsonl"
    row = _row()
    row["messages"][0]["content"] = [{"type": "image", "image": "https://example.invalid/image.png"}]
    _write_rows(path, [row])
    config = DiffusionGemmaDatasetConfig(train_path=str(path), hf_processor_path="test")
    dataset = DiffusionGemmaJSONLDataset(str(path), processor=_Processor(), config=config)
    with pytest.raises(ValueError, match="local file paths"):
        dataset[0]


def test_assistant_completion_cannot_be_supplied_as_prompt(tmp_path):
    path = tmp_path / "train.jsonl"
    row = _row()
    row["messages"][-1]["role"] = "assistant"
    _write_rows(path, [row])
    config = DiffusionGemmaDatasetConfig(train_path=str(path), hf_processor_path="test")
    with pytest.raises(ValueError, match="end before"):
        DiffusionGemmaJSONLDataset(str(path), processor=_Processor(), config=config)
