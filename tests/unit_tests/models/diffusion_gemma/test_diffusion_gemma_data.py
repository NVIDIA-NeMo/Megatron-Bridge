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


@pytest.mark.parametrize(
    ("examples", "message"),
    [
        ([], "empty"),
        ([{"input_ids": [1], "canvas_ids": []}], "1..2"),
        ([{"input_ids": [1], "canvas_ids": [1, 2, 3]}], "1..2"),
    ],
)
def test_collator_rejects_empty_or_invalid_canvas_batches(examples, message):
    with pytest.raises(ValueError, match=message):
        from megatron.bridge.models.diffusion_gemma.data import collate_diffusion_gemma

        collate_diffusion_gemma(examples, pad_token_id=0, canvas_length=2)


def test_collator_rejects_nonpositive_padding_and_mixed_image_batches():
    from megatron.bridge.models.diffusion_gemma.data import collate_diffusion_gemma

    with pytest.raises(ValueError, match="pad_to_multiple_of"):
        collate_diffusion_gemma(
            [{"input_ids": [1], "canvas_ids": [2]}], pad_token_id=0, canvas_length=2, pad_to_multiple_of=0
        )
    with pytest.raises(ValueError, match="Mixed image/text"):
        collate_diffusion_gemma(
            [
                {"input_ids": [1], "canvas_ids": [2]},
                {"input_ids": [1], "canvas_ids": [2], "pixel_values": [[[1.0]]], "image_position_ids": [[[0, 0]]]},
            ],
            pad_token_id=0,
            canvas_length=2,
        )


def test_collator_rejects_image_shape_mismatch():
    from megatron.bridge.models.diffusion_gemma.data import collate_diffusion_gemma

    with pytest.raises(ValueError, match="share one patch grid"):
        collate_diffusion_gemma(
            [
                {
                    "input_ids": [1],
                    "canvas_ids": [2],
                    "pixel_values": [[[1.0], [2.0]]],
                    "image_position_ids": [[[0, 0], [0, 1]]],
                },
                {"input_ids": [1], "canvas_ids": [2], "pixel_values": [[[1.0]]], "image_position_ids": [[[0, 0]]]},
            ],
            pad_token_id=0,
            canvas_length=2,
        )


def test_mock_dataset_validates_completion_and_image_configuration():
    from megatron.bridge.models.diffusion_gemma.data import MockDiffusionGemmaDatasetConfig

    with pytest.raises(ValueError, match="completion_length"):
        MockDiffusionGemmaDatasetConfig(completion_length=0).finalize()
    with pytest.raises(ValueError, match="image token ids"):
        MockDiffusionGemmaDatasetConfig(image_patches_per_side=3).finalize()


def test_mock_dataset_builds_deterministic_train_and_validation_splits():
    from megatron.bridge.data.base import DatasetBuildContext
    from megatron.bridge.models.diffusion_gemma.data import MockDiffusionGemmaDatasetConfig

    config = MockDiffusionGemmaDatasetConfig(prompt_length=6, completion_length=4, canvas_length=4, seed=8)
    train, valid, test = config.build_datasets(DatasetBuildContext(train_samples=0, valid_samples=0, test_samples=0))
    assert len(train) == len(valid) == 1
    assert test is None
    first = train[0]
    assert torch.equal(first["canvas_ids"], train[0]["canvas_ids"])
    assert torch.equal(first["mm_token_type_ids"], torch.zeros_like(first["input_ids"]))
    assert torch.equal(train[0]["canvas_ids"], valid[0]["canvas_ids"])
    assert not torch.equal(train[0]["input_ids"], valid[0]["input_ids"])


def test_mock_dataset_marks_synthetic_image_tokens_and_validates_pooling():
    from megatron.bridge.models.diffusion_gemma.data import MockDiffusionGemmaDatasetConfig, _MockDiffusionGemmaDataset

    config = MockDiffusionGemmaDatasetConfig(
        prompt_length=4,
        completion_length=2,
        canvas_length=2,
        image_token_id=7,
        boi_token_id=8,
        eoi_token_id=9,
        image_patches_per_side=4,
        patch_size=1,
        pooling_kernel_size=2,
    )
    example = _MockDiffusionGemmaDataset(config, size=1, offset=0)[0]
    assert example["mm_token_type_ids"].sum() == 4
    assert example["pixel_values"].shape == (1, 16, 3)
    assert example["image_position_ids"].shape == (1, 16, 2)
    batch = _MockDiffusionGemmaDataset(config, size=1, offset=0).collate_fn([example])
    assert batch["pixel_values"].shape == (1, 16, 3)

    config.image_patches_per_side = 3
    with pytest.raises(ValueError, match="divisible by pooling_kernel_size"):
        _MockDiffusionGemmaDataset(config, size=1, offset=0)[0]


def test_real_dataset_rejects_malformed_rows_and_missing_eos(tmp_path):
    class _NoEos(_Processor):
        def __init__(self):
            super().__init__()
            self.tokenizer.eos_token_id = None

    path = tmp_path / "bad.jsonl"
    _write_rows(path, [{"messages": [], "completion": 12}])
    config = DiffusionGemmaDatasetConfig(train_path=str(path), hf_processor_path="test")
    with pytest.raises(ValueError, match="messages list and completion string"):
        DiffusionGemmaJSONLDataset(str(path), processor=_Processor(), config=config)

    _write_rows(path, [_row("10")])
    with pytest.raises(ValueError, match="EOS token"):
        DiffusionGemmaJSONLDataset(str(path), processor=_NoEos(), config=config)


def test_dataset_appends_eos_and_preserves_original_messages(tmp_path):
    path = tmp_path / "train.jsonl"
    row = _row("10 11")
    _write_rows(path, [row])
    processor = _Processor()
    config = DiffusionGemmaDatasetConfig(train_path=str(path), hf_processor_path="test", canvas_length=4)
    dataset = DiffusionGemmaJSONLDataset(str(path), processor=processor, config=config)
    assert dataset.targets[0].tolist() == [10, 11, 99]
    assert dataset.rows[0] == row


def test_dataset_finalize_rejects_nonpositive_lengths_and_bad_canvas_multiple(tmp_path):
    path = tmp_path / "train.jsonl"
    for field in ("canvas_length", "max_encoder_length", "pad_to_multiple_of"):
        config = DiffusionGemmaDatasetConfig(train_path=str(path), hf_processor_path="test", **{field: 0})
        with pytest.raises(ValueError, match="positive"):
            config.finalize()
    config = DiffusionGemmaDatasetConfig(
        train_path=str(path), hf_processor_path="test", canvas_length=3, pad_to_multiple_of=2
    )
    with pytest.raises(ValueError, match="divisible"):
        config.finalize()


def test_dataset_build_datasets_uses_processor_and_explicit_splits(monkeypatch, tmp_path):
    import transformers

    from megatron.bridge.data.base import DatasetBuildContext

    paths = [tmp_path / name for name in ("train.jsonl", "valid.jsonl", "test.jsonl")]
    for path in paths:
        _write_rows(path, [_row("10")])
    processor = _Processor()
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", lambda path: processor)
    config = DiffusionGemmaDatasetConfig(
        train_path=str(paths[0]), validation_path=str(paths[1]), test_path=str(paths[2]), hf_processor_path="processor"
    )
    train, valid, test = config.build_datasets(DatasetBuildContext(train_samples=1, valid_samples=1, test_samples=1))
    assert [len(dataset) for dataset in (train, valid, test)] == [1, 1, 1]
