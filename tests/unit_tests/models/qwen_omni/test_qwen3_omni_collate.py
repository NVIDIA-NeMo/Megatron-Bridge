# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.bridge.data.collators.registry import resolve_model_collate
from megatron.bridge.data.datasets.direct_sft import DirectSFTDataset
from megatron.bridge.data.datasets.utils import IGNORE_INDEX
from megatron.bridge.models.qwen_omni.data.collate_fn import qwen3_omni_collate_fn


pytestmark = pytest.mark.unit


class _OmniTokenizer:
    pad_token_id = 0
    padding_side = "left"
    added_tokens_decoder = {}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        markers = {
            "<|im_start|>assistant\n": [100],
            "<|im_start|>system\n": [101],
            "<|im_start|>developer\n": [102],
            "<|im_start|>user\n": [103],
            "<|im_start|>tool\n": [104],
            "<|im_end|>\n": [105],
            "<|im_end|>": [105],
        }
        return markers.get(text, [1])


class Qwen3OmniMoeProcessor:
    """Small processor double whose class name exercises the runtime registry."""

    def __init__(self, tokenized_rows: list[list[int]] | None = None) -> None:
        self.tokenizer = _OmniTokenizer()
        self.calls = []
        self.tokenized_rows = tokenized_rows or [[103, 20, 105, 100, 30, 31, 105]]

    def apply_chat_template(self, conversations, *, tokenize=False, **kwargs):
        self.calls.append((conversations, tokenize, kwargs))
        assert tokenize is True
        batch_size = len(conversations)
        tokenized_rows = [self.tokenized_rows[index % len(self.tokenized_rows)] for index in range(batch_size)]
        max_length = max(len(row) for row in tokenized_rows)
        rows = torch.full((batch_size, max_length), self.tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros_like(rows)
        padding_side = kwargs["processor_kwargs"]["padding_side"]
        for index, row in enumerate(tokenized_rows):
            start = 0 if padding_side == "right" else max_length - len(row)
            rows[index, start : start + len(row)] = torch.tensor(row)
            attention_mask[index, start : start + len(row)] = 1
        return {
            "input_ids": rows,
            "attention_mask": attention_mask,
            "pixel_values": torch.ones(batch_size, 3, 4, 4),
            "image_grid_thw": torch.tensor([[1, 2, 2] for _ in range(batch_size)]),
            "pixel_values_videos": torch.ones(batch_size, 3, 4, 4),
            "video_grid_thw": torch.tensor([[2, 2, 2] for _ in range(batch_size)]),
            "video_second_per_grid": torch.ones(batch_size),
            "input_features": torch.ones(batch_size, 8, 4),
            "feature_attention_mask": torch.ones(batch_size, 4, dtype=torch.long),
        }


def _examples() -> list[dict]:
    tools = [{"type": "function", "function": {"name": "lookup"}}]
    return [
        {
            "conversation": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"/data/image-{index}.png"},
                        {"type": "audio", "audio": f"/data/audio-{index}.wav"},
                        {"type": "text", "text": "What happens next?"},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": "An event."}]},
            ],
            "tools": tools,
        }
        for index in range(2)
    ]


def test_registry_resolves_qwen3_omni_collator():
    assert resolve_model_collate("Qwen3OmniMoeProcessor") is qwen3_omni_collate_fn


def test_direct_sft_dataset_collates_qwen3_omni_media_and_loss_mask():
    processor = Qwen3OmniMoeProcessor()
    examples = _examples()
    dataset = DirectSFTDataset(
        base_examples=examples,
        target_length=2,
        processor=processor,
        sequence_length=8,
        pad_to_max_length=True,
        pad_to_multiple_of=1,
    )

    batch = dataset.collate_fn([dataset[0], dataset[1]])

    conversations, tokenize, kwargs = processor.calls[0]
    assert tokenize is True
    assert conversations == [example["conversation"] for example in examples]
    assert kwargs["processor_kwargs"] == {"padding": True, "padding_side": "right"}
    assert kwargs["tools"] == examples[0]["tools"]
    assert processor.tokenizer.padding_side == "left"
    assert batch["input_ids"].shape == (2, 8)
    assert batch["labels"][0].tolist() == [
        IGNORE_INDEX,
        IGNORE_INDEX,
        IGNORE_INDEX,
        30,
        31,
        105,
        IGNORE_INDEX,
        IGNORE_INDEX,
    ]
    assert batch["loss_mask"][0].tolist() == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0]
    assert batch["pixel_values"].shape == (2, 3, 4, 4)
    assert batch["pixel_values_videos"].shape == (2, 3, 4, 4)
    assert batch["input_features"].shape == (2, 8, 4)
    assert batch["feature_attention_mask"].shape == (2, 4)
    assert batch["audio_feature_lengths"].tolist() == [4, 4]
    assert batch["position_ids"].shape == (2, 8)


def test_qwen3_omni_collator_right_pads_before_truncating_mixed_lengths():
    short_row = [103, 20, 105, 100, 30, 31, 105]
    long_row = [103, *([20] * 70), 105, 100, 30, 31, 105]
    processor = Qwen3OmniMoeProcessor([short_row, long_row])
    examples = _examples()
    dataset = DirectSFTDataset(
        base_examples=examples,
        target_length=2,
        processor=processor,
        sequence_length=16,
        pad_to_multiple_of=1,
    )

    batch = dataset.collate_fn([dataset[0], dataset[1]])

    assert batch["attention_mask"][0].sum().item() == len(short_row)
    assert batch["input_ids"][0, : len(short_row)].tolist() == short_row
    assert batch["loss_mask"][0].sum().item() == 3


def test_qwen3_omni_collator_rejects_in_batch_packing():
    with pytest.raises(ValueError, match="does not support in-batch packing"):
        qwen3_omni_collate_fn([], SimpleNamespace(), enable_in_batch_packing=True)
