# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.bridge.data.collators.registry import resolve_model_collate
from megatron.bridge.models.muse_glimmer.data.collate_fn import muse_glimmer_collate_fn


class _Tokenizer:
    pad_token_id = 0
    padding_side = "left"
    all_special_ids = []

    def __call__(self, text, **kwargs):
        del kwargs
        return {"input_ids": [ord(char) for char in text]}


class _MuseGlimmerProcessor:
    def __init__(self) -> None:
        self.tokenizer = _Tokenizer()

    def apply_chat_template(self, conversations, **kwargs):
        assert kwargs["tokenize"] is True
        assert kwargs["padding"] is True
        assert self.tokenizer.padding_side == "right"
        batch_size = len(conversations)
        return {
            "input_ids": torch.tensor([[11, 12, 13, 14]] * batch_size),
            "attention_mask": torch.ones(batch_size, 4, dtype=torch.long),
            "pixel_values": torch.arange(24, dtype=torch.float32).reshape(2, 12),
            "image_grid_thw": torch.tensor([[1, 2, 4]]),
        }


def test_muse_glimmer_processor_resolves_family_collator():
    assert resolve_model_collate("MuseGlimmerProcessor") is muse_glimmer_collate_fn


def test_muse_glimmer_collator_builds_shifted_labels_and_visual_inputs(monkeypatch):
    monkeypatch.setattr(
        "megatron.bridge.models.muse_glimmer.data.collate_fn.build_assistant_loss_mask",
        lambda *args, **kwargs: torch.tensor([0.0, 0.0, 1.0, 1.0]),
    )
    processor = _MuseGlimmerProcessor()
    batch = muse_glimmer_collate_fn(
        [{"conversation": [{"role": "user", "content": "image"}, {"role": "assistant", "content": "ok"}]}],
        processor,
        pad_to_multiple_of=1,
    )

    assert processor.tokenizer.padding_side == "left"
    assert torch.equal(batch["labels"], torch.tensor([[-100, 13, 14, -100]]))
    assert torch.equal(batch["loss_mask"], torch.tensor([[0.0, 1.0, 1.0, 0.0]]))
    assert torch.equal(batch["visual_inputs"].pixel_values, torch.arange(24, dtype=torch.float32).reshape(2, 12))
    assert torch.equal(batch["visual_inputs"].image_grid_thw, torch.tensor([[1, 2, 4]]))
    assert "pixel_values" not in batch


def test_muse_glimmer_collator_rejects_in_batch_packing():
    with pytest.raises(ValueError, match="does not support in-batch packing"):
        muse_glimmer_collate_fn([], SimpleNamespace(), enable_in_batch_packing=True)
