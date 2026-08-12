# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

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
        assert kwargs["padding"] in {True, False}
        assert self.tokenizer.padding_side == "right"
        batch_size = len(conversations)
        length = 4 if kwargs["padding"] else 3 + len(conversations[0])
        return {
            "input_ids": torch.arange(11, 11 + length).unsqueeze(0).expand(batch_size, -1).clone(),
            "attention_mask": torch.ones(batch_size, length, dtype=torch.long),
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


def test_muse_glimmer_collator_packs_sequences_and_concatenates_visual_inputs(monkeypatch):
    monkeypatch.setattr(
        "megatron.bridge.models.muse_glimmer.data.collate_fn.build_assistant_loss_mask",
        lambda _example, input_ids, *_args, **_kwargs: torch.cat([torch.zeros(1), torch.ones(input_ids.numel() - 1)]),
    )
    processor = _MuseGlimmerProcessor()
    examples = [
        {"conversation": [{"role": "user", "content": "image"}]},
        {
            "conversation": [
                {"role": "user", "content": "image"},
                {"role": "assistant", "content": "ok"},
            ]
        },
    ]

    batch = muse_glimmer_collate_fn(
        examples,
        processor,
        sequence_length=16,
        enable_in_batch_packing=True,
        in_batch_packing_pad_to_multiple_of=4,
    )

    assert processor.tokenizer.padding_side == "left"
    assert batch["input_ids"].shape == (1, 12)
    assert torch.equal(batch["cu_seqlens_q"], torch.tensor([0, 4, 9], dtype=torch.int32))
    assert torch.equal(batch["cu_seqlens_q_padded"], torch.tensor([0, 4, 12], dtype=torch.int32))
    assert batch["visual_inputs"].pixel_values.shape == (4, 12)
    assert torch.equal(batch["visual_inputs"].image_grid_thw, torch.tensor([[1, 2, 4], [1, 2, 4]]))
