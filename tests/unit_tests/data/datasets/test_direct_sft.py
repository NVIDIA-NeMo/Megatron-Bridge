# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import pytest
import torch

from megatron.bridge.data.collators.sft import text_chat_collate_fn
from megatron.bridge.data.datasets.direct_sft import DirectSFTDataset


pytestmark = pytest.mark.unit


def _example():
    return {"conversation": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]}


class _Processor:
    pass


def _custom_collate(
    examples,
    processor,
    *,
    sequence_length=None,
    pad_to_max_length=False,
    pad_to_multiple_of=128,
    enable_in_batch_packing=False,
    in_batch_packing_pad_to_multiple_of=1,
):
    del examples, processor, sequence_length, pad_to_max_length, pad_to_multiple_of
    return {
        "enable_in_batch_packing": enable_in_batch_packing,
        "in_batch_packing_pad_to_multiple_of": in_batch_packing_pad_to_multiple_of,
    }


def test_direct_sft_dataset_repeats_examples_to_target_length():
    dataset = DirectSFTDataset(
        base_examples=[_example()],
        target_length=3,
        processor=_Processor(),
        collate_impl=_custom_collate,
    )

    assert len(dataset) == 3
    assert dataset[2] == dataset[0]


def test_direct_sft_dataset_binds_shared_text_collate():
    class _Tokenizer:
        pad_token_id = 0
        added_tokens_decoder = {}
        chat_template = "{% generation %}{{ messages }}{% endgeneration %}"

        def apply_chat_template(self, conversation, tokenize=True, **kwargs):
            del conversation, kwargs
            assert tokenize is True
            return {"input_ids": [7, 8, 9], "assistant_masks": [0, 1, 1]}

    example = {
        "messages": [
            {"role": "user", "content": "ping"},
            {"role": "assistant", "content": "pong"},
        ]
    }
    dataset = DirectSFTDataset(
        base_examples=[example],
        target_length=1,
        processor=_Tokenizer(),
        collate_impl=text_chat_collate_fn,
        pad_to_multiple_of=1,
    )

    batch = dataset.collate_fn([dataset[0]])

    assert batch["tokens"].tolist() == [[7, 8, 9]]
    assert batch["labels"].tolist() == [[8, 9, -100]]
    assert batch["loss_mask"].tolist() == [[1.0, 1.0, 0.0]]


def test_direct_sft_dataset_forwards_supported_packing_options():
    dataset = DirectSFTDataset(
        base_examples=[_example()],
        target_length=1,
        processor=_Processor(),
        collate_impl=_custom_collate,
        enable_in_batch_packing=True,
        in_batch_packing_pad_to_multiple_of=8,
    )

    batch = dataset.collate_fn([dataset[0]])

    assert batch == {
        "enable_in_batch_packing": True,
        "in_batch_packing_pad_to_multiple_of": 8,
    }


def test_direct_sft_dataset_rejects_collate_without_packing_support():
    def _legacy_collate(examples, processor):
        del processor
        return {"count": len(examples)}

    with pytest.raises(ValueError, match="does not accept enable_in_batch_packing=True"):
        DirectSFTDataset(
            base_examples=[_example()],
            target_length=1,
            processor=_Processor(),
            collate_impl=_legacy_collate,
            enable_in_batch_packing=True,
        )


def test_direct_sft_dataset_requires_registered_or_explicit_collate():
    with pytest.raises(ValueError, match="No VLM collate function is registered"):
        DirectSFTDataset(
            base_examples=[_example()],
            target_length=1,
            processor=_Processor(),
        )


def test_direct_sft_dataset_collate_returns_tensors():
    def _tensor_collate(examples, processor, **kwargs):
        del examples, processor, kwargs
        return {"tokens": torch.tensor([[1, 2]])}

    dataset = DirectSFTDataset(
        base_examples=[_example()],
        target_length=1,
        processor=_Processor(),
        collate_impl=_tensor_collate,
    )

    assert dataset.collate_fn([dataset[0]])["tokens"].tolist() == [[1, 2]]
