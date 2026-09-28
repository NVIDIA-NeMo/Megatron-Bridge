# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from examples.conversion.hf_to_megatron_generate_audio_lm import (
    SingleBatchIterator,
    alm_forward_step,
    process_audio_inputs,
)
from transformers.feature_extraction_utils import BatchFeature


pytestmark = pytest.mark.unit


@pytest.mark.parametrize("mask_name", ["feature_attention_mask", "input_features_mask"])
def test_audio_frame_mask_reaches_model(monkeypatch, mask_name):
    audio = object()
    mask = torch.tensor([[1, 1, 0]], dtype=torch.long)
    token_ids = torch.tensor([[10, 20]])
    features = torch.ones(1, 128, 3)
    processor = Mock()
    processor.feature_extractor.sampling_rate = 16000
    processor.apply_chat_template.return_value = "audio prompt"
    processor.return_value = BatchFeature({"input_ids": token_ids, "input_features": features, mask_name: mask})
    load = Mock(return_value=audio)
    monkeypatch.setitem(process_audio_inputs.__globals__, "load_audio", load)

    input_ids, input_features, feature_mask, messages = process_audio_inputs(processor, "clip.flac", "")

    load.assert_called_once_with("clip.flac", 16000)
    processor.assert_called_once_with(text="audio prompt", audio=[audio], return_tensors="pt", padding=True)
    assert messages[0]["content"][1]["text"] == ""
    assert input_ids is token_ids
    assert input_features is features
    assert feature_mask is mask

    # Verify the renamed mask survives the iterator and forward adapter too.
    model = Mock(return_value=torch.ones(1, 2, 8))
    iterator = SingleBatchIterator(
        input_ids, torch.arange(2)[None], torch.ones_like(input_ids), features, feature_mask
    )
    alm_forward_step(iterator, model)
    assert model.call_args.kwargs["feature_attention_mask"] is mask


def test_text_only_processing_has_no_audio_mask():
    processor = Mock()
    processor.return_value = SimpleNamespace(input_ids=torch.tensor([[10, 20]]))

    input_ids, features, mask, messages = process_audio_inputs(processor, None, "Hello")

    assert input_ids is processor.return_value.input_ids
    assert features is None
    assert mask is None
    assert messages == [{"role": "user", "content": "Hello"}]
