# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from megatron.bridge.models.minicpmo.modeling_minicpmo import MiniCPMOPerception


class NativeAudio:
    @staticmethod
    def get_audio_embedding(facade, data, **kwargs):
        # A padding-sensitive encoder exposes accidental cross-sample padding.
        return [[data["audio_features"].mean(dim=-1).transpose(0, 1) * facade.weight]]


def perception():
    model = MiniCPMOPerception.__new__(MiniCPMOPerception)
    torch.nn.Module.__init__(model)
    model._hf_model_class = NativeAudio
    model.config = SimpleNamespace(audio_chunk_length=1.0)
    return model


def test_unequal_audio_clips_match_separate_calls_and_gradients():
    model = perception()
    weight = torch.tensor(2.0, requires_grad=True)
    features = torch.tensor([[[1.0, 3.0, 99.0, 99.0]], [[2.0, 4.0, 6.0, 8.0]]], requires_grad=True)
    lengths = [torch.tensor([2]), torch.tensor([], dtype=torch.long), torch.tensor([4])]
    output = model.encode_audio_samples(
        SimpleNamespace(weight=weight), {"audio_features": features, "audio_feature_lens": lengths}, batch_size=3
    )
    assert output[1] == []
    torch.testing.assert_close(output[0][0], torch.tensor([[4.0]]))
    torch.testing.assert_close(output[2][0], torch.tensor([[10.0]]))
    (output[0][0].sum() + output[2][0].sum()).backward()
    torch.testing.assert_close(features.grad, torch.tensor([[[1.0, 1.0, 0.0, 0.0]], [[0.5, 0.5, 0.5, 0.5]]]))
    torch.testing.assert_close(weight.grad, torch.tensor(7.0))


@pytest.mark.parametrize(
    "lengths", [[torch.tensor([0])], [torch.tensor([5])], [torch.tensor([2, 3])], [torch.tensor([2.0])]]
)
def test_invalid_audio_lengths_fail_before_native_call(lengths):
    with pytest.raises(ValueError):
        perception().encode_audio_samples(
            SimpleNamespace(weight=1),
            {"audio_features": torch.ones(1, 1, 4), "audio_feature_lens": lengths},
            batch_size=1,
        )
