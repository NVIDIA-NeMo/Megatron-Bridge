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

"""Unit tests for the Inkling2 output layer's soft-cap and padded-vocabulary mask."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn


pytest.importorskip("transformers", minversion="5.23")

from megatron.bridge.models.inkling.modeling_inkling2.model import Inkling2OutputLayerMixin


pytestmark = pytest.mark.unit


class _Linear(nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.logits = logits
        self.output_size = logits.shape[-1]
        self.config = SimpleNamespace(inkling2_final_logit_softcap=30.0, inkling2_unpadded_vocab_size=5)

    def forward(self):
        return self.logits, None


class _Output(Inkling2OutputLayerMixin, _Linear):
    pass


def test_padded_vocabulary_gets_no_probability_and_a_finite_entropy():
    logits = torch.randn(3, 8, requires_grad=True)
    output, _ = _Output(logits)()
    probs = torch.softmax(output, dim=-1)
    assert torch.equal(probs[:, 5:], torch.zeros(3, 3))
    assert torch.allclose(output[:, :5], 30.0 * torch.tanh(logits[:, :5] / 30.0))
    # Entropy as RL trainers compute it, sum(p * logit) over the vocabulary: 0 * -inf would make it NaN.
    entropy = torch.logsumexp(output, -1) - (probs * output).sum(-1)
    assert torch.isfinite(entropy).all()
    entropy.sum().backward()
    assert torch.isfinite(logits.grad).all() and torch.equal(logits.grad[:, 5:], torch.zeros(3, 3))
