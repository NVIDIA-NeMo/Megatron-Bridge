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

"""CPU unit tests for the DSpark draft heads."""

import pytest
import torch

from megatron.bridge.training.post_training.dspark.heads import (
    ConfidenceHead,
    GatedMarkovHead,
    RNNHead,
    VanillaMarkov,
    build_markov_head,
)


B, N, L, V, H, R = 2, 3, 4, 16, 8, 5


def test_vanilla_markov_bias_is_lowrank_token_bias():
    torch.manual_seed(0)
    head = VanillaMarkov(vocab_size=V, markov_rank=R)
    base = torch.randn(B, N, L, V)
    prev = torch.randint(0, V, (B, N, L))
    out = head.apply_block_logits(base, token_ids=prev, hidden_states=None)
    expected = base + head.markov_w2(head.markov_w1(prev))
    assert out.shape == (B, N, L, V)
    assert torch.allclose(out, expected, atol=1e-5)


def test_empty_block_is_identity():
    head = VanillaMarkov(vocab_size=V, markov_rank=R)
    base = torch.randn(B, N, 0, V)
    prev = torch.randint(0, V, (B, N, 0))
    out = head.apply_block_logits(base, token_ids=prev, hidden_states=None)
    assert out.shape == (B, N, 0, V)


@pytest.mark.parametrize("head_type,cls", [("vanilla", VanillaMarkov), ("gated", GatedMarkovHead), ("rnn", RNNHead)])
def test_build_markov_head_dispatch_and_shape(head_type, cls):
    head = build_markov_head(markov_rank=R, markov_head_type=head_type, vocab_size=V, hidden_size=H)
    assert isinstance(head, cls)
    base = torch.randn(B, N, L, V)
    prev = torch.randint(0, V, (B, N, L))
    hidden = torch.randn(B, N, L, H)
    out = head.apply_block_logits(base, token_ids=prev, hidden_states=hidden)
    assert out.shape == (B, N, L, V)
    assert torch.isfinite(out).all()


def test_build_markov_head_disabled_when_rank_zero():
    assert build_markov_head(markov_rank=0, markov_head_type="vanilla", vocab_size=V, hidden_size=H) is None


def test_build_markov_head_rejects_unknown_type():
    with pytest.raises(ValueError, match="Unsupported markov_head_type"):
        build_markov_head(markov_rank=R, markov_head_type="bogus", vocab_size=V, hidden_size=H)


def test_gated_and_rnn_require_hidden_states():
    for head in (
        GatedMarkovHead(vocab_size=V, markov_rank=R, hidden_size=H),
        RNNHead(vocab_size=V, markov_rank=R, hidden_size=H),
    ):
        base = torch.randn(B, N, L, V)
        prev = torch.randint(0, V, (B, N, L))
        with pytest.raises(ValueError, match="requires hidden_states"):
            head.apply_block_logits(base, token_ids=prev, hidden_states=None)


def test_rnn_head_state_makes_positions_prefix_dependent():
    # Changing an earlier prefix token must change a later position's bias (state carries).
    torch.manual_seed(0)
    head = RNNHead(vocab_size=V, markov_rank=R, hidden_size=H)
    base = torch.zeros(1, 1, L, V)
    hidden = torch.randn(1, 1, L, H)
    prev_a = torch.tensor([[[1, 2, 3, 4]]])
    prev_b = torch.tensor([[[9, 2, 3, 4]]])  # differs only at position 0
    out_a = head.apply_block_logits(base, token_ids=prev_a, hidden_states=hidden)
    out_b = head.apply_block_logits(base, token_ids=prev_b, hidden_states=hidden)
    # The last position differs because the recurrent state depends on position 0.
    assert not torch.allclose(out_a[..., -1, :], out_b[..., -1, :])


def test_confidence_head_projects_to_scalar_logit():
    head = ConfidenceHead(input_dim=H)
    out = head(torch.randn(B, N, L, H))
    assert out.shape == (B, N, L)
