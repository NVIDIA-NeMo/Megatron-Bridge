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

"""CPU unit tests for the DSpark draft loss and acceptance metrics."""

import pytest
import torch

from megatron.bridge.training.post_training.dspark.loss import DSparkForwardOutput, dspark_loss


B, N, L, V = 2, 3, 4, 32


def _outputs(*, with_target: bool, with_confidence: bool, draft_equals_target: bool = False):
    torch.manual_seed(0)
    draft = torch.randn(B, N, L, V, requires_grad=True)
    target_ids = torch.randint(0, V, (B, N, L))
    eval_mask = torch.ones(B, N, L, dtype=torch.bool)
    keep = torch.ones(B, N, dtype=torch.bool)
    confidence = torch.randn(B, N, L) if with_confidence else None
    aligned = None
    if with_target:
        aligned = draft.detach().clone() if draft_equals_target else torch.randn(B, N, L, V)
    return DSparkForwardOutput(
        draft_logits=draft,
        target_ids=target_ids,
        eval_mask=eval_mask,
        block_keep_mask=keep,
        confidence_pred=confidence,
        aligned_target_logits=aligned,
    )


def test_acceptance_is_one_minus_tv_and_tau_when_distributions_match():
    # draft == target -> TV 0 -> accept_rate 1 -> tau = block_size + 1.
    out = _outputs(with_target=True, with_confidence=False, draft_equals_target=True)
    _, metrics = dspark_loss(out, l1_alpha=0.9, confidence_alpha=0.0)
    assert metrics["accept_rate"].item() == pytest.approx(1.0, abs=1e-5)
    assert metrics["tau"].item() == pytest.approx(L + 1.0, abs=1e-4)
    assert metrics["l1_loss"].item() == pytest.approx(0.0, abs=1e-5)


def test_mismatched_distributions_give_lower_acceptance():
    out = _outputs(with_target=True, with_confidence=False)
    _, metrics = dspark_loss(out, l1_alpha=0.9, confidence_alpha=0.0)
    assert 0.0 <= metrics["accept_rate"].item() < 1.0
    assert 1.0 <= metrics["tau"].item() <= L + 1.0


def test_ce_only_without_target_logits():
    out = _outputs(with_target=False, with_confidence=False)
    loss, metrics = dspark_loss(out, ce_alpha=0.1, l1_alpha=0.0, confidence_alpha=0.0)
    assert metrics["l1_loss"].item() == pytest.approx(0.0)
    assert metrics["confidence_loss"].item() == pytest.approx(0.0)
    # Loss reduces to ce_alpha * ce_loss.
    assert loss.item() == pytest.approx(0.1 * metrics["ce_loss"].item(), rel=1e-5)
    loss.backward()  # differentiable path


def test_three_terms_combine_with_alphas():
    out = _outputs(with_target=True, with_confidence=True)
    ce_a, l1_a, cf_a = 0.1, 0.9, 1.0
    loss, m = dspark_loss(out, ce_alpha=ce_a, l1_alpha=l1_a, confidence_alpha=cf_a)
    expected = ce_a * m["ce_loss"].item() + l1_a * m["l1_loss"].item() + cf_a * m["confidence_loss"].item()
    assert loss.item() == pytest.approx(expected, rel=1e-5)
    assert m["l1_loss"].item() > 0 and m["confidence_loss"].item() > 0
    loss.backward()


def test_missing_target_logits_raises_when_tv_requested():
    out = _outputs(with_target=False, with_confidence=False)
    with pytest.raises(ValueError, match="aligned_target_logits is required"):
        dspark_loss(out, l1_alpha=0.9)


def test_eval_mask_zero_positions_do_not_contribute():
    out = _outputs(with_target=True, with_confidence=False)
    out.eval_mask = torch.zeros(B, N, L, dtype=torch.bool)
    loss, metrics = dspark_loss(out, ce_alpha=0.1, l1_alpha=0.9, confidence_alpha=0.0)
    # All positions masked out: numerators are zero, so the loss is zero.
    assert loss.item() == pytest.approx(0.0, abs=1e-6)
    assert metrics["accept_rate"].item() == pytest.approx(0.0, abs=1e-6)
