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

from megatron.bridge.training.post_training.dspark.loss import (
    DSparkForwardOutput,
    assert_per_token_loss_config,
    dspark_loss,
)


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


def _ratio(report, key):
    """Global ratio of a reducible ``[numerator, denominator]`` reporting pair.

    No zero-denominator guard on purpose: ``training/train.py`` divides the reduced
    pair unguarded, so a test that silently substituted 0.0 here would hide exactly
    the NaN that a 0/0 pair produces in a real run.
    """
    numerator, denominator = report[key]
    return (numerator / denominator).item()


def _num(report, key):
    """Numerator of a reducible reporting pair."""
    return report[key][0].item()


def _den(report, key):
    """Denominator of a reducible reporting pair."""
    return report[key][1].item()


def test_acceptance_is_one_minus_tv_and_tau_when_distributions_match():
    # draft == target -> L1 0 -> accept_rate 1 -> tau = block_size + 1.
    out = _outputs(with_target=True, with_confidence=False, draft_equals_target=True)
    _, _, report = dspark_loss(out, l1_alpha=0.9, confidence_alpha=0.0)
    assert _ratio(report, "dspark accept rate") == pytest.approx(1.0, abs=1e-5)
    assert _ratio(report, "dspark tau") == pytest.approx(L + 1.0, abs=1e-4)
    assert _ratio(report, "dspark l1 loss") == pytest.approx(0.0, abs=1e-5)


def test_disjoint_distributions_give_raw_l1_of_two_and_zero_acceptance():
    """The training term is the *raw* L1 (paper Eq. 10), not ``0.5 * L1``.

    Two distributions with disjoint support are exactly L1 distance 2 apart, so a
    raw-L1 objective reports ``l1_loss == 2`` and a total-variation objective would
    report 1. Acceptance ``1 - 0.5 * L1`` is 0 in both readings; asserting the pair
    together pins which factor belongs where, so the loss and its documentation
    cannot drift apart again.
    """
    big = 30.0  # softmax(±big) is one-hot to well within fp32 resolution
    draft = torch.full((1, 1, 1, V), -big)
    draft[..., 0] = big
    draft.requires_grad_(True)
    target = torch.full((1, 1, 1, V), -big)
    target[..., 1] = big
    out = DSparkForwardOutput(
        draft_logits=draft,
        target_ids=torch.zeros(1, 1, 1, dtype=torch.long),
        eval_mask=torch.ones(1, 1, 1, dtype=torch.bool),
        block_keep_mask=torch.ones(1, 1, dtype=torch.bool),
        aligned_target_logits=target,
    )
    _, _, report = dspark_loss(out, ce_alpha=0.1, l1_alpha=0.9, confidence_alpha=0.0)
    assert _ratio(report, "dspark l1 loss") == pytest.approx(2.0, abs=1e-5)
    assert _ratio(report, "dspark accept rate") == pytest.approx(0.0, abs=1e-5)


def test_mismatched_distributions_give_lower_acceptance():
    out = _outputs(with_target=True, with_confidence=False)
    _, _, report = dspark_loss(out, l1_alpha=0.9, confidence_alpha=0.0)
    assert 0.0 <= _ratio(report, "dspark accept rate") < 1.0
    assert 1.0 <= _ratio(report, "dspark tau") <= L + 1.0


def test_ce_only_without_target_logits():
    out = _outputs(with_target=False, with_confidence=False)
    loss, num_tokens, report = dspark_loss(out, ce_alpha=0.1, l1_alpha=0.0, confidence_alpha=0.0)
    assert _ratio(report, "dspark l1 loss") == pytest.approx(0.0)
    assert _ratio(report, "dspark confidence loss") == pytest.approx(0.0)
    assert num_tokens.item() == B * N * L
    # Loss reduces to ce_alpha * ce_numerator.
    assert loss.item() == pytest.approx(0.1 * _num(report, "dspark ce loss"), rel=1e-5)
    loss.backward()  # differentiable path


def test_three_terms_combine_with_alphas():
    out = _outputs(with_target=True, with_confidence=True)
    ce_a, l1_a, cf_a = 0.1, 0.9, 1.0
    loss, _, report = dspark_loss(out, ce_alpha=ce_a, l1_alpha=l1_a, confidence_alpha=cf_a)
    expected = (
        ce_a * _num(report, "dspark ce loss")
        + l1_a * _num(report, "dspark l1 loss")
        + cf_a * _num(report, "dspark confidence loss")
    )
    assert loss.item() == pytest.approx(expected, rel=1e-5)
    assert _ratio(report, "dspark l1 loss") > 0 and _ratio(report, "dspark confidence loss") > 0
    loss.backward()


def test_returned_loss_is_unnormalized_numerator():
    """The loss must be a sum, not a mean: Megatron-Core divides once, globally."""
    out = _outputs(with_target=True, with_confidence=False)
    loss, _, report = dspark_loss(out, ce_alpha=0.1, l1_alpha=0.9, confidence_alpha=0.0)
    assert loss.item() == pytest.approx(_num(report, "dspark loss"), rel=1e-6)
    # A mean would be denominator-times smaller; the denominator here is >> 1.
    assert _den(report, "dspark loss") > 1.0


def test_uneven_micro_batches_reduce_to_the_global_token_mean():
    """Summing pairs across micro-batches gives the global mean, not a mean of means.

    One supervised token in the first micro-batch and three in the second: dividing
    each micro-batch by its own denominator and averaging over-weights the single
    token. Only ``sum(num) / sum(den)`` matches the full-batch reference.
    """
    torch.manual_seed(7)
    draft = torch.randn(1, 4, 1, V)
    target = torch.randn(1, 4, 1, V)
    target_ids = torch.randint(0, V, (1, 4, 1))
    keep = torch.ones(1, 4, dtype=torch.bool)

    def run(block_slice):
        out = DSparkForwardOutput(
            draft_logits=draft[:, block_slice],
            target_ids=target_ids[:, block_slice],
            eval_mask=torch.ones(1, block_slice.stop - block_slice.start, 1, dtype=torch.bool),
            block_keep_mask=keep[:, block_slice],
            aligned_target_logits=target[:, block_slice],
        )
        return dspark_loss(out, ce_alpha=0.1, l1_alpha=0.9, confidence_alpha=0.0)

    _, _, full = run(slice(0, 4))
    _, _, first = run(slice(0, 1))
    _, _, second = run(slice(1, 4))

    for key in ("dspark ce loss", "dspark l1 loss"):
        summed = (first[key][0] + second[key][0]) / (first[key][1] + second[key][1])
        assert summed.item() == pytest.approx(_ratio(full, key), rel=1e-5)
        # The biased mean-of-means differs, so this test actually discriminates.
        mean_of_means = 0.5 * (first[key][0] / first[key][1] + second[key][0] / second[key][1])
        assert mean_of_means.item() != pytest.approx(_ratio(full, key), rel=1e-3)


def test_chunked_terms_match_unchunked_and_keep_gradients():
    """Chunk size must not change either term's value or the gradient."""
    out_a = _outputs(with_target=True, with_confidence=False)
    out_b = _outputs(with_target=True, with_confidence=False)
    loss_a, _, report_a = dspark_loss(out_a, l1_alpha=0.9, confidence_alpha=0.0, chunk_tokens=10_000)
    loss_b, _, report_b = dspark_loss(out_b, l1_alpha=0.9, confidence_alpha=0.0, chunk_tokens=3)
    for key in ("dspark ce loss", "dspark l1 loss"):
        assert _num(report_a, key) == pytest.approx(_num(report_b, key), rel=1e-6)
    loss_a.backward()
    loss_b.backward()
    torch.testing.assert_close(out_a.draft_logits.grad, out_b.draft_logits.grad, rtol=1e-5, atol=1e-6)


def test_missing_target_logits_raises_when_l1_requested():
    out = _outputs(with_target=False, with_confidence=False)
    with pytest.raises(ValueError, match="aligned_target_logits is required"):
        dspark_loss(out, l1_alpha=0.9)


def test_eval_mask_zero_positions_do_not_contribute():
    out = _outputs(with_target=True, with_confidence=False)
    out.eval_mask = torch.zeros(B, N, L, dtype=torch.bool)
    loss, num_tokens, report = dspark_loss(out, ce_alpha=0.1, l1_alpha=0.9, confidence_alpha=0.0)
    # All positions masked out: numerators are zero, so the loss is zero.
    assert loss.item() == pytest.approx(0.0, abs=1e-6)
    assert num_tokens.item() == 0
    # Assert the pair, not its ratio. An empty micro-batch contributes 0/0, which is
    # correct for a reducible pair (it adds nothing to either sum) and only becomes a
    # NaN if every rank's every micro-batch is empty, exactly as "lm loss" behaves.
    for key in ("dspark ce loss", "dspark l1 loss", "dspark accept rate"):
        assert _num(report, key) == pytest.approx(0.0, abs=1e-6)
        assert _den(report, key) == pytest.approx(0.0, abs=1e-6)


def test_disabled_confidence_head_reports_a_finite_zero_not_nan():
    """A disabled term must share the live denominator, or the reducer logs NaN.

    ``training/train.py`` sums each reporting pair over micro-batches and the DP
    group and then divides ``val[0] / val[1]`` with no zero guard, so emitting a
    0/0 pair for a term that is off for the whole run would log NaN every step.
    """
    out = _outputs(with_target=True, with_confidence=False)
    _, _, report = dspark_loss(out, ce_alpha=0.1, l1_alpha=0.9, confidence_alpha=0.0)
    assert _den(report, "dspark confidence loss") > 0.0
    assert _ratio(report, "dspark confidence loss") == pytest.approx(0.0)


def test_acceptance_metrics_are_absent_without_a_teacher():
    """No teacher signal means no acceptance keys at all, rather than 0/0 pairs."""
    out = _outputs(with_target=False, with_confidence=False)
    _, _, report = dspark_loss(out, ce_alpha=0.1, l1_alpha=0.0, confidence_alpha=0.0)
    assert "dspark accept rate" not in report
    assert "dspark tau" not in report


class _Config:
    def __init__(self, calculate_per_token_loss):
        self.calculate_per_token_loss = calculate_per_token_loss


def test_assert_per_token_loss_config_rejects_the_biased_default():
    assert_per_token_loss_config(_Config(True))  # does not raise
    with pytest.raises(ValueError, match="calculate_per_token_loss=True"):
        assert_per_token_loss_config(_Config(False))


def test_assert_per_token_loss_config_rejects_a_config_without_the_field():
    class _Wrong:
        pass

    with pytest.raises(AttributeError):
        assert_per_token_loss_config(_Wrong())
