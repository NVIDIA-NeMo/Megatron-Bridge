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

import pytest
import torch

from megatron.bridge.models.diffusion_gemma.diffusion_gemma_step import corrupt_canvas
from megatron.bridge.models.diffusion_gemma.loss import (
    VARIANTS,
    diffusion_sft_logps,
    diffusion_sft_loss,
    loo_to_denoiser_logits,
    normalize_variant,
    reweight,
    rf_alpha,
    rf_alpha_dot,
)


pytestmark = pytest.mark.unit


def test_uniform_state_corruption_uses_one_t_per_example_and_respects_mask():
    x0 = torch.arange(12).reshape(2, 6)
    mask = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.float32)
    xt, t = corrupt_canvas(
        x0, mask, noise_min=0.2, noise_max=0.2, vocab_size=100, generator=torch.Generator().manual_seed(4)
    )
    assert t.shape == (2,)
    assert torch.all((xt[:, 3:] == x0[:, 3:])[0])
    assert torch.equal(xt[0, 3:], x0[0, 3:])


def test_sft_variants_are_finite_and_loo_changes_observed_token():
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 17, requires_grad=True)
    x0 = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    xt = torch.tensor([[2, 2, 3, 9], [5, 1, 7, 8]])
    t = torch.tensor([0.2, 0.7])
    mask = torch.ones(2, 4)
    plain = diffusion_sft_loss(logits, x0, xt, t, mask, variant="base-sft", vocab_size=17)
    loo = diffusion_sft_loss(logits, x0, xt, t, mask, variant="loo-ce", vocab_size=17)
    assert plain.shape == loo.shape == (2,)
    assert torch.isfinite(plain).all() and torch.isfinite(loo).all()
    shifted = loo_to_denoiser_logits(logits.detach(), xt, 1 - t[:, None], 17)
    assert not torch.equal(shifted, logits.detach())
    plain.mean().backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_reweighted_loss_matches_inverse_alpha_weight():
    logits = torch.zeros(1, 2, 4)
    x0 = torch.tensor([[0, 1]])
    xt = x0.clone()
    t = torch.tensor([0.5])
    mask = torch.ones(1, 2)
    base = diffusion_sft_loss(logits, x0, xt, t, mask, variant="base-sft", vocab_size=4)
    weighted = diffusion_sft_loss(logits, x0, xt, t, mask, variant="reweighted-ce", vocab_size=4)
    assert torch.allclose(weighted, base * 2)


def test_megatron_loss_contract_preserves_per_example_mean():
    from megatron.bridge.models.diffusion_gemma.diffusion_gemma_step import diffusion_loss_function

    torch.manual_seed(1)
    logits = torch.randn(3, 4, 11)
    x0 = torch.randint(0, 11, (3, 4))
    xt = torch.randint(0, 11, (3, 4))
    t = torch.tensor([0.1, 0.5, 0.9])
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1], [1, 0, 0, 0]], dtype=torch.float32)
    loss_sum, count, metrics = diffusion_loss_function(
        (logits, x0, xt, t, mask), variant="loo-ce", vocab_size=11, check_for_nan_in_loss=False
    )
    expected = diffusion_sft_loss(logits, x0, xt, t, mask, variant="loo-ce", vocab_size=11).mean()
    assert int(count) == 3
    assert torch.allclose(loss_sum / count, expected)
    assert torch.allclose(metrics["lm loss"][0] / metrics["lm loss"][1], expected)


def test_collator_left_pads_prompts_right_pads_canvas_and_respects_tp_multiple():
    from megatron.bridge.models.diffusion_gemma.data import collate_diffusion_gemma

    batch = collate_diffusion_gemma(
        [
            {"input_ids": [5, 6, 7], "canvas_ids": [8, 1]},
            {"input_ids": [9], "canvas_ids": [10, 11, 1]},
        ],
        pad_token_id=0,
        canvas_length=4,
        pad_to_multiple_of=4,
    )
    assert batch["input_ids"].tolist() == [[0, 5, 6, 7], [0, 0, 0, 9]]
    assert batch["attention_mask"].tolist() == [[0, 1, 1, 1], [0, 0, 0, 1]]
    assert batch["canvas_ids"].tolist() == [[8, 1, 0, 0], [10, 11, 1, 0]]
    assert batch["canvas_mask"].tolist() == [[1, 1, 0, 0], [1, 1, 1, 0]]


def test_combined_objective_keeps_loss_reporting_and_normalization_consistent():
    from megatron.bridge.models.diffusion_gemma.diffusion_gemma_step import diffusion_loss_function

    logits = torch.zeros(2, 3, 7, requires_grad=True)
    x0 = torch.zeros(2, 3).long()
    mask = torch.ones(2, 3)
    encoder_loss = torch.tensor(2.0, requires_grad=True)
    total, count, metrics = diffusion_loss_function(
        (logits, x0, x0, torch.tensor([0.4, 0.7]), mask, encoder_loss),
        variant="base-sft",
        vocab_size=7,
        decoder_loss_weight=0.5,
        encoder_loss_weight=0.25,
        check_for_nan_in_loss=False,
    )
    expected = 0.5 * torch.log(torch.tensor(7.0)) + 0.25 * encoder_loss
    assert torch.allclose(total / count, expected)
    assert torch.allclose(metrics["lm loss"][0] / metrics["lm loss"][1], expected)
    assert torch.allclose(metrics["encoder loss"][0] / metrics["encoder loss"][1], encoder_loss)
    (total / count).backward()
    assert torch.allclose(encoder_loss.grad, torch.tensor(0.25))
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_eval_passes_do_not_shift_training_noise_across_resume():
    from megatron.bridge.models.diffusion_gemma import diffusion_gemma_step as step

    def run(with_validation):
        step._MICROBATCH_COUNTERS["train"].update(step=None, index=0)
        step._MICROBATCH_COUNTERS["eval"].update(step=None, index=0)
        seeds = []
        for training_step in (3, 4):
            if with_validation and training_step == 4:
                for _ in range(5):  # validation at the boundary, before the next train step
                    step.next_microbatch_index("eval", 4)
            seeds.append(
                step.noise_seed(
                    seed=1234,
                    mode="train",
                    step=training_step,
                    microbatch=step.next_microbatch_index("train", training_step),
                    dp_rank=1,
                )
            )
        return seeds

    assert run(with_validation=False) == run(with_validation=True)


def test_noise_seed_separates_phase_step_microbatch_and_rank():
    from megatron.bridge.models.diffusion_gemma.diffusion_gemma_step import noise_seed

    base = dict(seed=7, mode="train", step=2, microbatch=0, dp_rank=0)
    seeds = {
        noise_seed(**base),
        noise_seed(**{**base, "mode": "eval"}),
        noise_seed(**{**base, "step": 3}),
        noise_seed(**{**base, "microbatch": 1}),
        noise_seed(**{**base, "dp_rank": 1}),
    }
    assert len(seeds) == 5
    assert noise_seed(**base) == noise_seed(**base)


def test_loss_variant_aliases_and_unknown_names_are_explicit():
    assert normalize_variant("loo-reweighted-ce") == "reweighted-loo-ce"
    assert all(normalize_variant(variant) == variant for variant in VARIANTS)
    with pytest.raises(ValueError, match="Unknown variant"):
        normalize_variant("not-a-loss")


def test_reweight_schedules_and_clips_near_terminal_noise():
    t = torch.tensor([0.0, 0.5, 0.99])
    alpha = rf_alpha(t)
    alpha_dot = rf_alpha_dot(t)
    assert torch.equal(reweight("base-sft", t, alpha, alpha_dot), torch.ones(3))
    assert torch.allclose(reweight("reweighted-ce", t, alpha, alpha_dot), torch.tensor([1.0, 2.0, 100.0]))
    clipped = diffusion_sft_loss(
        torch.zeros(3, 1, 5),
        torch.zeros(3, 1, dtype=torch.long),
        torch.zeros(3, 1, dtype=torch.long),
        t,
        torch.ones(3, 1),
        variant="reweighted-ce",
        vocab_size=5,
        diffusion_weight_clip=4.0,
    )
    assert torch.allclose(clipped, torch.log(torch.tensor(5.0)) * torch.tensor([1.0, 2.0, 4.0]))


def test_loss_masks_empty_rows_and_logps_are_negative_summed_loss():
    logits = torch.zeros(2, 3, 4)
    x0 = torch.tensor([[0, 1, 2], [1, 2, 3]])
    xt = x0.clone()
    t = torch.tensor([0.25, 0.75])
    mask = torch.tensor([[1, 0, 0], [0, 0, 0]], dtype=torch.float32)
    loss = diffusion_sft_loss(logits, x0, xt, t, mask, variant="base-sft", vocab_size=4)
    logps = diffusion_sft_logps(logits, x0, xt, t, mask, variant="base-sft", vocab_size=4)
    assert torch.allclose(loss, torch.tensor([torch.log(torch.tensor(4.0)), 0.0]))
    assert torch.allclose(logps, torch.tensor([-torch.log(torch.tensor(4.0)), 0.0]))
    assert torch.allclose(logps / mask.sum(-1).clamp(min=1.0), -loss)


def test_loo_shift_broadcasts_alpha_and_rejects_out_of_range_tokens():
    logits = torch.zeros(1, 2, 3)
    xt = torch.tensor([[1, 2]])
    shifted = loo_to_denoiser_logits(logits, xt, torch.tensor([[0.5]]), vocab_size=3)
    assert shifted.shape == logits.shape
    assert shifted[0, 0, 1] > 0 and shifted[0, 1, 2] > 0
    with pytest.raises(RuntimeError):
        loo_to_denoiser_logits(logits, torch.tensor([[3, 0]]), torch.tensor([[0.5]]), vocab_size=3)
