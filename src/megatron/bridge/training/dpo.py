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

"""DPO pair loss and the logprob reduction it shares with the offline reference scorer.

Both jobs reduce per-token NLL to per-sequence sums through the same function, so a
step-0 margin can only come from the models or the tokens, never from the reduction.
"""

from dataclasses import dataclass

import torch


def sequence_logprob_sums(per_token_nll: torch.Tensor, loss_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce per-token NLL to per-row completion logprob sums and token counts.

    Args:
        per_token_nll: ``[rows, seq]`` next-token negative log-likelihood, as returned
            by ``model(..., labels=...)``.
        loss_mask: ``[rows, seq]``; nonzero marks completion-label positions.

    Returns:
        ``(logprob_sums, num_tokens)``, both ``[rows]``: float32 ``-sum(nll * mask)``
        per row and the number of counted positions.
    """
    if per_token_nll.shape != loss_mask.shape:
        raise ValueError(
            f"per_token_nll and loss_mask must have the same shape; got {per_token_nll.shape} vs {loss_mask.shape}."
        )
    mask = loss_mask.float()
    logprob_sums = -(per_token_nll.float() * mask).sum(dim=-1)
    num_tokens = mask.sum(dim=-1).long()
    return logprob_sums, num_tokens


def split_pair_rows(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a row-interleaved tensor into its (chosen, rejected) halves."""
    if rows.shape[0] % 2 != 0:
        raise ValueError(f"Interleaved pair rows must come in an even count; got {rows.shape[0]}.")
    return rows[0::2], rows[1::2]


@dataclass
class DPOLossConfig:
    """DPO loss hyperparameters.

    Field names and defaults deliberately mirror NeMo-RL's ``DPOLossConfig``
    (``nemo_rl/algorithms/loss/loss_functions.py``) so runs are 1:1 comparable.

    Attributes:
        reference_policy_kl_penalty: The DPO beta: temperature on the margin inside the
            sigmoid, i.e. the strength of the implicit KL leash to the reference policy.
        preference_loss_weight: Weight of the preference (``-logsigmoid``) term.
        sft_loss_weight: Weight of the auxiliary NLL term on the chosen response;
            0 disables it.
        preference_average_log_probs: Average (instead of sum) per-token logprobs
            over completion tokens in the preference margins.
        sft_average_log_probs: Same, for the auxiliary SFT term.
    """

    reference_policy_kl_penalty: float = 0.05
    preference_loss_weight: float = 1.0
    sft_loss_weight: float = 0.0
    preference_average_log_probs: bool = False
    sft_average_log_probs: bool = False


def dpo_loss(
    config: DPOLossConfig, batch: dict[str, torch.Tensor], output_tensor: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """DPO pair loss over one row-interleaved microbatch.

    ``loss = sum over pairs of loss_multiplier * (w_p * -logsigmoid(beta * margin) + w_s * sft_nll)``
    where ``margin = (pi_chosen - pi_rejected) - (ref_chosen - ref_rejected)``.

    Stub pairs (``loss_multiplier == 0``) contribute nothing to the loss, the count, or any metric.

    Args:
        config: Loss hyperparameters (see ``DPOLossConfig``).
        batch: Collate output; uses ``loss_mask``, ``pair_id``, ``loss_multiplier``,
            ``ref_logprob_sum`` and (in average mode) ``ref_num_tokens``, all
            row-aligned, chosen@even/rejected@odd.
        output_tensor: ``[rows, seq]`` per-token NLL from ``model(..., labels=...)``.

    Returns:
        ``(loss_sum, num_live_pairs, metrics)`` for ``calculate_per_token_loss=True``. Every
        metric is a detached ``[sum, count]`` 2-tensor with count = live pairs, except
        ``live fraction`` whose count is all pairs. Per-pair metrics are reported unweighted.
    """
    chosen_ids, rejected_ids = split_pair_rows(batch["pair_id"])
    if not torch.equal(chosen_ids, rejected_ids):
        raise ValueError(
            "Rows are not interleaved (chosen, rejected) pairs: pair_id[0::2] != pair_id[1::2]. "
            "The batch must come from preference_collate_fn without reordering."
        )

    policy_sums, token_counts = sequence_logprob_sums(output_tensor, batch["loss_mask"])
    policy, ref = policy_sums, batch["ref_logprob_sum"].float()
    if config.preference_average_log_probs:
        policy = policy_sums / token_counts.clamp(min=1)
        ref = ref / batch["ref_num_tokens"].clamp(min=1)

    policy_chosen, policy_rejected = split_pair_rows(policy)
    ref_chosen, ref_rejected = split_pair_rows(ref)
    rewards_chosen = policy_chosen - ref_chosen
    rewards_rejected = policy_rejected - ref_rejected
    margin = rewards_chosen - rewards_rejected

    preference = -torch.nn.functional.logsigmoid(config.reference_policy_kl_penalty * margin)
    per_pair = config.preference_loss_weight * preference

    sft = torch.zeros_like(preference)
    if config.sft_loss_weight > 0:
        sft = -split_pair_rows(policy_sums)[0]
        if config.sft_average_log_probs:
            sft = sft / split_pair_rows(token_counts)[0].clamp(min=1)
        per_pair = per_pair + config.sft_loss_weight * sft

    multiplier = split_pair_rows(batch["loss_multiplier"])[0].float()
    loss = (per_pair * multiplier).sum()
    live_mask = (multiplier > 0).float()
    num_live_pairs = live_mask.sum().to(torch.int)

    def report(per_pair_values: torch.Tensor) -> torch.Tensor:
        live_sum = (per_pair_values.detach().float() * live_mask).sum()
        return torch.cat([live_sum.view(1), num_live_pairs.view(1)])

    metrics = {
        "dpo loss": torch.cat([loss.detach().view(1), num_live_pairs.view(1)]),
        "preference loss": report(preference),
        "sft loss": report(sft),
        "margin": report(margin),
        "accuracy": report((margin > 0).float()),
        "rewards chosen": report(rewards_chosen),
        "rewards rejected": report(rewards_rejected),
        "live fraction": torch.cat([num_live_pairs.view(1), num_live_pairs.new_full((1,), live_mask.numel())]),
    }
    return loss, num_live_pairs, metrics
