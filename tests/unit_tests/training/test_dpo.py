import pytest
import torch
import torch.nn.functional as F

from megatron.bridge.training.dpo import DPOLossConfig, dpo_loss, sequence_logprob_sums


def nemo_rl_dpo_reference(output_tensor: torch.Tensor, batch: dict[str, torch.Tensor], config: DPOLossConfig) -> float:
    """NeMo-RL DPOLossFn semantics (RL/nemo_rl/algorithms/loss/loss_functions.py),
    restated at pair granularity: the global per-live-pair mean of the total loss."""
    policy_sums, token_counts = sequence_logprob_sums(output_tensor, batch["loss_mask"])
    policy, ref = policy_sums, batch["ref_logprob_sum"].float()
    if config.preference_average_log_probs:
        policy = policy / token_counts.clamp(min=1)
        ref = ref / batch["ref_num_tokens"].clamp(min=1)
    rewards = policy - ref
    delta = rewards[0::2] - rewards[1::2]
    sample_mask = (batch["loss_multiplier"][0::2] > 0).float()
    live = sample_mask.sum()

    preference = (-F.logsigmoid(config.reference_policy_kl_penalty * delta) * sample_mask).sum() / live
    total = config.preference_loss_weight * preference
    if config.sft_loss_weight > 0:
        sft = -policy_sums[0::2]
        if config.sft_average_log_probs:
            sft = sft / token_counts[0::2].clamp(min=1)
        total = total + config.sft_loss_weight * (sft * sample_mask).sum() / live
    return total.item()


@pytest.mark.parametrize(
    "config",
    [
        DPOLossConfig(),
        DPOLossConfig(preference_average_log_probs=True),
        DPOLossConfig(sft_loss_weight=0.3, sft_average_log_probs=True),
    ],
    ids=["sum", "avg-pref", "sft-avg"],
)
def test_dpo_loss_matches_nemo_rl_reference_and_ignores_stub_pairs(config):
    """(loss_sum / live_pairs) must equal NeMo-RL DPOLossFn's global per-pair mean across sum/average
    modes and SFT mixing; a loss_multiplier=0 stub pair contributes nothing to the loss or its gradient."""
    torch.manual_seed(7)
    num_pairs, seq = 4, 6
    output_tensor = (torch.rand(2 * num_pairs, seq) * 3).requires_grad_()
    mask = (torch.rand(2 * num_pairs, seq) > 0.3).long()
    mask[:, 0] = 1  # no accidentally-empty live completion
    mask[4:6] = 0  # pair 2 is a stub: fully masked, multiplier 0
    batch = {
        "loss_mask": mask,
        "pair_id": torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]),
        "loss_multiplier": torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]),
        "ref_logprob_sum": torch.randn(2 * num_pairs) * 5 - 10,
        "ref_num_tokens": mask.sum(-1),
    }

    loss, num_live_pairs, metrics = dpo_loss(config, batch, output_tensor)

    assert num_live_pairs.item() == 3
    assert loss.item() / 3 == pytest.approx(nemo_rl_dpo_reference(output_tensor, batch, config), rel=1e-5)
    assert all(not value.requires_grad for value in metrics.values())
    loss.backward()
    assert torch.all(output_tensor.grad[4:6] == 0), "stub rows must receive no gradient"
    assert torch.all(output_tensor.grad[:4, 0] != 0), "live completion tokens must receive gradient"
