from types import SimpleNamespace

import pytest
import torch

from megatron.bridge.engine import AdamParams, ModelOutput, NormalizationPolicy, PreparedMicrobatch
from megatron.bridge.engine.accumulation import AccumulationWindow
from megatron.bridge.engine.losses import cross_entropy
from megatron.bridge.engine.optim import apply_adam_params
from megatron.bridge.engine.schedule import model_loss_mask


def test_accumulation_window_lifecycle_and_denominators():
    window = AccumulationWindow()
    assert not window.is_open
    window.add(num_loss_tokens=10, num_samples=2, loss_sum=3.0)
    window.add(num_loss_tokens=5, num_samples=1, loss_sum=1.5)
    assert window.is_open and window.num_calls == 2
    assert window.denominator(NormalizationPolicy.TOKEN_MEAN) == 15.0
    assert window.denominator(NormalizationPolicy.SAMPLE_MEAN) == 3.0
    assert window.denominator(NormalizationPolicy.NONE) is None
    assert window.loss_sum == 4.5
    window.reset()
    assert not window.is_open and window.num_loss_tokens == 0


def test_apply_adam_params_preserves_group_multipliers():
    base = {"weight_decay": 0.1, "betas": (0.9, 0.999), "eps": 1e-6}
    groups = [
        {"lr": 0.0, "lr_mult": 1.0, "wd_mult": 1.0, **base},
        {"lr": torch.tensor(0.0), "lr_mult": 0.5, "wd_mult": 0.0, **base},
    ]
    params = AdamParams(2e-5, beta1=0.8, beta2=0.95, eps=1e-8, weight_decay=0.1)
    apply_adam_params(SimpleNamespace(param_groups=groups), params)
    assert groups[0]["lr"] == 2e-5 and groups[0]["weight_decay"] == pytest.approx(0.1)
    assert groups[0]["betas"] == (0.8, 0.95) and groups[0]["eps"] == 1e-8
    assert groups[1]["lr"].item() == pytest.approx(1e-5) and groups[1]["weight_decay"] == 0.0


def test_cross_entropy_uses_active_mask_only():
    batch = PreparedMicrobatch(
        input_ids=torch.zeros(1, 4, dtype=torch.int64),
        target_ids=torch.zeros(1, 4, dtype=torch.int64),
        position_ids=torch.arange(4).unsqueeze(0),
        loss_mask=torch.tensor([[0.0, 1.0, 1.0, 1.0]]),
        padding_mask=torch.tensor([[False, False, False, True]]),  # last slot is padding even though loss_mask=1
        packed_seq_params=None,
    )
    output = ModelOutput(logprobs=torch.tensor([[-1.0, -2.0, -0.5, -9.0]], requires_grad=True))
    loss = cross_entropy(batch, output)
    assert loss.item() == pytest.approx(2.5)
    loss.backward()
    assert output.logprobs.grad.tolist() == [[0.0, -1.0, -1.0, 0.0]]


def test_model_loss_mask_follows_mtp_target_scope_and_never_includes_padding():
    batch = PreparedMicrobatch(
        input_ids=torch.zeros(1, 4, dtype=torch.int64),
        target_ids=torch.zeros(1, 4, dtype=torch.int64),
        position_ids=torch.arange(4).unsqueeze(0),
        loss_mask=torch.tensor([[0.0, 1.0, 1.0, 1.0]]),
        padding_mask=torch.tensor([[False, False, False, True]]),
        packed_seq_params=None,
    )
    primary = model_loss_mask(batch, "primary")
    all_valid = model_loss_mask(batch, "all_valid")
    assert primary.dtype == all_valid.dtype == torch.float32
    assert primary.tolist() == [[0.0, 1.0, 1.0, 0.0]]
    assert all_valid.tolist() == [[1.0, 1.0, 1.0, 0.0]]
    with pytest.raises(ValueError, match="mtp_target_scope"):
        model_loss_mask(batch, "response")
