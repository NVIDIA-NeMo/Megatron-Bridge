import math
from types import SimpleNamespace

import pytest
import torch

from megatron.bridge.engine import AdamParams, ModelOutput, NormalizationPolicy, PreparedMicrobatch
from megatron.bridge.engine.accumulation import AccumulationWindow
from megatron.bridge.engine.batching import MicrobatchMapping, SequenceMapping
from megatron.bridge.engine.losses import cross_entropy, importance_sampling, ppo_clip, target_logprobs, value_loss
from megatron.bridge.engine.optim import apply_adam_params
from megatron.bridge.engine.schedule import model_loss_mask, scored_tensor, split_loss_return, to_model_output
from megatron.bridge.engine.value_head import apply_value_head_model_settings, make_value_head_hook


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


def _rl_batch(**fields) -> PreparedMicrobatch:
    """Four slots: slot 0 is a prompt (loss_mask 0), slots 1-2 are active, slot 3 is padding."""
    return PreparedMicrobatch(
        input_ids=torch.zeros(1, 4, dtype=torch.int64),
        target_ids=torch.zeros(1, 4, dtype=torch.int64),
        position_ids=torch.arange(4).unsqueeze(0),
        loss_mask=torch.tensor([[0.0, 1.0, 1.0, 1.0]]),
        padding_mask=torch.tensor([[False, False, False, True]]),
        packed_seq_params=None,
        **{name: torch.tensor([value], dtype=torch.float32) for name, value in fields.items()},
    )


def _rl_output() -> ModelOutput:
    # Active logprob diffs vs old: +0.5 (ratio e^0.5 = 1.6487) and -0.1 (ratio e^-0.1 = 0.9048).
    return ModelOutput(logprobs=torch.tensor([[-9.0, -1.0, -2.1, -7.0]], requires_grad=True))


def test_importance_sampling_math_and_requirements():
    batch = _rl_batch(advantages=[0.0, 2.0, -1.0, 5.0], old_logprobs=[0.0, -1.5, -2.0, 0.0])
    output = _rl_output()
    loss, metrics = importance_sampling(batch, output)
    ratios = torch.tensor([0.5, -0.1]).exp()
    assert loss.item() == pytest.approx(-(ratios[0] * 2.0 + ratios[1] * -1.0).item(), rel=1e-6)
    assert metrics["abs_logprob_diff_sum"].item() == pytest.approx(0.6, rel=1e-6)
    loss.backward()
    grad = output.logprobs.grad[0]
    assert grad[0] == 0.0 and grad[3] == 0.0  # prompt and padding get no gradient
    assert grad[1].item() == pytest.approx(-(ratios[0] * 2.0).item(), rel=1e-6)
    with pytest.raises(ValueError, match="Datum.advantages"):
        importance_sampling(_rl_batch(old_logprobs=[0.0, 0.0, 0.0, 0.0]), _rl_output())


def test_importance_sampling_truncated_importance_weights():
    batch = _rl_batch(
        advantages=[0.0, 1.0, 1.0, 0.0],
        old_logprobs=[0.0, -1.5, -2.0, 0.0],
        rollout_logprobs=[0.0, -3.5, -2.0, 0.0],  # weight e^2 (truncated to 2.0) and weight 1
    )
    loss, metrics = importance_sampling(batch, _rl_output(), tis_coef=2.0)
    ratios = torch.tensor([0.5, -0.1]).exp()
    assert loss.item() == pytest.approx(-(2.0 * ratios[0] + 1.0 * ratios[1]).item(), rel=1e-6)
    assert metrics["tis_truncated_tokens"].item() == 1.0
    with pytest.raises(ValueError, match="rollout_logprobs"):
        importance_sampling(_rl_batch(advantages=[0.0] * 4, old_logprobs=[0.0] * 4), _rl_output(), tis_coef=2.0)


def test_ppo_clip_clips_both_sides_and_reports_metrics():
    # Slot 1: ratio 1.6487 above 1.2 with A>0 -> clipped term 1.2*A is the min.
    # Slot 2: ratio 0.9048 below 1-0.05 with A<0 -> clipped term 0.95*A is the min.
    batch = _rl_batch(advantages=[0.0, 2.0, -1.0, 5.0], old_logprobs=[0.0, -1.5, -2.0, 0.0])
    output = _rl_output()
    loss, metrics = ppo_clip(batch, output, clip_low=0.05, clip_high=0.2)
    assert loss.item() == pytest.approx(-(1.2 * 2.0 + 0.95 * -1.0), rel=1e-6)
    assert metrics["clipped_tokens"].item() == 2.0
    assert metrics["abs_logprob_diff_sum"].item() == pytest.approx(0.6, rel=1e-6)
    assert "kl_sum" not in metrics and "tis_truncated_tokens" not in metrics
    loss.backward()
    assert output.logprobs.grad.abs().sum().item() == 0.0  # fully clipped: no gradient anywhere


def test_ppo_clip_unclipped_gradient_kl_and_tis():
    batch = _rl_batch(
        advantages=[0.0, 2.0, -1.0, 5.0],
        old_logprobs=[0.0, -1.5, -2.0, 0.0],
        reference_logprobs=[0.0, -1.0, -2.1, 0.0],  # equal to logp at the active slots -> k3 = 0
        rollout_logprobs=[0.0, -1.5, -2.0, 0.0],  # equal to old -> importance weights exactly 1
    )
    output = _rl_output()
    loss, metrics = ppo_clip(batch, output, clip_low=1.0, clip_high=1.0, kl_beta=0.5, tis_coef=5.0)
    ratios = torch.tensor([0.5, -0.1]).exp()
    assert loss.item() == pytest.approx(-(ratios[0] * 2.0 + ratios[1] * -1.0).item(), rel=1e-6)
    assert metrics["kl_sum"].item() == pytest.approx(0.0, abs=1e-6)
    assert metrics["clipped_tokens"].item() == 0.0 and metrics["tis_truncated_tokens"].item() == 0.0
    loss.backward()
    grad = output.logprobs.grad[0]
    assert grad[1].item() == pytest.approx(-(ratios[0] * 2.0).item(), rel=1e-6)
    assert grad[2].item() == pytest.approx((ratios[1] * 1.0).item(), rel=1e-6)


def test_ppo_clip_overflowing_ratio_keeps_loss_finite_and_gradient_zero():
    # Slot 1: logp -1.0 vs old -100.0 is a log-ratio of 99; the clipped branch wins the minimum.
    batch = _rl_batch(advantages=[0.0, 1.0, 0.0, 0.0], old_logprobs=[0.0, -100.0, -2.1, 0.0])
    output = _rl_output()
    loss, metrics = ppo_clip(batch, output, clip_low=0.2, clip_high=0.2)
    assert loss.item() == pytest.approx(-1.2)
    assert metrics["clipped_tokens"].item() == 1.0
    loss.backward()
    assert torch.isfinite(output.logprobs.grad).all()
    assert output.logprobs.grad[0, 1].item() == 0.0


def test_truncated_importance_weight_overflow_stays_finite():
    batch = _rl_batch(
        advantages=[0.0, 1.0, 1.0, 0.0],
        old_logprobs=[0.0, -1.5, -2.0, 0.0],
        rollout_logprobs=[0.0, -100.5, -2.0, 0.0],  # old - rollout = 99 at slot 1
    )
    loss, metrics = importance_sampling(batch, _rl_output(), tis_coef=2.0)
    assert math.isfinite(loss.item()) and metrics["tis_truncated_tokens"].item() == 1.0


def test_losses_on_a_filler_give_zero_with_their_metric_keys():
    batch = _rl_batch()  # no optional fields at all, as a filler built by a custom pipeline
    batch.loss_mask = torch.zeros(1, 4)
    output = _rl_output()
    loss, metrics = importance_sampling(batch, output, tis_coef=1.0)
    assert loss.item() == 0.0 and loss.requires_grad
    assert set(metrics) == {"abs_logprob_diff_sum", "tis_truncated_tokens"}
    loss, metrics = ppo_clip(batch, output, clip_low=0.2, clip_high=0.2, kl_beta=0.1, tis_coef=1.0)
    assert loss.item() == 0.0 and loss.requires_grad
    assert set(metrics) == {"clipped_tokens", "abs_logprob_diff_sum", "kl_sum", "tis_truncated_tokens"}
    assert all(value.item() == 0.0 for value in metrics.values())
    values = ModelOutput(logprobs=None, values=torch.zeros(1, 4, requires_grad=True))
    loss, metrics = value_loss(batch, values, clip=0.5)
    assert loss.item() == 0.0 and set(metrics) == {"value_abs_error_sum", "value_clipped_tokens"}


def test_ppo_clip_kl_term_value_and_requirement():
    batch = _rl_batch(
        advantages=[0.0, 0.0, 0.0, 0.0],
        old_logprobs=[0.0, -1.0, -2.1, 0.0],
        reference_logprobs=[0.0, -2.0, -2.1, 0.0],  # delta = ref - logp = -1.0 at slot 1, 0 at slot 2
    )
    loss, metrics = ppo_clip(batch, _rl_output(), clip_low=0.2, clip_high=0.2, kl_beta=2.0)
    k3 = math.exp(-1.0) - (-1.0) - 1.0
    assert loss.item() == pytest.approx(2.0 * k3, rel=1e-6)
    assert metrics["kl_sum"].item() == pytest.approx(k3, rel=1e-6)
    batch = _rl_batch(advantages=[0.0] * 4, old_logprobs=[0.0] * 4)
    with pytest.raises(ValueError, match="reference_logprobs"):
        ppo_clip(batch, _rl_output(), clip_low=0.2, clip_high=0.2, kl_beta=1.0)


def test_value_loss_plain_and_clipped():
    batch = _rl_batch()
    batch.loss_inputs = {
        "returns": torch.tensor([[0.0, 1.0, -1.0, 9.0]]),
        "old_values": torch.tensor([[0.0, 0.0, 0.0, 0.0]]),
    }
    values = torch.tensor([[5.0, 0.5, 0.5, 5.0]], requires_grad=True)
    output = ModelOutput(logprobs=None, values=values)
    loss, metrics = value_loss(batch, output)
    assert loss.item() == pytest.approx(0.5 * (0.25 + 2.25))
    assert metrics["value_abs_error_sum"].item() == pytest.approx(2.0)
    loss.backward()
    assert values.grad.tolist() == [[0.0, -0.5, 1.5, 0.0]]
    # Clip 0.2 around old_values=0: slot 1 clipped value 0.2 -> error 0.64 > 0.25 (clipped wins);
    # slot 2 clipped value 0.2 -> error 1.44 < 2.25 (unclipped wins).
    loss, metrics = value_loss(batch, output, clip=0.2)
    assert loss.item() == pytest.approx(0.5 * (0.64 + 2.25))
    assert metrics["value_clipped_tokens"].item() == 1.0
    with pytest.raises(ValueError, match="value-head engine"):
        value_loss(batch, ModelOutput(logprobs=torch.zeros(1, 4)))
    with pytest.raises(ValueError, match="returns"):
        value_loss(_rl_batch(), output)


def test_value_head_model_settings_follow_bridge():
    model = SimpleNamespace(mtp_num_layers=1, mtp_enabled=True, share_embeddings_and_output_weights=True)
    assert apply_value_head_model_settings(model) == {
        "mtp_num_layers": 1,
        "mtp_enabled": True,
        "share_embeddings_and_output_weights": True,
    }
    assert model.mtp_num_layers is None and model.mtp_enabled is False
    assert model.share_embeddings_and_output_weights is False
    assert apply_value_head_model_settings(model) == {}
    bare = SimpleNamespace(mtp_num_layers=None)  # fields the provider lacks are left alone, not created
    assert apply_value_head_model_settings(bare) == {} and not hasattr(bare, "mtp_enabled")


def test_value_head_hook_loads_then_replaces(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "megatron.bridge.engine.value_head.load_pretrained_weights",
        lambda recipe, model, optimizer, path: calls.append(("load", path, optimizer)),
    )
    monkeypatch.setattr(
        "megatron.bridge.engine.value_head.create_value_head_hook",
        lambda hidden_size, sequence_parallel: lambda model: calls.append(("replace", hidden_size, sequence_parallel)),
    )
    recipe = SimpleNamespace(model=SimpleNamespace(hidden_size=8, sequence_parallel=True))
    model = ["chunk"]
    assert make_value_head_hook(recipe, "/ckpt")(model) is model
    assert calls == [("load", "/ckpt", None), ("replace", 8, True)]
    calls.clear()
    make_value_head_hook(recipe, None)(model)
    assert calls == [("replace", 8, True)]


def test_to_model_output_by_head():
    raw = torch.tensor([[1.0, 2.0]])
    lm = to_model_output(raw)
    assert lm.values is None and lm.logprobs.tolist() == [[-1.0, -2.0]] and scored_tensor(lm) is lm.logprobs
    value = to_model_output(raw, "value")
    assert value.logprobs is None and value.values is raw and scored_tensor(value) is raw
    with pytest.raises(ValueError, match="output_head"):
        to_model_output(raw, "reward")


def test_split_loss_return_accepts_scalar_or_scalar_with_metrics():
    loss, extra = split_loss_return(torch.tensor(1.0))
    assert loss.item() == 1.0 and extra == {}
    loss, extra = split_loss_return((torch.tensor(2.0), {"n": torch.tensor(3.0)}))
    assert loss.item() == 2.0 and extra["n"].item() == 3.0
    with pytest.raises(ValueError, match="scalar tensor"):
        split_loss_return(torch.ones(2))
    with pytest.raises(ValueError, match="metrics"):
        split_loss_return((torch.tensor(1.0), {"bad": torch.ones(2)}))


def test_target_logprobs_slices_per_datum_in_cp_local_order_with_grad():
    # Two Datums packed as [A0 A1 A2 | B0 B1 | pad]; this CP rank holds positions [4, 1, 0, 5] of the pack.
    mapping = MicrobatchMapping(
        segments=[SequenceMapping(index=7, offset=0, length=3), SequenceMapping(index=2, offset=3, length=2)],
        total_tokens=6,
        cp_indices=torch.tensor([4, 1, 0, 5]),
    )
    batch = _rl_batch()
    batch.mapping = mapping
    output = ModelOutput(logprobs=torch.tensor([[-0.5, -0.25, -0.0, -0.75]], requires_grad=True))
    slices = target_logprobs(batch, output)
    assert [s.tolist() for s in slices] == [[-0.0, -0.25], [-0.5]]
    slices[1].sum().backward()
    assert output.logprobs.grad.tolist() == [[1.0, 0.0, 0.0, 0.0]]
    batch.mapping = MicrobatchMapping(segments=[], total_tokens=0, cp_indices=torch.tensor([]), filler=True)
    assert target_logprobs(batch, output) == []
    batch.mapping = object()
    with pytest.raises(ValueError, match="MicrobatchMapping"):
        target_logprobs(batch, output)


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
