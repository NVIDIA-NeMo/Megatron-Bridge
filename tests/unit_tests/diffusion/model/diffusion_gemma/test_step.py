from unittest.mock import MagicMock

import pytest
import torch

from megatron.bridge.diffusion.models.diffusion_gemma.step import diffusion_gemma_loss, prepare_batch


pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def rerun_machine(monkeypatch):
    machine = MagicMock()
    machine.spiky_loss_factor = 10.0
    monkeypatch.setattr("megatron.bridge.training.losses.get_rerun_state_machine", lambda: machine)
    return machine


def shifted_batch():
    # Full clean row [1,2,3,4,5,9]: nonempty prompt [1,2], final response [3,4,5,9].
    return {
        "tokens": torch.tensor([[1, 2, 3, 4, 5]]),
        "labels": torch.tensor([[2, 3, 4, 5, 9]]),
        "loss_mask": torch.tensor([[0, 1, 1, 1, 1]]),
        "token_count": [6],
        "context_lengths": torch.tensor([2]),
    }


def test_reconstructs_shifted_final_eos_and_fills_partial_canvas():
    torch.manual_seed(4)
    batch = prepare_batch(shifted_batch(), canvas_length=3, eos_token_id=9, vocab_size=10)
    assert batch.input_ids.tolist() == [[1, 2, 3, 4, 5, 9, 9, 9]]
    assert batch.encoder_valid_mask.all()
    assert batch.targets.shape == (1, 3)
    assert batch.prefix_lengths.item() in (2, 5)
    assert torch.equal(batch.targets, batch.input_ids.gather(1, batch.canvas_positions))
    assert batch.encoder_valid_mask[:, 1:].sum() == 7


@pytest.mark.parametrize(
    "change,match",
    [
        ({"token_count": [7]}, "truncated"),
        ({"context_lengths": torch.tensor([0])}, "nonempty prompt"),
        ({"loss_mask": torch.tensor([[0, 1, 0, 1, 1]])}, "contiguous"),
        ({"loss_mask": torch.tensor([[1, 1, 1, 1, 1]])}, "contiguous"),
        ({"cu_seqlens": torch.tensor([0, 5])}, "packing"),
    ],
)
def test_rejects_invalid_shifted_rows(change, match):
    batch = shifted_batch() | change
    with pytest.raises(ValueError, match=match):
        prepare_batch(batch, canvas_length=3, eos_token_id=9, vocab_size=10)


def test_noise_replays_from_default_rng_state():
    state = torch.get_rng_state()
    first = prepare_batch(shifted_batch(), canvas_length=3, eos_token_id=9, vocab_size=10)
    torch.set_rng_state(state)
    second = prepare_batch(shifted_batch(), canvas_length=3, eos_token_id=9, vocab_size=10)
    assert torch.equal(first.noisy_tokens, second.noisy_tokens)
    assert torch.equal(first.prefix_lengths, second.prefix_lengths)
    assert torch.equal(first.self_conditioning_mask, second.self_conditioning_mask)


def test_loss_normalizes_objectives_separately_and_returns_legacy_pair():
    dlm = torch.tensor([[2.0, 4.0, 6.0]], requires_grad=True)
    ar = torch.tensor([[3.0, 5.0, 7.0]], requires_grad=True)
    loss, metrics = diffusion_gemma_loss(torch.tensor([[True, True, False]]), 0.5, (dlm, ar))
    assert loss.item() == pytest.approx(6.0)  # 12/3 + 0.5*8/2
    loss.backward()
    assert torch.allclose(dlm.grad, torch.full_like(dlm, 1 / 3))
    assert torch.allclose(ar.grad, torch.tensor([[0.25, 0.25, 0.0]]))
    assert metrics["diffusion loss"].tolist() == [12.0, 3.0]
    assert metrics["encoder loss"].tolist() == [8.0, 2.0]


def test_loss_uses_global_dp_counts_for_unequal_rows(monkeypatch):
    from megatron.bridge.diffusion.models.diffusion_gemma import step

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(step.parallel_state, "get_data_parallel_group", lambda: "dp")
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)
    # Local canvas count 3 plus peer count 3; local valid AR 2 plus peer AR 5.
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda counts, group: counts.add_(counts.new_tensor([3, 5])))
    dlm = torch.tensor([[2.0, 4.0, 6.0]], requires_grad=True)
    ar = torch.tensor([[3.0, 5.0, 7.0]], requires_grad=True)
    loss, metrics = diffusion_gemma_loss(torch.tensor([[True, True, False]]), 0.5, (dlm, ar))
    assert loss.item() == pytest.approx(2 * (12 / 6 + 0.5 * 8 / 7))
    loss.backward()
    assert torch.allclose(dlm.grad, torch.full_like(dlm, 2 / 6))
    assert torch.allclose(ar.grad, torch.tensor([[1 / 7, 1 / 7, 0.0]]))
    assert metrics["encoder loss"].tolist() == [8.0, 2.0]


def test_non_batch_padding_is_not_valid_encoder_context():
    first = shifted_batch()
    mixed = {
        "tokens": torch.cat((first["tokens"], torch.tensor([[1, 2, 9, 9, 9]]))),
        "labels": torch.cat((first["labels"], torch.tensor([[2, 9, 9, 9, 9]]))),
        "loss_mask": torch.cat((first["loss_mask"], torch.tensor([[0, 1, 0, 0, 0]]))),
        "token_count": [6, 3],
        "context_lengths": torch.tensor([2, 2]),
    }
    batch = prepare_batch(mixed, canvas_length=3, eos_token_id=9, vocab_size=10, self_conditioning_probability=0)
    assert batch.encoder_valid_mask.sum(dim=1).tolist() == [8, 5]
    assert not batch.self_conditioning_mask.any()
    assert batch.targets[1].tolist() == [9, 9, 9]
    assert (batch.noisy_tokens >= 0).all() and (batch.noisy_tokens < 10).all()


def test_step_passes_native_forward_contract_and_scores_all_clean_positions():
    from types import SimpleNamespace

    import torch.nn.functional as F

    from megatron.bridge.diffusion.models.diffusion_gemma.step import DiffusionGemmaStep

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.arange(10, dtype=torch.float32))
            self.config = SimpleNamespace(vocab_size=10)

        def forward(self, input_ids, position_ids, **kwargs):
            assert position_ids.shape == input_ids.shape
            assert kwargs["encoder_valid_mask"].all()
            assert kwargs["noisy_tokens"].shape == kwargs["canvas_positions"].shape == (1, 3)
            assert kwargs["self_conditioning_mask"].shape == (1,)
            return self.logits.expand(*input_ids.shape, 10), self.logits.expand(1, 3, 10)

        def compute_language_model_loss(self, labels, logits):
            return F.cross_entropy(logits.transpose(0, 1).flatten(0, 1), labels.flatten(), reduction="none").view_as(
                labels
            )

    cfg = SimpleNamespace(
        model=SimpleNamespace(
            overlap_moe_expert_parallel_comm=False, calculate_per_token_loss=False, recompute_granularity=None
        ),
        dataset=SimpleNamespace(),
        peft=None,
        rerun_state_machine=SimpleNamespace(check_for_nan_in_loss=False, check_for_spiky_loss=True),
    )
    state = SimpleNamespace(cfg=cfg, tokenizer=SimpleNamespace(eod=9))
    model = Model()
    outputs, loss_fn = DiffusionGemmaStep(canvas_length=3)(state, iter([shifted_batch()]), model)
    assert loss_fn.keywords == {"check_for_nan_in_loss": False, "check_for_spiky_loss": True}
    loss, metrics = loss_fn(outputs)
    assert outputs[0].shape == (1, 3)
    assert outputs[1].shape == (1, 7)
    assert metrics["diffusion loss"][1] == 3
    assert metrics["encoder loss"][1] == 7
    loss.backward()
    assert model.logits.grad is not None and torch.isfinite(model.logits.grad).all()


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
@pytest.mark.parametrize("field", ["noise_epsilon", "self_conditioning_probability", "encoder_loss_weight"])
def test_constructor_rejects_nonfinite_objective_settings(field, value):
    from megatron.bridge.diffusion.models.diffusion_gemma.step import DiffusionGemmaStep

    with pytest.raises(ValueError):
        DiffusionGemmaStep(**{field: value})


@pytest.mark.parametrize("canvas_length", [0, -1, float("nan"), 1.5])
def test_constructor_rejects_invalid_canvas_length(canvas_length):
    from megatron.bridge.diffusion.models.diffusion_gemma.step import DiffusionGemmaStep

    with pytest.raises(ValueError):
        DiffusionGemmaStep(canvas_length=canvas_length)


def test_loss_validates_final_objective_with_existing_rerun_checks(rerun_machine):
    outputs = (torch.tensor([[2.0, 4.0]]), torch.tensor([[3.0, 5.0]]))
    diffusion_gemma_loss(torch.tensor([[True, True]]), 1.0, outputs, check_for_spiky_loss=True)
    assert rerun_machine.validate_result.call_count == 3
    calls = rerun_machine.validate_result.call_args_list
    assert [call.kwargs["fatal"] for call in calls] == [True, True, False]
    assert [call.kwargs["result"].item() for call in calls] == [7.0, 7.0, 7.0]
    rerun_machine.reset_mock()
    diffusion_gemma_loss(torch.tensor([[True, True]]), 1.0, outputs, check_for_nan_in_loss=False)
    rerun_machine.validate_result.assert_not_called()


@pytest.mark.parametrize("invalid,check", [(float("nan"), 0), (float("inf"), 1)])
def test_nonfinite_final_loss_reaches_fatal_rerun_validation(rerun_machine, invalid, check):
    outputs = (torch.tensor([[invalid, 4.0]]), torch.tensor([[3.0, 5.0]]))
    diffusion_gemma_loss(torch.tensor([[True, True]]), 1.0, outputs)
    call = rerun_machine.validate_result.call_args_list[check]
    assert call.kwargs["fatal"]
    assert call.kwargs["rejection_func"](call.kwargs["result"])
