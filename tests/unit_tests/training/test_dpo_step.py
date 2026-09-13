import types

import pytest
import torch

from megatron.bridge.data.batch_utils import split_batch_into_microbatches
from megatron.bridge.data.datasets.preference import preference_collate_fn
from megatron.bridge.training.dpo import DPOLossConfig
from megatron.bridge.training.dpo_step import (
    _SEQ_DIM_KEYS,
    DPO_BATCH_KEYS,
    dpo_forward_step,
    trim_dpo_batch_padding,
    validate_dpo_batch,
)


def make_batch() -> dict[str, torch.Tensor]:
    return {
        "tokens": torch.zeros(2, 3, dtype=torch.long),
        "labels": torch.zeros(2, 3, dtype=torch.long),
        "loss_mask": torch.ones(2, 3, dtype=torch.long),
        "position_ids": torch.zeros(2, 3, dtype=torch.long),
        "pair_id": torch.tensor([0, 0]),
        "loss_multiplier": torch.ones(2),
        "ref_logprob_sum": torch.tensor([-1.0, -2.0]),
        "ref_num_tokens": torch.tensor([3, 3]),
    }


class FakeRerunStateMachine:
    def __init__(self):
        self.validated = []

    def validate_result(self, *, result, rejection_func, message, tolerance, fatal):
        self.validated.append(message)
        if rejection_func(result):
            raise AssertionError(message)


@pytest.fixture
def fake_rerun_state_machine(monkeypatch):
    machine = FakeRerunStateMachine()
    monkeypatch.setattr("megatron.bridge.training.losses.get_rerun_state_machine", lambda: machine)
    return machine


@pytest.mark.parametrize("missing_key", ["ref_logprob_sum", "loss_multiplier", "pair_id"])
def test_validate_dpo_batch_rejects_missing_or_none_key(missing_key):
    batch = make_batch()
    del batch[missing_key]
    with pytest.raises(ValueError, match=missing_key):
        validate_dpo_batch(batch)
    batch = make_batch()
    batch[missing_key] = None  # get_batch-style key dropping
    with pytest.raises(ValueError, match=missing_key):
        validate_dpo_batch(batch)


def make_pair_record(pair_id: int, ctx_len: int, chosen_comp: int, rejected_comp: int) -> dict:
    return {
        "pair_id": pair_id,
        "chosen_input_ids": list(range(10, 10 + ctx_len + chosen_comp)),
        "chosen_context_len": ctx_len,
        "rejected_input_ids": list(range(50, 50 + ctx_len + rejected_comp)),
        "rejected_context_len": ctx_len,
        "ref_chosen_logprob_sum": -1.0,
        "ref_chosen_num_tokens": chosen_comp,
        "ref_rejected_logprob_sum": -2.0,
        "ref_rejected_num_tokens": rejected_comp,
    }


def make_stub_record(pair_id: int) -> dict:
    """A zero-loss stub: two-token sides, loss_multiplier 0, live loss_mask only at column 0."""
    return {
        "pair_id": pair_id,
        "chosen_input_ids": [0, 0],
        "chosen_context_len": 1,
        "rejected_input_ids": [0, 0],
        "rejected_context_len": 1,
        "loss_multiplier": 0.0,
        "ref_chosen_logprob_sum": 0.0,
        "ref_chosen_num_tokens": 0,
        "ref_rejected_logprob_sum": 0.0,
        "ref_rejected_num_tokens": 0,
    }


def split_global_batch(records: list[dict], pad_to_multiple: int) -> list[dict[str, torch.Tensor]]:
    global_batch = preference_collate_fn(records, pad_token_id=0, pad_seq_length_to_mult=pad_to_multiple)
    global_batch.pop("attention_mask")  # None, not a tensor — split handles tensors
    return split_batch_into_microbatches(global_batch, num_microbatches=len(records))


@pytest.mark.parametrize("pad_to_multiple", [1, 8])
def test_trim_recovers_standalone_collate_shapes_and_content(pad_to_multiple):
    """A pair collated inside a wider global batch, split and trimmed, must be tensor-identical
    to the same pair collated alone (the mbs-1 scorer's shape), at the collate's multiple."""
    short = make_pair_record(0, ctx_len=2, chosen_comp=2, rejected_comp=1)
    long = make_pair_record(1, ctx_len=9, chosen_comp=40, rejected_comp=30)

    for record, microbatch in zip((short, long), split_global_batch([short, long], pad_to_multiple)):
        trimmed = trim_dpo_batch_padding(dict(microbatch), pad_to_multiple=pad_to_multiple)
        standalone = preference_collate_fn([record], pad_token_id=0, pad_seq_length_to_mult=pad_to_multiple)
        for key in _SEQ_DIM_KEYS:
            assert trimmed[key].shape[1] % pad_to_multiple == 0, key
            assert torch.equal(trimmed[key], standalone[key]), key
        # Row-aligned keys are untouched by the trim.
        assert torch.equal(trimmed["pair_id"], standalone["pair_id"])
        assert torch.equal(trimmed["ref_logprob_sum"], standalone["ref_logprob_sum"])


@pytest.mark.parametrize(("pad_to_multiple", "expected_width"), [(1, 2), (8, 8)])
def test_trim_floors_stub_pair_microbatch(pad_to_multiple, expected_width):
    """A stub-pair microbatch trims to width 2, not 1 (TE's fused-attention backward rejects
    seq length 1), rounded up to the collate multiple so SP's TP-divides-seq assert holds."""
    stub = make_stub_record(0)
    wide = make_pair_record(1, ctx_len=9, chosen_comp=40, rejected_comp=30)
    stub_microbatch, _ = split_global_batch([stub, wide], pad_to_multiple)

    assert bool(stub_microbatch["loss_mask"].any()), "stub rows carry a live mask at column 0"
    trimmed = trim_dpo_batch_padding(dict(stub_microbatch), pad_to_multiple=pad_to_multiple)
    for key in _SEQ_DIM_KEYS:
        assert trimmed[key].shape[1] == expected_width, key


def test_trim_keeps_all_stub_microbatch_unchanged():
    batch = make_batch()
    batch["loss_mask"] = torch.zeros(2, 3, dtype=torch.long)
    trimmed = trim_dpo_batch_padding(dict(batch))
    for key in _SEQ_DIM_KEYS:
        assert trimmed[key].shape == batch[key].shape


class FakeTimer:
    def start(self):
        pass

    def stop(self):
        pass


class FakeStragglerTimer:
    def __call__(self, bdata=False):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class FakeState:
    def __init__(self):
        self.timers = lambda *args, **kwargs: FakeTimer()
        self.straggler_timer = FakeStragglerTimer()
        self.cfg = types.SimpleNamespace(
            dpo=DPOLossConfig(),
            dataset=types.SimpleNamespace(pad_seq_length_to_mult=1),
            rerun_state_machine=types.SimpleNamespace(check_for_nan_in_loss=True, check_for_spiky_loss=False),
        )


def test_dpo_forward_step_wires_forward_and_loss_closure(monkeypatch, fake_rerun_state_machine):
    """The step must forward (tokens, position_ids, None mask, labels) and bind the
    full batch plus the rerun flags into the loss closure."""
    import megatron.bridge.training.dpo_step as dpo_step_module

    batch = make_batch()
    monkeypatch.setattr(dpo_step_module, "pipeline_stage_roles", lambda model: (True, True))
    monkeypatch.setattr(dpo_step_module, "get_dpo_batch", lambda it, _mult=1: validate_dpo_batch(next(it)))

    forward_calls = []

    def fake_model(*, input_ids, position_ids, attention_mask, labels):
        forward_calls.append((input_ids, position_ids, attention_mask, labels))
        return torch.full(labels.shape, 0.5)

    output_tensor, loss_closure = dpo_forward_step(FakeState(), iter([batch]), fake_model)

    assert len(forward_calls) == 1
    input_ids, position_ids, attention_mask, labels = forward_calls[0]
    assert input_ids is batch["tokens"]
    assert position_ids is batch["position_ids"]
    assert attention_mask is None
    assert labels is batch["labels"]

    loss, num_live_pairs, metrics = loss_closure(output_tensor)
    assert torch.isfinite(loss)
    assert num_live_pairs.item() == 1
    assert "dpo loss" in metrics
    assert set(DPO_BATCH_KEYS) <= set(batch)
    # check_for_nan_in_loss=True, check_for_spiky_loss=False per FakeState.cfg
    assert fake_rerun_state_machine.validated == [
        "found NaN in local forward loss calculation",
        "found Inf in local forward loss calculation",
    ]


@pytest.mark.parametrize("roles", [(True, False), (False, True)], ids=["first-stage", "last-stage"])
def test_dpo_forward_step_fetches_the_batch_on_the_edge_pipeline_stages(monkeypatch, roles):
    """The first stage needs the tokens, the last stage needs the labels and the reference sums."""
    import megatron.bridge.training.dpo_step as dpo_step_module

    batch = make_batch()
    monkeypatch.setattr(dpo_step_module, "pipeline_stage_roles", lambda model: roles)
    monkeypatch.setattr(dpo_step_module, "get_dpo_batch", lambda it, _mult=1: validate_dpo_batch(next(it)))
    iterator = iter([batch])

    def fake_model(*, input_ids, position_ids, attention_mask, labels):
        assert input_ids is batch["tokens"]
        return torch.full(labels.shape, 0.5)

    dpo_forward_step(FakeState(), iterator, fake_model)
    assert next(iterator, None) is None  # the one batch was consumed


def test_dpo_forward_step_skips_the_batch_on_a_middle_pipeline_stage(monkeypatch):
    """A middle stage forwards the previous stage's activations and never touches the iterator."""
    import megatron.bridge.training.dpo_step as dpo_step_module

    monkeypatch.setattr(dpo_step_module, "pipeline_stage_roles", lambda model: (False, False))

    def fail_if_fetched(*args, **kwargs):
        raise AssertionError("a middle stage must not fetch a batch")

    monkeypatch.setattr(dpo_step_module, "get_dpo_batch", fail_if_fetched)
    forward_calls = []

    def fake_model(*, input_ids, position_ids, attention_mask, labels):
        forward_calls.append((input_ids, position_ids, attention_mask, labels))
        return torch.zeros(2, 3, 8)

    output_tensor, loss_closure = dpo_forward_step(FakeState(), iter([make_batch()]), fake_model)
    assert forward_calls == [(None, None, None, None)]
    assert output_tensor.shape == (2, 3, 8)
    with pytest.raises(RuntimeError, match="last stage"):
        loss_closure(output_tensor)
