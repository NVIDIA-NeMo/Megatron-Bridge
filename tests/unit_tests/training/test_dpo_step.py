import types

import torch

import megatron.bridge.training.dpo_step as dpo_step_module
from megatron.bridge.data.batch_utils import split_batch_into_microbatches
from megatron.bridge.data.datasets.preference import preference_collate_fn
from megatron.bridge.training.dpo import DPOLossConfig
from megatron.bridge.training.dpo_step import _SEQ_DIM_KEYS, forward_step, trim_dpo_batch_padding


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


def test_trim_recovers_standalone_collate_shapes_and_content():
    """A pair collated inside a wider global batch, split and trimmed, must be tensor-identical to the
    same pair collated alone (the scorer's shape), rounded to the collate multiple for SP."""
    pad_to_multiple = 8
    short = make_pair_record(0, ctx_len=2, chosen_comp=2, rejected_comp=1)
    long = make_pair_record(1, ctx_len=9, chosen_comp=40, rejected_comp=30)
    global_batch = preference_collate_fn([short, long], pad_token_id=0, pad_seq_length_to_mult=pad_to_multiple)
    global_batch.pop("attention_mask")  # None, not a tensor — split handles tensors

    for record, microbatch in zip((short, long), split_batch_into_microbatches(global_batch, num_microbatches=2)):
        trimmed = trim_dpo_batch_padding(dict(microbatch), pad_to_multiple=pad_to_multiple)
        standalone = preference_collate_fn([record], pad_token_id=0, pad_seq_length_to_mult=pad_to_multiple)
        for key in _SEQ_DIM_KEYS:
            assert trimmed[key].shape[1] % pad_to_multiple == 0, key
            assert torch.equal(trimmed[key], standalone[key]), key
        assert torch.equal(trimmed["ref_logprob_sum"], standalone["ref_logprob_sum"])  # row keys untouched


class FakeStragglerTimer:
    def __call__(self, bdata=False):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def make_state() -> types.SimpleNamespace:
    timer = types.SimpleNamespace(start=lambda: None, stop=lambda: None)
    cfg = types.SimpleNamespace(
        dpo=DPOLossConfig(),
        dataset=types.SimpleNamespace(pad_seq_length_to_mult=1),
        rerun_state_machine=types.SimpleNamespace(check_for_nan_in_loss=False, check_for_spiky_loss=False),
    )
    return types.SimpleNamespace(timers=lambda *args, **kwargs: timer, straggler_timer=FakeStragglerTimer(), cfg=cfg)


def test_dpo_forward_step_wires_forward_and_loss_closure(monkeypatch):
    """The step forwards (tokens, position_ids, None mask, labels) and binds the batch into the loss closure."""
    batch = {
        "tokens": torch.zeros(2, 3, dtype=torch.long),
        "labels": torch.zeros(2, 3, dtype=torch.long),
        "loss_mask": torch.ones(2, 3, dtype=torch.long),
        "position_ids": torch.zeros(2, 3, dtype=torch.long),
        "pair_id": torch.tensor([0, 0]),
        "loss_multiplier": torch.ones(2),
        "ref_logprob_sum": torch.tensor([-1.0, -2.0]),
        "ref_num_tokens": torch.tensor([3, 3]),
    }
    monkeypatch.setattr(dpo_step_module, "pipeline_stage_roles", lambda model: (True, True))
    monkeypatch.setattr(dpo_step_module, "get_dpo_batch", lambda it, _mult=1: next(it))  # no .cuda() on CPU
    monkeypatch.setattr(dpo_step_module, "validate_local_loss", lambda *args, **kwargs: None)
    forward_calls = []

    def fake_model(*, input_ids, position_ids, attention_mask, labels):
        forward_calls.append((input_ids, position_ids, attention_mask, labels))
        return torch.full(labels.shape, 0.5)

    output_tensor, loss_closure = forward_step(make_state(), iter([batch]), fake_model)

    assert forward_calls == [(batch["tokens"], batch["position_ids"], None, batch["labels"])]
    loss, num_live_pairs, metrics = loss_closure(output_tensor)
    assert torch.isfinite(loss)
    assert num_live_pairs.item() == 1
    assert "dpo loss" in metrics
