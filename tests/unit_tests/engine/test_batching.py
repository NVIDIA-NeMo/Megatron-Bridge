import pytest
import torch

from megatron.bridge.engine import Datum, PreparedMicrobatch
from megatron.bridge.engine.batching import (
    PAD_TOKEN_ID,
    assign_to_dp_ranks,
    check_microbatches,
    default_postprocess,
    default_preprocess,
    filler_microbatch,
    group_into_microbatches,
    pack_sequences,
    sequence_pad_multiple,
    shard_for_cp,
    unpack_sequences,
    unshard_from_cp,
)


CPU = torch.device("cpu")


def _datum(n: int, loss_from: int = 0, **extra) -> Datum:
    ids = torch.arange(10, 10 + n + 1, dtype=torch.int64)
    mask = torch.zeros(n)
    mask[loss_from:] = 1.0
    return Datum(input_ids=ids[:-1], target_ids=ids[1:], loss_mask=mask, **extra)


def test_sequence_pad_multiple():
    assert sequence_pad_multiple(cp_size=1, tp_size=1, sequence_parallel=False) == 1
    assert sequence_pad_multiple(cp_size=8, tp_size=1, sequence_parallel=False) == 16
    assert sequence_pad_multiple(cp_size=8, tp_size=2, sequence_parallel=False) == 16
    assert sequence_pad_multiple(cp_size=2, tp_size=4, sequence_parallel=True) == 8


def test_group_into_microbatches_first_fit_with_padding():
    datums = [_datum(n) for n in [30, 10, 7, 20]]
    # padded to 16: 32, 16, 16, 32 with seq_length 48 -> [0,1] (48), [2,3] (48)
    assert group_into_microbatches(datums, seq_length=48, pad_multiple=16) == [[0, 1], [2, 3]]
    with pytest.raises(ValueError, match="above seq_length"):
        group_into_microbatches([_datum(49)], seq_length=48, pad_multiple=16)


def test_assign_to_dp_ranks_equalizes_counts():
    per_rank = assign_to_dp_ranks([[0], [1], [2], [3], [4]], dp_size=2)
    assert per_rank == [[[0], [2], [4]], [[1], [3], None]]


def test_pack_sequences_pads_and_records_mapping():
    datums = [_datum(5, loss_from=2, advantages=torch.full((5,), 0.5)), _datum(3, advantages=torch.ones(3))]
    datums[0].loss_inputs["teacher"] = torch.arange(5.0)
    mb = pack_sequences(datums, [0, 1], seq_length=64, pad_multiple=4, device=CPU)
    assert isinstance(mb, PreparedMicrobatch)
    assert mb.input_ids.shape == (1, 12)
    assert [(s.index, s.offset, s.length) for s in mb.mapping.segments] == [(0, 0, 5), (1, 8, 3)]
    assert mb.target_ids[0, 5:8].tolist() == [PAD_TOKEN_ID] * 3
    assert mb.target_ids[0, :5].tolist() == datums[0].target_ids.tolist()
    assert mb.padding_mask[0].tolist() == [False] * 5 + [True] * 3 + [False] * 3 + [True]
    assert mb.loss_mask[0].tolist() == [0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0]
    assert mb.active_mask[0].tolist() == [False, False, True, True, True, False, False, False, True, True, True, False]
    assert mb.advantages[0].tolist() == [0.5] * 5 + [0.0] * 3 + [1.0] * 3 + [0.0]
    assert mb.loss_inputs["teacher"][0].tolist() == [0, 1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0]
    assert mb.old_logprobs is None and mb.sample_inputs == [{}, {}]
    assert mb.packed_seq_params.qkv_format == "thd"
    assert mb.packed_seq_params.cu_seqlens_q.tolist() == [0, 5, 8]
    assert mb.packed_seq_params.cu_seqlens_q_padded.tolist() == [0, 8, 12]


def test_unpack_round_trip_full_and_local():
    datums = [_datum(5), _datum(3)]
    mb = shard_for_cp(pack_sequences(datums, [0, 1], seq_length=64, pad_multiple=4, device=CPU), None)
    values = torch.arange(12, dtype=torch.float32)
    full = unpack_sequences(mb, unshard_from_cp(mb, values, None), local_only=False)
    assert [(e.index, e.positions.tolist(), e.values.tolist()) for e in full] == [
        (0, [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]),
        (1, [0, 1, 2], [8, 9, 10]),
    ]
    local = unpack_sequences(mb, values, local_only=True)
    assert [(e.index, e.positions.tolist(), e.values.tolist()) for e in local] == [
        (e.index, e.positions.tolist(), e.values.tolist()) for e in full
    ]


def test_local_unpack_follows_cp_indices():
    mb = pack_sequences([_datum(6)], [0], seq_length=16, pad_multiple=2, device=CPU)
    mb.mapping.cp_indices = torch.tensor([5, 1, 4, 0])  # a CP shard holding pack positions in zigzag order
    entries = unpack_sequences(mb, torch.tensor([50.0, 10.0, 40.0, 0.0]), local_only=True)
    assert entries[0].positions.tolist() == [0, 1, 4, 5]
    assert entries[0].values.tolist() == [0.0, 10.0, 40.0, 50.0]


def test_filler_microbatch_contributes_nothing():
    mb = filler_microbatch(pad_multiple=4, device=CPU, layout="thd")
    assert mb.mapping.filler and mb.input_ids.shape == (1, 4)
    assert mb.loss_mask.sum() == 0 and mb.padding_mask.all()
    assert unpack_sequences(mb, torch.zeros(4), local_only=False) == []
    with pytest.raises(NotImplementedError, match="bshd"):
        filler_microbatch(pad_multiple=4, device=CPU, layout="bshd")


def test_default_preprocess_and_postprocess_single_rank():
    datums = [_datum(5, loss_from=2), _datum(3), _datum(2)]
    microbatches = default_preprocess(
        datums, layout="thd", seq_length=12, pad_multiple=4, dp_group=None, cp_group=None, device=CPU
    )
    assert [mb.input_ids.shape[1] for mb in microbatches] == [12, 4]  # [5->8 + 3->4], [2->4]
    logprobs = [-torch.arange(1.0, 13.0), -torch.arange(1.0, 5.0)]
    out = default_postprocess(microbatches, logprobs, gather_output=True, cp_group=None, dp_group=None)
    assert [t.tolist() for t in out.logprobs] == [[-1, -2, -3, -4, -5], [-9, -10, -11], [-1, -2]]
    assert out.sample_indices is None
    local = default_postprocess(microbatches, logprobs, gather_output=False, cp_group=None, dp_group=None)
    assert local.sample_indices == [0, 1, 2] and [p.tolist() for p in local.token_positions][2] == [0, 1]


def test_check_microbatches_rejects_contract_violations():
    mb = pack_sequences([_datum(3)], [0], seq_length=8, pad_multiple=1, device=CPU)
    with pytest.raises(ValueError, match="must be on CUDA"):
        check_microbatches([mb], layout="thd", vocab_size=100, dp_group=None)


def test_bshd_layout_is_an_explicit_placeholder():
    with pytest.raises(NotImplementedError, match="bshd"):
        default_preprocess(
            [_datum(4)], layout="bshd", seq_length=16, pad_multiple=1, dp_group=None, cp_group=None, device=CPU
        )
    with pytest.raises(ValueError, match="Unknown batch_layout"):
        default_preprocess(
            [_datum(4)], layout="x", seq_length=16, pad_multiple=1, dp_group=None, cp_group=None, device=CPU
        )
