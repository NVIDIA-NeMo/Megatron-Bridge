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
    datums[1].loss_inputs["teacher"] = torch.zeros(3)
    datums[0].loss_inputs["cls"] = torch.arange(5, dtype=torch.int64)
    datums[1].loss_inputs["cls"] = torch.full((3,), 7, dtype=torch.int64)
    mb = pack_sequences(datums, [0, 1], seq_length=64, pad_multiple=4, device=CPU)
    # Custom keys keep the caller's dtype; common fields are normalized to float32.
    assert mb.loss_inputs["cls"].dtype == torch.int64 and mb.advantages.dtype == torch.float32
    assert mb.loss_inputs["cls"][0].tolist() == [0, 1, 2, 3, 4, 0, 0, 0, 7, 7, 7, 0]
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


def test_filler_microbatch_copies_the_field_schema():
    real = Datum(
        input_ids=torch.zeros(3, dtype=torch.int64),
        target_ids=torch.zeros(3, dtype=torch.int64),
        loss_mask=torch.ones(3),
        advantages=torch.ones(3),
        loss_inputs={"returns": torch.ones(3, dtype=torch.float64)},
    )
    mb = filler_microbatch(pad_multiple=4, device=CPU, layout="thd", schema_from=real)
    assert mb.mapping.filler and not mb.active_mask.any()
    assert mb.advantages is not None and mb.advantages.abs().sum() == 0 and mb.old_logprobs is None
    assert mb.loss_inputs["returns"].dtype == torch.float64 and mb.loss_inputs["returns"].shape == mb.input_ids.shape


def test_filler_microbatch_contributes_nothing():
    mb = filler_microbatch(pad_multiple=4, device=CPU, layout="thd")
    assert mb.mapping.filler and mb.input_ids.shape == (1, 4)
    assert mb.loss_mask.sum() == 0 and mb.padding_mask.all()
    assert unpack_sequences(mb, torch.zeros(4), local_only=False) == []
    with pytest.raises(NotImplementedError, match="bshd"):
        filler_microbatch(pad_multiple=4, device=CPU, layout="bshd")


def test_default_preprocess_and_postprocess_single_rank():
    datums = [_datum(5, loss_from=2), _datum(3), _datum(2)]
    # [5->8 + 3->4], [2->4]; by default widths follow the content.
    variable = default_preprocess(
        datums, layout="thd", seq_length=12, pad_multiple=4, dp_group=None, cp_group=None, device=CPU
    )
    assert [mb.input_ids.shape[1] for mb in variable] == [12, 4]
    assert variable[1].padding_mask[0].tolist() == [False, False, True, True]
    # With pad_to_max_length every microbatch is padded to seq_length so kernels see one shape.
    microbatches = default_preprocess(
        datums,
        layout="thd",
        seq_length=12,
        pad_multiple=4,
        dp_group=None,
        cp_group=None,
        device=CPU,
        pad_to_max_length=True,
    )
    assert [mb.input_ids.shape[1] for mb in microbatches] == [12, 12]
    assert microbatches[1].padding_mask[0].tolist() == [False, False] + [True] * 10
    logprobs = [-torch.arange(1.0, 13.0), -torch.arange(1.0, 13.0)]
    out = default_postprocess(microbatches, logprobs, gather_output=True, cp_group=None, dp_group=None)
    assert [t.tolist() for t in out.logprobs] == [[-1, -2, -3, -4, -5], [-9, -10, -11], [-1, -2]]
    assert out.sample_indices is None
    local = default_postprocess(microbatches, logprobs, gather_output=False, cp_group=None, dp_group=None)
    assert local.sample_indices == [0, 1, 2] and [p.tolist() for p in local.token_positions][2] == [0, 1]


def test_check_microbatches_rejects_contract_violations():
    mb = pack_sequences([_datum(3)], [0], seq_length=8, pad_multiple=1, device=CPU)
    with pytest.raises(ValueError, match="must be on CUDA"):
        check_microbatches([mb], layout="thd", vocab_size=100, dp_group=None)


def _routed(n: int, layers: int = 3, topk: int = 2, start: int = 0) -> torch.Tensor:
    return torch.arange(start, start + n * layers * topk, dtype=torch.int16).view(n, layers, topk)


def test_pack_sequences_places_routing_by_segment_with_minus_one_padding():
    datums = [_datum(5, routed_experts=_routed(5)), _datum(3, routed_experts=_routed(3, start=100))]
    mb = pack_sequences(datums, [0, 1], seq_length=64, pad_multiple=4, device=CPU)
    assert mb.routed_experts.shape == (1, 12, 3, 2) and mb.routed_experts.dtype == torch.int64
    assert torch.equal(mb.routed_experts[0, :5], _routed(5).long())
    assert (mb.routed_experts[0, 5:8] == -1).all() and (mb.routed_experts[0, 11] == -1).all()
    assert torch.equal(mb.routed_experts[0, 8:11], _routed(3, start=100).long())
    assert "routed_experts" in mb.token_tensors()
    # The CP shard keeps the token axis in sync with every other field (identity split on one rank).
    sharded = shard_for_cp(mb, None)
    assert torch.equal(sharded.routed_experts, mb.routed_experts)
    # Fillers copy the schema as all-padding routing.
    filler = filler_microbatch(pad_multiple=4, device=CPU, layout="thd", schema_from=datums[0])
    assert filler.routed_experts.shape == (1, 4, 3, 2) and (filler.routed_experts == -1).all()
    # Without routing on the Datums the field stays unset.
    assert pack_sequences([_datum(3)], [0], seq_length=8, pad_multiple=1, device=CPU).routed_experts is None


def test_check_microbatches_validates_routing_against_the_replayable_shape():
    with_routing = pack_sequences(
        [_datum(3, routed_experts=_routed(3))], [0], seq_length=8, pad_multiple=1, device=CPU
    )
    plain = pack_sequences([_datum(3)], [0], seq_length=8, pad_multiple=1, device=CPU)
    with pytest.raises(ValueError, match="every microbatch of a call or for none"):
        check_microbatches([with_routing, plain], layout="thd", vocab_size=100, dp_group=None)
    with pytest.raises(ValueError, match="cannot replay routing"):
        check_microbatches([with_routing], layout="thd", vocab_size=100, dp_group=None, routed_shape=None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the strict check runs on CUDA tensors")
def test_check_microbatches_allows_the_sentinel_only_at_padding():
    cuda = torch.device("cuda")
    mb = pack_sequences([_datum(3, routed_experts=_routed(3))], [0], seq_length=8, pad_multiple=4, device=cuda)
    assert (mb.routed_experts[0, 3] == -1).all()  # the padded slot
    check_microbatches([mb], layout="thd", vocab_size=100, dp_group=None, routed_shape=(3, 2), num_experts=64)
    mb.routed_experts[0, 1, 0, 0] = -1  # a real token loses one routing entry
    with pytest.raises(ValueError, match="non-padding positions"):
        check_microbatches([mb], layout="thd", vocab_size=100, dp_group=None, routed_shape=(3, 2), num_experts=64)


def test_unshard_from_cp_keeps_trailing_dims():
    mb = shard_for_cp(pack_sequences([_datum(3)], [0], seq_length=8, pad_multiple=4, device=CPU), None)
    values = torch.arange(4 * 6, dtype=torch.float32).view(4, 3, 2)  # 3 tokens padded to 4
    assert torch.equal(unshard_from_cp(mb, values, None), values)


def test_bshd_layout_is_an_explicit_placeholder():
    with pytest.raises(NotImplementedError, match="bshd"):
        default_preprocess(
            [_datum(4)], layout="bshd", seq_length=16, pad_multiple=1, dp_group=None, cp_group=None, device=CPU
        )
    with pytest.raises(ValueError, match="Unknown batch_layout"):
        default_preprocess(
            [_datum(4)], layout="x", seq_length=16, pad_multiple=1, dp_group=None, cp_group=None, device=CPU
        )
