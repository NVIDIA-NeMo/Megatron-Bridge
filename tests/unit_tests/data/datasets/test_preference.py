from functools import partial

import pytest
import torch
from torch.utils.data import Dataset

from megatron.bridge.data.batch_utils import split_batch_into_microbatches
from megatron.bridge.data.datasets.preference import (
    build_preference_data_loader,
    load_ref_logprobs,
    preference_collate_fn,
    write_ref_logprobs,
)


PAD_ID = 0


class RecordsDataset(Dataset):
    """Serves ready-made pair records, so collate and sampler tests skip tokenization."""

    def __init__(self, records: list[dict]) -> None:
        self.records = records
        self.collate_fn = partial(preference_collate_fn, pad_token_id=PAD_ID)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        return dict(self.records[idx])


def make_records(num_pairs: int) -> list[dict]:
    """Token ids encode (pair, side): chosen 1000+p/2000+p (ctx/completion), rejected 3000+p/4000+p;
    ref sums encode the pair: p+0.25 (chosen) / p+0.75 (rejected)."""
    return [
        {
            "pair_id": p,
            "chosen_input_ids": [1000 + p] * (2 + p % 3) + [2000 + p] * (1 + p % 4),
            "chosen_context_len": 2 + p % 3,
            "rejected_input_ids": [3000 + p] * (2 + (p + 1) % 3) + [4000 + p] * (2 + (p + 2) % 3),
            "rejected_context_len": 2 + (p + 1) % 3,
            "ref_chosen_logprob_sum": p + 0.25,
            "ref_chosen_num_tokens": 1 + p % 4,
            "ref_rejected_logprob_sum": p + 0.75,
            "ref_rejected_num_tokens": 2 + (p + 2) % 3,
        }
        for p in range(num_pairs)
    ]


def test_loader_yields_whole_adjacent_pairs_and_covers_each_pair_once_across_ranks():
    """Every microbatch on every DP rank holds whole (chosen@even, rejected@odd) pairs with their
    reference columns row-aligned, and one epoch covers each pair exactly once with no overlap."""
    num_pairs, dp_size, gbs_rows, mbs_rows = 16, 2, 8, 2
    seen: dict[int, list[int]] = {}
    for rank in range(dp_size):
        loader = build_preference_data_loader(
            dataset=RecordsDataset(make_records(num_pairs)),
            micro_batch_size=mbs_rows,
            global_batch_size=gbs_rows,
            data_parallel_rank=rank,
            data_parallel_size=dp_size,
            consumed_samples=0,
            pin_memory=False,  # the sampler shuffles by default; adjacency must hold under shuffle
        )
        for step, global_batch in enumerate(loader):
            for mb in split_batch_into_microbatches(global_batch, gbs_rows // (mbs_rows * dp_size)):
                pair_id = mb["pair_id"]
                assert torch.equal(pair_id[::2], pair_id[1::2]), f"pair split across rows: {pair_id.tolist()}"
                assert (mb["tokens"][::2, 0] < 3000).all() and (mb["tokens"][1::2, 0] >= 3000).all()
                assert torch.equal(mb["ref_logprob_sum"][::2], pair_id[::2].float() + 0.25)
                assert torch.equal(mb["ref_logprob_sum"][1::2], pair_id[1::2].float() + 0.75)
                seen.setdefault(step, []).extend(pair_id[::2].tolist())

    assert len(seen) == num_pairs // (gbs_rows // 2)
    assert sorted(p for pairs in seen.values() for p in pairs) == list(range(num_pairs))
    assert all(len(pairs) == len(set(pairs)) for pairs in seen.values()), "ranks must not overlap within a step"


def test_collate_shift_mask_and_padding():
    """Hand-checked shift/mask/padding for one pair, incl. TP/SP divisibility."""
    record = {
        "pair_id": 7,
        "chosen_input_ids": [11, 12, 13, 21, 22],  # ctx=[11,12,13], completion=[21,22]
        "chosen_context_len": 3,
        "rejected_input_ids": [11, 12, 13, 31],  # ctx same, completion=[31]
        "rejected_context_len": 3,
        "ref_chosen_logprob_sum": -1.0,
        "ref_chosen_num_tokens": 2,
        "ref_rejected_logprob_sum": -2.0,
        "ref_rejected_num_tokens": 1,
    }
    out = preference_collate_fn([record], pad_token_id=PAD_ID, pad_seq_length_to_mult=8)

    assert out["tokens"].shape == (2, 8)  # max real length 4, ceiled to 8
    assert out["tokens"][0].tolist() == [11, 12, 13, 21, 0, 0, 0, 0]
    assert out["labels"][0].tolist() == [12, 13, 21, 22, 0, 0, 0, 0]
    # completion labels start at ctx_len-1=2: labels [21, 22] are trained on.
    assert out["loss_mask"][0].tolist() == [0, 0, 1, 1, 0, 0, 0, 0]
    assert out["tokens"][1].tolist() == [11, 12, 13, 0, 0, 0, 0, 0]
    assert out["labels"][1].tolist() == [12, 13, 31, 0, 0, 0, 0, 0]
    assert out["loss_mask"][1].tolist() == [0, 0, 1, 0, 0, 0, 0, 0]
    assert out["attention_mask"] is None


def test_ref_logprob_artifact_round_trips_and_must_cover_every_pair(tmp_path):
    rows = [
        {
            "pair_id": p,
            "ref_chosen_logprob_sum": p + 0.25,
            "ref_chosen_num_tokens": 2,
            "ref_rejected_logprob_sum": p + 0.75,
            "ref_rejected_num_tokens": 3,
        }
        for p in range(4)
    ]
    write_ref_logprobs(rows, str(tmp_path))

    mapping = load_ref_logprobs(str(tmp_path), expected_num_pairs=4)
    assert sorted(mapping) == [0, 1, 2, 3]
    assert mapping[2]["ref_chosen_logprob_sum"] == 2.25
    # A size mismatch means the artifact was scored on other rows: training would anchor to the wrong values.
    with pytest.raises(ValueError, match="cover pair_id"):
        load_ref_logprobs(str(tmp_path), expected_num_pairs=5)
