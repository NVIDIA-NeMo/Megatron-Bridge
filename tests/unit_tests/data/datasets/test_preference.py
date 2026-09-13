import json
from dataclasses import asdict
from functools import partial

import pytest
import torch
from torch.utils.data import Dataset

from megatron.bridge.data.batch_utils import split_batch_into_microbatches
from megatron.bridge.data.datasets.preference import (
    REF_LOGPROB_COLUMNS,
    ScoringFingerprint,
    build_preference_data_loader,
    load_ref_logprobs,
    pack_pairs_by_token_budget,
    pair_token_lengths,
    preference_collate_fn,
    ref_logprobs_from_rows,
    validate_scoring_metadata,
    write_ref_logprobs,
    write_scoring_metadata,
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
    """Synthetic records; token ids encode (pair, side): chosen 1000+p/2000+p (ctx/completion),
    rejected 3000+p/4000+p, ref sums p+0.25 (chosen) / p+0.75 (rejected)."""
    records = []
    for p in range(num_pairs):
        ctx_c, comp_c = 2 + p % 3, 1 + p % 4
        ctx_r, comp_r = 2 + (p + 1) % 3, 2 + (p + 2) % 3
        records.append(
            {
                "pair_id": p,
                "chosen_input_ids": [1000 + p] * ctx_c + [2000 + p] * comp_c,
                "chosen_context_len": ctx_c,
                "rejected_input_ids": [3000 + p] * ctx_r + [4000 + p] * comp_r,
                "rejected_context_len": ctx_r,
                "ref_chosen_logprob_sum": p + 0.25,
                "ref_chosen_num_tokens": comp_c,
                "ref_rejected_logprob_sum": p + 0.75,
                "ref_rejected_num_tokens": comp_r,
            }
        )
    return records


def iter_rank_microbatches(records, dp_size, gbs_rows, mbs_rows, consumed_rows=0, shuffle=True):
    """Yield (rank, step, microbatch) through the real loader + microbatch split."""
    num_microbatches = gbs_rows // (mbs_rows * dp_size)
    for rank in range(dp_size):
        loader = build_preference_data_loader(
            dataset=RecordsDataset(records),
            micro_batch_size=mbs_rows,
            global_batch_size=gbs_rows,
            data_parallel_rank=rank,
            data_parallel_size=dp_size,
            consumed_samples=consumed_rows,
            # sampler shuffles by default; adjacency must hold under shuffle
            pin_memory=False,
            shuffle=shuffle,
        )
        for step, global_batch in enumerate(loader):
            for microbatch in split_batch_into_microbatches(global_batch, num_microbatches):
                yield rank, step, microbatch


@pytest.mark.parametrize("dp_size", [1, 2, 4])
@pytest.mark.parametrize("mbs_rows", [2, 4])
def test_pair_adjacency_and_epoch_coverage(dp_size, mbs_rows):
    """Every microbatch on every rank holds whole, adjacent pairs; one epoch covers
    each pair exactly once with no overlap across ranks."""
    num_pairs, gbs_rows = 16, 8
    if gbs_rows % (mbs_rows * dp_size) != 0:
        pytest.skip("incompatible grid point")

    seen: dict[tuple[int, int], list[int]] = {}
    for rank, step, mb in iter_rank_microbatches(make_records(num_pairs), dp_size, gbs_rows, mbs_rows):
        assert mb["tokens"].shape[0] == mbs_rows
        pair_id = mb["pair_id"]
        # THE invariant: even row and its odd neighbor belong to the same pair.
        assert torch.equal(pair_id[::2], pair_id[1::2]), f"pair split across rows: {pair_id.tolist()}"
        # chosen@even / rejected@odd: token ids encode the side.
        assert (mb["tokens"][::2, 0] < 3000).all(), "even rows must be chosen"
        assert (mb["tokens"][1::2, 0] >= 3000).all(), "odd rows must be rejected"
        seen.setdefault((step, rank), []).extend(pair_id[::2].tolist())

    steps = {step for step, _ in seen}
    assert len(steps) == num_pairs // (gbs_rows // 2)
    all_pairs = [p for pairs in seen.values() for p in pairs]
    assert sorted(all_pairs) == list(range(num_pairs)), "epoch must cover every pair exactly once"
    for step in steps:
        step_pairs = [p for (s, r), pairs in seen.items() if s == step for p in pairs]
        assert len(step_pairs) == len(set(step_pairs)), "ranks must not overlap within a step"


def test_seed_reaches_the_batch_sampler():
    """Every DP rank must shuffle with the config seed, not its rank-local torch seed."""
    loader = build_preference_data_loader(
        dataset=RecordsDataset(make_records(8)),
        micro_batch_size=2,
        global_batch_size=4,
        data_parallel_rank=0,
        data_parallel_size=1,
        consumed_samples=0,
        pin_memory=False,
        seed=4321,
    )
    assert loader.batch_sampler.seed == 4321


def test_shuffle_false_consumes_pairs_in_source_row_order():
    """shuffle=False must yield pair ids exactly in dataset order — the contract
    step-synchronized parity runs against an external trainer rely on."""
    num_pairs, gbs_rows = 16, 8
    ordered = [
        pid
        for _, _, mb in iter_rank_microbatches(
            make_records(num_pairs), dp_size=1, gbs_rows=gbs_rows, mbs_rows=2, shuffle=False
        )
        for pid in mb["pair_id"][::2].tolist()
    ]
    assert ordered == list(range(num_pairs))


def test_ref_columns_ride_in_lockstep_with_rows():
    """Aux tensors must stay row-aligned through collate + microbatch split."""
    for _, _, mb in iter_rank_microbatches(make_records(8), dp_size=2, gbs_rows=4, mbs_rows=2):
        pair_id = mb["pair_id"].float()
        assert torch.equal(mb["ref_logprob_sum"][::2], pair_id[::2] + 0.25)
        assert torch.equal(mb["ref_logprob_sum"][1::2], pair_id[1::2] + 0.75)
        assert (mb["ref_num_tokens"] > 0).all()


def test_resume_matches_uninterrupted_run():
    """Rebuilding the loader with row-denominated consumed_samples continues
    the exact pair sequence (exercises the rows→pairs //2 translation)."""
    records, dp_size, gbs_rows, mbs_rows = make_records(24), 2, 8, 2

    def pair_sequence(consumed_rows):
        return [
            (rank, step, mb["pair_id"][::2].tolist())
            for rank, step, mb in iter_rank_microbatches(records, dp_size, gbs_rows, mbs_rows, consumed_rows)
        ]

    full = pair_sequence(consumed_rows=0)
    resumed = pair_sequence(consumed_rows=2 * gbs_rows)  # 2 completed steps
    steps_done = 2
    remaining = [(rank, step - steps_done, pairs) for rank, step, pairs in full if step >= steps_done]
    assert resumed == remaining


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
    assert out["position_ids"].shape == (2, 8)
    assert out["attention_mask"] is None


def test_collate_tensors_are_pinnable():
    """No collated tensor may alias memory across rows (stride 0): pin_memory
    allocates with identical strides and copy_s, which rejects overlapping writes —
    position_ids via bare expand() crashed every pin_memory=True consumer."""
    batch = preference_collate_fn(make_records(3), pad_token_id=PAD_ID)
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            assert 0 not in value.stride(), f"{key} aliases memory across rows: strides {value.stride()}"
            # The exact copy pin_memory performs must succeed.
            torch.empty_strided(value.shape, value.stride(), dtype=value.dtype).copy_(value)


def test_loader_rejects_odd_row_batch_sizes():
    common = dict(
        dataset=RecordsDataset(make_records(8)),
        data_parallel_rank=0,
        data_parallel_size=1,
    )
    with pytest.raises(ValueError, match="row-denominated and must be even"):
        build_preference_data_loader(micro_batch_size=3, global_batch_size=6, consumed_samples=0, **common)
    with pytest.raises(ValueError, match="row-denominated and must be even"):
        build_preference_data_loader(micro_batch_size=2, global_batch_size=5, consumed_samples=0, **common)
    with pytest.raises(ValueError, match="must be divisible"):
        build_preference_data_loader(micro_batch_size=4, global_batch_size=10, consumed_samples=0, **common)
    with pytest.raises(ValueError, match="consumed_samples must be even"):
        build_preference_data_loader(micro_batch_size=2, global_batch_size=4, consumed_samples=3, **common)


def test_collate_rejects_context_only_sequences():
    record = make_records(1)[0]
    record["chosen_context_len"] = len(record["chosen_input_ids"])  # no completion tokens
    with pytest.raises(ValueError, match="at least one completion"):
        preference_collate_fn([record], pad_token_id=PAD_ID)


def make_ref_rows(num_pairs: int) -> list[dict]:
    """Scorer-output rows; values encode the pair id so mixups are detectable."""
    return [
        {
            "pair_id": p,
            "ref_chosen_logprob_sum": p + 0.25,
            "ref_chosen_num_tokens": 2,
            "ref_rejected_logprob_sum": p + 0.75,
            "ref_rejected_num_tokens": 3,
        }
        for p in range(num_pairs)
    ]


def test_ref_logprobs_from_rows_keys_by_pair_id():
    mapping = ref_logprobs_from_rows(make_ref_rows(4), expected_num_pairs=4)
    assert sorted(mapping) == [0, 1, 2, 3]
    for p, ref in mapping.items():
        assert set(ref) == set(REF_LOGPROB_COLUMNS)
        assert ref["ref_chosen_logprob_sum"] == p + 0.25
        assert ref["ref_rejected_logprob_sum"] == p + 0.75


def test_ref_logprobs_from_rows_rejects_bad_artifacts():
    rows = make_ref_rows(4)

    with pytest.raises(ValueError, match="missing columns"):
        ref_logprobs_from_rows([{k: v for k, v in rows[0].items() if k != "ref_chosen_num_tokens"}], 1)
    with pytest.raises(ValueError, match="duplicate pair_id"):
        ref_logprobs_from_rows(rows + [rows[0]], 4)
    # A gap in coverage: pair 2 unscored — training would silently mis-anchor it.
    with pytest.raises(ValueError, match="cover pair_id"):
        ref_logprobs_from_rows([r for r in rows if r["pair_id"] != 2], 4)
    # Artifact from a different (larger) dataset.
    with pytest.raises(ValueError, match="cover pair_id"):
        ref_logprobs_from_rows(rows, 3)
    # Artifact from a smaller dataset.
    with pytest.raises(ValueError, match="cover pair_id"):
        ref_logprobs_from_rows(rows, 5)


def test_load_ref_logprobs_from_disk(tmp_path):
    rows = make_ref_rows(3)
    write_ref_logprobs(rows, str(tmp_path))

    mapping = load_ref_logprobs(str(tmp_path), expected_num_pairs=3)
    assert mapping == ref_logprobs_from_rows(rows, 3)
    with pytest.raises(ValueError, match="cover pair_id"):
        load_ref_logprobs(str(tmp_path), expected_num_pairs=4)


def fingerprint(**overrides) -> ScoringFingerprint:
    fields = {
        "dataset": "d",
        "split": "train",
        "tokenizer": "org/model",
        "max_seq_length": 2048,
        "prompt_key": None,
        "tensor_model_parallel_size": 1,
        "sequence_parallel": False,
    }
    fields.update(overrides)
    return ScoringFingerprint(**fields)


def write_metadata(tmp_path, drop: tuple[str, ...] = ()) -> str:
    metadata = {key: value for key, value in asdict(fingerprint()).items() if key not in drop}
    (tmp_path / "scoring_metadata.json").write_text(json.dumps(metadata))
    return str(tmp_path)


def test_validate_scoring_metadata_passes_on_match_and_returns_the_metadata(tmp_path):
    artifact = write_metadata(tmp_path)
    assert validate_scoring_metadata(artifact, fingerprint()) == asdict(fingerprint())


def test_validate_scoring_metadata_reports_every_mismatched_key(tmp_path):
    artifact = write_metadata(tmp_path)
    with pytest.raises(ValueError) as exc_info:
        validate_scoring_metadata(artifact, fingerprint(dataset="other", max_seq_length=4096))
    message = str(exc_info.value)
    assert "dataset" in message and "max_seq_length" in message
    assert "'d'" in message and "'other'" in message  # artifact vs expected values shown


def test_validate_scoring_metadata_missing_key_counts_as_mismatch(tmp_path):
    artifact = write_metadata(tmp_path, drop=("tokenizer",))
    with pytest.raises(ValueError, match="tokenizer"):
        validate_scoring_metadata(artifact, fingerprint())


def test_validate_scoring_metadata_missing_file_names_the_artifact(tmp_path):
    with pytest.raises(ValueError, match="scoring_metadata.json"):
        validate_scoring_metadata(str(tmp_path), fingerprint())


def test_write_and_load_ref_logprobs_round_trip_through_msc_with_no_staging(fake_msc):
    url = "msc://bucket/dpo/artifact"
    rows = make_ref_rows(3)

    write_ref_logprobs(rows, url)
    write_scoring_metadata(asdict(fingerprint()), url)

    assert set(fake_msc.blobs) == {f"{url}/ref_logprobs.jsonl", f"{url}/scoring_metadata.json"}

    mapping = load_ref_logprobs(url, expected_num_pairs=3)
    assert mapping == ref_logprobs_from_rows(rows, 3)
    validate_scoring_metadata(url, fingerprint())


class _FakeLengthDataset:
    """Minimal dataset stub: items expose the two token-id lists the length sweep reads."""

    def __init__(self, lengths: list[tuple[int, int]]):
        self._lengths = lengths

    def __len__(self) -> int:
        return len(self._lengths)

    def __getitem__(self, idx: int) -> dict:
        chosen_len, rejected_len = self._lengths[idx]
        return {"chosen_input_ids": [1] * chosen_len, "rejected_input_ids": [2] * rejected_len}


def test_pair_token_lengths_takes_max_of_sides():
    dataset = _FakeLengthDataset([(5, 3), (2, 8), (4, 4)])
    assert pair_token_lengths(dataset) == [5, 8, 4]


def test_token_budget_batches_cover_every_pair_exactly_once():
    lengths = [7, 2, 4096, 130, 2, 512, 512, 3000, 64, 2]
    batches = pack_pairs_by_token_budget(lengths, budget_tokens=8192)
    flat = sorted(idx for batch in batches for idx in batch)
    assert flat == list(range(len(lengths)))


def test_token_budget_padded_cost_invariant_holds_for_multi_pair_batches():
    lengths = [7, 2, 4096, 130, 2, 512, 512, 3000, 64, 2, 900, 900, 901]
    budget = 8192
    for batch in pack_pairs_by_token_budget(lengths, budget_tokens=budget):
        padded_cost = 2 * len(batch) * max(lengths[i] for i in batch)
        if len(batch) > 1:
            assert padded_cost <= budget


def test_token_budget_oversize_pair_gets_its_own_batch():
    lengths = [10, 5000, 10]
    batches = pack_pairs_by_token_budget(lengths, budget_tokens=4096)
    assert [1] in batches  # 2 * 5000 > 4096, still scored


def test_token_budget_batches_are_sorted_longest_first_and_deterministic():
    lengths = [3, 100, 3, 50, 100, 7]
    first = pack_pairs_by_token_budget(lengths, budget_tokens=1000)
    second = pack_pairs_by_token_budget(lengths, budget_tokens=1000)
    assert first == second
    assert first[0][0] == 1  # longest length wins; equal lengths tie-break by index


def test_token_budget_stub_flood_respects_max_pairs_cap():
    batches = pack_pairs_by_token_budget([2] * 300, budget_tokens=100_000, max_pairs_per_batch=64)
    assert max(len(batch) for batch in batches) == 64
    assert sum(len(batch) for batch in batches) == 300


def test_token_budget_rejects_non_positive_knobs():
    with pytest.raises(ValueError, match="budget_tokens"):
        pack_pairs_by_token_budget([4], budget_tokens=0)
    with pytest.raises(ValueError, match="max_pairs_per_batch"):
        pack_pairs_by_token_budget([4], budget_tokens=8, max_pairs_per_batch=0)
