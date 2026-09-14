# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from collections import Counter

import pytest

from megatron.bridge.data.sources.hf import (
    HFDatasetSourceConfig,
    blend_sft_rows,
    resolve_blend_weights,
)


pytestmark = pytest.mark.unit


def _rows(tag, count):
    return [{"prompt": f"{tag}{index}", "completion": str(index)} for index in range(count)]


def _tags(rows):
    return Counter(row["prompt"][0] for row in rows)


def test_weights_default_to_uniform():
    sources = [HFDatasetSourceConfig(dataset_name="squad"), HFDatasetSourceConfig(dataset_name="gsm8k")]

    assert resolve_blend_weights(sources, None) == [1.0, 1.0]


def test_weights_must_match_the_source_count():
    sources = [HFDatasetSourceConfig(dataset_name="squad")]

    with pytest.raises(ValueError, match="equal in number"):
        resolve_blend_weights(sources, [1.0, 2.0])


def test_weights_must_be_positive():
    sources = [HFDatasetSourceConfig(dataset_name="squad")]

    with pytest.raises(ValueError, match="must be positive"):
        resolve_blend_weights(sources, [0.0])


def test_a_blend_needs_a_source():
    with pytest.raises(ValueError, match="at least one source"):
        resolve_blend_weights([], None)


def test_a_weight_of_one_keeps_every_row_once():
    blended = blend_sft_rows([_rows("a", 50), _rows("b", 20)], [1.0, 1.0], seed=1234)

    assert _tags(blended) == {"a": 50, "b": 20}


def test_a_whole_weight_repeats_the_source():
    blended = blend_sft_rows([_rows("a", 10), _rows("b", 10)], [1.0, 3.0], seed=1234)

    assert _tags(blended) == {"a": 10, "b": 30}
    # Repeats are copies of the same rows, not new ones.
    assert len({row["prompt"] for row in blended if row["prompt"].startswith("b")}) == 10


def test_a_fractional_weight_draws_a_subset_without_replacement():
    blended = blend_sft_rows([_rows("a", 100)], [0.25], seed=1234)

    assert len(blended) == 25
    assert len({row["prompt"] for row in blended}) == 25


def test_a_mixed_weight_repeats_then_draws_the_remainder():
    blended = blend_sft_rows([_rows("a", 40)], [2.5], seed=1234)

    assert len(blended) == 100
    counts = Counter(row["prompt"] for row in blended)
    assert set(counts.values()) == {2, 3}


def test_blend_size_follows_the_data_not_the_schedule():
    # Epoch counts bound the pool by the sources, so a long run cycles the pool
    # instead of materializing one row per training sample.
    blended = blend_sft_rows([_rows("a", 30), _rows("b", 30)], [1.0, 2.0], seed=1234)

    assert len(blended) == 30 + 60


def test_blend_is_deterministic_for_a_seed():
    args = ([_rows("a", 60), _rows("b", 40)], [1.0, 2.0])

    first = blend_sft_rows(*args, seed=7)
    same = blend_sft_rows(*args, seed=7)
    other = blend_sft_rows(*args, seed=8)

    assert [row["prompt"] for row in first] == [row["prompt"] for row in same]
    assert [row["prompt"] for row in first] != [row["prompt"] for row in other]
    assert _tags(first) == _tags(other)


def test_sources_are_interleaved_rather_than_concatenated():
    blended = blend_sft_rows([_rows("a", 100), _rows("b", 100)], [1.0, 1.0], seed=1234)

    assert {row["prompt"][0] for row in blended[:20]} == {"a", "b"}


def test_an_empty_source_is_rejected():
    with pytest.raises(ValueError, match="produced no rows"):
        blend_sft_rows([_rows("a", 4), []], [1.0, 1.0], seed=1234)


def test_blended_rows_are_not_copied():
    rows = _rows("a", 4)

    blended = blend_sft_rows([rows], [1.0], seed=1234)

    assert all(any(row is original for original in rows) for row in blended)


@pytest.mark.parametrize("weight", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_weights_are_rejected(weight):
    # NaN and infinity pass a plain `<= 0` check, so they need a kind check.
    sources = [HFDatasetSourceConfig(dataset_name="squad")]

    with pytest.raises(ValueError, match="positive finite"):
        resolve_blend_weights(sources, [weight])


def test_a_positive_weight_keeps_a_row_even_when_its_share_is_under_one():
    # Half a row of a one-row source still has to keep that source present.
    blended = blend_sft_rows([_rows("a", 1)], [0.5], seed=1234)

    assert len(blended) == 1


def test_a_small_share_of_a_large_source_keeps_one_row():
    blended = blend_sft_rows([_rows("a", 100)], [0.004], seed=1234)

    assert len(blended) == 1


def test_a_source_with_a_sub_row_share_does_not_vanish_from_a_blend():
    blended = blend_sft_rows([_rows("a", 40), _rows("b", 1)], [1.0, 0.5], seed=1234)

    assert _tags(blended) == {"a": 40, "b": 1}


def test_a_fractional_share_rounds_up_to_a_whole_row():
    # 100 * 0.333 is 33.3, and a partial row cannot be drawn.
    blended = blend_sft_rows([_rows("a", 100)], [0.333], seed=1234)

    assert len(blended) == 34
