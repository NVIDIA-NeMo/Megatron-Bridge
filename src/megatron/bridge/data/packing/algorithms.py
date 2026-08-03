# Copyright (c) 2024-2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bin-packing algorithms and metrics for offline sequence packing."""

import collections
import logging
from typing import Dict, List, Sequence, Tuple, TypeVar

import numpy as np
from tqdm import tqdm

from megatron.bridge.utils.safe_pickle import safe_load_npy


PACKING_ALGOS = ["first_fit_decreasing", "first_fit_shuffle"]

logger = logging.getLogger(__name__)

_ItemT = TypeVar("_ItemT")


class _SegmentTree:
    def __init__(self, capacity: int):
        self._n = capacity
        self._tree = [0] * (4 * capacity)

    def _push_up(self, node: int):
        self._tree[node] = max(self._tree[2 * node], self._tree[2 * node + 1])

    def _update(self, node: int, start: int, end: int, idx: int, val: int):
        if start == end:
            self._tree[node] = val
            return
        mid = (start + end) // 2
        if idx <= mid:
            self._update(2 * node, start, mid, idx, val)
        else:
            self._update(2 * node + 1, mid + 1, end, idx, val)
        self._push_up(node)

    def _query(self, node: int, start: int, end: int, need: int) -> int:
        if self._tree[node] < need:
            return -1
        if start == end:
            return start
        mid = (start + end) // 2
        left = self._query(2 * node, start, mid, need)
        if left != -1:
            return left
        return self._query(2 * node + 1, mid + 1, end, need)

    def update(self, idx: int, val: int):
        self._update(1, 0, self._n - 1, idx, val)

    def query_first_fit(self, need: int) -> int:
        return self._query(1, 0, self._n - 1, need)


def first_fit(
    seqlens: Sequence[_ItemT],
    pack_size: int,
    *,
    item_lengths: Sequence[int] | None = None,
) -> list[list[_ItemT]]:
    """
    Packs sequences of varying lengths into bins using the First-Fit algorithm
    with a segment-tree index for O(N log N) performance.

    A segment-tree index over per-bin remaining capacity makes each placement O(log N)
    instead of a scan over every open bin.

    By default the entries of `seqlens` are themselves the lengths. Callers that pack
    objects rather than raw integers (for example diffusion samples keyed on padded
    query sequence length) pass `item_lengths` separately and get their original
    objects back in the bins.

    Args:
      seqlens: The entries to pack, in the order they should be considered. Integer
        lengths unless `item_lengths` is given, in which case these may be any objects.
      pack_size: The maximum capacity of each bin.
      item_lengths: Optional length of each entry in `seqlens`, in the same order. When
        omitted, each entry is used as its own length.

    Returns:
      A list of lists, where each inner list represents a bin and contains the
        entries assigned to that bin.

    Raises:
      ValueError: If `item_lengths` is given and does not have the same number of
        entries as `seqlens`.
    """
    lengths: Sequence[int] = seqlens if item_lengths is None else item_lengths  # type: ignore[assignment]
    if item_lengths is not None and len(seqlens) != len(item_lengths):
        raise ValueError(
            f"seqlens and item_lengths must have the same number of entries, "
            f"got {len(seqlens)} and {len(item_lengths)}"
        )
    if not seqlens:
        return []

    n = len(seqlens)
    tree = _SegmentTree(n)
    res: list[list[_ItemT]] = []
    remaining: list[int] = []

    for item, length in zip(seqlens, lengths):
        first_bin = tree.query_first_fit(length)
        # An unopened bin still reads as 0 remaining capacity, so a zero-length item can
        # match an index past the end of `res`; that case must open a bin, not index it.
        if first_bin == -1 or first_bin >= len(res):
            new_idx = len(res)
            res.append([item])
            remaining.append(pack_size - length)
            tree.update(new_idx, remaining[new_idx])
        else:
            res[first_bin].append(item)
            remaining[first_bin] -= length
            tree.update(first_bin, remaining[first_bin])
    return res


def first_fit_decreasing(
    seqlens: Sequence[_ItemT],
    pack_size: int,
    *,
    item_lengths: Sequence[int] | None = None,
) -> list[list[_ItemT]]:
    """
    Packs sequences of varying lengths into bins using the First-Fit Decreasing algorithm.

    This is a variation of the First-Fit algorithm where the sequences are sorted by decreasing length before packing.

    Like `first_fit`, callers that pack objects rather than raw integers (for example
    diffusion samples keyed on padded query sequence length) pass `item_lengths`
    separately: the entries are then ordered by those lengths, longest first, and the
    original objects are what end up in the bins.

    Args:
      seqlens: The entries to pack. Integer lengths unless `item_lengths` is given, in
        which case these may be any objects.
      pack_size: The maximum capacity of each bin.
      item_lengths: Optional length of each entry in `seqlens`, in the same order. When
        omitted, each entry is used as its own length.

    Returns:
      A list of lists, similar to the output of the 'first_fit' function.

    Raises:
      ValueError: If `item_lengths` is given and does not have the same number of
        entries as `seqlens`.
    """
    if item_lengths is None:
        return first_fit(sorted(seqlens, reverse=True), pack_size)
    if len(seqlens) != len(item_lengths):
        raise ValueError(
            f"seqlens and item_lengths must have the same number of entries, "
            f"got {len(seqlens)} and {len(item_lengths)}"
        )
    # Order entries by their supplied length, longest first, keeping each entry paired
    # with its length. `sorted(..., reverse=True)` is stable, so entries of equal length
    # keep their original relative order -- matching `sorted(items, reverse=True)` when
    # the items compare by that same length.
    order = sorted(range(len(seqlens)), key=lambda i: item_lengths[i], reverse=True)
    sorted_items = [seqlens[i] for i in order]
    sorted_lengths = [item_lengths[i] for i in order]
    return first_fit(sorted_items, pack_size, item_lengths=sorted_lengths)


def first_fit_shuffle(seqlens: List[int], pack_size: int) -> List[List[int]]:
    """
    Packs sequences of varying lengths into bins using the First-Fit with Shuffling algorithm.

    This variation shuffles the order of the sequences before applying the First-Fit algorithm.

    Args:
      seqlens: A list of integers, representing the lengths of the sequences to be packed.
      pack_size: The maximum capacity of each bin.

    Returns:
      A list of lists, similar to the output of the 'first_fit' function.
    """
    shuffled_seqlens = seqlens[:]
    np.random.shuffle(shuffled_seqlens)
    return first_fit(shuffled_seqlens, pack_size)


def create_hist(dataset: np.array, truncate_seq_len: int) -> Tuple[Dict[int, List[Dict]], List[int]]:
    """
    Creates a histogram of sequence lengths from a tokenized dataset.

    This function analyzes the tokenized dataset and creates a histogram showing the distribution of sequence lengths.

    Args:
      dataset: A NumPy array containing the tokenized sequences. Each element is a dictionary that contains at minimum
               the key `input_ids`.
      truncate_seq_len: The maximum sequence length to consider in the histogram.

    Returns:
      sequences: A dictionary where keys are sequence lengths and values are lists
                 of corresponding sequences from the dataset.
      histogram: A list representing the histogram data (number of sequences for each length).
    """
    logger.info("Creating histogram from tokenized dataset...")

    sequences = collections.defaultdict(list)
    counts = [0] * (truncate_seq_len + 1)
    num_skipped = 0

    for item_dict in dataset:
        # Minus 1 here to account for the fact that transformer input and label
        # have one less token than the full sequence.
        # Input is missing the last token and label is missing the first token
        # (this way the tokens are aligned for next token prediction).
        # We want pack size to be the length of the actual input and label, hence minus 1.
        seq_len = len(item_dict["input_ids"]) - 1
        if seq_len > truncate_seq_len:
            num_skipped += 1
            continue
        sequences[seq_len].append(item_dict)
        counts[seq_len] += 1

    if num_skipped:
        logger.warning(
            "Skipped %d sequences longer than the maximum packed sequence length %d",
            num_skipped,
            truncate_seq_len,
        )

    logger.debug("Histogram of sequence lengths")
    logger.debug(counts)

    histogram = []
    for seq_len in range(truncate_seq_len + 1):
        histogram.append(len(sequences[seq_len]))

    return sequences, histogram


def create_packing_strategy(
    histogram: List[int], pack_size: int, packing_algorithm: str = "first_fit"
) -> Tuple[List[List[int]], Dict[str, int]]:
    """
    Packs sequences into bins using the specified packing algorithm.

    This function takes the histogram of sequence lengths, desired pack size, and a string representing the packing
    algorithm to use. It then calls the corresponding function (e.g., 'first_fit_decreasing') and performs the
    packing process using only sequence lengths as input (without the actual sequences).

    Args:
          histogram: A list representing the histogram data (number of sequences for each length).
          pack_size: The maximum capacity of each bin.
          packing_algorithm: One of the supported packing algorithms from ['first_fit_decreasing', 'first_fit_shuffle']

    Returns:
          assignments: A list of lists, where each inner list represents a bin and contains the indices of the
                        sequence lengths assigned to that bin.
          pack_metadata: A dict that records packing metadata, for instance the max number of samples per bin.
    """

    logger.info(f"Packing sequences to length {pack_size}...")

    all_seq_lens = []
    for i, count in enumerate(histogram):
        all_seq_lens.extend([i] * count)

    packing_fn = globals()[packing_algorithm]
    assignments: list[list[int]] = packing_fn(all_seq_lens, pack_size)
    packed_seq_lens = [sum(x) for x in assignments]
    packing_factor = len(all_seq_lens) / len(packed_seq_lens)

    max_seqlen = max(all_seq_lens)
    max_samples_per_bin = max([len(b) for b in assignments])
    min_packed_seqlen = min(packed_seq_lens)
    packing_efficiency = sum(packed_seq_lens) / len(packed_seq_lens) / pack_size * 100

    packing_metadata = {
        "dataset_max_seqlen": max_seqlen,
        "max_samples_per_bin": max_samples_per_bin,
        "packing_factor": round(packing_factor, 2),
        "packing_efficiency": round(packing_efficiency, 2),
        "pack_size": pack_size,
        "min_packed_seqlen": min_packed_seqlen,
    }

    logger.debug("Packed sequence lengths:")
    logger.debug(packed_seq_lens)
    logger.info(f"Packing is {packing_efficiency:.2f}% efficient")
    logger.info(
        f">>>>> For pack size {pack_size}, average number of sequences per pack is n = {packing_factor:.3f} <<<<<"
    )
    return assignments, packing_metadata


def fill_packing_strategy(
    assignments: List[List[int]],
    sequences: Dict[int, List[Dict]],
    pack_size: int,
    pad_id: int,
) -> List[Dict]:
    """
    Fills the packing strategy with actual sequence data based on assignments and sequence information.
    This function takes the assignments generated by the packing algorithm (containing sequence length indices),
    the original sequences data, and the pack size. It iterates through the assignments, retrieves the corresponding
    sequences from the sequences dictionary, and constructs the final output data structure with input IDs, loss masks
    (if available), and starting indices for each sequence in a packed sequence.
    Args:
          assignments: A list of lists, where each inner list represents a bin and contains the indices of the
                        sequence lengths assigned to that bin (output of 'create_packing_strategy').
          sequences: A dictionary where keys are sequence lengths and values are lists of corresponding sequences
                      from the dataset (output of 'create_hist').
          pack_size: The maximum capacity of each bin.
          pad_id: The tokenizer's padding token.
    Returns:
          output_data: A list of dictionaries, where each dictionary represents a packed sequence with its input IDs,
                        loss mask (if available), and starting indices.
    """
    ifile_handles = dict()
    for seq_len in tqdm(range(pack_size + 1)):
        per_seq_data = sequences[seq_len]
        if len(per_seq_data) > 0:
            perm = np.random.permutation(len(per_seq_data))
            input_ids = np.array([x["input_ids"] for x in per_seq_data])[perm].tolist()
            try:
                loss_mask = np.array([x["loss_mask"] for x in per_seq_data])[perm].tolist()
                # roll loss mask by 1 to align with labels. We want to train on the output after the last context token
                loss_mask = [x[1:] + [False] for x in loss_mask]
            except KeyError:
                try:
                    loss_mask = np.array(
                        [
                            [
                                # (x['answer_start_idx'] - 1) because we want to train on the output
                                # after the last context token
                                idx >= (x["answer_start_idx"] - 1)
                                for idx in range(len(x["input_ids"]))
                            ]
                            for x in per_seq_data
                        ]
                    )[perm].tolist()
                except KeyError as err:
                    err_msg = "Key errors loss_mask and answer_start_idx missing in example - "
                    err_msg += f"{err} {per_seq_data[0]}"
                    logging.error(err_msg)
                    raise ValueError(err_msg)
            ifile_handles[seq_len] = (input_ids, loss_mask)
    input_ids, loss_mask, seq_start_id = {}, {}, {}
    for oindex, assignment in tqdm(enumerate(assignments), total=len(assignments)):
        _input_ids, _loss_mask, _seq_start_id = [], [], [0]
        for seq_length in assignment:
            _input_ids.extend(ifile_handles[seq_length][0].pop())
            _loss_mask.extend(ifile_handles[seq_length][1].pop())
            _seq_start_id.append(len(_input_ids))
        input_ids[oindex] = _input_ids
        loss_mask[oindex] = _loss_mask
        seq_start_id[oindex] = _seq_start_id[:-1]
    output_data = []
    for i in range(len(input_ids)):
        item_dict = {
            "input_ids": input_ids[i],
            "loss_mask": loss_mask[i],
            "seq_start_id": seq_start_id[i],
        }
        output_data.append(item_dict)
    assert all(not seq[0] for seq in ifile_handles.values()), "Error: There are items left over from the assignment"
    assert all(not seq[1] for seq in ifile_handles.values()), "Error: There are items left over from the assignment"
    return output_data


def get_seqlen_list(elem: Dict) -> Tuple[List[int], int]:
    """Extract per-sequence token counts from a packed dataset element.

    Args:
        elem: A packed dataset element with 'input_ids' and 'seq_start_id' fields.

    Returns:
        A tuple of (token_counts, tokens_minus_eos) where token_counts is a list of
        per-sequence token counts (excluding EOS) and tokens_minus_eos is the total
        token count excluding EOS tokens.
    """
    num_seq = len(elem["seq_start_id"])
    tokens_total = len(elem["input_ids"])
    tokens_minus_eos = tokens_total - num_seq

    seq_boundaries = elem["seq_start_id"] + [tokens_total]

    # subtract 1 to account for removing eos token
    token_counts = [seq_boundaries[i + 1] - seq_boundaries[i] - 1 for i in range(num_seq)]

    assert sum(token_counts) == tokens_minus_eos, (sum(token_counts), tokens_minus_eos)

    return token_counts, tokens_minus_eos


def calculate_avg_seqlen(
    dataset_file: str, gbs: int, max_seq_len: int, drop_remainder: bool
) -> Tuple[float, float, float, float]:
    """Calculate average sequence length statistics from a packed dataset.

    Args:
        dataset_file: Path to the packed dataset. Either a legacy ``.npy`` file, or a
            parquet spec (``.parquet`` / ``.pq``) which may be a single file, a glob
            pattern, or a directory -- resolved via ``resolve_packed_parquet_paths``
            so this matches the specs accepted by the FLOP-accounting existence gate.
        gbs: Global batch size used to determine how many rows to process.
        max_seq_len: Maximum sequence length (reserved for future use).
        drop_remainder: If True, drop rows that don't fill a complete batch.

    Returns:
        A tuple of (avg_seqlen_count, avg_seqlen_total, avg_seqlen_sq_individual, avg_seqlen_sq_per_row):
            - avg_seqlen_count: Average number of sequences per row.
            - avg_seqlen_total: Average total tokens (excluding EOS) per row.
            - avg_seqlen_sq_individual: Average of squared per-sequence lengths.
            - avg_seqlen_sq_per_row: Average of summed squared sequence lengths per row.

    Raises:
        ValueError: If no rows remain after applying drop_remainder, or if no sequences are found.
    """
    # Same predicate the packed-parquet loader (and flop_utils._packed_data_exists)
    # uses, so the format detected here matches the spec accepted by the existence
    # gate -- covers single file, glob pattern, and directory specs.
    from megatron.bridge.data.packing.paths import is_packed_parquet_spec, resolve_packed_parquet_paths

    if is_packed_parquet_spec(dataset_file):
        import pyarrow.parquet as pq

        # Resolve the spec (single file / glob / directory) to concrete shard paths
        # before reading, so a glob/dir spec that passes the existence gate does not
        # then crash pq.read_table (which cannot take a raw glob string).
        shards = resolve_packed_parquet_paths(dataset_file)
        table = pq.read_table(shards, columns=["input_ids", "seq_start_id"])
        ids, starts = table.column("input_ids"), table.column("seq_start_id")
        data = [{"input_ids": ids[i].as_py(), "seq_start_id": starts[i].as_py()} for i in range(table.num_rows)]
    else:
        with open(dataset_file, "rb") as f:
            data = safe_load_npy(f.read())

    total_len_accum = 0
    seqlen_sq_accum = 0
    seq_count_accum = 0

    rows_total = len(data)
    count = (rows_total // gbs) * gbs if drop_remainder else rows_total

    if count != rows_total:
        logger.info(f"Dropping {rows_total - count}, total was {rows_total}")

    for i, elem in enumerate(data):
        if i >= count:
            break
        seqlen_list, total_count = get_seqlen_list(elem)
        seqlen_sq_list = [s * s for s in seqlen_list]
        total_len_accum += total_count
        seqlen_sq_accum += sum(seqlen_sq_list)
        seq_count_accum += len(seqlen_list)

    if count == 0:
        raise ValueError(
            f"No rows to process: dataset has {rows_total} rows but gbs={gbs} with drop_remainder={drop_remainder}."
        )
    if seq_count_accum == 0:
        raise ValueError("No sequences found in dataset; cannot compute average sequence length.")

    avg_seqlen_count = seq_count_accum / count
    avg_seqlen_total = total_len_accum / count
    avg_seqlen_sq_individual = seqlen_sq_accum / seq_count_accum
    avg_seqlen_sq_per_row = seqlen_sq_accum / count

    return avg_seqlen_count, avg_seqlen_total, avg_seqlen_sq_individual, avg_seqlen_sq_per_row
