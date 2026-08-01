# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""MCore ``.bin/.idx`` storage for offline-packed GPT SFT samples.

Each IndexedDataset item contains one complete pack. Token IDs use the lower
31 bits of an int32 word and the target-aligned binary loss mask uses the high
bit. A versioned header stores the sequence start offsets. This preserves the
existing packed Parquet semantics while using MCore's mmap-backed storage.
"""

from __future__ import annotations

import bisect
import glob
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from megatron.core.datasets.indexed_dataset import (
    IndexedDataset,
    IndexedDatasetBuilder,
    get_bin_path,
    get_idx_path,
)
from megatron.core.datasets.object_storage_utils import ObjectStorageConfig
from megatron.core.msc_utils import MultiStorageClientFeature

from megatron.bridge.data.packing.gpt_sft import GPTSFTPackedDataset


if TYPE_CHECKING:
    from megatron.bridge.training.tokenizers.tokenizer import MegatronTokenizer


PACKED_SFT_SUFFIX = ".sft"
_ROW_MAGIC = 0x53465442  # ASCII "SFTB".
_FORMAT_VERSION = 1
_HEADER_WORDS = 3
_TOKEN_BITS_MASK = np.uint32(0x7FFFFFFF)
_LOSS_MASK_SHIFT = np.uint32(31)


def normalize_packed_indexed_prefix(path: str | Path) -> str:
    """Normalize a packed IndexedDataset path to its extension-free prefix."""
    path_str = str(path)
    if path_str.lower().endswith((".bin", ".idx")):
        return path_str[:-4]
    return path_str


def _glob_prefixes(pattern: str) -> list[str]:
    """Resolve a local or MSC glob to complete IndexedDataset prefixes."""
    if MultiStorageClientFeature.is_enabled():
        msc = MultiStorageClientFeature.import_package()
        if hasattr(msc, "glob"):
            matches = [str(path) for path in msc.glob(pattern)]
        else:
            pattern_path = msc.Path(pattern)
            matches = [str(path) for path in pattern_path.parent.glob(pattern_path.name)]
    else:
        matches = glob.glob(pattern)
    prefixes = {normalize_packed_indexed_prefix(path) for path in matches}
    return sorted(prefix for prefix in prefixes if IndexedDataset.exists(prefix))


def resolve_packed_indexed_prefixes(spec: str | Path) -> list[str]:
    """Resolve a packed SFT IndexedDataset prefix, pair file, glob, or directory.

    Args:
        spec: A single prefix, either pair file, glob, or directory containing
            ``*.sft.bin``/``*.sft.idx`` pairs.

    Returns:
        Sorted, normalized IndexedDataset prefixes.

    Raises:
        ValueError: If no complete dataset pair can be resolved.
    """
    spec_str = str(spec)
    if "*" in spec_str or "?" in spec_str:
        suffix = "" if spec_str.lower().endswith((".bin", ".idx")) else ".idx"
        prefixes = _glob_prefixes(f"{spec_str}{suffix}")
        if not prefixes:
            raise ValueError(f"No packed SFT .bin/.idx datasets found matching: {spec_str}")
        return prefixes

    if MultiStorageClientFeature.is_enabled():
        msc = MultiStorageClientFeature.import_package()
        path = msc.Path(spec_str)
        is_dir = path.is_dir() if hasattr(path, "is_dir") else False
    else:
        path = Path(spec_str)
        is_dir = path.is_dir()
    if is_dir:
        prefixes = _glob_prefixes(os.path.join(spec_str, f"*{PACKED_SFT_SUFFIX}.idx"))
        if not prefixes:
            raise ValueError(f"No packed SFT .bin/.idx datasets found in directory: {spec_str}")
        return prefixes

    prefix = normalize_packed_indexed_prefix(spec_str)
    if not IndexedDataset.exists(prefix):
        raise ValueError(
            f"Packed SFT IndexedDataset is incomplete or missing for prefix '{prefix}'; "
            f"expected both '{get_bin_path(prefix)}' and '{get_idx_path(prefix)}'"
        )
    return [prefix]


def is_packed_indexed_dataset(spec: str | Path) -> bool:
    """Return whether a path resolves to packed SFT ``.bin/.idx`` data."""
    try:
        return bool(resolve_packed_indexed_prefixes(spec))
    except ValueError:
        return False


def _as_integral_vector(value: Sequence[int | float] | np.ndarray, *, field: str) -> np.ndarray:
    """Validate and normalize one packed-row field as an integer vector."""
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{field} must be one-dimensional, got shape {array.shape}")
    if array.size == 0:
        return array.astype(np.int64)
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{field} must contain integers, got dtype {array.dtype}")
    return array.astype(np.int64, copy=False)


def encode_packed_row(row: Mapping[str, Sequence[int | float] | np.ndarray]) -> np.ndarray:
    """Encode a Parquet-compatible packed row as versioned int32 words.

    Args:
        row: Mapping containing ``input_ids``, ``loss_mask``, and
            ``seq_start_id``.

    Returns:
        Encoded one-dimensional int32 array.

    Raises:
        ValueError: If the packed row invariants are violated.
    """
    missing = {"input_ids", "loss_mask", "seq_start_id"} - row.keys()
    if missing:
        raise ValueError(f"Packed SFT row is missing required fields: {sorted(missing)}")

    input_ids = _as_integral_vector(row["input_ids"], field="input_ids")
    seq_start_id = _as_integral_vector(row["seq_start_id"], field="seq_start_id")
    loss_mask = np.asarray(row["loss_mask"])
    if loss_mask.ndim != 1:
        raise ValueError(f"loss_mask must be one-dimensional, got shape {loss_mask.shape}")
    if input_ids.size == 0:
        raise ValueError("input_ids must not be empty")
    if loss_mask.size != input_ids.size:
        raise ValueError(f"loss_mask length ({loss_mask.size}) != input_ids length ({input_ids.size})")
    if not np.all((loss_mask == 0) | (loss_mask == 1)):
        raise ValueError("loss_mask must contain only binary 0/1 values")
    if np.any(input_ids < 0) or np.any(input_ids > int(_TOKEN_BITS_MASK)):
        raise ValueError(f"input_ids must be in the range [0, {int(_TOKEN_BITS_MASK)}]")
    if seq_start_id.size == 0 or seq_start_id[0] != 0:
        raise ValueError("seq_start_id must start with 0")
    if np.any(seq_start_id < 0) or np.any(seq_start_id >= input_ids.size):
        raise ValueError("seq_start_id values must be within input_ids")
    if np.any(seq_start_id[1:] <= seq_start_id[:-1]):
        raise ValueError("seq_start_id values must be strictly increasing")

    token_words = input_ids.astype(np.uint32) | (loss_mask.astype(np.uint32) << _LOSS_MASK_SHIFT)
    header = np.asarray([_ROW_MAGIC, _FORMAT_VERSION, seq_start_id.size], dtype=np.int32)
    return np.concatenate((header, seq_start_id.astype(np.int32), token_words.view(np.int32)))


def decode_packed_row(encoded: np.ndarray, *, row_index: int) -> dict[str, np.ndarray | list[int]]:
    """Decode and validate one packed SFT IndexedDataset item."""
    if encoded.dtype != np.int32:
        raise ValueError(f"Packed SFT row {row_index} uses dtype {encoded.dtype}; expected int32")
    if encoded.ndim != 1 or encoded.size < _HEADER_WORDS + 2:
        raise ValueError(f"Packed SFT row {row_index} is too short to contain a valid sample")
    if int(encoded[0]) != _ROW_MAGIC:
        raise ValueError(f"Packed SFT row {row_index} has invalid magic {int(encoded[0]):#x}")
    if int(encoded[1]) != _FORMAT_VERSION:
        raise ValueError(
            f"Packed SFT row {row_index} uses unsupported format version {int(encoded[1])}; expected {_FORMAT_VERSION}"
        )

    boundary_count = int(encoded[2])
    token_offset = _HEADER_WORDS + boundary_count
    if boundary_count <= 0 or token_offset >= encoded.size:
        raise ValueError(f"Packed SFT row {row_index} has invalid boundary count {boundary_count}")
    seq_start_id = encoded[_HEADER_WORDS:token_offset].astype(np.int64)
    token_words = encoded[token_offset:].view(np.uint32)
    if seq_start_id[0] != 0:
        raise ValueError(f"Packed SFT row {row_index} sequence boundaries must start with 0")
    if np.any(seq_start_id < 0) or np.any(seq_start_id >= token_words.size):
        raise ValueError(f"Packed SFT row {row_index} has sequence boundaries outside the token range")
    if np.any(seq_start_id[1:] <= seq_start_id[:-1]):
        raise ValueError(f"Packed SFT row {row_index} sequence boundaries must be strictly increasing")
    return {
        "input_ids": (token_words & _TOKEN_BITS_MASK).astype(np.int64),
        "loss_mask": (token_words >> _LOSS_MASK_SHIFT).astype(np.int64),
        "seq_start_id": seq_start_id.tolist(),
    }


def write_packed_indexed_dataset(
    rows: Sequence[Mapping[str, Sequence[int | float] | np.ndarray]], output_path: str | Path
) -> str:
    """Write packed SFT rows to one MCore ``.bin/.idx`` pair."""
    if not rows:
        raise ValueError("Cannot write an empty packed SFT IndexedDataset")
    prefix = normalize_packed_indexed_prefix(output_path)
    builder = IndexedDatasetBuilder(get_bin_path(prefix), dtype=np.int32)
    for row in rows:
        builder.add_item(torch.from_numpy(encode_packed_row(row)))
        builder.end_document()
    builder.finalize(get_idx_path(prefix))
    return prefix


class PackedSFTIndexedDataset(Sequence[dict[str, np.ndarray | list[int]]]):
    """Read one or more packed SFT IndexedDataset shards as one sequence."""

    def __init__(
        self,
        path_spec: str | Path,
        *,
        mmap: bool = True,
        object_storage_config: ObjectStorageConfig | None = None,
    ) -> None:
        """Initialize the packed indexed reader."""
        self.prefixes = resolve_packed_indexed_prefixes(path_spec)
        self.datasets = [
            IndexedDataset(prefix, mmap=mmap, object_storage_config=object_storage_config) for prefix in self.prefixes
        ]
        self.offsets = [0]
        for dataset in self.datasets:
            self.offsets.append(self.offsets[-1] + len(dataset))
        if self.offsets[-1] == 0:
            raise ValueError(f"Packed SFT IndexedDataset is empty: {path_spec}")
        decode_packed_row(self.datasets[0][0], row_index=0)

    def __len__(self) -> int:
        """Return the row count across all shards."""
        return self.offsets[-1]

    def __getitem__(
        self, index: int | slice
    ) -> dict[str, np.ndarray | list[int]] | list[dict[str, np.ndarray | list[int]]]:
        """Decode one row or a Python slice."""
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return [self[item_index] for item_index in range(start, stop, step)]
        normalized_index = index + len(self) if index < 0 else index
        if normalized_index < 0 or normalized_index >= len(self):
            raise IndexError(f"Packed SFT row index out of range: {index}")
        dataset_index = bisect.bisect_right(self.offsets, normalized_index) - 1
        local_index = normalized_index - self.offsets[dataset_index]
        return decode_packed_row(self.datasets[dataset_index][local_index], row_index=normalized_index)


class GPTSFTPackedIndexedDataset(GPTSFTPackedDataset):
    """Packed GPT SFT training dataset backed by MCore IndexedDataset."""

    def __init__(
        self,
        file_path: str,
        tokenizer: MegatronTokenizer,
        return_cu_seqlen: bool = True,
        pad_cu_seqlens: bool = False,
        pack_metadata_file_path: str | None = None,
        mmap: bool = True,
        object_storage_config: ObjectStorageConfig | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize an indexed packed GPT SFT dataset."""
        self._path_spec = file_path
        self._mmap = mmap
        self._object_storage_config = object_storage_config
        super().__init__(
            file_path=file_path,
            tokenizer=tokenizer,
            return_cu_seqlen=return_cu_seqlen,
            pad_cu_seqlens=pad_cu_seqlens,
            pack_metadata_file_path=pack_metadata_file_path,
            **kwargs,
        )

    def _load_dataset(self) -> None:
        """Load the versioned packed SFT IndexedDataset rows."""
        self.indexed_dataset = PackedSFTIndexedDataset(
            self._path_spec, mmap=self._mmap, object_storage_config=self._object_storage_config
        )

    def __getitem__(self, index: int) -> dict[str, np.ndarray | list[int]]:
        """Load and decode one packed training sample."""
        is_padding_sample = index < 0
        if self.samples_mapping is not None:
            index = int(self.samples_mapping[index])
        row = self.indexed_dataset[index]
        input_ids = row["input_ids"]
        loss_mask = row["loss_mask"]
        seq_start_id = row["seq_start_id"]
        assert isinstance(input_ids, np.ndarray)
        assert isinstance(loss_mask, np.ndarray)
        assert isinstance(seq_start_id, list)
        if is_padding_sample or index < 0:
            loss_mask = np.zeros_like(loss_mask)
        return {
            "input_ids": input_ids,
            "seq_boundaries": [*seq_start_id, len(input_ids)],
            "loss_mask": loss_mask,
        }
