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
import contextlib
import fcntl
import glob
import logging
import os
import time
import uuid
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
from megatron.core.datasets.object_storage_utils import (
    ObjectStorageConfig,
    cache_index_file,
    get_index_cache_path,
    is_object_storage_path,
    parse_s3_path,
)
from megatron.core.msc_utils import MultiStorageClientFeature

from megatron.bridge.data.packing.gpt_sft import GPTSFTPackedDataset


if TYPE_CHECKING:
    from megatron.bridge.training.tokenizers.tokenizer import MegatronTokenizer


PACKED_SFT_SUFFIX = ".sft"
DEFAULT_OBJECT_STORAGE_BIN_CHUNK_NBYTES = 256 * 1024 * 1024
_ROW_MAGIC = 0x53465442  # ASCII "SFTB".
_FORMAT_VERSION = 1
_HEADER_WORDS = 3
_TOKEN_BITS_MASK = np.uint32(0x7FFFFFFF)
_LOSS_MASK_SHIFT = np.uint32(31)

logger = logging.getLogger(__name__)


def normalize_packed_indexed_prefix(path: str | Path) -> str:
    """Normalize a packed IndexedDataset path to its extension-free prefix."""
    path_str = str(path)
    if path_str.lower().endswith((".bin", ".idx")):
        return path_str[:-4]
    return path_str


def is_packed_indexed_spec(spec: str | Path) -> bool:
    """Return whether a path syntactically names canonical packed SFT indexed data."""
    spec_str = str(spec).lower()
    return spec_str.endswith((PACKED_SFT_SUFFIX, f"{PACKED_SFT_SUFFIX}.bin", f"{PACKED_SFT_SUFFIX}.idx"))


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
    return sorted(prefix for prefix in prefixes if _packed_indexed_pair_exists(prefix))


def _s3_object_exists(path: str) -> bool:
    """Check one S3 object without relying on MCore's pinned existence helper."""
    try:
        import boto3
        from botocore.exceptions import ClientError
    except ModuleNotFoundError as error:
        raise RuntimeError("Reading packed SFT data from s3:// requires boto3 and botocore") from error

    bucket, key = parse_s3_path(path)
    client = boto3.client("s3")
    try:
        client.head_object(Bucket=bucket, Key=key)
    except ClientError as error:
        response = error.response
        error_code = str(response.get("Error", {}).get("Code", ""))
        status_code = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if error_code in {"404", "NoSuchKey", "NotFound"} or status_code == 404:
            return False
        raise
    finally:
        close = getattr(client, "close", None)
        if close is not None:
            close()
    return True


def _packed_indexed_pair_exists(prefix: str) -> bool:
    """Return whether both files in one local, MSC, or S3 pair exist."""
    if prefix.startswith("s3://"):
        return _s3_object_exists(get_bin_path(prefix)) and _s3_object_exists(get_idx_path(prefix))
    if prefix.startswith("msc://"):
        msc = MultiStorageClientFeature.import_package()
        return msc.Path(get_bin_path(prefix)).exists() and msc.Path(get_idx_path(prefix)).exists()
    return IndexedDataset.exists(prefix)


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
    if spec_str.startswith("msc://") and not MultiStorageClientFeature.is_enabled():
        MultiStorageClientFeature.enable()
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
    if not _packed_indexed_pair_exists(prefix):
        raise ValueError(
            f"Packed SFT IndexedDataset is incomplete or missing for prefix '{prefix}'; "
            f"expected both '{get_bin_path(prefix)}' and '{get_idx_path(prefix)}'"
        )
    return [prefix]


def _refresh_directory_metadata(spec: str | Path) -> None:
    """Refresh the nearest local ancestor after a stale shared-filesystem miss."""
    spec_str = str(spec)
    if is_object_storage_path(spec_str):
        return
    base = spec_str.split("*", 1)[0].split("?", 1)[0]
    directory = Path(base if os.path.isdir(base) else os.path.dirname(base))
    while True:
        try:
            os.listdir(directory)
            return
        except OSError:
            parent = directory.parent
            if parent == directory:
                return
            directory = parent


def resolve_packed_indexed_prefixes_with_retry(
    spec: str | Path,
    *,
    max_attempts: int = 10,
    backoff_s: float = 1.0,
) -> list[str]:
    """Resolve packed indexed data with bounded NFS metadata refresh and retry."""
    if max_attempts <= 0:
        raise ValueError("max_attempts must be greater than 0")
    last_error: ValueError | None = None
    for attempt in range(max_attempts):
        try:
            return resolve_packed_indexed_prefixes(spec)
        except ValueError as error:
            last_error = error
            if attempt == max_attempts - 1:
                break
            _refresh_directory_metadata(spec)
            time.sleep(backoff_s * (attempt + 1))
    assert last_error is not None
    raise last_error


def is_packed_indexed_dataset(spec: str | Path) -> bool:
    """Return whether a path resolves to packed SFT ``.bin/.idx`` data."""
    try:
        return bool(resolve_packed_indexed_prefixes(spec))
    except ValueError:
        return False


def default_packed_index_cache_path() -> Path:
    """Return the standard local index-cache directory for remote packed data."""
    nemo_home = Path(os.getenv("NEMO_HOME", Path.home() / ".cache" / "nemo"))
    datasets_cache = Path(os.getenv("NEMO_DATASETS_CACHE", nemo_home / "datasets"))
    cache_path = datasets_cache / "packed_sft_index_cache"
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


def make_object_storage_config(
    cache_path: str | Path | None,
    *,
    bin_chunk_nbytes: int = DEFAULT_OBJECT_STORAGE_BIN_CHUNK_NBYTES,
) -> ObjectStorageConfig:
    """Build MCore's object-storage reader config from declarative values."""
    if bin_chunk_nbytes <= 0:
        raise ValueError("object_storage_bin_chunk_nbytes must be greater than 0")
    if cache_path is not None and is_object_storage_path(str(cache_path)):
        raise ValueError("object_storage_cache_path must be a local path visible to every rank")
    resolved_cache_path = default_packed_index_cache_path() if cache_path is None else Path(cache_path)
    resolved_cache_path.mkdir(parents=True, exist_ok=True)
    return ObjectStorageConfig(
        path_to_idx_cache=str(resolved_cache_path),
        bin_chunk_nbytes=bin_chunk_nbytes,
    )


def cache_packed_indexed_dataset_indices(
    path_spec: str | Path,
    *,
    cache_path: str | Path | None,
    bin_chunk_nbytes: int = DEFAULT_OBJECT_STORAGE_BIN_CHUNK_NBYTES,
) -> None:
    """Cache remote index files before all distributed ranks construct readers."""
    prefixes = resolve_packed_indexed_prefixes(path_spec)
    remote_prefixes = [prefix for prefix in prefixes if is_object_storage_path(prefix)]
    if not remote_prefixes:
        return
    object_storage_config = make_object_storage_config(
        cache_path,
        bin_chunk_nbytes=bin_chunk_nbytes,
    )
    for prefix in remote_prefixes:
        idx_path = get_idx_path(prefix)
        local_idx_path = get_index_cache_path(idx_path, object_storage_config)
        cache_index_file(idx_path, local_idx_path)


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


def _unlink_path(path: str) -> None:
    """Remove a local path when it exists."""
    Path(path).unlink(missing_ok=True)


def _cleanup_path(path: str) -> None:
    """Remove a temporary path without masking the primary write error."""
    try:
        _unlink_path(path)
    except OSError:
        logger.warning("Could not clean up temporary packed SFT path %s", path, exc_info=True)


def _replace_path(source: str, destination: str) -> None:
    """Atomically replace one local path."""
    os.replace(source, destination)


@contextlib.contextmanager
def _publication_lock(prefix: str, *, exclusive: bool):
    """Coordinate local readers and writers while a pair is opened or published."""
    lock_path = f"{prefix}.lock"
    with open(lock_path, "a+b") as lock_file:
        lock_mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        fcntl.flock(lock_file.fileno(), lock_mode)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _publish_temporary_pair(temporary_prefix: str, final_prefix: str) -> None:
    """Publish the data file first and roll back both files on failure."""
    temporary_bin = get_bin_path(temporary_prefix)
    temporary_idx = get_idx_path(temporary_prefix)
    final_bin = get_bin_path(final_prefix)
    final_idx = get_idx_path(final_prefix)
    backup_prefix = f"{final_prefix}.backup-{uuid.uuid4().hex}"
    backup_bin = get_bin_path(backup_prefix)
    backup_idx = get_idx_path(backup_prefix)

    with _publication_lock(final_prefix, exclusive=True):
        old_bin_backed_up = False
        old_idx_backed_up = False
        new_bin_published = False
        new_idx_published = False
        publication_succeeded = False
        rollback_succeeded = False
        try:
            # Remove the old commit point before moving its data. Readers take
            # the shared side of this lock while opening both files.
            if Path(final_idx).exists():
                _replace_path(final_idx, backup_idx)
                old_idx_backed_up = True
            if Path(final_bin).exists():
                _replace_path(final_bin, backup_bin)
                old_bin_backed_up = True
            _replace_path(temporary_bin, final_bin)
            new_bin_published = True
            _replace_path(temporary_idx, final_idx)
            new_idx_published = True
        except Exception as publication_error:
            try:
                if old_bin_backed_up:
                    _replace_path(backup_bin, final_bin)
                elif new_bin_published:
                    _cleanup_path(final_bin)
                if old_idx_backed_up:
                    _replace_path(backup_idx, final_idx)
                elif new_idx_published:
                    _cleanup_path(final_idx)
            except Exception as rollback_error:
                publication_error.add_note(f"Rollback also failed: {rollback_error}")
                raise publication_error from rollback_error
            rollback_succeeded = True
            raise
        else:
            publication_succeeded = True
        finally:
            # Preserve backups for manual recovery if rollback itself raises.
            if publication_succeeded or rollback_succeeded:
                _cleanup_path(backup_bin)
                _cleanup_path(backup_idx)


def write_packed_indexed_dataset(
    rows: Sequence[Mapping[str, Sequence[int | float] | np.ndarray]], output_path: str | Path
) -> str:
    """Write packed SFT rows to a transactionally published MCore pair."""
    if not rows:
        raise ValueError("Cannot write an empty packed SFT IndexedDataset")

    prefix = normalize_packed_indexed_prefix(output_path)
    if is_object_storage_path(prefix):
        raise ValueError(
            "Direct writes to object storage are not supported for packed SFT IndexedDataset; "
            "write a local pair and upload both files before training"
        )

    # Validate before creating any output, so a malformed later row cannot
    # truncate an existing valid pair.
    for row in rows:
        encode_packed_row(row)

    temporary_prefix = f"{prefix}.tmp-{uuid.uuid4().hex}"
    temporary_bin = get_bin_path(temporary_prefix)
    temporary_idx = get_idx_path(temporary_prefix)
    try:
        builder = IndexedDatasetBuilder(temporary_bin, dtype=np.int32)
        for row in rows:
            builder.add_item(torch.from_numpy(encode_packed_row(row)))
            builder.end_document()
        builder.finalize(temporary_idx)
        _publish_temporary_pair(temporary_prefix, prefix)
    finally:
        _cleanup_path(temporary_bin)
        _cleanup_path(temporary_idx)
    return prefix


class PackedSFTIndexedDataset(Sequence[dict[str, np.ndarray | list[int]]]):
    """Read one or more packed SFT IndexedDataset shards as one sequence."""

    def __init__(
        self,
        path_spec: str | Path,
        *,
        mmap: bool = True,
        object_storage_config: ObjectStorageConfig | None = None,
        object_storage_cache_path: str | Path | None = None,
        object_storage_bin_chunk_nbytes: int = DEFAULT_OBJECT_STORAGE_BIN_CHUNK_NBYTES,
    ) -> None:
        """Initialize the packed indexed reader."""
        self.prefixes = resolve_packed_indexed_prefixes_with_retry(path_spec)
        has_remote_prefix = any(is_object_storage_path(prefix) for prefix in self.prefixes)
        if has_remote_prefix and object_storage_config is None:
            object_storage_config = make_object_storage_config(
                object_storage_cache_path,
                bin_chunk_nbytes=object_storage_bin_chunk_nbytes,
            )

        self.datasets = []
        for prefix in self.prefixes:
            is_remote = is_object_storage_path(prefix)
            lock_context = contextlib.nullcontext() if is_remote else _publication_lock(prefix, exclusive=False)
            with lock_context:
                dataset = IndexedDataset(
                    prefix,
                    mmap=mmap and not is_remote,
                    object_storage_config=object_storage_config if is_remote else None,
                )
                if len(dataset) == 0:
                    raise ValueError(f"Packed SFT IndexedDataset shard is empty: {prefix}")
                try:
                    decode_packed_row(dataset[0], row_index=0)
                except ValueError as error:
                    raise ValueError(f"Invalid packed SFT IndexedDataset shard '{prefix}': {error}") from error
            self.datasets.append(dataset)

        self.offsets = [0]
        for dataset in self.datasets:
            self.offsets.append(self.offsets[-1] + len(dataset))

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
        object_storage_cache_path: str | Path | None = None,
        object_storage_bin_chunk_nbytes: int = DEFAULT_OBJECT_STORAGE_BIN_CHUNK_NBYTES,
        **kwargs: object,
    ) -> None:
        """Initialize an indexed packed GPT SFT dataset."""
        self._path_spec = file_path
        self._mmap = mmap
        self._object_storage_config = object_storage_config
        self._object_storage_cache_path = object_storage_cache_path
        self._object_storage_bin_chunk_nbytes = object_storage_bin_chunk_nbytes
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
            self._path_spec,
            mmap=self._mmap,
            object_storage_config=self._object_storage_config,
            object_storage_cache_path=self._object_storage_cache_path,
            object_storage_bin_chunk_nbytes=self._object_storage_bin_chunk_nbytes,
        )

    def __getitem__(self, index: int) -> dict[str, np.ndarray | list[int]]:
        """Load and decode one packed training sample."""
        is_padding_sample = index < 0
        if self.samples_mapping is not None:
            mapped_index = self.samples_mapping[index]
            if isinstance(mapped_index, np.ndarray):
                mapped_index = mapped_index.reshape(-1)[0]
            elif isinstance(mapped_index, (tuple, list)):
                mapped_index = mapped_index[0]
            index = int(mapped_index)
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
