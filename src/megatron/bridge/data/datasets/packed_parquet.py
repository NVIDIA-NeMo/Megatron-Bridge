# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Packed Parquet dataset support for SFT training.

This module provides GPTSFTPackedParquetDataset, which reads packed sequence data
from Parquet files as an alternative to the NumPy-based GPTSFTPackedDataset.

Supports multiple files via:
- Single file: "data.idx.parquet", "shard_0.parquet"
- Glob pattern: "data*.idx.parquet", "shard_*.parquet"
- Directory: "/path/to/data/" (globs for *.parquet and *.pq)

Key functions:
- is_packed_parquet_spec(): Check if a spec refers to packed Parquet data
- resolve_packed_parquet_paths(): Resolve a spec to actual file paths
"""

from __future__ import annotations

import bisect
import glob
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from megatron.core.msc_utils import MultiStorageClientFeature

from megatron.bridge.data.datasets.sft import GPTSFTPackedDataset


if TYPE_CHECKING:
    from megatron.bridge.training.tokenizers.tokenizer import MegatronTokenizer

logger = logging.getLogger(__name__)

# Required columns in packed Parquet schema
REQUIRED_COLUMNS = {"input_ids", "seq_start_id", "loss_mask"}


def is_packed_parquet_file(path) -> bool:
    """Check if a path refers to a packed Parquet file or pattern.

    Args:
        path: A Path object or string path.

    Returns:
        True if the path ends with .idx.parquet or .idx.pq, or contains a glob
        pattern that would match such files.
    """
    name = str(path).lower()
    # Matches both direct files and glob patterns (e.g., "data*.idx.parquet")
    # since both end with the extension.
    return name.endswith(".idx.parquet") or name.endswith(".idx.pq")


def is_packed_parquet_spec(spec: str | Path) -> bool:
    """Check if a spec refers to a packed Parquet source (file, directory, or glob).

    This predicate reflects what the dataset loader supports in packed mode:
    - Single .parquet/.idx.parquet/.idx.pq files
    - Glob patterns ending in .parquet/.idx.parquet/.idx.pq
    - Directories containing parquet files

    Args:
        spec: A path specification (file, directory, or glob pattern).

    Returns:
        True if the spec could refer to packed Parquet data.
    """
    spec_str = str(spec).lower()

    # Check for parquet file extensions (including glob patterns)
    if spec_str.endswith(".parquet") or spec_str.endswith(".pq"):
        return True

    # Check for glob patterns containing parquet extension
    if "*" in spec_str or "?" in spec_str:
        # Extract the pattern part after the last glob character
        return ".parquet" in spec_str or ".pq" in spec_str

    # For directories, try to resolve to parquet files
    # This is more robust than is_dir() on distributed filesystems (Lustre, S3, etc.)
    try:
        resolved = _resolve_parquet_paths(str(spec))
        return len(resolved) > 0
    except ValueError:
        pass

    # Fallback: check if it's a directory using filesystem abstraction
    if MultiStorageClientFeature.is_enabled():
        msc = MultiStorageClientFeature.import_package()
        msc_path = msc.Path(str(spec))
        return msc_path.is_dir() if hasattr(msc_path, "is_dir") else False
    else:
        return Path(spec).is_dir()


def _lazy_import_pyarrow():
    """Lazily import pyarrow and raise a clear error if not installed."""
    try:
        import pyarrow
        import pyarrow.parquet as pq

        return pyarrow, pq
    except ImportError as e:
        raise ImportError(
            "pyarrow is required for packed Parquet datasets but is not installed. "
            "Please reinstall megatron-bridge or run: pip install pyarrow>=14.0.0"
        ) from e


def _is_parquet_file(path: str) -> bool:
    """Check if a path refers to any Parquet file.

    Args:
        path: A string path.

    Returns:
        True if the path ends with .parquet or .pq (case-insensitive).
    """
    name = path.lower()
    return name.endswith(".parquet") or name.endswith(".pq")


def _resolve_parquet_paths(file_path: str) -> list[str]:
    """Resolve a file path specification to a list of actual file paths.

    Supports:
    - Single file: "data.idx.parquet", "shard_0.parquet"
    - Glob pattern: "data*.idx.parquet", "shard_*.parquet"
    - Directory: "/path/to/data/" (globs for *.parquet and *.pq)

    Args:
        file_path: Path specification (file, glob pattern, or directory).

    Returns:
        Sorted list of resolved file paths.

    Raises:
        ValueError: If no matching files are found.
    """
    path_str = str(file_path)

    # Check if it's a glob pattern
    if "*" in path_str or "?" in path_str:
        if MultiStorageClientFeature.is_enabled():
            msc = MultiStorageClientFeature.import_package()
            # MSC glob support - normalize to strings immediately
            if hasattr(msc, "glob"):
                paths = [str(p) for p in msc.glob(path_str)]
            else:
                # Fallback: try to use msc.Path with glob
                # Use msc.Path to split parent/pattern to handle URIs correctly
                msc_full_path = msc.Path(path_str)
                parent = str(msc_full_path.parent) if hasattr(msc_full_path, "parent") else None
                pattern = msc_full_path.name if hasattr(msc_full_path, "name") else None

                if parent is not None and pattern is not None:
                    msc_parent_path = msc.Path(parent)
                    if hasattr(msc_parent_path, "glob"):
                        paths = [str(p) for p in msc_parent_path.glob(pattern)]
                    else:
                        raise ValueError(f"MSC backend does not support glob operations for pattern: {path_str}")
                else:
                    raise ValueError(f"MSC backend does not support glob operations for pattern: {path_str}")
        else:
            paths = glob.glob(path_str)

        # Filter to only parquet files (accepts both *.parquet and *.idx.parquet)
        paths = [p for p in paths if _is_parquet_file(p)]
        paths = sorted(paths)

        if not paths:
            raise ValueError(
                f"No Parquet files found matching pattern: {path_str}. Files must end with .parquet or .pq"
            )
        return paths

    # Check if it's a directory
    if MultiStorageClientFeature.is_enabled():
        msc = MultiStorageClientFeature.import_package()
        msc_path = msc.Path(path_str)
        is_dir = msc_path.is_dir() if hasattr(msc_path, "is_dir") else False
    else:
        is_dir = Path(path_str).is_dir()

    if is_dir:
        # Glob for parquet files in directory (accepts both *.parquet and *.idx.parquet)
        paths = []
        for ext in ["*.parquet", "*.pq"]:
            pattern = os.path.join(path_str, ext)
            if MultiStorageClientFeature.is_enabled():
                msc = MultiStorageClientFeature.import_package()
                if hasattr(msc, "glob"):
                    # Normalize to strings immediately
                    paths.extend([str(p) for p in msc.glob(pattern)])
                elif hasattr(msc.Path(path_str), "glob"):
                    paths.extend([str(p) for p in msc.Path(path_str).glob(ext)])
            else:
                paths.extend(glob.glob(pattern))

        paths = sorted(set(paths))

        if not paths:
            raise ValueError(f"No Parquet files found in directory: {path_str}. Files must end with .parquet or .pq")
        return paths

    # Single file - verify it exists
    if MultiStorageClientFeature.is_enabled():
        msc = MultiStorageClientFeature.import_package()
        exists = msc.Path(path_str).exists()
    else:
        exists = Path(path_str).exists()

    if not exists:
        raise ValueError(f"Packed Parquet file not found: {path_str}")

    return [path_str]


def resolve_packed_parquet_paths(spec: str | Path) -> list[str]:
    """Resolve a packed parquet spec to a list of shard file paths.

    Public wrapper around the internal _resolve_parquet_paths function.
    Use this to validate and resolve packed parquet specs before dataset creation.

    Supports:
    - Single file: "data.idx.parquet", "shard_0.parquet"
    - Glob pattern: "data*.idx.parquet", "shard_*.parquet"
    - Directory: "/path/to/data/" (globs for *.parquet and *.pq)

    Args:
        spec: Path specification (file, glob pattern, or directory).

    Returns:
        Sorted list of resolved file paths.

    Raises:
        ValueError: If no matching files are found.
    """
    return _resolve_parquet_paths(str(spec))


def write_packed_parquet(
    rows: list[dict],
    output_path: str | Path,
    row_group_size: int = 500,
) -> None:
    """Write packed sequence data to a Parquet file.

    Args:
        rows: List of dicts with keys 'input_ids', 'loss_mask', 'seq_start_id'.
              This is the output format of fill_packing_strategy().
        output_path: Path to write the Parquet file.
        row_group_size: Number of rows per row group (default 500).
    """
    pa, pq = _lazy_import_pyarrow()

    table = pa.table(
        {
            "input_ids": [row["input_ids"] for row in rows],
            "loss_mask": [row["loss_mask"] for row in rows],
            "seq_start_id": [row["seq_start_id"] for row in rows],
        }
    )

    if MultiStorageClientFeature.is_enabled():
        msc = MultiStorageClientFeature.import_package()
        buf = pa.BufferOutputStream()
        pq.write_table(table, buf, row_group_size=row_group_size)
        with msc.open(str(output_path), "wb") as f:
            f.write(buf.getvalue().to_pybytes())
    else:
        pq.write_table(table, str(output_path), row_group_size=row_group_size)


class GPTSFTPackedParquetDataset(GPTSFTPackedDataset):
    """Dataset for packed sequences stored in Parquet format.

    This class reads packed training data from Parquet files with the naming convention
    *.idx.parquet or *.idx.pq. It inherits from GPTSFTPackedDataset to reuse the
    collate_fn() and loss-mask semantics.

    Supports multiple files via:
    - Single file: "data.idx.parquet"
    - Glob pattern: "data*.idx.parquet" or "shard_*.idx.pq"
    - Directory: "/path/to/data/" (globs for *.idx.parquet and *.idx.pq)

    The Parquet file(s) must contain the following columns:
        - input_ids: list<int32> - Token IDs for the packed sequence
        - seq_start_id: list<int32> - Start offsets for each sub-sequence within the pack
        - loss_mask: list<int8> - Per-token loss mask (0 or 1), same length as input_ids

    Example:
        >>> # Single file
        >>> dataset = GPTSFTPackedParquetDataset(
        ...     file_path="packed_data.idx.parquet",
        ...     tokenizer=tokenizer,
        ... )
        >>> # Multiple files via glob
        >>> dataset = GPTSFTPackedParquetDataset(
        ...     file_path="data/shard_*.idx.parquet",
        ...     tokenizer=tokenizer,
        ... )
    """

    def __init__(
        self,
        file_path: str,
        tokenizer: "MegatronTokenizer",
        return_cu_seqlen: bool = True,
        pad_cu_seqlens: bool = False,
        pack_metadata_file_path: str | None = None,
        **kwargs,
    ):
        """Initialize the packed Parquet dataset.

        Args:
            file_path: Path to packed Parquet file(s). Supports:
                - Single file: "data.idx.parquet"
                - Glob pattern: "data*.idx.parquet"
                - Directory: "/path/to/data/"
            tokenizer: The tokenizer to use.
            return_cu_seqlen: Whether to return cu_seqlen for THD attention kernel.
            pad_cu_seqlens: Whether to pad cu_seqlens for cudagraphs compatibility.
            pack_metadata_file_path: Path to the metadata JSON file for pad_cu_seqlens.
            **kwargs: Additional arguments passed to parent class.
        """
        # Initialize Parquet-specific state before calling parent __init__
        # (parent calls _load_dataset which needs these)
        self._file_path_spec: str = file_path  # Original specification (may be glob)
        self._parquet_paths: list[str] = []  # Resolved list of files
        self._num_rows: int = 0  # Total rows across all files
        self._file_offsets: list[int] = []  # Cumulative row counts: [0, rows_file0, rows_file0+rows_file1, ...]
        self._file_row_group_offsets: list[list[int]] = []  # Row group offsets per file

        # Lazy reader state (opened in worker processes after fork)
        # Maps file_idx -> (ParquetFile, handle)
        self._parquet_files: dict[int, tuple] = {}
        self._cached_file_idx: int | None = None
        self._cached_row_group_id: int | None = None
        self._cached_row_group_table = None

        # Call parent __init__ which will call _load_dataset() and _build_samples_mapping()
        super().__init__(
            file_path=file_path,
            tokenizer=tokenizer,
            return_cu_seqlen=return_cu_seqlen,
            pad_cu_seqlens=pad_cu_seqlens,
            pack_metadata_file_path=pack_metadata_file_path,
            **kwargs,
        )

    def _load_dataset(self):
        """Load Parquet metadata from all files and validate schemas.

        This method:
        1. Resolves the file path specification to actual files
        2. Reads metadata from each file (not actual data)
        3. Validates schemas contain required columns
        4. Builds cumulative indices for efficient row lookups

        The actual Parquet files are opened lazily in _ensure_reader() to survive
        DataLoader worker forking.
        """
        pyarrow, pq = _lazy_import_pyarrow()

        # Resolve file paths
        self._parquet_paths = _resolve_parquet_paths(self._file_path_spec)

        logger.info(f"Resolved {len(self._parquet_paths)} packed Parquet file(s) from: {self._file_path_spec}")

        # Build cumulative offsets
        self._file_offsets = [0]
        self._file_row_group_offsets = []

        for file_idx, parquet_path in enumerate(self._parquet_paths):
            # Read metadata only (not actual data)
            if MultiStorageClientFeature.is_enabled():
                msc = MultiStorageClientFeature.import_package()
                handle = msc.open(str(parquet_path), "rb")
                try:
                    if hasattr(handle, "seekable") and handle.seekable():
                        metadata = pq.read_metadata(handle)
                        handle.seek(0)
                        schema = pq.read_schema(handle)
                    else:
                        content = handle.read()
                        buffer = pyarrow.BufferReader(content)
                        pf = pq.ParquetFile(buffer)
                        metadata = pf.metadata
                        schema = pf.schema_arrow
                finally:
                    handle.close()
            else:
                metadata = pq.read_metadata(parquet_path)
                schema = pq.read_schema(parquet_path)

            # Validate schema on every file to catch malformed shards early
            schema_columns = set(schema.names)
            missing_columns = REQUIRED_COLUMNS - schema_columns
            if missing_columns:
                raise ValueError(
                    f"Packed Parquet file '{parquet_path}' is missing required columns: {missing_columns}. "
                    f"Required columns are: {REQUIRED_COLUMNS}. "
                    f"Found columns: {schema_columns}"
                )

            # Build row group offsets for this file
            row_group_offsets = [0]
            for i in range(metadata.num_row_groups):
                row_group_offsets.append(row_group_offsets[-1] + metadata.row_group(i).num_rows)
            self._file_row_group_offsets.append(row_group_offsets)

            # Update cumulative file offset
            file_rows = metadata.num_rows
            self._file_offsets.append(self._file_offsets[-1] + file_rows)

            logger.debug(
                f"  File {file_idx}: {parquet_path}, {file_rows} rows in {metadata.num_row_groups} row groups"
            )

        self._num_rows = self._file_offsets[-1]

        # Validate dataset is not empty
        if self._num_rows == 0:
            raise ValueError(f"Packed Parquet dataset is empty (0 rows) for path: {self._file_path_spec}")

        logger.info(
            f"Loaded packed Parquet dataset: {self._num_rows} total rows across {len(self._parquet_paths)} file(s)"
        )

    @staticmethod
    def validate_row(idx: int, input_ids: list, loss_mask: list, seq_start_id: list) -> None:
        """Validate packed row invariants.

        This is NOT called in the training hot path for performance reasons.
        Use it during data preparation or for debugging.

        Args:
            idx: Row index (for error messages).
            input_ids: Token IDs for the packed sequence.
            loss_mask: Per-token loss mask.
            seq_start_id: Start offsets for each sub-sequence.

        Raises:
            ValueError: If any invariant is violated.
        """
        if len(loss_mask) != len(input_ids):
            raise ValueError(f"Row {idx}: loss_mask length ({len(loss_mask)}) != input_ids length ({len(input_ids)})")

        if not seq_start_id or seq_start_id[0] != 0:
            raise ValueError(
                f"Row {idx}: seq_start_id must start with 0, got {seq_start_id[:5] if seq_start_id else []}"
            )

        for i, start in enumerate(seq_start_id):
            if start >= len(input_ids):
                raise ValueError(f"Row {idx}: seq_start_id[{i}]={start} >= len(input_ids)={len(input_ids)}")
            if i > 0 and start < seq_start_id[i - 1]:
                raise ValueError(
                    f"Row {idx}: seq_start_id is not non-decreasing at index {i}: {seq_start_id[i - 1]} > {start}"
                )

    def _ensure_reader(self, file_idx: int):
        """Lazily open a Parquet file for reading.

        Args:
            file_idx: Index of the file in self._parquet_paths.

        This method is called before accessing data and creates the ParquetFile
        reader if it doesn't exist. This lazy initialization ensures the reader
        survives DataLoader worker forking (each worker creates its own readers).
        """
        if file_idx in self._parquet_files:
            return self._parquet_files[file_idx][0]

        pyarrow, pq = _lazy_import_pyarrow()
        parquet_path = self._parquet_paths[file_idx]

        if MultiStorageClientFeature.is_enabled():
            msc = MultiStorageClientFeature.import_package()
            handle = msc.open(str(parquet_path), "rb")

            if hasattr(handle, "seekable") and handle.seekable():
                pf = pq.ParquetFile(handle)
                self._parquet_files[file_idx] = (pf, handle)
            else:
                # MVP fallback: load entire file into memory for non-seekable streams
                logger.warning(f"MSC stream is not seekable, loading entire Parquet file into memory: {parquet_path}")
                content = handle.read()
                handle.close()
                buffer = pyarrow.BufferReader(content)
                pf = pq.ParquetFile(buffer)
                self._parquet_files[file_idx] = (pf, None)
        else:
            pf = pq.ParquetFile(parquet_path)
            self._parquet_files[file_idx] = (pf, None)

        return self._parquet_files[file_idx][0]

    def close(self) -> None:
        """Close all open Parquet file handles.

        This method should be called when the dataset is no longer needed to
        release file handles, especially when using MSC backends. It is also
        called automatically by __del__.
        """
        parquet_files = getattr(self, "_parquet_files", None)
        if parquet_files is None:
            return

        for file_idx, (pf, handle) in list(parquet_files.items()):
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass  # Best effort cleanup
            # Also close ParquetFile if it has a close method
            if hasattr(pf, "close"):
                try:
                    pf.close()
                except Exception:
                    pass

        self._parquet_files.clear()
        self._cached_row_group_table = None
        self._cached_file_idx = None
        self._cached_row_group_id = None

    def __del__(self) -> None:
        """Cleanup on deletion."""
        self.close()

    def _build_samples_mapping(self):
        """Build epoch-level sample mapping for shuffling.

        Mirrors GPTSFTPackedDataset._build_samples_mapping() exactly,
        using self._num_rows instead of len(self.indexed_dataset).
        """
        if self.max_num_samples is not None:
            dataset_len = self._num_rows
            max_num_epochs = np.ceil(self.max_num_samples / dataset_len)
            indices = np.arange(dataset_len)[None, :].repeat(max_num_epochs, axis=0)
            [np.random.shuffle(x) for x in indices]
            self.samples_mapping = indices.reshape(1, -1).squeeze()[: self.max_num_samples]
        else:
            self.samples_mapping = None

    def __len__(self):
        """Return the number of samples in the dataset."""
        if self.samples_mapping is not None:
            return len(self.samples_mapping)
        return self._num_rows

    def _locate_row(self, global_idx: int) -> tuple[int, int, int]:
        """Map a global row index to (file_idx, row_group_id, row_in_group).

        Args:
            global_idx: Global row index across all files.

        Returns:
            Tuple of (file_idx, row_group_id, row_in_group).
        """
        # Find which file contains this row
        file_idx = bisect.bisect_right(self._file_offsets, global_idx) - 1
        row_in_file = global_idx - self._file_offsets[file_idx]

        # Find which row group within the file
        row_group_offsets = self._file_row_group_offsets[file_idx]
        row_group_id = bisect.bisect_right(row_group_offsets, row_in_file) - 1
        row_in_group = row_in_file - row_group_offsets[row_group_id]

        return file_idx, row_group_id, row_in_group

    def __getitem__(self, idx: int) -> dict:
        """Get a packed sample by index.

        Args:
            idx: Sample index. If samples_mapping exists, this is mapped to the
                actual row index. Negative indices return samples with zeroed loss_mask.

        Returns:
            dict with keys:
                - input_ids: list[int] - Token IDs
                - seq_boundaries: list[int] - Sequence boundaries (derived from seq_start_id)
                - loss_mask: list[int] - Per-token loss mask
        """
        # Apply sample mapping if exists
        if self.samples_mapping is not None:
            idx = self.samples_mapping[idx]

        # Handle negative indices (padding samples)
        # Use wrap-around semantics matching parent GPTSFTPackedDataset behavior
        is_padding_sample = idx < 0
        if is_padding_sample:
            idx = self._num_rows + idx  # -1 -> last row, -N -> Nth from end

        # Locate the row across files and row groups
        file_idx, row_group_id, row_in_group = self._locate_row(idx)

        # Ensure reader is initialized for this file
        pf = self._ensure_reader(file_idx)

        # Read row group with caching
        cache_key = (file_idx, row_group_id)
        if (self._cached_file_idx, self._cached_row_group_id) != cache_key:
            self._cached_row_group_table = pf.read_row_group(
                row_group_id, columns=["input_ids", "seq_start_id", "loss_mask"]
            )
            self._cached_file_idx = file_idx
            self._cached_row_group_id = row_group_id

        # Extract row values
        table = self._cached_row_group_table
        input_ids = table.column("input_ids")[row_in_group].as_py()
        seq_start_id = table.column("seq_start_id")[row_in_group].as_py()
        loss_mask = table.column("loss_mask")[row_in_group].as_py()

        # Compute derived field
        seq_boundaries = seq_start_id + [len(input_ids)]

        # For padding samples, zero out the loss mask
        if is_padding_sample:
            loss_mask = [0] * len(loss_mask)

        return {
            "input_ids": input_ids,
            "seq_boundaries": seq_boundaries,
            "loss_mask": loss_mask,
        }
