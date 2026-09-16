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

"""Ragged on-disk store for materialized tokenized SFT samples.

Stores variable-length per-sample arrays (e.g. ``input_ids``, ``loss_mask``)
as flat 1-D arrays plus an offsets array, so the materialized dataset never
has to exist in host memory as a collection of Python objects. This mirrors
the layout of Megatron-Core's ``.idx``/``.bin`` indexed datasets: offsets act
as the index and the flat data acts as the binary payload.

On-disk layout (one directory)::

    store_dir/
    ├── meta.json                  # version, column names, dtypes, item count
    ├── input_ids.npy              # flat int array, all samples concatenated
    ├── input_ids.offsets.npy      # int64, length N+1; sample i = data[offsets[i]:offsets[i+1]]
    ├── loss_mask.npy              # flat bool/int array (same ragged structure)
    ├── loss_mask.offsets.npy
    └── ...

All file I/O goes through :class:`~megatron.core.msc_utils.MultiStorageClientFeature`
when enabled, mirroring the rest of the packing pipeline.
"""

import json
import logging
import shutil
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from megatron.core.msc_utils import MultiStorageClientFeature


logger = logging.getLogger(__name__)

_META_FILE = "meta.json"
_FORMAT_VERSION = "1"
_CHUNK_DIR = "chunks"


def _save(path: str | Path, array: np.ndarray) -> None:
    if MultiStorageClientFeature.is_enabled():
        msc = MultiStorageClientFeature.import_package()
        msc.numpy.save(str(path), array)
    else:
        np.save(str(path), array, allow_pickle=False)


def _load(path: str | Path) -> np.ndarray:
    if MultiStorageClientFeature.is_enabled():
        msc = MultiStorageClientFeature.import_package()
        return msc.numpy.load(str(path), allow_pickle=False)
    return np.load(str(path), allow_pickle=False)


def _memmap(path: str | Path, dtype, mode: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    """Open a .npy file as a memory-mapped array.

    For read mode, ``np.load(mmap_mode="r")`` is used because it understands
    the .npy header; raw ``np.memmap`` would treat the header bytes as data.
    """
    if mode == "r":
        if MultiStorageClientFeature.is_enabled():
            msc = MultiStorageClientFeature.import_package()
            return msc.numpy.load(str(path), mmap_mode="r")
        return np.load(str(path), mmap_mode="r")
    if MultiStorageClientFeature.is_enabled():
        msc = MultiStorageClientFeature.import_package()
        return msc.numpy.memmap(str(path), dtype=dtype, mode=mode, shape=shape)
    return np.memmap(str(path), dtype=dtype, mode=mode, shape=shape)


def _create_data_file(path: str | Path, dtype, total: int, segments: list[str | Path]) -> None:
    """Stream segment files into a single .npy-formatted memmap.

    Segments are bounded by chunk size, so loading one segment at a time keeps
    peak RAM bounded regardless of total dataset size.  Each segment is read
    via ``np.load(mmap_mode="r")`` and copied into the output memmap, so the
    full token array is never resident in RAM.

    When MSC is enabled, ``open_memmap`` is not available on remote stores;
    we fall back to loading each segment and concatenating in memory before
    a single ``msc.numpy.save``.  This is no worse than the pre-ragged
    pipeline's ``np.concatenate`` of all chunks.
    """
    if MultiStorageClientFeature.is_enabled():
        msc = MultiStorageClientFeature.import_package()
        arrays = [_load(segment_path) for segment_path in segments]
        msc.numpy.save(str(path), np.concatenate(arrays))
        return
    data = np.lib.format.open_memmap(str(path), mode="w+", dtype=dtype, shape=(total,))
    position = 0
    for segment_path in segments:
        segment = np.load(str(segment_path), mmap_mode="r")
        data[position : position + len(segment)] = segment[:]
        position += len(segment)
        del segment
    data.flush()
    del data


def _as_flat_array(value) -> np.ndarray:
    """Convert a per-item value (list / torch tensor / numpy array) to a 1-D numpy array."""
    if isinstance(value, np.ndarray):
        return value if value.ndim == 1 else value.reshape(-1)
    array = np.asarray(value)
    return array if array.ndim == 1 else array.reshape(-1)


class FlatColumn:
    """A single ragged column backed by a flat data array and per-item offsets.

    Each sample ``i`` occupies ``data[offsets[i] : offsets[i+1]]``.
    ``offsets`` has length ``num_items + 1``; ``offsets[0]`` is always 0.

    For an absent sample (e.g. an item that has no ``loss_mask``), the
    entry is stored as zero-length — ``offsets[i] == offsets[i+1]`` — so
    ``lengths[i] > 0`` doubles as a presence flag.
    """

    def __init__(self, data: np.ndarray, offsets: np.ndarray):
        self.data = data
        self.offsets = offsets

    @property
    def num_items(self) -> int:
        return len(self.offsets) - 1

    @property
    def lengths(self) -> np.ndarray:
        """Per-item lengths (``np.diff(offsets)``)."""
        return np.diff(self.offsets)

    def get(self, index: int) -> np.ndarray:
        """Return a copy of sample *index* as a 1-D array."""
        start = int(self.offsets[index])
        end = int(self.offsets[index + 1])
        return self.data[start:end]

    def gather(self, indices: np.ndarray) -> np.ndarray:
        """Stack the given items into a 2-D array of shape ``(len(indices), L)``.

        All referenced items must share the same length ``L``, which holds
        for groups produced by ``create_hist_from_lengths`` (items are grouped
        by runtime sequence length).

        Raises:
            ValueError: if the items have differing lengths.
        """
        lengths = self.lengths[indices]
        if len(lengths) and not np.all(lengths == lengths[0]):
            raise ValueError(
                f"gather requires items of equal length, got lengths min={int(lengths.min())} max={int(lengths.max())}"
            )
        out = np.empty((len(indices), int(lengths[0]) if len(lengths) else 0), dtype=self.data.dtype)
        for row, index in enumerate(indices):
            start = int(self.offsets[index])
            end = int(self.offsets[index + 1])
            out[row] = self.data[start:end]
        return out


class RaggedStore:
    """Read-only view over a materialized ragged store on disk.

    Created by ``RaggedStoreWriter.finalize()`` or reopened via
    ``RaggedStore.open(path)``.  All column data is accessed through
    memory-mapped arrays, so the full dataset is never loaded into RAM.
    """

    def __init__(self, path: str | Path, columns: dict[str, FlatColumn]):
        self.path = Path(path)
        self.columns = columns

    @classmethod
    def open(cls, path: str | Path) -> "RaggedStore":
        """Reopen a previously finalized store from disk."""
        path = Path(path)
        if MultiStorageClientFeature.is_enabled():
            msc = MultiStorageClientFeature.import_package()
            with msc.open(str(path / _META_FILE), "r") as f:
                meta = json.load(f)
        else:
            with open(path / _META_FILE) as f:
                meta = json.load(f)
        columns = {}
        for name in meta["columns"]:
            data = _memmap(path / f"{name}.npy", dtype=meta["dtypes"][name], mode="r")
            offsets = _memmap(path / f"{name}.offsets.npy", dtype=np.int64, mode="r")
            columns[name] = FlatColumn(data, offsets)
        return cls(path, columns)

    def __contains__(self, name: str) -> bool:
        return name in self.columns

    def column(self, name: str) -> FlatColumn:
        return self.columns[name]

    def close(self) -> None:
        """Release memmaps and remove the backing directory."""
        for column in self.columns.values():
            data = column.data
            offsets = column.offsets
            mmap_obj = getattr(data, "_mmap", None)
            if mmap_obj is not None:
                mmap_obj.close()
            mmap_obj = getattr(offsets, "_mmap", None)
            if mmap_obj is not None:
                mmap_obj.close()
        self.columns = {}
        shutil.rmtree(self.path, ignore_errors=True)

    @property
    def num_items(self) -> int:
        for column in self.columns.values():
            return column.num_items
        return 0

    def input_ids_lengths(self) -> np.ndarray:
        """Per-sample stored token counts (``len(input_ids)`` for each item)."""
        return self.columns["input_ids"].lengths

    def __len__(self) -> int:
        return self.num_items

    def __getitem__(self, index: int) -> dict:
        """Return one sample as a dict of arrays, omitting absent (zero-length) columns."""
        sample = {}
        for name, column in self.columns.items():
            if column.lengths[index] > 0:
                sample[name] = column.get(index)
        return sample

    def __iter__(self):
        for index in range(self.num_items):
            yield self[index]


class _ColumnWriter:
    """Per-column writer that buffers items in RAM and flushes bounded segments to disk.

    Lifecycle::

        append(item_0)  →  append(item_1)  →  ...  →  flush()  →  append(...)  →  finalize()
          │                                          │
          └─ buffers in RAM (up to chunk_size)       └─ writes a segment .npy file, clears buffer

    finalize() concatenates all segments into a single .npy via streaming copy
    (``_create_data_file``), writes the offsets array, and returns a ``FlatColumn``
    backed by memmaps.
    """

    def __init__(self, name: str, chunk_dir: Path, chunk_size: int):
        self.name = name
        self._chunk_dir = chunk_dir
        self._chunk_size = chunk_size
        # RAM buffer: per-item arrays waiting to be flushed as one segment.
        self._buffer: list[np.ndarray] = []
        self._buffer_items: int = 0
        # Segment file names (relative to chunk_dir), populated by flush().
        self._segments: list[str] = []
        # Per-item end offsets into the future flat array.  offsets[0] = 0;
        # offsets[i+1] = offsets[i] + len(item_i).  Length = num_items + 1.
        self._offsets: list[int] = [0]
        # Running cursor = total tokens appended so far (end of last item).
        self._cursor: int = 0
        # Inferred from the first non-None value; used to allocate the output memmap.
        self._dtype: np.dtype | None = None

    def append(self, value) -> None:
        """Append one item's value for this column.

        ``value`` may be ``None`` (absent), in which case a zero-length entry
        is recorded so that ``lengths > 0`` can later serve as a presence flag.
        """
        if value is None:
            # Absent: record zero-length entry without advancing the cursor.
            self._offsets.append(self._cursor)
            return
        array = _as_flat_array(value)
        if self._dtype is None:
            self._dtype = array.dtype
        self._buffer.append(array)
        self._cursor += len(array)
        self._offsets.append(self._cursor)
        self._buffer_items += 1
        if self._buffer_items >= self._chunk_size:
            self.flush()

    def flush(self) -> None:
        """Write buffered arrays as a single concatenated segment file."""
        if not self._buffer:
            return
        segment = np.concatenate(self._buffer)
        segment_path = self._chunk_dir / f"{self.name}_seg_{len(self._segments):06d}.npy"
        _save(segment_path, segment)
        self._segments.append(segment_path.name)
        self._buffer = []
        self._buffer_items = 0

    def finalize(self, store_dir: Path) -> FlatColumn:
        """Flush remaining buffer, write offsets + data, and return a read-only FlatColumn."""
        self.flush()
        offsets = np.asarray(self._offsets, dtype=np.int64)
        _save(store_dir / f"{self.name}.offsets.npy", offsets)
        offsets_mm = _memmap(store_dir / f"{self.name}.offsets.npy", dtype=np.int64, mode="r")
        dtype = self._dtype if self._dtype is not None else np.dtype(np.float32)
        if not self._segments:
            # No data was ever appended for this column (e.g. empty dataset).
            data_path = store_dir / f"{self.name}.npy"
            _save(data_path, np.empty(0, dtype=dtype))
            return FlatColumn(_memmap(data_path, dtype=dtype, mode="r"), offsets_mm)
        # Compute total length and stream segments into the final .npy file.
        total = 0
        segments = []
        for segment_name in self._segments:
            segment = _load(self._chunk_dir / segment_name)
            total += len(segment)
            segments.append(self._chunk_dir / segment_name)
            del segment
        data_path = store_dir / f"{self.name}.npy"
        _create_data_file(data_path, dtype, total, segments)
        return FlatColumn(_memmap(data_path, dtype=dtype, mode="r"), offsets_mm)


class RaggedStoreWriter:
    """Writes items into a ragged store, flushing bounded chunks to disk.

    Items are dicts of per-sample arrays.  A column is treated as absent for an
    item when its value is ``None`` or the key is missing; absent values are
    stored as zero-length entries, so ``FlatColumn.lengths > 0`` doubles as a
    presence flag.  This matches the packing pipeline's fallback semantics where
    ``loss_mask`` may be absent and ``answer_start_idx`` is used instead.

    Usage::

        writer = RaggedStoreWriter(path, schema=("input_ids", "loss_mask"), chunk_size=5000)
        for item in dataset:
            writer.append(item)          # or writer.append_many(batch)
        store = writer.finalize()        # → RaggedStore (read-only, memmap-backed)
        ...
        store.close()                    # removes the backing directory
    """

    def __init__(self, path: str | Path, schema: Sequence[str], chunk_size: int = 5000):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._chunk_dir = self.path / _CHUNK_DIR
        self._chunk_dir.mkdir(exist_ok=True)
        self._schema = schema
        self._chunk_size = chunk_size
        # One _ColumnWriter per column name in the schema.
        self._writers = {name: _ColumnWriter(name, self._chunk_dir, chunk_size) for name in schema}
        self._num_items = 0

    def append(self, item: dict) -> None:
        """Append a single item, dispatching each value to its column writer."""
        for name, writer in self._writers.items():
            writer.append(item.get(name))
        self._num_items += 1

    def append_many(self, items) -> None:
        """Append a batch of items."""
        for item in items:
            self.append(item)

    def finalize(self) -> RaggedStore:
        """Finalize all columns, write meta.json, clean up segment files.

        Returns:
            A read-only ``RaggedStore`` backed by memory-mapped arrays.
        """
        columns = {name: writer.finalize(self.path) for name, writer in self._writers.items()}
        meta = {
            "version": _FORMAT_VERSION,
            "num_items": self._num_items,
            "columns": list(columns.keys()),
            "dtypes": {name: str(column.data.dtype) for name, column in columns.items()},
        }
        if MultiStorageClientFeature.is_enabled():
            msc = MultiStorageClientFeature.import_package()
            with msc.open(str(self.path / _META_FILE), "w") as f:
                json.dump(meta, f)
        else:
            with open(self.path / _META_FILE, "w") as f:
                json.dump(meta, f)
        # Clean up segment files now that the final .npy files exist.
        for column_writer in self._writers.values():
            for segment_name in column_writer._segments:
                try:
                    (self._chunk_dir / segment_name).unlink()
                except OSError:
                    pass
        try:
            self._chunk_dir.rmdir()
        except OSError:
            pass
        return RaggedStore(self.path, columns)
