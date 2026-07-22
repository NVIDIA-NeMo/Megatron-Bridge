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

"""Declarative configuration for offline GPT SFT sequence packing."""

import warnings
from dataclasses import dataclass
from pathlib import Path

from megatron.core.msc_utils import MultiStorageClientFeature

from megatron.bridge.data.packing.paths import is_packed_parquet_spec, resolve_packed_parquet_paths


@dataclass
class PackedSequenceSpecs:
    """Settings and optional artifact paths for offline sequence packing."""

    packed_sequence_size: int = -1
    tokenizer_model_name: str | None = None
    num_tokenizer_workers: int = -1
    packed_train_data_path: str | Path | None = None
    packed_val_data_path: str | Path | None = None
    packed_metadata_path: str | Path | None = None
    pad_cu_seqlens: bool = False
    pad_seq_to_mult: int | None = 1

    def __post_init__(self) -> None:
        """Validate alignment settings and any explicitly supplied artifacts."""
        if self.packed_train_data_path is not None:
            self._validate_packed_path("packed_train_data_path", self.packed_train_data_path)
        if self.packed_val_data_path is not None:
            self._validate_packed_path("packed_val_data_path", self.packed_val_data_path)
        if self.pad_seq_to_mult is not None and self.pad_seq_to_mult <= 0:
            raise ValueError("pad_seq_to_mult must be a positive integer when provided.")

    def _validate_packed_path(self, attr_name: str, path_value: str | Path) -> None:
        """Validate an explicitly supplied packed artifact path."""
        path_str = str(path_value)
        if path_str.lower().endswith(".npy"):
            warnings.warn(
                "The .npy packed sequence format is deprecated and will be removed in the next release. "
                f"Please use packed parquet format instead. Path: {path_str}",
                DeprecationWarning,
                stacklevel=2,
            )
            if MultiStorageClientFeature.is_enabled():
                msc = MultiStorageClientFeature.import_package()
                path_obj = msc.Path(path_str)
            else:
                path_obj = Path(path_str)
            if not path_obj.exists():
                raise FileNotFoundError(f"{attr_name} file does not exist: {path_str}")
            setattr(self, attr_name, path_obj)
            return

        if is_packed_parquet_spec(path_str):
            try:
                if not resolve_packed_parquet_paths(path_str):
                    raise FileNotFoundError(f"{attr_name} resolved to no files: {path_str}")
            except ValueError as error:
                raise FileNotFoundError(f"{attr_name} could not be resolved: {path_str}. Error: {error}") from error
            setattr(self, attr_name, path_str)
            return

        raise ValueError(
            f"{attr_name} must be a .npy file or a packed parquet spec "
            f"(file/directory/glob ending in .parquet or .pq): {path_str}"
        )
