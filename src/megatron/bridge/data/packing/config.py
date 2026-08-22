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

import math
import numbers
import warnings
from dataclasses import dataclass
from pathlib import Path

from megatron.core.msc_utils import MultiStorageClientFeature

from megatron.bridge.data.packing.paths import is_packed_parquet_spec, resolve_packed_parquet_paths


@dataclass
class PackedSequenceSpecs:
    """Settings and optional artifact paths for offline sequence packing."""

    packed_sequence_size: int = -1
    max_single_sequence_length: int | None = None
    tokenizer_model_name: str | None = None
    num_tokenizer_workers: int = -1
    packed_train_data_path: str | Path | None = None
    packed_train_data_blend: tuple[list[str | Path], list[float]] | None = None
    packed_val_data_path: str | Path | None = None
    packed_metadata_path: str | Path | None = None
    pad_cu_seqlens: bool = False
    """Pad cumulative sequence boundaries for full-iteration, whole-layer, or attention-scoped CUDA graphs."""
    pad_seq_to_mult: int | None = 1

    def __post_init__(self) -> None:
        """Validate alignment settings and any explicitly supplied artifacts."""
        if self.packed_train_data_path is not None and self.packed_train_data_blend is not None:
            raise ValueError("Set either packed_train_data_path or packed_train_data_blend, not both.")
        if self.packed_train_data_path is not None:
            self._validate_packed_path("packed_train_data_path", self.packed_train_data_path)
        if self.packed_train_data_blend is not None:
            self._validate_packed_train_data_blend()
        if self.packed_val_data_path is not None:
            self._validate_packed_path("packed_val_data_path", self.packed_val_data_path)
        if self.pad_seq_to_mult is not None and self.pad_seq_to_mult <= 0:
            raise ValueError("pad_seq_to_mult must be a positive integer when provided.")
        if self.max_single_sequence_length is not None:
            if self.max_single_sequence_length <= 0:
                raise ValueError("max_single_sequence_length must be a positive integer when provided.")
            if self.packed_sequence_size > 0 and self.max_single_sequence_length > self.packed_sequence_size:
                raise ValueError("max_single_sequence_length cannot exceed packed_sequence_size.")

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

    def _validate_packed_train_data_blend(self) -> None:
        """Validate weighted packed Parquet training sources."""
        blend = self.packed_train_data_blend
        assert blend is not None
        if not isinstance(blend, (tuple, list)) or len(blend) != 2:
            raise TypeError("packed_train_data_blend must be a (sources, weights) pair.")

        sources, weights = blend
        if isinstance(sources, (str, Path)) or not isinstance(sources, (tuple, list)):
            raise TypeError("packed_train_data_blend sources must be a list of packed Parquet specs.")
        if not isinstance(weights, (tuple, list)):
            raise TypeError("packed_train_data_blend weights must be a list of positive numbers.")
        if len(sources) < 2:
            raise ValueError("packed_train_data_blend requires at least two sources.")
        if len(sources) != len(weights):
            raise ValueError("packed_train_data_blend sources and weights must have the same length.")

        normalized_sources: list[str] = []
        normalized_weights: list[float] = []
        for source_index, source in enumerate(sources):
            source_str = str(source)
            if not is_packed_parquet_spec(source_str):
                raise ValueError(
                    "packed_train_data_blend only supports packed Parquet sources; "
                    f"source {source_index} is invalid: {source_str}"
                )
            try:
                if not resolve_packed_parquet_paths(source_str):
                    raise FileNotFoundError(f"source {source_index} resolved to no files: {source_str}")
            except ValueError as error:
                raise FileNotFoundError(
                    f"packed_train_data_blend source {source_index} could not be resolved: "
                    f"{source_str}. Error: {error}"
                ) from error
            normalized_sources.append(source_str)

        for weight_index, weight in enumerate(weights):
            if isinstance(weight, bool) or not isinstance(weight, numbers.Real):
                raise TypeError(f"packed_train_data_blend weight {weight_index} must be a number, got {weight!r}.")
            normalized_weight = float(weight)
            if not math.isfinite(normalized_weight) or normalized_weight <= 0.0:
                raise ValueError(
                    f"packed_train_data_blend weight {weight_index} must be finite and greater than 0, got {weight!r}."
                )
            normalized_weights.append(normalized_weight)

        try:
            weight_sum = math.fsum(normalized_weights)
        except OverflowError as error:
            raise ValueError("packed_train_data_blend weights must have a finite sum.") from error
        if not math.isfinite(weight_sum):
            raise ValueError("packed_train_data_blend weights must have a finite sum.")

        self.packed_train_data_blend = (normalized_sources, normalized_weights)
