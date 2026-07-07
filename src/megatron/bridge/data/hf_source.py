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

"""Declarative Hugging Face sources and shared loading/normalization."""

from dataclasses import dataclass, replace
from typing import Any

from datasets import concatenate_datasets, load_dataset

from megatron.bridge.data.base import validate_declarative_mapping
from megatron.bridge.data.hf_datasets.adapters import adapt_hf_dataset, prepare_hf_dataset_for_adapter


@dataclass(kw_only=True)
class HFDatasetSourceConfig:
    """Serializable source selection for one Hugging Face dataset split.

    ``schema_adapter`` is optional when rows already contain ``messages``,
    ``conversation``, or legacy ``conversations``. Dataset-specific schemas such
    as SQuAD, CORD, or CommonVoice select a registered adapter explicitly.
    """

    path_or_dataset: str
    split: str = "train"
    subset: str | list[str] | None = None
    load_kwargs: dict[str, Any] | None = None
    schema_adapter: str | None = None
    adapter_kwargs: dict[str, Any] | None = None

    def validate(self) -> None:
        """Validate declarative source and adapter settings."""
        if not self.path_or_dataset.strip():
            raise ValueError("path_or_dataset must be a non-empty string.")
        if not self.split.strip():
            raise ValueError("split must be a non-empty string.")
        if isinstance(self.subset, str) and not self.subset.strip():
            raise ValueError("subset must be non-empty when set.")
        if isinstance(self.subset, list) and (
            not self.subset or not all(isinstance(item, str) and item.strip() for item in self.subset)
        ):
            raise ValueError("subset lists must contain non-empty strings.")
        if self.subset is not None and not isinstance(self.subset, (str, list)):
            raise TypeError("subset must be a string, list of strings, or None.")
        if self.schema_adapter is not None and not self.schema_adapter.strip():
            raise ValueError("schema_adapter must be non-empty when set.")
        validate_declarative_mapping(self.load_kwargs, field_name="load_kwargs")
        validate_declarative_mapping(self.adapter_kwargs, field_name="adapter_kwargs")

    def with_split(self, split: str) -> "HFDatasetSourceConfig":
        """Return a copy selecting another split expression."""
        return replace(self, split=split)


def load_hf_dataset_source(source: HFDatasetSourceConfig) -> Any:
    """Load one declarative Hugging Face source without adapting its rows."""
    source.validate()
    kwargs = dict(source.load_kwargs or {})
    if isinstance(source.subset, list):
        datasets = [
            load_dataset(source.path_or_dataset, subset, split=source.split, **kwargs) for subset in source.subset
        ]
        return concatenate_datasets(datasets)
    if source.subset is None:
        return load_dataset(source.path_or_dataset, split=source.split, **kwargs)
    return load_dataset(source.path_or_dataset, source.subset, split=source.split, **kwargs)


def load_and_adapt_hf_dataset(source: HFDatasetSourceConfig) -> list[dict[str, Any]]:
    """Load a Hugging Face split and normalize it to canonical SFT rows."""
    dataset = load_hf_dataset_source(source)
    dataset = prepare_hf_dataset_for_adapter(
        dataset,
        adapter_name=source.schema_adapter,
        adapter_kwargs=source.adapter_kwargs,
    )
    return adapt_hf_dataset(
        dataset,
        adapter_name=source.schema_adapter,
        adapter_kwargs=source.adapter_kwargs,
    )
