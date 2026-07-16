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

"""Declarative Hugging Face sources, named presets, and shared normalization."""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
from datasets import concatenate_datasets, load_dataset

from megatron.bridge.data.base import validate_declarative_mapping
from megatron.bridge.data.sources.hf_adapters import (
    adapt_hf_dataset,
    prepare_hf_dataset_for_adapter,
    validate_hf_dataset_adapter,
)


def _has_unset_data_file(value: Any) -> bool:
    """Return whether a declarative ``data_files`` value contains no usable path."""
    if value is None:
        return True
    if isinstance(value, (str, Path)):
        return not str(value).strip()
    if isinstance(value, (list, tuple)):
        return not value or any(_has_unset_data_file(item) for item in value)
    if isinstance(value, dict):
        return not value or any(_has_unset_data_file(item) for item in value.values())
    return False


@dataclass(kw_only=True)
class HFDatasetSourceConfig:
    """Serializable source selection for one Hugging Face dataset split.

    Exactly one source mode is required. ``dataset_name`` selects a built-in
    dataset preset that owns its Hub path, subset, and schema adapter.
    ``path_or_dataset`` selects a custom source; ``schema_adapter`` is optional
    when its rows already match the selected chat or prompt-completion
    preprocessing schema.
    """

    dataset_name: str | None = None
    path_or_dataset: str | None = None
    split: str | None = None
    subset: str | list[str] | None = None
    load_kwargs: dict[str, Any] | None = None
    schema_adapter: str | None = None
    adapter_kwargs: dict[str, Any] | None = None

    def validate(self) -> None:
        """Validate declarative source and adapter settings."""
        validate_declarative_mapping(self.load_kwargs, field_name="load_kwargs")
        validate_declarative_mapping(self.adapter_kwargs, field_name="adapter_kwargs")
        if self.load_kwargs is not None and "data_files" in self.load_kwargs:
            if _has_unset_data_file(self.load_kwargs["data_files"]):
                raise ValueError("load_kwargs.data_files must contain non-empty paths.")
        if self.split is not None and (not isinstance(self.split, str) or not self.split.strip()):
            raise ValueError("split must be non-empty when set.")
        has_named_source = self.dataset_name is not None
        has_custom_source = self.path_or_dataset is not None
        if has_named_source == has_custom_source:
            raise ValueError("Exactly one Hugging Face source must be set: dataset_name or path_or_dataset.")
        if has_named_source:
            if not isinstance(self.dataset_name, str) or not self.dataset_name.strip():
                raise ValueError("dataset_name must be a non-empty string.")
            if self.dataset_name not in _HF_DATASET_PRESETS:
                choices = ", ".join(sorted(_HF_DATASET_PRESETS))
                raise ValueError(f"Unknown Hugging Face dataset preset: {self.dataset_name}. Available: {choices}")
            if self.subset is not None or self.schema_adapter is not None:
                raise ValueError("Named dataset presets own subset and schema_adapter; do not override them.")
            preset = _HF_DATASET_PRESETS[self.dataset_name]
            adapter_kwargs = self.adapter_kwargs or {}
            missing_kwargs = [
                key
                for key in preset.required_adapter_kwargs
                if not isinstance(adapter_kwargs.get(key), (str, Path)) or not str(adapter_kwargs[key]).strip()
            ]
            if missing_kwargs:
                missing = ", ".join(missing_kwargs)
                raise ValueError(f"Dataset preset {self.dataset_name} requires adapter_kwargs: {missing}.")
            if self.split is not None:
                _resolve_preset_split(self.dataset_name, preset, self.split)
        else:
            if not isinstance(self.path_or_dataset, str) or not self.path_or_dataset.strip():
                raise ValueError("path_or_dataset must be a non-empty string.")
        if isinstance(self.subset, str) and not self.subset.strip():
            raise ValueError("subset must be non-empty when set.")
        if isinstance(self.subset, list) and (
            not self.subset or not all(isinstance(item, str) and item.strip() for item in self.subset)
        ):
            raise ValueError("subset lists must contain non-empty strings.")
        if self.subset is not None and not isinstance(self.subset, (str, list)):
            raise TypeError("subset must be a string, list of strings, or None.")
        if self.schema_adapter is not None and (
            not isinstance(self.schema_adapter, str) or not self.schema_adapter.strip()
        ):
            raise ValueError("schema_adapter must be non-empty when set.")
        adapter_name = (
            _HF_DATASET_PRESETS[self.dataset_name].schema_adapter
            if self.dataset_name is not None
            else self.schema_adapter
        )
        validate_hf_dataset_adapter(adapter_name)
        duplicate_load_keys = {"path", "name", "split"}.intersection(self.load_kwargs or {})
        if duplicate_load_keys:
            keys = ", ".join(sorted(duplicate_load_keys))
            raise ValueError(f"load_kwargs must not override source fields: {keys}.")

    def with_split(self, split: str) -> "HFDatasetSourceConfig":
        """Return a copy selecting another split expression."""
        return replace(self, split=split)


@dataclass(frozen=True, kw_only=True)
class _HFDatasetPreset:
    """Resolved physical source metadata for a built-in dataset."""

    path_or_dataset: str
    schema_adapter: str | None
    split: str = "train"
    subset: str | list[str] | None = None
    load_kwargs: dict[str, Any] | None = None
    adapter_kwargs: dict[str, Any] | None = None
    required_adapter_kwargs: tuple[str, ...] = ()
    split_aliases: dict[str, str] | None = None
    supported_splits: tuple[str, ...] | None = None


_HF_DATASET_PRESETS: dict[str, _HFDatasetPreset] = {
    "cord_v2": _HFDatasetPreset(
        path_or_dataset="naver-clova-ix/cord-v2",
        schema_adapter="cord_v2",
        supported_splits=("train", "validation", "test"),
    ),
    "cv17": _HFDatasetPreset(
        path_or_dataset="ysdede/commonvoice_17_tr_fixed",
        schema_adapter="cv17",
        supported_splits=("train", "validation", "test", "validated"),
    ),
    "gsm8k": _HFDatasetPreset(
        path_or_dataset="openai/gsm8k",
        subset="main",
        schema_adapter="gsm8k",
        supported_splits=("train", "test"),
    ),
    "llava_video_178k": _HFDatasetPreset(
        path_or_dataset="lmms-lab/LLaVA-Video-178K",
        subset="0_30_s_nextqa",
        split="open_ended",
        schema_adapter="llava_video_178k",
        required_adapter_kwargs=("video_root_path",),
        split_aliases={"train": "open_ended"},
        supported_splits=("open_ended",),
    ),
    "medpix": _HFDatasetPreset(
        path_or_dataset="mmoukouba/MedPix-VQA",
        schema_adapter="medpix",
        supported_splits=("train", "validation"),
    ),
    "openmathinstruct2": _HFDatasetPreset(
        path_or_dataset="nvidia/OpenMathInstruct-2",
        split="train_1M",
        schema_adapter="openmathinstruct2",
        supported_splits=("train", "train_1M", "train_2M", "train_5M"),
    ),
    "openmathinstruct2_thinking": _HFDatasetPreset(
        path_or_dataset="nvidia/OpenMathInstruct-2",
        split="train_1M",
        schema_adapter="openmathinstruct2_thinking",
        supported_splits=("train", "train_1M", "train_2M", "train_5M"),
    ),
    "raven": _HFDatasetPreset(
        path_or_dataset="HuggingFaceM4/the_cauldron",
        subset="raven",
        schema_adapter="raven",
        supported_splits=("train",),
    ),
    "rdr": _HFDatasetPreset(
        path_or_dataset="quintend/rdr-items",
        schema_adapter="rdr",
        supported_splits=("train",),
    ),
    "squad": _HFDatasetPreset(
        path_or_dataset="rajpurkar/squad",
        schema_adapter="squad",
        supported_splits=("train", "validation"),
    ),
    "tulu3": _HFDatasetPreset(
        path_or_dataset="allenai/tulu-3-sft-mixture",
        schema_adapter=None,
        supported_splits=("train",),
    ),
}


def _resolve_preset_split(dataset_name: str, preset: _HFDatasetPreset, split: str) -> str:
    resolved_components: list[str] = []
    for component in split.split("+"):
        base_split, slice_separator, slice_expression = component.partition("[")
        resolved_base_split = (preset.split_aliases or {}).get(base_split, base_split)
        if preset.supported_splits is not None and resolved_base_split not in preset.supported_splits:
            supported = ", ".join(preset.supported_splits)
            raise ValueError(f"Dataset preset {dataset_name} only supports split(s): {supported}.")
        resolved_components.append(f"{resolved_base_split}{slice_separator}{slice_expression}")
    return "+".join(resolved_components)


def hf_dataset_supports_split(source: HFDatasetSourceConfig, split: str) -> bool:
    """Return whether a source declares support for a logical split."""
    if not isinstance(split, str) or not split.strip():
        raise ValueError("split must be a non-empty string.")
    source.validate()
    if source.dataset_name is None:
        return True
    preset = _HF_DATASET_PRESETS[source.dataset_name]
    try:
        _resolve_preset_split(source.dataset_name, preset, split)
    except ValueError:
        return False
    return True


def resolve_hf_dataset_source(source: HFDatasetSourceConfig) -> HFDatasetSourceConfig:
    """Resolve a named preset or custom source to complete physical metadata."""
    source.validate()
    if source.dataset_name is None:
        return replace(source, split=source.split or "train")

    preset = _HF_DATASET_PRESETS[source.dataset_name]
    load_kwargs = {**(preset.load_kwargs or {}), **(source.load_kwargs or {})} or None
    adapter_kwargs = {**(preset.adapter_kwargs or {}), **(source.adapter_kwargs or {})} or None
    split = _resolve_preset_split(source.dataset_name, preset, source.split or preset.split)
    resolved = HFDatasetSourceConfig(
        path_or_dataset=preset.path_or_dataset,
        split=split,
        subset=preset.subset,
        load_kwargs=load_kwargs,
        schema_adapter=preset.schema_adapter,
        adapter_kwargs=adapter_kwargs,
    )
    resolved.validate()
    return resolved


def load_hf_dataset_source(source: HFDatasetSourceConfig) -> Any:
    """Load one declarative Hugging Face source without adapting its rows."""
    resolved = resolve_hf_dataset_source(source)
    assert resolved.path_or_dataset is not None
    assert resolved.split is not None
    kwargs = dict(resolved.load_kwargs or {})
    if isinstance(resolved.subset, list):
        datasets = [
            load_dataset(resolved.path_or_dataset, subset, split=resolved.split, **kwargs)
            for subset in resolved.subset
        ]
        return concatenate_datasets(datasets)
    if resolved.subset is None:
        return load_dataset(resolved.path_or_dataset, split=resolved.split, **kwargs)
    return load_dataset(resolved.path_or_dataset, resolved.subset, split=resolved.split, **kwargs)


def prepare_hf_dataset_sources(sources: Sequence[HFDatasetSourceConfig]) -> None:
    """Materialize Hugging Face caches once before distributed readers start.

    Hugging Face cache creation is not reliable when multiple distributed ranks
    concurrently build the same source on a shared filesystem. Rank zero loads
    each requested source first, then broadcasts completion so every rank reads
    an already-stable cache. Single-process callers need no preparation because
    their normal load has no competing writer.
    """
    if (
        not sources
        or not torch.distributed.is_available()
        or not torch.distributed.is_initialized()
        or torch.distributed.get_world_size() == 1
    ):
        return

    rank = torch.distributed.get_rank()
    materialization_error: Exception | None = None
    if rank == 0:
        try:
            for source in sources:
                load_hf_dataset_source(source)
        except Exception as error:
            materialization_error = error

    status = [
        None if materialization_error is None else f"{type(materialization_error).__name__}: {materialization_error}"
    ]
    torch.distributed.broadcast_object_list(status, src=0)
    if status[0] is not None:
        if materialization_error is not None:
            raise materialization_error
        raise RuntimeError(f"Rank zero failed to materialize a Hugging Face dataset source: {status[0]}")


def load_and_adapt_hf_dataset(source: HFDatasetSourceConfig) -> list[dict[str, Any]]:
    """Load a Hugging Face split and normalize it to canonical SFT rows."""
    resolved = resolve_hf_dataset_source(source)
    dataset = load_hf_dataset_source(source)
    dataset = prepare_hf_dataset_for_adapter(
        dataset,
        adapter_name=resolved.schema_adapter,
        adapter_kwargs=resolved.adapter_kwargs,
    )
    return adapt_hf_dataset(
        dataset,
        adapter_name=resolved.schema_adapter,
        adapter_kwargs=resolved.adapter_kwargs,
    )
