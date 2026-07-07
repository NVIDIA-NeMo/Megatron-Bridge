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

"""Deprecated Hugging Face text SFT provider compatibility adapter."""

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from megatron.bridge.data.base import DatasetBuildContext, DatasetProvider
from megatron.bridge.data.builders.finetuning_dataset import GPTSFTDatasetConfig
from megatron.bridge.data.datasets.packed_sequence import PackedSequenceSpecs
from megatron.bridge.data.hf_source import HFDatasetSourceConfig


# =============================================================================
# Deprecated compatibility API
# =============================================================================


_LEGACY_TEXT_MAKER_DEFAULTS: dict[str, dict[str, Any]] = {
    "gsm8k": {"path_or_dataset": "openai/gsm8k", "subset": "main", "split": "train"},
    "openmathinstruct2": {"path_or_dataset": "nvidia/OpenMathInstruct-2", "split": "train_1M"},
    "openmathinstruct2_thinking": {
        "path_or_dataset": "nvidia/OpenMathInstruct-2",
        "split": "train_1M",
    },
    "squad": {"path_or_dataset": "rajpurkar/squad", "split": "train"},
}


def _get_gpt_sft_dataset_builder() -> type[Any]:
    from megatron.bridge.data.builders.finetuning_dataset import GPTSFTDatasetBuilder

    return GPTSFTDatasetBuilder


@dataclass(kw_only=True)
class HFTextSFTDatasetProvider(DatasetProvider):
    """Deprecated adapter from the former provider API to Config + Builder.

    Use :class:`GPTSFTDatasetConfig` with an :class:`HFDatasetSourceConfig`
    source in new code.
    """

    seq_length: int
    maker_name: str
    maker_kwargs: dict[str, Any] | None = None
    val_maker_kwargs: dict[str, Any] | None = None
    test_maker_kwargs: dict[str, Any] | None = None
    dataset_root: str | Path | None = None
    seed: int = 5678
    memmap_workers: int = 1
    max_train_samples: int | None = None
    enable_offline_packing: bool = False
    offline_packing_specs: PackedSequenceSpecs | None = None
    dataset_kwargs: dict[str, Any] | None = None
    val_proportion: float | None = None
    do_validation: bool = True
    do_test: bool = True
    rewrite: bool = False
    dataloader_type: Literal["single", "cyclic", "batch", "external"] | None = "batch"

    def __post_init__(self) -> None:
        warnings.warn(
            "HFTextSFTDatasetProvider is deprecated; use GPTSFTDatasetConfig with HFDatasetSourceConfig instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    def _to_config(self) -> GPTSFTDatasetConfig:
        adapter_name = self.maker_name.removeprefix("make_").removesuffix("_dataset")
        maker_kwargs = dict(_LEGACY_TEXT_MAKER_DEFAULTS.get(adapter_name, {}))
        maker_kwargs.update(self.maker_kwargs or {})
        path_or_dataset = str(maker_kwargs.pop("path_or_dataset", self.maker_name))
        split = str(maker_kwargs.pop("split", "train"))
        subset = maker_kwargs.pop("subset", None)
        adapter_keys = {
            "audio_column",
            "conversation_column",
            "conversations_column",
            "messages_column",
            "prompt",
            "remove_text_spaces",
            "text_column",
            "video_root_path",
        }
        adapter_kwargs = {key: maker_kwargs.pop(key) for key in list(maker_kwargs) if key in adapter_keys}
        return GPTSFTDatasetConfig(
            seq_length=self.seq_length,
            hf_dataset=HFDatasetSourceConfig(
                path_or_dataset=path_or_dataset,
                split=split,
                subset=subset,
                load_kwargs=maker_kwargs or None,
                schema_adapter=adapter_name,
                adapter_kwargs=adapter_kwargs or None,
            ),
            hf_validation_dataset=(
                self._split_source(self.val_maker_kwargs, default_split="validation") if self.do_validation else None
            ),
            hf_test_dataset=(
                self._split_source(self.test_maker_kwargs, default_split="test") if self.do_test else None
            ),
            hf_output_root=self.dataset_root,
            hf_validation_proportion=(
                self.val_proportion if self.do_validation and self.val_maker_kwargs is None else None
            ),
            hf_rewrite=self.rewrite,
            seed=self.seed,
            memmap_workers=self.memmap_workers,
            max_train_samples=self.max_train_samples,
            enable_offline_packing=self.enable_offline_packing,
            offline_packing_specs=self.offline_packing_specs,
            dataset_kwargs=self.dataset_kwargs,
            do_validation=self.do_validation,
            do_test=self.do_test,
            dataloader_type=self.dataloader_type,
            num_workers=self.num_workers,
            data_sharding=self.data_sharding,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            persistent_workers=self.persistent_workers,
            trust_remote_code=self.trust_remote_code,
        )

    def _split_source(self, kwargs: dict[str, Any] | None, *, default_split: str) -> HFDatasetSourceConfig | None:
        if kwargs is None:
            return None
        adapter_name = self.maker_name.removeprefix("make_").removesuffix("_dataset")
        values = dict(_LEGACY_TEXT_MAKER_DEFAULTS.get(adapter_name, {}))
        # Registered maker defaults describe the training split. Validation and
        # test inherit their own defaults unless callers explicitly override it.
        values.pop("split", None)
        values.update(self.maker_kwargs or {})
        values.update(kwargs)
        adapter_keys = {
            "audio_column",
            "conversation_column",
            "conversations_column",
            "messages_column",
            "prompt",
            "remove_text_spaces",
            "text_column",
            "video_root_path",
        }
        adapter_kwargs = {key: values.pop(key) for key in list(values) if key in adapter_keys}
        return HFDatasetSourceConfig(
            path_or_dataset=str(values.pop("path_or_dataset", self.maker_name)),
            split=str(values.pop("split", default_split)),
            subset=values.pop("subset", None),
            load_kwargs=values or None,
            schema_adapter=adapter_name,
            adapter_kwargs=adapter_kwargs or None,
        )

    def build_datasets(self, context: DatasetBuildContext) -> tuple[Any | None, Any | None, Any | None]:
        """Build datasets through the canonical GPT SFT builder."""
        if context.tokenizer is None:
            raise ValueError("HFTextSFTDatasetProvider requires a tokenizer in DatasetBuildContext.")
        builder_cls = _get_gpt_sft_dataset_builder()
        return tuple(builder_cls(config=self._to_config(), tokenizer=context.tokenizer).build())
