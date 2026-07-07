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

"""Deprecated provider adapter for Hugging Face conversation datasets."""

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import torch

from megatron.bridge.data.base import DatasetBuildContext, DatasetProvider
from megatron.bridge.data.builders.hf_sft_dataset import HFSFTDatasetConfig
from megatron.bridge.data.hf_datasets.makers import get_hf_dataset_maker
from megatron.bridge.data.hf_source import HFDatasetSourceConfig


# =============================================================================
# Deprecated compatibility API
# =============================================================================


@dataclass(kw_only=True)
class HFConversationDatasetProvider(DatasetProvider):
    """Deprecated constructor-compatible adapter for HF conversation data.

    Use :class:`HFSFTDatasetConfig` for primary training paths and
    :class:`HFSFTDatasetBuilder` for explicit runtime construction.
    """

    seq_length: int
    maker_name: str
    hf_processor_path: str | None = None
    maker_kwargs: Optional[Dict[str, Any]] = None
    val_maker_kwargs: Optional[Dict[str, Any]] = None
    test_maker_kwargs: Optional[Dict[str, Any]] = None
    do_validation: bool = True
    do_test: bool = True
    collate_impl: Optional[Callable[..., Dict[str, torch.Tensor]]] = None
    skip_getting_attention_mask_from_dataset: bool = True
    dataloader_type: Optional[Literal["single", "cyclic", "batch", "external"]] = "single"
    enable_in_batch_packing: bool = False
    defer_in_batch_packing_to_step: bool = False
    pad_to_max_length: bool = False
    pad_to_multiple_of: int = 128
    in_batch_packing_pad_to_multiple_of: int = 1

    def __post_init__(self) -> None:
        warnings.warn(
            "HFConversationDatasetProvider is deprecated; use HFSFTDatasetConfig instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    def _get_maker(self) -> Callable[..., List[Dict[str, Any]]]:
        """Return the configured maker for compatibility subclasses."""
        return get_hf_dataset_maker(self.maker_name)

    def _merged_maker_kwargs(self, overrides: dict[str, Any] | None, *, default_split: str) -> dict[str, Any]:
        values = dict(self.maker_kwargs or {})
        values.update(overrides or {})
        values.setdefault("split", default_split)
        return values

    def _source_from_kwargs(self, kwargs: dict[str, Any], *, default_split: str) -> HFDatasetSourceConfig:
        """Translate legacy maker kwargs into a declarative source."""
        return HFDatasetSourceConfig(
            path_or_dataset=str(kwargs.get("path_or_dataset", self.maker_name)),
            subset=kwargs.get("subset"),
            split=str(kwargs.get("split", default_split)),
        )

    def _to_config(self) -> HFSFTDatasetConfig:
        """Translate legacy declarative fields into the canonical config."""
        train_kwargs = self._merged_maker_kwargs(None, default_split="train")
        validation_kwargs = self._merged_maker_kwargs(self.val_maker_kwargs, default_split="validation")
        test_kwargs = self._merged_maker_kwargs(self.test_maker_kwargs, default_split="test")
        return HFSFTDatasetConfig(
            seq_length=self.seq_length,
            source=self._source_from_kwargs(train_kwargs, default_split="train"),
            validation_source=(
                self._source_from_kwargs(validation_kwargs, default_split="validation") if self.do_validation else None
            ),
            test_source=self._source_from_kwargs(test_kwargs, default_split="test") if self.do_test else None,
            hf_processor_path=self.hf_processor_path,
            do_validation=self.do_validation,
            do_test=self.do_test,
            skip_getting_attention_mask_from_dataset=self.skip_getting_attention_mask_from_dataset,
            dataloader_type=self.dataloader_type,
            num_workers=self.num_workers,
            data_sharding=self.data_sharding,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            persistent_workers=self.persistent_workers,
            trust_remote_code=self.trust_remote_code,
            enable_in_batch_packing=self.enable_in_batch_packing,
            defer_in_batch_packing_to_step=self.defer_in_batch_packing_to_step,
            pad_to_max_length=self.pad_to_max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            in_batch_packing_pad_to_multiple_of=self.in_batch_packing_pad_to_multiple_of,
        )

    def _load_processor_or_tokenizer(self, tokenizer: Any | None = None) -> Any:
        """Load the runtime processor through the canonical builder helper.

        This protected method remains for subclasses of the deprecated provider.
        """
        from megatron.bridge.data.builders import hf_sft_dataset as builder_module

        return builder_module.load_hf_sft_processor(self._to_config(), tokenizer)

    def _build_split_dataset(
        self,
        split: str,
        target_length: int,
        processor: Any,
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Any | None:
        """Build one split through the canonical helper.

        This protected method remains for subclasses of the deprecated provider.
        """
        from megatron.bridge.data.builders import hf_sft_dataset as builder_module

        maker_kwargs = self._merged_maker_kwargs(extra_kwargs, default_split=split)
        return builder_module.build_hf_sft_split(
            self._to_config(),
            self._source_from_kwargs(maker_kwargs, default_split=split),
            target_length,
            processor,
            maker=self._get_maker(),
            maker_kwargs=maker_kwargs,
            collate_impl=self.collate_impl,
        )

    def build_datasets(self, context: DatasetBuildContext) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
        """Build datasets through compatibility hooks backed by canonical helpers."""
        processor = self._load_processor_or_tokenizer(context.tokenizer)
        train_dataset = self._build_split_dataset("train", context.train_samples, processor)
        valid_dataset = (
            self._build_split_dataset("validation", context.valid_samples, processor, self.val_maker_kwargs)
            if self.do_validation
            else None
        )
        test_dataset = (
            self._build_split_dataset("test", context.test_samples, processor, self.test_maker_kwargs)
            if self.do_test
            else None
        )
        return train_dataset, valid_dataset, test_dataset
