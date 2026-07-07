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

from megatron.bridge.data.hf_datasets.makers import get_hf_dataset_maker
from megatron.bridge.training.config import (
    DatasetBuildContext,
    DatasetProvider,
    HFConversationDatasetConfig,
)


@dataclass(kw_only=True)
class HFConversationDatasetProvider(DatasetProvider):
    """Deprecated constructor-compatible adapter for HF conversation data.

    Use :class:`HFConversationDatasetConfig` for primary training paths and
    :class:`HFConversationDatasetBuilder` for explicit runtime construction.
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
            "HFConversationDatasetProvider is deprecated; use HFConversationDatasetConfig instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    def _get_maker(self) -> Callable[..., List[Dict[str, Any]]]:
        """Return the configured maker for compatibility subclasses."""
        return get_hf_dataset_maker(self.maker_name)

    def _to_config(self) -> HFConversationDatasetConfig:
        """Translate legacy declarative fields into the canonical config."""
        return HFConversationDatasetConfig(
            seq_length=self.seq_length,
            maker_name=self.maker_name,
            hf_processor_path=self.hf_processor_path,
            maker_kwargs=self.maker_kwargs,
            val_maker_kwargs=self.val_maker_kwargs,
            test_maker_kwargs=self.test_maker_kwargs,
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

    def build_datasets(self, context: DatasetBuildContext) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
        """Build datasets through the canonical runtime builder."""
        from megatron.bridge.data.builders.hf_conversation_dataset import HFConversationDatasetBuilder

        return HFConversationDatasetBuilder(
            self._to_config(),
            maker=self._get_maker(),
            collate_impl=self.collate_impl,
        ).build(context)
