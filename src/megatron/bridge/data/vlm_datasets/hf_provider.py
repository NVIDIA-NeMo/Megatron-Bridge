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

"""
Provider that builds conversation datasets from HuggingFace datasets.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import torch
from transformers import AutoProcessor

from megatron.bridge.data.vlm_datasets.conversation_dataset import VLMConversationDataset
from megatron.bridge.data.vlm_datasets.hf_dataset_makers import (
    make_cord_v2_dataset,
    make_cv17_dataset,
    make_medpix_dataset,
    make_rdr_dataset,
)
from megatron.bridge.training.config import DatasetBuildContext, DatasetProvider


@dataclass(kw_only=True)
class HFDatasetConversationProvider(DatasetProvider):
    """DatasetProvider that builds VLM conversation datasets from HF datasets.

    This provider leverages simple maker functions that return lists of examples
    with a "conversation" schema understood by model processors. It binds a
    HuggingFace `AutoProcessor` for the specified model and selects an
    appropriate collate function for batching.
    """

    # Required to match model.seq_length (enforced by ConfigContainer.validate)
    sequence_length: int

    # HF processor/model identifier (e.g., "Qwen/Qwen2.5-VL-3B-Instruct")
    hf_processor_path: str

    # Select which maker to use. Must match a function defined in makers module
    # like `make_rdr_dataset`, `make_cord_v2_dataset`, `make_medpix_dataset`, `make_cv17_dataset`.
    maker_name: str

    # Optional parameters forwarded to the selected maker
    maker_kwargs: Optional[Dict[str, Any]] = None

    # Optional collate override. If None, inferred from processor type.
    collate_impl: Optional[Callable[[list, Any], Dict[str, torch.Tensor]]] = None

    # Keep parity with GPTDatasetConfig usage in batching utilities
    skip_getting_attention_mask_from_dataset: bool = True

    # DataloaderConfig fields are inherited (num_workers, dataloader_type, etc.)
    dataloader_type: Optional[Literal["single", "cyclic", "external"]] = "single"

    def _get_maker(self) -> Callable[..., List[Dict[str, Any]]]:
        registry: Dict[str, Callable[..., List[Dict[str, Any]]]] = {
            "make_rdr_dataset": make_rdr_dataset,
            "make_cord_v2_dataset": make_cord_v2_dataset,
            "make_medpix_dataset": make_medpix_dataset,
            "make_cv17_dataset": make_cv17_dataset,
        }
        if self.maker_name in registry:
            return registry[self.maker_name]
        # Allow passing function name alias without prefix, e.g., "rdr" -> make_rdr_dataset
        alias_map = {
            "rdr": "make_rdr_dataset",
            "cord_v2": "make_cord_v2_dataset",
            "medpix": "make_medpix_dataset",
            "cv17": "make_cv17_dataset",
        }
        if self.maker_name in alias_map and alias_map[self.maker_name] in registry:
            return registry[alias_map[self.maker_name]]
        raise ValueError(f"Unknown maker_name: {self.maker_name}")

    def _build_split_dataset(
        self,
        split: str,
        target_length: int,
        processor: Any,
    ) -> Optional[VLMConversationDataset]:
        if target_length <= 0:
            return None
        maker = self._get_maker()
        kwargs = dict(self.maker_kwargs or {})
        kwargs.setdefault("split", split)
        base_examples = maker(**kwargs)  # type: ignore[misc]
        if not isinstance(base_examples, list) or len(base_examples) == 0:
            raise ValueError(f"Maker '{self.maker_name}' returned no examples for split='{split}'")
        return VLMConversationDataset(
            base_examples=base_examples,
            target_length=target_length,
            processor=processor,
            collate_impl=self.collate_impl,
        )

    def build_datasets(self, context: DatasetBuildContext) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
        # Bind processor for the requested model
        processor = AutoProcessor.from_pretrained(self.hf_processor_path, trust_remote_code=True)

        train_ds = self._build_split_dataset("train", context.train_samples, processor)
        valid_ds = self._build_split_dataset("validation", context.valid_samples, processor)
        test_ds = self._build_split_dataset("test", context.test_samples, processor)

        return train_ds, valid_ds, test_ds
