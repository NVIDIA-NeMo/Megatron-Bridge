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
Provider that builds conversation datasets from HuggingFace datasets for LLMs.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import torch
from transformers import AutoTokenizer

from megatron.bridge.data.llm_datasets.conversation_dataset import LLMConversationDataset
from megatron.bridge.data.llm_datasets.hf_dataset_makers import (
    make_openmathinstruct2_dataset,
    make_squad_v2_dataset,
)
from megatron.bridge.training.config import DatasetBuildContext, DatasetProvider


@dataclass(kw_only=True)
class HFDatasetConversationLLMProvider(DatasetProvider):
    """DatasetProvider that builds LLM conversation datasets from HF datasets.

    Uses a HuggingFace `AutoTokenizer` for the specified model and a default
    collate function that applies chat templates.
    """

    # Required to match model.seq_length (enforced by ConfigContainer.validate)
    sequence_length: int

    # HF tokenizer/model identifier (e.g., "meta-llama/Llama-3-8B-Instruct")
    hf_tokenizer_path: str

    # Maker name: currently supports "make_openmathinstruct2_dataset"
    maker_name: str

    maker_kwargs: Optional[Dict[str, Any]] = None

    # Optional collate override. If None, inferred default will be used
    collate_impl: Optional[Callable[[list, Any], Dict[str, torch.Tensor]]] = None

    # Keep parity with GPTDatasetConfig usage in batching utilities
    skip_getting_attention_mask_from_dataset: bool = True

    # DataloaderConfig fields are inherited (num_workers, dataloader_type, etc.)
    dataloader_type: Optional[Literal["single", "cyclic", "external"]] = "single"

    def _get_maker(self) -> Callable[..., List[Dict[str, Any]]]:
        registry: Dict[str, Callable[..., List[Dict[str, Any]]]] = {
            "make_openmathinstruct2_dataset": make_openmathinstruct2_dataset,
            "make_squad_v2_dataset": make_squad_v2_dataset,
            # aliases
            "openmathinstruct2": make_openmathinstruct2_dataset,
            "omi2": make_openmathinstruct2_dataset,
            "squad_v2": make_squad_v2_dataset,
        }
        if self.maker_name in registry:
            return registry[self.maker_name]
        raise ValueError(f"Unknown maker_name: {self.maker_name}")

    def _build_split_dataset(
        self,
        split: str,
        target_length: int,
        tokenizer: Any,
    ) -> Optional[LLMConversationDataset]:
        if target_length <= 0:
            return None
        maker = self._get_maker()
        kwargs = dict(self.maker_kwargs or {})
        kwargs.setdefault("split", split)
        try:
            base_examples = maker(**kwargs)  # type: ignore[misc]
        except Exception:
            # Some HF datasets do not provide non-train splits (e.g., test).
            # Treat missing non-train splits as absent and return None.
            if split != "train":
                return None
            raise
        if not isinstance(base_examples, list) or len(base_examples) == 0:
            # Allow empty non-train splits by treating them as absent.
            if split != "train":
                return None
            raise ValueError(f"Maker '{self.maker_name}' returned no examples for split='{split}'")
        return LLMConversationDataset(
            base_examples=base_examples,
            target_length=target_length,
            tokenizer=tokenizer,
            collate_impl=self.collate_impl,
        )

    def build_datasets(self, context: DatasetBuildContext) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
        # Bind tokenizer for the requested model
        tokenizer = AutoTokenizer.from_pretrained(self.hf_tokenizer_path, trust_remote_code=True)

        train_ds = self._build_split_dataset("train", context.train_samples, tokenizer)
        valid_ds = self._build_split_dataset("validation", context.valid_samples, tokenizer)
        test_ds = self._build_split_dataset("test", context.test_samples, tokenizer)

        return train_ds, valid_ds, test_ds
