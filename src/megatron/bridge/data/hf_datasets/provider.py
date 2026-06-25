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

"""Provider that builds conversation datasets from HuggingFace datasets."""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import torch
from transformers import AutoProcessor, AutoTokenizer

from megatron.bridge.data.hf_datasets.conversation_dataset import ConversationDataset
from megatron.bridge.data.hf_datasets.makers import (
    get_hf_dataset_maker,
)
from megatron.bridge.data.vlm_processing import get_processor_tokenizer
from megatron.bridge.models.hf_pretrained.utils import is_safe_repo
from megatron.bridge.training.config import DatasetBuildContext, DatasetProvider


logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class HFConversationDatasetProvider(DatasetProvider):
    """DatasetProvider that builds conversation datasets from Hugging Face datasets.

    This provider leverages simple maker functions that return lists of examples
    with a ``messages`` or ``conversation`` schema understood by model processors.
    It binds a Hugging Face processor/tokenizer for the specified model and
    selects an appropriate collate function for batching.

    HF data creation workflow:
        1. A maker function loads a Hugging Face dataset split and normalizes each
           row into Bridge's chat schema: ``messages`` for text-only rows or
           ``conversation`` for processor-ready multimodal rows.
        2. ``ConversationDataset`` repeats that normalized list to the requested
           Megatron sample count, then binds the selected collate implementation.
        3. The collate function renders chat templates, tokenizes the batch, and
           builds shifted labels/loss masks or model-specific visual inputs.
    """

    # Required to match model.seq_length (enforced by ConfigContainer.validate)
    seq_length: int

    # HF processor/model identifier (e.g., "Qwen/Qwen2.5-VL-3B-Instruct").
    # Text-only presets may leave this unset and use the training tokenizer from
    # DatasetBuildContext instead.
    hf_processor_path: str | None = None

    # Select which maker to use. Must match a function defined in makers module
    # like `make_rdr_dataset`, `make_cord_v2_dataset`, `make_medpix_dataset`, `make_cv17_dataset`.
    maker_name: str

    # Optional parameters forwarded to the selected maker (used for train split by default)
    maker_kwargs: Optional[Dict[str, Any]] = None

    # Per-split overrides: merged on top of maker_kwargs when building that split.
    # This allows different subset/split/prompt per data split (e.g. aishell "dev" vs "train").
    val_maker_kwargs: Optional[Dict[str, Any]] = None
    test_maker_kwargs: Optional[Dict[str, Any]] = None

    # Control whether optional validation/test splits are built.
    do_validation: bool = True
    do_test: bool = True

    # Optional collate override. If None, inferred from processor type.
    collate_impl: Optional[Callable[..., Dict[str, torch.Tensor]]] = None

    # Keep parity with GPTDatasetConfig usage in batching utilities
    skip_getting_attention_mask_from_dataset: bool = True

    # DataloaderConfig fields are inherited (num_workers, dataloader_type, etc.)
    dataloader_type: Optional[Literal["single", "cyclic", "batch", "external"]] = "single"

    # Enable batch-level online sequence packing (dataset-level packing is available in FinetuneDatasetProvider)
    enable_in_batch_packing: bool = False
    # Active user: Qwen3-VL. Its model step owns the packing path because it
    # needs the original tensors before building Qwen3-VL-specific packed
    # metadata. Qwen3.5-VL also sets this defensively while packing is off.
    defer_in_batch_packing_to_step: bool = False

    # Collate-time VLM padding controls. ConfigContainer sets pad_to_max_length
    # for PP/EP fixed-shape cases; otherwise collate pads to an efficient multiple
    # capped by seq_length.
    pad_to_max_length: bool = False
    pad_to_multiple_of: int = 128

    # Per-sample padding multiple used by collate-time in-batch packing.
    # ConfigContainer fills this from model CP/SP constraints when available.
    in_batch_packing_pad_to_multiple_of: int = 1

    def _get_maker(self) -> Callable[..., List[Dict[str, Any]]]:
        return get_hf_dataset_maker(self.maker_name)

    def _build_split_dataset(
        self,
        split: str,
        target_length: int,
        processor: Any,
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Optional[ConversationDataset]:
        if target_length <= 0:
            return None
        maker = self._get_maker()
        kwargs = dict(self.maker_kwargs or {})
        if extra_kwargs:
            kwargs.update(extra_kwargs)
        kwargs.setdefault("split", split)
        base_examples = maker(**kwargs)  # type: ignore[misc]
        if not isinstance(base_examples, list) or len(base_examples) == 0:
            raise ValueError(f"Maker '{self.maker_name}' returned no examples for split='{split}'")
        return ConversationDataset(
            base_examples=base_examples,
            target_length=target_length,
            processor=processor,
            collate_impl=self.collate_impl,
            sequence_length=self.seq_length,
            pad_to_max_length=self.pad_to_max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            enable_in_batch_packing=self.enable_in_batch_packing,
            defer_in_batch_packing_to_step=self.defer_in_batch_packing_to_step,
            in_batch_packing_pad_to_multiple_of=self.in_batch_packing_pad_to_multiple_of,
        )

    def _load_processor_or_tokenizer(self, tokenizer: Any | None = None) -> Any:
        if self.hf_processor_path is None:
            if tokenizer is None:
                raise ValueError("hf_processor_path must be set when no tokenizer is available in build context.")
            return get_processor_tokenizer(tokenizer)

        trust_remote_code = is_safe_repo(
            trust_remote_code=self.trust_remote_code,
            hf_path=self.hf_processor_path,
        )
        try:
            return AutoProcessor.from_pretrained(
                self.hf_processor_path,
                trust_remote_code=trust_remote_code,
            )
        except (OSError, ValueError):
            logger.debug(
                "AutoProcessor.from_pretrained failed for %s; falling back to AutoTokenizer.",
                self.hf_processor_path,
                exc_info=True,
            )
            return AutoTokenizer.from_pretrained(
                self.hf_processor_path,
                trust_remote_code=trust_remote_code,
            )

    def build_datasets(self, context: DatasetBuildContext) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
        # Bind processor for the requested model
        processor = self._load_processor_or_tokenizer(context.tokenizer)

        train_ds = self._build_split_dataset("train", context.train_samples, processor)
        valid_ds = (
            self._build_split_dataset("validation", context.valid_samples, processor, self.val_maker_kwargs)
            if self.do_validation
            else None
        )
        test_ds = (
            self._build_split_dataset("test", context.test_samples, processor, self.test_maker_kwargs)
            if self.do_test
            else None
        )

        return train_ds, valid_ds, test_ds
