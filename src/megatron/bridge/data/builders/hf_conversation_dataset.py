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

"""Runtime construction for direct Hugging Face conversation datasets."""

import logging
from collections.abc import Callable
from typing import Any

import torch
from transformers import AutoProcessor, AutoTokenizer

from megatron.bridge.data.hf_datasets.conversation_dataset import ConversationDataset
from megatron.bridge.data.hf_datasets.makers import get_hf_dataset_maker
from megatron.bridge.data.hf_datasets.text_collate import text_chat_collate_fn
from megatron.bridge.data.vlm_processing import get_processor_tokenizer
from megatron.bridge.models.hf_pretrained.utils import is_safe_repo
from megatron.bridge.training.config import DatasetBuildContext, HFConversationDatasetConfig


logger = logging.getLogger(__name__)

DatasetMaker = Callable[..., list[dict[str, Any]]]
CollateFunction = Callable[..., dict[str, torch.Tensor]]


def normalize_hf_conversation_processor(processor: Any) -> Any:
    """Ensure the runtime tokenizer can pad batched conversation text."""
    tokenizer = get_processor_tokenizer(processor)
    if getattr(tokenizer, "pad_token_id", None) is not None:
        return processor

    eos_token = getattr(tokenizer, "eos_token", None)
    if eos_token is not None:
        tokenizer.pad_token = eos_token
    else:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            tokenizer.pad_token_id = eos_token_id
    return processor


def load_hf_conversation_processor(config: HFConversationDatasetConfig, tokenizer: Any | None) -> Any:
    """Load the configured HF processor or adapt the training tokenizer."""
    if config.hf_processor_path is None:
        if tokenizer is None:
            raise ValueError("hf_processor_path must be set when no tokenizer is available in build context.")
        return normalize_hf_conversation_processor(get_processor_tokenizer(tokenizer))

    trust_remote_code = is_safe_repo(
        trust_remote_code=config.trust_remote_code,
        hf_path=config.hf_processor_path,
    )
    try:
        return normalize_hf_conversation_processor(
            AutoProcessor.from_pretrained(
                config.hf_processor_path,
                trust_remote_code=trust_remote_code,
            )
        )
    except (OSError, ValueError):
        logger.debug(
            "AutoProcessor.from_pretrained failed for %s; falling back to AutoTokenizer.",
            config.hf_processor_path,
            exc_info=True,
        )
        return normalize_hf_conversation_processor(
            AutoTokenizer.from_pretrained(
                config.hf_processor_path,
                trust_remote_code=trust_remote_code,
            )
        )


def load_hf_conversation_examples(
    config: HFConversationDatasetConfig,
    split: str,
    *,
    extra_kwargs: dict[str, Any] | None = None,
    maker: DatasetMaker | None = None,
) -> list[dict[str, Any]]:
    """Invoke one registered maker and validate its normalized examples."""
    selected_maker = maker if maker is not None else get_hf_dataset_maker(config.maker_name)
    kwargs = dict(config.maker_kwargs or {})
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    kwargs.setdefault("split", split)
    examples = selected_maker(**kwargs)
    if not isinstance(examples, list) or not examples:
        raise ValueError(f"Maker '{config.maker_name}' returned no examples for split='{split}'")
    if not all(isinstance(example, dict) for example in examples):
        raise TypeError(f"Maker '{config.maker_name}' must return a list of dictionaries for split='{split}'.")
    return examples


def select_hf_conversation_collate(
    examples: list[dict[str, Any]],
    collate_impl: CollateFunction | None = None,
) -> CollateFunction | None:
    """Select the shared text collator while leaving multimodal inference to the dataset."""
    if collate_impl is not None:
        return collate_impl
    if "messages" in examples[0] or "conversations" in examples[0]:
        return text_chat_collate_fn
    return None


def build_hf_conversation_split(
    config: HFConversationDatasetConfig,
    split: str,
    target_length: int,
    processor: Any,
    *,
    extra_kwargs: dict[str, Any] | None = None,
    maker: DatasetMaker | None = None,
    collate_impl: CollateFunction | None = None,
) -> ConversationDataset | None:
    """Build one requested conversation split."""
    if target_length <= 0:
        return None
    examples = load_hf_conversation_examples(config, split, extra_kwargs=extra_kwargs, maker=maker)
    return ConversationDataset(
        base_examples=examples,
        target_length=target_length,
        processor=processor,
        collate_impl=select_hf_conversation_collate(examples, collate_impl),
        sequence_length=config.seq_length,
        pad_to_max_length=config.pad_to_max_length,
        pad_to_multiple_of=config.pad_to_multiple_of,
        enable_in_batch_packing=config.enable_in_batch_packing,
        defer_in_batch_packing_to_step=config.defer_in_batch_packing_to_step,
        in_batch_packing_pad_to_multiple_of=config.in_batch_packing_pad_to_multiple_of,
    )


class HFConversationDatasetBuilder:
    """Build runtime conversation datasets from a serializable config.

    Args:
        config: Declarative Hugging Face conversation dataset settings.
        maker: Optional runtime maker override for tests or custom integrations.
        collate_impl: Optional runtime collate override for compatibility integrations.
    """

    def __init__(
        self,
        config: HFConversationDatasetConfig,
        *,
        maker: DatasetMaker | None = None,
        collate_impl: CollateFunction | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self._maker = maker
        self._collate_impl = collate_impl

    def build(
        self,
        context: DatasetBuildContext,
    ) -> tuple[ConversationDataset | None, ConversationDataset | None, ConversationDataset | None]:
        """Build train, validation, and test datasets for the requested sample counts."""
        processor = load_hf_conversation_processor(self.config, context.tokenizer)
        train_dataset = build_hf_conversation_split(
            self.config,
            "train",
            context.train_samples,
            processor,
            maker=self._maker,
            collate_impl=self._collate_impl,
        )
        valid_dataset = (
            build_hf_conversation_split(
                self.config,
                "validation",
                context.valid_samples,
                processor,
                extra_kwargs=self.config.val_maker_kwargs,
                maker=self._maker,
                collate_impl=self._collate_impl,
            )
            if self.config.do_validation
            else None
        )
        test_dataset = (
            build_hf_conversation_split(
                self.config,
                "test",
                context.test_samples,
                processor,
                extra_kwargs=self.config.test_maker_kwargs,
                maker=self._maker,
                collate_impl=self._collate_impl,
            )
            if self.config.do_test
            else None
        )
        return train_dataset, valid_dataset, test_dataset
