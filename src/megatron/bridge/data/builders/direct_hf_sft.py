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

"""Serializable config and runtime builder for direct Hugging Face SFT."""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Literal

import torch
from megatron.core.process_groups_config import ProcessGroupCollection
from transformers import AutoProcessor, AutoTokenizer

from megatron.bridge.data.base import DataloaderConfig, DatasetBuildContext, validate_declarative_mapping
from megatron.bridge.data.collators.sft import text_chat_collate_fn, text_prompt_completion_collate_fn
from megatron.bridge.data.conversation_processing import get_processor_tokenizer, is_text_only_chat_example
from megatron.bridge.data.datasets.direct_sft import DirectSFTDataset
from megatron.bridge.data.sft_processing import (
    ChatSFTPreprocessingConfig,
    SFTPreprocessingConfig,
    is_text_only_prompt_completion_example,
    normalize_sft_examples,
    validate_sft_preprocessing_config,
)
from megatron.bridge.data.sources.hf import (
    HFDatasetSourceConfig,
    hf_dataset_supports_split,
    load_and_adapt_hf_dataset,
    prepare_hf_dataset_sources,
)
from megatron.bridge.models.hf_pretrained.utils import is_safe_repo
from megatron.bridge.training.tokenizers.tokenizer import MegatronTokenizer


logger = logging.getLogger(__name__)

CollateFunction = Callable[..., dict[str, torch.Tensor]]


@dataclass(kw_only=True)
class DirectHFSFTDatasetConfig(DataloaderConfig):
    """Serializable configuration for direct Hugging Face SFT datasets.

    Chat preprocessing is the compatibility default for multimodal and
    conversation sources. New text recipes should select chat or paired-text
    preprocessing explicitly.
    """

    seq_length: int
    source: HFDatasetSourceConfig
    validation_source: HFDatasetSourceConfig | None = None
    test_source: HFDatasetSourceConfig | None = None
    preprocessing: SFTPreprocessingConfig = field(default_factory=ChatSFTPreprocessingConfig)
    hf_processor_path: str | None = None
    hf_processor_kwargs: dict[str, Any] | None = None
    do_validation: bool = True
    do_test: bool = True
    skip_getting_attention_mask_from_dataset: bool = True
    dataloader_type: Literal["single", "cyclic", "batch", "external"] | None = "cyclic"
    enable_in_batch_packing: bool = False
    defer_in_batch_packing_to_step: bool = False
    pad_to_max_length: bool = False
    pad_to_multiple_of: int = 128
    in_batch_packing_pad_to_multiple_of: int = 1

    def validate(self) -> None:
        """Validate declarative source and dataset settings."""
        if self.seq_length <= 0:
            raise ValueError("seq_length must be greater than 0.")
        validate_sft_preprocessing_config(self.preprocessing)
        self.source.validate()
        if self.do_validation and self.validation_source is not None:
            self._inherit_source_adapter_kwargs(self.validation_source)
            self.validation_source.validate()
        if self.do_test and self.test_source is not None:
            self._inherit_source_adapter_kwargs(self.test_source)
            self.test_source.validate()
        if self.hf_processor_path is not None and not self.hf_processor_path.strip():
            raise ValueError("hf_processor_path must be a non-empty string when set.")
        validate_declarative_mapping(self.hf_processor_kwargs, field_name="hf_processor_kwargs")
        if self.hf_processor_kwargs is not None and "trust_remote_code" in self.hf_processor_kwargs:
            raise ValueError(
                "hf_processor_kwargs must not override trust_remote_code; use the dataset trust policy instead."
            )
        if self.pad_to_multiple_of <= 0:
            raise ValueError("pad_to_multiple_of must be greater than 0.")
        if self.in_batch_packing_pad_to_multiple_of <= 0:
            raise ValueError("in_batch_packing_pad_to_multiple_of must be greater than 0.")

    def _inherit_source_adapter_kwargs(self, split_source: HFDatasetSourceConfig) -> None:
        """Fill unset adapter arguments on another split of the training source."""
        if split_source.dataset_name != self.source.dataset_name or not self.source.adapter_kwargs:
            return
        split_adapter_kwargs = dict(split_source.adapter_kwargs or {})
        for key, value in self.source.adapter_kwargs.items():
            if split_adapter_kwargs.get(key) is None:
                split_adapter_kwargs[key] = value
        split_source.adapter_kwargs = split_adapter_kwargs

    def finalize(self) -> None:
        """Finalize dataloader settings and validate this config."""
        super().finalize()
        self.validate()


def normalize_direct_hf_sft_processor(processor: Any) -> Any:
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


def load_direct_hf_sft_processor(config: DirectHFSFTDatasetConfig, tokenizer: Any | None) -> Any:
    """Load the configured HF processor or adapt the training tokenizer."""
    if config.hf_processor_path is None:
        if tokenizer is None:
            raise ValueError("hf_processor_path must be set when no tokenizer is available in build context.")
        return normalize_direct_hf_sft_processor(get_processor_tokenizer(tokenizer))

    trust_remote_code = is_safe_repo(
        trust_remote_code=config.trust_remote_code,
        hf_path=config.hf_processor_path,
    )
    processor_kwargs = dict(config.hf_processor_kwargs or {})
    try:
        return normalize_direct_hf_sft_processor(
            AutoProcessor.from_pretrained(
                config.hf_processor_path,
                trust_remote_code=trust_remote_code,
                **processor_kwargs,
            )
        )
    except (OSError, ValueError):
        logger.debug(
            "AutoProcessor.from_pretrained failed for %s; falling back to AutoTokenizer.",
            config.hf_processor_path,
            exc_info=True,
        )
        return normalize_direct_hf_sft_processor(
            AutoTokenizer.from_pretrained(
                config.hf_processor_path,
                trust_remote_code=trust_remote_code,
                **processor_kwargs,
            )
        )


def load_direct_hf_sft_examples(
    source: HFDatasetSourceConfig,
    preprocessing: SFTPreprocessingConfig,
) -> list[dict[str, Any]]:
    """Load and normalize one declarative Hugging Face source."""
    return normalize_sft_examples(load_and_adapt_hf_dataset(source), preprocessing)


def select_direct_hf_sft_collate(
    examples: list[dict[str, Any]],
    preprocessing: SFTPreprocessingConfig | None = None,
    collate_impl: CollateFunction | None = None,
) -> CollateFunction | None:
    """Select a shared text collator for the explicit preprocessing variant."""
    if collate_impl is not None:
        return collate_impl
    preprocessing = preprocessing or ChatSFTPreprocessingConfig()
    validate_sft_preprocessing_config(preprocessing)
    if isinstance(preprocessing, ChatSFTPreprocessingConfig):
        if all(is_text_only_chat_example(example) for example in examples):
            return partial(text_chat_collate_fn, loss_mode=preprocessing.loss_mode)
        if preprocessing.loss_mode != "assistant":
            raise ValueError("Multimodal direct-HF collators currently support only assistant chat loss.")
        return None
    if all(is_text_only_prompt_completion_example(example, preprocessing) for example in examples):
        return partial(text_prompt_completion_collate_fn, preprocessing=preprocessing)
    raise ValueError("Prompt-completion preprocessing supports text-only examples.")


def build_direct_hf_sft_split(
    config: DirectHFSFTDatasetConfig,
    source: HFDatasetSourceConfig,
    target_length: int,
    processor: Any,
    *,
    collate_impl: CollateFunction | None = None,
) -> DirectSFTDataset | None:
    """Build one requested direct-HF SFT split."""
    if target_length <= 0:
        return None
    from megatron.bridge.data.collators.registry import model_collate_required_for_all_examples

    examples = load_direct_hf_sft_examples(source, config.preprocessing)
    if collate_impl is None and model_collate_required_for_all_examples(type(processor).__name__):
        if not isinstance(config.preprocessing, ChatSFTPreprocessingConfig):
            raise ValueError(
                f"Processor type '{type(processor).__name__}' requires chat preprocessing through its "
                "model-owned collator."
            )
        selected_collate = None
    else:
        selected_collate = select_direct_hf_sft_collate(examples, config.preprocessing, collate_impl)
    return DirectSFTDataset(
        base_examples=examples,
        target_length=target_length,
        processor=processor,
        collate_impl=selected_collate,
        sequence_length=config.seq_length,
        pad_to_max_length=config.pad_to_max_length,
        pad_to_multiple_of=config.pad_to_multiple_of,
        enable_in_batch_packing=config.enable_in_batch_packing,
        defer_in_batch_packing_to_step=config.defer_in_batch_packing_to_step,
        in_batch_packing_pad_to_multiple_of=config.in_batch_packing_pad_to_multiple_of,
    )


class DirectHFSFTDatasetBuilder:
    """Build runtime SFT datasets from declarative Hugging Face sources."""

    def __init__(
        self,
        config: DirectHFSFTDatasetConfig,
        *,
        collate_impl: CollateFunction | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self._collate_impl = collate_impl

    def build(
        self,
        context: DatasetBuildContext,
    ) -> tuple[DirectSFTDataset | None, DirectSFTDataset | None, DirectSFTDataset | None]:
        """Build train, validation, and test datasets for requested sample counts."""
        if (
            self.config.do_validation
            and context.valid_samples > 0
            and self.config.validation_source is None
            and not hf_dataset_supports_split(self.config.source, "validation")
        ):
            raise ValueError(
                "The selected Hugging Face source has no validation split; disable validation or set one."
            )
        if (
            self.config.do_test
            and context.test_samples > 0
            and self.config.test_source is None
            and not hf_dataset_supports_split(self.config.source, "test")
        ):
            raise ValueError("The selected Hugging Face source has no test split; disable test or set one.")
        validation_source = self.config.validation_source or self.config.source.with_split("validation")
        test_source = self.config.test_source or self.config.source.with_split("test")
        requested_sources = []
        if context.train_samples > 0:
            requested_sources.append(self.config.source)
        if self.config.do_validation and context.valid_samples > 0:
            requested_sources.append(validation_source)
        if self.config.do_test and context.test_samples > 0:
            requested_sources.append(test_source)
        prepare_hf_dataset_sources(requested_sources)

        processor = load_direct_hf_sft_processor(self.config, context.tokenizer)
        train_dataset = build_direct_hf_sft_split(
            self.config,
            self.config.source,
            context.train_samples,
            processor,
            collate_impl=self._collate_impl,
        )
        valid_dataset = (
            build_direct_hf_sft_split(
                self.config,
                validation_source,
                context.valid_samples,
                processor,
                collate_impl=self._collate_impl,
            )
            if self.config.do_validation
            else None
        )
        test_dataset = (
            build_direct_hf_sft_split(
                self.config,
                test_source,
                context.test_samples,
                processor,
                collate_impl=self._collate_impl,
            )
            if self.config.do_test
            else None
        )
        return train_dataset, valid_dataset, test_dataset


def direct_hf_sft_train_valid_test_datasets_provider(
    train_val_test_num_samples: list[int],
    dataset_config: DirectHFSFTDatasetConfig,
    tokenizer: MegatronTokenizer | None = None,
    pg_collection: ProcessGroupCollection | None = None,
) -> tuple[DirectSFTDataset | None, DirectSFTDataset | None, DirectSFTDataset | None]:
    """Build direct-HF SFT datasets through the canonical runtime builder."""
    context = DatasetBuildContext(
        train_samples=train_val_test_num_samples[0],
        valid_samples=train_val_test_num_samples[1],
        test_samples=train_val_test_num_samples[2],
        tokenizer=tokenizer,
        pg_collection=pg_collection,
    )
    return DirectHFSFTDatasetBuilder(dataset_config).build(context)
