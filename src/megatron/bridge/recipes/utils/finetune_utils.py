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

"""Utility functions for finetuning recipes."""

from typing import Any

from megatron.bridge.data.hf_datasets.provider import HFDatasetConversationProvider
from megatron.bridge.data.hf_datasets.text_collate import text_chat_collate_fn
from megatron.bridge.peft.base import PEFT
from megatron.bridge.peft.dora import DoRA
from megatron.bridge.peft.lora import LoRA


def default_peft_config(peft_scheme: str | PEFT | None, **kwargs) -> PEFT | None:
    """Create default PEFT configuration matching NeMo2 exactly.

    Args:
        peft_scheme: PEFT scheme - 'lora', 'dora', PEFT instance, or None for full finetuning

    Returns:
        PEFT configuration or None for full finetuning
    """
    if peft_scheme is None:
        return None  # Full finetuning

    if isinstance(peft_scheme, PEFT):
        return peft_scheme  # User provided custom PEFT

    if isinstance(peft_scheme, str):
        if peft_scheme.lower() == "none":
            return None
        if peft_scheme.lower() == "lora":
            return LoRA(**kwargs)
        elif peft_scheme.lower() == "dora":
            return DoRA(**kwargs)
        else:
            raise ValueError(f"Unknown PEFT scheme: {peft_scheme}. Supported: 'lora', 'dora', or None")

    raise ValueError(f"Invalid peft type: {type(peft_scheme)}. Expected str, PEFT instance, or None")


def _text_hf_dataset_provider(
    *,
    seq_length: int,
    maker_name: str,
    maker_kwargs: dict[str, Any],
    val_maker_kwargs: dict[str, Any] | None = None,
    test_maker_kwargs: dict[str, Any] | None = None,
    skip_test: bool = True,
    num_workers: int = 2,
) -> HFDatasetConversationProvider:
    """Create a direct HF conversation provider for text-only SFT presets."""
    return HFDatasetConversationProvider(
        seq_length=seq_length,
        hf_processor_path=None,
        maker_name=maker_name,
        maker_kwargs=maker_kwargs,
        val_maker_kwargs=val_maker_kwargs,
        test_maker_kwargs=test_maker_kwargs,
        skip_test=skip_test,
        collate_impl=text_chat_collate_fn,
        dataloader_type="batch",
        num_workers=num_workers,
        data_sharding=True,
        pin_memory=True,
        persistent_workers=False,
        pack_sequences_in_batch=False,
        shuffle=False,
        seed=5678,
    )


def default_squad_config(
    seq_length: int, packed_sequence: bool = True, pad_seq_to_mult: int = 1
) -> HFDatasetConversationProvider:
    """Create default SQuAD dataset configuration for finetuning recipes.

    Args:
        seq_length: Sequence length for the dataset
        packed_sequence: Retained for API compatibility. The direct HF text
            collate path does not currently support runtime packing.
        pad_seq_to_mult: Retained for API compatibility with previous packed
            JSONL configs.

    Returns:
        HFDatasetConversationProvider configured for SQuAD finetuning

    Note:
        Uses consistent settings across all finetuning recipes:
        - SQuAD dataset with appropriate dataloader type
        - 10% validation slice
        - Seed 5678 (different from pretrain seed 1234)
    """
    del packed_sequence, pad_seq_to_mult

    return _text_hf_dataset_provider(
        maker_name="squad",
        maker_kwargs={
            "path_or_dataset": "rajpurkar/squad",
            "split": "train[:90%]",
        },
        val_maker_kwargs={"split": "train[90%:]"},
        seq_length=seq_length,
        num_workers=1,
    )


def default_openmathinstruct2_config(
    seq_length: int = 4096,
    packed_sequence: bool = False,
    pad_seq_to_mult: int = 1,
) -> HFDatasetConversationProvider:
    """Create default OpenMathInstruct-2 dataset configuration for finetuning recipes."""
    del packed_sequence, pad_seq_to_mult

    return _text_hf_dataset_provider(
        maker_name="openmathinstruct2",
        maker_kwargs={
            "path_or_dataset": "nvidia/OpenMathInstruct-2",
            "split": "train_1M",
        },
        val_maker_kwargs={"split": "train_1M[:5%]"},
        seq_length=seq_length,
        num_workers=2,
    )


def default_gsm8k_config(
    seq_length: int = 2048,
    packed_sequence: bool = False,
    pad_seq_to_mult: int = 1,
) -> HFDatasetConversationProvider:
    """Create default GSM8K dataset configuration for finetuning recipes.

    GSM8K (Grade School Math 8K) is a dataset of 8.5K high quality linguistically diverse
    grade school math word problems. See: https://huggingface.co/datasets/openai/gsm8k

    Args:
        seq_length: Sequence length for the dataset (default 2048, sufficient for GSM8K)
        packed_sequence: Retained for API compatibility. The direct HF text
            collate path does not currently support runtime packing.
        pad_seq_to_mult: Retained for API compatibility with previous packed
            JSONL configs.

    Returns:
        HFDatasetConversationProvider configured for GSM8K finetuning

    Note:
        - GSM8K has 7,473 train and 1,319 test examples
        - Loads the full DatasetDict so the published test split is used for evaluation
    """
    del packed_sequence, pad_seq_to_mult

    return _text_hf_dataset_provider(
        maker_name="gsm8k",
        maker_kwargs={
            "path_or_dataset": "openai/gsm8k",
            "subset": "main",
            "split": "train",
        },
        test_maker_kwargs={"split": "test"},
        skip_test=False,
        seq_length=seq_length,
        num_workers=2,
    )


def default_openmathinstruct2_thinking_packed_config(
    seq_length: int = 4096,
    packed_sequence: bool = False,
    pad_seq_to_mult: int = 1,
) -> HFDatasetConversationProvider:
    """Create OpenMathInstruct-2 dataset config with CoT in analysis channel, answer in final channel.

    Puts generated_solution (minus the trailing \boxed{N}) into the assistant thinking field
    (rendered as <|channel|>analysis) and #### {expected_answer} into the content field
    (rendered as <|channel|>final).

    Args:
        seq_length: Sequence length (default 4096)
        packed_sequence: Retained for API compatibility. The direct HF text
            collate path does not currently support runtime packing.
        pad_seq_to_mult: Retained for API compatibility with previous packed
            JSONL configs.
    """
    cfg = default_openmathinstruct2_config(
        seq_length=seq_length,
        packed_sequence=packed_sequence,
        pad_seq_to_mult=pad_seq_to_mult,
    )
    cfg.maker_name = "openmathinstruct2_thinking"
    return cfg
