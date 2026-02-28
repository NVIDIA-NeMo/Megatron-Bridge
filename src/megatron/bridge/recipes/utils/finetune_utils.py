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

from megatron.bridge.data.builders.hf_dataset import HFDatasetConfig
from megatron.bridge.data.datasets.packed_sequence import PackedSequenceSpecs
from megatron.bridge.data.hf_processors.gsm8k import process_gsm8k_example
from megatron.bridge.data.hf_processors.openmathinstruct2 import process_openmathinstruct2_example
from megatron.bridge.data.hf_processors.squad import process_squad_example
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


def default_squad_config(seq_length: int, packed_sequence: bool = True, pad_seq_to_mult: int = 1) -> HFDatasetConfig:
    """Create default SQuAD dataset configuration for finetuning recipes.

    Args:
        seq_length: Sequence length for the dataset
        packed_sequence: Whether to enable packed sequences for training efficiency
        pad_seq_to_mult: Optional multiple to pad each sequence to when packing
            (set to `2 * context_parallel_size` for THD CP runs).

    Returns:
        HFDatasetConfig configured for SQuAD finetuning

    Note:
        Uses consistent settings across all finetuning recipes:
        - SQuAD dataset with appropriate dataloader type
        - 10% validation split
        - Seed 5678 (different from pretrain seed 1234)
        - Packed sequences when enabled improve training efficiency
    """
    if packed_sequence:
        # Packed sequence configuration
        dataset_kwargs = {"pad_to_max_length": True}
        packed_sequence_specs = PackedSequenceSpecs(packed_sequence_size=seq_length, pad_seq_to_mult=pad_seq_to_mult)
    else:
        # Standard configuration
        dataset_kwargs = {}
        packed_sequence_specs = None

    # Use 'batch' sampler for variable-length finetuning
    # Samples full global batch to ensure consistent padding across all microbatches
    dataloader_type = "batch"

    return HFDatasetConfig(
        dataset_name="squad",
        process_example_fn=process_squad_example,
        seq_length=seq_length,
        seed=5678,  # Different from pretrain seed
        dataloader_type=dataloader_type,
        num_workers=1,
        do_validation=True,
        do_test=False,
        val_proportion=0.1,
        dataset_kwargs=dataset_kwargs,
        packed_sequence_specs=packed_sequence_specs,
        rewrite=False,
    )


def default_openmathinstruct2_config(
    seq_length: int = 4096,
    packed_sequence: bool = False,
    pad_seq_to_mult: int = 1,
) -> HFDatasetConfig:
    """Create default OpenMathInstruct-2 dataset configuration for finetuning recipes."""
    # Create packed sequence specs if needed
    packed_sequence_specs = None
    if packed_sequence:
        packed_sequence_specs = PackedSequenceSpecs(packed_sequence_size=seq_length, pad_seq_to_mult=pad_seq_to_mult)

    return HFDatasetConfig(
        dataset_name="nvidia/OpenMathInstruct-2",  # Hugging Face dataset name
        split="train_1M",  # Default to the 1M subset
        process_example_fn=process_openmathinstruct2_example,  # Processing function
        seq_length=seq_length,
        seed=5678,
        memmap_workers=1,
        # Dataloader config parameters
        dataloader_type="batch",
        do_validation=True,
        do_test=False,
        val_proportion=0.05,  # 950k train, 50k val
        num_workers=2,
        data_sharding=True,
        pin_memory=True,
        persistent_workers=False,
        packed_sequence_specs=packed_sequence_specs,
        rewrite=False,  # Rewrite existing processed files
    )


def default_gsm8k_config(
    seq_length: int = 2048,
    packed_sequence: bool = False,
    pad_seq_to_mult: int = 1,
) -> HFDatasetConfig:
    """Create default GSM8K dataset configuration for finetuning recipes.

    GSM8K (Grade School Math 8K) is a dataset of 8.5K high quality linguistically diverse
    grade school math word problems. See: https://huggingface.co/datasets/openai/gsm8k

    Args:
        seq_length: Sequence length for the dataset (default 2048, sufficient for GSM8K)
        packed_sequence: Whether to enable packed sequences for training efficiency
        pad_seq_to_mult: Optional multiple to pad each sequence to when packing
            (set to `2 * context_parallel_size` for THD CP runs).
        apply_tokenizer_chat_template: If True, apply the tokenizer's chat template
            when processing examples.
        dataset_subset: Dataset subset to use - 'main' (default) or 'socratic'

    Returns:
        HFDatasetConfig configured for GSM8K finetuning

    Note:
        - GSM8K has 7,473 train and 1,319 test examples
        - Since test split exists, we use it as validation (do_test=True, do_validation=False)
        - Uses 'batch' dataloader type for variable-length finetuning
    """
    # Create packed sequence specs if needed
    packed_sequence_specs = None
    if packed_sequence:
        packed_sequence_specs = PackedSequenceSpecs(packed_sequence_size=seq_length, pad_seq_to_mult=pad_seq_to_mult)

    return HFDatasetConfig(
        dataset_name="openai/gsm8k",  # Hugging Face dataset name
        dataset_subset="main",  # 'main' or 'socratic'
        split="train",  # Use train split for training
        process_example_fn=process_gsm8k_example,  # Processing function
        seq_length=seq_length,
        seed=5678,
        memmap_workers=1,
        # Dataloader config parameters
        dataloader_type="batch",
        do_validation=True,
        do_test=False,
        val_proportion=0.1,
        num_workers=2,
        data_sharding=True,
        pin_memory=True,
        persistent_workers=False,
        packed_sequence_specs=packed_sequence_specs,
        rewrite=False,
    )
