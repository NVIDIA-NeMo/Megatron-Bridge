# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""HuggingFace dataset provider for MIMO multi-encoder models."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple

from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoProcessor, AutoTokenizer

from megatron.bridge.data.mimo.base_provider import MimoDatasetProvider
from megatron.bridge.data.mimo.collate import mimo_collate_fn
from megatron.bridge.data.mimo.dataset import MimoDataset
from megatron.bridge.models.hf_pretrained.utils import is_safe_repo
from megatron.bridge.training.config import DatasetBuildContext


@dataclass(kw_only=True)
class HFMimoDatasetProvider(MimoDatasetProvider):
    """DatasetProvider for MIMO models using HuggingFace datasets.

    Loads datasets from HuggingFace Hub and applies per-modality processors
    to convert raw inputs (images, audio, text) into preprocessed tensors
    that MIMO encoder modules consume during training.

    For testing with synthetic data, use MockMimoProvider instead.

    Args:
        seq_length: Total sequence length for the model (encoder placeholders + text tokens).
            Must be greater than sum(encoder_seq_lengths.values()) to leave room for text.
            Text is truncated to fit: max_text_tokens = seq_length - total_encoder_tokens.
        hf_dataset_path: HuggingFace dataset identifier, e.g., "liuhaotian/LLaVA-Instruct-150K".
        hf_dataset_name: Optional dataset configuration name.
        hf_tokenizer_path: HuggingFace tokenizer identifier.
        processor_paths: Per-modality processor paths, e.g.,
            {"vision": "openai/clip-vit-large-patch14"}.
        special_token_ids: Per-encoder placeholder token IDs, e.g., {"vision": 32000}.
        encoder_seq_lengths: Per-encoder output sequence lengths, e.g., {"vision": 577}.
            Determines how many placeholder tokens to insert for each modality.
        modality_columns: Map modality name to dataset column, e.g., {"vision": "image"}.
        text_column: Column name for text data. Default: "text".
        train_split: Dataset split for training. Default: "train".
        valid_split: Dataset split for validation. Default: "validation".
        test_split: Dataset split for testing. Default: "test".
        trust_remote_code: Whether to trust remote code for HF models/processors.

    Example:
        >>> provider = HFMimoDatasetProvider(
        ...     seq_length=2048,
        ...     hf_dataset_path="liuhaotian/LLaVA-Instruct-150K",
        ...     hf_tokenizer_path="meta-llama/Llama-2-7b-hf",
        ...     processor_paths={"vision": "openai/clip-vit-large-patch14"},
        ...     special_token_ids={"vision": 32000},
        ...     encoder_seq_lengths={"vision": 577},  # CLIP ViT-L/14 output tokens
        ...     modality_columns={"vision": "image"},
        ... )
        >>> context = DatasetBuildContext(train_samples=10000, valid_samples=1000, test_samples=1000)
        >>> train_ds, valid_ds, test_ds = provider.build_datasets(context)
    """

    seq_length: int
    hf_dataset_path: str
    hf_dataset_name: Optional[str] = None
    hf_tokenizer_path: str = ""
    processor_paths: Dict[str, str] = field(default_factory=dict)
    special_token_ids: Dict[str, int] = field(default_factory=dict)
    encoder_seq_lengths: Dict[str, int] = field(default_factory=dict)
    modality_columns: Dict[str, str] = field(default_factory=dict)
    text_column: str = "text"
    train_split: str = "train"
    valid_split: str = "validation"
    test_split: str = "test"

    # Cached processors and tokenizer (loaded once)
    _processors: Optional[Dict[str, Any]] = field(default=None, repr=False)
    _tokenizer: Optional[Any] = field(default=None, repr=False)

    def _load_processors(self) -> Dict[str, Any]:
        """Load HuggingFace processors for each modality."""
        if self._processors is not None:
            return self._processors

        processors = {}
        for modality_name, processor_path in self.processor_paths.items():
            processors[modality_name] = AutoProcessor.from_pretrained(
                processor_path,
                trust_remote_code=is_safe_repo(
                    trust_remote_code=self.trust_remote_code,
                    hf_path=processor_path,
                ),
            )

        # Store for reuse
        object.__setattr__(self, "_processors", processors)
        return processors

    def _load_tokenizer(self) -> Any:
        """Load HuggingFace tokenizer."""
        if self._tokenizer is not None:
            return self._tokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self.hf_tokenizer_path,
            trust_remote_code=is_safe_repo(
                trust_remote_code=self.trust_remote_code,
                hf_path=self.hf_tokenizer_path,
            ),
        )

        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Store for reuse
        object.__setattr__(self, "_tokenizer", tokenizer)
        return tokenizer

    def _load_hf_dataset(self, split: str) -> Any:
        """Load a HuggingFace dataset split."""
        try:
            dataset = load_dataset(
                self.hf_dataset_path,
                name=self.hf_dataset_name,
                split=split,
                trust_remote_code=is_safe_repo(
                    trust_remote_code=self.trust_remote_code,
                    hf_path=self.hf_dataset_path,
                ),
            )
            return dataset
        except ValueError:
            # Split doesn't exist
            return None

    def _build_split_dataset(
        self,
        split: str,
        target_samples: int,
        processors: Dict[str, Any],
        tokenizer: Any,
    ) -> Optional[MimoDataset]:
        """Build dataset for a single split."""
        if target_samples <= 0:
            return None

        hf_dataset = self._load_hf_dataset(split)
        if hf_dataset is None:
            return None

        return MimoDataset(
            examples=hf_dataset,
            processors=processors,
            tokenizer=tokenizer,
            seq_length=self.seq_length,
            special_token_ids=self.special_token_ids,
            encoder_seq_lengths=self.encoder_seq_lengths,
            modality_columns=self.modality_columns,
            text_column=self.text_column,
            max_samples=target_samples,
        )

    def build_datasets(
        self, context: DatasetBuildContext
    ) -> Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset]]:
        """Build train, validation, and test datasets.

        Args:
            context: Build context with sample counts.

        Returns:
            Tuple of (train_dataset, valid_dataset, test_dataset).
            Any element can be None if split doesn't exist or sample count is 0.
        """
        processors = self._load_processors()
        tokenizer = self._load_tokenizer()

        train_ds = self._build_split_dataset(self.train_split, context.train_samples, processors, tokenizer)
        valid_ds = self._build_split_dataset(self.valid_split, context.valid_samples, processors, tokenizer)
        test_ds = self._build_split_dataset(self.test_split, context.test_samples, processors, tokenizer)

        return train_ds, valid_ds, test_ds

    def get_collate_fn(self) -> Callable:
        """Return collate function for MIMO datasets.

        Returns:
            Partial function of mimo_collate_fn with modality names pre-filled.
        """
        return partial(
            mimo_collate_fn,
            modality_names=list(self.modality_columns.keys()),
        )
