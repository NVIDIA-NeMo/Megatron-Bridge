# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Mock dataset provider for MIMO testing with synthetic multimodal data.

This module produces synthetic multimodal inputs (random images, audio, etc.)
that are compatible with HuggingFace processors. It follows the same pattern
as vlm_datasets/mock_provider.py - generating fake input data but using real
processors for preprocessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
from PIL import Image

from megatron.bridge.data.mimo.base_provider import MimoDatasetProvider
from megatron.bridge.data.mimo.dataset import MimoDataset
from megatron.bridge.models.hf_pretrained.utils import is_safe_repo
from megatron.bridge.training.config import DatasetBuildContext


def _generate_random_image(width: int, height: int, rng: np.random.Generator) -> Image.Image:
    """Generate a random RGB image."""
    pixels = rng.integers(low=0, high=256, size=(height, width, 3), dtype=np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def _generate_random_audio(duration_sec: float, sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    """Generate random audio waveform."""
    num_samples = int(duration_sec * sample_rate)
    # Generate random float32 audio in [-1, 1]
    return rng.uniform(-1.0, 1.0, size=(num_samples,)).astype(np.float32)


@dataclass(kw_only=True)
class MockMimoProvider(MimoDatasetProvider):
    """DatasetProvider for mock MIMO datasets with synthetic multimodal data.

    Generates synthetic multimodal inputs (random images, audio, etc.) and uses
    real HuggingFace processors to preprocess them. This tests the full data
    pipeline without requiring real datasets.

    Follows the same pattern as vlm_datasets/MockVLMConversationProvider.

    Args:
        seq_length: Total sequence length for the model (encoder placeholders + text tokens).
            Must be greater than sum(encoder_seq_lengths.values()) to leave room for text.
        processor_paths: Per-modality HF processor paths, e.g.,
            {"vision": "openai/clip-vit-large-patch14"}.
        tokenizer_path: HuggingFace tokenizer identifier.
        special_token_ids: Per-encoder placeholder token IDs, e.g., {"vision": 32000}.
        encoder_seq_lengths: Per-encoder output sequence lengths, e.g., {"vision": 577}.
            Determines how many placeholder tokens to insert for each modality.
        modality_configs: Per-modality generation config, e.g.,
            {"vision": {"type": "image", "width": 224, "height": 224}}.
        text_prompt: Default text prompt for synthetic examples.
        random_seed: Seed for random generation.

    Example:
        >>> provider = MockMimoProvider(
        ...     seq_length=2048,
        ...     processor_paths={"vision": "openai/clip-vit-large-patch14"},
        ...     tokenizer_path="meta-llama/Llama-2-7b-hf",
        ...     special_token_ids={"vision": 32000},
        ...     encoder_seq_lengths={"vision": 577},  # CLIP ViT-L/14 output tokens
        ...     modality_configs={"vision": {"type": "image", "width": 224, "height": 224}},
        ... )
        >>> context = DatasetBuildContext(train_samples=1000, valid_samples=100, test_samples=100)
        >>> train_ds, valid_ds, test_ds = provider.build_datasets(context)
    """

    seq_length: int
    processor_paths: Dict[str, str] = field(default_factory=dict)
    tokenizer_path: str = ""
    special_token_ids: Dict[str, int] = field(default_factory=dict)
    encoder_seq_lengths: Dict[str, int] = field(default_factory=dict)
    modality_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    text_prompt: str = "Describe this input."
    random_seed: int = 0
    trust_remote_code: bool = False

    # DataloaderConfig fields
    dataloader_type: Optional[Literal["single", "cyclic", "external"]] = "single"

    # Cached processors and tokenizer
    _processors: Optional[Dict[str, Any]] = field(default=None, repr=False)
    _tokenizer: Optional[Any] = field(default=None, repr=False)

    def _load_processors(self) -> Dict[str, Any]:
        """Load HuggingFace processors for each modality."""
        if self._processors is not None:
            return self._processors

        from transformers import AutoProcessor

        processors = {}
        for modality_name, processor_path in self.processor_paths.items():
            processors[modality_name] = AutoProcessor.from_pretrained(
                processor_path,
                trust_remote_code=is_safe_repo(
                    trust_remote_code=self.trust_remote_code,
                    hf_path=processor_path,
                ),
            )

        object.__setattr__(self, "_processors", processors)
        return processors

    def _load_tokenizer(self) -> Any:
        """Load HuggingFace tokenizer."""
        if self._tokenizer is not None:
            return self._tokenizer

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path,
            trust_remote_code=is_safe_repo(
                trust_remote_code=self.trust_remote_code,
                hf_path=self.tokenizer_path,
            ),
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        object.__setattr__(self, "_tokenizer", tokenizer)
        return tokenizer

    def _generate_synthetic_examples(
        self,
        size: int,
        seed_offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Generate synthetic multimodal examples.

        Args:
            size: Number of examples to generate.
            seed_offset: Offset to add to random seed for different splits.

        Returns:
            List of examples with synthetic modality data.
        """
        rng = np.random.default_rng(seed=self.random_seed + seed_offset)
        examples = []

        for i in range(size):
            example = {"text": f"{self.text_prompt} Sample {i}."}

            for modality_name, config in self.modality_configs.items():
                modality_type = config.get("type", "image")

                if modality_type == "image":
                    width = config.get("width", 224)
                    height = config.get("height", 224)
                    example[modality_name] = _generate_random_image(width, height, rng)

                elif modality_type == "audio":
                    duration = config.get("duration_sec", 3.0)
                    sample_rate = config.get("sample_rate", 16000)
                    example[modality_name] = _generate_random_audio(duration, sample_rate, rng)

                else:
                    # Default to image
                    example[modality_name] = _generate_random_image(224, 224, rng)

            examples.append(example)

        return examples

    def _build_split_dataset(
        self,
        size: int,
        processors: Dict[str, Any],
        tokenizer: Any,
        seed_offset: int = 0,
    ) -> Optional[MimoDataset]:
        """Build dataset for a single split."""
        if size <= 0:
            return None

        examples = self._generate_synthetic_examples(size, seed_offset)

        # Build modality_columns mapping (modality_name -> column_name)
        # In synthetic data, we use modality_name as column_name
        modality_columns = {name: name for name in self.modality_configs.keys()}

        return MimoDataset(
            examples=examples,
            processors=processors,
            tokenizer=tokenizer,
            seq_length=self.seq_length,
            special_token_ids=self.special_token_ids,
            encoder_seq_lengths=self.encoder_seq_lengths,
            modality_columns=modality_columns,
            text_column="text",
        )

    def build_datasets(
        self, context: DatasetBuildContext
    ) -> Tuple[Optional[MimoDataset], Optional[MimoDataset], Optional[MimoDataset]]:
        """Build train, validation, and test datasets with synthetic data.

        Args:
            context: Build context with sample counts.

        Returns:
            Tuple of (train_dataset, valid_dataset, test_dataset).
        """
        processors = self._load_processors()
        tokenizer = self._load_tokenizer()

        train_ds = self._build_split_dataset(context.train_samples, processors, tokenizer, seed_offset=0)
        valid_ds = self._build_split_dataset(context.valid_samples, processors, tokenizer, seed_offset=1000000)
        test_ds = self._build_split_dataset(context.test_samples, processors, tokenizer, seed_offset=2000000)

        return train_ds, valid_ds, test_ds

    def get_collate_fn(self) -> Callable:
        """Return collate function for MIMO datasets.

        Returns:
            Partial function of mimo_collate_fn with modality names pre-filled.
        """
        from megatron.bridge.data.mimo.collate import mimo_collate_fn

        return partial(
            mimo_collate_fn,
            modality_names=list(self.modality_configs.keys()),
        )
