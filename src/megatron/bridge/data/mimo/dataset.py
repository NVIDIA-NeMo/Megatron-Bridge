# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Dataset wrapper for MIMO multi-encoder models."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import torch
from torch.utils.data import Dataset


class MimoDataset(Dataset):
    """Dataset for MIMO models with per-modality preprocessing.

    Wraps a data source (HuggingFace dataset or list of examples) and applies
    per-modality processors to convert raw inputs (images, audio, etc.) into
    preprocessed tensors (pixel_values, input_features) that encoders consume
    during the forward pass.

    Args:
        examples: Data source - either a HuggingFace Dataset or a list of dicts.
        processors: Dict mapping modality name to HF processor, e.g.,
            {"vision": AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")}.
        tokenizer: Tokenizer for text processing.
        seq_length: Total sequence length for the model (encoder placeholders + text tokens).
            Must be greater than sum(encoder_seq_lengths.values()) to leave room for text.
            Text is truncated to fit: max_text_tokens = seq_length - total_encoder_tokens.
        special_token_ids: Per-encoder placeholder token IDs, e.g., {"vision": 32000}.
        encoder_seq_lengths: Per-encoder output sequence lengths, e.g., {"vision": 577}.
            Determines how many placeholder tokens to insert for each modality.
            For CLIP ViT-L/14 with 224x224 images, this would be 577 (576 patches + 1 CLS).
        modality_columns: Dict mapping modality name to column name in dataset,
            e.g., {"vision": "image", "audio": "audio_path"}.
        text_column: Column name for text/conversation data. Default: "text".
        max_samples: Optional limit on dataset size for debugging.
        preprocess_fn: Optional function to preprocess each example before
            modality processing.

    Example:
        >>> from datasets import load_dataset
        >>> from transformers import AutoProcessor, AutoTokenizer
        >>>
        >>> # Using HuggingFace Dataset
        >>> hf_ds = load_dataset("liuhaotian/LLaVA-Instruct-150K", split="train")
        >>> processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>>
        >>> dataset = MimoDataset(
        ...     examples=hf_ds,
        ...     processors={"vision": processor},
        ...     tokenizer=tokenizer,
        ...     seq_length=2048,
        ...     special_token_ids={"vision": 32000},
        ...     encoder_seq_lengths={"vision": 577},  # CLIP ViT-L/14 output tokens
        ...     modality_columns={"vision": "image"},
        ... )
        >>>
        >>> # Or using a simple list of dicts for testing/prototyping
        >>> examples = [
        ...     {"text": "Describe this image.", "image": "img1.jpg"},
        ...     {"text": "What do you see?", "image": "img2.jpg"},
        ... ]
        >>> dataset = MimoDataset(
        ...     examples=examples,
        ...     processors={"vision": processor},
        ...     tokenizer=tokenizer,
        ...     seq_length=2048,
        ...     special_token_ids={"vision": 32000},
        ...     encoder_seq_lengths={"vision": 577},
        ...     modality_columns={"vision": "image"},
        ... )
    """

    def __init__(
        self,
        examples: Any,  # HF Dataset or List[Dict]
        processors: Dict[str, Any],
        tokenizer: Any,
        seq_length: int,
        special_token_ids: Dict[str, int],
        encoder_seq_lengths: Dict[str, int],
        modality_columns: Dict[str, str],
        text_column: str = "text",
        max_samples: Optional[int] = None,
        preprocess_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> None:
        # Validate encoder tokens fit within seq_length
        total_encoder_tokens = sum(encoder_seq_lengths.values())
        if total_encoder_tokens >= seq_length:
            raise ValueError(
                f"Total encoder tokens ({total_encoder_tokens}) must be less than "
                f"seq_length ({seq_length}) to leave room for text tokens. "
                f"encoder_seq_lengths: {encoder_seq_lengths}"
            )

        self.examples = examples
        self.processors = processors
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.special_token_ids = special_token_ids
        self.encoder_seq_lengths = encoder_seq_lengths
        self.modality_columns = modality_columns
        self.text_column = text_column
        self.preprocess_fn = preprocess_fn

        # Limit dataset size if requested
        self._size = len(examples)
        if max_samples is not None and max_samples > 0:
            self._size = min(self._size, max_samples)

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single example with preprocessed modality inputs.

        Returns:
            Dict containing:
                - input_ids: Tokenized text with placeholder tokens
                - labels: Same as input_ids (for causal LM training)
                - attention_mask: Attention mask
                - position_ids: Position indices
                - modality_inputs: Dict[str, Any] with preprocessed inputs per modality
                  e.g., {"vision": {"pixel_values": tensor, ...}}
        """
        if idx >= self._size:
            raise IndexError(f"Index {idx} out of range for dataset of size {self._size}")

        example = self.examples[idx]

        # Apply custom preprocessing if provided
        if self.preprocess_fn is not None:
            example = self.preprocess_fn(example)

        # Process each modality
        modality_inputs: Dict[str, Dict[str, Any]] = {}
        for modality_name, column_name in self.modality_columns.items():
            if column_name not in example or example[column_name] is None:
                continue

            raw_input = example[column_name]
            processor = self.processors.get(modality_name)

            if processor is not None:
                # Apply HF processor to get preprocessed inputs
                # This typically returns pixel_values for vision, input_features for audio
                processed = processor(raw_input, return_tensors="pt")
                # Remove batch dimension added by processor
                modality_inputs[modality_name] = {
                    k: v.squeeze(0) if isinstance(v, torch.Tensor) else v for k, v in processed.items()
                }

        # Process text with placeholder tokens
        text = example.get(self.text_column, "")
        input_ids = self._tokenize_with_placeholders(text, modality_inputs)

        # Create attention mask and position ids
        attention_mask = torch.ones_like(input_ids)
        position_ids = torch.arange(len(input_ids))

        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "modality_inputs": modality_inputs,
        }

    def _tokenize_with_placeholders(
        self,
        text: str,
        modality_inputs: Dict[str, Dict[str, Any]],
    ) -> torch.Tensor:
        """Tokenize text and insert placeholder tokens for each modality.

        For each modality present, inserts N placeholder tokens at the beginning
        of the sequence, where N = encoder_seq_lengths[modality_name]. This matches
        the number of embeddings the encoder will produce, enabling 1:1 replacement
        during the model forward pass.

        Args:
            text: Raw text to tokenize.
            modality_inputs: Dict of preprocessed modality inputs.

        Returns:
            Token IDs tensor with placeholder tokens inserted.
        """
        # Tokenize the text
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.seq_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)

        # Insert placeholder tokens for each modality at the beginning
        # The order follows the order of modality_inputs (Python 3.7+ dict ordering)
        prefix_tokens = []
        for modality_name in modality_inputs.keys():
            if modality_name in self.special_token_ids:
                token_id = self.special_token_ids[modality_name]
                num_tokens = self.encoder_seq_lengths.get(modality_name, 1)
                prefix_tokens.extend([token_id] * num_tokens)

        if prefix_tokens:
            prefix = torch.tensor(prefix_tokens, dtype=input_ids.dtype)
            # Truncate text tokens to make room for placeholders
            max_text_len = self.seq_length - len(prefix_tokens)
            input_ids = input_ids[:max_text_len]
            input_ids = torch.cat([prefix, input_ids])

        # Pad or truncate to seq_length
        if len(input_ids) < self.seq_length:
            pad_len = self.seq_length - len(input_ids)
            pad_token_id = self.tokenizer.pad_token_id or 0
            padding = torch.full((pad_len,), pad_token_id, dtype=input_ids.dtype)
            input_ids = torch.cat([input_ids, padding])
        else:
            input_ids = input_ids[: self.seq_length]

        return input_ids
