# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Unit tests for MimoDataset."""

import pytest
import torch

from megatron.bridge.data.mimo.dataset import MimoDataset


class MockExamples:
    """Mock data source (HuggingFace dataset or list) for testing."""

    def __init__(self, size: int = 100):
        self._size = size
        self._data = [
            {
                "text": f"Sample text {i} with some content.",
                "image": f"image_{i}.jpg",  # Fake image path
                "audio": f"audio_{i}.wav",  # Fake audio path
            }
            for i in range(size)
        ]

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, idx: int):
        return self._data[idx]


class MockProcessor:
    """Mock HuggingFace processor for testing."""

    def __init__(self, output_key: str = "pixel_values", output_shape: tuple = (3, 224, 224)):
        self.output_key = output_key
        self.output_shape = output_shape

    def __call__(self, inputs, return_tensors: str = "pt"):
        # Return mock processed output
        return {
            self.output_key: torch.randn(1, *self.output_shape),
        }


class MockTokenizer:
    """Mock HuggingFace tokenizer for testing."""

    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.pad_token = "<pad>"
        self.pad_token_id = 0
        self.eos_token = "</s>"

    def __call__(
        self,
        text: str,
        truncation: bool = True,
        max_length: int = 512,
        return_tensors: str = "pt",
    ):
        # Generate fake token IDs based on text length
        num_tokens = min(len(text.split()) * 2, max_length)
        input_ids = torch.randint(1, self.vocab_size, (1, num_tokens))
        return {"input_ids": input_ids}


class TestMimoDataset:
    """Test suite for MimoDataset."""

    def test_basic_construction(self):
        """Test basic dataset construction."""
        examples = MockExamples(size=50)
        processors = {"vision": MockProcessor()}
        tokenizer = MockTokenizer()

        dataset = MimoDataset(
            examples=examples,
            processors=processors,
            tokenizer=tokenizer,
            seq_length=128,
            special_token_ids={"vision": 32000},
            encoder_seq_lengths={"vision": 1},
            modality_columns={"vision": "image"},
        )

        assert len(dataset) == 50

    def test_max_samples_limit(self):
        """Test that max_samples limits dataset size."""
        examples = MockExamples(size=100)
        processors = {"vision": MockProcessor()}
        tokenizer = MockTokenizer()

        dataset = MimoDataset(
            examples=examples,
            processors=processors,
            tokenizer=tokenizer,
            seq_length=128,
            special_token_ids={"vision": 32000},
            encoder_seq_lengths={"vision": 1},
            modality_columns={"vision": "image"},
            max_samples=25,
        )

        assert len(dataset) == 25

    def test_getitem_returns_expected_keys(self):
        """Test that __getitem__ returns expected dict keys."""
        examples = MockExamples(size=10)
        processors = {"vision": MockProcessor()}
        tokenizer = MockTokenizer()

        dataset = MimoDataset(
            examples=examples,
            processors=processors,
            tokenizer=tokenizer,
            seq_length=128,
            special_token_ids={"vision": 32000},
            encoder_seq_lengths={"vision": 1},
            modality_columns={"vision": "image"},
        )

        item = dataset[0]

        assert "input_ids" in item
        assert "labels" in item
        assert "attention_mask" in item
        assert "position_ids" in item
        assert "modality_inputs" in item

    def test_getitem_shapes(self):
        """Test that __getitem__ returns correct tensor shapes."""
        seq_length = 64
        examples = MockExamples(size=10)
        processors = {"vision": MockProcessor(output_shape=(3, 224, 224))}
        tokenizer = MockTokenizer()

        dataset = MimoDataset(
            examples=examples,
            processors=processors,
            tokenizer=tokenizer,
            seq_length=seq_length,
            special_token_ids={"vision": 32000},
            encoder_seq_lengths={"vision": 1},
            modality_columns={"vision": "image"},
        )

        item = dataset[0]

        assert item["input_ids"].shape == (seq_length,)
        assert item["labels"].shape == (seq_length,)
        assert item["attention_mask"].shape == (seq_length,)
        assert item["position_ids"].shape == (seq_length,)

    def test_modality_inputs_present(self):
        """Test that modality_inputs contains processed tensors."""
        examples = MockExamples(size=10)
        processors = {"vision": MockProcessor(output_shape=(3, 224, 224))}
        tokenizer = MockTokenizer()

        dataset = MimoDataset(
            examples=examples,
            processors=processors,
            tokenizer=tokenizer,
            seq_length=64,
            special_token_ids={"vision": 32000},
            encoder_seq_lengths={"vision": 1},
            modality_columns={"vision": "image"},
        )

        item = dataset[0]

        assert "vision" in item["modality_inputs"]
        assert "pixel_values" in item["modality_inputs"]["vision"]
        # Shape should be (3, 224, 224) - batch dim squeezed
        assert item["modality_inputs"]["vision"]["pixel_values"].shape == (3, 224, 224)

    def test_multiple_modalities(self):
        """Test dataset with multiple modalities."""
        examples = MockExamples(size=10)
        processors = {
            "vision": MockProcessor(output_key="pixel_values", output_shape=(3, 224, 224)),
            "audio": MockProcessor(output_key="input_features", output_shape=(128, 3000)),
        }
        tokenizer = MockTokenizer()

        dataset = MimoDataset(
            examples=examples,
            processors=processors,
            tokenizer=tokenizer,
            seq_length=64,
            special_token_ids={"vision": 32000, "audio": 32001},
            encoder_seq_lengths={"vision": 1, "audio": 1},
            modality_columns={"vision": "image", "audio": "audio"},
        )

        item = dataset[0]

        assert "vision" in item["modality_inputs"]
        assert "audio" in item["modality_inputs"]
        assert "pixel_values" in item["modality_inputs"]["vision"]
        assert "input_features" in item["modality_inputs"]["audio"]

    def test_placeholder_token_inserted(self):
        """Test that N placeholder tokens are inserted in input_ids based on encoder_seq_lengths."""
        examples = MockExamples(size=10)
        processors = {"vision": MockProcessor()}
        tokenizer = MockTokenizer()

        vision_placeholder = 32000
        encoder_seq_length = 10  # Insert 10 placeholder tokens

        dataset = MimoDataset(
            examples=examples,
            processors=processors,
            tokenizer=tokenizer,
            seq_length=64,
            special_token_ids={"vision": vision_placeholder},
            encoder_seq_lengths={"vision": encoder_seq_length},
            modality_columns={"vision": "image"},
        )

        item = dataset[0]

        # First N tokens should all be the vision placeholder
        for i in range(encoder_seq_length):
            assert item["input_ids"][i].item() == vision_placeholder, (
                f"Position {i} should be placeholder token {vision_placeholder}"
            )

        # Token at position N should NOT be the placeholder (should be text)
        assert item["input_ids"][encoder_seq_length].item() != vision_placeholder, (
            f"Position {encoder_seq_length} should not be placeholder token"
        )

    def test_index_out_of_range(self):
        """Test that accessing out-of-range index raises error."""
        examples = MockExamples(size=10)
        processors = {"vision": MockProcessor()}
        tokenizer = MockTokenizer()

        dataset = MimoDataset(
            examples=examples,
            processors=processors,
            tokenizer=tokenizer,
            seq_length=64,
            special_token_ids={"vision": 32000},
            encoder_seq_lengths={"vision": 1},
            modality_columns={"vision": "image"},
        )

        with pytest.raises(IndexError):
            _ = dataset[100]

    def test_custom_text_column(self):
        """Test using custom text column name."""
        # Create dataset with different column name
        examples = MockExamples(size=10)
        # Modify data to use different column
        for item in examples._data:
            item["content"] = item.pop("text")

        processors = {"vision": MockProcessor()}
        tokenizer = MockTokenizer()

        dataset = MimoDataset(
            examples=examples,
            processors=processors,
            tokenizer=tokenizer,
            seq_length=64,
            special_token_ids={"vision": 32000},
            encoder_seq_lengths={"vision": 1},
            modality_columns={"vision": "image"},
            text_column="content",
        )

        # Should not raise
        item = dataset[0]
        assert item["input_ids"].shape == (64,)

    def test_list_as_examples(self):
        """Test that a plain list works as examples."""
        examples = [
            {"text": "Hello world", "image": "img1.jpg"},
            {"text": "Another sample", "image": "img2.jpg"},
        ]
        processors = {"vision": MockProcessor()}
        tokenizer = MockTokenizer()

        dataset = MimoDataset(
            examples=examples,
            processors=processors,
            tokenizer=tokenizer,
            seq_length=64,
            special_token_ids={"vision": 32000},
            encoder_seq_lengths={"vision": 1},
            modality_columns={"vision": "image"},
        )

        assert len(dataset) == 2
        item = dataset[0]
        assert "input_ids" in item


class TestMimoDatasetPreprocessing:
    """Test preprocessing functionality."""

    def test_custom_preprocess_fn(self):
        """Test that custom preprocess_fn is applied."""
        examples = MockExamples(size=10)
        processors = {"vision": MockProcessor()}
        tokenizer = MockTokenizer()

        def custom_preprocess(example):
            example["text"] = example["text"].upper()
            return example

        dataset = MimoDataset(
            examples=examples,
            processors=processors,
            tokenizer=tokenizer,
            seq_length=64,
            special_token_ids={"vision": 32000},
            encoder_seq_lengths={"vision": 1},
            modality_columns={"vision": "image"},
            preprocess_fn=custom_preprocess,
        )

        # Should not raise
        item = dataset[0]
        assert "input_ids" in item


class TestMimoDatasetEncoderSeqLengths:
    """Test encoder_seq_lengths functionality."""

    def test_encoder_seq_lengths_validation(self):
        """Test that encoder_seq_lengths >= seq_length raises ValueError."""
        examples = MockExamples(size=10)
        processors = {"vision": MockProcessor()}
        tokenizer = MockTokenizer()

        # encoder_seq_lengths (100) >= seq_length (64) should raise
        with pytest.raises(ValueError, match="must be less than"):
            MimoDataset(
                examples=examples,
                processors=processors,
                tokenizer=tokenizer,
                seq_length=64,
                special_token_ids={"vision": 32000},
                encoder_seq_lengths={"vision": 100},  # Too large!
                modality_columns={"vision": "image"},
            )

    def test_encoder_seq_lengths_validation_equal(self):
        """Test that encoder_seq_lengths == seq_length raises ValueError."""
        examples = MockExamples(size=10)
        processors = {"vision": MockProcessor()}
        tokenizer = MockTokenizer()

        # encoder_seq_lengths (64) == seq_length (64) should raise
        with pytest.raises(ValueError, match="must be less than"):
            MimoDataset(
                examples=examples,
                processors=processors,
                tokenizer=tokenizer,
                seq_length=64,
                special_token_ids={"vision": 32000},
                encoder_seq_lengths={"vision": 64},  # Equal, no room for text!
                modality_columns={"vision": "image"},
            )

    def test_multiple_modality_placeholders(self):
        """Test correct placeholder insertion for multiple modalities."""
        examples = MockExamples(size=10)
        processors = {
            "vision": MockProcessor(output_key="pixel_values", output_shape=(3, 224, 224)),
            "audio": MockProcessor(output_key="input_features", output_shape=(128, 3000)),
        }
        tokenizer = MockTokenizer()

        vision_placeholder = 32000
        audio_placeholder = 32001
        vision_seq_len = 10
        audio_seq_len = 5

        dataset = MimoDataset(
            examples=examples,
            processors=processors,
            tokenizer=tokenizer,
            seq_length=64,
            special_token_ids={"vision": vision_placeholder, "audio": audio_placeholder},
            encoder_seq_lengths={"vision": vision_seq_len, "audio": audio_seq_len},
            modality_columns={"vision": "image", "audio": "audio"},
        )

        item = dataset[0]

        # First vision_seq_len tokens should be vision placeholder
        for i in range(vision_seq_len):
            assert item["input_ids"][i].item() == vision_placeholder, f"Position {i} should be vision placeholder"

        # Next audio_seq_len tokens should be audio placeholder
        for i in range(vision_seq_len, vision_seq_len + audio_seq_len):
            assert item["input_ids"][i].item() == audio_placeholder, f"Position {i} should be audio placeholder"

        # Token after all placeholders should not be a placeholder
        first_text_pos = vision_seq_len + audio_seq_len
        assert item["input_ids"][first_text_pos].item() not in (vision_placeholder, audio_placeholder), (
            f"Position {first_text_pos} should be text, not placeholder"
        )

    def test_encoder_seq_lengths_text_truncation(self):
        """Test that text is properly truncated when encoder tokens take most of seq_length."""
        examples = MockExamples(size=10)
        processors = {"vision": MockProcessor()}
        tokenizer = MockTokenizer()

        vision_placeholder = 32000
        encoder_seq_length = 50  # Takes most of the 64 seq_length

        dataset = MimoDataset(
            examples=examples,
            processors=processors,
            tokenizer=tokenizer,
            seq_length=64,
            special_token_ids={"vision": vision_placeholder},
            encoder_seq_lengths={"vision": encoder_seq_length},
            modality_columns={"vision": "image"},
        )

        item = dataset[0]

        # Should have exactly seq_length tokens
        assert item["input_ids"].shape == (64,)

        # First encoder_seq_length tokens should be placeholders
        for i in range(encoder_seq_length):
            assert item["input_ids"][i].item() == vision_placeholder

        # Remaining tokens should be text (or padding)
        # Just verify they're not placeholders
        for i in range(encoder_seq_length, 64):
            assert item["input_ids"][i].item() != vision_placeholder
