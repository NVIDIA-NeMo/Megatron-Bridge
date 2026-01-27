# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
            modality_columns={"vision": "image", "audio": "audio"},
        )
        
        item = dataset[0]
        
        assert "vision" in item["modality_inputs"]
        assert "audio" in item["modality_inputs"]
        assert "pixel_values" in item["modality_inputs"]["vision"]
        assert "input_features" in item["modality_inputs"]["audio"]
    
    def test_placeholder_token_inserted(self):
        """Test that placeholder tokens are inserted in input_ids."""
        examples = MockExamples(size=10)
        processors = {"vision": MockProcessor()}
        tokenizer = MockTokenizer()
        
        vision_placeholder = 32000
        
        dataset = MimoDataset(
            examples=examples,
            processors=processors,
            tokenizer=tokenizer,
            seq_length=64,
            special_token_ids={"vision": vision_placeholder},
            modality_columns={"vision": "image"},
        )
        
        item = dataset[0]
        
        # First token should be the vision placeholder
        assert item["input_ids"][0].item() == vision_placeholder
    
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
            modality_columns={"vision": "image"},
            preprocess_fn=custom_preprocess,
        )
        
        # Should not raise
        item = dataset[0]
        assert "input_ids" in item
