# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Unit tests for MIMO collate functions."""

import torch

from megatron.bridge.data.mimo.collate import mimo_collate_fn


def make_sample(
    seq_length: int = 64,
    modalities: dict = None,
) -> dict:
    """Create a sample item for testing."""
    if modalities is None:
        modalities = {"vision": {"pixel_values": torch.randn(3, 224, 224)}}

    return {
        "input_ids": torch.randint(0, 1000, (seq_length,)),
        "labels": torch.randint(0, 1000, (seq_length,)),
        "attention_mask": torch.ones(seq_length),
        "position_ids": torch.arange(seq_length),
        "modality_inputs": modalities,
    }


class TestMimoCollateFn:
    """Test suite for mimo_collate_fn."""

    def test_basic_collation(self):
        """Test basic batch collation."""
        batch = [make_sample() for _ in range(4)]

        result = mimo_collate_fn(batch, modality_names=["vision"])

        assert "input_ids" in result
        assert "labels" in result
        assert "attention_mask" in result
        assert "position_ids" in result
        assert "modality_inputs" in result

    def test_batch_dimension(self):
        """Test that batch dimension is correct."""
        batch_size = 8
        seq_length = 64
        batch = [make_sample(seq_length=seq_length) for _ in range(batch_size)]

        result = mimo_collate_fn(batch, modality_names=["vision"])

        assert result["input_ids"].shape == (batch_size, seq_length)
        assert result["labels"].shape == (batch_size, seq_length)
        assert result["attention_mask"].shape == (batch_size, seq_length)
        assert result["position_ids"].shape == (batch_size, seq_length)

    def test_modality_inputs_batched(self):
        """Test that modality inputs are properly batched."""
        batch_size = 4
        image_shape = (3, 224, 224)

        batch = [
            make_sample(modalities={"vision": {"pixel_values": torch.randn(*image_shape)}}) for _ in range(batch_size)
        ]

        result = mimo_collate_fn(batch, modality_names=["vision"])

        assert "vision" in result["modality_inputs"]
        assert "pixel_values" in result["modality_inputs"]["vision"]
        assert result["modality_inputs"]["vision"]["pixel_values"].shape == (batch_size, *image_shape)

    def test_multiple_modalities(self):
        """Test collation with multiple modalities."""
        batch_size = 4

        batch = [
            make_sample(
                modalities={
                    "vision": {"pixel_values": torch.randn(3, 224, 224)},
                    "audio": {"input_features": torch.randn(128, 3000)},
                }
            )
            for _ in range(batch_size)
        ]

        result = mimo_collate_fn(batch, modality_names=["vision", "audio"])

        assert "vision" in result["modality_inputs"]
        assert "audio" in result["modality_inputs"]
        assert result["modality_inputs"]["vision"]["pixel_values"].shape == (batch_size, 3, 224, 224)
        assert result["modality_inputs"]["audio"]["input_features"].shape == (batch_size, 128, 3000)

    def test_empty_batch(self):
        """Test handling of empty batch."""
        result = mimo_collate_fn([], modality_names=["vision"])
        assert result == {}

    def test_missing_modality_in_some_items(self):
        """Test handling when some items lack a modality."""
        # Item 0 has vision, item 1 doesn't
        batch = [
            make_sample(modalities={"vision": {"pixel_values": torch.randn(3, 224, 224)}}),
            make_sample(modalities={}),  # No modality
        ]

        result = mimo_collate_fn(batch, modality_names=["vision"])

        # Should still have basic fields
        assert result["input_ids"].shape[0] == 2
        # Vision should only have 1 item (from first sample)
        if "vision" in result["modality_inputs"]:
            assert result["modality_inputs"]["vision"]["pixel_values"].shape[0] == 1

    def test_multiple_tensors_per_modality(self):
        """Test modality with multiple tensor outputs."""
        batch_size = 4

        batch = [
            make_sample(
                modalities={
                    "vision": {
                        "pixel_values": torch.randn(3, 224, 224),
                        "image_grid_thw": torch.tensor([1, 14, 14]),
                    }
                }
            )
            for _ in range(batch_size)
        ]

        result = mimo_collate_fn(batch, modality_names=["vision"])

        assert "pixel_values" in result["modality_inputs"]["vision"]
        assert "image_grid_thw" in result["modality_inputs"]["vision"]
