# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""Unit tests for MIMO checkpointing utilities.

Tests cover:
- MimoOptimizerWrapper sharded_state_dict and load_state_dict
- slice_batch_for_mimo utility
"""

import pytest
from unittest.mock import MagicMock


class TestMimoOptimizerWrapper:
    """Tests for MimoOptimizerWrapper class."""
    
    def test_sharded_state_dict(self):
        """Test sharded_state_dict collects from all optimizers."""
        from megatron.bridge.training.mimo_checkpointing import MimoOptimizerWrapper
        
        mock_llm_opt = MagicMock()
        mock_llm_opt.is_stub_optimizer = False
        mock_llm_opt.sharded_state_dict.return_value = {"key1": "value1"}
        
        mock_vision_opt = MagicMock()
        mock_vision_opt.is_stub_optimizer = False
        mock_vision_opt.sharded_state_dict.return_value = {"key2": "value2"}
        
        optimizers = {"llm": mock_llm_opt, "vision": mock_vision_opt}
        wrapper = MimoOptimizerWrapper(optimizers)
        
        model_sd = {"model": "state"}
        sharded_sd = wrapper.sharded_state_dict(model_sd)
        
        assert "llm" in sharded_sd
        assert "vision" in sharded_sd
        assert sharded_sd["llm"] == {"key1": "value1"}
        assert sharded_sd["vision"] == {"key2": "value2"}
    
    def test_sharded_state_dict_skips_stub(self):
        """Test that stub optimizers are skipped."""
        from megatron.bridge.training.mimo_checkpointing import MimoOptimizerWrapper
        
        mock_llm_opt = MagicMock()
        mock_llm_opt.is_stub_optimizer = True
        
        mock_vision_opt = MagicMock()
        mock_vision_opt.is_stub_optimizer = False
        mock_vision_opt.sharded_state_dict.return_value = {"key": "value"}
        
        optimizers = {"llm": mock_llm_opt, "vision": mock_vision_opt}
        wrapper = MimoOptimizerWrapper(optimizers)
        
        sharded_sd = wrapper.sharded_state_dict(None)
        
        assert "llm" not in sharded_sd
        assert "vision" in sharded_sd
    
    def test_sharded_state_dict_skips_none(self):
        """Test that None optimizers are skipped."""
        from megatron.bridge.training.mimo_checkpointing import MimoOptimizerWrapper
        
        mock_vision_opt = MagicMock()
        mock_vision_opt.is_stub_optimizer = False
        mock_vision_opt.sharded_state_dict.return_value = {"key": "value"}
        
        optimizers = {"llm": None, "vision": mock_vision_opt}
        wrapper = MimoOptimizerWrapper(optimizers)
        
        sharded_sd = wrapper.sharded_state_dict(None)
        
        assert "llm" not in sharded_sd
        assert "vision" in sharded_sd
    
    def test_load_state_dict(self):
        """Test load_state_dict distributes to per-module optimizers."""
        from megatron.bridge.training.mimo_checkpointing import MimoOptimizerWrapper
        
        mock_llm_opt = MagicMock()
        mock_vision_opt = MagicMock()
        
        optimizers = {"llm": mock_llm_opt, "vision": mock_vision_opt}
        wrapper = MimoOptimizerWrapper(optimizers)
        
        state_dict = {
            "llm": {"param1": 1},
            "vision": {"param2": 2},
        }
        
        wrapper.load_state_dict(state_dict)
        
        mock_llm_opt.load_state_dict.assert_called_once_with({"param1": 1})
        mock_vision_opt.load_state_dict.assert_called_once_with({"param2": 2})
    
    def test_load_state_dict_handles_missing(self):
        """Test load_state_dict handles missing modules gracefully."""
        from megatron.bridge.training.mimo_checkpointing import MimoOptimizerWrapper
        
        mock_llm_opt = MagicMock()
        mock_vision_opt = MagicMock()
        
        optimizers = {"llm": mock_llm_opt, "vision": mock_vision_opt}
        wrapper = MimoOptimizerWrapper(optimizers)
        
        # Only llm in state_dict
        state_dict = {"llm": {"param1": 1}}
        
        wrapper.load_state_dict(state_dict)
        
        mock_llm_opt.load_state_dict.assert_called_once()
        mock_vision_opt.load_state_dict.assert_not_called()
    
    def test_load_state_dict_none(self):
        """Test load_state_dict handles None gracefully."""
        from megatron.bridge.training.mimo_checkpointing import MimoOptimizerWrapper
        
        mock_llm_opt = MagicMock()
        optimizers = {"llm": mock_llm_opt}
        wrapper = MimoOptimizerWrapper(optimizers)
        
        wrapper.load_state_dict(None)  # Should not raise
        
        mock_llm_opt.load_state_dict.assert_not_called()


class TestSliceBatchForMimo:
    """Tests for slice_batch_for_mimo utility."""
    
    def test_slice_batch_basic(self):
        """Test basic batch slicing."""
        import torch
        from megatron.bridge.data.mimo.dp_utils import slice_batch_for_mimo
        
        batch = {
            "tokens": torch.arange(32).reshape(8, 4),
            "labels": torch.arange(32).reshape(8, 4),
        }
        
        # DP size 4, rank 0 should get first 2 samples
        sliced = slice_batch_for_mimo(batch, dp_rank=0, dp_size=4)
        
        assert sliced["tokens"].shape[0] == 2
        assert sliced["labels"].shape[0] == 2
        assert sliced["tokens"][0, 0].item() == 0
    
    def test_slice_batch_different_ranks(self):
        """Test slicing for different DP ranks."""
        import torch
        from megatron.bridge.data.mimo.dp_utils import slice_batch_for_mimo
        
        batch = {"tokens": torch.arange(16).reshape(4, 4)}
        
        # DP size 2, rank 1 should get second half
        sliced = slice_batch_for_mimo(batch, dp_rank=1, dp_size=2)
        
        assert sliced["tokens"].shape[0] == 2
        assert sliced["tokens"][0, 0].item() == 8
    
    def test_slice_batch_dp_size_one(self):
        """Test that dp_size=1 returns original batch."""
        import torch
        from megatron.bridge.data.mimo.dp_utils import slice_batch_for_mimo
        
        batch = {"tokens": torch.arange(16).reshape(4, 4)}
        
        sliced = slice_batch_for_mimo(batch, dp_rank=0, dp_size=1)
        
        assert sliced["tokens"].shape[0] == 4
        assert torch.equal(sliced["tokens"], batch["tokens"])
    
    def test_slice_batch_handles_lists(self):
        """Test slicing handles list values."""
        import torch
        from megatron.bridge.data.mimo.dp_utils import slice_batch_for_mimo
        
        batch = {
            "tokens": torch.arange(16).reshape(4, 4),
            "metadata": ["a", "b", "c", "d"],
        }
        
        sliced = slice_batch_for_mimo(batch, dp_rank=0, dp_size=2)
        
        assert sliced["tokens"].shape[0] == 2
        assert sliced["metadata"] == ["a", "b"]
    
    def test_slice_batch_raises_on_indivisible(self):
        """Test error on batch size not divisible by dp_size."""
        import torch
        from megatron.bridge.data.mimo.dp_utils import slice_batch_for_mimo
        
        batch = {"tokens": torch.arange(12).reshape(3, 4)}
        
        with pytest.raises(ValueError, match="not divisible"):
            slice_batch_for_mimo(batch, dp_rank=0, dp_size=2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
