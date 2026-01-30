# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""Unit tests for MIMO forward step functions."""

from unittest.mock import MagicMock, Mock, patch

import pytest
import torch


class TestLossFunc:
    """Test cases for loss_func()."""
    
    def test_loss_computation(self):
        """Test loss is computed correctly with mask."""
        from megatron.bridge.training.mimo_step import loss_func
        
        # Create test data
        output_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
        loss_mask = torch.tensor([1.0, 1.0, 0.0, 1.0])  # Mask out 3rd element
        
        total_loss, num_tokens, metrics = loss_func(loss_mask, output_tensor)
        
        # Expected: (1.0*1 + 2.0*1 + 3.0*0 + 4.0*1) = 7.0
        assert total_loss.item() == 7.0
        # Expected tokens: 3 (sum of mask)
        assert num_tokens.item() == 3
        # Check metrics dict structure
        assert 'lm loss' in metrics
    
    def test_loss_with_all_ones_mask(self):
        """Test loss with all-ones mask."""
        from megatron.bridge.training.mimo_step import loss_func
        
        output_tensor = torch.tensor([1.0, 2.0, 3.0])
        loss_mask = torch.ones(3)
        
        total_loss, num_tokens, metrics = loss_func(loss_mask, output_tensor)
        
        assert total_loss.item() == 6.0
        assert num_tokens.item() == 3
    
    def test_loss_with_all_zeros_mask(self):
        """Test loss with all-zeros mask."""
        from megatron.bridge.training.mimo_step import loss_func
        
        output_tensor = torch.tensor([1.0, 2.0, 3.0])
        loss_mask = torch.zeros(3)
        
        total_loss, num_tokens, metrics = loss_func(loss_mask, output_tensor)
        
        assert total_loss.item() == 0.0
        assert num_tokens.item() == 0


class TestGetBatch:
    """Test cases for get_batch()."""
    
    def test_returns_none_for_none_iterator(self):
        """Test returns None when iterator is None."""
        from megatron.bridge.training.mimo_step import get_batch
        
        result = get_batch(None)
        assert result is None
    
    def test_returns_none_on_stop_iteration(self):
        """Test returns None when iterator is exhausted."""
        from megatron.bridge.training.mimo_step import get_batch
        
        empty_iter = iter([])
        result = get_batch(empty_iter)
        assert result is None
    
    def test_returns_batch_from_iterator(self):
        """Test returns batch from iterator."""
        from megatron.bridge.training.mimo_step import get_batch
        
        batch = {'input_ids': torch.tensor([1, 2, 3])}
        data_iter = iter([batch])
        
        result = get_batch(data_iter)
        
        assert result is not None
        assert 'input_ids' in result


class TestForwardStep:
    """Test cases for forward_step()."""
    
    @patch('megatron.bridge.training.mimo_step.is_pp_last_stage')
    def test_forward_step_last_stage(self, mock_is_last):
        """Test forward step at last pipeline stage returns loss func."""
        from megatron.bridge.training.mimo_step import forward_step
        
        mock_is_last.return_value = True
        
        # Create mock state
        mock_state = MagicMock()
        
        # Create mock model
        mock_model = MagicMock()
        mock_output = torch.tensor([1.0, 2.0])  # Scalar-like output
        mock_loss_mask = torch.ones(2)
        mock_model.return_value = (mock_output, mock_loss_mask)
        
        # Create mock iterator
        batch = {'input_ids': torch.tensor([1, 2])}
        data_iter = iter([batch])
        
        output, loss_fn = forward_step(mock_state, data_iter, mock_model)
        
        # At last stage, should return loss function
        assert loss_fn is not None
        assert callable(loss_fn)
    
    @patch('megatron.bridge.training.mimo_step.is_pp_last_stage')
    def test_forward_step_intermediate_stage(self, mock_is_last):
        """Test forward step at intermediate stage returns None for loss func."""
        from megatron.bridge.training.mimo_step import forward_step
        
        mock_is_last.return_value = False
        
        mock_state = MagicMock()
        mock_model = MagicMock()
        mock_model.return_value = (torch.tensor([1.0]), None)
        
        batch = {'input_ids': torch.tensor([1, 2])}
        data_iter = iter([batch])
        
        output, loss_fn = forward_step(mock_state, data_iter, mock_model)
        
        # Intermediate stage should return None for loss_fn
        assert loss_fn is None
    
    @patch('megatron.bridge.training.mimo_step.is_pp_last_stage')
    def test_forward_step_rejects_dict_at_last_stage(self, mock_is_last):
        """Test forward step raises error if dict returned at last stage."""
        from megatron.bridge.training.mimo_step import forward_step
        
        mock_is_last.return_value = True
        
        mock_state = MagicMock()
        mock_model = MagicMock()
        # Return dict (incorrect for last stage)
        mock_model.return_value = ({'encoder': torch.tensor([1.0])}, None)
        
        batch = {'input_ids': torch.tensor([1, 2])}
        data_iter = iter([batch])
        
        with pytest.raises(ValueError, match="Last pipeline stage must return scalar loss"):
            forward_step(mock_state, data_iter, mock_model)
    
    @patch('megatron.bridge.training.mimo_step.is_pp_last_stage')
    def test_forward_step_uses_global_state_signature(self, mock_is_last):
        """Test forward step uses 3-arg signature with GlobalState."""
        from megatron.bridge.training.mimo_step import forward_step
        import inspect
        
        sig = inspect.signature(forward_step)
        params = list(sig.parameters.keys())
        
        # Should have state as first parameter
        assert params[0] == 'state'
        assert len(params) == 3
