#!/usr/bin/env python3
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

"""Tests for flop_utils module."""

import pytest
from dataclasses import dataclass
from unittest.mock import MagicMock

from megatron.bridge.training.utils.flop_utils import num_floating_point_operations
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.models import GPTModelProvider, T5ModelProvider
from megatron.bridge.training.tokenizers.config import TokenizerConfig


@dataclass
class MockModelConfig:
    """Mock model configuration for testing."""
    
    # Basic model parameters
    num_layers: int = 12
    hidden_size: int = 768
    seq_length: int = 1024
    num_attention_heads: int = 12
    ffn_hidden_size: int = 3072
    
    # Hybrid model parameters
    is_hybrid_model: bool = False
    hybrid_override_pattern: list = None
    hybrid_attention_ratio: float = 0.0
    hybrid_mlp_ratio: float = 0.0
    
    # Mamba parameters
    mamba_state_dim: int = 128
    mamba_head_dim: int = 64
    mamba_num_groups: int = 8
    mamba_num_heads: int = 128
    
    # Attention parameters
    group_query_attention: bool = False
    num_query_groups: int = 8
    kv_channels: int = None
    
    # MLP parameters
    gated_linear_unit: bool = False
    
    # MoE parameters
    num_moe_experts: int = None
    moe_layer_freq: int = 1
    moe_router_topk: int = 1
    moe_ffn_hidden_size: int = None
    moe_shared_expert_intermediate_size: int = None
    
    # MTP parameters
    mtp_num_layers: int = None
    
    # Multi-latent attention parameters
    multi_latent_attention: bool = False
    q_lora_rank: int = None
    qk_head_dim: int = 64
    qk_pos_emb_head_dim: int = 0
    kv_lora_rank: int = 0
    v_head_dim: int = 64


@dataclass
class MockTokenizerConfig:
    """Mock tokenizer configuration for testing."""
    
    padded_vocab_size: int = 50257


class TestNumFloatingPointOperations:
    """Test cases for num_floating_point_operations function."""

    def test_standard_transformer_model(self):
        """Test FLOPs calculation for a standard Transformer model."""
        # Create mock config
        model_config = MockModelConfig()
        tokenizer_config = MockTokenizerConfig()
        
        cfg = MagicMock()
        cfg.model = model_config
        cfg.tokenizer = tokenizer_config
        
        batch_size = 32
        
        # Calculate expected FLOPs manually for verification
        # This is a simplified calculation based on the transformer_flops function
        expansion_factor = 3 * 2 * 2  # 12x from the function
        num_layers = model_config.num_layers
        hidden_size = model_config.hidden_size
        seq_length = model_config.seq_length
        ffn_hidden_size = model_config.ffn_hidden_size
        vocab_size = tokenizer_config.padded_vocab_size
        
        # MLP FLOPs
        mlp_flops = expansion_factor * num_layers * hidden_size * ffn_hidden_size
        
        # Attention FLOPs (simplified)
        attn_flops = expansion_factor * num_layers * hidden_size * hidden_size * 2
        
        # Logits FLOPs
        logits_flops = 3 * 2 * hidden_size * vocab_size
        
        expected_flops = batch_size * seq_length * (mlp_flops + attn_flops + logits_flops)
        
        result = num_floating_point_operations(cfg, batch_size)
        
        assert result > 0
        assert isinstance(result, (int, float))
        # Allow for some tolerance due to complex calculations
        assert abs(result - expected_flops) / expected_flops < 0.1

    def test_hybrid_model_with_pattern(self):
        """Test FLOPs calculation for a hybrid model with override pattern."""
        model_config = MockModelConfig(
            is_hybrid_model=True,
            hybrid_override_pattern=['*', 'M', '*', '-', 'M', '*']  # 3 attn, 2 mamba, 1 mlp
        )
        tokenizer_config = MockTokenizerConfig()
        
        cfg = MagicMock()
        cfg.model = model_config
        cfg.tokenizer = tokenizer_config
        
        batch_size = 16
        
        result = num_floating_point_operations(cfg, batch_size)
        
        assert result > 0
        assert isinstance(result, (int, float))

    def test_hybrid_model_with_ratios(self):
        """Test FLOPs calculation for a hybrid model with attention/MLP ratios."""
        model_config = MockModelConfig(
            is_hybrid_model=True,
            hybrid_attention_ratio=0.5,  # 50% attention layers
            hybrid_mlp_ratio=0.25,       # 25% MLP layers
            num_layers=12                # 25% mamba layers (remaining)
        )
        tokenizer_config = MockTokenizerConfig()
        
        cfg = MagicMock()
        cfg.model = model_config
        cfg.tokenizer = tokenizer_config
        
        batch_size = 8
        
        result = num_floating_point_operations(cfg, batch_size)
        
        assert result > 0
        assert isinstance(result, (int, float))

    def test_hybrid_model_with_gqa(self):
        """Test FLOPs calculation for a hybrid model with Group Query Attention."""
        model_config = MockModelConfig(
            is_hybrid_model=True,
            hybrid_attention_ratio=1.0,  # All attention layers
            group_query_attention=True,
            num_query_groups=4,
            kv_channels=64
        )
        tokenizer_config = MockTokenizerConfig()
        
        cfg = MagicMock()
        cfg.model = model_config
        cfg.tokenizer = tokenizer_config
        
        batch_size = 4
        
        result = num_floating_point_operations(cfg, batch_size)
        
        assert result > 0
        assert isinstance(result, (int, float))

    def test_hybrid_model_with_swiglu(self):
        """Test FLOPs calculation for a hybrid model with SwiGLU activation."""
        model_config = MockModelConfig(
            is_hybrid_model=True,
            hybrid_mlp_ratio=1.0,  # All MLP layers
            gated_linear_unit=True
        )
        tokenizer_config = MockTokenizerConfig()
        
        cfg = MagicMock()
        cfg.model = model_config
        cfg.tokenizer = tokenizer_config
        
        batch_size = 2
        
        result = num_floating_point_operations(cfg, batch_size)
        
        assert result > 0
        assert isinstance(result, (int, float))

    def test_transformer_with_moe(self):
        """Test FLOPs calculation for a Transformer model with MoE."""
        model_config = MockModelConfig(
            num_moe_experts=8,
            moe_layer_freq=2,  # Every 2nd layer is MoE
            moe_router_topk=2,
            moe_ffn_hidden_size=4096
        )
        tokenizer_config = MockTokenizerConfig()
        
        cfg = MagicMock()
        cfg.model = model_config
        cfg.tokenizer = tokenizer_config
        
        batch_size = 32
        
        result = num_floating_point_operations(cfg, batch_size)
        
        assert result > 0
        assert isinstance(result, (int, float))

    def test_transformer_with_mtp(self):
        """Test FLOPs calculation for a Transformer model with MTP."""
        model_config = MockModelConfig(
            mtp_num_layers=4
        )
        tokenizer_config = MockTokenizerConfig()
        
        cfg = MagicMock()
        cfg.model = model_config
        cfg.tokenizer = tokenizer_config
        
        batch_size = 16
        
        result = num_floating_point_operations(cfg, batch_size)
        
        assert result > 0
        assert isinstance(result, (int, float))

    def test_transformer_with_multi_latent_attention(self):
        """Test FLOPs calculation for a Transformer model with Multi-Latent Attention."""
        model_config = MockModelConfig(
            multi_latent_attention=True,
            q_lora_rank=16,
            qk_head_dim=64,
            v_head_dim=64
        )
        tokenizer_config = MockTokenizerConfig()
        
        cfg = MagicMock()
        cfg.model = model_config
        cfg.tokenizer = tokenizer_config
        
        batch_size = 8
        
        result = num_floating_point_operations(cfg, batch_size)
        
        assert result > 0
        assert isinstance(result, (int, float))

    def test_transformer_with_multi_latent_attention_no_lora(self):
        """Test FLOPs calculation for MLA without LoRA."""
        model_config = MockModelConfig(
            multi_latent_attention=True,
            q_lora_rank=None,
            qk_head_dim=64,
            v_head_dim=64
        )
        tokenizer_config = MockTokenizerConfig()
        
        cfg = MagicMock()
        cfg.model = model_config
        cfg.tokenizer = tokenizer_config
        
        batch_size = 4
        
        result = num_floating_point_operations(cfg, batch_size)
        
        assert result > 0
        assert isinstance(result, (int, float))

    def test_different_batch_sizes(self):
        """Test that FLOPs scale linearly with batch size."""
        model_config = MockModelConfig()
        tokenizer_config = MockTokenizerConfig()
        
        cfg = MagicMock()
        cfg.model = model_config
        cfg.tokenizer = tokenizer_config
        
        batch_size_1 = 8
        batch_size_2 = 16
        
        result_1 = num_floating_point_operations(cfg, batch_size_1)
        result_2 = num_floating_point_operations(cfg, batch_size_2)
        
        # FLOPs should scale linearly with batch size
        expected_ratio = batch_size_2 / batch_size_1
        actual_ratio = result_2 / result_1
        
        assert abs(actual_ratio - expected_ratio) < 0.01

    def test_different_sequence_lengths(self):
        """Test that FLOPs scale with sequence length."""
        model_config_1 = MockModelConfig(seq_length=512)
        model_config_2 = MockModelConfig(seq_length=1024)
        tokenizer_config = MockTokenizerConfig()
        
        cfg_1 = MagicMock()
        cfg_1.model = model_config_1
        cfg_1.tokenizer = tokenizer_config
        
        cfg_2 = MagicMock()
        cfg_2.model = model_config_2
        cfg_2.tokenizer = tokenizer_config
        
        batch_size = 16
        
        result_1 = num_floating_point_operations(cfg_1, batch_size)
        result_2 = num_floating_point_operations(cfg_2, batch_size)
        
        # FLOPs should be higher for longer sequences
        assert result_2 > result_1

    def test_different_model_sizes(self):
        """Test that FLOPs scale with model size."""
        model_config_1 = MockModelConfig(
            num_layers=6,
            hidden_size=512,
            ffn_hidden_size=2048
        )
        model_config_2 = MockModelConfig(
            num_layers=12,
            hidden_size=768,
            ffn_hidden_size=3072
        )
        tokenizer_config = MockTokenizerConfig()
        
        cfg_1 = MagicMock()
        cfg_1.model = model_config_1
        cfg_1.tokenizer = tokenizer_config
        
        cfg_2 = MagicMock()
        cfg_2.model = model_config_2
        cfg_2.tokenizer = tokenizer_config
        
        batch_size = 8
        
        result_1 = num_floating_point_operations(cfg_1, batch_size)
        result_2 = num_floating_point_operations(cfg_2, batch_size)
        
        # Larger model should have more FLOPs
        assert result_2 > result_1

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with minimal model
        model_config = MockModelConfig(
            num_layers=1,
            hidden_size=64,
            seq_length=128,
            num_attention_heads=1,
            ffn_hidden_size=256
        )
        tokenizer_config = MockTokenizerConfig(padded_vocab_size=1000)
        
        cfg = MagicMock()
        cfg.model = model_config
        cfg.tokenizer = tokenizer_config
        
        batch_size = 1
        
        result = num_floating_point_operations(cfg, batch_size)
        
        assert result > 0
        assert isinstance(result, (int, float))

    def test_hybrid_model_all_mamba(self):
        """Test hybrid model with all Mamba layers."""
        model_config = MockModelConfig(
            is_hybrid_model=True,
            hybrid_override_pattern=['M'] * 12  # All Mamba layers
        )
        tokenizer_config = MockTokenizerConfig()
        
        cfg = MagicMock()
        cfg.model = model_config
        cfg.tokenizer = tokenizer_config
        
        batch_size = 4
        
        result = num_floating_point_operations(cfg, batch_size)
        
        assert result > 0
        assert isinstance(result, (int, float))

    def test_hybrid_model_all_mlp(self):
        """Test hybrid model with all MLP layers."""
        model_config = MockModelConfig(
            is_hybrid_model=True,
            hybrid_override_pattern=['-'] * 12  # All MLP layers
        )
        tokenizer_config = MockTokenizerConfig()
        
        cfg = MagicMock()
        cfg.model = model_config
        cfg.tokenizer = tokenizer_config
        
        batch_size = 4
        
        result = num_floating_point_operations(cfg, batch_size)
        
        assert result > 0
        assert isinstance(result, (int, float))

    def test_hybrid_model_all_attention(self):
        """Test hybrid model with all attention layers."""
        model_config = MockModelConfig(
            is_hybrid_model=True,
            hybrid_override_pattern=['*'] * 12  # All attention layers
        )
        tokenizer_config = MockTokenizerConfig()
        
        cfg = MagicMock()
        cfg.model = model_config
        cfg.tokenizer = tokenizer_config
        
        batch_size = 4
        
        result = num_floating_point_operations(cfg, batch_size)
        
        assert result > 0
        assert isinstance(result, (int, float))

    def test_moe_with_list_pattern(self):
        """Test MoE with list-based layer frequency pattern."""
        model_config = MockModelConfig(
            num_moe_experts=4,
            moe_layer_freq=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],  # Alternating pattern
            moe_router_topk=1
        )
        tokenizer_config = MockTokenizerConfig()
        
        cfg = MagicMock()
        cfg.model = model_config
        cfg.tokenizer = tokenizer_config
        
        batch_size = 8
        
        result = num_floating_point_operations(cfg, batch_size)
        
        assert result > 0
        assert isinstance(result, (int, float))

    def test_moe_with_shared_expert(self):
        """Test MoE with shared expert."""
        model_config = MockModelConfig(
            num_moe_experts=8,
            moe_layer_freq=2,
            moe_router_topk=2,
            moe_shared_expert_intermediate_size=1024
        )
        tokenizer_config = MockTokenizerConfig()
        
        cfg = MagicMock()
        cfg.model = model_config
        cfg.tokenizer = tokenizer_config
        
        batch_size = 16
        
        result = num_floating_point_operations(cfg, batch_size)
        
        assert result > 0
        assert isinstance(result, (int, float))

    def test_moe_with_mtp_and_last_layer_moe(self):
        """Test MoE with MTP where last layer is MoE."""
        model_config = MockModelConfig(
            num_moe_experts=4,
            moe_layer_freq=2,  # Every 2nd layer is MoE, so last layer (12th) is MoE
            mtp_num_layers=2
        )
        tokenizer_config = MockTokenizerConfig()
        
        cfg = MagicMock()
        cfg.model = model_config
        cfg.tokenizer = tokenizer_config
        
        batch_size = 8
        
        result = num_floating_point_operations(cfg, batch_size)
        
        assert result > 0
        assert isinstance(result, (int, float))

    def test_moe_with_mtp_and_last_layer_dense(self):
        """Test MoE with MTP where last layer is dense."""
        model_config = MockModelConfig(
            num_moe_experts=4,
            moe_layer_freq=3,  # Every 3rd layer is MoE, so last layer (12th) is dense
            mtp_num_layers=2
        )
        tokenizer_config = MockTokenizerConfig()
        
        cfg = MagicMock()
        cfg.model = model_config
        cfg.tokenizer = tokenizer_config
        
        batch_size = 8
        
        result = num_floating_point_operations(cfg, batch_size)
        
        assert result > 0
        assert isinstance(result, (int, float))
