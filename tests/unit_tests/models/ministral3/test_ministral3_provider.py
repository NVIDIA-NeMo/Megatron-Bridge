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

import pytest
import torch

from megatron.bridge.models.ministral3.ministral3_provider import (
    Ministral3ModelProvider,
    Ministral3ModelProvider3B,
    Ministral3ModelProvider8B,
    Ministral3ModelProvider14B,
    MinistralTEDotProductAttention,
)


pytestmark = pytest.mark.unit


class TestMinistral3ModelProvider:
    """Test cases for Ministral3ModelProvider base class."""

    def test_ministral3_model_provider_initialization(self):
        """Test Ministral3ModelProvider can be initialized with default values."""
        provider = Ministral3ModelProvider(
            num_layers=26,
            hidden_size=3072,
            num_attention_heads=32,
        )

        # Check required transformer config fields
        assert provider.num_layers == 26
        assert provider.hidden_size == 3072
        assert provider.num_attention_heads == 32

        # Check Ministral3-inherited defaults from Mistral
        assert provider.normalization == "RMSNorm"
        assert provider.gated_linear_unit is True
        assert provider.position_embedding_type == "yarn"
        assert provider.add_bias_linear is False
        assert provider.seq_length == 32768  # Default
        assert provider.attention_dropout == 0.0
        assert provider.hidden_dropout == 0.0
        assert provider.share_embeddings_and_output_weights is False
        assert provider.layernorm_epsilon == 1e-5
        assert provider.rotary_base == 1000000
        assert provider.num_query_groups == 8

    def test_ministral3_vl_specific_defaults(self):
        """Test Ministral3ModelProvider VL-specific default configuration."""
        provider = Ministral3ModelProvider(
            num_layers=26,
            hidden_size=3072,
            num_attention_heads=32,
        )

        # Check VL-specific defaults
        assert provider.scatter_embedding_sequence_parallel is False
        assert provider.image_token_id == 10

        # Check freeze options defaults
        assert provider.freeze_language_model is False
        assert provider.freeze_vision_model is False
        assert provider.freeze_vision_projection is False

    def test_ministral3_yarn_rope_defaults(self):
        """Test Ministral3ModelProvider YARN RoPE default configuration."""
        provider = Ministral3ModelProvider(
            num_layers=26,
            hidden_size=3072,
            num_attention_heads=32,
        )

        # Check YARN RoPE defaults
        assert provider.yarn_rotary_scaling_factor == 16.0
        assert provider.yarn_original_max_position_embeddings == 16384
        assert provider.yarn_beta_fast == 32.0
        assert provider.yarn_beta_slow == 1.0
        assert provider.yarn_correction_range_round_to_int is False
        assert provider.yarn_mscale == 1.0
        assert provider.yarn_mscale_all_dim == 1.0

    def test_ministral3_custom_image_token_id(self):
        """Test Ministral3ModelProvider with custom image token ID."""
        provider = Ministral3ModelProvider(
            num_layers=26,
            hidden_size=3072,
            num_attention_heads=32,
            image_token_id=100,
        )

        assert provider.image_token_id == 100

    def test_ministral3_freeze_options(self):
        """Test Ministral3ModelProvider with freeze options."""
        provider = Ministral3ModelProvider(
            num_layers=26,
            hidden_size=3072,
            num_attention_heads=32,
            freeze_language_model=True,
            freeze_vision_model=True,
            freeze_vision_projection=True,
        )

        assert provider.freeze_language_model is True
        assert provider.freeze_vision_model is True
        assert provider.freeze_vision_projection is True

    def test_ministral3_provide_method_exists(self):
        """Test that provide method exists and is callable."""
        provider = Ministral3ModelProvider(
            num_layers=26,
            hidden_size=3072,
            num_attention_heads=32,
        )

        assert hasattr(provider, "provide")
        assert callable(provider.provide)

    def test_ministral3_provide_language_model_method_exists(self):
        """Test that provide_language_model method exists and is callable."""
        provider = Ministral3ModelProvider(
            num_layers=26,
            hidden_size=3072,
            num_attention_heads=32,
        )

        assert hasattr(provider, "provide_language_model")
        assert callable(provider.provide_language_model)


class TestMinistral3ModelProvider3B:
    """Test cases for Ministral3ModelProvider3B."""

    def test_ministral3_3b_initialization(self):
        """Test Ministral3ModelProvider3B can be initialized with correct defaults."""
        provider = Ministral3ModelProvider3B()

        # Check 3B specific configuration
        assert provider.hidden_size == 3072
        assert provider.ffn_hidden_size == 9216
        assert provider.num_layers == 26
        assert provider.share_embeddings_and_output_weights is True


class TestMinistral3ModelProvider8B:
    """Test cases for Ministral3ModelProvider8B."""

    def test_ministral3_8b_initialization(self):
        """Test Ministral3ModelProvider8B can be initialized with correct defaults."""
        provider = Ministral3ModelProvider8B()

        # Check 8B specific configuration
        assert provider.hidden_size == 4096
        assert provider.ffn_hidden_size == 14336
        assert provider.num_layers == 34


class TestMinistral3ModelProvider14B:
    """Test cases for Ministral3ModelProvider14B."""

    def test_ministral3_14b_initialization(self):
        """Test Ministral3ModelProvider14B can be initialized with correct defaults."""
        provider = Ministral3ModelProvider14B()

        # Check 14B specific configuration
        assert provider.hidden_size == 5120
        assert provider.ffn_hidden_size == 16384
        assert provider.num_layers == 40
        assert provider.rotary_base == 1000000000.0


class TestGetLlama4AttnScale:
    """Test cases for _get_llama_4_attn_scale function used in MinistralTEDotProductAttention.

    This function computes attention scaling based on Llama 4 attention parameters.
    The key change in PR 1997 is that it now handles different query shapes for
    packed (3D) vs unpacked (4D) tensors.
    """

    # Use the actual production implementation
    _get_llama_4_attn_scale = staticmethod(MinistralTEDotProductAttention._get_llama_4_attn_scale)

    def test_unpacked_4d_query_shape(self):
        """Test attention scaling with unpacked 4D query shape [seq_len, batch, num_heads, head_dim]."""
        seq_len = 8
        batch_size = 2
        num_heads = 4
        head_dim = 64

        positions_ids = torch.arange(seq_len)
        beta = 0.1
        max_position_embeddings = 16384
        query_shape = (seq_len, batch_size, num_heads, head_dim)

        scaling = self._get_llama_4_attn_scale(positions_ids, beta, max_position_embeddings, query_shape)

        # Output should have shape [seq_len, 1, 1, 1] for broadcasting
        assert scaling.shape == (seq_len, 1, 1, 1)

        # First position should have scaling = 1 (since log(1 + 0) = 0)
        expected_first = 1 + beta * torch.log(torch.tensor(1.0))
        assert torch.isclose(scaling[0, 0, 0, 0], expected_first, atol=1e-6)

    def test_packed_3d_query_shape(self):
        """Test attention scaling with packed 3D query shape [seq_len, num_heads, head_dim]."""
        seq_len = 16
        num_heads = 8
        head_dim = 32

        positions_ids = torch.arange(seq_len)
        beta = 0.2
        max_position_embeddings = 8192
        query_shape = (seq_len, num_heads, head_dim)

        scaling = self._get_llama_4_attn_scale(positions_ids, beta, max_position_embeddings, query_shape)

        # Output should have shape [seq_len, 1, 1] for broadcasting (3D - 1 = 2 dims added)
        assert scaling.shape == (seq_len, 1, 1)

        # Verify scaling values are computed correctly
        expected = 1 + beta * torch.log(1 + torch.floor(positions_ids / max_position_embeddings))
        assert torch.allclose(scaling.squeeze(), expected, atol=1e-6)

    def test_scaling_formula_correctness(self):
        """Test that the scaling formula matches expected Llama 4 attention scaling."""
        positions_ids = torch.tensor([0, 1, 100, 1000, 16384, 32768])
        beta = 0.15
        max_position_embeddings = 16384
        query_shape = (6, 1, 1, 1)

        scaling = self._get_llama_4_attn_scale(positions_ids, beta, max_position_embeddings, query_shape)

        # Manual computation of expected values
        # For position 0: 1 + 0.15 * log(1 + 0) = 1
        # For position 16384: 1 + 0.15 * log(1 + 1) = 1 + 0.15 * log(2)
        # For position 32768: 1 + 0.15 * log(1 + 2) = 1 + 0.15 * log(3)

        expected_0 = 1.0
        expected_16384 = 1 + beta * torch.log(torch.tensor(2.0))
        expected_32768 = 1 + beta * torch.log(torch.tensor(3.0))

        assert torch.isclose(scaling[0].squeeze(), torch.tensor(expected_0), atol=1e-6)
        assert torch.isclose(scaling[4].squeeze(), expected_16384, atol=1e-6)
        assert torch.isclose(scaling[5].squeeze(), expected_32768, atol=1e-6)

    def test_beta_zero_returns_ones(self):
        """Test that beta=0 returns all ones (no scaling)."""
        positions_ids = torch.arange(10)
        beta = 0.0
        max_position_embeddings = 4096
        query_shape = (10, 4, 64)

        scaling = self._get_llama_4_attn_scale(positions_ids, beta, max_position_embeddings, query_shape)

        assert torch.allclose(scaling.squeeze(), torch.ones(10), atol=1e-6)

    def test_different_query_shapes_get_correct_dims(self):
        """Test that different query shapes result in correct number of dimensions added."""
        positions_ids = torch.arange(4)
        beta = 0.1
        max_position_embeddings = 1000

        # 2D query shape
        query_shape_2d = (4, 32)
        scaling_2d = self._get_llama_4_attn_scale(positions_ids, beta, max_position_embeddings, query_shape_2d)
        assert scaling_2d.shape == (4, 1)  # 2-1 = 1 dim added

        # 3D query shape (packed THD)
        query_shape_3d = (4, 8, 32)
        scaling_3d = self._get_llama_4_attn_scale(positions_ids, beta, max_position_embeddings, query_shape_3d)
        assert scaling_3d.shape == (4, 1, 1)  # 3-1 = 2 dims added

        # 4D query shape (unpacked BSHD)
        query_shape_4d = (4, 2, 8, 32)
        scaling_4d = self._get_llama_4_attn_scale(positions_ids, beta, max_position_embeddings, query_shape_4d)
        assert scaling_4d.shape == (4, 1, 1, 1)  # 4-1 = 3 dims added

    def test_broadcasting_compatibility(self):
        """Test that scaling tensor is broadcastable to query tensor."""
        seq_len = 8
        num_heads = 4
        head_dim = 64

        positions_ids = torch.arange(seq_len)
        beta = 0.1
        max_position_embeddings = 16384

        # Test for 3D packed format
        query_3d = torch.randn(seq_len, num_heads, head_dim)
        scaling_3d = self._get_llama_4_attn_scale(positions_ids, beta, max_position_embeddings, query_3d.shape)

        # Broadcasting should work
        result_3d = query_3d * scaling_3d.to(query_3d.dtype)
        assert result_3d.shape == query_3d.shape

        # Test for 4D unpacked format
        batch = 2
        query_4d = torch.randn(seq_len, batch, num_heads, head_dim)
        scaling_4d = self._get_llama_4_attn_scale(positions_ids, beta, max_position_embeddings, query_4d.shape)

        # Broadcasting should work
        result_4d = query_4d * scaling_4d.to(query_4d.dtype)
        assert result_4d.shape == query_4d.shape

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_gpu_tensor_support(self):
        """Test that the function works with GPU tensors if available."""
        positions_ids = torch.arange(8, device="cuda")
        beta = 0.1
        max_position_embeddings = 1024
        query_shape = (8, 4, 32)

        scaling = self._get_llama_4_attn_scale(positions_ids, beta, max_position_embeddings, query_shape)

        assert scaling.device.type == "cuda"
        assert scaling.shape == (8, 1, 1)

    def test_dtype_preservation(self):
        """Test that output dtype matches input positions_ids dtype."""
        positions_ids_float32 = torch.arange(4, dtype=torch.float32)
        positions_ids_float64 = torch.arange(4, dtype=torch.float64)
        beta = 0.1
        max_position_embeddings = 100
        query_shape = (4, 2, 8)

        scaling_32 = self._get_llama_4_attn_scale(positions_ids_float32, beta, max_position_embeddings, query_shape)
        scaling_64 = self._get_llama_4_attn_scale(positions_ids_float64, beta, max_position_embeddings, query_shape)

        # Note: torch.arange with int creates int tensors, but the function uses float operations
        # The scaling result will be float due to log operation
        assert scaling_32.dtype == torch.float32
        assert scaling_64.dtype == torch.float64
