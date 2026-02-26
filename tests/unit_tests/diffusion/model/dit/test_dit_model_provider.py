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

from unittest.mock import MagicMock, patch

import pytest
import torch

from megatron.bridge.diffusion.models.dit.dit_model import DiTCrossAttentionModel
from megatron.bridge.diffusion.models.dit.dit_model_provider import DiTModelProvider


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
class TestDiTModelProvider:
    """Test class for DiTModelProvider."""

    def setup_method(self, method):
        """Set up test fixtures before each test method."""
        self.hidden_size = 1152
        self.num_layers = 28
        self.num_attention_heads = 16

    def test_provide(self, monkeypatch):
        """Test that provide() returns a DiTCrossAttentionModel instance."""
        # Mock parallel_state functions
        from megatron.core import parallel_state

        monkeypatch.setattr(parallel_state, "is_pipeline_first_stage", lambda: True)
        monkeypatch.setattr(parallel_state, "is_pipeline_last_stage", lambda: True)
        monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_world_size", lambda: 1)
        monkeypatch.setattr(parallel_state, "get_pipeline_model_parallel_world_size", lambda: 1)
        monkeypatch.setattr(parallel_state, "get_data_parallel_world_size", lambda with_context_parallel=False: 1)
        monkeypatch.setattr(parallel_state, "get_context_parallel_world_size", lambda: 1)
        monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_group", lambda **kwargs: None)
        monkeypatch.setattr(parallel_state, "get_data_parallel_group", lambda **kwargs: None)
        monkeypatch.setattr(parallel_state, "get_context_parallel_group", lambda **kwargs: None)
        monkeypatch.setattr(parallel_state, "get_tensor_and_data_parallel_group", lambda **kwargs: None)

        # Create provider instance
        provider = DiTModelProvider(
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_attention_heads=self.num_attention_heads,
        )

        # Call provide method
        model = provider.provide()

        # Check that the model is an instance of DiTCrossAttentionModel
        assert isinstance(model, DiTCrossAttentionModel), f"Expected DiTCrossAttentionModel, got {type(model)}"

        # Check that the model config matches provider config
        assert model.config.hidden_size == self.hidden_size, (
            f"Expected hidden_size {self.hidden_size}, got {model.config.hidden_size}"
        )
        assert model.config.num_layers == self.num_layers, (
            f"Expected num_layers {self.num_layers}, got {model.config.num_layers}"
        )
        assert model.config.num_attention_heads == self.num_attention_heads, (
            f"Expected num_attention_heads {self.num_attention_heads}, got {model.config.num_attention_heads}"
        )

    def test_configure_vae(self):
        """Test that configure_vae() dynamically imports the VAE module."""
        # Create a mock VAE class
        mock_vae_instance = MagicMock()
        mock_vae_class = MagicMock()
        mock_vae_class.from_pretrained.return_value = mock_vae_instance

        # Mock the dynamic_import function
        with patch("megatron.bridge.diffusion.models.dit.dit_model_provider.dynamic_import") as mock_dynamic_import:
            mock_dynamic_import.return_value = mock_vae_class

            # Create provider instance
            provider = DiTModelProvider(
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                num_attention_heads=self.num_attention_heads,
                vae_module="megatron.bridge.diffusion.common.tokenizers.cosmos.cosmos1.causal_video_tokenizer.CausalVideoTokenizer",
                vae_name="Cosmos-0.1-Tokenizer-CV4x8x8",
                vae_cache_folder="/path/to/cache",
            )

            # Call configure_vae
            vae_result = provider.configure_vae()

            # Verify that dynamic_import was called with correct module path
            mock_dynamic_import.assert_called_once_with(
                "megatron.bridge.diffusion.common.tokenizers.cosmos.cosmos1.causal_video_tokenizer.CausalVideoTokenizer"
            )

            # Verify that from_pretrained was called with correct parameters
            mock_vae_class.from_pretrained.assert_called_once_with(
                "Cosmos-0.1-Tokenizer-CV4x8x8", cache_dir="/path/to/cache"
            )

            # Verify that the returned value is the mock VAE instance
            assert vae_result is mock_vae_instance, "Expected the VAE instance to be returned"
