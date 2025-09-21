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

from megatron.bridge.models.nemotron.nemotron_provider import NemotronModelProvider, squared_relu


@pytest.mark.unit
class TestNemotronModelProvider:
    """Test cases for base NemotronModelProvider class."""

    def test_nemotron_model_provider_initialization(self):
        """Test NemotronModelProvider can be initialized with default values."""
        provider = NemotronModelProvider()

        # Check Nemotron-specific defaults
        assert provider.normalization == "LayerNorm"
        assert provider.activation_func is squared_relu
        assert provider.position_embedding_type == "rope"
        assert provider.share_embeddings_and_output_weights is False
        assert provider.add_bias_linear is False
        assert provider.hidden_dropout == 0.0
        assert provider.attention_dropout == 0.0
        assert provider.rotary_percent == 0.5
        assert provider.bias_dropout_add_fusion is False
        assert provider.layernorm_zero_centered_gamma is True
        assert provider.cross_entropy_loss_fusion is True
