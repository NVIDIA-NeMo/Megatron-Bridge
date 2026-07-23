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

from unittest.mock import Mock, patch

import pytest
import torch

from megatron.bridge.models.common.te_layers import TERowParallelLinearLayerNorm


class TestTERowParallelLinearLayerNorm:
    """Test cases for the TERowParallelLinearLayerNorm module shared by Gemma3 and EXAONE 4.0."""

    def test_forward_applies_post_layernorm(self):
        """Forward applies Post-LN to the linear output and passes the None bias through."""
        layer = TERowParallelLinearLayerNorm.__new__(TERowParallelLinearLayerNorm)
        normed = torch.randn(2, 512)
        layer.post_layernorm = Mock(return_value=normed)

        linear_output = torch.randn(2, 512)
        with patch.object(
            TERowParallelLinearLayerNorm.__bases__[0], "forward", return_value=(linear_output, None)
        ) as mock_super_forward:
            x = torch.randn(2, 1024)
            output, bias = layer.forward(x)

        mock_super_forward.assert_called_once_with(x)
        layer.post_layernorm.assert_called_once_with(linear_output)
        assert output is normed
        assert bias is None

    def test_forward_rejects_deferred_bias(self):
        """Forward raises ValueError when the wrapped linear returns a deferred bias.

        The deferred bias is added downstream by the bias-dropout-add fusion, so
        applying Post-LN before that addition would be numerically incorrect.
        """
        layer = TERowParallelLinearLayerNorm.__new__(TERowParallelLinearLayerNorm)
        layer.post_layernorm = Mock()

        with patch.object(
            TERowParallelLinearLayerNorm.__bases__[0],
            "forward",
            return_value=(torch.randn(2, 512), torch.randn(512)),
        ):
            with pytest.raises(ValueError, match="add_bias_linear=False"):
                layer.forward(torch.randn(2, 1024))

        layer.post_layernorm.assert_not_called()
