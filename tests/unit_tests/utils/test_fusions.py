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

import logging
import os
from unittest.mock import MagicMock, patch

from megatron.bridge.utils.fusions import (
    can_enable_gradient_accumulation_fusion,
    validate_rope_fusion_compatibility,
)


class TestCanEnableGradientAccumulationFusion:
    """Test can_enable_gradient_accumulation_fusion function."""

    def test_gradient_accumulation_fusion_success(self):
        """Test gradient accumulation fusion success when kernel is available."""
        original_import = __import__

        with patch("builtins.__import__") as mock_import:
            mock_kernel = MagicMock()

            def side_effect(name, *args, **kwargs):
                if name == "fused_weight_gradient_mlp_cuda":
                    return mock_kernel
                else:
                    return original_import(name, *args, **kwargs)

            mock_import.side_effect = side_effect

            result = can_enable_gradient_accumulation_fusion()
            assert result is True

    def test_gradient_accumulation_fusion_not_available(self, caplog):
        """Test gradient accumulation fusion when kernel is not available."""
        original_import = __import__

        with (
            patch("megatron.bridge.utils.fusions.LOG_FUSION_DISABLE", True),
            patch("builtins.__import__") as mock_import,
        ):

            def side_effect(name, *args, **kwargs):
                if name == "fused_weight_gradient_mlp_cuda":
                    raise ImportError("No module named 'fused_weight_gradient_mlp_cuda'")
                else:
                    return original_import(name, *args, **kwargs)

            mock_import.side_effect = side_effect

            with caplog.at_level(logging.WARNING):
                result = can_enable_gradient_accumulation_fusion()

            assert result is False
            assert "gradient_accumulation_fusion requires FusedLayerNorm" in caplog.text


class TestValidateRopeFusionCompatibility:
    """Test validate_rope_fusion_compatibility function."""

    def test_rope_fusion_disabled(self):
        """Test validation when RoPE fusion is disabled."""
        mock_config = MagicMock()
        mock_config.apply_rope_fusion = False

        result = validate_rope_fusion_compatibility(mock_config)
        assert result is True

    def test_rope_fusion_incompatible_with_multi_latent_attention(self, caplog):
        """Test RoPE fusion incompatibility with multi_latent_attention."""
        mock_config = MagicMock()
        mock_config.apply_rope_fusion = True

        # Set attributes directly on the mock config
        mock_config.multi_latent_attention = True
        mock_config.rotary_interleaved = False

        with patch("megatron.bridge.utils.fusions.LOG_FUSION_DISABLE", True), caplog.at_level(logging.WARNING):
            result = validate_rope_fusion_compatibility(mock_config)

        assert result is True

    def test_rope_fusion_with_rotary_interleaved_success(self):
        """Test RoPE fusion with rotary_interleaved when TE version is sufficient."""
        mock_config = MagicMock()
        mock_config.apply_rope_fusion = True

        # Set attributes directly on the mock config
        mock_config.multi_latent_attention = False
        mock_config.rotary_interleaved = True

        with (
            patch("megatron.core.utils.is_te_min_version", return_value=True),
            patch("megatron.core.utils.get_te_version", return_value="2.2.0"),
        ):
            result = validate_rope_fusion_compatibility(mock_config)
            assert result is True

    def test_rope_fusion_with_rotary_interleaved_old_te_version(self, caplog):
        """Test RoPE fusion with rotary_interleaved when TE version is too old."""
        mock_config = MagicMock()
        mock_config.apply_rope_fusion = True

        # Set attributes directly on the mock config
        mock_config.multi_latent_attention = False
        mock_config.rotary_interleaved = True

        with (
            patch("megatron.core.utils.is_te_min_version", return_value=False),
            patch("megatron.core.utils.get_te_version", return_value="1.0.0"),
            patch("megatron.bridge.utils.fusions.LOG_FUSION_DISABLE", True),
        ):
            with caplog.at_level(logging.WARNING):
                result = validate_rope_fusion_compatibility(mock_config)

            assert result is False
            assert "apply_rope_fusion with rotary_interleaved requires TE >= 2.2.0.dev0" in caplog.text

    def test_rope_fusion_with_rotary_interleaved_no_te(self, caplog):
        """Test RoPE fusion with rotary_interleaved when TE is not available."""
        mock_config = MagicMock()
        mock_config.apply_rope_fusion = True
        original_import = __import__

        # Set attributes directly on the mock config
        mock_config.multi_latent_attention = False
        mock_config.rotary_interleaved = True

        with (
            patch("builtins.__import__") as mock_import,
            patch("megatron.bridge.utils.fusions.LOG_FUSION_DISABLE", True),
        ):

            def import_side_effect(name, *args, **kwargs):
                if name == "megatron.core.utils":
                    raise ImportError("No module named 'megatron.core.utils'")
                else:
                    return original_import(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect

            with caplog.at_level(logging.WARNING):
                result = validate_rope_fusion_compatibility(mock_config)

            assert result is False
            assert "apply_rope_fusion with rotary_interleaved requires Transformer Engine" in caplog.text

    def test_rope_fusion_basic_compatibility(self):
        """Test basic RoPE fusion compatibility without additional features."""
        mock_config = MagicMock()
        mock_config.apply_rope_fusion = True

        # Set attributes directly on the mock config
        mock_config.multi_latent_attention = False
        mock_config.rotary_interleaved = False

        result = validate_rope_fusion_compatibility(mock_config)
        assert result is True


class TestEnvironmentVariableHandling:
    """Test environment variable handling for fusion warnings."""

    def test_fusion_warnings_enabled_by_default(self):
        """Test that fusion warnings are enabled by default."""
        with patch.dict(os.environ, {}, clear=True):
            # Re-import the module to reset the LOG_FUSION_DISABLE variable
            import importlib

            import megatron.bridge.utils.fusions

            importlib.reload(megatron.bridge.utils.fusions)

            assert megatron.bridge.utils.fusions.LOG_FUSION_DISABLE is True

    def test_fusion_warnings_disabled_by_env_var(self):
        """Test that fusion warnings can be disabled via environment variable."""
        with patch.dict(os.environ, {"MEGATRON_SUPPRESS_FUSION_WARNINGS": "1"}):
            # Re-import the module to reset the LOG_FUSION_DISABLE variable
            import importlib

            import megatron.bridge.utils.fusions

            importlib.reload(megatron.bridge.utils.fusions)

            assert megatron.bridge.utils.fusions.LOG_FUSION_DISABLE is False

    def test_fusion_warnings_enabled_by_env_var_zero(self):
        """Test that fusion warnings are enabled when env var is set to '0'."""
        with patch.dict(os.environ, {"MEGATRON_SUPPRESS_FUSION_WARNINGS": "0"}):
            # Re-import the module to reset the LOG_FUSION_DISABLE variable
            import importlib

            import megatron.bridge.utils.fusions

            importlib.reload(megatron.bridge.utils.fusions)

            assert megatron.bridge.utils.fusions.LOG_FUSION_DISABLE is True
