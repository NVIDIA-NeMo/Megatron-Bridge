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

"""Unit tests for GLM 4.5 Vision-Language Model (modeling_glm_45v.py)."""

from unittest.mock import Mock, patch

import pytest
import torch


@pytest.fixture
def mock_vision_config():
    """Create a mock vision config for GLM-4.5V."""
    config = Mock()
    config.hidden_size = 1152
    config.spatial_merge_size = 2
    return config


@pytest.fixture
def mock_gpt_provider(mock_vision_config):
    """Create a mock GPTModelProvider for GLM-4.5V."""
    provider = Mock()
    provider.vision_config = mock_vision_config
    provider.share_embeddings_and_output_weights = False
    provider.sequence_parallel = False
    provider.hidden_size = 4096

    # Mock the language model
    mock_lm = Mock()
    mock_lm.shared_embedding_or_output_weight.return_value = None
    mock_lm.embedding = Mock(return_value=torch.randn(10, 2, 4096))
    mock_lm.forward = Mock(return_value=torch.randn(2, 10, 4096))
    mock_lm.set_input_tensor = Mock()
    mock_lm.parameters = Mock(return_value=iter([torch.nn.Parameter(torch.randn(10, 10))]))
    provider.provide_language_model = Mock(return_value=mock_lm)

    return provider


@pytest.fixture
def mock_visual():
    """Create a mock visual module with all required subcomponents."""
    visual = Mock()
    visual.patch_embed = Mock()
    visual.patch_embed.parameters = Mock(return_value=iter([torch.nn.Parameter(torch.randn(10, 10))]))
    visual.blocks = Mock()
    visual.blocks.parameters = Mock(return_value=iter([torch.nn.Parameter(torch.randn(10, 10))]))
    visual.merger = Mock()
    visual.merger.parameters = Mock(return_value=iter([torch.nn.Parameter(torch.randn(10, 10))]))
    return visual


class TestGLM45VModelInitialization:
    """Test GLM45VModel initialization."""

    @patch("megatron.bridge.models.glm_vl.modeling_glm_45v.is_transformers_min_version")
    @patch("transformers.models.glm4v.modeling_glm4v.Glm4vVisionModel")
    @patch("megatron.bridge.models.glm_vl.modeling_glm_45v.hook_hf_module_setattr_for_tp_grad_sync")
    @patch("megatron.bridge.models.glm_vl.modeling_glm_45v.Glm4vModel")
    def test_init_with_pre_process(
        self,
        mock_glm4v_model,
        mock_hook_hf,
        mock_vision_cls,
        mock_version_check,
        mock_gpt_provider,
        mock_visual,
    ):
        """Test initialization with pre_process=True creates vision and language components."""
        mock_version_check.return_value = True
        mock_vision_cls._from_config.return_value = mock_visual

        from megatron.bridge.models.glm_vl.modeling_glm_45v import GLM45VModel

        model = GLM45VModel(config=mock_gpt_provider, pre_process=True, post_process=True, vp_stage=2)

        # Check model state
        assert model.pre_process is True
        assert model.post_process is True
        assert model.vp_stage == 2
        assert hasattr(model, "visual")
        assert hasattr(model, "language_model")

        # Check HF methods are bound
        for method in ["get_image_features", "get_video_features", "get_rope_index", "get_placeholder_mask"]:
            assert hasattr(model, method) and callable(getattr(model, method))

        # Check components were initialized correctly
        mock_vision_cls._from_config.assert_called_once_with(mock_gpt_provider.vision_config)
        mock_hook_hf.assert_called_once()
        mock_gpt_provider.provide_language_model.assert_called_once_with(
            pre_process=True, post_process=True, vp_stage=2
        )

        # Test set_input_tensor delegation
        tensor = torch.randn(10, 2, 4096)
        model.set_input_tensor(tensor)
        model.language_model.set_input_tensor.assert_called_once_with(tensor)

    @patch("megatron.bridge.models.glm_vl.modeling_glm_45v.is_transformers_min_version")
    @patch("megatron.bridge.models.glm_vl.modeling_glm_45v.Glm4vModel")
    def test_init_without_pre_process(self, mock_glm4v_model, mock_version_check, mock_gpt_provider):
        """Test initialization with pre_process=False skips vision components."""
        mock_version_check.return_value = True

        from megatron.bridge.models.glm_vl.modeling_glm_45v import GLM45VModel

        model = GLM45VModel(config=mock_gpt_provider, pre_process=False, post_process=True)

        assert model.pre_process is False
        assert not hasattr(model, "visual")
        assert hasattr(model, "language_model")

    @patch("megatron.bridge.models.glm_vl.modeling_glm_45v.is_transformers_min_version")
    def test_init_raises_on_old_transformers(self, mock_version_check, mock_gpt_provider):
        """Test initialization raises RuntimeError for old transformers versions."""
        mock_version_check.return_value = False

        from megatron.bridge.models.glm_vl.modeling_glm_45v import GLM45VModel

        with pytest.raises(RuntimeError, match="transformers version .* is not supported"):
            GLM45VModel(config=mock_gpt_provider)


class TestGLM45VModelForward:
    """Test GLM45VModel forward pass."""

    @patch("megatron.bridge.models.glm_vl.modeling_glm_45v.is_transformers_min_version")
    @patch("transformers.models.glm4v.modeling_glm4v.Glm4vVisionModel")
    @patch("megatron.bridge.models.glm_vl.modeling_glm_45v.hook_hf_module_setattr_for_tp_grad_sync")
    @patch("megatron.bridge.models.glm_vl.modeling_glm_45v.Glm4vModel")
    def test_forward_text_only(
        self,
        mock_glm4v_model,
        mock_hook_hf,
        mock_vision_cls,
        mock_version_check,
        mock_gpt_provider,
        mock_visual,
    ):
        """Test forward pass with text-only input."""
        mock_version_check.return_value = True
        mock_vision_cls._from_config.return_value = mock_visual

        from megatron.bridge.models.glm_vl.modeling_glm_45v import GLM45VModel

        model = GLM45VModel(config=mock_gpt_provider, pre_process=True, post_process=True)
        model.get_rope_index = Mock(return_value=(torch.arange(10).unsqueeze(0).expand(2, -1), torch.zeros(2, 10)))

        model.forward(input_ids=torch.randint(0, 1000, (2, 10)), attention_mask=torch.ones(2, 10))

        model.language_model.embedding.assert_called_once()
        model.language_model.forward.assert_called_once()

    @patch("megatron.bridge.models.glm_vl.modeling_glm_45v.is_transformers_min_version")
    @patch("transformers.models.glm4v.modeling_glm4v.Glm4vVisionModel")
    @patch("megatron.bridge.models.glm_vl.modeling_glm_45v.hook_hf_module_setattr_for_tp_grad_sync")
    @patch("megatron.bridge.models.glm_vl.modeling_glm_45v.Glm4vModel")
    def test_forward_with_images(
        self,
        mock_glm4v_model,
        mock_hook_hf,
        mock_vision_cls,
        mock_version_check,
        mock_gpt_provider,
        mock_visual,
    ):
        """Test forward pass with image input processes vision features."""
        mock_version_check.return_value = True
        mock_vision_cls._from_config.return_value = mock_visual

        from megatron.bridge.models.glm_vl.modeling_glm_45v import GLM45VModel

        model = GLM45VModel(config=mock_gpt_provider, pre_process=True, post_process=True)
        model.get_rope_index = Mock(return_value=(torch.arange(10).unsqueeze(0).expand(2, -1), torch.zeros(2, 10)))

        # Setup image feature mocks with correct dimensions for masked_scatter
        batch_size, seq_len, hidden_size, num_tokens = 2, 10, 4096, 4
        model.get_image_features = Mock(return_value=[torch.randn(batch_size * num_tokens, hidden_size)])
        image_mask = torch.zeros(batch_size, seq_len, hidden_size, dtype=torch.bool)
        image_mask[:, :num_tokens, :] = True
        model.get_placeholder_mask = Mock(return_value=(image_mask, None))

        model.forward(
            input_ids=torch.randint(0, 1000, (batch_size, seq_len)),
            pixel_values=torch.randn(1, 3, 224, 224),
            image_grid_thw=torch.tensor([[1, 14, 14]]),
        )

        model.get_image_features.assert_called_once()
        model.get_placeholder_mask.assert_called()

    @patch("megatron.bridge.models.glm_vl.modeling_glm_45v.is_transformers_min_version")
    @patch("transformers.models.glm4v.modeling_glm4v.Glm4vVisionModel")
    @patch("megatron.bridge.models.glm_vl.modeling_glm_45v.hook_hf_module_setattr_for_tp_grad_sync")
    @patch("megatron.bridge.models.glm_vl.modeling_glm_45v.Glm4vModel")
    def test_forward_with_videos(
        self,
        mock_glm4v_model,
        mock_hook_hf,
        mock_vision_cls,
        mock_version_check,
        mock_gpt_provider,
        mock_visual,
    ):
        """Test forward pass with video input processes video features."""
        mock_version_check.return_value = True
        mock_vision_cls._from_config.return_value = mock_visual

        from megatron.bridge.models.glm_vl.modeling_glm_45v import GLM45VModel

        model = GLM45VModel(config=mock_gpt_provider, pre_process=True, post_process=True)
        model.get_rope_index = Mock(return_value=(torch.arange(10).unsqueeze(0).expand(2, -1), torch.zeros(2, 10)))

        # Setup video feature mocks with correct dimensions
        batch_size, seq_len, hidden_size, num_tokens = 2, 10, 4096, 4
        model.get_video_features = Mock(return_value=[torch.randn(batch_size * num_tokens, hidden_size)])
        video_mask = torch.zeros(batch_size, seq_len, hidden_size, dtype=torch.bool)
        video_mask[:, :num_tokens, :] = True
        model.get_placeholder_mask = Mock(return_value=(None, video_mask))

        model.forward(
            input_ids=torch.randint(0, 1000, (batch_size, seq_len)),
            pixel_values_videos=torch.randn(1, 8, 3, 224, 224),
            video_grid_thw=torch.tensor([[8, 14, 14]]),
        )

        model.get_video_features.assert_called_once()

    @patch("megatron.bridge.models.glm_vl.modeling_glm_45v.is_transformers_min_version")
    @patch("megatron.bridge.models.glm_vl.modeling_glm_45v.Glm4vModel")
    def test_forward_without_pre_process(self, mock_glm4v_model, mock_version_check, mock_gpt_provider):
        """Test forward pass with pre_process=False skips embedding."""
        mock_version_check.return_value = True

        from megatron.bridge.models.glm_vl.modeling_glm_45v import GLM45VModel

        model = GLM45VModel(config=mock_gpt_provider, pre_process=False, post_process=True)
        model.get_rope_index = Mock(return_value=(torch.arange(10).unsqueeze(0).expand(2, -1), torch.zeros(2, 10)))

        model.forward(input_ids=torch.randint(0, 1000, (2, 10)))

        model.language_model.embedding.assert_not_called()
        model.language_model.forward.assert_called_once()


class TestGLM45VModelFreeze:
    """Test GLM45VModel freeze functionality."""

    @patch("megatron.bridge.models.glm_vl.modeling_glm_45v.is_transformers_min_version")
    @patch("transformers.models.glm4v.modeling_glm4v.Glm4vVisionModel")
    @patch("megatron.bridge.models.glm_vl.modeling_glm_45v.hook_hf_module_setattr_for_tp_grad_sync")
    @patch("megatron.bridge.models.glm_vl.modeling_glm_45v.Glm4vModel")
    def test_freeze_all_components(
        self,
        mock_glm4v_model,
        mock_hook_hf,
        mock_vision_cls,
        mock_version_check,
        mock_gpt_provider,
        mock_visual,
    ):
        """Test freezing all model components (language, vision, projection)."""
        mock_version_check.return_value = True
        mock_vision_cls._from_config.return_value = mock_visual

        from megatron.bridge.models.glm_vl.modeling_glm_45v import GLM45VModel

        model = GLM45VModel(config=mock_gpt_provider, pre_process=True)
        model.freeze(freeze_language_model=True, freeze_vision_model=True, freeze_vision_projection=True)

        # Verify all components were frozen
        model.language_model.parameters.assert_called()
        model.visual.patch_embed.parameters.assert_called()
        model.visual.blocks.parameters.assert_called()
        model.visual.merger.parameters.assert_called()

    @patch("megatron.bridge.models.glm_vl.modeling_glm_45v.is_transformers_min_version")
    @patch("megatron.bridge.models.glm_vl.modeling_glm_45v.Glm4vModel")
    def test_freeze_without_visual_module(self, mock_glm4v_model, mock_version_check, mock_gpt_provider):
        """Test freeze works safely when visual module doesn't exist (pre_process=False)."""
        mock_version_check.return_value = True

        from megatron.bridge.models.glm_vl.modeling_glm_45v import GLM45VModel

        model = GLM45VModel(config=mock_gpt_provider, pre_process=False)

        # Should not raise even when trying to freeze vision components
        model.freeze(freeze_language_model=True, freeze_vision_model=True, freeze_vision_projection=True)
        model.language_model.parameters.assert_called()
