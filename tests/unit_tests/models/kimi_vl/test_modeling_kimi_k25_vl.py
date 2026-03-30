# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Unit tests for Kimi K2.5 VL model, focusing on _merge_input_ids_with_image_features."""

from unittest.mock import Mock, patch

import pytest
import torch


# Constants matching Kimi K2.5 defaults
IMAGE_TOKEN_ID = 163605
PAD_TOKEN_ID = 163839
IGNORE_INDEX = -100
HIDDEN_DIM = 64  # Small for testing


class MergeTestHelper:
    """Lightweight wrapper to call _merge_input_ids_with_image_features without full model."""

    def __init__(self):
        self.media_placeholder_token_id = IMAGE_TOKEN_ID
        self.config = Mock()
        self.config.pad_token_id = PAD_TOKEN_ID
        self.config.ignore_index = IGNORE_INDEX

    def merge(
        self, image_features, inputs_embeds, input_ids, attention_mask=None, labels=None, target_seq_length=None
    ):
        from megatron.bridge.models.kimi_vl.modeling_kimi_k25_vl import KimiK25VLModel

        return KimiK25VLModel._merge_input_ids_with_image_features(
            self, image_features, inputs_embeds, input_ids, attention_mask, labels, target_seq_length
        )


@pytest.fixture
def helper():
    return MergeTestHelper()


def _make_inputs(batch_size, seq_len, hidden_dim=HIDDEN_DIM):
    """Create basic input tensors."""
    input_ids = torch.randint(1000, 2000, (batch_size, seq_len))
    inputs_embeds = torch.randn(batch_size, seq_len, hidden_dim)
    return input_ids, inputs_embeds


def _make_image_features(num_features, hidden_dim=HIDDEN_DIM):
    """Create a single image's features."""
    return torch.randn(num_features, hidden_dim)


# ===========================================================================
# Pre-expanded mode: num_placeholders == total_image_features
# ===========================================================================
class TestMergePreExpanded:
    """Test pre-expanded mode where placeholders already match features 1:1."""

    def test_single_image_shape_unchanged(self, helper):
        """Output shape should equal input shape (no expansion)."""
        batch_size, text_len, num_features = 1, 5, 8
        input_ids, inputs_embeds = _make_inputs(batch_size, text_len + num_features)
        # Place image tokens at positions 2..9
        input_ids[0, 2 : 2 + num_features] = IMAGE_TOKEN_ID

        image_features = [_make_image_features(num_features)]
        embedding, attn_mask, labels, pos_ids = helper.merge(image_features, inputs_embeds, input_ids)

        assert embedding.shape == inputs_embeds.shape
        assert attn_mask.shape == (batch_size, text_len + num_features)
        assert pos_ids.shape == (batch_size, text_len + num_features)

    def test_features_placed_at_placeholder_positions(self, helper):
        """Image features should replace placeholder embeddings exactly."""
        batch_size, num_features = 1, 4
        input_ids = torch.tensor([[100, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 200]])
        inputs_embeds = torch.zeros(batch_size, 6, HIDDEN_DIM)

        feat = torch.ones(num_features, HIDDEN_DIM) * 42.0
        embedding, _, _, _ = helper.merge([feat], inputs_embeds, input_ids)

        # Positions 1-4 should have the image features
        assert torch.allclose(embedding[0, 1:5], feat)
        # Positions 0 and 5 should remain zero (original text embeddings)
        assert torch.allclose(embedding[0, 0], torch.zeros(HIDDEN_DIM))
        assert torch.allclose(embedding[0, 5], torch.zeros(HIDDEN_DIM))

    def test_multi_image(self, helper):
        """Multiple images with pre-expanded placeholders."""
        # 2 images: 3 features + 2 features = 5 placeholders
        input_ids = torch.tensor(
            [[100, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 200, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 300]]
        )
        inputs_embeds = torch.zeros(1, 8, HIDDEN_DIM)

        feat1 = torch.ones(3, HIDDEN_DIM) * 1.0
        feat2 = torch.ones(2, HIDDEN_DIM) * 2.0
        embedding, _, _, _ = helper.merge([feat1, feat2], inputs_embeds, input_ids)

        assert torch.allclose(embedding[0, 1:4], feat1)
        assert torch.allclose(embedding[0, 5:7], feat2)

    def test_with_labels(self, helper):
        """Labels at image positions should be set to ignore_index."""
        input_ids = torch.tensor([[100, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 200]])
        inputs_embeds = torch.randn(1, 4, HIDDEN_DIM)
        labels = torch.tensor([[10, 20, 30, 40]])

        feat = _make_image_features(2)
        _, _, final_labels, _ = helper.merge([feat], inputs_embeds, input_ids, labels=labels)

        assert final_labels[0, 0].item() == 10
        assert final_labels[0, 1].item() == IGNORE_INDEX
        assert final_labels[0, 2].item() == IGNORE_INDEX
        assert final_labels[0, 3].item() == 40

    def test_without_labels(self, helper):
        """Labels should be None when not provided."""
        input_ids = torch.tensor([[IMAGE_TOKEN_ID, IMAGE_TOKEN_ID]])
        inputs_embeds = torch.randn(1, 2, HIDDEN_DIM)
        feat = _make_image_features(2)

        _, _, final_labels, _ = helper.merge([feat], inputs_embeds, input_ids)
        assert final_labels is None

    def test_position_ids(self, helper):
        """Position IDs should be sequential for non-padded tokens."""
        input_ids = torch.tensor([[100, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 200]])
        inputs_embeds = torch.randn(1, 4, HIDDEN_DIM)

        _, _, _, pos_ids = helper.merge([_make_image_features(2)], inputs_embeds, input_ids)

        assert pos_ids[0].tolist() == [0, 1, 2, 3]

    def test_batch_size_2(self, helper):
        """Pre-expanded mode with batch_size=2."""
        # Each sample has 2 image placeholders
        input_ids = torch.tensor(
            [
                [100, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 200],
                [300, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 400],
            ]
        )
        inputs_embeds = torch.zeros(2, 4, HIDDEN_DIM)

        feat1 = torch.ones(2, HIDDEN_DIM) * 1.0
        feat2 = torch.ones(2, HIDDEN_DIM) * 2.0
        embedding, _, _, _ = helper.merge([feat1, feat2], inputs_embeds, input_ids)

        assert torch.allclose(embedding[0, 1:3], feat1)
        assert torch.allclose(embedding[1, 1:3], feat2)


# ===========================================================================
# Truncated pre-expanded mode: num_images < num_placeholders < total_features
# ===========================================================================
class TestMergeTruncatedPreExpanded:
    """Test truncated pre-expanded mode where seq_length cutoff removed some placeholders."""

    def test_single_image_truncated(self, helper):
        """Truncated: 50 placeholders but image has 64 features."""
        text_tokens = 10
        num_placeholders = 50
        num_features = 64
        seq_len = text_tokens + num_placeholders

        input_ids = torch.randint(1000, 2000, (1, seq_len))
        input_ids[0, text_tokens:] = IMAGE_TOKEN_ID
        inputs_embeds = torch.zeros(1, seq_len, HIDDEN_DIM)

        feat = torch.arange(num_features * HIDDEN_DIM, dtype=torch.float).reshape(num_features, HIDDEN_DIM)
        embedding, attn_mask, _, pos_ids = helper.merge([feat], inputs_embeds, input_ids)

        # Shape should be unchanged (no expansion)
        assert embedding.shape == (1, seq_len, HIDDEN_DIM)
        # Only first 50 features should be used
        assert torch.allclose(embedding[0, text_tokens:], feat[:num_placeholders])

    def test_multi_image_last_truncated(self, helper):
        """Two images [32, 32] features but only 48 placeholders total."""
        text_tokens = 4
        num_placeholders = 48  # > 2 images but < 64 total features
        seq_len = text_tokens + num_placeholders

        input_ids = torch.randint(1000, 2000, (1, seq_len))
        input_ids[0, text_tokens:] = IMAGE_TOKEN_ID
        inputs_embeds = torch.zeros(1, seq_len, HIDDEN_DIM)

        feat1 = torch.ones(32, HIDDEN_DIM) * 1.0
        feat2 = torch.ones(32, HIDDEN_DIM) * 2.0
        embedding, _, _, _ = helper.merge([feat1, feat2], inputs_embeds, input_ids)

        assert embedding.shape == (1, seq_len, HIDDEN_DIM)
        # First image: all 32 features used
        assert torch.allclose(embedding[0, text_tokens : text_tokens + 32], feat1)
        # Second image: only first 16 features used (48 - 32 = 16)
        assert torch.allclose(embedding[0, text_tokens + 32 :], feat2[:16])


# ===========================================================================
# Dynamic expansion mode: num_placeholders <= num_images (1 per image)
# ===========================================================================
class TestMergeDynamicExpansion:
    """Test dynamic expansion mode where 1 placeholder expands to N features."""

    def test_single_image_expansion(self, helper):
        """1 placeholder should expand to N image feature tokens."""
        num_features = 8
        # input: [text, IMAGE, text] → 3 tokens
        input_ids = torch.tensor([[100, IMAGE_TOKEN_ID, 200]])
        inputs_embeds = torch.randn(1, 3, HIDDEN_DIM)

        feat = _make_image_features(num_features)
        embedding, attn_mask, labels, pos_ids = helper.merge([feat], inputs_embeds, input_ids)

        # Expanded length = 2 text tokens + 8 image features = 10
        expected_len = 2 + num_features
        assert embedding.shape == (1, expected_len, HIDDEN_DIM)
        assert attn_mask.shape == (1, expected_len)
        assert pos_ids.shape == (1, expected_len)

    def test_multi_image_expansion(self, helper):
        """2 placeholders should expand to N1 + N2 features."""
        n1, n2 = 4, 6
        # input: [text, IMAGE, text, IMAGE, text] → 5 tokens, 2 placeholders
        input_ids = torch.tensor([[100, IMAGE_TOKEN_ID, 200, IMAGE_TOKEN_ID, 300]])
        inputs_embeds = torch.randn(1, 5, HIDDEN_DIM)

        feat1 = _make_image_features(n1)
        feat2 = _make_image_features(n2)
        embedding, _, _, _ = helper.merge([feat1, feat2], inputs_embeds, input_ids)

        # Expanded: 3 text tokens + 4 + 6 image features = 13
        assert embedding.shape == (1, 13, HIDDEN_DIM)

    def test_with_labels_expansion(self, helper):
        """Labels should be expanded and image positions set to ignore_index."""
        num_features = 4
        input_ids = torch.tensor([[100, IMAGE_TOKEN_ID, 200]])
        inputs_embeds = torch.randn(1, 3, HIDDEN_DIM)
        labels = torch.tensor([[10, 20, 30]])

        feat = _make_image_features(num_features)
        _, _, final_labels, _ = helper.merge([feat], inputs_embeds, input_ids, labels=labels)

        # Expanded length: 2 text + 4 image = 6
        assert final_labels.shape[1] == 6
        # Image positions should be IGNORE_INDEX
        image_label_count = (final_labels == IGNORE_INDEX).sum().item()
        assert image_label_count >= num_features

    def test_attention_mask_all_ones_no_padding(self, helper):
        """Attention mask should be all 1s when there's no padding."""
        input_ids = torch.tensor([[100, IMAGE_TOKEN_ID, 200]])
        inputs_embeds = torch.randn(1, 3, HIDDEN_DIM)

        feat = _make_image_features(4)
        _, attn_mask, _, _ = helper.merge([feat], inputs_embeds, input_ids)

        assert attn_mask.sum().item() == attn_mask.numel()  # All 1s

    def test_position_ids_sequential(self, helper):
        """Position IDs should be sequential after expansion."""
        input_ids = torch.tensor([[100, IMAGE_TOKEN_ID, 200]])
        inputs_embeds = torch.randn(1, 3, HIDDEN_DIM)

        feat = _make_image_features(4)
        _, _, _, pos_ids = helper.merge([feat], inputs_embeds, input_ids)

        expected_len = 2 + 4  # 2 text + 4 image
        assert pos_ids[0].tolist() == list(range(expected_len))

    def test_with_target_seq_length(self, helper):
        """target_seq_length should control output size."""
        target = 20
        input_ids = torch.tensor([[100, IMAGE_TOKEN_ID, 200]])
        inputs_embeds = torch.randn(1, 3, HIDDEN_DIM)

        feat = _make_image_features(4)
        embedding, attn_mask, _, _ = helper.merge([feat], inputs_embeds, input_ids, target_seq_length=target)

        assert embedding.shape == (1, target, HIDDEN_DIM)
        assert attn_mask.shape == (1, target)

    def test_image_features_in_output(self, helper):
        """Verify image features are present in the expanded output."""
        input_ids = torch.tensor([[100, IMAGE_TOKEN_ID, 200]])
        inputs_embeds = torch.zeros(1, 3, HIDDEN_DIM)

        feat = torch.ones(4, HIDDEN_DIM) * 99.0
        embedding, _, _, _ = helper.merge([feat], inputs_embeds, input_ids)

        # At least 4 positions should have the image feature value
        has_feat = (embedding[0] == 99.0).all(dim=-1)
        assert has_feat.sum().item() == 4

    def test_text_embeddings_preserved(self, helper):
        """Verify text embeddings are preserved in the expanded output."""
        text_embed_val = 42.0
        input_ids = torch.tensor([[100, IMAGE_TOKEN_ID, 200]])
        inputs_embeds = torch.full((1, 3, HIDDEN_DIM), text_embed_val)

        feat = torch.zeros(4, HIDDEN_DIM)
        embedding, _, _, _ = helper.merge([feat], inputs_embeds, input_ids)

        # Text positions should retain their value
        has_text = (embedding[0] == text_embed_val).all(dim=-1)
        assert has_text.sum().item() == 2  # 2 text tokens


# ===========================================================================
# Model initialization tests (with mocking for dynamic module loading)
# ===========================================================================
class TestKimiK25VLModelInit:
    """Test KimiK25VLModel initialization."""

    def _make_mock_config(self):
        """Create mock config for model init."""
        config = Mock()
        config.hf_model_path = "/path/to/model"
        config.share_embeddings_and_output_weights = False
        config.sequence_parallel = False
        config.media_placeholder_token_id = IMAGE_TOKEN_ID

        mock_lm = Mock()
        mock_lm.shared_embedding_or_output_weight.return_value = None
        mock_lm.set_input_tensor = Mock()
        config.provide_language_model = Mock(return_value=mock_lm)
        return config

    @patch("megatron.bridge.models.kimi_vl.modeling_kimi_k25_vl.get_class_from_dynamic_module")
    @patch("megatron.bridge.models.kimi_vl.modeling_kimi_k25_vl.hook_hf_module_setattr_for_tp_grad_sync")
    @patch("megatron.bridge.models.hf_pretrained.safe_config_loader.safe_load_config_with_retry")
    def test_init_with_pre_process(self, mock_safe_load, mock_hook, mock_get_class):
        """Test initialization with pre_process=True creates vision components."""
        config = self._make_mock_config()

        mock_vision_config = Mock()
        mock_vision_config.merge_kernel_size = (2, 2)
        mock_safe_load_result = Mock()
        mock_safe_load_result.vision_config = mock_vision_config
        mock_safe_load.return_value = mock_safe_load_result

        # Mock dynamic class loading
        mock_encoder_cls = type("MoonViT3dEncoder", (), {"_bridge_init_patched": False})
        mock_module = Mock()
        mock_module.MoonViT3dEncoder = mock_encoder_cls

        mock_vit = Mock()
        mock_projector = Mock()

        def side_effect(name, path):
            if "MoonViT3dPretrainedModel" in name:
                cls = Mock(return_value=mock_vit)
                cls.__module__ = "test_module"
                return cls
            elif "PatchMergerMLP" in name:
                return Mock(return_value=mock_projector)
            elif "ProjectorConfig" in name:
                return Mock(return_value=Mock())
            elif "VisionTowerConfig" in name:
                return Mock(return_value=Mock())
            elif "MoonViT3dEncoder" in name:
                return mock_encoder_cls
            return Mock()

        mock_get_class.side_effect = side_effect

        with patch("importlib.import_module", return_value=mock_module):
            from megatron.bridge.models.kimi_vl.modeling_kimi_k25_vl import KimiK25VLModel

            model = KimiK25VLModel(config=config, pre_process=True, post_process=True)

        assert model.pre_process is True
        assert model.post_process is True
        assert hasattr(model, "vision_tower")
        assert hasattr(model, "mm_projector")
        assert hasattr(model, "language_model")

    def test_init_without_pre_process(self):
        """Test initialization with pre_process=False skips vision components."""
        config = self._make_mock_config()

        from megatron.bridge.models.kimi_vl.modeling_kimi_k25_vl import KimiK25VLModel

        model = KimiK25VLModel(config=config, pre_process=False, post_process=True)

        assert model.pre_process is False
        assert not hasattr(model, "vision_tower")
        assert not hasattr(model, "mm_projector")
        assert hasattr(model, "language_model")

    def test_set_input_tensor_delegates(self):
        """Test set_input_tensor delegates to language model."""
        config = self._make_mock_config()

        from megatron.bridge.models.kimi_vl.modeling_kimi_k25_vl import KimiK25VLModel

        model = KimiK25VLModel(config=config, pre_process=False, post_process=True)
        tensor = torch.randn(10, 2, 64)
        model.set_input_tensor(tensor)
        model.language_model.set_input_tensor.assert_called_once_with(tensor)


class TestKimiK25VLModelFreeze:
    """Test model freezing."""

    def test_freeze_language_model(self):
        """Test freezing language model."""
        config = Mock()
        config.hf_model_path = "/path/to/model"
        config.share_embeddings_and_output_weights = False
        config.sequence_parallel = False
        config.media_placeholder_token_id = IMAGE_TOKEN_ID

        mock_lm = Mock()
        mock_lm.shared_embedding_or_output_weight.return_value = None
        mock_lm.parameters = Mock(return_value=iter([torch.nn.Parameter(torch.randn(4, 4))]))
        config.provide_language_model = Mock(return_value=mock_lm)

        from megatron.bridge.models.kimi_vl.modeling_kimi_k25_vl import KimiK25VLModel

        model = KimiK25VLModel(config=config, pre_process=False, post_process=True)
        model.freeze(freeze_language_model=True, freeze_vision_model=False, freeze_vision_projection=False)

        mock_lm.parameters.assert_called()
