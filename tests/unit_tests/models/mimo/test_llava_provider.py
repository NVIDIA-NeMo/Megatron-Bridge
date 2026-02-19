# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""Unit tests for LLaVA MIMO Provider."""

from unittest.mock import Mock

import pytest
import torch.nn.functional as F
from megatron.core.models.gpt import GPTModel
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
from megatron.core.models.vision.multimodal_projector import MultimodalProjector

from megatron.bridge.models.mimo.llava_provider import LlavaMimoProvider
from megatron.bridge.models.transformer_config import TransformerConfig


class TestLlavaMimoProvider:
    """Test cases for LlavaMimoProvider."""

    def test_initialization_with_vision_encoder(self):
        """Test LlavaMimoProvider initializes correctly with vision encoder."""
        mock_vision_encoder = Mock

        provider = LlavaMimoProvider(
            vision_encoder_module=mock_vision_encoder,
        )

        # Check language model spec was created
        assert provider.language_model_spec is not None
        assert provider.language_model_spec.module == GPTModel
        assert "config" in provider.language_model_spec.params
        assert provider.language_model_spec.params["vocab_size"] == 32256
        assert provider.language_model_spec.params["max_sequence_length"] == 4096
        assert provider.language_model_spec.params["position_embedding_type"] == "rope"

        # Check modality submodules spec was created
        assert "images" in provider.modality_submodules_spec
        vision_spec = provider.modality_submodules_spec["images"]
        assert vision_spec.module == VisionModalitySubmodules
        assert "encoders" in vision_spec.submodules
        assert "clip_encoder" in vision_spec.submodules["encoders"]

        # Check special token IDs
        assert provider.special_token_ids == {"images": 32000}

    def test_error_when_vision_encoder_missing(self):
        """Test ValueError raised when vision_encoder_module is None."""
        with pytest.raises(ValueError, match="vision_encoder_module must be provided"):
            LlavaMimoProvider(vision_encoder_module=None)

    def test_default_language_config_generation(self):
        """Test that default Vicuna-7B config is created correctly."""
        mock_vision_encoder = Mock

        provider = LlavaMimoProvider(
            vision_encoder_module=mock_vision_encoder,
        )

        config = provider.language_config
        assert config.num_layers == 32
        assert config.hidden_size == 4096
        assert config.num_attention_heads == 32
        assert config.num_query_groups == 32
        assert config.ffn_hidden_size == 11008
        assert config.normalization == "RMSNorm"
        assert config.activation_func == F.silu
        assert config.gated_linear_unit is True
        assert config.add_bias_linear is False
        assert config.attention_dropout == 0.0
        assert config.hidden_dropout == 0.0

    def test_custom_language_config(self):
        """Test that custom language config is used when provided."""
        mock_vision_encoder = Mock
        custom_config = TransformerConfig(
            num_layers=16,
            hidden_size=2048,
            num_attention_heads=16,
            num_query_groups=16,
            ffn_hidden_size=5504,
        )

        provider = LlavaMimoProvider(
            vision_encoder_module=mock_vision_encoder,
            language_config=custom_config,
        )

        # Verify custom config is used
        assert provider.language_config == custom_config
        assert provider.language_config.num_layers == 16
        assert provider.language_config.hidden_size == 2048

        # Verify language model spec uses custom config
        assert provider.language_model_spec.params["config"] == custom_config

        # Verify projector uses custom config's hidden_size
        vision_spec = provider.modality_submodules_spec["images"]
        projector_spec = vision_spec.submodules["input_projections"][0]
        assert projector_spec.params["config"].hidden_size == 2048

    def test_custom_vision_encoder_params(self):
        """Test that custom vision encoder params are propagated."""
        mock_vision_encoder = Mock
        custom_params = {"pretrained": True, "freeze": False, "resolution": 224}

        provider = LlavaMimoProvider(
            vision_encoder_module=mock_vision_encoder,
            vision_encoder_params=custom_params,
        )

        # Check encoder params were propagated
        vision_spec = provider.modality_submodules_spec["images"]
        encoder_spec = vision_spec.submodules["encoders"]["clip_encoder"]
        assert encoder_spec.params == custom_params
        assert encoder_spec.params["pretrained"] is True
        assert encoder_spec.params["resolution"] == 224

    def test_custom_vocab_and_token_ids(self):
        """Test custom vocab size and special token IDs."""
        mock_vision_encoder = Mock

        provider = LlavaMimoProvider(
            vision_encoder_module=mock_vision_encoder,
            vocab_size=50000,
            image_special_token_id=40000,
        )

        # Check custom vocab size
        assert provider.vocab_size == 50000
        assert provider.language_model_spec.params["vocab_size"] == 50000

        # Check custom special token ID
        assert provider.image_special_token_id == 40000
        assert provider.special_token_ids == {"images": 40000}

    def test_custom_vision_projector_input_size(self):
        """Test custom vision projector input size."""
        mock_vision_encoder = Mock

        provider = LlavaMimoProvider(
            vision_encoder_module=mock_vision_encoder,
            vision_projector_input_size=768,  # CLIP ViT-B/16 output size
        )

        # Check projector input size
        vision_spec = provider.modality_submodules_spec["images"]
        projector_spec = vision_spec.submodules["input_projections"][0]
        assert projector_spec.params["input_size"] == 768

    def test_vision_submodule_spec_structure(self):
        """Test vision submodule spec has correct structure."""
        mock_vision_encoder = Mock

        provider = LlavaMimoProvider(
            vision_encoder_module=mock_vision_encoder,
        )

        vision_spec = provider.modality_submodules_spec["images"]

        # Check top-level structure
        assert vision_spec.module == VisionModalitySubmodules
        assert vision_spec.params == {}
        assert "encoders" in vision_spec.submodules
        assert "input_projections" in vision_spec.submodules

        # Check encoder structure
        encoders = vision_spec.submodules["encoders"]
        assert isinstance(encoders, dict)
        assert "clip_encoder" in encoders
        encoder_spec = encoders["clip_encoder"]
        assert encoder_spec.module == mock_vision_encoder

        # Check projector structure
        projections = vision_spec.submodules["input_projections"]
        assert isinstance(projections, list)
        assert len(projections) == 1
        projector_spec = projections[0]
        assert projector_spec.module == MultimodalProjector
        assert projector_spec.params["projector_type"] == "mlp"
        assert projector_spec.params["input_size"] == 1024  # Default CLIP ViT-L/14

    def test_vision_projector_config(self):
        """Test vision projector has correct 2-layer MLP config."""
        mock_vision_encoder = Mock

        provider = LlavaMimoProvider(
            vision_encoder_module=mock_vision_encoder,
        )

        vision_spec = provider.modality_submodules_spec["images"]
        projector_spec = vision_spec.submodules["input_projections"][0]

        # Check projector config
        projector_config = projector_spec.params["config"]
        assert projector_config.num_layers == 2
        assert projector_config.hidden_size == 4096  # Matches default language hidden_size
        assert projector_config.num_attention_heads == 1

        # Check projector has MLP submodules
        assert projector_spec.params["submodules"] is not None

    def test_inherits_from_mimo_model_provider(self):
        """Test that LlavaMimoProvider inherits from MimoModelProvider."""
        mock_vision_encoder = Mock

        provider = LlavaMimoProvider(
            vision_encoder_module=mock_vision_encoder,
        )

        # Check it has MimoModelProvider attributes
        assert hasattr(provider, "language_model_spec")
        assert hasattr(provider, "modality_submodules_spec")
        assert hasattr(provider, "special_token_ids")
        assert hasattr(provider, "mimo_parallelism_config")
        assert hasattr(provider, "provide")
        assert hasattr(provider, "build_infra")

    def test_can_set_parallelism_config(self):
        """Test that parallelism config can be set on LlavaMimoProvider."""
        from megatron.bridge.models.mimo.mimo_config import (
            MimoParallelismConfig,
            ModuleParallelismConfig,
        )

        mock_vision_encoder = Mock
        mimo_config = MimoParallelismConfig(
            module_parallelisms={
                "llm": ModuleParallelismConfig(tensor_model_parallel_size=4),
            }
        )

        provider = LlavaMimoProvider(
            vision_encoder_module=mock_vision_encoder,
            mimo_parallelism_config=mimo_config,
        )

        assert provider.mimo_parallelism_config == mimo_config

    def test_can_set_freezing_options(self):
        """Test that freezing options can be set on LlavaMimoProvider."""
        mock_vision_encoder = Mock

        provider = LlavaMimoProvider(
            vision_encoder_module=mock_vision_encoder,
            freeze_language_model=True,
            freeze_modality_encoders={"images": True},
            freeze_modality_projections={"images": False},
        )

        assert provider.freeze_language_model is True
        assert provider.freeze_modality_encoders == {"images": True}
        assert provider.freeze_modality_projections == {"images": False}

    def test_defaults_match_documentation(self):
        """Test that default values match documentation."""
        mock_vision_encoder = Mock

        provider = LlavaMimoProvider(
            vision_encoder_module=mock_vision_encoder,
        )

        # Check defaults from docstring
        assert provider.vocab_size == 32256  # Vicuna vocab size
        assert provider.image_special_token_id == 32000
        assert provider.vision_projector_input_size == 1024  # CLIP ViT-L/14
        assert provider.language_config.num_layers == 32  # Vicuna-7B style
