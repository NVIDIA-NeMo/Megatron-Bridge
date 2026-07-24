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
from transformers import Mistral3Config

from megatron.bridge import AutoBridge
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.ministral3.ministral3_bridge import Ministral3Bridge
from megatron.bridge.models.ministral3.ministral3_provider import Ministral3ModelProvider


@pytest.fixture
def mock_text_config():
    """Create a mock text config for Ministral3."""
    config = Mock()
    config.num_hidden_layers = 26
    config.hidden_size = 3072
    config.intermediate_size = 9216
    config.num_attention_heads = 32
    config.num_key_value_heads = 8
    config.head_dim = 96
    config.max_position_embeddings = 262144
    config.vocab_size = 131072
    config.rope_parameters = {
        "rope_theta": 1000000,
        "original_max_position_embeddings": 16384,
        "llama_4_scaling_beta": 0.5,
    }
    config.tie_word_embeddings = True
    return config


@pytest.fixture
def mock_vision_config():
    """Create a mock vision config for Ministral3."""
    config = Mock()
    config.hidden_size = 1152
    config.intermediate_size = 4304
    config.num_hidden_layers = 27
    config.num_attention_heads = 16
    config.patch_size = 14
    config.image_size = 896
    return config


@pytest.fixture
def mock_hf_config(mock_text_config, mock_vision_config):
    """Create a mock HF config for Ministral3."""
    config = Mock()
    config.text_config = mock_text_config
    config.vision_config = mock_vision_config
    config.tie_word_embeddings = True
    config.torch_dtype = torch.bfloat16

    # VL-specific token IDs
    config.image_token_index = 10

    return config


@pytest.fixture
def mock_hf_pretrained(mock_hf_config):
    """Create a mock HF pretrained VLM."""
    pretrained = Mock(spec=PreTrainedCausalLM)
    pretrained.config = mock_hf_config
    return pretrained


@pytest.fixture
def ministral3_bridge():
    """Create a Ministral3Bridge instance."""
    return Ministral3Bridge()


class TestMinistral3BridgeProviderBridge:
    """Test provider_bridge method functionality."""

    def test_provider_bridge_basic_config(self, ministral3_bridge, mock_hf_pretrained):
        """Test provider_bridge creates correct provider with basic config."""
        provider = ministral3_bridge.provider_bridge(mock_hf_pretrained)

        assert isinstance(provider, Ministral3ModelProvider)

        # Check basic transformer config
        assert provider.num_layers == 26
        assert provider.hidden_size == 3072
        assert provider.ffn_hidden_size == 9216
        assert provider.vocab_size == 131072

    def test_provider_bridge_rotary_config(self, ministral3_bridge, mock_hf_pretrained):
        """Test provider_bridge creates correct rotary configuration."""
        provider = ministral3_bridge.provider_bridge(mock_hf_pretrained)

        # Check rotary base configuration
        assert provider.rotary_base == 1000000

    def test_provider_bridge_hf_config_attribute(self, ministral3_bridge, mock_hf_pretrained):
        """Test provider_bridge sets hf_config attribute."""
        provider = ministral3_bridge.provider_bridge(mock_hf_pretrained)

        # Should set hf_config
        assert provider.hf_config is mock_hf_pretrained.config

    @pytest.mark.parametrize("text_tie_word_embeddings", [True, False])
    def test_provider_bridge_tie_word_embeddings(
        self, ministral3_bridge, mock_hf_pretrained, text_tie_word_embeddings
    ):
        """The nested language config controls input/output embedding sharing."""
        mock_hf_pretrained.config.tie_word_embeddings = not text_tie_word_embeddings
        mock_hf_pretrained.config.text_config.tie_word_embeddings = text_tie_word_embeddings

        provider = ministral3_bridge.provider_bridge(mock_hf_pretrained)

        assert provider.share_embeddings_and_output_weights is text_tie_word_embeddings
        assert mock_hf_pretrained.config.tie_word_embeddings is text_tie_word_embeddings
        assert provider.hf_config.tie_word_embeddings is text_tie_word_embeddings

    def test_provider_bridge_with_custom_vocab_size(self, ministral3_bridge, mock_hf_pretrained):
        """Test provider_bridge with custom vocabulary size."""
        mock_hf_pretrained.config.text_config.vocab_size = 150000
        provider = ministral3_bridge.provider_bridge(mock_hf_pretrained)
        assert provider.vocab_size == 150000

    def test_provider_bridge_with_custom_rope_theta(self, ministral3_bridge, mock_hf_pretrained):
        """Test provider_bridge with custom RoPE theta."""
        mock_hf_pretrained.config.text_config.rope_parameters["rope_theta"] = 500000
        provider = ministral3_bridge.provider_bridge(mock_hf_pretrained)
        assert provider.rotary_base == 500000

    def test_provider_bridge_with_different_layer_counts(self, ministral3_bridge, mock_hf_pretrained):
        """Test provider_bridge with different layer counts."""
        test_layer_counts = [26, 34, 40]

        for num_layers in test_layer_counts:
            mock_hf_pretrained.config.text_config.num_hidden_layers = num_layers
            provider = ministral3_bridge.provider_bridge(mock_hf_pretrained)
            assert provider.num_layers == num_layers


class TestMinistral3BridgeMappingRegistry:
    """Test mapping_registry method functionality."""

    def test_mapping_registry_returns_correct_type(self, ministral3_bridge):
        """Test mapping_registry returns MegatronMappingRegistry."""
        registry = ministral3_bridge.mapping_registry()

        assert isinstance(registry, MegatronMappingRegistry)

    def test_mapping_registry_contains_required_mappings(self, ministral3_bridge):
        """Test mapping_registry contains all required parameter mappings."""
        registry = ministral3_bridge.mapping_registry()

        # Extract mappings - registry should contain mappings for common parameters
        mappings = registry.mappings
        assert len(mappings) > 0

        # Check that we have mappings for embeddings, output layer, layernorms
        mapping_names = []
        for mapping in mappings:
            # Collect Megatron param pattern
            if hasattr(mapping, "megatron_param"):
                mapping_names.append(str(getattr(mapping, "megatron_param")))
            # Collect HF param pattern(s)
            hf = getattr(mapping, "hf_param", None)
            if isinstance(hf, dict):
                mapping_names.extend([str(v) for v in hf.values()])
            elif isinstance(hf, str):
                mapping_names.append(hf)

        # Should contain word embeddings mapping
        has_embeddings = any("embed_tokens" in name or "word_embeddings" in name for name in mapping_names)
        assert has_embeddings, "Should contain embeddings mapping"

        # Should contain norm layer mapping
        has_norm = any("norm" in name for name in mapping_names)
        assert has_norm, "Should contain norm layer mapping"

    def test_mapping_registry_vision_tower_params(self, ministral3_bridge):
        """Test mapping_registry handles vision tower parameters correctly."""
        registry = ministral3_bridge.mapping_registry()

        # Should contain vision tower parameter mappings
        mappings = registry.mappings
        mapping_names = []
        for mapping in mappings:
            if hasattr(mapping, "megatron_param"):
                mapping_names.append(str(getattr(mapping, "megatron_param")))
            hf = getattr(mapping, "hf_param", None)
            if isinstance(hf, dict):
                mapping_names.extend([str(v) for v in hf.values()])
            elif isinstance(hf, str):
                mapping_names.append(hf)

        has_vision_tower = any("vision_tower" in name for name in mapping_names)
        assert has_vision_tower, "Should contain vision tower parameter mappings"

    def test_mapping_registry_multimodal_projector_params(self, ministral3_bridge):
        """Test mapping_registry handles multimodal projector parameters correctly."""
        registry = ministral3_bridge.mapping_registry()

        # Should contain multimodal projector parameter mappings
        mappings = registry.mappings
        mapping_names = []
        for mapping in mappings:
            if hasattr(mapping, "megatron_param"):
                mapping_names.append(str(getattr(mapping, "megatron_param")))
            hf = getattr(mapping, "hf_param", None)
            if isinstance(hf, dict):
                mapping_names.extend([str(v) for v in hf.values()])
            elif isinstance(hf, str):
                mapping_names.append(hf)

        has_projector = any("multi_modal_projector" in name for name in mapping_names)
        assert has_projector, "Should contain multimodal projector parameter mappings"

    def test_mapping_registry_qkv_mappings(self, ministral3_bridge):
        """Test mapping_registry contains QKV parameter mappings."""
        registry = ministral3_bridge.mapping_registry()

        mappings = registry.mappings
        mapping_names = []
        for mapping in mappings:
            if hasattr(mapping, "megatron_param"):
                mapping_names.append(str(getattr(mapping, "megatron_param")))
            hf = getattr(mapping, "hf_param", None)
            if isinstance(hf, dict):
                mapping_names.extend([str(v) for v in hf.values()])
            elif isinstance(hf, str):
                mapping_names.append(hf)

        # Should contain QKV mappings
        has_qkv = any("linear_qkv" in name for name in mapping_names)
        assert has_qkv, "Should contain QKV mappings"

    def test_mapping_registry_mlp_mappings(self, ministral3_bridge):
        """Test mapping_registry contains MLP parameter mappings."""
        registry = ministral3_bridge.mapping_registry()

        mappings = registry.mappings
        mapping_names = []
        for mapping in mappings:
            if hasattr(mapping, "megatron_param"):
                mapping_names.append(str(getattr(mapping, "megatron_param")))
            hf = getattr(mapping, "hf_param", None)
            if isinstance(hf, dict):
                mapping_names.extend([str(v) for v in hf.values()])
            elif isinstance(hf, str):
                mapping_names.append(hf)

        # Should contain MLP mappings
        has_mlp = any("mlp" in name for name in mapping_names)
        assert has_mlp, "Should contain MLP mappings"

    def test_mapping_registry_attention_mappings(self, ministral3_bridge):
        """Test mapping_registry contains attention parameter mappings."""
        registry = ministral3_bridge.mapping_registry()

        mappings = registry.mappings
        mapping_names = []
        for mapping in mappings:
            if hasattr(mapping, "megatron_param"):
                mapping_names.append(str(getattr(mapping, "megatron_param")))
            hf = getattr(mapping, "hf_param", None)
            if isinstance(hf, dict):
                mapping_names.extend([str(v) for v in hf.values()])
            elif isinstance(hf, str):
                mapping_names.append(hf)

        # Should contain attention mappings
        has_attention = any("self_attn" in name or "self_attention" in name for name in mapping_names)
        assert has_attention, "Should contain attention mappings"


class TestMinistral3BridgeEdgeCases:
    """Test edge cases and error conditions."""

    def test_provider_bridge_with_minimal_config(self, ministral3_bridge):
        """Test provider_bridge with minimal HF config."""
        minimal_pretrained = Mock(spec=PreTrainedCausalLM)
        minimal_config = Mock()

        # Create minimal text config
        text_config = Mock()
        text_config.num_hidden_layers = 26
        text_config.hidden_size = 3072
        text_config.intermediate_size = 9216
        text_config.num_attention_heads = 32
        text_config.num_key_value_heads = 8
        text_config.head_dim = 96
        text_config.max_position_embeddings = 262144
        text_config.vocab_size = 131072
        text_config.rope_parameters = {"rope_theta": 1000000}
        text_config.tie_word_embeddings = False

        minimal_config.text_config = text_config
        minimal_config.tie_word_embeddings = False
        minimal_config.torch_dtype = torch.bfloat16
        minimal_config.image_token_index = 10
        minimal_pretrained.config = minimal_config

        provider = ministral3_bridge.provider_bridge(minimal_pretrained)

        assert isinstance(provider, Ministral3ModelProvider)
        assert provider.num_layers == 26
        assert provider.hidden_size == 3072

    def test_provider_bridge_with_different_hidden_sizes(self, ministral3_bridge, mock_hf_pretrained):
        """Test provider_bridge with different hidden sizes."""
        test_hidden_sizes = [3072, 4096, 5120]

        for hidden_size in test_hidden_sizes:
            mock_hf_pretrained.config.text_config.hidden_size = hidden_size
            provider = ministral3_bridge.provider_bridge(mock_hf_pretrained)
            assert provider.hidden_size == hidden_size

    def test_provider_bridge_with_different_ffn_sizes(self, ministral3_bridge, mock_hf_pretrained):
        """Test provider_bridge with different FFN intermediate sizes."""
        test_ffn_sizes = [9216, 14336, 16384]

        for ffn_size in test_ffn_sizes:
            mock_hf_pretrained.config.text_config.intermediate_size = ffn_size
            provider = ministral3_bridge.provider_bridge(mock_hf_pretrained)
            assert provider.ffn_hidden_size == ffn_size


class TestMinistral3BridgeCompatibility:
    """Test compatibility with different HF model configurations."""

    def test_provider_bridge_with_group_query_attention(self, ministral3_bridge, mock_hf_pretrained):
        """Test provider_bridge with group query attention (default for Ministral3)."""
        mock_hf_pretrained.config.text_config.num_attention_heads = 32
        mock_hf_pretrained.config.text_config.num_key_value_heads = 8

        provider = ministral3_bridge.provider_bridge(mock_hf_pretrained)

        # Ministral3 uses GQA by default
        assert provider.num_attention_heads == 32
        # num_query_groups should be set from provider defaults

    def test_provider_bridge_with_different_vocab_sizes(self, ministral3_bridge, mock_hf_pretrained):
        """Test provider_bridge with different vocabulary sizes."""
        test_vocab_sizes = [100000, 131072, 150000]

        for vocab_size in test_vocab_sizes:
            mock_hf_pretrained.config.text_config.vocab_size = vocab_size
            provider = ministral3_bridge.provider_bridge(mock_hf_pretrained)
            assert provider.vocab_size == vocab_size

    @pytest.mark.parametrize(
        "num_layers,hidden_size,ffn_hidden_size,rope_theta,tie_word_embeddings",
        [
            (26, 3072, 9216, 1000000.0, True),
            (34, 4096, 14336, 1000000.0, False),
            (40, 5120, 16384, 1000000000.0, False),
        ],
        ids=["3b", "8b", "14b"],
    )
    def test_provider_bridge_exact_base_size_configs(
        self,
        ministral3_bridge,
        mock_hf_pretrained,
        num_layers,
        hidden_size,
        ffn_hidden_size,
        rope_theta,
        tie_word_embeddings,
    ):
        """Map the architecture and embedding-sharing values of each Base size."""
        text_config = mock_hf_pretrained.config.text_config
        text_config.num_hidden_layers = num_layers
        text_config.hidden_size = hidden_size
        text_config.intermediate_size = ffn_hidden_size
        text_config.rope_parameters["rope_theta"] = rope_theta
        text_config.tie_word_embeddings = tie_word_embeddings

        provider = ministral3_bridge.provider_bridge(mock_hf_pretrained)

        assert provider.num_layers == num_layers
        assert provider.hidden_size == hidden_size
        assert provider.ffn_hidden_size == ffn_hidden_size
        assert provider.rotary_base == rope_theta
        assert provider.share_embeddings_and_output_weights is tie_word_embeddings

    def test_provider_bridge_uses_composite_config_contract(self, ministral3_bridge):
        """Test the real composite config maps text dimensions and top-level fields."""
        hf_config = Mistral3Config(
            architectures=["Mistral3ForConditionalGeneration"],
            dtype="float16",
            image_token_index=42,
            tie_word_embeddings=False,
            text_config={
                "model_type": "ministral3",
                "hidden_size": 256,
                "intermediate_size": 768,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 64,
                "max_position_embeddings": 4096,
                "vocab_size": 16384,
                "rope_parameters": {"rope_type": "default", "rope_theta": 1000000.0},
                "tie_word_embeddings": True,
            },
            vision_config={
                "hidden_size": 128,
                "intermediate_size": 256,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "patch_size": 14,
                "image_size": 28,
            },
        )
        hf_pretrained = Mock(spec=PreTrainedCausalLM)
        hf_pretrained.config = hf_config

        provider = ministral3_bridge.provider_bridge(hf_pretrained)

        assert provider.hidden_size == hf_config.text_config.hidden_size
        assert provider.ffn_hidden_size == hf_config.text_config.intermediate_size
        assert provider.num_layers == hf_config.text_config.num_hidden_layers
        assert provider.num_attention_heads == hf_config.text_config.num_attention_heads
        assert provider.num_query_groups == hf_config.text_config.num_key_value_heads
        assert provider.kv_channels == hf_config.text_config.head_dim
        assert provider.seq_length == hf_config.text_config.max_position_embeddings
        assert provider.vocab_size == hf_config.text_config.vocab_size
        assert provider.share_embeddings_and_output_weights is hf_config.text_config.tie_word_embeddings
        assert provider.params_dtype == torch.float16
        assert provider.fp16 is True
        assert provider.bf16 is False
        assert provider.image_token_id == hf_config.image_token_index
        assert provider.hf_config is hf_config

    def test_auto_config_export_preserves_composite_config_semantics(self, ministral3_bridge, tmp_path):
        """Test the production auto-config path synthesizes a valid composite config."""
        checkpoint_config = Mistral3Config(
            architectures=["Mistral3ForConditionalGeneration"],
            dtype="bfloat16",
            tie_word_embeddings=False,
            text_config={
                "model_type": "ministral3",
                "hidden_size": 256,
                "intermediate_size": 768,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 64,
                "max_position_embeddings": 4096,
                "vocab_size": 16384,
                "rope_parameters": {"rope_type": "default", "rope_theta": 1000000.0},
                "tie_word_embeddings": True,
            },
            vision_config={
                "hidden_size": 128,
                "intermediate_size": 256,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "head_dim": 32,
                "patch_size": 14,
                "image_size": 28,
            },
        )
        checkpoint_pretrained = Mock(spec=PreTrainedCausalLM)
        checkpoint_pretrained.config = checkpoint_config
        checkpoint_provider = ministral3_bridge.provider_bridge(checkpoint_pretrained)

        reference_config = Mistral3Config(
            architectures=["Mistral3ForConditionalGeneration"],
            dtype="float16",
            image_token_index=99,
            multimodal_projector_bias=True,
            projector_hidden_act="silu",
            spatial_merge_size=4,
            tie_word_embeddings=True,
            vision_feature_layer=-2,
            text_config={
                "model_type": "ministral3",
                "hidden_size": 512,
                "intermediate_size": 1536,
                "num_hidden_layers": 4,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "head_dim": 64,
                "max_position_embeddings": 8192,
                "vocab_size": 32768,
                "rope_parameters": {"rope_type": "default", "rope_theta": 2000000.0},
                "sliding_window": 2048,
                "tie_word_embeddings": True,
                "use_cache": False,
            },
            vision_config={
                "hidden_size": 192,
                "intermediate_size": 384,
                "num_hidden_layers": 3,
                "num_attention_heads": 6,
                "head_dim": 32,
                "patch_size": 16,
                "image_size": 32,
            },
        )
        reference_rope_parameters = dict(reference_config.text_config.rope_parameters)
        reference_vision_config = reference_config.vision_config.to_dict()

        checkpoint_dir = tmp_path / "checkpoint"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "run_config.yaml").write_text("model: {}\n")

        with (
            patch("transformers.AutoConfig.from_pretrained", return_value=reference_config),
            patch(
                "megatron.bridge.training.model_load_save.load_model_config",
                return_value=(checkpoint_provider, None),
            ),
        ):
            auto_bridge = AutoBridge.from_auto_config(str(checkpoint_dir), "mistralai/reference")

        synthesized_config = auto_bridge.hf_pretrained
        assert isinstance(synthesized_config, Mistral3Config)

        assert synthesized_config.text_config.hidden_size == checkpoint_provider.hidden_size
        assert synthesized_config.text_config.intermediate_size == checkpoint_provider.ffn_hidden_size
        assert synthesized_config.text_config.num_hidden_layers == checkpoint_provider.num_layers
        assert synthesized_config.text_config.num_attention_heads == checkpoint_provider.num_attention_heads
        assert synthesized_config.text_config.num_key_value_heads == checkpoint_provider.num_query_groups
        assert synthesized_config.text_config.head_dim == checkpoint_provider.kv_channels
        assert synthesized_config.text_config.max_position_embeddings == checkpoint_provider.seq_length
        assert synthesized_config.text_config.vocab_size == checkpoint_provider.vocab_size

        assert synthesized_config.architectures == ["Mistral3ForConditionalGeneration"]
        assert synthesized_config.model_type == "mistral3"
        assert synthesized_config.tie_word_embeddings is checkpoint_provider.share_embeddings_and_output_weights
        assert (
            synthesized_config.text_config.tie_word_embeddings
            is checkpoint_provider.share_embeddings_and_output_weights
        )
        assert synthesized_config.dtype == checkpoint_provider.params_dtype

        assert synthesized_config.image_token_index == reference_config.image_token_index
        assert synthesized_config.multimodal_projector_bias is reference_config.multimodal_projector_bias
        assert synthesized_config.projector_hidden_act == reference_config.projector_hidden_act
        assert synthesized_config.spatial_merge_size == reference_config.spatial_merge_size
        assert synthesized_config.vision_feature_layer == reference_config.vision_feature_layer
        assert synthesized_config.text_config.model_type == reference_config.text_config.model_type
        assert synthesized_config.text_config.rope_parameters == reference_rope_parameters
        assert synthesized_config.text_config.sliding_window == reference_config.text_config.sliding_window
        assert synthesized_config.text_config.use_cache is reference_config.text_config.use_cache
        assert synthesized_config.vision_config.to_dict() == reference_vision_config

        synthesized_provider = auto_bridge.to_megatron_provider(load_weights=False)
        assert synthesized_provider.num_attention_heads == checkpoint_provider.num_attention_heads
        assert synthesized_provider.num_query_groups == checkpoint_provider.num_query_groups
        assert synthesized_provider.kv_channels == checkpoint_provider.kv_channels
        assert synthesized_provider.share_embeddings_and_output_weights is True
        assert synthesized_provider.params_dtype == torch.bfloat16
        assert synthesized_provider.fp16 is False
        assert synthesized_provider.bf16 is True
