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

from unittest.mock import Mock

import pytest

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.hf_pretrained.vlm import PreTrainedVLM
from megatron.bridge.models.nemotron_omni.nemotron_omni_bridge import NemotronOmniBridge
from megatron.bridge.models.nemotron_omni.nemotron_omni_provider import (
    NemotronNano3Bv3OmniModelProvider,
    NemotronNano12Bv2OmniModelProvider,
)
from megatron.bridge.models.nemotron_vl.nemotron_vl_bridge import NemotronVLBridge


@pytest.fixture
def mock_llm_config():
    cfg = Mock()
    cfg.num_hidden_layers = 52
    cfg.hidden_size = 2688
    cfg.intermediate_size = 10752
    cfg.num_attention_heads = 28
    cfg.num_key_value_heads = 4
    cfg.initializer_range = 0.02
    cfg.layer_norm_epsilon = 1e-5
    cfg.vocab_size = 262144
    cfg.max_position_embeddings = 131072
    cfg.torch_dtype = "bfloat16"
    cfg.n_routed_experts = 128
    cfg.hybrid_override_pattern = "MEMEM"
    return cfg


@pytest.fixture
def mock_sound_config():
    cfg = Mock()
    cfg.model_type = "parakeet"
    cfg.hidden_size = 1024
    cfg.projection_hidden_size = 4096
    cfg.num_hidden_layers = 24
    cfg.num_attention_heads = 8
    cfg.intermediate_size = 4096
    cfg.num_mel_bins = 128
    cfg.subsampling_factor = 8
    cfg.conv_kernel_size = 9
    cfg.to_dict = Mock(return_value={
        "hidden_size": 1024,
        "projection_hidden_size": 4096,
        "num_hidden_layers": 24,
        "num_attention_heads": 8,
        "intermediate_size": 4096,
        "num_mel_bins": 128,
        "subsampling_factor": 8,
        "conv_kernel_size": 9,
    })
    return cfg


@pytest.fixture
def mock_vision_config():
    cfg = Mock()
    cfg.args = {"separate_video_embedder": True}
    return cfg


@pytest.fixture
def mock_hf_config_with_sound(mock_llm_config, mock_sound_config):
    cfg = Mock()
    cfg.llm_config = mock_llm_config
    cfg.sound_config = mock_sound_config
    cfg.sound_context_token_id = 29
    cfg.projector_hidden_size = 20480
    # No temporal compression by default
    cfg.video_temporal_patch_size = 1
    cfg.separate_video_embedder = False
    cfg.vision_config = None
    return cfg


@pytest.fixture
def mock_hf_config_no_sound(mock_llm_config):
    cfg = Mock()
    cfg.llm_config = mock_llm_config
    cfg.sound_config = None
    cfg.projector_hidden_size = 20480
    cfg.video_temporal_patch_size = 1
    cfg.separate_video_embedder = False
    cfg.vision_config = None
    del cfg.sound_config
    return cfg


@pytest.fixture
def mock_hf_config_with_conv3d(mock_llm_config, mock_sound_config, mock_vision_config):
    cfg = Mock()
    cfg.llm_config = mock_llm_config
    cfg.sound_config = mock_sound_config
    cfg.sound_context_token_id = 29
    cfg.projector_hidden_size = 20480
    cfg.video_temporal_patch_size = 2
    cfg.separate_video_embedder = True
    cfg.vision_config = mock_vision_config
    return cfg


@pytest.fixture
def mock_hf_pretrained_with_sound(mock_hf_config_with_sound):
    pretrained = Mock(spec=PreTrainedVLM)
    pretrained.config = mock_hf_config_with_sound
    return pretrained


@pytest.fixture
def mock_hf_pretrained_no_sound(mock_hf_config_no_sound):
    pretrained = Mock(spec=PreTrainedVLM)
    pretrained.config = mock_hf_config_no_sound
    return pretrained


@pytest.fixture
def mock_hf_pretrained_with_conv3d(mock_hf_config_with_conv3d):
    pretrained = Mock(spec=PreTrainedVLM)
    pretrained.config = mock_hf_config_with_conv3d
    return pretrained


@pytest.fixture
def omni_bridge():
    return NemotronOmniBridge()


class TestNemotronOmniBridgeInitialization:
    def test_bridge_inherits_vl(self, omni_bridge):
        assert isinstance(omni_bridge, NemotronVLBridge)
        assert isinstance(omni_bridge, NemotronOmniBridge)

    def test_bridge_has_required_methods(self, omni_bridge):
        assert hasattr(omni_bridge, "provider_bridge")
        assert callable(omni_bridge.provider_bridge)
        assert hasattr(omni_bridge, "mapping_registry")
        assert callable(omni_bridge.mapping_registry)


class TestNemotronOmniBridgeProviderBridge:
    def test_returns_moe_provider_with_sound(self, omni_bridge, mock_hf_pretrained_with_sound):
        provider = omni_bridge.provider_bridge(mock_hf_pretrained_with_sound)
        assert isinstance(provider, NemotronNano3Bv3OmniModelProvider)
        assert provider.has_sound is True
        assert provider.sound_hidden_size == 1024
        assert provider.sound_projection_hidden_size == 4096
        assert provider.sound_context_token_id == 29
        # Temporal compression defaults when not configured
        assert provider.video_temporal_patch_size == 1
        assert provider.separate_video_embedder is False

    def test_returns_dense_provider_without_moe(self, omni_bridge, mock_hf_pretrained_with_sound, mock_llm_config):
        mock_llm_config.n_routed_experts = None
        del mock_llm_config.n_routed_experts
        provider = omni_bridge.provider_bridge(mock_hf_pretrained_with_sound)
        assert isinstance(provider, NemotronNano12Bv2OmniModelProvider)

    def test_no_sound_config_sets_has_sound_false(self, omni_bridge, mock_hf_pretrained_no_sound):
        provider = omni_bridge.provider_bridge(mock_hf_pretrained_no_sound)
        assert provider.has_sound is False
        # Temporal defaults also apply
        assert provider.video_temporal_patch_size == 1
        assert provider.separate_video_embedder is False

    def test_reads_llm_config_correctly(self, omni_bridge, mock_hf_pretrained_with_sound):
        provider = omni_bridge.provider_bridge(mock_hf_pretrained_with_sound)
        assert provider.num_layers == 52
        assert provider.hidden_size == 2688
        assert provider.num_attention_heads == 28
        assert provider.vocab_size == 262144

    def test_reads_conv3d_config(self, omni_bridge, mock_hf_pretrained_with_conv3d):
        provider = omni_bridge.provider_bridge(mock_hf_pretrained_with_conv3d)
        assert provider.video_temporal_patch_size == 2
        assert provider.separate_video_embedder is True
        # Sound config still read correctly alongside conv3d
        assert provider.has_sound is True
        assert provider.sound_hidden_size == 1024


class TestNemotronOmniBridgeMappingRegistry:
    def test_returns_registry(self, omni_bridge):
        registry = omni_bridge.mapping_registry()
        assert isinstance(registry, MegatronMappingRegistry)
        assert len(registry.mappings) > 0

    def test_contains_vl_mappings(self, omni_bridge):
        registry = omni_bridge.mapping_registry()
        params = _collect_megatron_params(registry)

        assert any("vision_model" in p for p in params)
        assert any("language_model" in p for p in params)
        assert any("vision_projection" in p for p in params)
        # Conv3d video embedder mapping inherited from VL bridge
        assert "llava_model.vision_model.video_embedder.weight" in params

    def test_contains_sound_projection_mappings(self, omni_bridge):
        registry = omni_bridge.mapping_registry()
        params = _collect_megatron_params(registry)

        assert any("sound_projection" in p for p in params)
        sound_proj_params = [p for p in params if "sound_projection" in p]
        assert len(sound_proj_params) == 3  # norm.weight, linear_fc1.weight, linear_fc2.weight

    def test_sound_projection_uses_auto_mapping(self, omni_bridge):
        registry = omni_bridge.mapping_registry()
        sound_proj_mappings = [
            m for m in registry.mappings
            if hasattr(m, "megatron_param") and "sound_projection" in str(m.megatron_param)
        ]
        for m in sound_proj_mappings:
            assert isinstance(m, AutoMapping)

    def test_contains_sound_encoder_mappings(self, omni_bridge):
        registry = omni_bridge.mapping_registry()
        params = _collect_megatron_params(registry)

        assert any("sound_model" in p for p in params)
        sound_encoder_params = [p for p in params if "sound_model" in p]
        # 29 per-layer wildcard mappings + 10 subsampling conv + 2 subsampling linear = 41
        assert len(sound_encoder_params) > 30

    def test_sound_encoder_uses_replicated_mapping(self, omni_bridge):
        registry = omni_bridge.mapping_registry()
        sound_encoder_mappings = [
            m for m in registry.mappings
            if hasattr(m, "megatron_param") and "sound_model" in str(m.megatron_param)
        ]
        for m in sound_encoder_mappings:
            assert isinstance(m, ReplicatedMapping)

    def test_sound_encoder_uses_wildcards(self, omni_bridge):
        registry = omni_bridge.mapping_registry()
        layer_mappings = [
            m for m in registry.mappings
            if hasattr(m, "megatron_param")
            and "sound_model" in str(m.megatron_param)
            and "layers" in str(m.megatron_param)
        ]
        for m in layer_mappings:
            assert "layers.*." in str(m.megatron_param), (
                f"Expected wildcard in {m.megatron_param}"
            )

    def test_subsampling_conv_has_explicit_indices(self, omni_bridge):
        registry = omni_bridge.mapping_registry()
        sub_mappings = [
            m for m in registry.mappings
            if hasattr(m, "megatron_param")
            and "subsampling.conv." in str(m.megatron_param)
        ]
        assert len(sub_mappings) == 10  # 5 conv layers x 2 (weight + bias)
        meg_indices = sorted(set(
            str(m.megatron_param).split("subsampling.conv.")[1].split(".")[0]
            for m in sub_mappings
        ))
        assert meg_indices == ["0", "2", "3", "5", "6"]

    def test_superset_of_vl_mappings(self, omni_bridge):
        vl_bridge = NemotronVLBridge()
        vl_registry = vl_bridge.mapping_registry()
        omni_registry = omni_bridge.mapping_registry()
        assert len(omni_registry.mappings) > len(vl_registry.mappings)


class TestNemotronOmniBridgeConv3d:
    def test_separate_video_embedder_from_vision_config_args(self, omni_bridge, mock_llm_config, mock_sound_config, mock_vision_config):
        """When separate_video_embedder is not on the top-level config, fall back to vision_config.args."""
        cfg = Mock()
        cfg.llm_config = mock_llm_config
        cfg.sound_config = mock_sound_config
        cfg.sound_context_token_id = 29
        cfg.projector_hidden_size = 20480
        cfg.video_temporal_patch_size = 2
        cfg.vision_config = mock_vision_config
        del cfg.separate_video_embedder

        pretrained = Mock(spec=PreTrainedVLM)
        pretrained.config = cfg
        provider = omni_bridge.provider_bridge(pretrained)
        assert provider.separate_video_embedder is True

    def test_separate_video_embedder_falls_back_to_false(self, omni_bridge, mock_llm_config, mock_sound_config):
        """When neither top-level nor vision_config.args has separate_video_embedder, default to False."""
        cfg = Mock()
        cfg.llm_config = mock_llm_config
        cfg.sound_config = mock_sound_config
        cfg.sound_context_token_id = 29
        cfg.projector_hidden_size = 20480
        cfg.video_temporal_patch_size = 2
        cfg.vision_config = None
        del cfg.separate_video_embedder

        pretrained = Mock(spec=PreTrainedVLM)
        pretrained.config = cfg
        provider = omni_bridge.provider_bridge(pretrained)
        assert provider.separate_video_embedder is False

    def test_video_embedder_maps_to_correct_hf_key(self, omni_bridge):
        registry = omni_bridge.mapping_registry()
        video_embedder_mappings = [
            m for m in registry.mappings
            if hasattr(m, "megatron_param")
            and "video_embedder" in str(m.megatron_param)
        ]
        assert len(video_embedder_mappings) == 1
        m = video_embedder_mappings[0]
        assert str(m.hf_param) == "vision_model.radio_model.model.patch_generator.video_embedder.weight"

    def test_conv3d_config_coexists_with_sound(self, omni_bridge, mock_hf_pretrained_with_conv3d):
        """Conv3d and sound config should both be read from the same HF config."""
        provider = omni_bridge.provider_bridge(mock_hf_pretrained_with_conv3d)
        assert provider.video_temporal_patch_size == 2
        assert provider.separate_video_embedder is True
        assert provider.has_sound is True
        assert provider.sound_hidden_size == 1024


def _collect_megatron_params(registry):
    """Extract all megatron_param strings from a registry."""
    params = []
    for m in registry.mappings:
        if hasattr(m, "megatron_param"):
            params.append(str(m.megatron_param))
        hf = getattr(m, "hf_param", None)
        if isinstance(hf, dict):
            params.extend(str(v) for v in hf.values())
    return params
