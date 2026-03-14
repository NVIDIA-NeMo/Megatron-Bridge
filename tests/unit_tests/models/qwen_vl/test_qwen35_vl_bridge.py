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

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import WeightConversionTask
from megatron.bridge.models.hf_pretrained.vlm import PreTrainedVLM
from megatron.bridge.models.qwen_vl.qwen35_vl_bridge import Qwen35VLBridge, Qwen35VLMoEBridge
from megatron.bridge.models.qwen_vl.qwen35_vl_provider import (
    _TRANSFORMERS_HAS_QWEN3_5,
    _TRANSFORMERS_HAS_QWEN3_5_MOE,
    Qwen35VLModelProvider,
    Qwen35VLMoEModelProvider,
)


pytestmark = pytest.mark.skipif(not _TRANSFORMERS_HAS_QWEN3_5, reason="transformers does not have qwen3_5 support")


def _make_dense_text_config():
    """Create a mock text config matching Qwen3.5-27B dense architecture."""
    cfg = Mock(spec=[])
    cfg.num_hidden_layers = 64
    cfg.hidden_size = 5120
    cfg.intermediate_size = 17408
    cfg.num_attention_heads = 24
    cfg.num_key_value_heads = 4
    cfg.initializer_range = 0.02
    cfg.rms_norm_eps = 1e-6
    cfg.vocab_size = 248320
    cfg.max_position_embeddings = 262144
    cfg.rope_theta = 10000000.0
    cfg.tie_word_embeddings = False
    cfg.hidden_act = "silu"
    cfg.attention_bias = False
    cfg.head_dim = 256
    cfg.full_attention_interval = 4
    cfg.rope_parameters = {"partial_rotary_factor": 0.25, "rope_theta": 10000000.0}
    cfg.rope_scaling = {"mrope_section": [11, 11, 10]}
    cfg.linear_conv_kernel_dim = 4
    cfg.linear_key_head_dim = 128
    cfg.linear_value_head_dim = 128
    cfg.linear_num_key_heads = 16
    cfg.linear_num_value_heads = 48
    cfg.bos_token_id = 248045
    cfg.eos_token_id = 248044
    cfg.num_nextn_predict_layers = None
    cfg.torch_dtype = "bfloat16"
    return cfg


def _make_moe_text_config():
    """Create a mock text config matching Qwen3.5-397B-A17B MoE architecture."""
    cfg = Mock(spec=[])
    cfg.num_hidden_layers = 60
    cfg.hidden_size = 4096
    cfg.intermediate_size = 1024
    cfg.num_attention_heads = 32
    cfg.num_key_value_heads = 2
    cfg.initializer_range = 0.02
    cfg.rms_norm_eps = 1e-6
    cfg.vocab_size = 248320
    cfg.max_position_embeddings = 262144
    cfg.rope_theta = 10000000.0
    cfg.tie_word_embeddings = False
    cfg.hidden_act = "silu"
    cfg.attention_bias = False
    cfg.head_dim = 256
    cfg.full_attention_interval = 4
    cfg.rope_parameters = {"partial_rotary_factor": 0.25, "rope_theta": 10000000.0}
    cfg.rope_scaling = {"mrope_section": [11, 11, 10]}
    cfg.linear_conv_kernel_dim = 4
    cfg.linear_key_head_dim = 128
    cfg.linear_value_head_dim = 128
    cfg.linear_num_key_heads = 16
    cfg.linear_num_value_heads = 64
    cfg.moe_intermediate_size = 1024
    cfg.num_experts = 512
    cfg.num_experts_per_tok = 10
    cfg.shared_expert_intermediate_size = 4096
    cfg.bos_token_id = 248045
    cfg.eos_token_id = 248046
    cfg.num_nextn_predict_layers = None
    cfg.torch_dtype = "bfloat16"
    return cfg


def _make_vision_config():
    """Create a minimal mock vision config."""
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig

    return Qwen3_5VisionConfig()


def _make_mock_pretrained(text_config, vision_config, tie_word_embeddings=False):
    pretrained = Mock(spec=PreTrainedVLM)
    config = Mock()
    config.text_config = text_config
    config.vision_config = vision_config
    config.tie_word_embeddings = tie_word_embeddings
    config.vision_start_token_id = 248053
    config.vision_end_token_id = 248054
    config.image_token_id = 248056
    config.video_token_id = 248057
    config.audio_token_id = 248076
    pretrained.config = config
    return pretrained


# =====================================================================
# Tests for Qwen35VLBridge (Dense)
# =====================================================================


class TestQwen35VLBridgeInitialization:
    def test_bridge_initialization(self):
        bridge = Qwen35VLBridge()
        assert isinstance(bridge, Qwen35VLBridge)

    def test_bridge_has_required_methods(self):
        bridge = Qwen35VLBridge()
        assert hasattr(bridge, "provider_bridge") and callable(bridge.provider_bridge)
        assert hasattr(bridge, "mapping_registry") and callable(bridge.mapping_registry)


class TestQwen35VLBridgeProviderBridge:
    @pytest.fixture
    def bridge(self):
        return Qwen35VLBridge()

    @pytest.fixture
    def mock_pretrained(self):
        return _make_mock_pretrained(_make_dense_text_config(), _make_vision_config())

    def test_provider_bridge_returns_correct_type(self, bridge, mock_pretrained):
        provider = bridge.provider_bridge(mock_pretrained)
        assert isinstance(provider, Qwen35VLModelProvider)

    def test_provider_bridge_basic_config(self, bridge, mock_pretrained):
        provider = bridge.provider_bridge(mock_pretrained)
        assert provider.num_layers == 64
        assert provider.hidden_size == 5120
        assert provider.ffn_hidden_size == 17408
        assert provider.num_attention_heads == 24
        assert provider.num_query_groups == 4
        assert provider.vocab_size == 248320

    def test_provider_bridge_hybrid_architecture(self, bridge, mock_pretrained):
        provider = bridge.provider_bridge(mock_pretrained)
        assert provider.layernorm_zero_centered_gamma is True
        assert provider.attention_output_gate is True
        assert provider.experimental_attention_variant == "gated_delta_net"
        assert provider.linear_attention_freq == 4

    def test_provider_bridge_gdn_params(self, bridge, mock_pretrained):
        provider = bridge.provider_bridge(mock_pretrained)
        assert provider.linear_conv_kernel_dim == 4
        assert provider.linear_key_head_dim == 128
        assert provider.linear_value_head_dim == 128
        assert provider.linear_num_key_heads == 16
        assert provider.linear_num_value_heads == 48

    def test_provider_bridge_vl_config(self, bridge, mock_pretrained):
        provider = bridge.provider_bridge(mock_pretrained)
        assert provider.position_embedding_type == "mrope"
        assert provider.mrope_section == [11, 11, 10]
        assert provider.head_dim == 256
        assert provider.rotary_percent == 0.25

    def test_provider_bridge_token_ids(self, bridge, mock_pretrained):
        provider = bridge.provider_bridge(mock_pretrained)
        assert provider.bos_token_id == 248045
        assert provider.eos_token_id == 248044
        assert provider.vision_start_token_id == 248053
        assert provider.vision_end_token_id == 248054
        assert provider.image_token_id == 248056
        assert provider.video_token_id == 248057

    def test_provider_bridge_common_settings(self, bridge, mock_pretrained):
        provider = bridge.provider_bridge(mock_pretrained)
        assert provider.normalization == "RMSNorm"
        assert provider.gated_linear_unit is True
        assert provider.add_qkv_bias is False
        assert provider.add_bias_linear is False
        assert provider.qk_layernorm is True
        assert provider.hidden_dropout == 0.0

    @patch.object(Qwen35VLBridge, "dtype_from_hf")
    def test_provider_bridge_dtype_handling(self, mock_dtype, bridge, mock_pretrained):
        mock_dtype.return_value = torch.bfloat16
        provider = bridge.provider_bridge(mock_pretrained)
        assert provider.bf16 is True
        assert provider.params_dtype == torch.bfloat16

    def test_provider_bridge_tied_embeddings(self, bridge):
        text_config = _make_dense_text_config()
        text_config.tie_word_embeddings = True
        pretrained = _make_mock_pretrained(text_config, _make_vision_config(), tie_word_embeddings=True)
        provider = bridge.provider_bridge(pretrained)
        assert provider.share_embeddings_and_output_weights is True


class TestQwen35VLBridgeMappingRegistry:
    @pytest.fixture
    def bridge(self):
        return Qwen35VLBridge()

    def _get_mapping_names(self, registry):
        names = []
        for mapping in registry.mappings:
            if hasattr(mapping, "megatron_param"):
                names.append(str(getattr(mapping, "megatron_param")))
            hf = getattr(mapping, "hf_param", None)
            if isinstance(hf, dict):
                names.extend([str(v) for v in hf.values()])
            elif isinstance(hf, str):
                names.append(hf)
        return names

    def test_mapping_registry_type(self, bridge):
        registry = bridge.mapping_registry()
        assert isinstance(registry, MegatronMappingRegistry)
        assert len(registry.mappings) > 0

    def test_mapping_registry_has_embeddings(self, bridge):
        names = self._get_mapping_names(bridge.mapping_registry())
        assert any("embed_tokens" in n or "word_embeddings" in n for n in names)

    def test_mapping_registry_has_output_layer(self, bridge):
        names = self._get_mapping_names(bridge.mapping_registry())
        assert any("lm_head" in n or "output_layer" in n for n in names)

    def test_mapping_registry_has_gdn_mappings(self, bridge):
        names = self._get_mapping_names(bridge.mapping_registry())
        assert any("in_proj" in n for n in names), "Should contain GDN in_proj mappings"
        assert any("out_proj" in n for n in names), "Should contain GDN out_proj mappings"
        assert any("A_log" in n for n in names), "Should contain GDN A_log mappings"
        assert any("conv1d" in n for n in names), "Should contain GDN conv1d mappings"
        assert any("out_norm" in n or "linear_attn.norm" in n for n in names)

    def test_mapping_registry_has_dense_mlp(self, bridge):
        names = self._get_mapping_names(bridge.mapping_registry())
        assert any("gate_proj" in n for n in names), "Should contain gate_proj for dense MLP"
        assert any("up_proj" in n for n in names), "Should contain up_proj for dense MLP"
        assert any("down_proj" in n for n in names), "Should contain down_proj for dense MLP"

    def test_mapping_registry_has_no_moe(self, bridge):
        names = self._get_mapping_names(bridge.mapping_registry())
        assert not any("router" in n or "experts" in n for n in names), "Dense model should not have MoE mappings"

    def test_mapping_registry_has_vision_params(self, bridge):
        names = self._get_mapping_names(bridge.mapping_registry())
        assert any("visual" in n or "vision_model" in n for n in names)

    def test_mapping_registry_has_qkv(self, bridge):
        names = self._get_mapping_names(bridge.mapping_registry())
        assert any("linear_qkv" in n for n in names)

    def test_mapping_registry_has_vision_patch_embed(self, bridge):
        names = self._get_mapping_names(bridge.mapping_registry())
        assert any("patch_embed" in n for n in names)


# =====================================================================
# Tests for Qwen35VLMoEBridge
# =====================================================================


@pytest.mark.skipif(not _TRANSFORMERS_HAS_QWEN3_5_MOE, reason="transformers does not have qwen3_5_moe support")
class TestQwen35VLMoEBridgeInitialization:
    def test_bridge_initialization(self):
        bridge = Qwen35VLMoEBridge()
        assert isinstance(bridge, Qwen35VLMoEBridge)


@pytest.mark.skipif(not _TRANSFORMERS_HAS_QWEN3_5_MOE, reason="transformers does not have qwen3_5_moe support")
class TestQwen35VLMoEBridgeProviderBridge:
    @pytest.fixture
    def bridge(self):
        return Qwen35VLMoEBridge()

    @pytest.fixture
    def mock_pretrained(self):
        from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeVisionConfig

        return _make_mock_pretrained(_make_moe_text_config(), Qwen3_5MoeVisionConfig())

    def test_provider_bridge_returns_correct_type(self, bridge, mock_pretrained):
        provider = bridge.provider_bridge(mock_pretrained)
        assert isinstance(provider, Qwen35VLMoEModelProvider)

    def test_provider_bridge_basic_config(self, bridge, mock_pretrained):
        provider = bridge.provider_bridge(mock_pretrained)
        assert provider.num_layers == 60
        assert provider.hidden_size == 4096
        assert provider.num_attention_heads == 32
        assert provider.num_query_groups == 2
        assert provider.vocab_size == 248320

    def test_provider_bridge_moe_config(self, bridge, mock_pretrained):
        provider = bridge.provider_bridge(mock_pretrained)
        assert provider.num_moe_experts == 512
        assert provider.moe_router_topk == 10
        assert provider.moe_ffn_hidden_size == 1024
        assert provider.moe_shared_expert_gate is True
        assert provider.moe_grouped_gemm is True

    def test_provider_bridge_hybrid_architecture(self, bridge, mock_pretrained):
        provider = bridge.provider_bridge(mock_pretrained)
        assert provider.experimental_attention_variant == "gated_delta_net"
        assert provider.linear_attention_freq == 4
        assert provider.layernorm_zero_centered_gamma is True

    def test_provider_bridge_gdn_params(self, bridge, mock_pretrained):
        provider = bridge.provider_bridge(mock_pretrained)
        assert provider.linear_num_value_heads == 64
        assert provider.linear_key_head_dim == 128
        assert provider.linear_value_head_dim == 128

    def test_provider_bridge_token_ids(self, bridge, mock_pretrained):
        provider = bridge.provider_bridge(mock_pretrained)
        assert provider.bos_token_id == 248045
        assert provider.eos_token_id == 248046
        assert provider.image_token_id == 248056


@pytest.mark.skipif(not _TRANSFORMERS_HAS_QWEN3_5_MOE, reason="transformers does not have qwen3_5_moe support")
class TestQwen35VLMoEBridgeMappingRegistry:
    @pytest.fixture
    def bridge(self):
        return Qwen35VLMoEBridge()

    def _get_mapping_names(self, registry):
        names = []
        for mapping in registry.mappings:
            if hasattr(mapping, "megatron_param"):
                names.append(str(getattr(mapping, "megatron_param")))
            hf = getattr(mapping, "hf_param", None)
            if isinstance(hf, dict):
                names.extend([str(v) for v in hf.values()])
            elif isinstance(hf, str):
                names.append(hf)
        return names

    def test_mapping_registry_type(self, bridge):
        registry = bridge.mapping_registry()
        assert isinstance(registry, MegatronMappingRegistry)

    def test_mapping_registry_has_moe_mappings(self, bridge):
        names = self._get_mapping_names(bridge.mapping_registry())
        assert any("router" in n or "gate.weight" in n for n in names), "Should contain MoE router"
        assert any("experts" in n for n in names), "Should contain expert MLPs"
        assert any("shared_expert" in n for n in names), "Should contain shared experts"

    def test_mapping_registry_has_gdn_mappings(self, bridge):
        names = self._get_mapping_names(bridge.mapping_registry())
        assert any("in_proj" in n for n in names)
        assert any("A_log" in n for n in names)
        assert any("conv1d" in n for n in names)

    def test_mapping_registry_has_vision_params(self, bridge):
        names = self._get_mapping_names(bridge.mapping_registry())
        assert any("visual" in n or "vision_model" in n for n in names)


@pytest.mark.skipif(not _TRANSFORMERS_HAS_QWEN3_5_MOE, reason="transformers does not have qwen3_5_moe support")
class TestQwen35VLMoEBridgeExport:
    def test_maybe_modify_converted_hf_weight_stacks_fused_expert_keys(self, monkeypatch):
        bridge = Qwen35VLMoEBridge()
        bridge.hf_config = Mock()
        bridge.hf_config.text_config = Mock()
        bridge.hf_config.text_config.num_experts = 2

        monkeypatch.setattr(
            "megatron.bridge.models.qwen_vl.qwen3_vl_bridge.parallel_state.get_expert_model_parallel_world_size",
            lambda: 1,
        )

        task0 = Mock()
        task0.param_name = "language_model.decoder.layers.0.mlp.experts.linear_fc1.weight0"
        task1 = Mock()
        task1.param_name = "language_model.decoder.layers.0.mlp.experts.linear_fc1.weight1"

        key = "model.language_model.layers.0.mlp.experts.gate_up_proj"
        result0 = bridge.maybe_modify_converted_hf_weight(task0, {key: torch.ones(2, 2)}, {})
        result1 = bridge.maybe_modify_converted_hf_weight(task1, {key: 2 * torch.ones(2, 2)}, {})

        assert result0 == {}
        torch.testing.assert_close(
            result1[key],
            torch.stack([torch.ones(2, 2), 2 * torch.ones(2, 2)], dim=0),
        )
        assert bridge.hf_weights_cache == {}

    def test_maybe_modify_converted_hf_weight_keeps_explicit_mtp_expert_keys(self, monkeypatch):
        bridge = Qwen35VLMoEBridge()
        bridge.hf_config = Mock()
        bridge.hf_config.text_config = Mock()
        bridge.hf_config.text_config.num_experts = 256

        monkeypatch.setattr(
            "megatron.bridge.models.qwen_vl.qwen3_vl_bridge.parallel_state.get_expert_model_parallel_world_size",
            lambda: 8,
        )

        task = Mock()
        task.param_name = "language_model.mtp.layers.0.mtp_model_layer.mlp.experts.linear_fc1.weight0"
        converted = {
            "mtp.layers.0.mlp.experts.0.gate_proj.weight": torch.ones(2, 2),
            "mtp.layers.0.mlp.experts.32.gate_proj.weight": 2 * torch.ones(2, 2),
            "mtp.layers.0.mlp.experts.0.up_proj.weight": 3 * torch.ones(2, 2),
            "mtp.layers.0.mlp.experts.32.up_proj.weight": 4 * torch.ones(2, 2),
        }

        result = bridge.maybe_modify_converted_hf_weight(task, dict(converted), {})

        assert result.keys() == converted.keys()
        for key in converted:
            torch.testing.assert_close(result[key], converted[key])
        assert bridge.hf_weights_cache == {}

    def test_stream_weights_megatron_to_hf_skips_mtp_duplicate_embedding_export(self, monkeypatch):
        bridge = Qwen35VLMoEBridge()
        calls = []

        class DummyMapping:
            def megatron_to_hf(self, weight, module):
                calls.append((weight, module))
                return {}

        config = SimpleNamespace(share_embeddings_and_output_weights=False)
        stage0 = SimpleNamespace(pre_process=True, config=config)
        wrapped_stage1 = SimpleNamespace(
            module=SimpleNamespace(
                pre_process=False,
                config=config,
                language_model=SimpleNamespace(mtp_process=True),
            )
        )
        duplicate_module = Mock()
        task = WeightConversionTask(
            param_name="language_model.embedding.word_embeddings.weight",
            global_param_name="language_model.embedding.word_embeddings.weight",
            mapping=DummyMapping(),
            pp_rank=1,
            vp_stage=1,
            megatron_module=duplicate_module,
            param_weight=torch.ones(1),
        )

        monkeypatch.setattr(
            Qwen35VLMoEBridge,
            "_with_progress_tracking",
            lambda self, tasks, *_args, **_kwargs: tasks,
        )
        monkeypatch.setattr(
            Qwen35VLMoEBridge,
            "maybe_modify_converted_hf_weight",
            lambda self, *_args, **_kwargs: _args[1],
        )
        monkeypatch.setattr(
            "megatron.bridge.models.conversion.model_bridge.unwrap_model",
            lambda model, *_args, **_kwargs: [model[0]]
            if isinstance(model, list)
            else getattr(model, "module", model),
        )

        weights = list(
            bridge.stream_weights_megatron_to_hf(
                [stage0, wrapped_stage1],
                SimpleNamespace(),
                cpu=False,
                show_progress=False,
                conversion_tasks=[task],
                merge_adapter_weights=False,
            )
        )

        assert weights == []
        assert len(calls) == 1
        assert calls[0] == (None, None)

    def test_stream_weights_megatron_to_hf_keeps_pre_process_embedding_owner(self, monkeypatch):
        bridge = Qwen35VLMoEBridge()
        calls = []
        embedding_weight = torch.ones(1)
        embedding_module = Mock()

        class DummyMapping:
            def megatron_to_hf(self, weight, module):
                calls.append((weight, module))
                return {"hf.weight": weight}

        stage0 = SimpleNamespace(
            pre_process=True,
            config=SimpleNamespace(share_embeddings_and_output_weights=False),
            language_model=SimpleNamespace(mtp_process=False),
        )
        task = WeightConversionTask(
            param_name="language_model.embedding.word_embeddings.weight",
            global_param_name="language_model.embedding.word_embeddings.weight",
            mapping=DummyMapping(),
            pp_rank=0,
            vp_stage=0,
            megatron_module=embedding_module,
            param_weight=embedding_weight,
        )

        monkeypatch.setattr(
            Qwen35VLMoEBridge,
            "_with_progress_tracking",
            lambda self, tasks, *_args, **_kwargs: tasks,
        )
        monkeypatch.setattr(
            Qwen35VLMoEBridge,
            "maybe_modify_converted_hf_weight",
            lambda self, *_args, **_kwargs: _args[1],
        )
        monkeypatch.setattr(
            "megatron.bridge.models.conversion.model_bridge.unwrap_model",
            lambda model, *_args, **_kwargs: [model[0]]
            if isinstance(model, list)
            else getattr(model, "module", model),
        )

        weights = list(
            bridge.stream_weights_megatron_to_hf(
                [stage0],
                SimpleNamespace(),
                cpu=False,
                show_progress=False,
                conversion_tasks=[task],
                merge_adapter_weights=False,
            )
        )

        assert len(weights) == 1
        assert weights[0].param_name == "hf.weight"
        assert len(calls) == 1
        assert calls[0][0] is embedding_weight
        assert calls[0][1] is embedding_module
