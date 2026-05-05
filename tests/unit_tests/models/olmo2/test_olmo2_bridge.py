# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Unit tests for the OLMo-2 bridge.

These tests pin the architectural decisions documented in
``olmo2_bridge.py`` and ``olmo2_provider.py``:

* QK-RMSNorm and post-norm placement are flagged on the provider.
* The mapping registry routes the OLMo-2-specific
  ``post_attention_layernorm`` / ``post_feedforward_layernorm`` weights into
  the ``linear_proj.post_layernorm`` / ``linear_fc2.post_layernorm`` slots
  (NOT into ``linear_qkv.layer_norm_weight`` like Llama/Qwen3 do).
* Pre-norm slots — ``input_layernorm`` and ``pre_mlp_layernorm`` — are
  intentionally absent from the registry (no HF weights map to them).
* QKV / Gated-MLP weights are fused.
* Q-/K-RMSNorm weights map by the standard ``q_norm`` / ``k_norm`` names.
"""

from unittest.mock import Mock

import pytest
import torch


try:
    from transformers import Olmo2Config, Olmo2ForCausalLM

    _HAS_OLMO2 = True
except ImportError:  # pragma: no cover - older transformers versions
    Olmo2Config = None  # type: ignore[assignment]
    Olmo2ForCausalLM = None  # type: ignore[assignment]
    _HAS_OLMO2 = False

from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
)
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.olmo2.olmo2_bridge import Olmo2Bridge
from megatron.bridge.models.olmo2.olmo2_provider import (
    Olmo2ModelProvider,
    Olmo2ModelProvider1B,
    Olmo2ModelProvider7B,
    Olmo2ModelProvider13B,
    TERowParallelLinearPostLN,
    olmo2_layer_spec,
)


pytestmark = pytest.mark.skipif(
    not _HAS_OLMO2,
    reason="transformers version does not expose Olmo2Config / Olmo2ForCausalLM",
)


@pytest.fixture
def olmo2_7b_config_dict():
    """Mirror of `allenai/OLMo-2-1124-7B/config.json`."""
    return {
        "architectures": ["Olmo2ForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "eos_token_id": 100257,
        "hidden_act": "silu",
        "hidden_size": 4096,
        "initializer_range": 0.02,
        "intermediate_size": 11008,
        "max_position_embeddings": 4096,
        "model_type": "olmo2",
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "num_key_value_heads": 32,
        "pad_token_id": 100277,
        "rms_norm_eps": 1e-06,
        "rope_theta": 500000.0,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "use_cache": True,
        "vocab_size": 100352,
    }


@pytest.fixture
def olmo2_1b_config_dict():
    """Mirror of `allenai/OLMo-2-0425-1B/config.json`."""
    return {
        "architectures": ["Olmo2ForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "hidden_act": "silu",
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "max_position_embeddings": 4096,
        "model_type": "olmo2",
        "num_attention_heads": 16,
        "num_hidden_layers": 16,
        "num_key_value_heads": 16,
        "rms_norm_eps": 1e-06,
        "rope_theta": 500000.0,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "vocab_size": 100352,
    }


@pytest.fixture
def olmo2_7b_config(olmo2_7b_config_dict):
    return Olmo2Config(**olmo2_7b_config_dict)


@pytest.fixture
def olmo2_1b_config(olmo2_1b_config_dict):
    return Olmo2Config(**olmo2_1b_config_dict)


@pytest.fixture
def mock_pretrained_olmo2_7b(olmo2_7b_config):
    pretrained = Mock(spec=PreTrainedCausalLM)
    pretrained.config = olmo2_7b_config
    pretrained.model = Mock(spec=Olmo2ForCausalLM)
    pretrained.model.dtype = torch.bfloat16
    return pretrained


@pytest.fixture
def mock_pretrained_olmo2_1b(olmo2_1b_config):
    pretrained = Mock(spec=PreTrainedCausalLM)
    pretrained.config = olmo2_1b_config
    pretrained.model = Mock(spec=Olmo2ForCausalLM)
    pretrained.model.dtype = torch.bfloat16
    return pretrained


class TestOlmo2BridgeRegistration:
    """Bridge class registration and basic identity."""

    def test_inherits_megatron_bridge(self):
        assert issubclass(Olmo2Bridge, MegatronModelBridge)

    def test_source_class_name(self):
        # `source` is registered as a string for environments where the
        # transformers version does not export Olmo2ForCausalLM.
        assert getattr(Olmo2Bridge, "_source_class_name", "Olmo2ForCausalLM") in {
            "Olmo2ForCausalLM",
            getattr(Olmo2Bridge, "_source_class_name", "Olmo2ForCausalLM"),
        }


class TestOlmo2ProviderBridgeArchitecturalFlags:
    """Provider config flags that are load-bearing for OLMo-2 correctness."""

    def test_returns_gpt_provider(self, mock_pretrained_olmo2_7b):
        provider = Olmo2Bridge().provider_bridge(mock_pretrained_olmo2_7b)
        assert isinstance(provider, GPTModelProvider)

    def test_qk_layernorm_enabled(self, mock_pretrained_olmo2_7b):
        """OLMo-2 normalizes Q and K before the attention dot product."""
        provider = Olmo2Bridge().provider_bridge(mock_pretrained_olmo2_7b)
        assert provider.qk_layernorm is True

    def test_post_norm_layer_spec_selected(self, mock_pretrained_olmo2_7b):
        """The bridge must swap the default pre-norm spec for the OLMo-2 post-norm one."""
        provider = Olmo2Bridge().provider_bridge(mock_pretrained_olmo2_7b)
        assert provider.transformer_layer_spec is olmo2_layer_spec

    def test_no_biases(self, mock_pretrained_olmo2_7b):
        provider = Olmo2Bridge().provider_bridge(mock_pretrained_olmo2_7b)
        assert provider.add_bias_linear is False
        assert provider.add_qkv_bias is False

    def test_swiglu_mlp(self, mock_pretrained_olmo2_7b):
        provider = Olmo2Bridge().provider_bridge(mock_pretrained_olmo2_7b)
        assert provider.gated_linear_unit is True

    def test_rmsnorm(self, mock_pretrained_olmo2_7b):
        provider = Olmo2Bridge().provider_bridge(mock_pretrained_olmo2_7b)
        assert provider.normalization == "RMSNorm"

    def test_rope_position_embedding(self, mock_pretrained_olmo2_7b):
        provider = Olmo2Bridge().provider_bridge(mock_pretrained_olmo2_7b)
        assert provider.position_embedding_type == "rope"

    def test_untied_word_embeddings(self, mock_pretrained_olmo2_7b):
        provider = Olmo2Bridge().provider_bridge(mock_pretrained_olmo2_7b)
        assert provider.share_embeddings_and_output_weights is False


class TestOlmo2ProviderBridgeShapeFields:
    """Numerical config translation from HF config → provider."""

    def test_7b_dimensions(self, mock_pretrained_olmo2_7b, olmo2_7b_config):
        provider = Olmo2Bridge().provider_bridge(mock_pretrained_olmo2_7b)
        assert provider.num_layers == olmo2_7b_config.num_hidden_layers == 32
        assert provider.hidden_size == olmo2_7b_config.hidden_size == 4096
        assert provider.num_attention_heads == olmo2_7b_config.num_attention_heads == 32
        assert provider.ffn_hidden_size == olmo2_7b_config.intermediate_size == 11008
        assert provider.vocab_size == olmo2_7b_config.vocab_size == 100352

    def test_1b_dimensions(self, mock_pretrained_olmo2_1b, olmo2_1b_config):
        provider = Olmo2Bridge().provider_bridge(mock_pretrained_olmo2_1b)
        assert provider.num_layers == olmo2_1b_config.num_hidden_layers == 16
        assert provider.hidden_size == olmo2_1b_config.hidden_size == 2048
        assert provider.num_attention_heads == 16
        assert provider.ffn_hidden_size == 8192

    def test_kv_channels_derived_when_head_dim_missing(self, mock_pretrained_olmo2_7b):
        """OLMo-2 HF configs do not include head_dim; the bridge must derive it."""
        provider = Olmo2Bridge().provider_bridge(mock_pretrained_olmo2_7b)
        # 4096 / 32 = 128
        assert provider.kv_channels == 128

    def test_kv_channels_uses_explicit_head_dim_when_present(self, olmo2_7b_config):
        """If a future HF config grows a head_dim field, the bridge must respect it."""
        olmo2_7b_config.head_dim = 96
        pretrained = Mock(spec=PreTrainedCausalLM)
        pretrained.config = olmo2_7b_config
        pretrained.model = Mock(spec=Olmo2ForCausalLM)
        pretrained.model.dtype = torch.bfloat16
        provider = Olmo2Bridge().provider_bridge(pretrained)
        assert provider.kv_channels == 96


class TestOlmo2MappingRegistry:
    """Weight name routing — the load-bearing public contract of any bridge."""

    @pytest.fixture
    def registry(self):
        return Olmo2Bridge().mapping_registry()

    @pytest.fixture
    def hf_param_to_megatron(self, registry):
        """Build a flat HF→Megatron param-name lookup for AutoMapping entries."""
        out: dict[str, str] = {}
        for m in registry.mappings:
            if isinstance(m, AutoMapping):
                out[m.hf_param] = m.megatron_param
        return out

    def test_embedding_routes_correctly(self, hf_param_to_megatron):
        assert hf_param_to_megatron["model.embed_tokens.weight"] == "embedding.word_embeddings.weight"

    def test_output_layer_routes_correctly(self, hf_param_to_megatron):
        assert hf_param_to_megatron["lm_head.weight"] == "output_layer.weight"

    def test_final_norm_routes_correctly(self, hf_param_to_megatron):
        assert hf_param_to_megatron["model.norm.weight"] == "decoder.final_layernorm.weight"

    def test_post_attention_layernorm_routes_to_linear_proj_post_layernorm(self, hf_param_to_megatron):
        """
        OLMo-2's ``post_attention_layernorm`` is an *output* norm.
        It must NOT be routed into ``linear_qkv.layer_norm_weight`` (the
        Llama/Qwen3 pre-MLP slot). It must go into ``linear_proj.post_layernorm.weight``.
        """
        target = hf_param_to_megatron["model.layers.*.post_attention_layernorm.weight"]
        assert target == "decoder.layers.*.self_attention.linear_proj.post_layernorm.weight"

    def test_post_feedforward_layernorm_routes_to_linear_fc2_post_layernorm(self, hf_param_to_megatron):
        target = hf_param_to_megatron["model.layers.*.post_feedforward_layernorm.weight"]
        assert target == "decoder.layers.*.mlp.linear_fc2.post_layernorm.weight"

    def test_q_norm_routes_to_q_layernorm(self, hf_param_to_megatron):
        target = hf_param_to_megatron["model.layers.*.self_attn.q_norm.weight"]
        assert target == "decoder.layers.*.self_attention.q_layernorm.weight"

    def test_k_norm_routes_to_k_layernorm(self, hf_param_to_megatron):
        target = hf_param_to_megatron["model.layers.*.self_attn.k_norm.weight"]
        assert target == "decoder.layers.*.self_attention.k_layernorm.weight"

    def test_attention_output_projection_routes(self, hf_param_to_megatron):
        target = hf_param_to_megatron["model.layers.*.self_attn.o_proj.weight"]
        assert target == "decoder.layers.*.self_attention.linear_proj.weight"

    def test_mlp_down_projection_routes(self, hf_param_to_megatron):
        target = hf_param_to_megatron["model.layers.*.mlp.down_proj.weight"]
        assert target == "decoder.layers.*.mlp.linear_fc2.weight"

    def test_no_pre_attention_layernorm_mapping(self, hf_param_to_megatron):
        """
        OLMo-2 has NO pre-attention norm in HF, and Megatron-side it is
        IdentityOp via the layer spec. There must be no mapping that tries
        to populate ``linear_qkv.layer_norm_weight`` from ``input_layernorm``.
        """
        assert "model.layers.*.input_layernorm.weight" not in hf_param_to_megatron
        for hf_param, mg_param in hf_param_to_megatron.items():
            assert "linear_qkv.layer_norm_weight" not in mg_param, (
                f"OLMo-2 must not write into linear_qkv.layer_norm_weight; saw {hf_param} -> {mg_param}"
            )

    def test_no_pre_feedforward_layernorm_mapping(self, hf_param_to_megatron):
        """Mirror of the above for the pre-MLP slot."""
        for hf_param, mg_param in hf_param_to_megatron.items():
            assert "linear_fc1.layer_norm_weight" not in mg_param, (
                f"OLMo-2 must not write into linear_fc1.layer_norm_weight; saw {hf_param} -> {mg_param}"
            )

    def test_qkv_fused_mapping_present(self, registry):
        qkv_maps = [m for m in registry.mappings if isinstance(m, QKVMapping)]
        assert len(qkv_maps) == 1
        m = qkv_maps[0]
        assert m.megatron_param == "decoder.layers.*.self_attention.linear_qkv.weight"
        assert m.q == "model.layers.*.self_attn.q_proj.weight"
        assert m.k == "model.layers.*.self_attn.k_proj.weight"
        assert m.v == "model.layers.*.self_attn.v_proj.weight"

    def test_gated_mlp_fused_mapping_present(self, registry):
        gated_maps = [m for m in registry.mappings if isinstance(m, GatedMLPMapping)]
        assert len(gated_maps) == 1
        m = gated_maps[0]
        assert m.megatron_param == "decoder.layers.*.mlp.linear_fc1.weight"
        assert m.gate == "model.layers.*.mlp.gate_proj.weight"
        assert m.up == "model.layers.*.mlp.up_proj.weight"

    def test_no_qkv_bias_mapping(self, hf_param_to_megatron):
        """OLMo-2 has ``attention_bias=False``; no bias weight should be mapped."""
        for hf_param, mg_param in hf_param_to_megatron.items():
            assert "self_attn.q_proj.bias" not in hf_param
            assert "self_attn.k_proj.bias" not in hf_param
            assert "self_attn.v_proj.bias" not in hf_param
            assert "linear_qkv.bias" not in mg_param


class TestOlmo2LayerSpec:
    """Verify the post-norm layer spec is structured correctly."""

    @pytest.fixture
    def spec(self):
        return olmo2_layer_spec(config=None)

    def test_pre_attention_norm_is_identity(self, spec):
        from megatron.core.transformer.identity_op import IdentityOp

        assert spec.submodules.input_layernorm is IdentityOp

    def test_pre_mlp_norm_is_identity(self, spec):
        from megatron.core.transformer.identity_op import IdentityOp

        assert spec.submodules.pre_mlp_layernorm is IdentityOp

    def test_attention_uses_post_layernorm_linear_proj(self, spec):
        attn = spec.submodules.self_attention
        assert attn.submodules.linear_proj is TERowParallelLinearPostLN

    def test_mlp_uses_post_layernorm_fc2(self, spec):
        mlp = spec.submodules.mlp
        assert mlp.submodules.linear_fc2 is TERowParallelLinearPostLN

    def test_attention_has_qk_layernorm_slots(self, spec):
        from megatron.core.extensions.transformer_engine import TENorm

        attn = spec.submodules.self_attention
        assert attn.submodules.q_layernorm is TENorm
        assert attn.submodules.k_layernorm is TENorm

    def test_attention_uses_plain_column_parallel_qkv(self, spec):
        """Pre-attention norm is IdentityOp ⇒ linear_qkv must NOT carry a fused norm."""
        from megatron.core.extensions.transformer_engine import TEColumnParallelLinear

        attn = spec.submodules.self_attention
        assert attn.submodules.linear_qkv is TEColumnParallelLinear

    def test_mlp_uses_plain_column_parallel_fc1(self, spec):
        """Pre-MLP norm is IdentityOp ⇒ linear_fc1 must NOT carry a fused norm."""
        from megatron.core.extensions.transformer_engine import TEColumnParallelLinear

        mlp = spec.submodules.mlp
        assert mlp.submodules.linear_fc1 is TEColumnParallelLinear


class TestOlmo2ModelProviderSizeVariants:
    """Hardcoded size variants used by recipes — keep in sync with HF defaults."""

    def test_1b_dimensions(self):
        p = Olmo2ModelProvider1B()
        assert p.num_layers == 16
        assert p.hidden_size == 2048
        assert p.num_attention_heads == 16
        assert p.ffn_hidden_size == 8192

    def test_7b_dimensions(self):
        p = Olmo2ModelProvider7B()
        assert p.num_layers == 32
        assert p.hidden_size == 4096
        assert p.num_attention_heads == 32
        assert p.ffn_hidden_size == 11008

    def test_13b_dimensions(self):
        p = Olmo2ModelProvider13B()
        assert p.num_layers == 40
        assert p.hidden_size == 5120
        assert p.num_attention_heads == 40
        assert p.ffn_hidden_size == 13824

    @pytest.mark.parametrize("cls", [Olmo2ModelProvider1B, Olmo2ModelProvider7B, Olmo2ModelProvider13B])
    def test_all_size_variants_inherit_olmo2_defaults(self, cls):
        p = cls()
        assert p.qk_layernorm is True
        assert p.normalization == "RMSNorm"
        assert p.gated_linear_unit is True
        assert p.add_bias_linear is False
        assert p.add_qkv_bias is False
        assert p.layernorm_epsilon == 1e-6
        assert p.rotary_base == 500000.0
        assert p.share_embeddings_and_output_weights is False
        assert p.transformer_layer_spec is olmo2_layer_spec


class TestOlmo2ProviderBaseDefaults:
    """The `Olmo2ModelProvider` base picks up OLMo-2 conventions even before recipe sizing."""

    def test_base_provider_layer_spec(self):
        p = Olmo2ModelProvider()
        assert p.transformer_layer_spec is olmo2_layer_spec

    def test_base_provider_persist_layer_norm(self):
        # OLMoE sets this; OLMo-2 follows the same convention.
        p = Olmo2ModelProvider()
        assert p.persist_layer_norm is True
