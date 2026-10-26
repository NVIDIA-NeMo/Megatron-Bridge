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

"""Unit tests for the builder-backed Gemma 4 text configs and builders."""

import dataclasses
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from megatron.core.activations import fast_gelu
from torch import nn
from transformers import Gemma4TextConfig

from megatron.bridge.models.conversion.model_bridge import ModelConfigNotSupportedError
from megatron.bridge.models.diffusion_gemma.diffusion_gemma_bridge import DiffusionGemmaBridge
from megatron.bridge.models.gemma.gemma4_bridge import Gemma4Bridge
from megatron.bridge.models.gemma.gemma4_builder import (
    Gemma4DenseModelBuilder,
    Gemma4ModelBuilder,
    gemma4_moe_layer_spec,
)
from megatron.bridge.models.gemma.gemma4_config import (
    Gemma4DenseModelConfig,
    Gemma4DenseTransformerConfig,
    Gemma4ModelConfig,
    Gemma4TransformerConfig,
)
from megatron.bridge.models.gemma.gemma4_provider import Gemma4DenseProvider, Gemma4ModelProvider
from megatron.bridge.models.gemma.modeling_gemma4 import Gemma4OutputLayer
from megatron.bridge.models.gemma_vl.gemma4_vl_bridge import Gemma4VLBridge
from megatron.bridge.models.gpt.gpt_builder import GPTModelBuilder


pytestmark = pytest.mark.unit

_BUILDER = "megatron.bridge.models.gemma.gemma4_builder"

_TEXT_CONFIG = {
    "attention_bias": False,
    "attention_dropout": 0.0,
    "attention_k_eq_v": True,
    "head_dim": 64,
    "global_head_dim": 128,
    "hidden_activation": "gelu_pytorch_tanh",
    "hidden_size": 256,
    "intermediate_size": 512,
    "max_position_embeddings": 2048,
    "num_attention_heads": 4,
    "num_hidden_layers": 6,
    "num_key_value_heads": 2,
    "num_global_key_value_heads": 1,
    "rms_norm_eps": 1e-6,
    "sliding_window": 128,
    "vocab_size": 1000,
    "final_logit_softcapping": 30.0,
    "layer_types": ["sliding_attention"] * 5 + ["full_attention"],
    "rope_parameters": {
        "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1_000_000.0, "rope_type": "proportional"},
        "sliding_attention": {"rope_theta": 10_000.0, "rope_type": "default"},
    },
}


def _dense_hf_config() -> Gemma4TextConfig:
    return Gemma4TextConfig(
        **_TEXT_CONFIG,
        enable_moe_block=False,
        num_kv_shared_layers=2,
        use_double_wide_mlp=True,
        hidden_size_per_layer_input=32,
        vocab_size_per_layer_input=1024,
    )


def _moe_hf_config() -> Gemma4TextConfig:
    return Gemma4TextConfig(
        **_TEXT_CONFIG,
        enable_moe_block=True,
        num_experts=8,
        top_k_experts=2,
        moe_intermediate_size=64,
        hidden_size_per_layer_input=0,
    )


def _provider_and_model_config(hf_config):
    provider = Gemma4Bridge().provider_bridge(SimpleNamespace(config=hf_config))
    model_config = Gemma4Bridge().hf_config_to_model_config(hf_config)
    return provider, model_config


class TestGemma4ModelConfigFromHF:
    def test_dense_config_types(self):
        model_config = Gemma4Bridge().hf_config_to_model_config(_dense_hf_config())

        assert type(model_config) is Gemma4DenseModelConfig
        assert type(model_config.transformer) is Gemma4DenseTransformerConfig
        assert model_config.get_builder_cls() is Gemma4DenseModelBuilder

    def test_moe_config_types(self):
        model_config = Gemma4Bridge().hf_config_to_model_config(_moe_hf_config())

        assert type(model_config) is Gemma4ModelConfig
        assert type(model_config.transformer) is Gemma4TransformerConfig
        assert model_config.get_builder_cls() is Gemma4ModelBuilder
        assert model_config.transformer_layer_spec is gemma4_moe_layer_spec

    def test_dense_architecture_fields(self):
        model_config = Gemma4Bridge().hf_config_to_model_config(_dense_hf_config())

        assert model_config.position_embedding_type == "rope"
        assert model_config.share_embeddings_and_output_weights is True
        assert model_config.make_vocab_size_divisible_by == 128
        transformer = model_config.transformer
        assert transformer.window_size == (127, 0)
        assert transformer.window_attn_skip_freq == [True] * 5 + [False]
        assert transformer.global_kv_channels == 128
        assert transformer.num_global_query_groups == 1
        assert transformer.num_kv_shared_layers == 2
        assert transformer.use_double_wide_mlp is True
        assert transformer.per_layer_embed_dim == 32
        assert transformer.per_layer_embed_vocab_size == 1024
        assert transformer.scale_embeddings_by_hidden_size is True
        assert transformer.activation_func is fast_gelu

    def test_moe_architecture_fields(self):
        model_config = Gemma4Bridge().hf_config_to_model_config(_moe_hf_config())

        assert model_config.rotary_base == (10_000.0, 1_000_000.0)
        transformer = model_config.transformer
        assert transformer.interleaved_attn_pattern == (5, 1)
        assert transformer.global_head_dim == 128
        assert transformer.num_global_key_value_heads == 1
        assert transformer.num_moe_experts == 8
        assert transformer.moe_router_topk == 2
        assert transformer.moe_ffn_hidden_size == 64
        assert transformer.softmax_scale == 1.0
        assert transformer.qk_layernorm is True

    @pytest.mark.parametrize("make_hf_config", [_dense_hf_config, _moe_hf_config], ids=["dense", "moe"])
    def test_builder_config_matches_legacy_provider(self, make_hf_config):
        """Every field the provider and the builder config share has the same value."""
        provider, model_config = _provider_and_model_config(make_hf_config())

        mismatches = {}
        for owner in (model_config, model_config.transformer):
            for field in dataclasses.fields(owner):
                # The layer spec is chosen by the builder, not read from the provider.
                if field.name in {"transformer", "transformer_layer_spec"} or not hasattr(provider, field.name):
                    continue
                expected = getattr(provider, field.name)
                actual = getattr(owner, field.name)
                if actual != expected:
                    mismatches[field.name] = (expected, actual)

        assert mismatches == {}

    @pytest.mark.parametrize("make_hf_config", [_dense_hf_config, _moe_hf_config], ids=["dense", "moe"])
    def test_export_config_matches_legacy_provider(self, make_hf_config):
        provider, model_config = _provider_and_model_config(make_hf_config())

        assert Gemma4Bridge.megatron_to_hf_config(model_config) == Gemma4Bridge.megatron_to_hf_config(provider)

    @pytest.mark.parametrize("make_hf_config", [_dense_hf_config, _moe_hf_config], ids=["dense", "moe"])
    def test_serialized_config_restores_class_and_fields(self, make_hf_config):
        model_config = Gemma4Bridge().hf_config_to_model_config(make_hf_config())

        restored = type(model_config).from_dict(model_config.as_dict())

        assert type(restored) is type(model_config)
        assert type(restored.transformer) is type(model_config.transformer)
        assert restored.transformer.activation_func is fast_gelu
        assert restored.transformer_layer_spec is model_config.transformer_layer_spec
        assert restored.as_dict() == model_config.as_dict()

    @pytest.mark.parametrize("make_hf_config", [_dense_hf_config, _moe_hf_config], ids=["dense", "moe"])
    def test_checkpoint_run_config_reload(self, make_hf_config, tmp_path):
        """A config saved to run_config.yaml reloads as the same Gemma 4 config and exports identically."""
        from megatron.bridge.recipes.gpt import vanilla_gpt_pretrain_config
        from megatron.bridge.training.model_load_save import load_model_config

        model_config = Gemma4Bridge().hf_config_to_model_config(make_hf_config())
        run_config = vanilla_gpt_pretrain_config()
        run_config.model = model_config
        run_config.to_yaml(str(tmp_path / "run_config.yaml"))

        loaded, _ = load_model_config(str(tmp_path))

        assert type(loaded) is type(model_config)
        assert type(loaded.transformer) is type(model_config.transformer)
        assert loaded.transformer_layer_spec is model_config.transformer_layer_spec
        assert Gemma4Bridge.megatron_to_hf_config(loaded) == Gemma4Bridge.megatron_to_hf_config(model_config)

    def test_legacy_providers_still_built(self):
        bridge = Gemma4Bridge()

        assert isinstance(bridge.provider_bridge(SimpleNamespace(config=_dense_hf_config())), Gemma4DenseProvider)
        assert isinstance(bridge.provider_bridge(SimpleNamespace(config=_moe_hf_config())), Gemma4ModelProvider)


class TestGemma4ModelConfigRouting:
    def test_text_bridge_converts_through_builder(self):
        assert Gemma4Bridge.MODEL_CONFIG_CLASS is Gemma4ModelConfig
        assert Gemma4Bridge.USE_MODEL_CONFIG_FOR_CONVERSION is True

    def test_vl_bridge_text_mode_returns_text_builder_config(self):
        bridge = Gemma4VLBridge()
        bridge.gemma4_conversion_mode = "text"

        model_config = bridge.hf_config_to_model_config(SimpleNamespace(text_config=_dense_hf_config()))

        assert type(model_config) is Gemma4DenseModelConfig
        assert Gemma4VLBridge.USE_MODEL_CONFIG_FOR_CONVERSION is False

    @pytest.mark.parametrize("mode", ["auto", "vl", "audio"])
    def test_vl_bridge_rejects_builder_config_outside_text_mode(self, mode):
        bridge = Gemma4VLBridge()
        bridge.gemma4_conversion_mode = mode

        with pytest.raises(ModelConfigNotSupportedError, match="GEMMA4_CONVERSION_MODE=text"):
            bridge.hf_config_to_model_config(SimpleNamespace(text_config=_dense_hf_config()))

    def test_diffusion_gemma_stays_on_providers(self):
        bridge = DiffusionGemmaBridge()
        bridge.gemma4_conversion_mode = "text"

        assert DiffusionGemmaBridge.MODEL_CONFIG_CLASS is None
        assert DiffusionGemmaBridge.USE_MODEL_CONFIG_FOR_CONVERSION is False
        with pytest.raises(ModelConfigNotSupportedError, match="DiffusionGemmaBridge"):
            bridge.hf_config_to_model_config(SimpleNamespace(text_config=_dense_hf_config()))


class _IdentityOutputLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, hidden_states, *args, **kwargs):
        return hidden_states, None


class _FakeGPTModel(nn.Module):
    def __init__(self, config, vocab_size=1024):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.embedding = nn.Identity()
        self.output_layer = _IdentityOutputLayer(config)
        self.setup_embeddings_and_output_layer = MagicMock()


class TestGemma4DenseModelBuilder:
    @pytest.fixture
    def model_config(self):
        return Gemma4Bridge().hf_config_to_model_config(_dense_hf_config())

    def _build(self, model_config, **kwargs):
        seen = {}

        def fake_gpt_model(**gpt_kwargs):
            config = gpt_kwargs["config"]
            seen["kwargs"] = gpt_kwargs
            seen["dual_rope_during_build"] = (
                config.sliding_window_rope_base,
                config.full_attention_rope_base,
                config.full_attention_rope_partial_factor,
            )
            return _FakeGPTModel(config)

        with (
            patch(f"{_BUILDER}.GPTModel", side_effect=fake_gpt_model),
            patch(f"{_BUILDER}.get_gemma4_layer_spec", return_value="dense-spec") as layer_spec,
            patch(f"{_BUILDER}.Gemma4DenseRotaryEmbedding", return_value="dual-rope") as rope,
            patch(f"{_BUILDER}._attach_ple_modules") as attach_ple,
            patch(f"{_BUILDER}.wire_gemma4_kv_sharing") as wire_kv,
            patch(f"{_BUILDER}._install_ple_forward") as ple_forward,
            patch(f"{_BUILDER}.install_gemma4_dense_load_state_aliases") as aliases,
        ):
            model = Gemma4DenseModelBuilder(model_config).build_model(pg_collection="pgs", **kwargs)
        mocks = SimpleNamespace(
            layer_spec=layer_spec,
            rope=rope,
            attach_ple=attach_ple,
            wire_kv=wire_kv,
            ple_forward=ple_forward,
            aliases=aliases,
        )
        return model, seen, mocks

    def test_builds_gpt_model_like_the_legacy_provider(self, model_config):
        model, seen, mocks = self._build(model_config)
        transformer = model_config.transformer

        gpt_kwargs = seen["kwargs"]
        assert gpt_kwargs["config"] is transformer
        assert gpt_kwargs["transformer_layer_spec"] == "dense-spec"
        # Padding uses make_vocab_size_divisible_by only, like Gemma4DenseProvider.build().
        assert gpt_kwargs["vocab_size"] == 1024
        assert gpt_kwargs["max_sequence_length"] == model_config.seq_length
        assert gpt_kwargs["share_embeddings_and_output_weights"] is True
        assert gpt_kwargs["pre_process"] is True and gpt_kwargs["post_process"] is True
        assert gpt_kwargs["pg_collection"] == "pgs"
        mocks.layer_spec.assert_called_once_with(transformer)
        mocks.rope.assert_called_once_with(transformer)
        assert model.rotary_pos_emb == "dual-rope"
        mocks.attach_ple.assert_called_once_with(model, transformer, transformer)
        mocks.wire_kv.assert_called_once_with(model)
        mocks.ple_forward.assert_called_once_with(model)
        mocks.aliases.assert_called_once_with(model)
        assert isinstance(model.output_layer, Gemma4OutputLayer)

    def test_hides_dual_rope_bases_only_while_building(self, model_config):
        _, seen, _ = self._build(model_config)
        transformer = model_config.transformer

        assert seen["dual_rope_during_build"] == (None, None, None)
        assert transformer.sliding_window_rope_base == 10_000.0
        assert transformer.full_attention_rope_base == 1_000_000.0
        assert transformer.full_attention_rope_partial_factor == 0.25

    def test_skips_per_layer_embeddings_without_pre_process(self, model_config):
        _, _, mocks = self._build(model_config, pre_process=False)

        mocks.attach_ple.assert_not_called()

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("pipeline_model_parallel_size", 2, "PP=1"),
            ("context_parallel_size", 2, "CP=1"),
        ],
    )
    def test_rejects_unsupported_parallelism(self, model_config, field, value, match):
        setattr(model_config.transformer, field, value)

        with pytest.raises(NotImplementedError, match=match):
            Gemma4DenseModelBuilder(model_config).build_model(pg_collection="pgs")

    def test_rejects_virtual_pipeline_stage(self, model_config):
        with pytest.raises(NotImplementedError, match="PP=1"):
            Gemma4DenseModelBuilder(model_config).build_model(pg_collection="pgs", vp_stage=0)


class TestGemma4ModelBuilder:
    @pytest.fixture
    def model_config(self):
        return Gemma4Bridge().hf_config_to_model_config(_moe_hf_config())

    def _build(self, model_config):
        seen = {}

        def fake_build_model(builder, pg_collection, pre_process=None, post_process=None, vp_stage=None):
            seen["rotary_base"] = builder._model_config.rotary_base
            seen["layer_spec"] = builder._model_config.transformer_layer_spec
            return _FakeGPTModel(builder._model_config.transformer)

        scaled_embedding = nn.Identity()
        with (
            patch.object(GPTModelBuilder, "build_model", autospec=True, side_effect=fake_build_model),
            patch(f"{_BUILDER}.Gemma3LanguageModelEmbedding", return_value=scaled_embedding) as embedding,
            patch(f"{_BUILDER}.Gemma4RotaryEmbedding", return_value="dual-rope") as rope,
            patch(f"{_BUILDER}._install_tied_kv") as tied_kv,
        ):
            model = Gemma4ModelBuilder(model_config).build_model(pg_collection="pgs")
        return (
            model,
            seen,
            SimpleNamespace(embedding=embedding, scaled_embedding=scaled_embedding, rope=rope, tied_kv=tied_kv),
        )

    def test_builds_with_the_sliding_rope_base_then_installs_dual_rope(self, model_config):
        model, seen, mocks = self._build(model_config)
        transformer = model_config.transformer

        assert seen["rotary_base"] == 10_000.0
        assert model_config.rotary_base == (10_000.0, 1_000_000.0)
        assert model.embedding is mocks.scaled_embedding
        assert mocks.embedding.call_args.kwargs["config"] is transformer
        rope_kwargs = mocks.rope.call_args.kwargs
        assert rope_kwargs["rotary_base"] == 1_000_000.0
        assert rope_kwargs["rotary_base_local"] == 10_000.0
        assert rope_kwargs["global_kv_channels"] == transformer.global_head_dim
        assert rope_kwargs["global_rotary_percent"] == transformer.global_rotary_percent
        assert model.rotary_pos_emb == "dual-rope"
        assert isinstance(model.output_layer, Gemma4OutputLayer)
        model.setup_embeddings_and_output_layer.assert_called_once_with()
        mocks.tied_kv.assert_called_once_with(model, transformer)

    def test_falls_back_to_the_gemma4_layer_spec(self, model_config):
        model_config.transformer_layer_spec = None

        _, seen, _ = self._build(model_config)

        assert seen["layer_spec"] is gemma4_moe_layer_spec
        assert model_config.transformer_layer_spec is None

    def test_rejects_single_rotary_base(self, model_config):
        model_config.rotary_base = 10_000

        with pytest.raises(ValueError, match="sliding base, global base"):
            Gemma4ModelBuilder(model_config).build_model(pg_collection="pgs")


class TestGemma4MoELayerSpec:
    def test_uses_the_transformer_config_of_a_builder_config(self):
        model_config = Gemma4Bridge().hf_config_to_model_config(_moe_hf_config())

        with patch(f"{_BUILDER}.gemma4_block_spec", return_value="block-spec") as block_spec:
            assert gemma4_moe_layer_spec(model_config) == "block-spec"

        assert block_spec.call_args.args == (model_config.transformer,)

    def test_accepts_a_transformer_config(self):
        transformer = Gemma4Bridge().hf_config_to_model_config(_moe_hf_config()).transformer

        with patch(f"{_BUILDER}.gemma4_block_spec", return_value="block-spec") as block_spec:
            gemma4_moe_layer_spec(transformer)

        assert block_spec.call_args.args == (transformer,)


@pytest.mark.parametrize("config_class", [Gemma4DenseTransformerConfig, Gemma4TransformerConfig])
def test_transformer_configs_declare_the_final_logit_softcapping(config_class):
    assert config_class(num_layers=2, hidden_size=64, num_attention_heads=4).final_logit_softcapping == 30.0


def test_dense_finalize_marks_double_wide_checkpoints_heterogeneous():
    transformer = Gemma4DenseTransformerConfig(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        use_double_wide_mlp=True,
        bf16=True,
        params_dtype=torch.bfloat16,
    )

    transformer.finalize()

    assert transformer.hetereogenous_dist_checkpoint is True


def test_moe_finalize_rejects_bypassed_execution_modes():
    transformer = Gemma4TransformerConfig(
        num_layers=2, hidden_size=64, num_attention_heads=4, moe_shared_expert_overlap=True
    )

    with pytest.raises(ValueError, match="moe_shared_expert_overlap=True"):
        transformer.finalize()
