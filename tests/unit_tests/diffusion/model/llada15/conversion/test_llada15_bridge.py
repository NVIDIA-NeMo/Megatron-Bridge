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

"""Unit tests for LLaDA15Bridge: provider mapping and weight-name registry.

Uses a flat ``types.SimpleNamespace`` mock of the OLMo-style ``LLaDAConfig``
(LLaDA1.5 has no ``text_config``). No HuggingFace download, no safetensors, no
GPU — the tests validate config→provider field mapping and the *structure* of
the weight-name registry (not numeric weight conversion, which is covered by
the GPU round-trip script).
"""

import functools
import types
from dataclasses import fields
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from megatron.bridge.diffusion.conversion.llada15.llada15_bridge import LLaDA15Bridge
from megatron.bridge.diffusion.models.llada15.llada15_attention import LLaDA15TEDotProductAttention
from megatron.bridge.diffusion.models.llada15.llada15_provider import LLaDA15ModelProvider, llada15_layer_spec
from megatron.bridge.models.conversion.param_mapping import AutoMapping, GatedMLPMapping, QKVMapping
from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.model_load_save import build_and_load_model, load_model_config
from megatron.bridge.utils.instantiate_utils import instantiate


pytestmark = [pytest.mark.unit]


def _make_hf_config(
    d_model=4096,
    n_heads=32,
    n_kv_heads=32,
    n_layers=32,
    mlp_hidden_size=12288,
    vocab_size=126464,
    embedding_size=126464,
    rope_theta=500000.0,
    rms_norm_eps=1e-5,
    weight_tying=False,
    attention_layer_norm=False,
):
    """Flat OLMo-style LLaDA config mock (no text_config)."""
    return types.SimpleNamespace(
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        n_layers=n_layers,
        mlp_hidden_size=mlp_hidden_size,
        mlp_ratio=4,
        vocab_size=vocab_size,
        embedding_size=embedding_size,
        max_sequence_length=4096,
        rms_norm_eps=rms_norm_eps,
        rope_theta=rope_theta,
        weight_tying=weight_tying,
        include_bias=False,
        include_qkv_bias=False,
        attention_layer_norm=attention_layer_norm,
        residual_dropout=0.0,
        attention_dropout=0.0,
        # LLaDA1.5's config.json sets no dtype; HF configs still expose torch_dtype.
        torch_dtype=None,
    )


class DummyHFPretrained:
    def __init__(self, hf_config):
        self.config = hf_config


class TestProviderBridge:
    def _provider(self, **kw):
        return LLaDA15Bridge().provider_bridge(DummyHFPretrained(_make_hf_config(**kw)))

    def test_returns_llada15_provider(self):
        assert isinstance(self._provider(), LLaDA15ModelProvider)

    def test_num_layers(self):
        assert self._provider(n_layers=16).num_layers == 16

    def test_hidden_size(self):
        assert self._provider(d_model=2048).hidden_size == 2048

    def test_attention_heads(self):
        p = self._provider(n_heads=16, n_kv_heads=16)
        assert p.num_attention_heads == 16
        assert p.num_query_groups == 16

    def test_ffn_hidden_size(self):
        assert self._provider(mlp_hidden_size=8192).ffn_hidden_size == 8192

    def test_vocab_prefers_embedding_size(self):
        # embedding_size (padded vocab) takes precedence over vocab_size.
        assert self._provider(vocab_size=120000, embedding_size=126464).vocab_size == 126464

    def test_rotary_base_from_rope_theta(self):
        assert self._provider(rope_theta=500000.0).rotary_base == 500000.0

    def test_position_embedding_type_is_rope(self):
        # Full RoPE handled by Megatron; must not be "none" or "learned_absolute".
        assert self._provider().position_embedding_type == "rope"

    def test_normalization_is_rmsnorm(self):
        assert self._provider().normalization == "RMSNorm"

    def test_share_embeddings_follows_weight_tying(self):
        assert self._provider(weight_tying=False).share_embeddings_and_output_weights is False
        assert self._provider(weight_tying=True).share_embeddings_and_output_weights is True

    def test_qk_layernorm_follows_config(self):
        assert self._provider(attention_layer_norm=False).qk_layernorm is False
        assert self._provider(attention_layer_norm=True).qk_layernorm is True


class TestMappingRegistry:
    def setup_method(self):
        self.registry = LLaDA15Bridge().mapping_registry()
        self.mappings = list(self.registry)

    def test_registry_not_empty(self):
        assert len(self.mappings) > 0

    def test_megatron_keys_are_bare(self):
        # Targets a bare GPTModel, not a VLM wrapper.
        for m in self.mappings:
            mp = getattr(m, "megatron_param", "")
            assert not mp.startswith("language_model."), f"unexpected prefix: {mp}"

    def test_qkv_mapping_separate_qkv(self):
        qkv = [m for m in self.mappings if isinstance(m, QKVMapping)]
        assert len(qkv) == 1
        assert "linear_qkv" in qkv[0].megatron_param
        hf = qkv[0].hf_param
        assert hf["q"].endswith("q_proj.weight")
        assert hf["k"].endswith("k_proj.weight")
        assert hf["v"].endswith("v_proj.weight")

    def test_gated_mlp_gate_is_ff_proj_up_is_up_proj(self):
        gated = [m for m in self.mappings if isinstance(m, GatedMLPMapping)]
        assert len(gated) == 1
        assert "linear_fc1" in gated[0].megatron_param
        hf = gated[0].hf_param
        # SwiGLU: ff_proj is the gate (gets SiLU), up_proj is the linear up.
        assert hf["gate"].endswith("ff_proj.weight")
        assert hf["up"].endswith("up_proj.weight")

    def test_word_embeddings_mapping(self):
        m = [
            x
            for x in self.mappings
            if isinstance(x, AutoMapping) and getattr(x, "megatron_param", "") == "embedding.word_embeddings.weight"
        ]
        assert len(m) == 1
        assert m[0].hf_param == "model.transformer.wte.weight"

    def test_output_layer_mapping(self):
        m = [
            x
            for x in self.mappings
            if isinstance(x, AutoMapping) and getattr(x, "megatron_param", "") == "output_layer.weight"
        ]
        assert len(m) == 1
        assert m[0].hf_param == "model.transformer.ff_out.weight"

    def test_final_layernorm_mapping(self):
        m = [
            x
            for x in self.mappings
            if isinstance(x, AutoMapping) and getattr(x, "megatron_param", "") == "decoder.final_layernorm.weight"
        ]
        assert len(m) == 1
        assert m[0].hf_param == "model.transformer.ln_f.weight"


def _legacy_provider(hf_config) -> LLaDA15ModelProvider:
    """Build the provider the way LLaDA15Bridge.provider_bridge() did before builder-backed conversion."""
    hidden_size = hf_config.d_model
    num_attention_heads = hf_config.n_heads
    vocab_size = hf_config.embedding_size or hf_config.vocab_size
    return LLaDA15ModelProvider(
        num_layers=hf_config.n_layers,
        hidden_size=hidden_size,
        ffn_hidden_size=(
            hf_config.mlp_hidden_size if hf_config.mlp_hidden_size is not None else hf_config.mlp_ratio * hidden_size
        ),
        num_attention_heads=num_attention_heads,
        num_query_groups=hf_config.n_kv_heads if hf_config.n_kv_heads is not None else hf_config.n_heads,
        kv_channels=hidden_size // num_attention_heads,
        vocab_size=vocab_size,
        seq_length=hf_config.max_sequence_length,
        layernorm_epsilon=hf_config.rms_norm_eps,
        rotary_base=hf_config.rope_theta,
        rotary_percent=1.0,
        share_embeddings_and_output_weights=hf_config.weight_tying,
        add_bias_linear=hf_config.include_bias,
        add_qkv_bias=bool(hf_config.include_bias or hf_config.include_qkv_bias),
        normalization="RMSNorm",
        gated_linear_unit=True,
        activation_func=F.silu,
        qk_layernorm=hf_config.attention_layer_norm,
        hidden_dropout=hf_config.residual_dropout,
        attention_dropout=hf_config.attention_dropout,
        bf16=True,
        params_dtype=torch.bfloat16,
        autocast_dtype=torch.bfloat16,
        make_vocab_size_divisible_by=LLaDA15Bridge().make_vocab_size_divisible_by(vocab_size),
    )


def _config_variant(name):
    """Return the LLaDA1.5 config with one architecture option changed."""
    hf_config = _make_hf_config()
    if name == "tied":
        hf_config.weight_tying = True
    elif name == "qk_norm":
        hf_config.attention_layer_norm = True
    elif name == "mha_default":
        hf_config.n_kv_heads = None
    elif name == "ffn_ratio":
        hf_config.mlp_hidden_size = None
    elif name == "vocab_fallback":
        hf_config.embedding_size = None
        hf_config.vocab_size = 126000
    elif name == "bias":
        hf_config.include_bias = True
    elif name == "qkv_bias":
        hf_config.include_qkv_bias = True
    elif name == "dropout":
        hf_config.residual_dropout = 0.1
        hf_config.attention_dropout = 0.05
    elif name == "fp16_checkpoint":
        hf_config.torch_dtype = "float16"
    return hf_config


_CONFIG_VARIANTS = [
    "default",
    "tied",
    "qk_norm",
    "mha_default",
    "ffn_ratio",
    "vocab_fallback",
    "bias",
    "qkv_bias",
    "dropout",
    "fp16_checkpoint",
]


def _comparable(value):
    """functools.partial compares by identity, so compare its structure instead."""
    if isinstance(value, functools.partial):
        return value.func, value.args, value.keywords
    return value


def _save_run_config(model_config: BridgeGPTModelConfig, directory) -> None:
    """Write the model config as a checkpoint's run_config.yaml."""
    ConfigContainer(
        model=model_config,
        train=None,
        optimizer=None,
        scheduler=None,
        dataset=None,
        logger=None,
        tokenizer=None,
        checkpoint=None,
    ).to_yaml(str(directory / "run_config.yaml"))


class TestLLaDA15BuilderConfig:
    """Builder-backed ModelConfig path for LLaDA1.5."""

    def test_provider_bridge_is_inherited_compatibility_only(self):
        assert "provider_bridge" not in LLaDA15Bridge.__dict__

    def test_conversion_uses_builder_config(self):
        assert LLaDA15Bridge.MODEL_CONFIG_CLASS is BridgeGPTModelConfig
        assert LLaDA15Bridge.USE_MODEL_CONFIG_FOR_CONVERSION is True
        assert LLaDA15Bridge.PROVIDER_CLASS is LLaDA15ModelProvider

    def test_hf_config_to_model_config_uses_direct_mapping(self):
        bridge = LLaDA15Bridge()
        with (
            patch.object(bridge, "provider_bridge", side_effect=AssertionError("provider path used")),
            patch.object(bridge, "hf_config_to_provider_kwargs", side_effect=AssertionError("provider kwargs used")),
        ):
            result = bridge.hf_config_to_model_config(_make_hf_config())

        assert type(result) is BridgeGPTModelConfig
        assert result.transformer_layer_spec is llada15_layer_spec
        assert result.position_embedding_type == "rope"
        assert result.share_embeddings_and_output_weights is False
        assert result.num_layers == 32
        assert result.hidden_size == 4096
        assert result.ffn_hidden_size == 12288
        assert result.num_query_groups == 32
        assert result.kv_channels == 128
        assert result.vocab_size == 126464
        assert result.seq_length == 4096
        assert result.rotary_base == 500000.0
        assert result.activation_func is F.silu
        assert result.bf16 is True
        assert result.fp16 is False
        assert result.params_dtype == torch.bfloat16

    @pytest.mark.parametrize("variant", _CONFIG_VARIANTS)
    def test_model_config_matches_branch_and_legacy_provider(self, variant):
        hf_config = _config_variant(variant)
        bridge = LLaDA15Bridge()

        model_config = bridge.hf_config_to_model_config(hf_config)
        with pytest.warns(FutureWarning, match=r"deprecated.*get_model_config.*get_model"):
            provider = bridge.provider_bridge(DummyHFPretrained(hf_config))
        legacy_provider = _legacy_provider(hf_config)

        assert type(provider) is LLaDA15ModelProvider
        # The provider path is unchanged on every provider field.
        for field in fields(legacy_provider):
            assert getattr(provider, field.name) == getattr(legacy_provider, field.name), field.name
        provider_fields = {field.name for field in fields(provider)}
        model_config_fields = {field.name for field in fields(model_config)}
        model_config_fields.update(field.name for field in fields(model_config.transformer))
        shared_fields = sorted(provider_fields & model_config_fields)
        assert len(shared_fields) > 300
        for field_name in shared_fields:
            assert getattr(model_config, field_name) == getattr(provider, field_name), field_name

        model_config.finalize()
        provider.finalize()
        for field_name in shared_fields:
            assert _comparable(getattr(model_config, field_name)) == _comparable(getattr(provider, field_name)), (
                field_name
            )

    def test_layer_spec_matches_between_provider_and_builder_config(self):
        """Both paths build the TE GPT layer with bidirectional LLaDA1.5 core attention."""
        hf_config = _make_hf_config(n_layers=2, d_model=128, n_heads=4, n_kv_heads=4, mlp_hidden_size=256)
        bridge = LLaDA15Bridge()
        model_config = bridge.hf_config_to_model_config(hf_config)
        with pytest.warns(FutureWarning):
            provider = bridge.provider_bridge(DummyHFPretrained(hf_config))

        builder_spec = llada15_layer_spec(model_config)
        assert builder_spec.submodules.self_attention.submodules.core_attention is LLaDA15TEDotProductAttention
        assert repr(builder_spec) == repr(llada15_layer_spec(provider))

    def test_run_config_round_trip_restores_layer_spec(self, tmp_path):
        model_config = LLaDA15Bridge().hf_config_to_model_config(_make_hf_config())

        _save_run_config(model_config, tmp_path)
        loaded, _ = load_model_config(str(tmp_path))

        assert type(loaded) is BridgeGPTModelConfig
        assert loaded.transformer_layer_spec is llada15_layer_spec

    @pytest.mark.parametrize("saved_cpu_init", [True, False])
    @pytest.mark.parametrize("use_cpu_init", [False, True])
    def test_load_builds_on_requested_device(self, tmp_path, saved_cpu_init, use_cpu_init):
        """The load request, not the flag a CPU import saved, decides where the model is built."""
        model_config = LLaDA15Bridge().hf_config_to_model_config(_make_hf_config())
        # AutoBridge.import_ckpt() builds on CPU and saves use_cpu_initialization=True.
        model_config.transformer.use_cpu_initialization = saved_cpu_init
        _save_run_config(model_config, tmp_path)
        loaded, _ = load_model_config(str(tmp_path))
        assert loaded.transformer.use_cpu_initialization is saved_cpu_init

        # Built chunks are modules: the loader keeps the outer config on each one for later saves.
        model = [torch.nn.Module()]
        cpu_init_at_build = []

        def build_distributed_models(*args, **kwargs):
            cpu_init_at_build.append(loaded.transformer.use_cpu_initialization)
            return model

        with (
            patch.object(loaded.get_builder_cls(), "build_distributed_models", build_distributed_models),
            patch("megatron.bridge.training.model_load_save._get_or_initialize_pg_collection"),
            patch("megatron.bridge.training.checkpointing._load_model_weights_from_checkpoint"),
        ):
            result = build_and_load_model(
                str(tmp_path), loaded, use_cpu_init=use_cpu_init, skip_temp_dist_context=True
            )

        assert result is model
        assert cpu_init_at_build == [use_cpu_init]
        assert result[0].model_config is loaded

    def test_legacy_private_layer_spec_target_resolves(self):
        """Provider checkpoints saved before the rename name the spec by its private alias."""
        target = "megatron.bridge.diffusion.models.llada15.llada15_provider._llada15_layer_spec"

        assert instantiate({"_target_": target, "_call_": False}) is llada15_layer_spec
