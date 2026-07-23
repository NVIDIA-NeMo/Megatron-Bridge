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

"""Unit tests for the Gemma 4 E4B pre-training recipe."""

import importlib.util
import os
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import pytest

from megatron.bridge.models.gemma.gemma4_bridge import Gemma4Bridge
from megatron.bridge.models.gemma.gemma4_provider import Gemma4DenseProvider
from tests.unit_tests.recipes.recipe_test_utils import patch_recipe_module_global


def _minimal_pretrain_common():
    cfg = types.SimpleNamespace()
    cfg.tokenizer = types.SimpleNamespace()
    cfg.dataset = types.SimpleNamespace()
    cfg.train = types.SimpleNamespace()
    cfg.validation = types.SimpleNamespace()
    cfg.scheduler = types.SimpleNamespace()
    cfg.optimizer = types.SimpleNamespace()
    cfg.ddp = types.SimpleNamespace()
    return cfg


class _FakeAutoBridge:
    def __init__(self, provider):
        self.provider = provider
        self.hf_paths = []
        self.load_weights = []
        self.conversion_modes = []

    def from_hf_pretrained(self, hf_path):
        self.hf_paths.append(hf_path)
        self.conversion_modes.append(os.environ.get("GEMMA4_CONVERSION_MODE"))
        return self

    def to_megatron_provider(self, load_weights=True):
        self.load_weights.append(load_weights)
        return self.provider


def _load_gemma4_recipe_module():
    """Load the Gemma4 recipe without importing the umbrella recipes package."""
    bridge_root = Path(__file__).resolve().parents[3]
    recipes_root = bridge_root / "src" / "megatron" / "bridge" / "recipes"

    recipes_pkg = types.ModuleType("megatron.bridge.recipes")
    recipes_pkg.__path__ = [str(recipes_root)]
    sys.modules.setdefault("megatron.bridge.recipes", recipes_pkg)

    common_mod = types.ModuleType("megatron.bridge.recipes.common")
    common_mod._pretrain_common = _minimal_pretrain_common
    sys.modules["megatron.bridge.recipes.common"] = common_mod

    utils_pkg = types.ModuleType("megatron.bridge.recipes.utils")
    utils_pkg.__path__ = [str(recipes_root / "utils")]
    sys.modules.setdefault("megatron.bridge.recipes.utils", utils_pkg)

    tokenizer_mod = types.ModuleType("megatron.bridge.recipes.utils.tokenizer_utils")
    tokenizer_mod.DEFAULT_NULL_TOKENIZER_VOCAB_SIZE = 32000
    sys.modules["megatron.bridge.recipes.utils.tokenizer_utils"] = tokenizer_mod

    recipe_path = recipes_root / "gemma" / "gemma4.py"
    spec = importlib.util.spec_from_file_location("_gemma4_recipe_under_test", recipe_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Minimal HF config that mirrors google/gemma-4-E4B-it
# ---------------------------------------------------------------------------


@pytest.fixture
def hf_config_e4b():
    """Minimal HF config mirroring google/gemma-4-E4B-it."""
    cfg = Mock(spec=[])
    cfg.num_hidden_layers = 42
    cfg.hidden_size = 2560
    cfg.intermediate_size = 10240
    cfg.num_attention_heads = 8
    cfg.num_key_value_heads = 2
    cfg.head_dim = 256
    cfg.global_head_dim = 512
    cfg.num_global_key_value_heads = 2
    cfg.rms_norm_eps = 1e-6
    cfg.vocab_size = 262143
    cfg.vocab_size_per_layer_input = 262144
    cfg.hidden_size_per_layer_input = 256
    cfg.max_position_embeddings = 131072
    cfg.enable_moe_block = False
    cfg.num_kv_shared_layers = 18
    cfg.rope_parameters = {
        "sliding_attention": {"rope_theta": 10000.0},
        "full_attention": {"rope_theta": 1000000.0, "partial_rotary_factor": 0.25},
    }
    cfg.layer_types = None
    return cfg


@pytest.fixture
def bridge_provider(hf_config_e4b):
    return Gemma4Bridge()._build_dense_provider(hf_config_e4b)


@pytest.fixture
def recipe_module():
    return _load_gemma4_recipe_module()


@pytest.fixture
def fake_autobridge(recipe_module, hf_config_e4b, monkeypatch):
    provider = Gemma4Bridge()._build_dense_provider(hf_config_e4b)
    fake = _FakeAutoBridge(provider)
    patch_recipe_module_global(monkeypatch, recipe_module, "AutoBridge", fake)
    return fake


@pytest.fixture
def recipe_config(recipe_module, fake_autobridge):
    return recipe_module.gemma4_e4b_pretrain_config()


@pytest.fixture
def recipe_provider(recipe_config):
    return recipe_config.model


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGemma4RecipeAutoBridge:
    def test_recipe_uses_autobridge_for_text_provider(self, recipe_module, fake_autobridge, monkeypatch):
        monkeypatch.setenv("GEMMA4_CONVERSION_MODE", "vl")

        cfg = recipe_module.gemma4_e4b_pretrain_config()

        assert isinstance(cfg.model, Gemma4DenseProvider)
        assert fake_autobridge.hf_paths == [recipe_module._GEMMA4_E4B_HF_PATH]
        assert fake_autobridge.load_weights == [False]
        assert fake_autobridge.conversion_modes == ["text"]
        assert os.environ["GEMMA4_CONVERSION_MODE"] == "vl"

    def test_recipe_clears_scoped_text_mode_when_unset(self, recipe_module, fake_autobridge, monkeypatch):
        monkeypatch.delenv("GEMMA4_CONVERSION_MODE", raising=False)

        recipe_module.gemma4_e4b_pretrain_config()

        assert fake_autobridge.conversion_modes == ["text"]
        assert "GEMMA4_CONVERSION_MODE" not in os.environ

    def test_text_conversion_mode_restores_env_on_exception(self, recipe_module, monkeypatch):
        monkeypatch.setenv("GEMMA4_CONVERSION_MODE", "vl")

        with pytest.raises(RuntimeError, match="boom"):
            with recipe_module._gemma4_text_conversion_mode():
                assert os.environ["GEMMA4_CONVERSION_MODE"] == "text"
                raise RuntimeError("boom")

        assert os.environ["GEMMA4_CONVERSION_MODE"] == "vl"


class TestGemma4RecipeProviderType:
    def test_recipe_returns_dense_provider(self, recipe_provider):
        assert isinstance(recipe_provider, Gemma4DenseProvider)

    def test_bridge_returns_dense_provider(self, bridge_provider):
        assert isinstance(bridge_provider, Gemma4DenseProvider)


class TestGemma4RecipeOverrides:
    def test_recipe_runtime_overrides(self, recipe_config):
        assert recipe_config.tokenizer.tokenizer_type == "NullTokenizer"
        assert recipe_config.tokenizer.tokenizer_model is None
        assert recipe_config.tokenizer.vocab_size == 32000
        assert recipe_config.dataset.blend is None
        assert recipe_config.dataset.seq_length == 4096
        assert recipe_config.model.seq_length == 4096
        assert recipe_config.model.tensor_model_parallel_size == 2
        assert recipe_config.model.pipeline_model_parallel_size == 1
        assert recipe_config.model.transformer_impl == "local"
        assert recipe_config.model.masked_softmax_fusion is False
        assert recipe_config.model.gradient_accumulation_fusion is False


class TestGemma4RecipeProviderDrift:
    """Critical fields must match between recipe and bridge-derived provider."""

    # seq_length is intentionally excluded: the recipe uses a shorter default
    # training sequence length, while the bridge mirrors the HF max position.
    CRITICAL_FIELDS = [
        "num_layers",
        "hidden_size",
        "ffn_hidden_size",
        "num_attention_heads",
        "num_query_groups",
        "kv_channels",
        "global_kv_channels",
        "num_global_query_groups",
        "vocab_size",
        "make_vocab_size_divisible_by",
        "normalization",
        "layernorm_epsilon",
        "gated_linear_unit",
        "add_bias_linear",
        "attention_dropout",
        "hidden_dropout",
        "window_size",
        "window_attn_skip_freq",
        "sliding_window_rope_base",
        "full_attention_rope_base",
        "full_attention_rope_partial_factor",
        "num_kv_shared_layers",
        "per_layer_embed_vocab_size",
        "per_layer_embed_dim",
    ]

    @pytest.mark.parametrize("field", CRITICAL_FIELDS)
    def test_field_matches_bridge(self, recipe_provider, bridge_provider, field):
        recipe_val = getattr(recipe_provider, field)
        bridge_val = getattr(bridge_provider, field)
        assert recipe_val == bridge_val, (
            f"Recipe and bridge-derived provider differ on '{field}': "
            f"recipe={recipe_val!r}, bridge={bridge_val!r}. "
            f"Update gemma4_e4b_pretrain_config() to match Gemma4Bridge._build_dense_provider()."
        )
