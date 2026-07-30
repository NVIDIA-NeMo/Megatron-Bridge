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

"""The Megatron-Core GPT facade must work with and without the upstream module."""

import builtins
import dataclasses
import importlib
import inspect

import pytest
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.models.base import ModelConfig

from megatron.bridge.compat import mcore_gpt, mcore_gpt_fallback


pytestmark = pytest.mark.unit

try:
    upstream = importlib.import_module("megatron.training.models.gpt")
except ModuleNotFoundError as error:
    if error.name != "megatron.training.models.gpt":
        raise
    upstream = None


def _public_names(module) -> set[str]:
    """Return public classes and functions defined in a module."""
    return {
        name
        for name, obj in vars(module).items()
        if not name.startswith("_") and getattr(obj, "__module__", None) == module.__name__
    }


def test_facade_exposes_the_symbols_bridge_imports():
    for name in ("GPTModelConfig", "GPTModelBuilder", "mtp_block_spec"):
        assert hasattr(mcore_gpt, name)


def test_facade_selects_the_available_implementation():
    expected = mcore_gpt_fallback if upstream is None else upstream

    assert issubclass(mcore_gpt.GPTModelConfig, expected.GPTModelConfig)
    assert mcore_gpt.GPTModelBuilder is expected.GPTModelBuilder
    assert mcore_gpt.mtp_block_spec is expected.mtp_block_spec


def test_facade_config_round_trips_through_stable_targets():
    transformer = TransformerConfig(num_layers=1, hidden_size=16, num_attention_heads=1)
    config = mcore_gpt.GPTModelConfig(transformer=transformer)
    serialized = config.as_dict()

    assert serialized["_target_"] == "megatron.bridge.compat.mcore_gpt.GPTModelConfig"
    assert serialized["_builder_"] == "megatron.bridge.compat.mcore_gpt.GPTModelBuilder"

    restored = ModelConfig.from_dict(serialized)
    assert type(restored) is mcore_gpt.GPTModelConfig
    assert restored.get_builder_cls() is mcore_gpt.GPTModelBuilder


def test_legacy_upstream_targets_are_normalized():
    data = {
        "_target_": "megatron.training.models.gpt.GPTModelConfig",
        "_builder_": "megatron.training.models.gpt.GPTModelBuilder",
    }

    assert mcore_gpt.normalize_gpt_config_targets(data) == {
        "_target_": "megatron.bridge.compat.mcore_gpt.GPTModelConfig",
        "_builder_": "megatron.bridge.compat.mcore_gpt.GPTModelBuilder",
    }
    assert data["_target_"] == "megatron.training.models.gpt.GPTModelConfig"


def test_facade_uses_fallback_only_when_upstream_module_is_missing(monkeypatch):
    real_import = builtins.__import__

    def import_with_missing_upstream(name, *args, **kwargs):
        if name == "megatron.training.models.gpt":
            raise ModuleNotFoundError(f"No module named '{name}'", name=name)
        return real_import(name, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", import_with_missing_upstream)
        _, config_cls, _ = mcore_gpt._load_gpt_symbols()
        assert config_cls is mcore_gpt_fallback.GPTModelConfig


def test_facade_reraises_transitive_module_not_found(monkeypatch):
    real_import = builtins.__import__

    def import_with_missing_dependency(name, *args, **kwargs):
        if name == "megatron.training.models.gpt":
            raise ModuleNotFoundError("No module named 'missing_dependency'", name="missing_dependency")
        return real_import(name, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", import_with_missing_dependency)
        with pytest.raises(ModuleNotFoundError, match="missing_dependency"):
            mcore_gpt._load_gpt_symbols()


@pytest.mark.skipif(upstream is None, reason="Upstream GPT module is unavailable on Megatron-Core 0.18.x")
def test_fallback_public_surface_matches_upstream():
    missing = _public_names(upstream) - _public_names(mcore_gpt_fallback)
    assert not missing, f"fallback is missing upstream symbols: {sorted(missing)}"


@pytest.mark.skipif(upstream is None, reason="Upstream GPT module is unavailable on Megatron-Core 0.18.x")
def test_fallback_config_fields_match_upstream():
    fallback_fields = {field.name for field in dataclasses.fields(mcore_gpt_fallback.GPTModelConfig)}
    upstream_fields = {field.name for field in dataclasses.fields(upstream.GPTModelConfig)}
    assert fallback_fields == upstream_fields, (
        "GPTModelConfig drifted upstream; refresh "
        "src/megatron/bridge/compat/mcore_gpt_fallback.py from megatron/training/models/gpt.py"
    )


@pytest.mark.skipif(upstream is None, reason="Upstream GPT module is unavailable on Megatron-Core 0.18.x")
def test_fallback_builder_methods_match_upstream():
    def methods(cls):
        return {name for name, _ in inspect.getmembers(cls, inspect.isfunction) if not name.startswith("_")}

    assert methods(mcore_gpt_fallback.GPTModelBuilder) == methods(upstream.GPTModelBuilder)


@pytest.mark.skipif(upstream is None, reason="Upstream GPT module is unavailable on Megatron-Core 0.18.x")
@pytest.mark.parametrize("method_name", ["build_model", "build_distributed_models"])
def test_fallback_builder_method_signatures_match_upstream(method_name):
    assert inspect.signature(getattr(mcore_gpt_fallback.GPTModelBuilder, method_name)) == inspect.signature(
        getattr(upstream.GPTModelBuilder, method_name)
    )


@pytest.mark.skipif(upstream is None, reason="Upstream GPT module is unavailable on Megatron-Core 0.18.x")
def test_fallback_mtp_block_spec_signature_matches_upstream():
    assert inspect.signature(mcore_gpt_fallback.mtp_block_spec) == inspect.signature(upstream.mtp_block_spec)
