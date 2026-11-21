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

"""Tests for the Megatron-Core CUDA graph helpers in ``megatron.bridge.utils.cuda_graph``."""

import dataclasses
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from megatron.core.transformer import cuda_graphs as mcore_cuda_graphs
from megatron.core.transformer.transformer_config import TransformerConfig

from megatron.bridge.models.hybrid.hybrid_builder import HybridModelConfig
from megatron.bridge.utils.cuda_graph import enable_hybrid_cuda_graph_pairs, get_vision_cuda_graph_seq_length


pytestmark = pytest.mark.unit

_PAIR_FIELD = "cuda_graph_pair_hybrid_layers"


@dataclasses.dataclass
class _WithField:
    cuda_graph_pair_hybrid_layers: bool = False


@dataclasses.dataclass
class _WithFactoryField:  # no class attribute, only an instance one
    cuda_graph_pair_hybrid_layers: bool = dataclasses.field(default_factory=lambda: False)


@dataclasses.dataclass
class _WithoutField:
    cuda_graph_impl: str = "transformer_engine"


@dataclasses.dataclass
class _TransformerConfigWithField(TransformerConfig):
    """A transformer config from a Megatron-Core that declares the pairing field."""

    cuda_graph_pair_hybrid_layers: bool = False


def _small_transformer_config(config_class: type[TransformerConfig]) -> TransformerConfig:
    return config_class(num_layers=2, hidden_size=64, num_attention_heads=4)


@pytest.mark.parametrize("config_class", [_WithField, _WithFactoryField])
def test_enable_hybrid_cuda_graph_pairs_sets_a_declared_field(config_class: type) -> None:
    config = config_class()

    assert enable_hybrid_cuda_graph_pairs(config) is True
    assert config.cuda_graph_pair_hybrid_layers is True


def test_enable_hybrid_cuda_graph_pairs_does_not_invent_the_field() -> None:
    """Without the Megatron-Core field the helper changes nothing: an invented attribute would be silently unused."""
    config = _WithoutField()

    assert enable_hybrid_cuda_graph_pairs(config) is False
    assert not hasattr(config, _PAIR_FIELD)
    assert vars(config) == {"cuda_graph_impl": "transformer_engine"}


@pytest.mark.parametrize("transformer_class", [TransformerConfig, _TransformerConfigWithField])
def test_enable_hybrid_cuda_graph_pairs_reaches_the_hybrid_transformer_config(
    transformer_class: type[TransformerConfig],
) -> None:
    """A Hybrid model config forwards the field to its transformer config and never stores it itself."""
    transformer = _small_transformer_config(transformer_class)
    config = HybridModelConfig(transformer=transformer)
    has_field = any(field.name == _PAIR_FIELD for field in dataclasses.fields(transformer_class))

    assert enable_hybrid_cuda_graph_pairs(config) is has_field
    assert _PAIR_FIELD not in vars(config)
    if has_field:
        assert transformer.cuda_graph_pair_hybrid_layers is True
    else:
        assert not hasattr(transformer, _PAIR_FIELD)


def test_megatron_core_cuda_graph_pair_contract() -> None:
    """Megatron-Core's ``TransformerConfig.cuda_graph_pair_hybrid_layers`` is an opt-in flag that the helper sets.

    The helper's guard turns a missing field into a no-op, so a rename in Megatron-Core would leave the recipes on one
    graph per Hybrid position without any error. Once the pinned Megatron-Core has the field, this pins its name and
    its off-by-default value.
    """
    fields = {field.name: field for field in dataclasses.fields(TransformerConfig)}
    if _PAIR_FIELD not in fields:
        pytest.skip("Megatron-Core predates TransformerConfig.cuda_graph_pair_hybrid_layers")

    assert fields[_PAIR_FIELD].default is False
    config = _small_transformer_config(TransformerConfig)
    assert enable_hybrid_cuda_graph_pairs(config) is True
    assert config.cuda_graph_pair_hybrid_layers is True


@pytest.mark.parametrize("helper_name", ["get_vision_te_cuda_graph_seq_length", "get_vision_cuda_graph_seq_length"])
def test_vision_graph_length_supports_mcore_helper_names(monkeypatch, helper_name: str) -> None:
    """Both MCore revisions preserve the config and explicit fallback length."""
    monkeypatch.delattr(mcore_cuda_graphs, "get_vision_te_cuda_graph_seq_length", raising=False)
    monkeypatch.delattr(mcore_cuda_graphs, "get_vision_cuda_graph_seq_length", raising=False)
    helper = Mock(return_value=123)
    monkeypatch.setattr(mcore_cuda_graphs, helper_name, helper, raising=False)
    config = SimpleNamespace()

    assert get_vision_cuda_graph_seq_length(config, default_seq_length=73) == 123
    helper.assert_called_once_with(config, default_seq_length=73)


def test_vision_graph_length_prefers_current_helper_without_hiding_errors(monkeypatch) -> None:
    current = Mock(side_effect=ValueError("invalid vision configuration"))
    legacy = Mock(return_value=123)
    monkeypatch.setattr(mcore_cuda_graphs, "get_vision_te_cuda_graph_seq_length", current, raising=False)
    monkeypatch.setattr(mcore_cuda_graphs, "get_vision_cuda_graph_seq_length", legacy, raising=False)

    with pytest.raises(ValueError, match="invalid vision configuration"):
        get_vision_cuda_graph_seq_length(SimpleNamespace())
    legacy.assert_not_called()


def test_vision_graph_length_reports_missing_mcore_helpers(monkeypatch) -> None:
    monkeypatch.delattr(mcore_cuda_graphs, "get_vision_te_cuda_graph_seq_length", raising=False)
    monkeypatch.delattr(mcore_cuda_graphs, "get_vision_cuda_graph_seq_length", raising=False)

    with pytest.raises(
        ImportError, match="neither get_vision_te_cuda_graph_seq_length nor get_vision_cuda_graph_seq_length"
    ):
        get_vision_cuda_graph_seq_length(SimpleNamespace())


def test_vision_graph_length_delegates_to_installed_mcore() -> None:
    """Exercise the real helper API without constructing a vision encoder or CUDA state."""
    config = SimpleNamespace(num_position_embeddings=4096, spatial_merge_size=2)
    assert get_vision_cuda_graph_seq_length(config) == 1024
    assert get_vision_cuda_graph_seq_length(SimpleNamespace(), default_seq_length=73) == 73
