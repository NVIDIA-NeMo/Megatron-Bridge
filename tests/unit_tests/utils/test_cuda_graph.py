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

import pytest
from megatron.core.transformer.transformer_config import TransformerConfig

from megatron.bridge.models.hybrid.hybrid_builder import HybridModelConfig
from megatron.bridge.utils.cuda_graph import enable_hybrid_cuda_graph_pairs


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
