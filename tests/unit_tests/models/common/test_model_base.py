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

import copy
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import ClassVar

import pytest
import torch

from megatron.bridge.models.common.base import ModelBuilder, ModelConfig, compose_hooks
from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig
from megatron.bridge.models.hybrid import BridgeHybridModelConfig
from megatron.bridge.models.transformer_config import TransformerConfig
from megatron.bridge.utils.instantiate_utils import (
    _ALLOWED_TARGET_PREFIXES,
    InstantiationException,
    register_allowed_target_prefix,
    target_allowlist,
)


pytestmark = pytest.mark.unit


@dataclass
class DummyModelConfig(ModelConfig):
    builder: ClassVar[str] = ""
    value: int = 42
    name: str = "test"


class DummyModelBuilder(ModelBuilder):
    def build_model(self, pg_collection, pre_process=None, post_process=None, vp_stage=None):
        return None

    def build_distributed_models(self, pg_collection, **kwargs):
        return []


DummyModelConfig.builder = f"{DummyModelBuilder.__module__}.DummyModelBuilder"


@dataclass
class DummySubConfig:
    x: int = 1
    y: str = "sub"


@dataclass
class DummyDerivedSubConfig:
    value: int = 1
    derived: int = field(init=False, default=2)


def dummy_callable() -> None:
    """Placeholder callable used as a field default in DummyNestedModelConfig."""


@dataclass
class DummyNestedModelConfig(ModelConfig):
    builder: ClassVar[str] = ""
    sub: DummySubConfig = field(default_factory=DummySubConfig)
    fn_field: Callable = dummy_callable
    extra: int = 0


DummyNestedModelConfig.builder = f"{DummyModelBuilder.__module__}.DummyModelBuilder"


@dataclass
class DummyDerivedModelConfig(ModelConfig):
    builder: ClassVar[str] = ""
    sub: DummyDerivedSubConfig = field(default_factory=DummyDerivedSubConfig)


DummyDerivedModelConfig.builder = f"{DummyModelBuilder.__module__}.DummyModelBuilder"


@pytest.fixture(autouse=True)
def _register_test_target_prefix():
    prefix = f"{__name__}."
    added_to_bridge = prefix not in _ALLOWED_TARGET_PREFIXES
    added_to_mlm = prefix not in target_allowlist.allowed_prefixes

    register_allowed_target_prefix(prefix)
    yield

    if added_to_bridge:
        _ALLOWED_TARGET_PREFIXES.discard(prefix)
    if added_to_mlm:
        target_allowlist.remove_prefix(prefix)


def test_model_config_get_builder_cls_uses_validated_target() -> None:
    cfg = DummyModelConfig()

    assert cfg.get_builder_cls() is DummyModelBuilder


@pytest.mark.parametrize("fn_field", [dummy_callable, torch.nn.functional.relu])
def test_model_config_from_dict_round_trips_nested_config(fn_field: Callable[..., object]) -> None:
    original = DummyNestedModelConfig(sub=DummySubConfig(x=7, y="nested"), fn_field=fn_field, extra=99)

    cfg = ModelConfig.from_dict(original.as_dict())

    assert isinstance(cfg, DummyNestedModelConfig)
    assert cfg.extra == 99
    assert isinstance(cfg.sub, DummySubConfig)
    assert cfg.sub.x == 7
    assert cfg.sub.y == "nested"
    assert cfg.fn_field is fn_field


def test_model_config_from_dict_ignores_non_init_derived_fields() -> None:
    original = DummyDerivedModelConfig(sub=DummyDerivedSubConfig(value=7))

    cfg = ModelConfig.from_dict(original.as_dict())

    assert isinstance(cfg, DummyDerivedModelConfig)
    assert cfg.sub.value == 7
    assert cfg.sub.derived == 2


def test_model_config_from_dict_rejects_disallowed_target() -> None:
    data = DummyNestedModelConfig().as_dict()
    data["sub"] = {"_target_": "os.system"}

    with pytest.raises(InstantiationException, match="is not allowed"):
        ModelConfig.from_dict(data)


def test_model_config_from_dict_rejects_disallowed_builder() -> None:
    data = DummyModelConfig().as_dict()
    data["_builder_"] = "os.system"

    with pytest.raises(InstantiationException, match="is not allowed"):
        ModelConfig.from_dict(data)


def identity_hook(model: list[torch.nn.Module]) -> list[torch.nn.Module]:
    """Pre-wrap hook that returns the model unchanged."""
    return model


@pytest.mark.parametrize("config_cls", [BridgeGPTModelConfig, BridgeHybridModelConfig])
def test_builder_config_deepcopy_keeps_outer_fields(
    config_cls: type[BridgeGPTModelConfig | BridgeHybridModelConfig],
) -> None:
    original = config_cls(
        transformer=TransformerConfig(num_layers=2, hidden_size=16, num_attention_heads=2),
        seq_length=64,
        vocab_size=128,
    )
    original.pre_wrap_hooks.append(identity_hook)

    copied = copy.deepcopy(original)

    assert type(copied) is config_cls
    assert copied.seq_length == 64
    assert copied.vocab_size == 128
    assert copied.num_layers == 2
    assert copied.transformer is not original.transformer
    assert copied.pre_wrap_hooks == [identity_hook]
    assert copied.pre_wrap_hooks is not original.pre_wrap_hooks

    copied.hidden_size = 32
    assert copied.transformer.hidden_size == 32
    assert original.hidden_size == 16


def test_compose_hooks_preserves_order() -> None:
    call_log = []

    def first(value):
        call_log.append("first")
        return value + 1

    def second(value):
        call_log.append("second")
        return value * 2

    assert compose_hooks([first, second])(3) == 8
    assert call_log == ["first", "second"]
