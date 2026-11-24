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

"""A model can keep modules out of PEFT through ``peft_excluded_module_paths()`` (see ``peft.base``)."""

import pytest
import torch.nn as nn

from megatron.bridge.peft.base import excluded_module_paths
from megatron.bridge.peft.lora import LoRA
from megatron.bridge.peft.lora_layers import LoRALinear


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_a = nn.Linear(8, 8)
        self.linear_b = nn.Linear(8, 8)


class _PlainModel(nn.Module):
    """Two targets at the top level and two in a block; no exclusions."""

    def __init__(self):
        super().__init__()
        self.linear_a = nn.Linear(8, 8)
        self.linear_b = nn.Linear(8, 8)
        self.block = _Block()


class _SharingModel(_PlainModel):
    """``block`` shares its parameters with the top-level linears by aliasing, so neither side may be wrapped."""

    def __init__(self):
        super().__init__()
        self.block.linear_a.weight = self.linear_a.weight
        self.block.linear_a.bias = self.linear_a.bias

    def peft_excluded_module_paths(self):
        return {"linear_a", "block.linear_a"}


class _GeneratorModel(_SharingModel):
    """Declares the paths through a fresh generator on every call."""

    def peft_excluded_module_paths(self):
        return (path for path in ("linear_a", "block.linear_a"))


class _StoredIteratorModel(_SharingModel):
    """Declares the paths through one stored iterator, which can be consumed only once, and counts the reads."""

    def __init__(self):
        super().__init__()
        self._paths = iter(("linear_a", "block.linear_a"))
        self.reads = 0

    def peft_excluded_module_paths(self):
        self.reads += 1
        return self._paths


class _BrokenModel(_PlainModel):
    def peft_excluded_module_paths(self):
        return "linear_a"


class _NotIterableModel(_PlainModel):
    def peft_excluded_module_paths(self):
        return 42


class _BadElementModel(_PlainModel):
    def peft_excluded_module_paths(self):
        return ["linear_a", ""]


class _Wrapper(nn.Module):
    """A Float16Module-like wrapper that exposes the wrapped model as ``module``."""

    def __init__(self, module):
        super().__init__()
        self.module = module


def _lora():
    return LoRA(target_modules=["linear_a", "linear_b"], dim=4, alpha=8, dropout=0.0)


@pytest.mark.unit
def test_excluded_modules_keep_their_base_linears():
    model = _lora()(_SharingModel(), training=True)
    assert isinstance(model.linear_b, LoRALinear)
    assert isinstance(model.block.linear_b, LoRALinear)
    assert type(model.linear_a) is nn.Linear
    assert type(model.block.linear_a) is nn.Linear
    # The shared parameters are still one object, frozen; only the adapters train.
    assert model.block.linear_a.weight is model.linear_a.weight
    assert not model.linear_a.weight.requires_grad
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert trainable and all(".adapter." in name for name in trainable)
    assert {name.split(".adapter.")[0] for name in trainable} == {"linear_b", "block.linear_b"}


@pytest.mark.unit
def test_a_generator_declaration_is_honoured():
    model = _lora()(_GeneratorModel(), training=True)
    assert type(model.linear_a) is nn.Linear and type(model.block.linear_a) is nn.Linear
    assert isinstance(model.linear_b, LoRALinear) and isinstance(model.block.linear_b, LoRALinear)


@pytest.mark.unit
def test_a_cached_iterator_declaration_is_honoured(caplog):
    # PEFT.__call__ reads the declaration once per root and reuses it for the validation and transform walks,
    # so an iterator that is exhausted by its first consumer still protects both shared modules.
    model = _lora()(_StoredIteratorModel(), training=True)
    assert model.reads == 1
    assert type(model.linear_a) is nn.Linear and type(model.block.linear_a) is nn.Linear
    assert model.block.linear_a.weight is model.linear_a.weight and not model.linear_a.weight.requires_grad
    assert isinstance(model.linear_b, LoRALinear) and isinstance(model.block.linear_b, LoRALinear)
    # A target that exists only behind the exclusions is still reported as unmatched and adapts nothing.
    lora = LoRA(target_modules=["linear_a"], dim=4, alpha=8, dropout=0.0)
    with caplog.at_level("WARNING", logger="megatron.bridge.peft.module_matcher"):
        model = lora(_StoredIteratorModel(), training=True)
    assert model.reads == 1
    assert not any(isinstance(module, LoRALinear) for module in model.modules())
    assert any(
        "No modules matched" in record.getMessage() and "linear_a" in record.getMessage() for record in caplog.records
    )


@pytest.mark.unit
def test_models_without_the_method_are_transformed_as_before():
    model = _lora()(_PlainModel(), training=True)
    assert all(
        isinstance(m, LoRALinear) for m in (model.linear_a, model.linear_b, model.block.linear_a, model.block.linear_b)
    )


@pytest.mark.unit
def test_exclusions_apply_per_pipeline_chunk():
    sharing, plain = _lora()([_SharingModel(), _PlainModel()], training=True)
    assert type(sharing.linear_a) is nn.Linear and isinstance(sharing.linear_b, LoRALinear)
    assert isinstance(plain.linear_a, LoRALinear) and isinstance(plain.linear_b, LoRALinear)


@pytest.mark.unit
def test_excluded_module_paths_helper():
    assert excluded_module_paths(_PlainModel()) == frozenset()
    assert excluded_module_paths(_SharingModel()) == {"linear_a", "block.linear_a"}
    # Behind a wrapper the paths gain the wrapper's attribute name, as the walk reports them.
    assert excluded_module_paths(_Wrapper(_SharingModel())) == {"module.linear_a", "module.block.linear_a"}


@pytest.mark.unit
def test_the_walk_sees_the_wrapper_prefix():
    wrapper = _lora()(_Wrapper(_SharingModel()), training=True)
    assert type(wrapper.module.linear_a) is nn.Linear and type(wrapper.module.block.linear_a) is nn.Linear
    assert isinstance(wrapper.module.linear_b, LoRALinear) and isinstance(wrapper.module.block.linear_b, LoRALinear)


@pytest.mark.unit
def test_targets_that_only_name_excluded_modules_are_reported_as_unmatched(caplog):
    lora = LoRA(target_modules=["linear_a"], dim=4, alpha=8, dropout=0.0)
    with caplog.at_level("WARNING", logger="megatron.bridge.peft.module_matcher"):
        model = lora(_SharingModel(), training=True)
    assert not any(isinstance(module, LoRALinear) for module in model.modules())
    assert any(
        "No modules matched" in record.getMessage() and "linear_a" in record.getMessage() for record in caplog.records
    )
    # With both targets, only the one that exists nowhere outside the exclusions is reported.
    caplog.clear()
    with caplog.at_level("WARNING", logger="megatron.bridge.peft.module_matcher"):
        _lora()(_SharingModel(), training=True)
    messages = [record.getMessage() for record in caplog.records if "No modules matched" in record.getMessage()]
    assert messages and all("linear_a" in message and "linear_b" not in message for message in messages)
    # A target that is adaptable in another chunk, or in a model without exclusions, is not reported.
    caplog.clear()
    with caplog.at_level("WARNING", logger="megatron.bridge.peft.module_matcher"):
        _lora()([_SharingModel(), _PlainModel()], training=True)
        _lora()(_PlainModel(), training=True)
    assert not any("No modules matched" in record.getMessage() for record in caplog.records)


@pytest.mark.unit
@pytest.mark.parametrize("model_cls", [_BrokenModel, _NotIterableModel, _BadElementModel])
def test_malformed_exclusions_are_rejected(model_cls):
    with pytest.raises(TypeError, match="peft_excluded_module_paths"):
        _lora()(model_cls(), training=True)
