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

import importlib.util
from pathlib import Path
from unittest.mock import Mock

import pytest


pytestmark = pytest.mark.unit


@pytest.fixture
def distillation_example():
    path = Path(__file__).parents[3] / "examples" / "distillation" / "llama" / "distill_llama32_3b-1b.py"
    spec = importlib.util.spec_from_file_location("distill_llama32_example", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_distillation_example_prefers_builder_configs(distillation_example, monkeypatch):
    student_bridge = Mock()
    teacher_bridge = Mock()
    model_config = object()
    student_bridge.get_model_config.return_value = model_config
    teacher_bridge.get_model_config.return_value = model_config
    monkeypatch.setattr(
        distillation_example.AutoBridge,
        "from_hf_pretrained",
        Mock(side_effect=[student_bridge, teacher_bridge]),
    )
    converted = Mock(pre_wrap_hooks=[Mock()])
    converter = Mock(return_value=converted)
    monkeypatch.setattr(distillation_example, "convert_to_distillation_provider", converter)
    kd_config = Mock()

    result = distillation_example._build_distillation_model_config(kd_config)

    assert result is converted
    converter.assert_called_once_with(model_config, model_config, kd_config)
    student_bridge.to_megatron_provider.assert_not_called()
    teacher_bridge.to_megatron_provider.assert_not_called()


def test_distillation_example_rejects_builder_config_without_weight_hooks(distillation_example, monkeypatch):
    student_bridge = Mock()
    teacher_bridge = Mock()
    model_config_without_hooks = Mock(pre_wrap_hooks=[])
    converted_provider = object()
    student_bridge.to_megatron_provider.return_value = object()
    teacher_bridge.to_megatron_provider.return_value = object()
    monkeypatch.setattr(
        distillation_example.AutoBridge,
        "from_hf_pretrained",
        Mock(side_effect=[student_bridge, teacher_bridge]),
    )
    converter = Mock(side_effect=[model_config_without_hooks, converted_provider])
    monkeypatch.setattr(distillation_example, "convert_to_distillation_provider", converter)

    result = distillation_example._build_distillation_model_config(Mock())

    assert result is converted_provider
    student_bridge.to_megatron_provider.assert_called_once_with(load_weights=True)
    teacher_bridge.to_megatron_provider.assert_called_once_with(load_weights=True)


def test_distillation_example_falls_back_for_unsupported_builder_configs(distillation_example, monkeypatch):
    student_bridge = Mock()
    teacher_bridge = Mock()
    student_config = object()
    teacher_config = object()
    student_provider = object()
    teacher_provider = object()
    student_bridge.get_model_config.return_value = student_config
    teacher_bridge.get_model_config.return_value = teacher_config
    student_bridge.to_megatron_provider.return_value = student_provider
    teacher_bridge.to_megatron_provider.return_value = teacher_provider
    monkeypatch.setattr(
        distillation_example.AutoBridge,
        "from_hf_pretrained",
        Mock(side_effect=[student_bridge, teacher_bridge]),
    )
    converted = object()
    converter = Mock(
        side_effect=[
            AssertionError("Student provider must be a subclass of GPTModelProvider or HybridModelProvider."),
            converted,
        ]
    )
    monkeypatch.setattr(distillation_example, "convert_to_distillation_provider", converter)
    kd_config = Mock()

    result = distillation_example._build_distillation_model_config(kd_config)

    assert result is converted
    assert converter.call_args_list[0].args == (student_config, teacher_config, kd_config)
    assert converter.call_args_list[1].args == (student_provider, teacher_provider, kd_config)
    student_bridge.to_megatron_provider.assert_called_once_with(load_weights=True)
    teacher_bridge.to_megatron_provider.assert_called_once_with(load_weights=True)


def test_distillation_example_does_not_hide_unexpected_assertions(distillation_example, monkeypatch):
    student_bridge = Mock()
    teacher_bridge = Mock()
    monkeypatch.setattr(
        distillation_example.AutoBridge,
        "from_hf_pretrained",
        Mock(side_effect=[student_bridge, teacher_bridge]),
    )
    converter = Mock(side_effect=AssertionError("unexpected conversion failure"))
    monkeypatch.setattr(distillation_example, "convert_to_distillation_provider", converter)

    with pytest.raises(AssertionError, match="unexpected conversion failure"):
        distillation_example._build_distillation_model_config(Mock())

    student_bridge.to_megatron_provider.assert_not_called()
    teacher_bridge.to_megatron_provider.assert_not_called()
