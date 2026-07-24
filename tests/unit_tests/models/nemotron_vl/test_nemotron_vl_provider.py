# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock

from megatron.bridge.models.nemotron_vl import modeling_nemotron_vl
from megatron.bridge.models.nemotron_vl import nemotron_vl_provider as provider_module
from megatron.bridge.models.nemotron_vl.nemotron_vl_provider import (
    NemotronVLModelProvider,
    get_language_mlp_submodules,
)


class TestNemotronVLModelProvider:
    def test_provider_initialization_minimal(self):
        provider = NemotronVLModelProvider(
            hybrid_layer_pattern="M-M-M-M*-M-M-M-M*-M-M-M-M-M*",
            hidden_size=5120,
            num_attention_heads=40,
        )
        provider.finalize()

        # Core fields
        assert provider.num_layers == 28
        assert provider.hidden_size == 5120
        assert provider.num_attention_heads == 40

        # VL-specific defaults
        assert provider.scatter_embedding_sequence_parallel is False
        assert provider.attention_softmax_in_fp32 is True
        assert provider.vision_model_type == "radio"
        assert provider.language_model_type == "nemotron5-hybrid-12b"

        # Freeze defaults
        assert provider.freeze_language_model is False
        assert provider.freeze_vision_model is False
        assert provider.freeze_vision_projection is False

        # API presence
        assert hasattr(provider, "provide")
        assert callable(provider.provide)
        assert hasattr(provider, "provide_language_model")
        assert callable(provider.provide_language_model)

    def test_provider_freeze_overrides(self):
        provider = NemotronVLModelProvider(
            hidden_size=5120,
            num_attention_heads=40,
            freeze_language_model=True,
            freeze_vision_model=True,
            freeze_vision_projection=True,
        )
        provider.finalize()

        assert provider.freeze_language_model is True
        assert provider.freeze_vision_model is True
        assert provider.freeze_vision_projection is True

    def test_provide_uses_provider_as_runtime_config(self, monkeypatch):
        provider = NemotronVLModelProvider()
        llava_model = SimpleNamespace()
        model = SimpleNamespace()
        model_factory = Mock(return_value=model)

        monkeypatch.setattr(provider_module, "LLaVAModel", Mock(return_value=llava_model))
        monkeypatch.setattr(provider_module, "get_vit_layer_with_transformer_engine_spec", Mock(return_value=object()))
        monkeypatch.setattr(provider_module, "get_language_mlp_submodules", Mock(return_value=object()))
        monkeypatch.setattr(modeling_nemotron_vl, "NemotronVLModel", model_factory)

        assert provider.provide() is model
        model_factory.assert_called_once_with(config=provider, llava_model=llava_model)

    def test_language_mlp_submodules_from_object_specs(self):
        mlp_submodules = SimpleNamespace(linear_fc1=object(), linear_fc2=object())
        language_spec = SimpleNamespace(
            submodules=SimpleNamespace(
                mlp_layer=SimpleNamespace(submodules=SimpleNamespace(mlp=SimpleNamespace(submodules=mlp_submodules)))
            )
        )

        assert get_language_mlp_submodules(language_spec) is mlp_submodules

    def test_language_mlp_submodules_from_partial_specs(self):
        mlp_submodules = SimpleNamespace(linear_fc1=object(), linear_fc2=object())
        language_spec = partial(
            object,
            submodules=SimpleNamespace(
                mlp_layer=partial(
                    object,
                    submodules=SimpleNamespace(mlp=partial(object, submodules=mlp_submodules)),
                )
            ),
        )

        assert get_language_mlp_submodules(language_spec) is mlp_submodules
