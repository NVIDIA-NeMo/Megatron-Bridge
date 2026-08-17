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

"""Unit tests for Step37ModelProvider."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.stepfun.step35_provider import Step35ModelProvider
from megatron.bridge.models.stepfun.step37_provider import Step37ModelProvider


pytestmark = pytest.mark.unit


def _provider(**overrides):
    kwargs = dict(num_layers=2, hidden_size=64, num_attention_heads=4)
    kwargs.update(overrides)
    provider = Step37ModelProvider(**kwargs)
    provider._pg_collection = SimpleNamespace()
    return provider


class TestStep37ProviderDefaults:
    def test_multimodal_defaults(self):
        provider = _provider()

        assert provider.vision_config is None
        assert provider.image_token_id == 128001
        assert provider.understand_projector_stride == 2
        assert provider.projector_bias is False
        assert provider.language_max_sequence_length == 262144

    def test_position_embedding_type_is_rope(self):
        # Overrides GPTModelProvider's "learned_absolute" so recipe-only paths
        # get a legal pairing with rotary_base_per_layer.
        assert _provider().position_embedding_type == "rope"

    def test_freeze_and_pp_gating_defaults(self):
        provider = _provider()

        assert provider.freeze_language_model is False
        assert provider.freeze_vision_model is False
        assert provider.freeze_vision_projection is False
        assert provider.add_encoder is True
        assert provider.add_decoder is True

    def test_inherits_step35_text_decoder_provider(self):
        provider = _provider()

        assert isinstance(provider, Step35ModelProvider)
        assert isinstance(provider, GPTModelProvider)


class TestStep37Provide:
    @patch("megatron.bridge.models.stepfun.step37_provider.Step37Model")
    @patch("megatron.bridge.models.stepfun.step37_provider.get_step37_text_layer_spec")
    def test_provide_falls_back_to_step37_layer_spec_helper(self, mock_spec_helper, mock_model_cls):
        provider = _provider(transformer_layer_spec=None)

        result = provider.provide(pre_process=True, post_process=False, vp_stage=None)

        mock_spec_helper.assert_called_once_with(provider, vp_stage=None)
        assert result is mock_model_cls.return_value
        _, kwargs = mock_model_cls.call_args
        assert kwargs["language_transformer_layer_spec"] is mock_spec_helper.return_value
        assert kwargs["language_transformer_config"] is provider
        assert kwargs["pre_process"] is True
        assert kwargs["post_process"] is False

    @patch("megatron.bridge.models.stepfun.step37_provider.Step37Model")
    def test_provide_honours_callable_layer_spec_override(self, mock_model_cls):
        spec_override = Mock(name="layer_spec_callable")
        provider = _provider(transformer_layer_spec=spec_override)

        provider.provide(vp_stage=1)

        spec_override.assert_called_once_with(provider, vp_stage=1)
        _, kwargs = mock_model_cls.call_args
        assert kwargs["language_transformer_layer_spec"] is spec_override.return_value

    @patch("megatron.bridge.models.stepfun.step37_provider.Step37Model")
    def test_provide_passes_vision_and_gating_fields(self, mock_model_cls):
        vision_config = SimpleNamespace(hidden_size=32)
        provider = _provider(
            transformer_layer_spec=Mock(),
            vision_config=vision_config,
            add_encoder=False,
            add_decoder=True,
        )

        provider.provide()

        _, kwargs = mock_model_cls.call_args
        assert kwargs["vision_transformer_config"] is vision_config
        assert kwargs["add_encoder"] is False
        assert kwargs["add_decoder"] is True

    @patch("megatron.bridge.models.stepfun.step37_provider.Step37Model")
    def test_provide_does_not_freeze_by_default(self, mock_model_cls):
        provider = _provider(transformer_layer_spec=Mock())

        model = provider.provide()

        model.freeze.assert_not_called()

    @pytest.mark.parametrize(
        "freeze_field",
        ["freeze_language_model", "freeze_vision_model", "freeze_vision_projection"],
    )
    @patch("megatron.bridge.models.stepfun.step37_provider.Step37Model")
    def test_provide_freezes_when_any_flag_set(self, mock_model_cls, freeze_field):
        provider = _provider(transformer_layer_spec=Mock(), **{freeze_field: True})

        model = provider.provide()

        model.freeze.assert_called_once_with(
            freeze_language_model=provider.freeze_language_model,
            freeze_vision_model=provider.freeze_vision_model,
            freeze_vision_projection=provider.freeze_vision_projection,
        )

    @patch.object(GPTModelProvider, "provide")
    def test_provide_language_model_delegates_to_gpt_provider(self, mock_gpt_provide):
        provider = _provider()

        result = provider.provide_language_model(pre_process=True, post_process=True, vp_stage=2)

        mock_gpt_provide.assert_called_once_with(provider, pre_process=True, post_process=True, vp_stage=2)
        assert result is mock_gpt_provide.return_value
