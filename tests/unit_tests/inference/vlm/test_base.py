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

from unittest.mock import MagicMock, patch

import pytest

from megatron.bridge.inference.vlm.base import generate, setup_inference_wrapper
from megatron.bridge.inference.vlm.qwenvl_inference_wrapper import QwenVLInferenceWrapper
from megatron.bridge.models.qwen_vl import Qwen3VLModelProvider, Qwen25VLModelProvider


class TestSetupInferenceWrapper:
    """Tests for setup_inference_wrapper function."""

    @patch("megatron.bridge.inference.vlm.base.QwenVLInferenceWrapper")
    def test_setup_inference_wrapper_qwen25(self, mock_wrapper_cls, mock_tokenizer):
        mock_model = MagicMock()
        mock_model.config = MagicMock(spec=Qwen25VLModelProvider)
        mock_model.config.hidden_size = 1024

        _wrapper = setup_inference_wrapper(mock_model, mock_tokenizer)

        mock_wrapper_cls.assert_called_once()
        # Check InferenceWrapperConfig was created with correct hidden_size
        # Args are positional: (model, InferenceWrapperConfig)
        call_args = mock_wrapper_cls.call_args
        inference_config = call_args[0][1]  # Second positional argument
        assert inference_config.hidden_size == 1024

    @patch("megatron.bridge.inference.vlm.base.QwenVLInferenceWrapper")
    def test_setup_inference_wrapper_qwen3(self, mock_wrapper_cls, mock_tokenizer):
        mock_model = MagicMock()
        mock_model.config = MagicMock(spec=Qwen3VLModelProvider)
        mock_model.config.language_transformer_config = MagicMock()
        mock_model.config.language_transformer_config.hidden_size = 2048

        _wrapper = setup_inference_wrapper(mock_model, mock_tokenizer)

        mock_wrapper_cls.assert_called_once()
        # Check InferenceWrapperConfig was created with correct hidden_size
        # Args are positional: (model, InferenceWrapperConfig)
        call_args = mock_wrapper_cls.call_args
        inference_config = call_args[0][1]  # Second positional argument
        assert inference_config.hidden_size == 2048

    def test_setup_inference_wrapper_invalid(self, mock_tokenizer):
        mock_model = MagicMock()
        mock_model.config = MagicMock()  # Not Qwen config

        with pytest.raises(ValueError):
            setup_inference_wrapper(mock_model, mock_tokenizer)


class TestGenerate:
    """Tests for generate function."""

    @patch("megatron.bridge.inference.vlm.base.VLMEngine")
    @patch("megatron.bridge.inference.vlm.base.QwenVLTextGenerationController")
    def test_generate_qwen(self, mock_qwen_controller, mock_engine, mock_tokenizer, mock_image_processor):
        mock_wrapper = MagicMock(spec=QwenVLInferenceWrapper)

        generate(
            wrapped_model=mock_wrapper,
            tokenizer=mock_tokenizer,
            image_processor=mock_image_processor,
            prompts=["test"],
            images=["image"],
            processor="processor",
        )

        mock_qwen_controller.assert_called()
        mock_engine.assert_called()
        mock_engine.return_value.generate.assert_called()

    @patch("megatron.bridge.inference.vlm.base.VLMEngine")
    @patch("megatron.bridge.inference.vlm.base.VLMTextGenerationController")
    def test_generate_vlm(self, mock_vlm_controller, mock_engine, mock_tokenizer, mock_image_processor):
        mock_wrapper = MagicMock()  # Not QwenVLInferenceWrapper

        generate(
            wrapped_model=mock_wrapper,
            tokenizer=mock_tokenizer,
            image_processor=mock_image_processor,
            prompts=["test"],
            images=["image"],
        )

        mock_vlm_controller.assert_called()
        mock_engine.assert_called()
        mock_engine.return_value.generate.assert_called()

    @patch("megatron.bridge.inference.vlm.base.VLMEngine")
    @patch("megatron.bridge.inference.vlm.base.QwenVLTextGenerationController")
    def test_generate_with_inference_params(self, mock_qwen_controller, mock_engine, mock_tokenizer, mock_image_processor):
        from megatron.core.inference.common_inference_params import CommonInferenceParams

        mock_wrapper = MagicMock(spec=QwenVLInferenceWrapper)
        inference_params = CommonInferenceParams(num_tokens_to_generate=100)

        generate(
            wrapped_model=mock_wrapper,
            tokenizer=mock_tokenizer,
            image_processor=mock_image_processor,
            prompts=["test"],
            images=["image"],
            processor="processor",
            inference_params=inference_params,
        )

        # Verify generate was called with the provided inference params
        mock_engine.return_value.generate.assert_called()
        call_args = mock_engine.return_value.generate.call_args
        assert call_args[1]["common_inference_params"] == inference_params
