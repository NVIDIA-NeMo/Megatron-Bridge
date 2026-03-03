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
import torch

from megatron.bridge.inference.vlm.base import generate, setup_inference_wrapper, setup_model_and_tokenizer
from megatron.bridge.inference.vlm.qwenvl_inference_wrapper import QwenVLInferenceWrapper
from megatron.bridge.models.qwen_vl import Qwen3VLModelProvider, Qwen25VLModelProvider


class TestSetupModelAndTokenizer:
    """Tests for setup_model_and_tokenizer function."""

    @patch("megatron.bridge.inference.vlm.base.setup_inference_wrapper")
    @patch("megatron.bridge.inference.vlm.base.AutoProcessor")
    @patch("megatron.bridge.inference.vlm.base.AutoBridge")
    @patch("megatron.bridge.inference.vlm.base.get_hf_model_id_from_checkpoint")
    @patch("megatron.bridge.inference.vlm.base.print_rank_0")
    def test_setup_model_and_tokenizer_basic(
        self,
        mock_print_rank_0,
        mock_get_hf_model_id,
        mock_auto_bridge,
        mock_auto_processor,
        mock_setup_inference_wrapper,
    ):
        """Test basic setup_model_and_tokenizer flow."""
        # Setup mocks
        mock_get_hf_model_id.return_value = "Qwen/Qwen2.5-VL-3B"

        mock_bridge = MagicMock()
        mock_auto_bridge.from_hf_pretrained.return_value = mock_bridge

        mock_model_provider = MagicMock()
        mock_bridge.to_megatron_provider.return_value = mock_model_provider

        # Create mock model that will be returned by load_megatron_model
        mock_model = MagicMock()
        mock_model.config = MagicMock(spec=Qwen25VLModelProvider)
        # Make cuda() return the same mock so eval() is called on it
        mock_model.cuda.return_value = mock_model
        mock_bridge.load_megatron_model.return_value = [mock_model]

        # Setup processor mock
        mock_processor = MagicMock()
        mock_processor.tokenizer.pad_token = None
        mock_processor.tokenizer.eos_token = "<|endoftext|>"
        mock_auto_processor.from_pretrained.return_value = mock_processor

        # Setup inference wrapper mock
        mock_wrapped_model = MagicMock()
        mock_setup_inference_wrapper.return_value = mock_wrapped_model

        # Call the function
        result_model, result_processor = setup_model_and_tokenizer(
            megatron_model_path="/path/to/checkpoint",
            tp=2,
            pp=1,
        )

        # Assertions
        mock_print_rank_0.assert_called()
        mock_get_hf_model_id.assert_called_once_with("/path/to/checkpoint")
        mock_auto_bridge.from_hf_pretrained.assert_called_once_with("Qwen/Qwen2.5-VL-3B")
        mock_bridge.to_megatron_provider.assert_called_once_with(load_weights=False)

        # Verify model provider configuration
        assert mock_model_provider.tensor_model_parallel_size == 2
        assert mock_model_provider.pipeline_model_parallel_size == 1
        assert mock_model_provider.pipeline_dtype == torch.bfloat16
        assert mock_model_provider.parallel_output is False
        mock_model_provider.finalize.assert_called_once()
        mock_model_provider.initialize_model_parallel.assert_called_once_with(seed=0)

        # Verify load_megatron_model was called correctly
        mock_bridge.load_megatron_model.assert_called_once()
        call_args = mock_bridge.load_megatron_model.call_args
        assert call_args[0][0] == "/path/to/checkpoint"
        assert call_args[1]["mp_overrides"]["tensor_model_parallel_size"] == 2
        assert call_args[1]["mp_overrides"]["pipeline_model_parallel_size"] == 1
        assert call_args[1]["wrap_with_ddp"] is False

        # Verify model was set to eval mode
        mock_model.eval.assert_called_once()

        # Verify pad_token was set
        assert mock_processor.tokenizer.pad_token == "<|endoftext|>"

        # Verify setup_inference_wrapper was called
        mock_setup_inference_wrapper.assert_called_once()

        # Verify return values
        assert result_model == mock_wrapped_model
        assert result_processor == mock_processor

    @patch("megatron.bridge.inference.vlm.base.setup_inference_wrapper")
    @patch("megatron.bridge.inference.vlm.base.AutoProcessor")
    @patch("megatron.bridge.inference.vlm.base.AutoBridge")
    @patch("megatron.bridge.inference.vlm.base.get_hf_model_id_from_checkpoint")
    @patch("megatron.bridge.inference.vlm.base.print_rank_0")
    def test_setup_model_and_tokenizer_with_existing_pad_token(
        self,
        mock_print_rank_0,
        mock_get_hf_model_id,
        mock_auto_bridge,
        mock_auto_processor,
        mock_setup_inference_wrapper,
    ):
        """Test that pad_token is not overwritten if already set."""
        mock_get_hf_model_id.return_value = "Qwen/Qwen2.5-VL-3B"

        mock_bridge = MagicMock()
        mock_auto_bridge.from_hf_pretrained.return_value = mock_bridge
        mock_bridge.to_megatron_provider.return_value = MagicMock()

        mock_model = MagicMock()
        mock_model.config = MagicMock(spec=Qwen25VLModelProvider)
        mock_model.cuda.return_value = mock_model
        mock_bridge.load_megatron_model.return_value = [mock_model]

        # Setup processor with existing pad_token
        mock_processor = MagicMock()
        mock_processor.tokenizer.pad_token = "<|pad|>"
        mock_processor.tokenizer.eos_token = "<|endoftext|>"
        mock_auto_processor.from_pretrained.return_value = mock_processor

        mock_setup_inference_wrapper.return_value = MagicMock()

        setup_model_and_tokenizer(megatron_model_path="/path/to/checkpoint")

        # Verify pad_token was NOT changed
        assert mock_processor.tokenizer.pad_token == "<|pad|>"

    @patch("megatron.bridge.inference.vlm.base.setup_inference_wrapper")
    @patch("megatron.bridge.inference.vlm.base.AutoProcessor")
    @patch("megatron.bridge.inference.vlm.base.AutoBridge")
    @patch("megatron.bridge.inference.vlm.base.get_hf_model_id_from_checkpoint")
    @patch("megatron.bridge.inference.vlm.base.print_rank_0")
    def test_setup_model_and_tokenizer_grad_scale_func_set_to_none(
        self,
        mock_print_rank_0,
        mock_get_hf_model_id,
        mock_auto_bridge,
        mock_auto_processor,
        mock_setup_inference_wrapper,
    ):
        """Test that grad_scale_func is set to None for inference."""
        mock_get_hf_model_id.return_value = "Qwen/Qwen2.5-VL-3B"

        mock_bridge = MagicMock()
        mock_auto_bridge.from_hf_pretrained.return_value = mock_bridge
        mock_bridge.to_megatron_provider.return_value = MagicMock()

        # Create mock model with config that has grad_scale_func
        mock_model = MagicMock()
        # Don't use spec here so we can freely set/check grad_scale_func
        mock_model.config = MagicMock()
        mock_model.config.grad_scale_func = MagicMock()  # Initially not None
        # Make cuda() return the same mock so grad_scale_func is set on it
        mock_model.cuda.return_value = mock_model
        mock_bridge.load_megatron_model.return_value = [mock_model]

        mock_processor = MagicMock()
        mock_processor.tokenizer.pad_token = "<|pad|>"
        mock_auto_processor.from_pretrained.return_value = mock_processor

        mock_setup_inference_wrapper.return_value = MagicMock()

        setup_model_and_tokenizer(megatron_model_path="/path/to/checkpoint")

        # Verify grad_scale_func was set to None
        assert mock_model.config.grad_scale_func is None

    @patch("megatron.bridge.inference.vlm.base.setup_inference_wrapper")
    @patch("megatron.bridge.inference.vlm.base.AutoProcessor")
    @patch("megatron.bridge.inference.vlm.base.AutoBridge")
    @patch("megatron.bridge.inference.vlm.base.get_hf_model_id_from_checkpoint")
    @patch("megatron.bridge.inference.vlm.base.print_rank_0")
    def test_setup_model_and_tokenizer_default_params(
        self,
        mock_print_rank_0,
        mock_get_hf_model_id,
        mock_auto_bridge,
        mock_auto_processor,
        mock_setup_inference_wrapper,
    ):
        """Test that default parameters are applied correctly."""
        mock_get_hf_model_id.return_value = "Qwen/Qwen2.5-VL-3B"

        mock_bridge = MagicMock()
        mock_auto_bridge.from_hf_pretrained.return_value = mock_bridge

        mock_model_provider = MagicMock()
        mock_bridge.to_megatron_provider.return_value = mock_model_provider

        mock_model = MagicMock()
        mock_model.config = MagicMock(spec=Qwen25VLModelProvider)
        mock_model.cuda.return_value = mock_model
        mock_bridge.load_megatron_model.return_value = [mock_model]

        mock_processor = MagicMock()
        mock_processor.tokenizer.pad_token = "<|pad|>"
        mock_auto_processor.from_pretrained.return_value = mock_processor

        mock_setup_inference_wrapper.return_value = MagicMock()

        # Call with only required parameter
        setup_model_and_tokenizer(megatron_model_path="/path/to/checkpoint")

        # Verify default values are used
        assert mock_model_provider.tensor_model_parallel_size == 1  # default tp
        assert mock_model_provider.pipeline_model_parallel_size == 1  # default pp

        # Verify load_megatron_model default mp_overrides
        call_args = mock_bridge.load_megatron_model.call_args
        assert call_args[1]["mp_overrides"]["tensor_model_parallel_size"] == 1
        assert call_args[1]["mp_overrides"]["pipeline_model_parallel_size"] == 1


class TestSetupInferenceWrapper:
    """Tests for setup_inference_wrapper function."""

    @patch("megatron.bridge.inference.vlm.base.QwenVLInferenceWrapper")
    def test_setup_inference_wrapper_qwen25(self, mock_wrapper_cls, mock_tokenizer):
        """Test Qwen25 setup with module.language_model.decoder structure."""

        # Create mock objects with nested structure
        class MockObject:
            pass

        mock_decoder = MagicMock()

        # Build the nested structure: model.module.language_model.decoder
        mock_language_model = MockObject()
        mock_language_model.decoder = mock_decoder
        mock_language_model.vocab_size = 151936

        mock_module = MockObject()
        mock_module.language_model = mock_language_model

        mock_model = MockObject()
        mock_model.module = mock_module
        mock_model.config = MagicMock(spec=Qwen25VLModelProvider)
        mock_model.config.hidden_size = 1024
        mock_model.cuda = MagicMock(return_value=mock_model)
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.eval = MagicMock()

        _wrapper = setup_inference_wrapper(mock_model, mock_tokenizer)

        # Verify decoder was exposed at module level
        assert hasattr(mock_module, "decoder")
        assert mock_module.decoder is mock_decoder

        mock_wrapper_cls.assert_called_once()

    @patch("megatron.bridge.inference.vlm.base.QwenVLInferenceWrapper")
    def test_setup_inference_wrapper_qwen3(self, mock_wrapper_cls, mock_tokenizer):
        # Create a simple object without module attribute to avoid infinite loop
        class MockObject:
            pass

        mock_model = MockObject()
        mock_model.config = MagicMock(spec=Qwen3VLModelProvider)
        mock_model.config.language_transformer_config = MagicMock()
        mock_model.config.language_transformer_config.hidden_size = 2048
        mock_model.cuda = MagicMock(return_value=mock_model)
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.eval = MagicMock()

        _wrapper = setup_inference_wrapper(mock_model, mock_tokenizer)

        mock_wrapper_cls.assert_called_once()

    def test_setup_inference_wrapper_invalid(self, mock_tokenizer):
        # Create a simple object without module attribute to avoid infinite loop
        class MockObject:
            pass

        mock_model = MockObject()
        mock_model.config = MagicMock()  # Not Qwen config
        mock_model.cuda = MagicMock(return_value=mock_model)
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.eval = MagicMock()

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
    def test_generate_with_sampling_params(
        self, mock_qwen_controller, mock_engine, mock_tokenizer, mock_image_processor
    ):
        from megatron.core.inference.sampling_params import SamplingParams

        mock_wrapper = MagicMock(spec=QwenVLInferenceWrapper)
        sampling_params = SamplingParams(num_tokens_to_generate=100)

        generate(
            wrapped_model=mock_wrapper,
            tokenizer=mock_tokenizer,
            image_processor=mock_image_processor,
            prompts=["test"],
            images=["image"],
            processor="processor",
            sampling_params=sampling_params,
        )

        # Verify generate was called with the provided inference params
        mock_engine.return_value.generate.assert_called()
        call_args = mock_engine.return_value.generate.call_args
        assert call_args[1]["sampling_params"] == sampling_params
