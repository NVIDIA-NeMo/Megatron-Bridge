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

from megatron.bridge.inference.vlm.vlm_inference_controller import (
    QwenVLTextGenerationController,
    TokenizerWrapper,
    VLMTextGenerationController,
)


class TestTokenizerWrapper:
    def test_init(self, mock_tokenizer):
        wrapper = TokenizerWrapper(mock_tokenizer)
        assert wrapper.eod == 100
        assert wrapper.vocab_size is None
        assert wrapper._tokenizer == mock_tokenizer

    def test_tokenize(self, mock_tokenizer):
        wrapper = TokenizerWrapper(mock_tokenizer)

        wrapper.tokenize("test")

        mock_tokenizer.encode.assert_called_with("test", add_special_tokens=False)

    def test_detokenize(self, mock_tokenizer):
        wrapper = TokenizerWrapper(mock_tokenizer)

        wrapper.detokenize([1, 2, 3])

        mock_tokenizer.decode.assert_called_with([1, 2, 3], skip_special_tokens=False)


class TestVLMTextGenerationController:
    """Tests for VLMTextGenerationController.

    Since the controller inherits from SimpleTextGenerationController which has
    complex initialization, we mock parent __init__ for most tests.
    """

    @pytest.fixture
    def controller(self, mock_tokenizer, mock_image_processor):
        """Create a VLMTextGenerationController with mocked parent initialization."""
        with patch.object(VLMTextGenerationController, "__init__", lambda self, *args, **kwargs: None):
            controller = VLMTextGenerationController.__new__(VLMTextGenerationController)
            controller.tokenizer = TokenizerWrapper(mock_tokenizer)
            controller.image_processor = mock_image_processor
            controller.inference_wrapped_model = MagicMock()
            return controller

    def test_tokenize_prompt_no_image(self, controller, mock_tokenizer, mock_image_processor):
        tokens, image_dict = controller.tokenize_prompt("test", None)

        assert tokens == [1, 2, 3]
        assert "pixel_values" in image_dict
        assert image_dict["pixel_values"].shape == (1, 4, 3, 224, 224)

    def test_tokenize_prompt_with_image(self, controller, mock_tokenizer, mock_image_processor):
        image = MagicMock()

        tokens, image_dict = controller.tokenize_prompt("test", image)

        assert tokens == [1, 2, 3]
        mock_image_processor.preprocess.assert_called_with(image, return_tensors="pt")
        assert "pixel_values" in image_dict

    def test_prep_inference_input(self, controller):
        prompts_tokens = torch.tensor([[1, 2, 3]])
        active_requests = {1: MagicMock(encoder_prompt="image_data")}

        controller.prep_inference_input(prompts_tokens, active_requests)

        controller.inference_wrapped_model.prep_inference_input.assert_called_with(
            prompts_tokens=prompts_tokens, image_dict=["image_data"]
        )


class TestQwenVLTextGenerationController:
    """Tests for QwenVLTextGenerationController."""

    @pytest.fixture
    def controller(self, mock_tokenizer, mock_image_processor):
        """Create a QwenVLTextGenerationController with mocked parent initialization."""
        mock_processor = MagicMock()
        mock_processor.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "pixel_values": "pixel_values",
            "image_grid_thw": "image_grid_thw",
        }

        with patch.object(QwenVLTextGenerationController, "__init__", lambda self, *args, **kwargs: None):
            controller = QwenVLTextGenerationController.__new__(QwenVLTextGenerationController)
            controller.image_processor = mock_image_processor
            controller.processor = mock_processor
            controller.inference_wrapped_model = MagicMock()

            # Set up the QwenVLTokenizer matching the actual implementation
            class QwenVLTokenizer(TokenizerWrapper):
                def detokenize(self, tokens):
                    new_tokens = []
                    for token in tokens:
                        if token == 151652:
                            new_tokens.append(token)
                            new_tokens.append(151655)
                        elif token != 151655:
                            new_tokens.append(token)
                    return self._tokenizer.decode(new_tokens, skip_special_tokens=False)

            controller.tokenizer = QwenVLTokenizer(mock_tokenizer)
            return controller

    def test_tokenize_prompt(self, controller):
        tokens, image_dict = controller.tokenize_prompt("test", "image")

        assert tokens == [1, 2, 3]
        assert image_dict["pixel_values"] == "pixel_values"
        assert image_dict["image_grid_thw"] == "image_grid_thw"

    def test_tokenize_prompt_no_pixel_values(self, mock_tokenizer, mock_image_processor):
        """Test tokenize_prompt when processor returns no pixel_values."""
        mock_processor = MagicMock()
        mock_processor.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
        }

        with patch.object(QwenVLTextGenerationController, "__init__", lambda self, *args, **kwargs: None):
            controller = QwenVLTextGenerationController.__new__(QwenVLTextGenerationController)
            controller.image_processor = mock_image_processor
            controller.processor = mock_processor
            controller.tokenizer = TokenizerWrapper(mock_tokenizer)

            tokens, image_dict = controller.tokenize_prompt("test", None)

            assert tokens == [1, 2, 3]
            assert image_dict is None

    def test_tokenizer_detokenize_with_special_token_151652(self, controller, mock_tokenizer):
        """Test that token 151652 is followed by 151655 during detokenization."""
        tokens = [151652, 1]
        controller.tokenizer.detokenize(tokens)

        # 151652 should be followed by 151655
        mock_tokenizer.decode.assert_called()
        call_args = mock_tokenizer.decode.call_args[0][0]
        assert call_args == [151652, 151655, 1]

    def test_tokenizer_detokenize_filters_out_151655(self, controller, mock_tokenizer):
        """Test that standalone token 151655 is filtered out during detokenization."""
        tokens = [1, 151655, 2, 3]
        controller.tokenizer.detokenize(tokens)

        # 151655 should be filtered out when it appears standalone
        mock_tokenizer.decode.assert_called()
        call_args = mock_tokenizer.decode.call_args[0][0]
        assert call_args == [1, 2, 3]

    def test_tokenizer_detokenize_regular_tokens(self, controller, mock_tokenizer):
        """Test that regular tokens pass through unchanged."""
        tokens = [1, 2, 3, 100, 200]
        controller.tokenizer.detokenize(tokens)

        mock_tokenizer.decode.assert_called()
        call_args = mock_tokenizer.decode.call_args[0][0]
        assert call_args == [1, 2, 3, 100, 200]

    def test_tokenizer_detokenize_complex_sequence(self, controller, mock_tokenizer):
        """Test detokenization with a complex sequence containing multiple special tokens."""
        # Sequence: regular, 151652 (should add 151655), 151655 (should be filtered), regular
        tokens = [10, 151652, 151655, 20]
        controller.tokenizer.detokenize(tokens)

        mock_tokenizer.decode.assert_called()
        call_args = mock_tokenizer.decode.call_args[0][0]
        # 10 -> 10
        # 151652 -> 151652, 151655
        # 151655 -> filtered out
        # 20 -> 20
        assert call_args == [10, 151652, 151655, 20]

    def test_init_creates_qwen_tokenizer(self, mock_tokenizer, mock_image_processor):
        """Test that __init__ properly creates the QwenVLTokenizer with correct behavior.

        This test exercises lines 94-110 of vlm_inference_controller.py by calling the
        actual QwenVLTextGenerationController.__init__ while mocking only the parent class.
        """
        mock_processor = MagicMock()
        mock_inference_model = MagicMock()

        # Patch the direct parent's __init__ to avoid complex initialization chain
        with patch.object(VLMTextGenerationController, "__init__", return_value=None):
            controller = QwenVLTextGenerationController(
                mock_inference_model, mock_tokenizer, mock_image_processor, mock_processor
            )

            # Verify the QwenVLTokenizer was created and assigned
            assert controller.tokenizer is not None
            assert controller.tokenizer._tokenizer == mock_tokenizer
            assert controller.processor == mock_processor

            # Verify the tokenizer is the custom QwenVLTokenizer (not the base TokenizerWrapper)
            # by checking that its detokenize method has the special token handling

            # Test 1: Token 151652 should have 151655 appended
            controller.tokenizer.detokenize([151652])
            call_args = mock_tokenizer.decode.call_args[0][0]
            assert call_args == [151652, 151655], "Token 151652 should be followed by 151655"

            # Test 2: Token 151655 alone should be filtered out
            mock_tokenizer.decode.reset_mock()
            controller.tokenizer.detokenize([151655])
            call_args = mock_tokenizer.decode.call_args[0][0]
            assert call_args == [], "Standalone 151655 should be filtered out"

            # Test 3: Regular tokens should pass through unchanged
            mock_tokenizer.decode.reset_mock()
            controller.tokenizer.detokenize([1, 2, 3])
            call_args = mock_tokenizer.decode.call_args[0][0]
            assert call_args == [1, 2, 3], "Regular tokens should pass through"
