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

import pytest
import torch
from unittest.mock import MagicMock, patch
from megatron.bridge.inference.vlm.vlm_inference_controller import (
    TokenizerWrapper,
    VLMTextGenerationController,
    QwenVLTextGenerationController
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
        with patch.object(VLMTextGenerationController, '__init__', lambda self, *args, **kwargs: None):
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
        mock_image_processor.preprocess.assert_called_with(image, return_tensors='pt')
        assert "pixel_values" in image_dict

    def test_prep_inference_input(self, controller):
        prompts_tokens = torch.tensor([[1, 2, 3]])
        active_requests = {
            1: MagicMock(encoder_prompt="image_data")
        }
        
        controller.prep_inference_input(prompts_tokens, active_requests)
        
        controller.inference_wrapped_model.prep_inference_input.assert_called_with(
            prompts_tokens=prompts_tokens,
            image_dict=["image_data"]
        )


class TestQwenVLTextGenerationController:
    """Tests for QwenVLTextGenerationController."""
    
    @pytest.fixture
    def controller(self, mock_tokenizer, mock_image_processor):
        """Create a QwenVLTextGenerationController with mocked parent initialization."""
        mock_processor = MagicMock()
        mock_processor.return_value = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'pixel_values': 'pixel_values',
            'image_grid_thw': 'image_grid_thw'
        }
        
        with patch.object(QwenVLTextGenerationController, '__init__', lambda self, *args, **kwargs: None):
            controller = QwenVLTextGenerationController.__new__(QwenVLTextGenerationController)
            controller.image_processor = mock_image_processor
            controller.processor = mock_processor
            controller.inference_wrapped_model = MagicMock()
            
            # Set up the QwenVLTokenizer which is a nested class
            class QwenVLTokenizer(TokenizerWrapper):
                def detokenize(self, tokens):
                    new_tokens = []
                    for token in tokens:
                        if token == 151652:
                            new_tokens.append(token)
                            new_tokens.append(151655)
                        else:
                            new_tokens.append(token)
                    return self._tokenizer.decode(new_tokens, skip_special_tokens=False)
            
            controller.tokenizer = QwenVLTokenizer(mock_tokenizer)
            return controller

    def test_tokenize_prompt(self, controller):
        tokens, image_dict = controller.tokenize_prompt("test", "image")
        
        assert tokens == [1, 2, 3]
        assert image_dict['pixel_values'] == 'pixel_values'
        assert image_dict['image_grid_thw'] == 'image_grid_thw'

    def test_tokenize_prompt_no_pixel_values(self, mock_tokenizer, mock_image_processor):
        """Test tokenize_prompt when processor returns no pixel_values."""
        mock_processor = MagicMock()
        mock_processor.return_value = {
            'input_ids': torch.tensor([[1, 2, 3]]),
        }
        
        with patch.object(QwenVLTextGenerationController, '__init__', lambda self, *args, **kwargs: None):
            controller = QwenVLTextGenerationController.__new__(QwenVLTextGenerationController)
            controller.image_processor = mock_image_processor
            controller.processor = mock_processor
            controller.tokenizer = TokenizerWrapper(mock_tokenizer)
            
            tokens, image_dict = controller.tokenize_prompt("test", None)
            
            assert tokens == [1, 2, 3]
            assert image_dict is None

    def test_tokenizer_detokenize(self, controller, mock_tokenizer):
        # Test special token replacement
        tokens = [151652, 1]
        controller.tokenizer.detokenize(tokens)
        
        # 151652 should be followed by 151655
        mock_tokenizer.decode.assert_called()
        call_args = mock_tokenizer.decode.call_args[0][0]
        assert call_args == [151652, 151655, 1]
