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

from megatron.bridge.inference.vlm.qwenvl_inference_wrapper import QwenVLInferenceWrapper


class TestQwenVLInferenceWrapper:
    """Tests for QwenVLInferenceWrapper methods.

    Since QwenVLInferenceWrapper inherits from AbstractModelInferenceWrapper which
    has complex distributed initialization, we mock the parent __init__ and test
    the methods directly.
    """

    @pytest.fixture
    def wrapper(self, mock_model):
        """Create a QwenVLInferenceWrapper with mocked parent initialization."""
        with patch.object(QwenVLInferenceWrapper, "__init__", lambda self, *args, **kwargs: None):
            wrapper = QwenVLInferenceWrapper.__new__(QwenVLInferenceWrapper)
            wrapper.model = mock_model
            wrapper.inference_params = None
            return wrapper

    def test_prep_inference_input(self, wrapper):
        prompts_tokens = torch.tensor([[1, 2, 3]])
        pixel_values = torch.randn(1, 3, 224, 224)
        image_grid_thw = torch.tensor([1, 1, 1])
        image_dict = [{"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}]

        with (
            patch("torch.Tensor.cuda", side_effect=lambda non_blocking=False: pixel_values),
            patch.object(image_grid_thw, "cuda", return_value=image_grid_thw),
        ):
            result = wrapper.prep_inference_input(prompts_tokens, image_dict)

        assert "input_ids" in result
        assert "pixel_values" in result
        assert "image_grid_thw" in result
        assert result["input_ids"].equal(prompts_tokens)
        # inference_params is now inference_context in newer megatron-core
        # Just verify it was set (not None)
        assert wrapper.inference_params is not None

    def test_prep_inference_input_no_image(self, wrapper):
        prompts_tokens = torch.tensor([[1, 2, 3]])
        image_dict = [None]

        result = wrapper.prep_inference_input(prompts_tokens, image_dict)

        assert "input_ids" in result
        assert result["pixel_values"] is None
        assert result["image_grid_thw"] is None

    def test_get_batch_for_context_window(self, wrapper):
        inference_input = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "pixel_values": MagicMock(),
            "image_grid_thw": MagicMock(),
        }

        result = wrapper.get_batch_for_context_window(inference_input, 0, 2)

        assert result["input_ids"].equal(torch.tensor([[1, 2]]))
        assert result["pixel_values"] == inference_input["pixel_values"]
        assert result["image_grid_thw"] == inference_input["image_grid_thw"]

    def test_forward_pass_without_pipeline_parallel(self, wrapper):
        inference_input = {"input_ids": torch.tensor([[1, 2, 3]])}
        wrapper.model.return_value = torch.tensor([[[0.1, 0.9]]])

        result = wrapper.forward_pass_without_pipeline_parallel(inference_input)

        wrapper.model.assert_called_once()
        assert result.equal(torch.tensor([[[0.1, 0.9]]]))
