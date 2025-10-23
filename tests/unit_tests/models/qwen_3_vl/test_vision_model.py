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

"""
Unit tests for Qwen3VL Vision Model implementation.

Run with: pytest tests/unit_tests/models/qwen_3_vl/test_vision_model.py
"""

import pytest
import torch
import numpy as np
from PIL import Image
from unittest.mock import Mock
from transformers import AutoProcessor
from megatron.bridge.models.qwen_3_vl.vision_model import Qwen3VLVisionModel


@pytest.fixture(scope="module")
def processor():
    """Load HuggingFace processor once for all tests."""
    return AutoProcessor.from_pretrained("Qwen/Qwen3-VL-30B-A3B-Instruct")


@pytest.fixture
def mock_transformer_config():
    """Get mock transformer config."""
    config = Mock()
    config.bf16 = True
    return config


@pytest.fixture
def vision_model(mock_transformer_config):
    """Create vision model instance."""
    model = Qwen3VLVisionModel(config=mock_transformer_config)
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")
    return model


@pytest.fixture
def random_image():
    """Generate a random PIL image."""
    # Create random RGB image (224x224 is typical for vision models)
    random_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(random_array)


class TestVisionModel:
    """Test suite for Qwen3VL Vision Model."""

    def test_vision_model_forward(self, vision_model, processor, random_image):
        """Test vision model forward pass with processor-generated inputs."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Use processor with random image (no URL download needed)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": random_image,  # Pass PIL Image directly
                    },
                    {
                        "type": "text",
                        "text": "Describe this image."
                    }
                ]
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        pixel_values = inputs["pixel_values"].to(device)
        image_grid_thw = inputs.get("image_grid_thw", None)
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.to(device)
        
        # Forward pass
        with torch.no_grad():
            hidden_states, deepstack_feature_lists = vision_model(
                hidden_states=pixel_values,
                grid_thw=image_grid_thw
            )
        
        # Verify outputs have expected properties
        assert hidden_states is not None, "hidden_states should not be None"
        assert isinstance(deepstack_feature_lists, list), "deepstack_feature_lists should be a list"
        assert len(deepstack_feature_lists) > 0, "deepstack_feature_lists should not be empty"
        assert hidden_states.ndim >= 2, "hidden_states should have at least 2 dimensions"