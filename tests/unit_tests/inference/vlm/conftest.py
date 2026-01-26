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
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import InferenceWrapperConfig


@pytest.fixture(autouse=True)
def mock_parallel_state():
    """Mock megatron parallel state functions to avoid initialization errors."""
    with patch('megatron.core.parallel_state.get_tensor_model_parallel_group', return_value=None), \
         patch('megatron.core.parallel_state.get_pipeline_model_parallel_group', return_value=None), \
         patch('megatron.core.parallel_state.is_pipeline_first_stage', return_value=True), \
         patch('megatron.core.parallel_state.is_pipeline_last_stage', return_value=True):
        yield

@pytest.fixture
def mock_model():
    # Create a mock that is explicitly not iterable
    model = MagicMock(spec=torch.nn.Module) 
    model.config = MagicMock()
    # Explicitly remove __iter__ to be safe, though spec=torch.nn.Module might handle it if Module isn't iterable
    # But MagicMock might still add it.
    del model.__iter__
    return model

@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 100
    tokenizer.vocab_size = 1000
    tokenizer.encode.return_value = [1, 2, 3]
    tokenizer.decode.return_value = "decoded text"
    return tokenizer

@pytest.fixture
def mock_image_processor():
    processor = MagicMock()
    processor.size = {'height': 224, 'width': 224}
    processor.preprocess.return_value = {
        'pixel_values': [torch.randn(1, 3, 224, 224)],
        'aspect_ratio_ids': [torch.tensor([0])],
        'num_tiles': [0]
    }
    return processor

@pytest.fixture
def mock_inference_wrapper_config():
    config = InferenceWrapperConfig(
        hidden_size=1024,
        params_dtype=torch.float32,
        inference_batch_times_seqlen_threshold=1000,
        padded_vocab_size=1000
    )
    # Ensure inference_max_requests is set (it might be a property or default in real class)
    # If it's a property that depends on other things, setting it explicitly on the object might work if it's not a slot class preventing it.
    # But InferenceWrapperConfig is likely a dataclass or simple class.
    # Let's check if we can set it. If not, we might need to mock the config object itself if it has complex logic.
    # For now, assuming we can set it or it has a default.
    # Actually, looking at the error in test_vlm_engine, the config there was a MagicMock from mock_controller.
    # This fixture is used for wrapper tests.
    return config

