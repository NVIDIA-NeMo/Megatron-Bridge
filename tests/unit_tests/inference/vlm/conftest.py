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


@pytest.fixture(autouse=True)
def mock_parallel_state():
    """Mock megatron parallel state functions to avoid initialization errors."""
    with (
        patch("megatron.core.parallel_state.get_tensor_model_parallel_group", return_value=None),
        patch("megatron.core.parallel_state.get_pipeline_model_parallel_group", return_value=None),
        patch("megatron.core.parallel_state.is_pipeline_first_stage", return_value=True),
        patch("megatron.core.parallel_state.is_pipeline_last_stage", return_value=True),
    ):
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
    processor.size = {"height": 224, "width": 224}
    processor.preprocess.return_value = {
        "pixel_values": [torch.randn(1, 3, 224, 224)],
        "aspect_ratio_ids": [torch.tensor([0])],
        "num_tiles": [0],
    }
    return processor
