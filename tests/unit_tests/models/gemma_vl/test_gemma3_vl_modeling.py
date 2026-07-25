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

from unittest.mock import Mock

import torch

from megatron.bridge.models.gemma_vl.modeling_gemma3_vl import Gemma3VLModel


def test_attention_mask_is_available_on_later_pipeline_stages():
    """Later PP stages must receive the same image-bidirectional mask."""
    model = object.__new__(Gemma3VLModel)
    model.config = Mock(image_token_id=99)
    model.pre_process = False
    input_ids = torch.tensor([[1, 99, 99, 2]])

    mask = model._compute_attention_mask(input_ids)

    assert mask is not None
    assert mask[0, 0, 1, 2].item() is False
    assert mask[0, 0, 0, 3].item() is True


def test_attention_mask_requires_input_ids():
    """Mask construction is skipped only when token IDs are unavailable."""
    model = object.__new__(Gemma3VLModel)

    assert model._compute_attention_mask(None) is None
