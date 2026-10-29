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

from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn as nn

from megatron.bridge.models.gemma_vl.modeling_gemma3_vl import Gemma3VLModel


def test_attention_bias_is_available_on_later_pipeline_stages():
    """Later PP stages must receive the same image-bidirectional bias."""
    model = object.__new__(Gemma3VLModel)
    model.config = Mock(image_token_id=99, window_size=512)
    model.pre_process = False
    input_ids = torch.tensor([[1, 99, 99, 2]])

    biases = model._compute_attention_biases(input_ids, dtype=torch.bfloat16)

    assert biases is not None
    local_bias, global_bias = biases
    assert local_bias.dtype is torch.bfloat16
    assert global_bias.dtype is torch.bfloat16
    assert local_bias[0, 0, 1, 2].item() == 0.0
    assert global_bias[0, 0, 1, 2].item() == 0.0
    assert local_bias[0, 0, 0, 3].item() == torch.finfo(torch.bfloat16).min
    assert global_bias[0, 0, 0, 3].item() == torch.finfo(torch.bfloat16).min


def test_attention_bias_requires_input_ids():
    """Bias construction is skipped only when token IDs are unavailable."""
    model = object.__new__(Gemma3VLModel)

    assert model._compute_attention_biases(None, dtype=torch.bfloat16) is None


def test_attention_bias_keeps_separate_image_blocks_isolated():
    """Only tokens in the same contiguous image block can see each other bidirectionally."""
    model = object.__new__(Gemma3VLModel)
    model.config = Mock(image_token_id=99, window_size=512)
    input_ids = torch.tensor([[1, 99, 99, 2, 99, 99, 3]])

    biases = model._compute_attention_biases(input_ids, dtype=torch.float32)

    assert biases is not None
    local_bias, global_bias = biases
    for bias in (local_bias, global_bias):
        assert bias[0, 0, 1, 2].item() == 0.0
        assert bias[0, 0, 4, 5].item() == 0.0
        assert bias[0, 0, 1, 5].item() == torch.finfo(torch.float32).min
        assert bias[0, 0, 3, 6].item() == torch.finfo(torch.float32).min
        assert torch.count_nonzero(bias[0, 0].tril()).item() == 0


def test_local_attention_bias_includes_sliding_window():
    """Local layers apply the sliding window in bias while global layers do not."""
    model = object.__new__(Gemma3VLModel)
    model.config = Mock(image_token_id=99, window_size=3)
    input_ids = torch.tensor([[1, 99, 99, 2, 3]])

    biases = model._compute_attention_biases(input_ids, dtype=torch.float32)

    assert biases is not None
    local_bias, global_bias = biases
    assert local_bias[0, 0, 4, 0].item() == torch.finfo(torch.float32).min
    assert global_bias[0, 0, 4, 0].item() == 0.0
    assert local_bias[0, 0, 1, 2].item() == 0.0


def test_forward_passes_attention_bias_to_language_model():
    """The language model receives fused-attention bias rather than a custom mask."""
    model = object.__new__(Gemma3VLModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        image_token_id=99,
        window_size=512,
        params_dtype=torch.bfloat16,
        sequence_parallel=False,
        _pg_collection=None,
    )
    model.pre_process = False
    model.language_model = SimpleNamespace(forward=Mock(return_value=torch.tensor(1.0)))
    input_ids = torch.tensor([[1, 99, 99, 2]])

    with patch(
        "megatron.bridge.models.gemma_vl.modeling_gemma3_vl.slice_batch_for_context_parallel",
        return_value=(None, None, None, None, None),
    ):
        model(input_ids=input_ids)

    call_kwargs = model.language_model.forward.call_args.kwargs
    assert call_kwargs["attention_mask"] is None
    local_bias, global_bias = call_kwargs["extra_block_kwargs"]["attention_bias"]
    assert local_bias[0, 0, 1, 2].item() == 0.0
    assert global_bias[0, 0, 1, 2].item() == 0.0
    assert local_bias[0, 0, 0, 3].item() == torch.finfo(torch.bfloat16).min
    assert global_bias[0, 0, 0, 3].item() == torch.finfo(torch.bfloat16).min
