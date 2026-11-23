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

"""Regress the global-window bug exposed by the encoder auxiliary objective."""

from types import SimpleNamespace

import pytest
import torch
from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.transformer.enums import AttnMaskType

from megatron.bridge.models.gemma.modeling_gemma4 import Gemma4TEDotProductAttention


pytestmark = pytest.mark.unit


def test_bidirectional_global_layers_do_not_acquire_a_sliding_window(monkeypatch):
    # The constructor's window/mask decisions need no GPU kernels or process groups.
    monkeypatch.setattr(TEDotProductAttention, "__init__", lambda self, **kwargs: torch.nn.Module.__init__(self))
    config = SimpleNamespace(
        window_size=4,
        interleaved_attn_pattern=["sliding_attention", "full_attention"],
        text_config=SimpleNamespace(use_bidirectional_attention="vision"),
        global_image_bidirectional_attention=True,
    )
    local = Gemma4TEDotProductAttention(
        config=config, layer_number=1, attn_mask_type=AttnMaskType.causal, attention_type="self"
    )
    global_layer = Gemma4TEDotProductAttention(
        config=config, layer_number=2, attn_mask_type=AttnMaskType.causal, attention_type="self"
    )
    assert local._image_bidirectional_attention and global_layer._image_bidirectional_attention
    assert local._image_attention_window == 4
    assert global_layer._image_attention_window is None
