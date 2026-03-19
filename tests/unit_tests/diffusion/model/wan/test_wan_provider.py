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

import torch
import torch.nn as nn
from megatron.core import parallel_state

import megatron.bridge.diffusion.models.wan.wan_model as wan_model_module
from megatron.bridge.diffusion.models.wan.wan_model import WanModel
from megatron.bridge.diffusion.models.wan.wan_provider import WanModelProvider


def test_wan_model_provider_provide_returns_model(monkeypatch):
    # Force pipeline stage booleans to avoid dependency on initialized model parallel
    monkeypatch.setattr(parallel_state, "is_pipeline_first_stage", lambda: True, raising=False)
    monkeypatch.setattr(parallel_state, "is_pipeline_last_stage", lambda: True, raising=False)
    # Avoid querying uninitialized PP groups
    monkeypatch.setattr(parallel_state, "get_pipeline_model_parallel_world_size", lambda: 1, raising=False)

    # Bypass Megatron's ProcessGroupCollection usage inside TransformerBlock during construction.
    # CI does not initialize distributed groups; a dummy block suffices for construction checks.
    class DummyTransformerBlock(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.input_tensor = None

        def set_input_tensor(self, input_tensor):
            self.input_tensor = input_tensor

        def forward(self, hidden_states, **kwargs):
            return hidden_states

    monkeypatch.setattr(wan_model_module, "TransformerBlock", DummyTransformerBlock, raising=False)

    provider = WanModelProvider(
        num_layers=2,  # keep small
        hidden_size=64,
        ffn_hidden_size=128,
        num_attention_heads=4,
        layernorm_epsilon=1e-6,
        normalization="RMSNorm",
        layernorm_zero_centered_gamma=False,
        layernorm_across_heads=True,
        add_qkv_bias=True,
        rotary_interleaved=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        fp16_lm_cross_entropy=False,
        parallel_output=True,
        bf16=False,
        params_dtype=torch.float32,
        qkv_format="sbhd",
        seq_length=128,
        share_embeddings_and_output_weights=False,
        vocab_size=32000,
        make_vocab_size_divisible_by=128,
        in_channels=4,
        out_channels=4,
        patch_spatial=2,
        patch_temporal=1,
        freq_dim=16,
        text_len=32,
        text_dim=64,
    )
    # Ensure config supplies fields expected by core attention
    provider.kv_channels = provider.hidden_size // provider.num_attention_heads
    provider.num_query_groups = provider.num_attention_heads
    model = provider.provide()
    assert isinstance(model, WanModel)
    # Sanity check key config properties were plumbed
    assert model.config.hidden_size == 64
    assert model.config.num_attention_heads == 4
    assert model.config.text_dim == 64
