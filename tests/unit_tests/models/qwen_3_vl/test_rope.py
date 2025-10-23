
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
# See the License for the specific l

"""
Run with: pytest tests/unit_tests/models/qwen_3_vl/test_rope.py
"""

import pytest
import torch
from megatron.bridge.models.qwen_3_vl.rope import Qwen3VLTextRotaryEmbedding
from transformers import Qwen3VLMoeTextConfig
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeTextRotaryEmbedding


@pytest.fixture(scope="module")
def hf_config():
    """Load HuggingFace config once for all tests."""
    return Qwen3VLMoeTextConfig.from_pretrained("Qwen/Qwen3-VL-30B-A3B-Instruct")


class TestQwen3VLTextRotaryEmbedding:
    """Test suite for Qwen3VL Text Rotary Embedding."""
    
    def test_qwen3_vl_text_rotary_embedding(self, hf_config):
        """Test that MBridge RoPE output matches HuggingFace implementation."""
        hf_rope_embedding = Qwen3VLMoeTextRotaryEmbedding(hf_config)
        mbridge_rope_embedding = Qwen3VLTextRotaryEmbedding(hf_config)

        seq_len = 1024
        batch_size = 1
        rand_hidden_states = torch.randn(batch_size, seq_len, hf_config.hidden_size)

        position_ids_2d = torch.arange(seq_len).unsqueeze(0)  # shape: (1, 1024)
        position_ids_3d = position_ids_2d[None, ...].expand(3, batch_size, -1)  # shape: (3, 1, 1024)

        mrope_section = [24, 20, 20]
        
        # Get HF outputs: (bs, seq_len, head_dim*2) for both cos and sin
        hf_cos, hf_sin = hf_rope_embedding(rand_hidden_states, position_ids_3d)

        # Get MBridge output: (seq_len, bs, 1, head_dim/2) raw freqs with attention_scaling applied
        mbridge_rope_output = mbridge_rope_embedding(position_ids_3d, mrope_section)

        # MBridge returns raw freqs with attention_scaling already applied
        # Megatron Core will compute cos/sin internally, but for testing we compute them here
        mbridge_cos = mbridge_rope_output.cos().squeeze(2)  # (seq_len, bs, head_dim/2)
        mbridge_sin = mbridge_rope_output.sin().squeeze(2)  # (seq_len, bs, head_dim/2)
        
        # Transpose MBridge to match HF shape: (seq_len, bs, head_dim/2) -> (bs, seq_len, head_dim/2)
        mbridge_cos = mbridge_cos.transpose(0, 1)
        mbridge_sin = mbridge_sin.transpose(0, 1)

        # HF doubles the dimension with torch.cat, but we only have half
        # So we need to compare the actual frequency values (not doubled)
        # HF applies the pattern: [freq_0, freq_1, ..., freq_n, freq_0, freq_1, ..., freq_n]
        # MBridge just has: [freq_0, freq_1, ..., freq_n]
        # Let's compare the first half of HF with MBridge
        head_dim_half = hf_cos.shape[-1] // 2
        torch.testing.assert_close(hf_cos[..., :head_dim_half], mbridge_cos, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(hf_sin[..., :head_dim_half], mbridge_sin, rtol=1e-4, atol=1e-4)