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
Run with: pytest tests/unit_tests/models/qwen_vl/modelling_qwen3_vl/test_rope.py
"""

import pytest
import torch
from transformers import Qwen3VLMoeTextConfig
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeTextRotaryEmbedding

from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.rope import Qwen3VLMultimodalRotaryEmbedding


@pytest.fixture(scope="module")
def hf_config():
    """Load HuggingFace config once for all tests."""
    return Qwen3VLMoeTextConfig.from_pretrained("Qwen/Qwen3-VL-30B-A3B-Instruct")


class TestQwen3VLTextRotaryEmbedding:
    """Test suite for Qwen3VL Text Rotary Embedding."""

    def test_qwen3_vl_text_rotary_embedding(self, hf_config):
        """Test that MBridge RoPE output matches HuggingFace implementation."""
        hf_rope_embedding = Qwen3VLMoeTextRotaryEmbedding(hf_config)
        mbridge_rope_embedding = Qwen3VLMultimodalRotaryEmbedding(
            kv_channels=hf_config.head_dim,
            rotary_base=hf_config.rope_theta,
        )

        seq_len = 1024
        batch_size = 1
        rand_hidden_states = torch.randn(batch_size, seq_len, hf_config.hidden_size)

        position_ids_2d = torch.arange(seq_len).unsqueeze(0)  # shape: (1, 1024)
        position_ids_3d = position_ids_2d[None, ...].expand(3, batch_size, -1)  # shape: (3, 1, 1024)

        mrope_section = [24, 20, 20]

        # Get HF outputs: (bs, seq_len, head_dim) for both cos and sin
        hf_cos, hf_sin = hf_rope_embedding(rand_hidden_states, position_ids_3d)

        # Get MBridge output: (seq_len, bs, 1, head_dim) raw freqs with concatenation applied
        # The implementation now concatenates freqs: emb = torch.cat((freqs, freqs), dim=-1)
        mbridge_rope_output = mbridge_rope_embedding(position_ids_3d, mrope_section)

        if torch.cuda.is_available():
            hf_cos = hf_cos.to(torch.cuda.current_device())
            hf_sin = hf_sin.to(torch.cuda.current_device())
            mbridge_rope_output = mbridge_rope_output.to(torch.cuda.current_device())

        # MBridge returns concatenated freqs with attention_scaling already applied
        # Megatron Core will compute cos/sin internally, but for testing we compute them here
        mbridge_cos = mbridge_rope_output.cos().squeeze(2)  # (seq_len, bs, head_dim)
        mbridge_sin = mbridge_rope_output.sin().squeeze(2)  # (seq_len, bs, head_dim)

        # Transpose MBridge to match HF shape: (seq_len, bs, head_dim) -> (bs, seq_len, head_dim)
        mbridge_cos = mbridge_cos.transpose(0, 1)
        mbridge_sin = mbridge_sin.transpose(0, 1)

        # Both HF and MBridge now have the same shape with concatenated freqs
        # HF applies the pattern: [freq_0, freq_1, ..., freq_n, freq_0, freq_1, ..., freq_n]
        # MBridge also applies: torch.cat((freqs, freqs), dim=-1)
        # So they should match exactly
        torch.testing.assert_close(hf_cos, mbridge_cos, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(hf_sin, mbridge_sin, rtol=1e-4, atol=1e-4)

    def test_qwen3_vl_text_rotary_embedding_2d_position_ids(self, hf_config):
        """Test Qwen3VLMultimodalRotaryEmbedding with 2D position_ids (should auto-expand to 3D)."""
        mbridge_rope_embedding = Qwen3VLMultimodalRotaryEmbedding(
            kv_channels=hf_config.head_dim,
            rotary_base=hf_config.rope_theta,
        )

        seq_len = 512
        batch_size = 2

        # Test with 2D position_ids (should be expanded to 3D internally)
        position_ids_2d = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)  # shape: (bs, seq_len)

        mrope_section = [24, 20, 20]

        # Get MBridge output with 2D position_ids
        mbridge_rope_output = mbridge_rope_embedding(position_ids_2d, mrope_section)

        # Verify output shape: (seq_len, bs, 1, head_dim)
        assert mbridge_rope_output.shape[0] == seq_len
        assert mbridge_rope_output.shape[1] == batch_size
        assert mbridge_rope_output.shape[2] == 1

    def test_qwen3_vl_text_rotary_embedding_default_mrope_section(self, hf_config):
        """Test Qwen3VLMultimodalRotaryEmbedding with None mrope_section (should use default)."""
        mbridge_rope_embedding = Qwen3VLMultimodalRotaryEmbedding(
            kv_channels=hf_config.head_dim,
            rotary_base=hf_config.rope_theta,
        )

        seq_len = 256
        batch_size = 1

        position_ids_3d = torch.arange(seq_len).unsqueeze(0).unsqueeze(0).expand(3, batch_size, -1)

        # Test with mrope_section=None (should use self.mrope_section)
        mbridge_rope_output = mbridge_rope_embedding(position_ids_3d, mrope_section=None)

        # Verify output shape
        assert mbridge_rope_output.shape[0] == seq_len
        assert mbridge_rope_output.shape[1] == batch_size

    def test_qwen3_vl_moe_text_rotary_embedding(self, hf_config):
        """Test Qwen3VLMultimodalRotaryEmbedding forward pass."""
        mbridge_rope_embedding = Qwen3VLMultimodalRotaryEmbedding(
            kv_channels=hf_config.head_dim,
            rotary_base=hf_config.rope_theta,
        )

        seq_len = 512
        batch_size = 2

        position_ids_3d = torch.arange(seq_len).unsqueeze(0).unsqueeze(0).expand(3, batch_size, -1)
        mrope_section = [24, 20, 20]

        # Forward pass through MoE RoPE
        output = mbridge_rope_embedding(position_ids_3d, mrope_section)

        # Verify output shape: (seq_len, bs, 1, head_dim)
        assert output.shape[0] == seq_len
        assert output.shape[1] == batch_size
        assert output.shape[2] == 1
        assert output.dtype == torch.float32 or output.dtype == torch.bfloat16 or output.dtype == torch.float16
