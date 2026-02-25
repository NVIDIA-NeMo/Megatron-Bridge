# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
from megatron.core.packed_seq_params import PackedSeqParams

from megatron.bridge.diffusion.models.dit.dit_data_process import (
    encode_seq_length,
    get_batch_on_this_cp_rank,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
class TestEncodeSeqLength:
    """Test suite for encode_seq_length and get_batch_on_this_cp_rank functions."""

    def setup_method(self, method):
        """Set up test fixtures before each test method."""
        # Create basic batch for encode_seq_length tests
        self.basic_batch = {
            "seq_len_q": torch.tensor([10, 20, 15], dtype=torch.int32, device="cuda"),
            "seq_len_kv": torch.tensor([5, 10, 8], dtype=torch.int32, device="cuda"),
            "seq_len_q_padded": torch.tensor([12, 24, 16], dtype=torch.int32, device="cuda"),
            "seq_len_kv_padded": torch.tensor([8, 12, 10], dtype=torch.int32, device="cuda"),
            "video": torch.randn(3, 100, 512, device="cuda"),
        }

        # Create packed sequence parameters for get_batch_on_this_cp_rank tests
        # Note: For cp_size=2, total_tokens must be divisible by (world_size * 2) = 4
        cu_seqlens_q = torch.tensor([0, 10, 30, 45], dtype=torch.int32, device="cuda")
        cu_seqlens_kv = torch.tensor([0, 6, 16, 24], dtype=torch.int32, device="cuda")
        cu_seqlens_q_padded = torch.tensor([0, 12, 36, 52], dtype=torch.int32, device="cuda")
        cu_seqlens_kv_padded = torch.tensor([0, 8, 20, 32], dtype=torch.int32, device="cuda")

        # Create batch with packed parameters for context parallelism tests
        self.batch_with_cp = {
            "video": torch.randn(3, 52, 512, device="cuda"),
            "loss_mask": torch.ones(3, 52, device="cuda"),
            "pos_ids": torch.arange(52, device="cuda").unsqueeze(0).expand(3, -1),
            "context_embeddings": torch.randn(3, 32, 512, device="cuda"),
            "context_mask": torch.ones(3, 32, device="cuda"),
            "packed_seq_params": {
                "self_attention": PackedSeqParams(
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_kv=cu_seqlens_q,
                    cu_seqlens_q_padded=cu_seqlens_q_padded,
                    cu_seqlens_kv_padded=cu_seqlens_q_padded,
                    qkv_format="thd",
                ),
                "cross_attention": PackedSeqParams(
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_kv=cu_seqlens_kv,
                    cu_seqlens_q_padded=cu_seqlens_q_padded,
                    cu_seqlens_kv_padded=cu_seqlens_kv_padded,
                    qkv_format="thd",
                ),
            },
        }

    def test_encode_seq_length_with_seq_lens(self):
        """Test encode_seq_length creates packed_seq_params when seq_len_q and seq_len_kv are present."""
        qkv_format = "thd"
        result = encode_seq_length(self.basic_batch, format=qkv_format)

        # Check that packed_seq_params is created
        assert "packed_seq_params" in result
        assert "self_attention" in result["packed_seq_params"]
        assert "cross_attention" in result["packed_seq_params"]

        # Check self_attention params
        self_attn = result["packed_seq_params"]["self_attention"]
        assert isinstance(self_attn, PackedSeqParams)
        assert self_attn.qkv_format == qkv_format

        # Verify cumulative sum for q (self_attention uses cu_seqlens_q for both q and kv)
        expected_cu_seqlens_q = torch.tensor([0, 10, 30, 45], dtype=torch.int32, device="cuda")
        assert torch.equal(self_attn.cu_seqlens_q, expected_cu_seqlens_q)
        assert torch.equal(self_attn.cu_seqlens_kv, expected_cu_seqlens_q)

        # Verify cumulative sum for q_padded
        expected_cu_seqlens_q_padded = torch.tensor([0, 12, 36, 52], dtype=torch.int32, device="cuda")
        assert torch.equal(self_attn.cu_seqlens_q_padded, expected_cu_seqlens_q_padded)
        assert torch.equal(self_attn.cu_seqlens_kv_padded, expected_cu_seqlens_q_padded)

        # Check cross_attention params
        cross_attn = result["packed_seq_params"]["cross_attention"]
        assert isinstance(cross_attn, PackedSeqParams)
        assert cross_attn.qkv_format == qkv_format

        # Verify cumulative sum for kv (cross_attention uses different kv lengths)
        expected_cu_seqlens_kv = torch.tensor([0, 5, 15, 23], dtype=torch.int32, device="cuda")
        assert torch.equal(cross_attn.cu_seqlens_q, expected_cu_seqlens_q)
        assert torch.equal(cross_attn.cu_seqlens_kv, expected_cu_seqlens_kv)

        # Verify cumulative sum for kv_padded
        expected_cu_seqlens_kv_padded = torch.tensor([0, 8, 20, 30], dtype=torch.int32, device="cuda")
        assert torch.equal(cross_attn.cu_seqlens_q_padded, expected_cu_seqlens_q_padded)
        assert torch.equal(cross_attn.cu_seqlens_kv_padded, expected_cu_seqlens_kv_padded)

    def test_get_batch_on_this_cp_rank(self, monkeypatch):
        """Test that get_batch_on_this_cp_rank returns data unchanged when cp_size=1."""
        # Stub parallel_state functions to avoid requiring initialization
        from megatron.core import parallel_state

        monkeypatch.setattr(parallel_state, "get_context_parallel_world_size", lambda: 1, raising=False)

        # Store original shapes
        original_video_shape = self.batch_with_cp["video"].shape
        original_context_shape = self.batch_with_cp["context_embeddings"].shape

        # Clone the batch to ensure we can compare with original
        original_video = self.batch_with_cp["video"].clone()
        original_context = self.batch_with_cp["context_embeddings"].clone()

        result = get_batch_on_this_cp_rank(self.batch_with_cp)

        # Data should remain unchanged when cp_size=1
        assert result["video"].shape == original_video_shape
        assert result["context_embeddings"].shape == original_context_shape
        assert torch.equal(result["video"], original_video)
        assert torch.equal(result["context_embeddings"], original_context)

    def test_get_batch_on_this_cp_rank_with_context_parallelism(self, monkeypatch):
        """Test that get_batch_on_this_cp_rank partitions data when cp_size>1."""
        # Stub parallel_state functions with cp_size=2, cp_rank=0
        from megatron.core import parallel_state

        monkeypatch.setattr(parallel_state, "get_context_parallel_world_size", lambda: 2, raising=False)
        monkeypatch.setattr(parallel_state, "get_context_parallel_rank", lambda: 0, raising=False)

        result = get_batch_on_this_cp_rank(self.batch_with_cp)

        # Verify that data tensors were modified (should be partitioned)
        # For self-attention keys (video, loss_mask, pos_ids) - 52 tokens / 2 ranks = 26 tokens
        assert result["video"].shape[1] == 26  # Partitioned to half
        assert result["loss_mask"].shape[1] == 26
        assert result["pos_ids"].shape[1] == 26

        # For cross-attention keys (context_embeddings, context_mask) - 32 tokens / 2 ranks = 16 tokens
        assert result["context_embeddings"].shape[1] == 16  # Partitioned to half
        assert result["context_mask"].shape[1] == 16
