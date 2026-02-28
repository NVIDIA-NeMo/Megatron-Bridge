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

from megatron.bridge.diffusion.data.dit.dit_taskencoder import DiTTaskEncoder, PosID3D


class TestDiTTaskEncoder:
    """Test class for DiTTaskEncoder."""

    def test_encode_sample(self, monkeypatch):
        """Test the encode_sample method with valid input."""
        # Mock parallel_state functions
        from unittest.mock import MagicMock

        from megatron.core import parallel_state
        from megatron.energon import WorkerConfig

        monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_world_size", lambda: 1)
        monkeypatch.setattr(parallel_state, "get_context_parallel_world_size", lambda: 1)

        # Mock WorkerConfig for @stateless decorator
        mock_worker_config = MagicMock()
        mock_worker_config.worker_seed.return_value = 42
        monkeypatch.setattr(WorkerConfig, "active_worker_config", mock_worker_config)

        patch_spatial = 2
        patch_temporal = 2
        task_encoder = DiTTaskEncoder(
            seq_length=256,
            patch_spatial=patch_spatial,
            patch_temporal=patch_temporal,
            text_embedding_max_length=384,
        )

        C, T, H, W = 16, 8, 8, 10
        video_latent = torch.randn(1, C, T, H, W, dtype=torch.bfloat16)
        text_embedding_seq_len = 178
        text_hidden_dim = 512
        text_embeddings = torch.randn(text_embedding_seq_len, text_hidden_dim, dtype=torch.bfloat16)

        sample = {
            "pth": video_latent,
            "json": {"resolution": [H, W], "fps": 24, "duration": 1.0},
            "pickle": text_embeddings,
            "__key__": "test_sample_001",
            "__restore_key__": ("test_sample_001",),
            "__subflavors__": ["video"],
        }

        # Call encode_sample
        result = task_encoder.encode_sample(sample)

        # Verify the output structure
        assert hasattr(result, "video"), "Result should have 'video' attribute"
        assert hasattr(result, "context_embeddings"), "Result should have 'context_embeddings' attribute"
        assert hasattr(result, "context_mask"), "Result should have 'context_mask' attribute"
        assert hasattr(result, "loss_mask"), "Result should have 'loss_mask' attribute"
        assert hasattr(result, "seq_len_q"), "Result should have 'seq_len_q' attribute"
        assert hasattr(result, "seq_len_q_padded"), "Result should have 'seq_len_q_padded' attribute"
        assert hasattr(result, "seq_len_kv"), "Result should have 'seq_len_kv' attribute"
        assert hasattr(result, "seq_len_kv_padded"), "Result should have 'seq_len_kv_padded' attribute"
        assert hasattr(result, "pos_ids"), "Result should have 'pos_ids' attribute"
        assert hasattr(result, "latent_shape"), "Result should have 'latent_shape' attribute"

        expected_seq_len = (T // patch_temporal) * (H // patch_spatial) * (W // patch_spatial)
        expected_seq_len_q_padded = 64 * 2
        expected_video_features = patch_spatial * patch_spatial * patch_temporal * C
        expected_seq_len_kv_padded = 64 * 3

        assert result.video.shape[0] == expected_seq_len_q_padded, (
            f"Expected video seq_len {expected_seq_len_q_padded}, got {result.video.shape[0]}"
        )
        assert result.video.shape[1] == expected_video_features, (
            f"Expected video feature dim {expected_video_features}, got {result.video.shape[1]}"
        )

        assert result.context_embeddings.shape[0] == expected_seq_len_kv_padded, (
            f"Expected context_embeddings seq_len {expected_seq_len_kv_padded}, got {result.context_embeddings.shape[0]}"
        )

        # Verify dtypes
        assert result.video.dtype == torch.bfloat16
        assert result.context_embeddings.dtype == torch.bfloat16
        assert result.context_mask.dtype == torch.bfloat16
        assert result.loss_mask.dtype == torch.bfloat16
        assert result.seq_len_q.dtype == torch.int32
        assert result.seq_len_q_padded.dtype == torch.int32
        assert result.seq_len_kv.dtype == torch.int32
        assert result.seq_len_kv_padded.dtype == torch.int32
        assert result.latent_shape.dtype == torch.int32
        assert result.pos_ids.dtype == torch.int64  # TODO: should it be changed to int32?

        # Verify no NaN or Inf values
        assert not torch.isnan(result.video).any(), "Video output contains NaN values"
        assert not torch.isinf(result.video).any(), "Video output contains Inf values"

        assert result.seq_len_q.item() == expected_seq_len, (
            f"Expected seq_len_q {expected_seq_len}, got {result.seq_len_q.item()}"
        )
        print(
            f"result.seq_len_q_padded.item() = {result.seq_len_q_padded.item()}, expected_seq_len_q_padded = {expected_seq_len_q_padded}"
        )
        assert result.seq_len_q_padded.item() == expected_seq_len_q_padded, (
            f"Expected seq_len_q_padded {expected_seq_len_q_padded}, got {result.seq_len_q_padded.item()}"
        )
        assert result.seq_len_kv.item() == text_embedding_seq_len, (
            f"Expected seq_len_kv {text_embedding_seq_len}, got {result.seq_len_kv.item()}"
        )
        assert result.seq_len_kv_padded.item() == expected_seq_len_kv_padded, (
            f"Expected seq_len_kv_padded {expected_seq_len_kv_padded}, got {result.seq_len_kv_padded.item()}"
        )

        # # Verify latent_shape
        assert torch.equal(result.latent_shape, torch.tensor([C, T, H, W], dtype=torch.int32)), (
            "latent_shape does not match original video shape"
        )

        # # Verify pos_ids shape
        assert result.pos_ids.shape[0] == expected_seq_len_q_padded, (
            f"Expected pos_ids seq_len {expected_seq_len_q_padded}, got {result.pos_ids.shape[0]}"
        )
        assert result.pos_ids.shape[1] == 3, (
            f"Expected pos_ids to have 3 dimensions (T, H, W), got {result.pos_ids.shape[1]}"
        )

        # # Verify metadata
        assert result.__key__ == "test_sample_001", "Key mismatch"
        assert result.video_metadata == sample["json"], "Metadata mismatch"

        print("encode_sample test passed successfully with output shapes:")
        print(f"  video: {result.video.shape}")
        print(f"  context_embeddings: {result.context_embeddings.shape}")
        print(f"  pos_ids: {result.pos_ids.shape}")


class TestPosID3D:
    """Test class for PosID3D."""

    def test_get_pos_id_3d_values(self):
        """Test that get_pos_id_3d returns correct position values."""
        pos_id = PosID3D(max_t=8, max_h=16, max_w=16)

        t, h, w = 3, 4, 5
        result = pos_id.get_pos_id_3d(t=t, h=h, w=w)

        # Check shape
        assert result.shape == (t, h, w, 3), f"Expected shape ({t}, {h}, {w}, 3), got {result.shape}"

        # Check dtype
        assert result.dtype == torch.int64 or result.dtype == torch.long, (
            f"Expected dtype torch.int64 or torch.long, got {result.dtype}"
        )

        # Check that values are correct for specific positions
        # First position (0, 0, 0) should be [0, 0, 0]
        assert torch.equal(result[0, 0, 0], torch.tensor([0, 0, 0])), (
            f"Position [0, 0, 0] should be [0, 0, 0], got {result[0, 0, 0]}"
        )

        # Position (1, 2, 3) should be [1, 2, 3]
        assert torch.equal(result[1, 2, 3], torch.tensor([1, 2, 3])), (
            f"Position [1, 2, 3] should be [1, 2, 3], got {result[1, 2, 3]}"
        )

        # Last position (t-1, h-1, w-1) should be [t-1, h-1, w-1]
        assert torch.equal(result[t - 1, h - 1, w - 1], torch.tensor([t - 1, h - 1, w - 1])), (
            f"Position [{t - 1}, {h - 1}, {w - 1}] should be [{t - 1}, {h - 1}, {w - 1}], got {result[t - 1, h - 1, w - 1]}"
        )

        # Verify all temporal positions in first spatial location
        for i in range(t):
            assert result[i, 0, 0, 0] == i, (
                f"Temporal position at [{i}, 0, 0, 0] should be {i}, got {result[i, 0, 0, 0]}"
            )

        # Verify all height positions in first t and w location
        for i in range(h):
            assert result[0, i, 0, 1] == i, (
                f"Height position at [0, {i}, 0, 1] should be {i}, got {result[0, i, 0, 1]}"
            )

        # Verify all width positions in first t and h location
        for i in range(w):
            assert result[0, 0, i, 2] == i, f"Width position at [0, 0, {i}, 2] should be {i}, got {result[0, 0, i, 2]}"
