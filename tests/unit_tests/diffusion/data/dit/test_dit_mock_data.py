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

from megatron.bridge.diffusion.data.dit.dit_mock_datamodule import mock_batch


def test_mock_batch():
    """Unit test for mock_batch function."""
    # Test parameters
    F_latents = 2
    H_latents = 32
    W_latents = 48
    patch_temporal = 1
    patch_spatial = 2
    number_packed_samples = 3
    context_seq_len = 77
    context_embeddings_dim = 768

    # Generate mock batch
    batch = mock_batch(
        F_latents=F_latents,
        H_latents=H_latents,
        W_latents=W_latents,
        patch_temporal=patch_temporal,
        patch_spatial=patch_spatial,
        number_packed_samples=number_packed_samples,
        context_seq_len=context_seq_len,
        context_embeddings_dim=context_embeddings_dim,
    )

    # Calculate expected dimensions
    C = 16  # channels (hardcoded in mock_batch)
    T = F_latents
    H = H_latents
    W = W_latents
    seq_len_q = (T // patch_temporal) * (H // patch_spatial) * (W // patch_spatial)
    total_seq_len_q = seq_len_q * number_packed_samples
    total_seq_len_kv = context_seq_len * number_packed_samples

    # Verify batch structure and shapes
    assert "video" in batch
    assert "context_embeddings" in batch
    assert "context_mask" in batch
    assert "loss_mask" in batch
    assert "seq_len_q" in batch
    assert "seq_len_q_padded" in batch
    assert "seq_len_kv" in batch
    assert "seq_len_kv_padded" in batch
    assert "latent_shape" in batch
    assert "pos_ids" in batch
    assert "video_metadata" in batch

    # Check video shape: [1, total_seq_len_q, patch_features]
    patch_features = patch_spatial * patch_spatial * patch_temporal * C
    assert batch["video"].shape == (1, total_seq_len_q, patch_features)
    assert batch["video"].dtype == torch.bfloat16

    # Check context embeddings shape: [1, total_seq_len_kv, context_embeddings_dim]
    assert batch["context_embeddings"].shape == (1, total_seq_len_kv, context_embeddings_dim)
    assert batch["context_embeddings"].dtype == torch.bfloat16

    # Check context mask shape: [1, total_seq_len_kv]
    assert batch["context_mask"].shape == (1, total_seq_len_kv)
    assert batch["context_mask"].dtype == torch.bfloat16

    # Check loss mask shape: [1, total_seq_len_q]
    assert batch["loss_mask"].shape == (1, total_seq_len_q)
    assert batch["loss_mask"].dtype == torch.bfloat16

    # Check sequence length tensors
    assert batch["seq_len_q"].shape == (number_packed_samples,)
    assert batch["seq_len_q_padded"].shape == (number_packed_samples,)
    assert batch["seq_len_kv"].shape == (number_packed_samples,)
    assert batch["seq_len_kv_padded"].shape == (number_packed_samples,)

    # Check all seq_len_q values are correct
    assert torch.all(batch["seq_len_q"] == seq_len_q)
    assert torch.all(batch["seq_len_q_padded"] == seq_len_q)
    assert torch.all(batch["seq_len_kv"] == context_seq_len)
    assert torch.all(batch["seq_len_kv_padded"] == context_seq_len)

    # Check latent shape tensor
    assert batch["latent_shape"].shape == (number_packed_samples, 4)
    expected_latent_shape = torch.tensor([C, T, H, W], dtype=torch.int32)
    for i in range(number_packed_samples):
        assert torch.all(batch["latent_shape"][i] == expected_latent_shape)

    # Check pos_ids shape
    assert batch["pos_ids"].shape == (1, total_seq_len_q, 3)  # 3D position encoding

    # Check video metadata
    assert len(batch["video_metadata"]) == number_packed_samples
    for i, metadata in enumerate(batch["video_metadata"]):
        assert "caption" in metadata
        assert metadata["caption"] == f"Mock video sample {i}"

    print("All tests passed!")
