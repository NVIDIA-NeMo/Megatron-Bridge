# Copyright (c) 2025, NVIDIA CORPORATION.
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

from typing import List

import torch

from megatron.bridge.diffusion.data.common.diffusion_sample import DiffusionSample
from megatron.bridge.diffusion.data.common.diffusion_task_encoder_with_sp import (
    DiffusionTaskEncoderWithSequencePacking,
)


class ConcreteDiffusionTaskEncoder(DiffusionTaskEncoderWithSequencePacking):
    """Concrete implementation for testing."""

    def encode_sample(self, sample: dict) -> dict:
        """Simple implementation for testing purposes."""
        return sample

    def batch(self, samples: List[DiffusionSample]) -> dict:
        """Simple batch implementation that returns first sample as dict."""
        if len(samples) == 1:
            sample = samples[0]
            return dict(
                video=sample.video.unsqueeze(0),
                context_embeddings=sample.context_embeddings.unsqueeze(0),
                context_mask=sample.context_mask.unsqueeze(0) if sample.context_mask is not None else None,
                loss_mask=sample.loss_mask.unsqueeze(0) if sample.loss_mask is not None else None,
                seq_len_q=sample.seq_len_q,
                seq_len_q_padded=sample.seq_len_q_padded,
                seq_len_kv=sample.seq_len_kv,
                seq_len_kv_padded=sample.seq_len_kv_padded,
                pos_ids=sample.pos_ids.unsqueeze(0) if sample.pos_ids is not None else None,
                latent_shape=sample.latent_shape,
                video_metadata=sample.video_metadata,
            )
        else:
            # For multiple samples, just return a simple dict
            return {"samples": samples}


def create_diffusion_sample(key: str, seq_len: int, video_shape=(16, 8), embedding_dim=128) -> DiffusionSample:
    """Helper function to create a DiffusionSample for testing."""
    return DiffusionSample(
        __key__=key,
        __restore_key__=(),
        __subflavor__=None,
        __subflavors__=["default"],
        video=torch.randn(seq_len, video_shape[0]),
        context_embeddings=torch.randn(10, embedding_dim),
        context_mask=torch.ones(10),
        loss_mask=torch.ones(seq_len),
        seq_len_q=torch.tensor([seq_len], dtype=torch.int32),
        seq_len_q_padded=torch.tensor([seq_len], dtype=torch.int32),
        seq_len_kv=torch.tensor([10], dtype=torch.int32),
        seq_len_kv_padded=torch.tensor([10], dtype=torch.int32),
        pos_ids=torch.arange(seq_len).unsqueeze(1),
        latent_shape=torch.tensor([4, 2, 4, 4], dtype=torch.int32),
        video_metadata={"fps": 30, "resolution": "512x512"},
    )


def test_select_samples_to_pack():
    """Test select_samples_to_pack method."""
    # Create encoder with seq_length=20
    encoder = ConcreteDiffusionTaskEncoder(seq_length=20)

    # Create samples with different sequence lengths
    samples = [
        create_diffusion_sample("sample_1", seq_len=8),
        create_diffusion_sample("sample_2", seq_len=12),
        create_diffusion_sample("sample_3", seq_len=5),
        create_diffusion_sample("sample_4", seq_len=7),
        create_diffusion_sample("sample_5", seq_len=3),
    ]

    # Call select_samples_to_pack
    result = encoder.select_samples_to_pack(samples)

    # Verify result is a list of lists
    assert isinstance(result, list), "Result should be a list"
    assert all(isinstance(group, list) for group in result), "All elements should be lists"

    # Verify all samples are included
    all_samples = [sample for group in result for sample in group]
    assert len(all_samples) == len(samples), "All samples should be included"

    # Verify no bin exceeds seq_length
    for group in result:
        total_seq_len = sum(sample.seq_len_q.item() for sample in group)
        assert total_seq_len <= encoder.seq_length, (
            f"Bin with total {total_seq_len} exceeds seq_length {encoder.seq_length}"
        )

    # Verify that bins are non-empty
    assert all(len(group) > 0 for group in result), "No bin should be empty"

    print(f"✓ Successfully packed {len(samples)} samples into {len(result)} bins")
    print(f"  Bin sizes: {[sum(s.seq_len_q.item() for s in group) for group in result]}")


def test_pack_selected_samples():
    """Test pack_selected_samples method."""
    encoder = ConcreteDiffusionTaskEncoder(seq_length=100)

    # Create multiple samples to pack
    sample_1_length = 10
    sample_2_length = 15
    sample_3_length = 8
    sample_1 = create_diffusion_sample("sample_1", seq_len=sample_1_length)
    sample_2 = create_diffusion_sample("sample_2", seq_len=sample_2_length)
    sample_3 = create_diffusion_sample("sample_3", seq_len=sample_3_length)

    samples_to_pack = [sample_1, sample_2, sample_3]

    # Pack the samples
    packed_sample = encoder.pack_selected_samples(samples_to_pack)

    # Verify the packed sample is a DiffusionSample
    assert isinstance(packed_sample, DiffusionSample), "Result should be a DiffusionSample"

    # Verify __key__ is concatenated
    expected_key = "sample_1,sample_2,sample_3"
    assert packed_sample.__key__ == expected_key, f"Key should be '{expected_key}'"

    # Verify video is concatenated along dim 0
    expected_video_len = 10 + 15 + 8
    assert packed_sample.video.shape[0] == expected_video_len, f"Video should have length {expected_video_len}"

    # Verify context_embeddings is concatenated
    expected_context_len = 10 * 3  # 3 samples with 10 embeddings each
    assert packed_sample.context_embeddings.shape[0] == expected_context_len, (
        f"Context embeddings should have length {expected_context_len}"
    )

    # Verify context_mask is concatenated
    assert packed_sample.context_mask.shape[0] == expected_context_len, (
        f"Context mask should have length {expected_context_len}"
    )

    # Verify loss_mask is concatenated
    assert packed_sample.loss_mask.shape[0] == expected_video_len, f"Loss mask should have length {expected_video_len}"

    # Verify seq_len_q is concatenated
    assert packed_sample.seq_len_q.shape[0] == 3, "seq_len_q should have 3 elements"
    assert torch.equal(
        packed_sample.seq_len_q, torch.tensor([sample_1_length, sample_2_length, sample_3_length], dtype=torch.int32)
    ), "seq_len_q values incorrect"

    assert packed_sample.seq_len_q_padded.shape[0] == 3, "seq_len_q_padded should have 3 elements"
    assert torch.equal(
        packed_sample.seq_len_q_padded,
        torch.tensor([sample_1_length, sample_2_length, sample_3_length], dtype=torch.int32),
    ), "seq_len_q_padded values incorrect"

    assert packed_sample.seq_len_kv.shape[0] == 3, "seq_len_kv should have 3 elements"
    assert torch.equal(packed_sample.seq_len_kv, torch.tensor([10, 10, 10], dtype=torch.int32)), (
        "seq_len_kv values incorrect"
    )

    assert packed_sample.seq_len_kv_padded.shape[0] == 3, "seq_len_kv_padded should have 3 elements"
    assert torch.equal(packed_sample.seq_len_kv_padded, torch.tensor([10, 10, 10], dtype=torch.int32)), (
        "seq_len_kv_padded values incorrect"
    )

    assert packed_sample.latent_shape.shape[0] == 3, "latent_shape should have 3 rows"
    assert isinstance(packed_sample.video_metadata, list), "video_metadata should be a list"
    assert len(packed_sample.video_metadata) == 3, "video_metadata should have 3 elements"

    print(f"✓ Successfully packed {len(samples_to_pack)} samples")
    print(f"  Packed video shape: {packed_sample.video.shape}")
    print(f"  Packed context embeddings shape: {packed_sample.context_embeddings.shape}")
