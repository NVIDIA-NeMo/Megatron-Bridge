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

"""
Run with: pytest tests/unit_tests/models/qwen_omni/modeling_qwen3_omni/test_rope.py
"""

import torch

from megatron.bridge.models.qwen_omni.modeling_qwen3_omni.rope import get_rope_index


class TestQwen3OmniMoeRope:
    """Test suite for Qwen3OmniMoe utility functions."""

    def test_get_rope_index_text_only(self):
        """Test get_rope_index with text-only input."""
        batch_size, seq_len = 2, 8
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))

        position_ids, deltas = get_rope_index(
            spatial_merge_size=2,
            image_token_id=151655,
            video_token_id=151656,
            audio_token_id=151675,
            vision_start_token_id=151652,
            audio_start_token_id=151669,
            position_id_per_seconds=13,
            input_ids=input_ids,
        )

        assert position_ids.shape == (3, batch_size, seq_len)
        assert deltas.shape == (batch_size, 1)

    def test_get_rope_index_with_attention_mask(self):
        """Test get_rope_index with attention mask"""
        batch_size, seq_len = 1, 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones((batch_size, seq_len))

        position_ids, deltas = get_rope_index(
            spatial_merge_size=2,
            image_token_id=151655,
            video_token_id=151656,
            audio_token_id=151675,
            vision_start_token_id=151652,
            audio_start_token_id=151669,
            position_id_per_seconds=13,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        assert position_ids.shape == (3, batch_size, seq_len)
        assert deltas.shape == (batch_size, 1)

    def test_get_rope_index_with_image(self):
        """Test get_rope_index with image grid"""
        batch_size, seq_len = 1, 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        # Insert image tokens
        input_ids[0, 4] = 151652  # vision_start_token_id
        input_ids[0, 5] = 151655  # image_token_id
        image_grid_thw = torch.tensor([[1, 4, 4]])

        position_ids, deltas = get_rope_index(
            spatial_merge_size=2,
            image_token_id=151655,
            video_token_id=151656,
            audio_token_id=151675,
            vision_start_token_id=151652,
            audio_start_token_id=151669,
            position_id_per_seconds=13,
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
        )

        assert position_ids.shape == (3, batch_size, seq_len)
        assert deltas.shape == (batch_size, 1)

    def test_get_rope_index_with_video(self):
        """Test get_rope_index with video grid"""
        batch_size, seq_len = 1, 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        # Insert vidion tokens
        input_ids[0, 4] = 151652  # vision_start_token_id
        input_ids[0, 5] = 151656  # video_token_id
        video_grid_thw = torch.tensor([[1, 4, 4]])
        second_per_grids = torch.tensor([2])

        position_ids, deltas = get_rope_index(
            spatial_merge_size=2,
            image_token_id=151655,
            video_token_id=151656,
            audio_token_id=151675,
            vision_start_token_id=151652,
            audio_start_token_id=151669,
            position_id_per_seconds=13,
            input_ids=input_ids,
            video_grid_thw=video_grid_thw,
            second_per_grids=second_per_grids,
        )

        assert position_ids.shape == (3, batch_size, seq_len)
        assert deltas.shape == (batch_size, 1)

    def test_get_rope_index_with_audio_in_video(self):
        """Test get_rope_index with audio grid"""
        batch_size, seq_len = 1, 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        # Insert vidion tokens
        input_ids[0, 4] = 151652  # vision_start_token_id
        input_ids[0, 5] = 151669  # audio_start_token_id
        input_ids[0, 6] = 151656  # video_token_id
        input_ids[0, 7] = 151675  # audio_token_id
        video_grid_thw = torch.tensor([[1, 4, 4]])
        audio_seqlens = torch.tensor([1])
        second_per_grids = torch.tensor([2])

        position_ids, deltas = get_rope_index(
            spatial_merge_size=2,
            image_token_id=151655,
            video_token_id=151656,
            audio_token_id=151675,
            vision_start_token_id=151652,
            audio_start_token_id=151669,
            position_id_per_seconds=13,
            input_ids=input_ids,
            video_grid_thw=video_grid_thw,
            use_audio_in_video=True,
            audio_seqlens=audio_seqlens,
            second_per_grids=second_per_grids,
        )

        assert position_ids.shape == (3, batch_size, seq_len)
        assert deltas.shape == (batch_size, 1)
