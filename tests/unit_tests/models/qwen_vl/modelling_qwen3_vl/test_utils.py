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

"""Unit tests for Qwen3VL utils functions."""

from types import SimpleNamespace

import torch

from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.utils import get_rope_index, split_deepstack_embs


class TestQwen3VLUtils:
    """Test suite for Qwen3VL utility functions."""

    def test_split_deepstack_embs_no_tp(self):
        """Test split_deepstack_embs with tp_size=1."""
        visual_pos_masks = torch.tensor([[True, False, True], [False, True, False]])
        deepstack_visual_embeds = [torch.randn(3, 64), torch.randn(3, 64)]

        masks_out, embeds_out = split_deepstack_embs(visual_pos_masks, deepstack_visual_embeds, tp_size=1)

        assert torch.equal(masks_out, visual_pos_masks)
        assert len(embeds_out) == len(deepstack_visual_embeds)

    def test_split_deepstack_embs_with_tp(self):
        """Test split_deepstack_embs with tp_size=2."""
        visual_pos_masks = torch.tensor([[True, True, False, False]])
        deepstack_visual_embeds = [torch.randn(2, 64)]

        masks_out, embeds_out = split_deepstack_embs(visual_pos_masks, deepstack_visual_embeds, tp_size=2, tp_rank=0)

        assert masks_out.shape[0] == 1
        assert len(embeds_out) == 1

    def test_get_rope_index_text_only(self):
        """Test get_rope_index with text-only input."""
        batch_size, seq_len = 2, 8
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))

        position_ids, deltas = get_rope_index(
            spatial_merge_size=2,
            image_token_id=151655,
            video_token_id=151656,
            vision_start_token_id=151652,
            input_ids=input_ids,
        )

        assert position_ids.shape == (3, batch_size, seq_len)
        assert deltas.shape == (batch_size, 1)

    def test_get_rope_index_with_attention_mask(self):
        """Test get_rope_index with attention mask."""
        batch_size, seq_len = 2, 8
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones((batch_size, seq_len))

        position_ids, deltas = get_rope_index(
            spatial_merge_size=2,
            image_token_id=151655,
            video_token_id=151656,
            vision_start_token_id=151652,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        assert position_ids.shape == (3, batch_size, seq_len)
        assert deltas.shape == (batch_size, 1)

    def test_get_rope_index_with_image(self):
        """Test get_rope_index with image grid."""
        batch_size, seq_len = 1, 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        # Insert vision tokens
        input_ids[0, 4] = 151652  # vision_start_token_id
        input_ids[0, 5] = 151655  # image_token_id
        image_grid_thw = torch.tensor([[1, 4, 4]])  # t=1, h=4, w=4

        position_ids, deltas = get_rope_index(
            spatial_merge_size=2,
            image_token_id=151655,
            video_token_id=151656,
            vision_start_token_id=151652,
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
        )

        assert position_ids.shape == (3, batch_size, seq_len)
        assert deltas.shape == (batch_size, 1)

    def test_get_rope_index_packed_seq_params_builds_mask(self):
        """Test get_rope_index builds attention mask from packed sequence params."""
        batch_size, seq_len = 2, 5
        input_ids = torch.zeros((batch_size, seq_len), dtype=torch.long)
        packed_seq_params = SimpleNamespace(cu_seqlens_q=torch.tensor([0, 3, 5], dtype=torch.int32))

        position_ids, deltas = get_rope_index(
            spatial_merge_size=2,
            image_token_id=151655,
            video_token_id=151656,
            vision_start_token_id=151652,
            input_ids=input_ids,
            packed_seq_params=packed_seq_params,
        )

        expected_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=input_ids.dtype)
        expected_positions = expected_mask.long().cumsum(-1) - 1
        expected_positions.masked_fill_(expected_mask == 0, 1)
        expected_positions = expected_positions.unsqueeze(0).expand(3, -1, -1)
        expected_max = expected_positions.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
        expected_deltas = expected_max + 1 - expected_mask.shape[-1]

        assert torch.equal(position_ids, expected_positions)
        assert torch.equal(deltas, expected_deltas)

    def test_get_rope_index_packed_seq_params_fallback_dense_mask(self):
        """Test get_rope_index falls back to dense mask when cu_seqlens is missing."""
        batch_size, seq_len = 2, 4
        input_ids = torch.zeros((batch_size, seq_len), dtype=torch.long)
        packed_seq_params = SimpleNamespace(cu_seqlens_q=torch.tensor([0], dtype=torch.int32))

        position_ids, deltas = get_rope_index(
            spatial_merge_size=2,
            image_token_id=151655,
            video_token_id=151656,
            vision_start_token_id=151652,
            input_ids=input_ids,
            packed_seq_params=packed_seq_params,
        )

        expected_positions = torch.arange(seq_len, dtype=input_ids.dtype).view(1, 1, -1).expand(3, batch_size, -1)
        expected_deltas = torch.zeros((batch_size, 1), dtype=input_ids.dtype)

        assert torch.equal(position_ids, expected_positions)
        assert torch.equal(deltas, expected_deltas)
