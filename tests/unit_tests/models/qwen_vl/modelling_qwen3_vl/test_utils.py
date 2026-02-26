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

import datetime
import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.rope import get_rope_index
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.transformer_config import Qwen3VLTransformerConfig
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.utils import (
    AllGatherVisionEmbeddings,
    PatchMergerSubmodules,
    Qwen3VLVisionPatchEmbed,
    Qwen3VLVisionPatchMerger,
    Qwen3VLVisionRotaryEmbedding,
    collapse_thw,
    expand_thw,
    get_vision_cp_data,
    preprocess_packed_seqs,
    qwen3vl_cp_split,
    reorganize_inputs,
    split_data_cp_rank,
    split_deepstack_embs,
    split_part_by_cp_tp,
)


"""
Test utils functions for Qwen3VL model.
    Run with: torchrun --nproc_per_node=2 tests/unit_tests/models/qwen_vl/modelling_qwen3_vl/test_utils.py
"""


class TestQwen3VLUtils:
    """Test suite for Qwen3VL utility functions."""

    @classmethod
    def setup_class(cls):
        if not dist.is_initialized():
            rank = int(os.environ.get("RANK", "0"))
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            world_size = int(os.environ.get("WORLD_SIZE", "1"))

            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = "29500"
            os.environ["RANK"] = str(rank)
            os.environ["LOCAL_RANK"] = str(local_rank)
            os.environ["WORLD_SIZE"] = str(world_size)

            torch.cuda.set_device(local_rank)

            dist.init_process_group(
                backend="nccl",
                world_size=world_size,
                rank=rank,
                timeout=datetime.timedelta(minutes=30),
            )

    @classmethod
    def teardown_class(cls):
        """Teardown distributed process group once after all tests in this class."""
        if dist.is_initialized():
            dist.destroy_process_group()

    def _setup_parallel_state(self, tp_size=1, ep_size=1, pp_size=1, cp_size=1):
        """Setup Megatron parallel state with specified parallelism configuration.

        Args:
            tp_size: Tensor model parallel size
            ep_size: Expert model parallel size
            pp_size: Pipeline model parallel size
            cp_size: Context parallel size
        """
        # Clean up any existing parallel state before initializing
        if parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()

        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=pp_size,
            virtual_pipeline_model_parallel_size=None,
            context_parallel_size=cp_size,
            expert_model_parallel_size=ep_size,
            expert_tensor_parallel_size=1,
        )

        model_parallel_cuda_manual_seed(123)

    def destroy_parallel_state(self):
        """Destroy Megatron parallel state."""
        parallel_state.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="VisionRotaryEmbedding uses CUDA")
    def test_vision_patch_embed(self):
        """Test Qwen3VLVisionPatchEmbed forward pass with representative config."""
        config = Qwen3VLTransformerConfig(
            hidden_size=128,
            num_attention_heads=2,
            patch_size=16,
            temporal_patch_size=2,
            in_channels=3,
            num_layers=1,
        )
        module = Qwen3VLVisionPatchEmbed(config)
        # Input: [N, C, T, H, W] after view -> N = batch*t*h*w patches per (t,h,w) patch
        n_patches = 4
        hidden = torch.randn(
            n_patches, config.in_channels, config.temporal_patch_size, config.patch_size, config.patch_size
        )
        out = module(hidden)
        assert out.shape == (n_patches, config.hidden_size)
        torch.cuda.empty_cache()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="VisionRotaryEmbedding uses CUDA")
    def test_vision_rotary_embedding(self):
        """Test Qwen3VLVisionRotaryEmbedding forward."""
        dim = 64
        seqlen = 10
        module = Qwen3VLVisionRotaryEmbedding(dim=dim)
        freqs = module(seqlen)
        assert freqs.shape == (seqlen, dim // 2)

    def test_patch_merger(self):
        """Test Qwen3VLVisionPatchMerger forward (requires parallel state)."""
        self._setup_parallel_state(tp_size=1, ep_size=1, pp_size=1, cp_size=1)
        config = Qwen3VLTransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=2,
            patch_size=16,
            spatial_merge_size=2,
            out_hidden_size=128,
        )
        from megatron.core.extensions.transformer_engine import (
            TEColumnParallelLinear,
            TENorm,
            TERowParallelLinear,
        )

        submodules = PatchMergerSubmodules(
            patch_norm=TENorm,
            linear_fc1=TEColumnParallelLinear,
            linear_fc2=TERowParallelLinear,
        )
        merger = Qwen3VLVisionPatchMerger(config, submodules).cuda()
        seqlens = 128
        x = torch.randn(seqlens, config.hidden_size).cuda()
        out = merger(x)
        assert out.shape == (seqlens // config.spatial_merge_size**2, config.out_hidden_size)
        torch.cuda.empty_cache()
        self.destroy_parallel_state()

    def test_split_part_by_cp_tp(self):
        """
        For a small configuration where we can reason about exact values:
        cp_size = 2, tp_size = 2, split_size = 8
        part_list = [0,1,2,3,4,5,6,7]
        """
        cp_size = 2
        tp_size = 2
        split_size = 2 * cp_size * tp_size  # 8

        # cp_rank=0: chunks [0,1] and [6,7]
        assert split_part_by_cp_tp(cp_size, 0, tp_size, 0, split_size) == [0, 1]
        assert split_part_by_cp_tp(cp_size, 0, tp_size, 1, split_size) == [6, 7]

        # cp_rank=1: chunks [2,3] and [4,5]
        assert split_part_by_cp_tp(cp_size, 1, tp_size, 0, split_size) == [2, 3]
        assert split_part_by_cp_tp(cp_size, 1, tp_size, 1, split_size) == [4, 5]

    def test_reorganize_inputs(self):
        """Test reorganize_inputs for image-only and video-only."""
        image_token_id = 151655
        video_token_id = 151656
        # Image only: no videos
        input_ids = torch.tensor([1, image_token_id, image_token_id, 2], dtype=torch.long)
        pixel_values = torch.randn(4, 3, 2, 4, 4)  # 4 patches, C, T, H, W
        image_grid_thw = torch.tensor([[1, 2, 2]], dtype=torch.long)
        pv, grid, mask = reorganize_inputs(
            input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            image_token_id=image_token_id,
            video_token_id=video_token_id,
        )
        assert pv.shape[0] == 4
        assert grid.shape == (1, 3)
        assert mask is not None
        # Video only: no images
        input_ids = torch.tensor([1, video_token_id, video_token_id, 2], dtype=torch.long)
        pixel_values_videos = torch.randn(4, 3, 2, 4, 4)
        video_grid_thw = torch.tensor([[1, 2, 2]], dtype=torch.long)
        pv, grid, mask = reorganize_inputs(
            input_ids,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            image_token_id=image_token_id,
            video_token_id=video_token_id,
        )

        assert pv.shape[0] == 4
        assert grid.shape == (1, 3)

    def test_split_data_cp_rank(self):
        """Test split_data_cp_rank zigzag split along seq_dim."""
        # seq_dim=1, shape (2, 8, 4) -> 8 must be divisible by 2*cp_size
        val = torch.arange(2 * 8 * 4, dtype=torch.float).view(2, 8, 4)
        out = split_data_cp_rank(val, cp_size=2, seq_dim=1, cp_rank=0)
        assert out.shape == (2, 4, 4)
        out1 = split_data_cp_rank(val, cp_size=2, seq_dim=1, cp_rank=1)
        assert out1.shape == (2, 4, 4)
        # Zigzag: rank 0 gets indices 0,3 (first and last quarter of 8)
        expected_0 = torch.cat([val[:, 0:2], val[:, 6:8]], dim=1)
        assert torch.equal(out, expected_0)

    def test_expand_thw(self):
        """Test expand_thw repeats rows by first column."""
        thw = torch.tensor([[2, 4, 4], [1, 2, 2]], dtype=torch.long)
        out = expand_thw(thw)
        assert out.shape == (3, 3)
        assert out[0, 0].item() == 1 and out[1, 0].item() == 1 and out[2, 0].item() == 1
        assert (out[:, 1:] == thw[:, 1:].repeat_interleave(thw[:, 0], dim=0)).all()

    def test_collapse_thw(self):
        """Test collapse_thw and roundtrip with expand_thw."""
        thw = torch.tensor([[2, 4, 4], [1, 2, 2], [3, 1, 1]], dtype=torch.long)
        expanded = expand_thw(thw)
        collapsed = collapse_thw(expanded)
        assert collapsed.shape == thw.shape
        assert torch.equal(collapsed, thw)
        # Single row
        single = torch.tensor([[1, 2, 2]], dtype=torch.long)
        assert torch.equal(collapse_thw(single), single)
        # Two rows, same (t,h,w) -> count 2
        two = torch.tensor([[1, 2, 2], [1, 2, 2]], dtype=torch.long)
        assert torch.equal(collapse_thw(two), torch.tensor([[2, 2, 2]], dtype=torch.long))

    def test_qwen3vl_cp_split(self):
        """Test qwen3vl_cp_split with cp_size=1 (no split) and cp_size=2."""
        pixel_values = torch.randn(16, 64)
        image_grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long)

        pv, grid, cp_img_num, images_padded = qwen3vl_cp_split(2, pixel_values, image_grid_thw)
        assert pv is not None
        assert grid is not None
        assert len(cp_img_num) == 2
        assert all(not p for p in images_padded)

    def test_get_vision_cp_data(self):
        """Test get_vision_cp_data returns correct slice for cp_rank."""
        vision_data = torch.randn(12, 64)  # 8+4 tokens for two grid rows
        vision_grid_thw = torch.tensor([[1, 2, 4], [1, 2, 2]], dtype=torch.long)  # 8+4=12 tokens
        square_merge_size = 4
        cp_img_num = [1, 1]
        images_padded = [False, False]
        data, grid, seqlens = get_vision_cp_data(
            vision_data,
            vision_grid_thw,
            square_merge_size,
            cp_img_num,
            images_padded,
            cp_rank=0,
            cp_size=2,
        )
        assert data.shape[0] == 8
        assert grid.shape == (1, 3)
        assert len(seqlens) == 2
        data1, grid1, _ = get_vision_cp_data(
            vision_data,
            vision_grid_thw,
            square_merge_size,
            cp_img_num,
            images_padded,
            cp_rank=1,
            cp_size=2,
        )
        assert data1.shape[0] == 4
        assert grid1.shape == (1, 3)

    @pytest.mark.skipif(
        not torch.cuda.is_available() or int(os.environ.get("WORLD_SIZE", "1")) < 2,
        reason="Requires at least 2 GPUs",
    )
    def test_preprocess_packed_seqs(self):
        """Test preprocess_packed_seqs with pg_collection (or mpu when init)."""
        tp_size = 1
        cp_size = 2
        self._setup_parallel_state(tp_size=tp_size, cp_size=cp_size)

        batch_size, seq_len = 2, 8
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        ids_out, packed = preprocess_packed_seqs(input_ids, attention_mask, pre_process=True)
        assert packed.max_seqlen_q == packed.max_seqlen_kv
        assert ids_out.shape[0] == 1
        assert ids_out.shape[1] == attention_mask.sum().item() // cp_size

        self.destroy_parallel_state()

    @pytest.mark.skipif(
        not torch.cuda.is_available() or int(os.environ.get("WORLD_SIZE", "1")) < 2,
        reason="Requires at least 2 GPUs",
    )
    def test_allgather_vision_embeddings(self):
        """Test AllGatherVisionEmbeddings forward/backward."""
        tp_size = 1
        cp_size = 2
        self._setup_parallel_state(tp_size=tp_size, cp_size=cp_size)
        local_seqlen = 4
        hidden_size = 64
        seqlens_on_cp_ranks = [
            torch.tensor([local_seqlen], dtype=torch.long, device=torch.cuda.current_device()) for _ in range(cp_size)
        ]
        input_ = torch.randn(
            local_seqlen, hidden_size, dtype=torch.float, device=torch.cuda.current_device(), requires_grad=True
        )
        output = AllGatherVisionEmbeddings.apply(
            input_, seqlens_on_cp_ranks, parallel_state.get_context_parallel_group()
        )
        assert output.shape == (local_seqlen * cp_size, hidden_size)
        output.sum().backward()
        assert input_.grad is not None

        self.destroy_parallel_state()

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
