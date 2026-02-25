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

import pytest
import torch
from einops import rearrange
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.transformer_config import TransformerConfig

from megatron.bridge.diffusion.data.dit.dit_taskencoder import pos_id_3d
from megatron.bridge.diffusion.models.dit.dit_model import DiTCrossAttentionModel


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
class TestDiTCrossAttentionModel:
    """Test class for DiTCrossAttentionModel."""

    def setup_method(self, method):
        """Set up test fixtures before each test method."""
        self.batch_size = 2
        self.hidden_size = 512
        self.num_layers = 1
        self.num_attention_heads = 4
        # Dimensions chosen so that seq_len = (max_frames/patch_temporal) * (max_img_h/patch_spatial) * (max_img_w/patch_spatial)
        # seq_len = (8/2) * (8/2) * (8/2) = 4 * 4 * 4 = 64
        self.max_img_h = 8
        self.max_img_w = 8
        self.max_frames = 16
        self.patch_spatial = 2
        self.patch_temporal = 2
        self.seq_len = (
            (self.max_frames // self.patch_temporal)
            * (self.max_img_h // self.patch_spatial)
            * (self.max_img_w // self.patch_spatial)
        )
        self.in_channels = 16
        self.out_channels = 16
        self.crossattn_seq_len = 256

    def test_forward_full_pipeline(self, monkeypatch):
        """Test the forward method with full pre/post processing."""
        # Mock parallel_state functions
        from megatron.core import parallel_state

        monkeypatch.setattr(parallel_state, "is_pipeline_first_stage", lambda: True)
        monkeypatch.setattr(parallel_state, "is_pipeline_last_stage", lambda: True)
        monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_world_size", lambda: 1)
        monkeypatch.setattr(parallel_state, "get_pipeline_model_parallel_world_size", lambda: 1)
        monkeypatch.setattr(parallel_state, "get_data_parallel_world_size", lambda with_context_parallel=False: 1)
        monkeypatch.setattr(parallel_state, "get_context_parallel_world_size", lambda: 1)
        monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_group", lambda **kwargs: None)
        monkeypatch.setattr(parallel_state, "get_data_parallel_group", lambda **kwargs: None)
        monkeypatch.setattr(parallel_state, "get_context_parallel_group", lambda **kwargs: None)
        monkeypatch.setattr(parallel_state, "get_tensor_and_data_parallel_group", lambda **kwargs: None)

        # Create config
        config = TransformerConfig(
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            use_cpu_initialization=True,
            perform_initialization=True,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            sequence_parallel=False,
            add_bias_linear=False,
        )

        # Create model
        model = (
            DiTCrossAttentionModel(
                config=config,
                pre_process=True,
                post_process=True,
                max_img_h=self.max_img_h,
                max_img_w=self.max_img_w,
                max_frames=self.max_frames,
                patch_spatial=self.patch_spatial,
                patch_temporal=self.patch_temporal,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
            )
            .cuda()
            .to(torch.bfloat16)
        )

        # Create input tensors on CUDA
        # x should be [B, S, C] where C = in_channels * patch_spatial^2
        x = torch.randn(
            self.batch_size,
            self.seq_len,
            self.in_channels * self.patch_spatial**2,
            dtype=torch.bfloat16,
            device="cuda",
        )
        x = x.reshape(1, -1, self.in_channels * self.patch_spatial**2)

        import math

        sigma_min = 0.0002
        sigma_max = 80.0
        c_noise_min = 0.25 * math.log(sigma_min)
        c_noise_max = 0.25 * math.log(sigma_max)
        timesteps = (torch.rand(1, device="cuda") * (c_noise_max - c_noise_min) + c_noise_min).to(torch.bfloat16)

        # crossattn_emb should be [B, S_cross, D]
        crossattn_emb = torch.randn(
            self.batch_size, self.crossattn_seq_len, self.hidden_size, dtype=torch.bfloat16, device="cuda"
        )
        crossattn_emb = crossattn_emb.reshape(1, -1, self.hidden_size)
        pos_ids = rearrange(
            pos_id_3d.get_pos_id_3d(
                t=self.max_frames // self.patch_temporal,
                h=self.max_img_h // self.patch_spatial,
                w=self.max_img_w // self.patch_spatial,
            ),
            "T H W d -> (T H W) d",
        )
        pos_ids = pos_ids.unsqueeze(0).expand(self.batch_size, -1, -1)
        pos_ids = pos_ids.reshape(1, -1, 3).cuda()

        cu_seqlens_q = torch.arange(
            0, (self.batch_size + 1) * self.seq_len, self.seq_len, dtype=torch.int32, device="cuda"
        )
        cu_seqlens_kv_cross = torch.arange(
            0, (self.batch_size + 1) * self.crossattn_seq_len, self.crossattn_seq_len, dtype=torch.int32, device="cuda"
        )

        packed_seq_params = {
            "self_attention": PackedSeqParams(
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_q,
                cu_seqlens_q_padded=cu_seqlens_q,
                cu_seqlens_kv_padded=cu_seqlens_q,
                qkv_format="thd",
            ),
            "cross_attention": PackedSeqParams(
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv_cross,
                cu_seqlens_q_padded=cu_seqlens_q,
                cu_seqlens_kv_padded=cu_seqlens_kv_cross,
                qkv_format="thd",
            ),
        }

        # Run forward pass with original crossattn_emb
        with torch.no_grad():
            output_original = model(
                x=x,
                timesteps=timesteps,
                crossattn_emb=crossattn_emb,
                pos_ids=pos_ids,
                packed_seq_params=packed_seq_params,
            )

        # Verify output shape
        # Expected output: [B, S, patch_spatial^2 * patch_temporal * out_channels]
        expected_out_channels = self.patch_spatial**2 * self.patch_temporal * self.out_channels
        expected_shape = (1, self.batch_size * self.seq_len, expected_out_channels)

        assert output_original.shape == expected_shape, (
            f"Expected output shape {expected_shape}, got {output_original.shape}"
        )

        # Verify output is not NaN or Inf
        assert not torch.isnan(output_original).any(), "Output contains NaN values"
        assert not torch.isinf(output_original).any(), "Output contains Inf values"

        # Verify output dtype
        assert output_original.dtype == torch.bfloat16, (
            f"Expected output dtype torch.bfloat16, got {output_original.dtype}"
        )

        print(f"Forward pass successful with output shape: {output_original.shape}.")
