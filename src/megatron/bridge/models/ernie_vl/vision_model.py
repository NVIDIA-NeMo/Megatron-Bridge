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
Megatron-Core native Vision Transformer for ERNIE 4.5 VL.

This module implements the ERNIE 4.5 VL DFN-style ViT using Megatron-Core
TransformerBlock infrastructure instead of the HuggingFace implementation.
This enables TP-native attention and MLP layers for better distributed
training performance.

Architecture (matching HF DFNRopeVisionTransformerPretrainedModel):
    - PatchEmbed: nn.Linear(C * P * P, embed_dim, bias=False)
    - 2D RoPE: Non-interleaved rotate_half with spatial_merge_size reordering
    - 32x TransformerLayer (TE-backed):
        - LayerNorm(1280, eps=1e-6) -> QKV(1280, 3*1280, bias=True) -> Attention -> Proj
        - LayerNorm(1280, eps=1e-6) -> FC1(1280, 5120) -> quick_gelu -> FC2(5120, 1280)
    - Final LayerNorm(1280, eps=1e-6)
    - Per-image packed sequence attention via PackedSeqParams (thd format)

Key differences from Qwen3VL MG ViT:
    - PatchEmbed uses nn.Linear (not Conv3d) on pre-flattened patches
    - No positional embedding interpolation (ERNIE uses pure 2D RoPE)
    - No deepstack feature extraction
    - Non-interleaved RoPE (rotate_half style, rotary_interleaved=False)
    - No PatchMerger (merging is done by the resampler)
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core import InferenceParams
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlock

from megatron.bridge.models.ernie_vl.vision_transformer_config import ErnieVisionTransformerConfig


class ErnieVisionPatchEmbed(nn.Module):
    """Patch embedding for ERNIE 4.5 VL ViT.

    Unlike Qwen3VL which uses Conv3d on raw image tensors, ERNIE's processor
    pre-flattens each patch into a vector of size [C * patch_size^2] = [588],
    so patch embedding is a simple linear projection.

    Args:
        in_channels: Number of input channels (default 3).
        patch_size: Patch size in pixels (default 14).
        embed_dim: Embedding dimension (default 1280).
    """

    def __init__(self, in_channels: int = 3, patch_size: int = 14, embed_dim: int = 1280):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        # HF PatchEmbed: nn.Linear(in_channels * patch_size^2, embed_dim, bias=False)
        self.proj = nn.Linear(in_channels * patch_size * patch_size, embed_dim, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project pre-flattened patches to embedding space.

        Args:
            hidden_states: [total_patches, C * patch_size^2] (e.g., [N, 588])

        Returns:
            [total_patches, embed_dim] (e.g., [N, 1280])
        """
        return self.proj(hidden_states.to(dtype=self.proj.weight.dtype))


class ErnieVisionRotaryEmbedding(nn.Module):
    """1D rotary embedding frequency table for ERNIE ViT 2D RoPE.

    Computes a frequency table of shape [max_seqlen, dim//2] which is then
    indexed by 2D (H, W) position IDs to produce per-token RoPE embeddings.

    This matches HF's ``VisionRotaryEmbedding`` in the ERNIE 4.5 VL model.

    Args:
        dim: Half of the per-head dimension (head_dim // 2).
            For ERNIE ViT: head_dim = 1280 / 16 = 80, so dim = 40.
        theta: RoPE base frequency (default 10000.0).
    """

    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, seqlen: int) -> torch.Tensor:
        """Compute frequency table for positions 0..seqlen-1.

        Args:
            seqlen: Maximum sequence length to compute frequencies for.

        Returns:
            Tensor of shape [seqlen, dim] containing outer product of
            position indices and inverse frequencies.
        """
        if not hasattr(self, "inv_freq"):
            inv_freq = 1.0 / (
                self.theta
                ** (
                    torch.arange(0, self.dim, 2, dtype=torch.float) / self.dim
                )
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class ErnieVLVisionModel(VisionModule):
    """Megatron-Core native ERNIE 4.5 VL Vision Transformer.

    Implements the DFN-style ViT with 2D RoPE using MCore TransformerBlock
    for TP-native distributed training.

    Architecture:
        1. PatchEmbed (nn.Linear, replicated)
        2. 2D RoPE computation (per-image H/W position lookup with spatial merge reordering)
        3. TransformerBlock (32 ViT layers with TE modules)
        4. Final LayerNorm

    Unlike the HF-wrapped version, this implementation:
        - Uses TE-backed attention and MLP layers for TP support
        - Leverages MCore's PackedSeqParams for per-image variable-length attention
        - Enables activation recomputation through TransformerBlock

    Args:
        transformer_config: ErnieVisionTransformerConfig with ViT hyperparameters.
        transformer_layer_spec: ModuleSpec for each ViT transformer layer.
    """

    def __init__(
        self,
        transformer_config: ErnieVisionTransformerConfig,
        transformer_layer_spec: ModuleSpec,
    ) -> None:
        super().__init__(config=transformer_config)

        self.spatial_merge_size = transformer_config.spatial_merge_size
        self.patch_size = transformer_config.patch_size

        # Patch embedding: nn.Linear (replicated across TP ranks)
        self.patch_embed = ErnieVisionPatchEmbed(
            in_channels=transformer_config.in_channels,
            patch_size=transformer_config.patch_size,
            embed_dim=transformer_config.hidden_size,
        )

        # 1D frequency table for 2D RoPE lookup
        head_dim = transformer_config.hidden_size // transformer_config.num_attention_heads
        self.rotary_pos_emb = ErnieVisionRotaryEmbedding(head_dim // 2)

        self.model_type = ModelType.encoder_or_decoder

        # Transformer layers (32 ViT blocks with TE modules)
        self.decoder = TransformerBlock(
            config=transformer_config,
            spec=transformer_layer_spec,
            pre_process=True,
            post_process=True,
            post_layer_norm=True,  # Apply final LN after last block
        )

        self.input_tensor = None

    def set_input_tensor(self, input_tensor: torch.Tensor) -> None:
        """Set input tensor (for pipeline parallelism, currently not used for ViT)."""
        self.input_tensor = input_tensor

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """Compute 2D RoPE positional embeddings for all tokens.

        For each image/video frame, computes (H, W) position IDs with
        spatial_merge_size reordering (grouping merge_size x merge_size
        patches together), then looks up the frequency table.

        The spatial merge reordering ensures that patches within each
        spatial merge unit (2x2 by default) are consecutive in the
        sequence, matching the resampler's spatial pooling pattern.

        Args:
            grid_thw: [num_images, 3] tensor of (T, H, W) grid dimensions
                      for each image/video.

        Returns:
            Tensor of shape [total_tokens, head_dim] containing the
            concatenated cos/sin frequencies for 2D RoPE.
        """
        merge_size = self.spatial_merge_size

        # Compute frequency table up to the maximum spatial dimension
        max_hw = int(grid_thw[:, 1:].max().item())
        freq_table = self.rotary_pos_emb(max_hw)  # [max_hw, dim//2]
        device = freq_table.device

        total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for num_frames, height, width in grid_thw:
            merged_h = height // merge_size
            merged_w = width // merge_size

            # Build position IDs with spatial merge reordering:
            # Within each merged_h x merged_w block, iterate over
            # merge_size x merge_size sub-positions
            block_rows = torch.arange(merged_h, device=device)
            block_cols = torch.arange(merged_w, device=device)
            intra_row = torch.arange(merge_size, device=device)
            intra_col = torch.arange(merge_size, device=device)

            # Full-resolution (H, W) positions
            row_idx = (
                block_rows[:, None, None, None] * merge_size
                + intra_row[None, None, :, None]
            )
            col_idx = (
                block_cols[None, :, None, None] * merge_size
                + intra_col[None, None, None, :]
            )

            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

            coords = torch.stack((row_idx, col_idx), dim=-1)  # [H*W, 2]

            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)

            num_tokens = coords.shape[0]
            pos_ids[offset: offset + num_tokens] = coords
            offset += num_tokens

        # Look up frequency table by position IDs: [total_tokens, 2, dim//2]
        embeddings = freq_table[pos_ids]
        # Flatten to [total_tokens, head_dim//2 * 2 = head_dim]
        embeddings = embeddings.flatten(1)
        return embeddings

    def build_packed_seq_params(
        self,
        grid_thw: torch.Tensor,
    ) -> PackedSeqParams:
        """Build PackedSeqParams for per-image variable-length attention.

        Each frame in each image/video is treated as a separate sequence
        for attention computation. This enables per-image attention without
        cross-image contamination.

        Args:
            grid_thw: [num_images, 3] tensor of (T, H, W) grid dimensions.

        Returns:
            PackedSeqParams with cu_seqlens for thd-format attention.
        """
        # Each frame is a separate sequence: seqlen = H * W
        seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        )
        cu_seqlens = seqlens.cumsum(dim=0)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0).int()

        max_seqlen_q = seqlens.max()
        return PackedSeqParams(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            qkv_format="thd",
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_q,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
        inference_params: Optional[InferenceParams] = None,
        extra_block_kwargs: Optional[dict] = None,
    ) -> torch.Tensor:
        """Forward pass of the ERNIE ViT.

        Args:
            hidden_states: Pre-flattened pixel patches [total_patches, C*P*P].
            grid_thw: [num_images, 3] tensor of (T, H, W) grid dimensions.
            inference_params: Inference parameters (currently unused for ViT).
            extra_block_kwargs: Extra kwargs to pass to TransformerBlock.

        Returns:
            Vision features of shape [total_patches, hidden_size].
        """
        assert grid_thw is not None
        assert self.input_tensor is None
        assert inference_params is None

        # 1. Patch embedding: [total_patches, C*P*P] -> [total_patches, embed_dim]
        hidden_states = self.patch_embed(hidden_states)

        seq_len = hidden_states.size(0)

        # 2. Compute 2D RoPE frequencies: [total_patches, head_dim]
        #    rot_pos_emb() returns raw frequency values (theta * position) of
        #    shape [total_patches, head_dim//2] = [N, 40].
        #
        #    HF's apply_rotary_pos_emb_vision tiles cos/sin along the last dim:
        #        cos = freqs.cos().unsqueeze(1).tile(1, 1, 2)   # (N,40) -> (N,1,80)
        #    so the effective frequency pattern is [f0..f39, f0..f39] (doubled).
        #    This means ALL head_dim dimensions get rotated (rot_dim == head_dim).
        #
        #    MCore's _apply_rotary_pos_emb_bshd uses freqs.shape[-1] as rot_dim.
        #    To match HF's tiling, we must duplicate the frequencies:
        #        [f0..f39] -> [f0..f39, f0..f39]  (shape 80)
        #    so that rot_dim == head_dim and the rotation covers all dimensions.
        #
        #    With rotary_interleaved=False, MCore's _rotate_half splits at the
        #    midpoint [-x2, x1] which is identical to HF's rotate_half.
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, 1, 1, -1)
        rotary_pos_emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)

        # 3. Reshape for TransformerBlock: [total_tokens, 1, embed_dim]
        #    (seq_len, batch=1, hidden_size)
        hidden_states = hidden_states[:, None]

        # 4. Forward through transformer layers
        hidden_states = self.decoder(
            hidden_states=hidden_states,
            attention_mask=None,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=self.build_packed_seq_params(grid_thw),
            **(extra_block_kwargs or {}),
        )

        # 5. Remove batch dimension: [total_tokens, 1, hidden_size] -> [total_tokens, hidden_size]
        hidden_states = hidden_states.squeeze(1)

        return hidden_states
