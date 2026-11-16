# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Inkling2 image encoder: a Perception Encoder-style vision transformer and its decoder adapter.

No Transformers implementation of this encoder exists, so it is defined here
with the reference implementation's parameter names and arithmetic. Inputs are
merged 40x40 patches with two identical temporal slices; each is split into
four native 20x20 patches. A vision transformer with a CLS token, interpolated
learned position embeddings, 2D rotary embeddings, and LayerScale encodes each
image bidirectionally; neighboring 2x2 features are then concatenated and
projected to the decoder width.
"""

import torch
import torch.nn.functional as F
from torch import nn


def interpolate_position_embeddings(
    grid: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor, heights: torch.Tensor, widths: torch.Tensor
) -> torch.Tensor:
    """Bilinearly resample a learned ``[G, G, D]`` position grid at each token's patch center."""

    def weights(indices, targets, size):
        center = (size / targets.float()) * (indices.float() + 0.5)
        distance = torch.arange(size, device=grid.device)[None, :] + 0.5
        result = (1 - (distance - center[:, None]).abs()).clamp(min=0)
        return result / result.sum(dim=1, keepdim=True)

    result = torch.einsum(
        "ni,nj,ijd->nd",
        weights(rows, heights, grid.shape[0]),
        weights(cols, widths, grid.shape[1]),
        grid.float(),
    )
    return result.to(grid.dtype)


def apply_2d_rotary(query: torch.Tensor, key: torch.Tensor, positions: torch.Tensor):
    """Rotate interleaved channel pairs by column angles (first half) and row angles (second half)."""
    dim = query.shape[-1]
    frequencies = 1.0 / (10000 ** (torch.arange(0, dim, 4, device=query.device, dtype=torch.float32) / dim))
    angles = torch.cat(
        (positions[:, 1, None].float() * frequencies, positions[:, 0, None].float() * frequencies), dim=-1
    )
    cos, sin = torch.cos(angles)[:, None, :], torch.sin(angles)[:, None, :]

    def rotate(value):
        pairs = value.float().reshape(*value.shape[:-1], -1, 2)
        even, odd = pairs[..., 0], pairs[..., 1]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2).to(value.dtype)

    return rotate(query), rotate(key)


class Inkling2VisionAttention(nn.Module):
    """Bidirectional multi-head attention within each image."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.num_heads = heads
        self.head_dim = width // heads
        self.q_proj = nn.Linear(width, width)
        self.k_proj = nn.Linear(width, width)
        self.v_proj = nn.Linear(width, width)
        self.out_proj = nn.Linear(width, width)

    def forward(self, hidden: torch.Tensor, positions: torch.Tensor, lengths: list[int]) -> torch.Tensor:
        """Attend within each image span of the flattened token sequence."""
        shape = (-1, self.num_heads, self.head_dim)
        query, key, value = (
            projection(hidden).reshape(shape) for projection in (self.q_proj, self.k_proj, self.v_proj)
        )
        query, key = apply_2d_rotary(query, key, positions)
        outputs = [
            F.scaled_dot_product_attention(
                q.transpose(0, 1).unsqueeze(0),
                k.transpose(0, 1).unsqueeze(0),
                v.transpose(0, 1).unsqueeze(0),
                scale=self.head_dim**-0.5,
            )
            .squeeze(0)
            .transpose(0, 1)
            for q, k, v in zip(query.split(lengths), key.split(lengths), value.split(lengths))
        ]
        return self.out_proj(torch.cat(outputs).reshape(hidden.shape))


class Inkling2VisionMLP(nn.Module):
    """GELU feed-forward block."""

    def __init__(self, width: int, intermediate_size: int) -> None:
        super().__init__()
        self.c_fc = nn.Linear(width, intermediate_size)
        self.c_proj = nn.Linear(intermediate_size, width)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Apply the feed-forward block."""
        return self.c_proj(F.gelu(self.c_fc(hidden)))


class Inkling2VisionLayerScale(nn.Module):
    """Learned per-channel residual-branch scale."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(width))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Scale the residual branch."""
        return hidden * self.gamma


class Inkling2VisionLayer(nn.Module):
    """Pre-LayerNorm transformer layer with LayerScale."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        width, eps = config["hidden_size"], config["layer_norm_eps"]
        self.ln_1 = nn.LayerNorm(width, eps=eps)
        self.self_attn = Inkling2VisionAttention(width, config["num_attention_heads"])
        self.ln_2 = nn.LayerNorm(width, eps=eps)
        self.mlp = Inkling2VisionMLP(width, int(width * config["mlp_ratio"]))
        self.ls_1 = Inkling2VisionLayerScale(width)
        self.ls_2 = Inkling2VisionLayerScale(width)

    def forward(self, hidden: torch.Tensor, positions: torch.Tensor, lengths: list[int]) -> torch.Tensor:
        """Apply attention and MLP residual branches."""
        hidden = hidden + self.ls_1(self.self_attn(self.ln_1(hidden), positions, lengths))
        return hidden + self.ls_2(self.mlp(self.ln_2(hidden)))


class Inkling2VisionEmbeddings(nn.Module):
    """Native-patch embedding, CLS token, and learned position grid."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        width = config["hidden_size"]
        self.patch_embedding = nn.Linear(3 * config["native_patch_size"] ** 2, width, bias=False)
        self.position_embedding = nn.Parameter(torch.zeros(config["posemb_grid_size"] ** 2, width))
        self.class_embedding = nn.Parameter(torch.zeros(width))
        self.cls_position_embedding = nn.Parameter(torch.zeros(width))

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        """Embed flattened native patches."""
        return self.patch_embedding(pixels)


class Inkling2VisionBackbone(nn.Module):
    """Embeddings, pre-LayerNorm, and transformer layers."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.embeddings = Inkling2VisionEmbeddings(config)
        self.ln_pre = nn.LayerNorm(config["hidden_size"], eps=config["layer_norm_eps"])
        self.layers = nn.ModuleList([Inkling2VisionLayer(config) for _ in range(config["num_hidden_layers"])])


class Inkling2VisionEncoder(nn.Module):
    """Encode preprocessed image patches into decoder-width embeddings, one per merged patch."""

    def __init__(self, vision_config: dict, text_hidden_size: int) -> None:
        super().__init__()
        self.config = dict(vision_config)
        self.patch_size = vision_config["native_patch_size"]
        self.merge_size = vision_config["adapter_merge_size"]
        self.temporal_patch_size = vision_config["temporal_patch_size"]
        self.backbone = Inkling2VisionBackbone(vision_config)
        projection = vision_config["projection_hidden_dim"]
        bias = vision_config["projector_bias"]
        self.proj_in = nn.Linear(self.merge_size**2 * vision_config["hidden_size"], projection, bias=bias)
        self.proj_out = nn.Linear(projection, text_hidden_size, bias=bias)

    def _preprocess(self, patches: torch.Tensor) -> torch.Tensor:
        # The processor already scales pixels to [-1, 1]; the two temporal slices are averaged.
        pixels = patches.mean(dim=1)
        return pixels.permute(0, 3, 1, 2).contiguous().to(self.proj_out.weight.dtype)

    def _embed(self, patches: torch.Tensor, grid: list[list[int]]):
        merge, size = self.merge_size, self.patch_size
        pixels, positions, heights, widths = [], [], [], []
        for image, (_, height, width) in zip(patches.split([h * w for _, h, w in grid]), grid):
            native = image.reshape(height, width, 3, merge, size, merge, size)
            native = native.permute(0, 3, 1, 5, 4, 6, 2).reshape(-1, size * size * 3)
            pixels.extend((native.new_zeros((1, native.shape[1])), native))
            rows = torch.arange(height * width * merge * merge, device=patches.device)
            coordinates = torch.stack((rows // (width * merge), rows % (width * merge)), dim=-1)
            positions.extend((coordinates.new_zeros((1, 2)), coordinates))
            heights.append(rows.new_full((rows.numel() + 1,), height * merge))
            widths.append(rows.new_full((rows.numel() + 1,), width * merge))
        pixels = torch.cat(pixels)
        positions = torch.cat(positions)
        lengths = [h * w * merge * merge + 1 for _, h, w in grid]
        starts = torch.tensor([0, *lengths[:-1]], device=patches.device).cumsum(0)
        embeddings = self.backbone.embeddings
        hidden = embeddings(pixels)
        grid_size = self.config["posemb_grid_size"]
        hidden = hidden + interpolate_position_embeddings(
            embeddings.position_embedding.reshape(grid_size, grid_size, -1),
            positions[:, 0],
            positions[:, 1],
            torch.cat(heights),
            torch.cat(widths),
        )
        cls = (embeddings.class_embedding + embeddings.cls_position_embedding).to(hidden.dtype)
        hidden = hidden.index_put((starts,), cls.expand(len(lengths), -1))
        positions = positions + 1
        positions[starts] = 0
        return self.backbone.ln_pre(hidden), positions, lengths

    def _pool(self, hidden: torch.Tensor, grid: list[list[int]], lengths: list[int]) -> torch.Tensor:
        merge, width = self.merge_size, hidden.shape[-1]
        return torch.cat(
            [
                image[1:]
                .reshape(h, merge, w, merge, width)
                .permute(0, 2, 1, 3, 4)
                .reshape(h * w, merge * merge * width)
                for image, (_, h, w) in zip(hidden.split(lengths), grid)
            ]
        )

    def forward(self, patches: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """Encode images.

        Args:
            patches: ``[N, T, P * M, P * M, 3]`` merged patches scaled to [-1, 1] (mean and
                standard deviation 0.5, as the release's ``processor_config.json`` sets), one
                per decoder image token, where ``P`` is the native patch size and
                ``M`` the adapter merge size.
            grid_thw: ``[num_images, 3]`` merged-token grids with temporal size 1.

        Returns:
            Image embeddings ``[N, text_hidden_size]``.
        """
        side = self.patch_size * self.merge_size
        if patches.ndim != 5 or tuple(patches.shape[1:]) != (self.temporal_patch_size, side, side, 3):
            raise ValueError(f"Unexpected Inkling2 image patch shape {tuple(patches.shape)}")
        grid = grid_thw.tolist()
        if any(t != 1 or h <= 0 or w <= 0 for t, h, w in grid):
            raise ValueError("Inkling2 images require nonempty grids with temporal size 1")
        if sum(h * w for _, h, w in grid) != patches.shape[0]:
            raise ValueError("Image grids do not match the number of image patches")
        if not grid:
            return self.proj_out.weight.new_empty((0, self.proj_out.out_features))
        hidden, positions, lengths = self._embed(self._preprocess(patches), grid)
        for layer in self.backbone.layers:
            hidden = layer(hidden, positions, lengths)
        return self.proj_out(F.gelu(self.proj_in(self._pool(hidden, grid, lengths))))
