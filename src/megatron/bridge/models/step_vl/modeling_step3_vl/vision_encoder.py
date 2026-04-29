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

# Adapted from stepfun-ai/Step3-VL-10B vision_encoder.py (Apache 2.0 license)

from typing import Literal, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from transformers.activations import ACT2FN

from megatron.bridge.models.step_vl.modeling_step3_vl.configuration import (
    StepRoboticsVisionEncoderConfig,
)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


def _apply_rotary_emb(
    freqs: torch.Tensor,
    t: torch.Tensor,
    start_index: int = 0,
    scale: float = 1.0,
    seq_dim: int = -2,
) -> torch.Tensor:
    dtype = t.dtype
    if t.ndim == 3:
        seq_len = t.shape[seq_dim]
        freqs = freqs[-seq_len:]
    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim
    t_left = t[..., :start_index]
    t_mid = t[..., start_index:end_index]
    t_right = t[..., end_index:]
    t_mid = (t_mid * freqs.cos() * scale) + (_rotate_half(t_mid) * freqs.sin() * scale)
    return torch.cat((t_left, t_mid, t_right), dim=-1).type(dtype)


class EncoderRope2D(nn.Module):
    """Cacheable 2-D rotary positional embedding for the vision encoder."""

    def __init__(
        self,
        dim: int,
        max_grid_height: int,
        max_grid_width: int,
        use_cls_token: bool = False,
        theta: Union[int, float] = 10000,
        max_freq: int = 10,
        num_freqs: int = 1,
        theta_rescale_factor: float = 1.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_grid_height = max_grid_height
        self.max_grid_width = max_grid_width
        self.use_cls_token = use_cls_token
        self.theta = theta * theta_rescale_factor ** (dim / (dim - 2))
        self.max_freq = max_freq
        self.num_freqs = num_freqs
        cache = self._compute_2d_freqs()
        self.register_buffer("freqs_cache", cache, persistent=False)

    def _compute_inv_freq(self, base: Union[int, float], dim: int) -> torch.Tensor:
        return 1.0 / (base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    def _compute_freqs(self, t: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
        freqs = torch.einsum("..., f -> ... f", t.type(inv_freq.dtype), inv_freq)
        return repeat(freqs, "... n -> ... (n r)", r=2)

    def _compute_2d_freqs(self) -> torch.Tensor:
        grid_h = torch.arange(self.max_grid_height, dtype=torch.float)
        grid_w = torch.arange(self.max_grid_width, dtype=torch.float)
        if self.use_cls_token:
            grid_h = grid_h + 1
            grid_w = grid_w + 1
        inv_freq = self._compute_inv_freq(self.theta, self.dim // 2)
        freqs_h = self._compute_freqs(grid_h, inv_freq)[:, None].expand(
            self.max_grid_height, self.max_grid_width, -1
        )
        freqs_w = self._compute_freqs(grid_w, inv_freq)[None, :].expand(
            self.max_grid_height, self.max_grid_width, -1
        )
        freqs = torch.cat([freqs_w, freqs_h], dim=-1).reshape(
            self.max_grid_height * self.max_grid_width, -1
        )
        if self.use_cls_token:
            freqs = torch.cat([torch.zeros(1, freqs.shape[-1]), freqs], dim=0)
        return freqs[None, None, ...]

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, grid_hw: tuple[int, int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if grid_hw[0] != self.max_grid_height or grid_hw[1] != self.max_grid_width:
            rows = torch.arange(grid_hw[0], device=q.device).view(-1, 1)
            cols = torch.arange(grid_hw[1], device=q.device).view(1, -1)
            positions = (rows * self.max_grid_width + cols).reshape(-1).long()
            if self.use_cls_token:
                positions = torch.cat([torch.zeros(1, device=q.device, dtype=torch.long), positions + 1])
            freqs = self.freqs_cache.index_select(2, positions)
        else:
            freqs = self.freqs_cache
        q = _apply_rotary_emb(freqs, q)
        k = _apply_rotary_emb(freqs, k)
        return q, k


class EncoderLayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), init_values))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states * self.gamma


class EncoderMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str) -> None:
        super().__init__()
        self.c_fc = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.act_fn = ACT2FN[hidden_act]
        self.c_proj = nn.Linear(intermediate_size, hidden_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.c_proj(self.act_fn(self.c_fc(hidden_states)))


class EncoderVisionAttention(nn.Module):
    """Multi-head self-attention with optional 2-D RoPE."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        max_grid_height: int,
        max_grid_width: int,
        use_cls_token: bool = False,
        use_rope2d: bool = True,
        rope_theta: Union[int, float] = 10000,
        rope_max_freq: int = 10,
        rope_num_freqs: int = 1,
        rope_theta_rescale_factor: float = 1.0,
        rope_freqs_for: Literal["lang", "pixel", "constant"] = "lang",
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads}).")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.in_proj_weight = nn.Parameter(torch.zeros(hidden_size * 3, hidden_size))
        self.in_proj_bias = nn.Parameter(torch.zeros(hidden_size * 3))
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)

        self.rope: Optional[EncoderRope2D] = None
        if use_rope2d:
            self.rope = EncoderRope2D(
                dim=self.head_dim,
                max_grid_height=max_grid_height,
                max_grid_width=max_grid_width,
                use_cls_token=use_cls_token,
                theta=rope_theta,
                max_freq=rope_max_freq,
                num_freqs=rope_num_freqs,
                theta_rescale_factor=rope_theta_rescale_factor,
            )

    def forward(self, hidden_states: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        qkv = F.linear(hidden_states, self.in_proj_weight, self.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        if self.rope is not None:
            q, k = self.rope(q, k, grid_hw=grid_hw)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=False, scale=self.scale)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, self.num_heads * self.head_dim)
        return self.out_proj(attn_output)


class EncoderVisionBlock(nn.Module):
    """Single ViT transformer block (pre-norm attention + pre-norm MLP with LayerScale)."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        hidden_act: str,
        layer_norm_eps: float,
        ls_init_value: Optional[float] = None,
        max_grid_height: Optional[int] = None,
        max_grid_width: Optional[int] = None,
        use_cls_token: bool = False,
        use_rope2d: bool = True,
        rope_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        rope_kwargs = rope_kwargs or {}
        self.attn = EncoderVisionAttention(
            hidden_size,
            num_heads,
            max_grid_height=max_grid_height,
            max_grid_width=max_grid_width,
            use_cls_token=use_cls_token,
            use_rope2d=use_rope2d,
            **rope_kwargs,
        )
        self.ln_1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        intermediate = int(hidden_size * mlp_ratio)
        self.mlp = EncoderMLP(hidden_size, intermediate, hidden_act)
        self.ls_1 = EncoderLayerScale(hidden_size, ls_init_value)
        self.ls_2 = EncoderLayerScale(hidden_size, ls_init_value)

    def forward(self, hidden_states: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.attn(self.ln_1(hidden_states), grid_hw=grid_hw)
        hidden_states = residual + self.ls_1(hidden_states)
        residual = hidden_states
        hidden_states = self.mlp(self.ln_2(hidden_states))
        hidden_states = residual + self.ls_2(hidden_states)
        return hidden_states


class EncoderVisionTransformer(nn.Module):
    """Stack of ViT encoder blocks."""

    def __init__(
        self,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        hidden_act: str,
        layer_norm_eps: float,
        ls_init_value: Optional[float] = None,
        max_grid_height: Optional[int] = None,
        max_grid_width: Optional[int] = None,
        use_cls_token: bool = False,
        use_rope2d: bool = True,
        rope_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.layers = depth
        rope_kwargs = rope_kwargs or {}
        self.resblocks = nn.ModuleList(
            [
                EncoderVisionBlock(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    hidden_act,
                    layer_norm_eps,
                    max_grid_height=max_grid_height,
                    max_grid_width=max_grid_width,
                    use_cls_token=use_cls_token,
                    use_rope2d=use_rope2d,
                    ls_init_value=ls_init_value,
                    rope_kwargs=rope_kwargs,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, hidden_states: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
        for block in self.resblocks:
            hidden_states = block(hidden_states, grid_hw=grid_hw)
        return hidden_states


class StepRoboticsVisionEncoder(nn.Module):
    """
    Step3-VL vision encoder.

    Implements a custom ViT with 2-D RoPE, LayerScale residuals, and two
    strided Conv2d downsampler layers (vit_downsampler1 / vit_downsampler2).
    The downsampler weights are kept inside this module so that all vision
    parameters live under a single ``vision_model.**`` namespace.
    """

    def __init__(self, config: StepRoboticsVisionEncoderConfig) -> None:
        super().__init__()
        self.config = config

        self.hidden_size = config.width
        self.num_heads = config.heads
        self.num_hidden_layers = config.layers
        self.patch_size = config.patch_size
        self.image_size = config.image_size
        self.use_cls_token = getattr(config, "use_cls_token", False)
        self.use_rope2d = getattr(config, "use_rope2d", True)
        self.use_abs_posemb = getattr(config, "use_abs_posemb", True)
        self.layer_norm_eps = config.layer_norm_eps
        self.mlp_ratio = getattr(config, "mlp_ratio", 8960 / 1536)
        self.ls_init_value = getattr(config, "ls_init_value", None)
        self.hidden_act = config.hidden_act
        self.use_ln_pre = getattr(config, "use_ln_pre", True)
        self.use_ln_post = getattr(config, "use_ln_post", False)

        self.conv1 = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.ln_pre = (
            nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps) if self.use_ln_pre else nn.Identity()
        )
        self.ln_post = (
            nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps) if self.use_ln_post else nn.Identity()
        )

        grid_size = self.image_size // self.patch_size
        self.base_grid = (grid_size, grid_size)

        if self.use_cls_token:
            self.class_embedding = nn.Parameter(torch.randn(self.hidden_size) * (self.hidden_size**-0.5))
        else:
            self.class_embedding = None

        if self.use_abs_posemb:
            self.posemb_grid_size = self.image_size // self.patch_size
            self.positional_embedding = nn.Parameter(
                (self.hidden_size**-0.5)
                * torch.randn(
                    int(self.use_cls_token) + self.posemb_grid_size**2,
                    self.hidden_size,
                )
            )

        self.transformer = EncoderVisionTransformer(
            embed_dim=self.hidden_size,
            depth=self.num_hidden_layers,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
            ls_init_value=self.ls_init_value,
            max_grid_height=self.base_grid[0],
            max_grid_width=self.base_grid[1],
            use_cls_token=self.use_cls_token,
            use_rope2d=self.use_rope2d,
            rope_kwargs={
                "rope_theta": getattr(config, "rope_theta", 10000),
                "rope_max_freq": getattr(config, "rope_max_freq", 10),
                "rope_num_freqs": getattr(config, "rope_num_freqs", 1),
                "rope_theta_rescale_factor": getattr(config, "rope_theta_rescale_factor", 1.0),
                "rope_freqs_for": getattr(config, "rope_freqs_for", "lang"),
            },
        )

        # Spatial downsampler: 1536 → 3072 → 6144 via stride-2 Conv2d
        self.vit_downsampler1 = nn.Conv2d(self.hidden_size, self.hidden_size * 2, kernel_size=3, stride=2, padding=1)
        self.vit_downsampler2 = nn.Conv2d(
            self.hidden_size * 2, self.hidden_size * 4, kernel_size=3, stride=2, padding=1
        )

    def _sample_abs_posemb(self, grid_h: int, grid_w: int) -> torch.Tensor:
        if self.posemb_grid_size == grid_h and self.posemb_grid_size == grid_w:
            return self.positional_embedding[None, ...]
        pos_embed = self.positional_embedding
        if self.use_cls_token:
            cls_embed, pos_embed = pos_embed[:1], pos_embed[1:]
        pos_embed = (
            pos_embed.reshape(1, self.posemb_grid_size, self.posemb_grid_size, -1).permute(0, 3, 1, 2).contiguous()
        )
        pos_embed = F.interpolate(pos_embed, size=(grid_h, grid_w), mode="bilinear", align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(-1, self.hidden_size)
        if self.use_cls_token:
            pos_embed = torch.cat([cls_embed, pos_embed], dim=0)
        return pos_embed[None, ...]

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, C, H, W)

        Returns:
            hidden_states: (B, Gh*Gw, D) patch embeddings after all transformer blocks.
        """
        bsz, _, height, width = pixel_values.shape
        grid_h, grid_w = height // self.patch_size, width // self.patch_size

        hidden_state = self.conv1(pixel_values).flatten(2).transpose(1, 2)  # (B, Gh*Gw, D)

        if self.use_cls_token:
            cls_token = self.class_embedding.view(1, 1, -1).expand(bsz, -1, -1)
            hidden_state = torch.cat([cls_token, hidden_state], dim=1)

        if self.use_abs_posemb:
            hidden_state = hidden_state + self._sample_abs_posemb(grid_h, grid_w)

        hidden_state = self.ln_pre(hidden_state)
        hidden_state = self.transformer(hidden_state, grid_hw=(grid_h, grid_w))

        if self.use_ln_post:
            hidden_state = self.ln_post(hidden_state)

        if self.use_cls_token:
            hidden_state = hidden_state[:, 1:, :]

        return hidden_state
