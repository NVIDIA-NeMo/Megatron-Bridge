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

"""Vision-encoder building blocks for Step3.7.

The module mirrors the upstream HuggingFace ``vision_encoder.py`` shipped
inside ``stepfun-ai/step3p7_flash_bf16``: a 2D-RoPE Perception-Encoder G/14
ViT with LayerScale-gated residuals. All attribute names match the reference
implementation so the safetensors weights can be loaded by name with no
renaming required.
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate last dimension halves (used by RoPE)."""
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return x.reshape(*x.shape[:-2], -1)


def apply_rotary_emb(
    freqs: torch.Tensor,
    t: torch.Tensor,
    start_index: int = 0,
    scale: float = 1.0,
    seq_dim: int = -2,
) -> torch.Tensor:
    """Apply 2D rotary embeddings to queries / keys."""
    dtype = t.dtype

    if t.ndim == 3:
        seq_len = t.shape[seq_dim]
        freqs = freqs[-seq_len:]

    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim
    assert rot_dim <= t.shape[-1], f"feature dimension {t.shape[-1]} is too small for rot_dim {rot_dim}"

    t_left, t, t_right = (
        t[..., :start_index],
        t[..., start_index:end_index],
        t[..., end_index:],
    )
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    out = torch.cat((t_left, t, t_right), dim=-1)
    return out.type(dtype)


class EncoderRope2D(nn.Module):
    """Cacheable 2D rotary positional embedding (matches HF reference)."""

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
    ):
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
        freqs = 1.0 / (base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        return freqs

    def _compute_freqs(self, t: torch.Tensor, inv_freq: torch.Tensor):
        freqs = torch.einsum("..., f -> ... f", t.type(inv_freq.dtype), inv_freq)
        freqs = freqs.repeat_interleave(2, dim=-1)
        return freqs

    def _compute_2d_freqs(self) -> torch.Tensor:
        grid_h_range = torch.arange(self.max_grid_height, dtype=torch.float)
        grid_w_range = torch.arange(self.max_grid_width, dtype=torch.float)
        if self.use_cls_token:
            grid_h_range += 1
            grid_w_range += 1
        inv_freq = self._compute_inv_freq(self.theta, self.dim // 2)
        freqs_h = self._compute_freqs(grid_h_range, inv_freq)[:, None].expand(
            self.max_grid_height, self.max_grid_width, -1
        )
        freqs_w = self._compute_freqs(grid_w_range, inv_freq)[None, :].expand(
            self.max_grid_height, self.max_grid_width, -1
        )
        freqs = torch.cat([freqs_w, freqs_h], dim=-1).reshape(self.max_grid_height * self.max_grid_width, -1)
        if self.use_cls_token:
            freqs = torch.cat([torch.zeros(1, freqs.shape[-1]), freqs], dim=0)
        freqs = freqs[None, None, ...]
        return freqs

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        grid_hw: Tuple[int, int],
    ):
        if grid_hw[0] != self.max_grid_height or grid_hw[1] != self.max_grid_width:
            rows = torch.arange(grid_hw[0], device=q.device).view(-1, 1)
            cols = torch.arange(grid_hw[1], device=q.device).view(1, -1)
            positions = (rows * self.max_grid_width + cols).reshape(-1).to(torch.long)
            if self.use_cls_token:
                positions = torch.cat([torch.zeros(1, device=q.device), positions + 1], dim=0)
            freqs = self.freqs_cache.index_select(2, positions)
        else:
            freqs = self.freqs_cache
        q = apply_rotary_emb(freqs, q)
        k = apply_rotary_emb(freqs, k)
        return q, k


class EncoderLayerScale(nn.Module):
    """Per-channel residual scaling (γ stored as ``ls_{1,2}.gamma`` in the HF checkpoint)."""

    def __init__(self, dim: int, init_values: float):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), init_values))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:  # (B, L, D)
        return hidden_states * self.gamma


class EncoderMLP(nn.Module):
    """Feed-forward network used inside each transformer block."""

    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str):
        super().__init__()
        self.c_fc = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.act_fn = ACT2FN[hidden_act]
        self.c_proj = nn.Linear(intermediate_size, hidden_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.c_proj(self.act_fn(self.c_fc(hidden_states)))


class EncoderVisionAttention(nn.Module):
    """Multi-head self attention with optional 2D RoPE (matches HF reference)."""

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
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads}).")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        # Fused QKV — name matches the HF safetensors index entries
        # ``vision_model.transformer.resblocks.*.attn.in_proj_{weight,bias}``.
        self.in_proj_weight = nn.Parameter(torch.zeros(hidden_size * 3, hidden_size))
        self.in_proj_bias = nn.Parameter(torch.zeros(hidden_size * 3))
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)

        self.rope = None
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_hw: Tuple[int, int],
    ) -> torch.Tensor:
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
    """A single PE-G/14 Vision Transformer block."""

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
    ):
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_hw: Tuple[int, int],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        hidden_states = self.attn(hidden_states, grid_hw=grid_hw)
        hidden_states = residual + self.ls_1(hidden_states)

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.ls_2(hidden_states)
        return hidden_states


class EncoderVisionTransformer(nn.Module):
    """Stack of PE-G/14 encoder blocks (``vision_model.transformer.resblocks``)."""

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
    ):
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_hw: Tuple[int, int],
    ) -> torch.Tensor:
        for block in self.resblocks:
            hidden_states = block(hidden_states, grid_hw=grid_hw)
        return hidden_states
