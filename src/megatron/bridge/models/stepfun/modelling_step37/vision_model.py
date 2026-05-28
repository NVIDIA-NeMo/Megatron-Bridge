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

"""Step3.7 vision tower (Perception-Encoder G/14 + downsamplers).

Module names mirror ``vision_model.*`` in the HF ``Step37Model``
checkpoint so safetensors weights can be loaded by direct AutoMapping:

    vision_model.conv1.weight
    vision_model.ln_pre.{weight,bias}
    vision_model.positional_embedding
    vision_model.transformer.resblocks.{N}.attn.{in_proj_weight,in_proj_bias,
                                                  out_proj.{weight,bias}}
    vision_model.transformer.resblocks.{N}.ln_{1,2}.{weight,bias}
    vision_model.transformer.resblocks.{N}.ls_{1,2}.gamma
    vision_model.transformer.resblocks.{N}.mlp.{c_fc,c_proj}.{weight,bias}
    vision_model.vit_downsampler{1,2}.{weight,bias}
"""

from __future__ import annotations

import torch
import torch.nn as nn

from megatron.bridge.models.stepfun.modelling_step37.utils import (
    EncoderVisionTransformer,
)


class Step37VisionModel(nn.Module):
    """Perception-Encoder G/14 vision tower used by Step3.7.

    The module layout and parameter names match the HF
    ``StepRoboticsVisionEncoder`` checkpoint, which is what makes the
    Megatron-Bridge weight loader a direct AutoMapping for every vision
    parameter. The two ``vit_downsampler`` convolutions live on this module
    (matching the HF safetensors); ``forward`` runs the whole PE-G/14 trunk
    plus both downsamplers and returns ``[N, P', output_dim]`` in one call.
    The final ``vit_large_projector`` linear is owned by
    :class:`ImageInsertEmbedding` (in ``image_insert_embedding.py``) and is
    applied during the embedding/fusion step, not in the vision tower.
    """

    def __init__(self, vision_config):
        super().__init__()
        self.config = vision_config

        self.hidden_size = vision_config.width
        self.num_heads = vision_config.heads
        self.num_hidden_layers = vision_config.layers
        self.patch_size = vision_config.patch_size
        self.image_size = vision_config.image_size
        self.use_cls_token = getattr(vision_config, "use_cls_token", False)
        self.use_rope2d = getattr(vision_config, "use_rope2d", True)
        self.use_abs_posemb = getattr(vision_config, "use_abs_posemb", True)
        self.layer_norm_eps = vision_config.layer_norm_eps
        self.mlp_ratio = getattr(vision_config, "mlp_ratio", 8960 / 1536)
        self.ls_init_value = getattr(vision_config, "ls_init_value", None)
        self.hidden_act = vision_config.hidden_act
        self.use_ln_pre = getattr(vision_config, "use_ln_pre", False)
        self.use_ln_post = getattr(vision_config, "use_ln_post", True)

        self.conv1 = nn.Conv2d(
            in_channels=vision_config.num_channels,
            out_channels=self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.ln_pre = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps) if self.use_ln_pre else nn.Identity()
        self.ln_post = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps) if self.use_ln_post else nn.Identity()

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
                "rope_theta": getattr(vision_config, "rope_theta", 10000),
                "rope_max_freq": getattr(vision_config, "rope_max_freq", 10),
                "rope_num_freqs": getattr(vision_config, "rope_num_freqs", 1),
                "rope_theta_rescale_factor": getattr(vision_config, "rope_theta_rescale_factor", 1.0),
                "rope_freqs_for": getattr(vision_config, "rope_freqs_for", "lang"),
            },
        )

        self.vit_downsampler1 = nn.Conv2d(
            self.hidden_size,
            self.hidden_size * 2,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.vit_downsampler2 = nn.Conv2d(
            self.hidden_size * 2,
            self.hidden_size * 4,
            kernel_size=3,
            stride=2,
            padding=1,
        )

    def sample_abs_posemb(self, grid_h: int, grid_w: int):
        if self.posemb_grid_size == grid_h and self.posemb_grid_size == grid_w:
            return self.positional_embedding[None, ...]

        import torch.nn.functional as F  # local to avoid top-level conflict

        pos_embed = self.positional_embedding
        if self.use_cls_token:
            cls_token_embed, pos_embed = pos_embed[:1], pos_embed[1:]

        pos_embed = (
            pos_embed.reshape(1, self.posemb_grid_size, self.posemb_grid_size, -1).permute(0, 3, 1, 2).contiguous()
        )
        pos_embed = F.interpolate(
            pos_embed,
            size=(grid_h, grid_w),
            mode="bilinear",
            align_corners=False,
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(-1, self.hidden_size)

        if self.use_cls_token:
            pos_embed = torch.cat([cls_token_embed, pos_embed], dim=0)

        return pos_embed[None, ...]

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run the PE-G/14 trunk **+ both downsamplers** in one call.

        Steps: conv1 patchify → optional CLS → optional abs-pos-emb →
        ``ln_pre`` → 47×VisionBlock → optional ``ln_post`` → drop CLS →
        reshape to spatial → ``vit_downsampler1`` (3×3, stride 2) →
        ``vit_downsampler2`` (3×3, stride 2) → flatten + transpose.

        Args:
            pixel_values: float tensor of shape ``[B, C, H, W]`` with
                ``H = W = image_size`` (728 for the released checkpoint).

        Returns:
            Tensor of shape ``[B, P', output_dim]`` (e.g. ``[B, 169, 6144]``
            for 728² inputs through the released checkpoint). ``P'`` is
            ``(Gh/4)*(Gw/4)`` — the spatial grid after two stride-2
            downsamplers — and ``output_dim`` is ``vit_downsampler2``'s
            output channel count.
        """
        bsz, _, height, width = pixel_values.shape
        grid_h, grid_w = height // self.patch_size, width // self.patch_size

        hidden_state = self.conv1(pixel_values)  # [B, D, Gh, Gw]
        hidden_state = hidden_state.flatten(2).transpose(1, 2)  # [B, Gh*Gw, D]

        if self.use_cls_token:
            cls_token = self.class_embedding.view(1, 1, -1).expand(bsz, -1, -1)
            hidden_state = torch.cat([cls_token, hidden_state], dim=1)

        if self.use_abs_posemb:
            pos_emb = self.sample_abs_posemb(grid_h, grid_w)
            hidden_state = hidden_state + pos_emb
        hidden_state = self.ln_pre(hidden_state)
        hidden_state = self.transformer(hidden_state, grid_hw=(grid_h, grid_w))

        if self.use_ln_post:
            hidden_state = self.ln_post(hidden_state)

        if self.use_cls_token:
            hidden_state = hidden_state[:, 1:, :]

        # Spatial reshape + 2× downsampler, producing the final
        # ``[B, P', output_dim]`` image features in a single forward call.
        B, P = hidden_state.shape[:2]
        HW = int(P**0.5)
        image_features = hidden_state.permute(0, 2, 1).view(B, -1, HW, HW)
        image_features = self.vit_downsampler1(image_features)
        image_features = self.vit_downsampler2(image_features)

        B, C, HW, _ = image_features.shape
        image_features = image_features.view(B, -1, HW * HW).permute(0, 2, 1)
        return image_features
