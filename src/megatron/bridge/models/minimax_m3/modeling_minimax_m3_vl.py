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

"""MiniMax-M3 vision-language model components.

The released MiniMax-M3 checkpoint predates the native Transformers model
implementation. Keeping the small vision stack local lets the Bridge support
the full published checkpoint across the supported Transformers range while
preserving its legacy parameter namespace exactly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.tensor_parallel.mappings import scatter_to_sequence_parallel_region
from megatron.core.transformer.module import MegatronModule
from torch import nn

from megatron.bridge.utils.common_utils import (
    hook_hf_module_setattr_for_tp_grad_sync,
    slice_batch_for_context_parallel,
)


if TYPE_CHECKING:
    from megatron.core.packed_seq_params import PackedSeqParams


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    """Read a field from either a config object or a plain dictionary."""
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


class MiniMaxM3VisionPatchEmbeddings(nn.Module):
    """Conv3d patch embedding used by the MiniMax-M3 vision tower."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.patch_size = int(_config_value(config, "patch_size", 14))
        self.temporal_patch_size = int(_config_value(config, "temporal_patch_size", 2))
        self.num_channels = int(_config_value(config, "num_channels", 3))
        hidden_size = int(_config_value(config, "hidden_size", 1280))
        kernel_size = (self.temporal_patch_size, self.patch_size, self.patch_size)
        self.patch_embedding = nn.Conv3d(
            self.num_channels,
            hidden_size,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=False,
        )

        # The released checkpoint stores this one vision tensor in FP32. Mark it
        # so Bridge's mixed-precision wrapper does not truncate it to BF16.
        self.patch_embedding.float()
        self.patch_embedding._keep_in_float32_parameter_names = ("weight",)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Embed flattened image/video patches."""
        pixel_values = pixel_values.reshape(
            -1,
            self.num_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        hidden_states = self.patch_embedding(pixel_values.to(dtype=self.patch_embedding.weight.dtype))
        return hidden_states.reshape(-1, hidden_states.shape[1])


class MiniMaxM3Vision3DRotaryEmbedding(nn.Module):
    """Build MiniMax-M3's temporal/height/width rotary embeddings."""

    def __init__(self, head_dim: int, *, theta: float, spatial_merge_size: int) -> None:
        super().__init__()
        rope_dims = 2 * (head_dim // 2)
        self.axis_dim = 2 * ((rope_dims // 3) // 2)
        self.theta = theta
        self.spatial_merge_size = spatial_merge_size

    def forward(
        self,
        grid_thw: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cosine and sine tables in spatial-merge patch order."""
        merge_size = self.spatial_merge_size
        coordinates = []
        for temporal, height, width in grid_thw.tolist():
            if height % merge_size or width % merge_size:
                raise ValueError(
                    "MiniMax-M3 vision grid height and width must be divisible by "
                    f"spatial_merge_size={merge_size}, got {(temporal, height, width)}."
                )
            height_ids = torch.arange(height).unsqueeze(1).expand(-1, width)
            height_ids = (
                height_ids.reshape(height // merge_size, merge_size, width // merge_size, merge_size)
                .permute(0, 2, 1, 3)
                .flatten()
            )
            width_ids = torch.arange(width).unsqueeze(0).expand(height, -1)
            width_ids = (
                width_ids.reshape(height // merge_size, merge_size, width // merge_size, merge_size)
                .permute(0, 2, 1, 3)
                .flatten()
            )
            temporal_ids = torch.arange(temporal).repeat_interleave(height * width)
            coordinates.append(
                torch.stack(
                    [
                        temporal_ids,
                        height_ids.repeat(temporal),
                        width_ids.repeat(temporal),
                    ],
                    dim=-1,
                )
            )

        if not coordinates:
            raise ValueError("MiniMax-M3 vision input requires at least one image/video grid.")
        coordinates_tensor = torch.cat(coordinates).to(device=device, dtype=torch.float32)
        inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, self.axis_dim, 2, dtype=torch.float32, device=device) / self.axis_dim)
        )
        frequencies = torch.cat(
            [coordinates_tensor[:, axis : axis + 1] * inv_freq for axis in range(3)],
            dim=-1,
        )
        embeddings = torch.cat([frequencies, frequencies], dim=-1)
        return embeddings.cos().to(dtype=dtype), embeddings.sin().to(dtype=dtype)


def _rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    first, second = hidden_states.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_vision_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rotary_dim = cosine.shape[-1]
    cosine = cosine[None, :, None, :]
    sine = sine[None, :, None, :]
    query_rotary, query_pass = query[..., :rotary_dim], query[..., rotary_dim:]
    key_rotary, key_pass = key[..., :rotary_dim], key[..., rotary_dim:]
    query_rotary = query_rotary * cosine + _rotate_half(query_rotary) * sine
    key_rotary = key_rotary * cosine + _rotate_half(key_rotary) * sine
    return (
        torch.cat((query_rotary, query_pass), dim=-1),
        torch.cat((key_rotary, key_pass), dim=-1),
    )


class MiniMaxM3VisionAttention(nn.Module):
    """CLIP-style self-attention with MiniMax-M3 3D RoPE."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.hidden_size = int(_config_value(config, "hidden_size", 1280))
        self.num_heads = int(_config_value(config, "num_attention_heads", 16))
        if self.hidden_size % self.num_heads:
            raise ValueError(f"Vision hidden_size={self.hidden_size} must be divisible by num_heads={self.num_heads}.")
        self.head_dim = self.hidden_size // self.num_heads
        self.dropout = float(_config_value(config, "attention_dropout", 0.0))
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Apply non-causal vision self-attention."""
        batch_size, sequence_length, _ = hidden_states.shape
        target_shape = (batch_size, sequence_length, self.num_heads, self.head_dim)
        query = self.q_proj(hidden_states).reshape(target_shape)
        key = self.k_proj(hidden_states).reshape(target_shape)
        value = self.v_proj(hidden_states).reshape(target_shape).transpose(1, 2)
        query, key = _apply_vision_rope(query, key, *position_embeddings)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            scale=self.head_dim**-0.5,
        )
        output = output.transpose(1, 2).reshape(batch_size, sequence_length, self.hidden_size)
        return self.out_proj(output)


class MiniMaxM3VisionMLP(nn.Module):
    """GELU feed-forward block for the vision tower."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        hidden_size = int(_config_value(config, "hidden_size", 1280))
        intermediate_size = int(_config_value(config, "intermediate_size", 5120))
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply the vision MLP."""
        return self.fc2(F.gelu(self.fc1(hidden_states)))


class MiniMaxM3VisionEncoderLayer(nn.Module):
    """Pre-norm MiniMax-M3 vision transformer layer."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        hidden_size = int(_config_value(config, "hidden_size", 1280))
        epsilon = float(_config_value(config, "layer_norm_eps", 1e-5))
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=epsilon)
        self.self_attn = MiniMaxM3VisionAttention(config)
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=epsilon)
        self.mlp = MiniMaxM3VisionMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Apply attention and MLP residual blocks."""
        hidden_states = hidden_states + self.self_attn(self.layer_norm1(hidden_states), position_embeddings)
        return hidden_states + self.mlp(self.layer_norm2(hidden_states))


class MiniMaxM3VisionEncoder(nn.Module):
    """Container preserving the checkpoint's ``encoder.layers`` namespace."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        num_layers = int(_config_value(config, "num_hidden_layers", 32))
        self.layers = nn.ModuleList([MiniMaxM3VisionEncoderLayer(config) for _ in range(num_layers)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Run all vision transformer layers."""
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings)
        return hidden_states


class MiniMaxM3VisionModel(nn.Module):
    """MiniMax-M3 Conv3d + 3D-RoPE vision encoder."""

    def __init__(self, config: Any, *, spatial_merge_size: int) -> None:
        super().__init__()
        hidden_size = int(_config_value(config, "hidden_size", 1280))
        num_heads = int(_config_value(config, "num_attention_heads", 16))
        epsilon = float(_config_value(config, "layer_norm_eps", 1e-5))
        rope_parameters = _config_value(config, "rope_parameters", {}) or {}
        rope_theta = float(
            _config_value(
                config,
                "rope_theta",
                rope_parameters.get("rope_theta", 10000.0) if isinstance(rope_parameters, dict) else 10000.0,
            )
        )
        self.embeddings = MiniMaxM3VisionPatchEmbeddings(config)
        # Keep the checkpoint's historical typo in the module name.
        self.pre_layrnorm = nn.LayerNorm(hidden_size, eps=epsilon)
        self.encoder = MiniMaxM3VisionEncoder(config)
        self.rotary_embedding = MiniMaxM3Vision3DRotaryEmbedding(
            hidden_size // num_heads,
            theta=rope_theta,
            spatial_merge_size=spatial_merge_size,
        )

    def forward(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> torch.Tensor:
        """Return unprojected patch features with shape ``[1, patches, hidden]``."""
        hidden_states = self.embeddings(pixel_values).to(dtype=self.pre_layrnorm.weight.dtype)
        position_embeddings = self.rotary_embedding(
            image_grid_thw,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        hidden_states = self.pre_layrnorm(hidden_states).unsqueeze(0)
        return self.encoder(hidden_states, position_embeddings)


class MiniMaxM3VisionTower(nn.Module):
    """Wrapper preserving the checkpoint's ``vision_tower.vision_model`` path."""

    def __init__(self, config: Any, *, spatial_merge_size: int) -> None:
        super().__init__()
        self.vision_model = MiniMaxM3VisionModel(config, spatial_merge_size=spatial_merge_size)

    def forward(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> torch.Tensor:
        """Encode image or video patches."""
        return self.vision_model(pixel_values, image_grid_thw)


class MiniMaxM3ProjectorMLP(nn.Module):
    """Biased GELU MLP with the released projector parameter names."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int, *, bias: bool) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(input_size, hidden_size, bias=bias)
        self.linear_2 = nn.Linear(hidden_size, output_size, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project vision features."""
        return self.linear_2(F.gelu(self.linear_1(hidden_states)))


class _MiniMaxM3CheckpointWeight(nn.Module):
    """Frozen checkpoint tensor with a conventional ``weight`` leaf."""

    def __init__(self, shape: tuple[int, ...], *, dtype: torch.dtype) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(shape, dtype=dtype), requires_grad=False)


class MiniMaxM3LightningIndexerState(nn.Module):
    """Checkpoint-only state for one Lightning Indexer layer.

    Megatron currently executes full causal attention instead of MiniMax-M3's
    block-sparse selection path. Keeping these tensors as frozen parameters
    makes checkpoint conversion lossless without adding trainable
    parameters or coupling the shared conversion path to the source checkpoint.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        index_n_heads = int(config.index_n_heads)
        index_head_dim = int(config.index_head_dim)
        dtype = _config_value(config, "params_dtype", torch.float32) or torch.float32

        self.q_proj = _MiniMaxM3CheckpointWeight(
            (index_n_heads * index_head_dim, hidden_size),
            dtype=dtype,
        )
        self.k_proj = _MiniMaxM3CheckpointWeight(
            (index_head_dim, hidden_size),
            dtype=dtype,
        )
        self.q_norm = _MiniMaxM3CheckpointWeight((index_head_dim,), dtype=dtype)
        self.k_norm = _MiniMaxM3CheckpointWeight((index_head_dim,), dtype=dtype)


class MiniMaxM3VLModel(MegatronModule):
    """MiniMax-M3 vision tower, projectors, and Megatron language model."""

    def __init__(
        self,
        config: Any,
        pre_process: bool = True,
        post_process: bool = True,
        vp_stage: int | None = None,
    ) -> None:
        super().__init__(config=config)
        self.pre_process = pre_process
        self.post_process = post_process
        self.vp_stage = vp_stage

        if pre_process:
            if config.vision_config is None:
                raise ValueError("MiniMax-M3 VLM construction requires vision_config.")
            vision_hidden_size = int(_config_value(config.vision_config, "hidden_size", 1280))
            self.vision_tower = MiniMaxM3VisionTower(
                config.vision_config,
                spatial_merge_size=config.spatial_merge_size,
            )
            self.multi_modal_projector = MiniMaxM3ProjectorMLP(
                vision_hidden_size,
                config.projector_hidden_size,
                config.hidden_size,
                bias=config.multimodal_projector_bias,
            )
            self.patch_merge_mlp = MiniMaxM3ProjectorMLP(
                config.hidden_size * (config.spatial_merge_size**2),
                config.projector_hidden_size,
                config.hidden_size,
                bias=config.multimodal_projector_bias,
            )
            for module in (self.vision_tower, self.multi_modal_projector, self.patch_merge_mlp):
                hook_hf_module_setattr_for_tp_grad_sync(module)

        self.language_model = config.provide_language_model(
            pre_process=pre_process,
            post_process=post_process,
            vp_stage=vp_stage,
        )
        sparse_layer_indices = set(config.lightning_indexer_layers)
        local_layers = getattr(getattr(self.language_model, "decoder", None), "layers", ())
        self.lightning_indexers = nn.ModuleDict()
        for layer in local_layers:
            layer_idx = int(layer.layer_number) - 1
            if layer_idx in sparse_layer_indices:
                self.lightning_indexers[str(layer_idx)] = MiniMaxM3LightningIndexerState(config)
        self.share_embeddings_and_output_weights = config.share_embeddings_and_output_weights
        self.shared_embedding_or_output_weight = self.language_model.shared_embedding_or_output_weight

    @property
    def decoder(self) -> nn.Module | None:
        """Expose the text decoder for Megatron-Core inference inspection."""
        return getattr(self.language_model, "decoder", None)

    def set_input_tensor(self, input_tensor: torch.Tensor | list[torch.Tensor]) -> None:
        """Set the pipeline input tensor on the language model."""
        self.language_model.set_input_tensor(input_tensor)

    def _project_vision(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.vision_tower(pixel_values, grid_thw).squeeze(0)
        hidden_states = self.multi_modal_projector(hidden_states)
        merge_factor = self.config.spatial_merge_size**2
        if hidden_states.shape[0] % merge_factor:
            raise ValueError(
                f"Vision patch count {hidden_states.shape[0]} must be divisible by merge factor {merge_factor}."
            )
        hidden_states = hidden_states.reshape(hidden_states.shape[0] // merge_factor, -1)
        return self.patch_merge_mlp(hidden_states)

    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """Encode and project image patches into language-model tokens."""
        return self._project_vision(pixel_values, image_grid_thw)

    def get_video_features(
        self,
        pixel_values_videos: torch.Tensor,
        video_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """Encode and project video patches into language-model tokens."""
        return self._project_vision(pixel_values_videos, video_grid_thw)

    @staticmethod
    def _scatter_features(
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor | None,
        *,
        token_id: int,
        features: torch.Tensor | None,
        modality: str,
    ) -> torch.Tensor:
        if features is None:
            return inputs_embeds
        if input_ids is None:
            raise ValueError(f"input_ids are required when scattering MiniMax-M3 {modality} features.")
        mask = (input_ids == token_id).unsqueeze(-1).expand_as(inputs_embeds)
        if inputs_embeds[mask].numel() != features.numel():
            token_count = int(mask[..., 0].sum().item())
            raise ValueError(
                f"MiniMax-M3 {modality} feature count does not match placeholder tokens: "
                f"{features.shape[0]} features for {token_count} tokens."
            )
        return inputs_embeds.masked_scatter(mask, features.to(inputs_embeds.device, inputs_embeds.dtype))

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        inference_context: BaseInferenceContext | None = None,
        runtime_gather_output: bool | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        extra_block_kwargs: dict[str, Any] | None = None,
        *,
        inference_params: BaseInferenceContext | None = None,
        loss_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run the multimodal forward path and return a CP-local mask when supplied."""
        # The Transformers processor emits this metadata, but MiniMax-M3's text
        # backbone uses standard 1D RoPE and does not consume token-type IDs.
        del mm_token_type_ids

        if self.pre_process:
            if inputs_embeds is None:
                if input_ids is None:
                    raise ValueError("MiniMax-M3 VLM forward requires input_ids or inputs_embeds.")
                inputs_embeds = self.language_model.embedding(input_ids=input_ids, position_ids=None)
                inputs_embeds = inputs_embeds.transpose(0, 1).contiguous()

            image_features = None
            if pixel_values is not None:
                if image_grid_thw is None:
                    raise ValueError("image_grid_thw is required with pixel_values.")
                image_features = self.get_image_features(pixel_values, image_grid_thw)
            video_features = None
            if pixel_values_videos is not None:
                if video_grid_thw is None:
                    raise ValueError("video_grid_thw is required with pixel_values_videos.")
                video_features = self.get_video_features(pixel_values_videos, video_grid_thw)

            inputs_embeds = self._scatter_features(
                inputs_embeds,
                input_ids,
                token_id=self.config.image_token_id,
                features=image_features,
                modality="image",
            )
            inputs_embeds = self._scatter_features(
                inputs_embeds,
                input_ids,
                token_id=self.config.video_token_id,
                features=video_features,
                modality="video",
            )
            inputs_embeds = inputs_embeds.transpose(0, 1).contiguous()

        inputs_embeds, labels, loss_mask, position_ids, attention_mask = slice_batch_for_context_parallel(
            inputs_embeds=inputs_embeds,
            labels=labels,
            loss_mask=loss_mask,
            position_ids=position_ids,
            attention_mask=attention_mask,
            packed_seq_params=packed_seq_params,
            pg_collection=self.config._pg_collection,
        )
        if self.config.sequence_parallel and inputs_embeds is not None:
            tp_group = self.config._pg_collection.tp if self.config._pg_collection is not None else None
            inputs_embeds = scatter_to_sequence_parallel_region(inputs_embeds, group=tp_group)

        output = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=inputs_embeds,
            labels=labels,
            loss_mask=loss_mask,
            inference_context=inference_context,
            runtime_gather_output=runtime_gather_output,
            packed_seq_params=packed_seq_params,
            extra_block_kwargs=extra_block_kwargs,
            inference_params=inference_params,
        )
        # Existing MiniMax-M3 text recipes use gpt_step, which owns the loss
        # mask and expects a tensor. VLM training passes the mask into the model
        # so vlm_step can consume the CP-local slice returned here.
        if loss_mask is not None:
            return output, loss_mask
        return output

    def freeze(
        self,
        *,
        freeze_language_model: bool,
        freeze_vision_model: bool,
        freeze_vision_projection: bool,
    ) -> None:
        """Freeze selected VLM components."""
        modules = []
        if freeze_language_model:
            modules.append(self.language_model)
        if freeze_vision_model and hasattr(self, "vision_tower"):
            modules.append(self.vision_tower)
        if freeze_vision_projection and hasattr(self, "multi_modal_projector"):
            modules.extend((self.multi_modal_projector, self.patch_merge_mlp))
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad = False


__all__ = [
    "MiniMaxM3LightningIndexerState",
    "MiniMaxM3VLModel",
    "MiniMaxM3VisionModel",
]
