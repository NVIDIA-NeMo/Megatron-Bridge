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

"""Megatron modeling components for Muse Glimmer."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core.extensions.transformer_engine import TENorm
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.backends import get_backend
from megatron.core.models.hybrid.hybrid_block import HybridStack, HybridStackSubmodules
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.mappings import scatter_to_sequence_parallel_region
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.transformer.utils import ensure_metadata_has_dp_cp_group
from megatron.core.utils import get_pg_rank
from torch import Tensor
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from transformer_engine.pytorch import LayerNorm as TELayerNorm

from megatron.bridge.models.gemma.modules import extend_instance
from megatron.bridge.models.muse_glimmer.muse_glimmer_config import (
    MuseGlimmerModelConfig,
    MuseGlimmerVisionModelConfig,
)
from megatron.bridge.utils.common_utils import (
    hook_hf_module_setattr_for_tp_grad_sync,
    slice_batch_for_context_parallel,
)


if TYPE_CHECKING:
    from megatron.core.inference.contexts import BaseInferenceContext
    from megatron.core.packed_seq_params import PackedSeqParams


class MuseGlimmerSelfAttention(SelfAttention):
    """MCore self-attention with Muse's residual-branch post norm."""

    def __init__(self, config: TransformerConfig, *args: Any, **kwargs: Any) -> None:
        super().__init__(config, *args, **kwargs)
        # TE RMSNorm always registers an affine weight, while Muse Q/K norms are
        # parameter-free; use native RMSNorm to preserve the HF checkpoint contract.
        self.q_layernorm = nn.RMSNorm(
            self.hidden_size_per_attention_head,
            eps=config.layernorm_epsilon,
            elementwise_affine=False,
        )
        self.k_layernorm = nn.RMSNorm(
            self.hidden_size_per_attention_head,
            eps=config.layernorm_epsilon,
            elementwise_affine=False,
        )
        self.post_layernorm = TENorm(
            config,
            config.hidden_size,
            eps=float(getattr(config, "post_norm_epsilon", 1e-8)),
        )

    def forward(self, *args: Any, **kwargs: Any) -> tuple[Tensor, Tensor | None]:
        output, bias = super().forward(*args, **kwargs)
        return self.post_layernorm(output), bias


class MuseGlimmerMLP(MLP):
    """MCore gated MLP with Muse's residual-branch post norm."""

    def __init__(
        self,
        config: TransformerConfig,
        submodules: MLPSubmodules,
        ffn_hidden_size: int | None = None,
        **kwargs: Any,
    ) -> None:
        ffn_hidden_size = config.ffn_hidden_size if ffn_hidden_size is None else ffn_hidden_size
        super().__init__(
            config=config,
            submodules=submodules,
            ffn_hidden_size=ffn_hidden_size,
            **kwargs,
        )
        self.post_layernorm = TENorm(
            config,
            config.hidden_size,
            eps=float(getattr(config, "post_norm_epsilon", 1e-8)),
        )

    def forward(self, *args: Any, **kwargs: Any) -> tuple[Tensor, Tensor | None]:
        output, bias = super().forward(*args, **kwargs)
        return self.post_layernorm(output), bias


def get_muse_glimmer_hybrid_stack_spec(config: TransformerConfig) -> ModuleSpec:
    """Build the native Hybrid stack spec used by the Muse decoder."""
    backend = get_backend(config.transformer_impl)
    return ModuleSpec(
        module=HybridStack,
        submodules=HybridStackSubmodules(
            attention_layer=ModuleSpec(
                module=TransformerLayer,
                submodules=TransformerLayerSubmodules(
                    input_layernorm=TENorm,
                    self_attention=ModuleSpec(
                        module=MuseGlimmerSelfAttention,
                        params={"attn_mask_type": AttnMaskType.causal},
                        submodules=SelfAttentionSubmodules(
                            linear_qkv=backend.column_parallel_linear(),
                            core_attention=backend.core_attention(),
                            linear_proj=backend.row_parallel_linear(),
                        ),
                    ),
                    self_attn_bda=get_bias_dropout_add,
                    pre_mlp_layernorm=TENorm,
                    mlp=functools.partial(
                        MuseGlimmerMLP.as_mlp_submodule,
                        submodules=MLPSubmodules(
                            linear_fc1=backend.column_parallel_linear(),
                            linear_fc2=backend.row_parallel_linear(),
                        ),
                    ),
                    mlp_bda=get_bias_dropout_add,
                ),
            ),
        ),
    )


class MuseGlimmerOutputLayerMixin(nn.Module):
    """Apply the Muse logit multiplier followed by tanh soft-capping."""

    def forward(self, *args: Any, **kwargs: Any) -> tuple[Tensor, Tensor | None]:
        output, bias = super().forward(*args, **kwargs)
        output = output * self.config.output_multiplier
        softcap = self.config.final_logit_softcapping
        if softcap:
            output = softcap * torch.tanh(output / softcap)
        return output, bias


def _vision_cu_seqlens(grid_thw: Tensor) -> Tensor:
    lengths = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
    return F.pad(lengths.cumsum(dim=0, dtype=torch.int32), (1, 0), value=0)


def _vision_position_ids(grid_thw: Tensor) -> Tensor:
    positions = []
    for temporal, height, width in grid_thw.tolist():
        rows, columns = torch.meshgrid(
            torch.arange(int(height), device=grid_thw.device),
            torch.arange(int(width), device=grid_thw.device),
            indexing="ij",
        )
        positions.append(torch.stack((columns.flatten(), rows.flatten()), dim=-1).repeat(int(temporal), 1) + 1)
    return torch.cat(positions, dim=0)


def _vision_window_index(
    grid_thw: Tensor,
    *,
    window_size: int,
) -> tuple[Tensor, Tensor]:
    indices = []
    cumulative_lengths = [0]
    offset = 0
    for temporal, height, width in grid_thw.tolist():
        temporal, height, width = int(temporal), int(height), int(width)
        index = torch.arange(temporal * height * width).reshape(temporal, height, width)
        pad_height = (-height) % window_size
        pad_width = (-width) % window_size
        padded = F.pad(index, (0, pad_width, 0, pad_height), value=-1)
        num_windows_height = (height + pad_height) // window_size
        num_windows_width = (width + pad_width) // window_size
        padded = padded.reshape(
            temporal,
            num_windows_height,
            window_size,
            num_windows_width,
            window_size,
        )
        padded = padded.permute(0, 1, 3, 2, 4).reshape(-1, window_size, window_size)
        lengths = (padded != -1).sum(dim=(1, 2))
        flattened = padded.flatten()
        indices.append(flattened[flattened != -1] + offset)
        cumulative_lengths.extend((lengths.cumsum(0) + cumulative_lengths[-1]).tolist())
        offset += temporal * height * width
    return (
        torch.cat(indices).to(device=grid_thw.device),
        torch.unique_consecutive(torch.tensor(cumulative_lengths, device=grid_thw.device, dtype=torch.int32)),
    )


def _bilinear_indices_and_weights(grid_thw: Tensor, side: int) -> tuple[Tensor, Tensor]:
    index_parts: list[list[Tensor]] = [[] for _ in range(4)]
    weight_parts: list[list[Tensor]] = [[] for _ in range(4)]
    for temporal, height, width in grid_thw.tolist():
        temporal, height, width = int(temporal), int(height), int(width)
        height_grid = (torch.arange(height, device=grid_thw.device).float() + 0.5) * (side / height) - 0.5
        width_grid = (torch.arange(width, device=grid_thw.device).float() + 0.5) * (side / width) - 0.5
        height_floor = torch.floor(height_grid).long()
        width_floor = torch.floor(width_grid).long()
        height_ceil = height_floor + 1
        width_ceil = width_floor + 1
        height_fraction = height_grid - height_floor.float()
        width_fraction = width_grid - width_floor.float()
        height_floor_valid = (height_floor >= 0) & (height_floor < side)
        height_ceil_valid = (height_ceil >= 0) & (height_ceil < side)
        width_floor_valid = (width_floor >= 0) & (width_floor < side)
        width_ceil_valid = (width_ceil >= 0) & (width_ceil < side)
        height_floor = height_floor.clamp(0, side - 1)
        height_ceil = height_ceil.clamp(0, side - 1)
        width_floor = width_floor.clamp(0, side - 1)
        width_ceil = width_ceil.clamp(0, side - 1)
        corner_indices = [
            (height_floor[:, None] * side + width_floor[None, :]).flatten(),
            (height_floor[:, None] * side + width_ceil[None, :]).flatten(),
            (height_ceil[:, None] * side + width_floor[None, :]).flatten(),
            (height_ceil[:, None] * side + width_ceil[None, :]).flatten(),
        ]
        corner_weights = [
            (
                (1 - height_fraction)[:, None]
                * (1 - width_fraction)[None, :]
                * (height_floor_valid[:, None] & width_floor_valid[None, :])
            ).flatten(),
            (
                (1 - height_fraction)[:, None]
                * width_fraction[None, :]
                * (height_floor_valid[:, None] & width_ceil_valid[None, :])
            ).flatten(),
            (
                height_fraction[:, None]
                * (1 - width_fraction)[None, :]
                * (height_ceil_valid[:, None] & width_floor_valid[None, :])
            ).flatten(),
            (
                height_fraction[:, None]
                * width_fraction[None, :]
                * (height_ceil_valid[:, None] & width_ceil_valid[None, :])
            ).flatten(),
        ]
        for corner in range(4):
            index_parts[corner].append(corner_indices[corner].repeat(temporal))
            weight_parts[corner].append(corner_weights[corner].repeat(temporal))
    return (
        torch.stack([torch.cat(part) for part in index_parts]),
        torch.stack([torch.cat(part) for part in weight_parts]),
    )


class MuseGlimmerVisionPatchEmbedder(nn.Module):
    """Linear patch embedding plus exact half-pixel bilinear position lookup."""

    def __init__(self, config: MuseGlimmerVisionModelConfig) -> None:
        super().__init__()
        if config.pos_emb_height != config.pos_emb_width:
            raise ValueError("Muse Glimmer currently requires a square learned vision position table.")
        patch_dimension = config.patch_temporal * 3 * config.patch_size**2
        self.patch_embedding = nn.Linear(patch_dimension, config.hidden_size, bias=False)
        self.position_embedding_table = nn.Embedding(
            config.pos_emb_height * config.pos_emb_width,
            config.hidden_size,
        )
        self.num_grid_per_side = config.pos_emb_height

    def forward(self, pixel_values: Tensor, grid_thw: Tensor) -> Tensor:
        embeddings = self.patch_embedding(pixel_values.to(self.patch_embedding.weight.dtype)).reshape(
            pixel_values.shape[0], -1
        )
        indices, weights = _bilinear_indices_and_weights(grid_thw, self.num_grid_per_side)
        position_embeddings = (self.position_embedding_table(indices) * weights[:, :, None]).sum(dim=0)
        return embeddings + position_embeddings.to(embeddings.dtype)


def _rotate_half(hidden_states: Tensor) -> Tensor:
    first, second = hidden_states.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class MuseGlimmerVisionRotaryEmbedding(nn.Module):
    """Muse 2-D RoPE with [W, H, W, H] frequency interleaving."""

    def __init__(self, config: MuseGlimmerVisionModelConfig) -> None:
        super().__init__()
        head_dim = config.hidden_size // config.num_attention_heads
        spatial_dim = head_dim // 2
        inverse_frequency = 1.0 / (
            config.rotary_base ** (torch.arange(0, spatial_dim, 2, dtype=torch.float32) / spatial_dim)
        )
        self.register_buffer("inv_freq", inverse_frequency, persistent=False)

    def forward(self, hidden_states: Tensor, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        del hidden_states
        width_frequency = torch.outer(position_ids[:, 0].float(), self.inv_freq.float())
        height_frequency = torch.outer(position_ids[:, 1].float(), self.inv_freq.float())
        frequency = torch.cat(
            [width_frequency, height_frequency, width_frequency, height_frequency],
            dim=-1,
        )
        return frequency.cos(), frequency.sin()


class MuseGlimmerVisionAttention(nn.Module):
    """Bidirectional packed vision attention matching the HF parameter layout."""

    def __init__(self, config: MuseGlimmerVisionModelConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scaling = self.head_dim**-0.5
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

    def forward(
        self,
        hidden_states: Tensor,
        cu_seqlens: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
    ) -> Tensor:
        sequence_length = hidden_states.shape[0]
        query = self.q_proj(hidden_states).reshape(sequence_length, self.num_heads, self.head_dim)
        key = self.k_proj(hidden_states).reshape(sequence_length, self.num_heads, self.head_dim)
        value = self.v_proj(hidden_states).reshape(sequence_length, self.num_heads, self.head_dim)
        cosine, sine = position_embeddings
        query_dtype, key_dtype = query.dtype, key.dtype
        cosine = cosine[:, None, :].float()
        sine = sine[:, None, :].float()
        query = ((query.float() * cosine) + (_rotate_half(query.float()) * sine)).to(query_dtype)
        key = ((key.float() * cosine) + (_rotate_half(key.float()) * sine)).to(key_dtype)

        outputs = []
        boundaries = cu_seqlens.tolist()
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            q = query[start:end].transpose(0, 1).unsqueeze(0)
            k = key[start:end].transpose(0, 1).unsqueeze(0)
            v = value[start:end].transpose(0, 1).unsqueeze(0)
            attention_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
            attention_weights = F.softmax(attention_weights, dim=-1, dtype=torch.float32).to(q.dtype)
            output = torch.matmul(attention_weights, v)
            outputs.append(output.squeeze(0).transpose(0, 1))
        return self.proj(torch.cat(outputs, dim=0).reshape(sequence_length, -1))


class MuseGlimmerVisionMLP(nn.Module):
    """Muse vision feed-forward network."""

    def __init__(self, config: MuseGlimmerVisionModelConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.fc2(F.gelu(self.fc1(hidden_states)))


class MuseGlimmerVisionEncoderLayer(nn.Module):
    """Pre-normalized Muse vision transformer layer."""

    def __init__(self, config: MuseGlimmerVisionModelConfig, transformer_config: TransformerConfig) -> None:
        super().__init__()
        norm_device = (
            "cpu"
            if transformer_config.use_cpu_initialization
            else "meta"
            if transformer_config.init_model_with_meta_device
            else torch.cuda.current_device()
        )
        norm_kwargs = {
            "normalized_shape": config.hidden_size,
            "eps": config.layer_norm_epsilon,
            "sequence_parallel": False,
            "zero_centered_gamma": False,
            "device": norm_device,
            "dtype": transformer_config.params_dtype,
        }
        self.norm1 = TELayerNorm(**norm_kwargs)
        self.attn = MuseGlimmerVisionAttention(config)
        self.norm2 = TELayerNorm(**norm_kwargs)
        self.mlp = MuseGlimmerVisionMLP(config)

    def forward(
        self,
        hidden_states: Tensor,
        cu_seqlens: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
    ) -> Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), cu_seqlens, position_embeddings)
        return hidden_states + self.mlp(self.norm2(hidden_states))


class MuseGlimmerVisionModel(nn.Module):
    """Muse vision tower with 3:1 window/full attention and 2x2 pixel shuffle."""

    def __init__(
        self,
        config: MuseGlimmerVisionModelConfig,
        transformer_config: TransformerConfig,
        *,
        recompute_layers: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.recompute_layers = recompute_layers
        self.patch_embedder = MuseGlimmerVisionPatchEmbedder(config)
        self.rotary_emb = MuseGlimmerVisionRotaryEmbedding(config)
        norm_device = (
            "cpu"
            if transformer_config.use_cpu_initialization
            else "meta"
            if transformer_config.init_model_with_meta_device
            else torch.cuda.current_device()
        )
        norm_kwargs = {
            "normalized_shape": config.hidden_size,
            "eps": config.layer_norm_epsilon,
            "sequence_parallel": False,
            "zero_centered_gamma": False,
            "device": norm_device,
            "dtype": transformer_config.params_dtype,
        }
        self.ln_pre = TELayerNorm(**norm_kwargs)
        self.layers = nn.ModuleList(
            [MuseGlimmerVisionEncoderLayer(config, transformer_config) for _ in range(config.num_hidden_layers)]
        )
        self.ln_post = TELayerNorm(**norm_kwargs)

    def _pixel_shuffle(self, hidden_states: Tensor, grid_thw: Tensor) -> Tensor:
        outputs = []
        offset = 0
        merge_size = self.config.merge_size
        hidden_size = hidden_states.shape[-1]
        for temporal, height, width in grid_thw.tolist():
            temporal, height, width = int(temporal), int(height), int(width)
            if height % merge_size or width % merge_size:
                raise ValueError(
                    f"Vision grid {(temporal, height, width)} must be divisible by merge_size={merge_size}."
                )
            token_count = temporal * height * width
            chunk = hidden_states[offset : offset + token_count]
            permutation = torch.arange(height * width, device=hidden_states.device)
            permutation = (
                permutation.reshape(height // merge_size, merge_size, width // merge_size, merge_size)
                .permute(0, 2, 1, 3)
                .reshape(-1)
            )
            frame_offsets = torch.arange(temporal, device=hidden_states.device)[:, None] * height * width
            permutation = (permutation[None, :] + frame_offsets).reshape(-1)
            merged_tokens = temporal * (height // merge_size) * (width // merge_size)
            chunk = chunk[permutation].reshape(merged_tokens, merge_size * merge_size, hidden_size)
            outputs.append(chunk.permute(0, 2, 1).contiguous().reshape(merged_tokens, -1))
            offset += token_count
        return torch.cat(outputs, dim=0)

    def forward(self, pixel_values: Tensor, grid_thw: Tensor) -> Tensor:
        full_cu_seqlens = _vision_cu_seqlens(grid_thw)
        window_index, window_cu_seqlens = _vision_window_index(
            grid_thw,
            window_size=self.config.pos_emb_height,
        )
        hidden_states = self.ln_pre(self.patch_embedder(pixel_values, grid_thw))
        hidden_states = hidden_states[window_index]
        position_ids = _vision_position_ids(grid_thw)[window_index]
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer_type, layer in zip(self.config.layer_types, self.layers):
            cu_seqlens = full_cu_seqlens if layer_type == "full_attention" else window_cu_seqlens
            if self.recompute_layers and self.training:
                hidden_states = torch_checkpoint(
                    layer,
                    hidden_states,
                    cu_seqlens,
                    position_embeddings,
                    use_reentrant=False,
                )
            else:
                hidden_states = layer(hidden_states, cu_seqlens, position_embeddings)
        hidden_states = hidden_states[torch.argsort(window_index)]
        return self._pixel_shuffle(self.ln_post(hidden_states), grid_thw)


class MuseGlimmerVisionAdapter(nn.Module):
    """Two-layer GELU adapter used before projection into the text width."""

    def __init__(self, config: MuseGlimmerModelConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.vision_output_size, config.projector_hidden_size, bias=False)
        self.fc2 = nn.Linear(config.projector_hidden_size, config.projector_hidden_size, bias=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return F.gelu(self.fc2(F.gelu(self.fc1(hidden_states))))


class MuseGlimmerModel(HybridModel):
    """Native MCore Hybrid model with the Muse vision modules attached."""

    _EMPTY_NORM_EXTRA_STATE_SUFFIXES = (
        "input_layernorm._extra_state",
        "self_attention.post_layernorm._extra_state",
        "pre_mlp_layernorm._extra_state",
        "mlp.post_layernorm._extra_state",
        "decoder.final_norm._extra_state",
        "vision_tower.ln_pre._extra_state",
        "vision_tower.ln_post._extra_state",
        "norm1._extra_state",
        "norm2._extra_state",
    )
    _CENTERED_NORM_WEIGHT_SUFFIXES = (
        "input_layernorm.weight",
        "self_attention.post_layernorm.weight",
        "pre_mlp_layernorm.weight",
        "mlp.post_layernorm.weight",
    )

    def __init__(
        self,
        config: MuseGlimmerModelConfig,
        hybrid_stack_spec: ModuleSpec,
        vocab_size: int,
        pg_collection: ProcessGroupCollection,
        *,
        pre_process: bool,
        post_process: bool,
        vp_stage: int | None,
    ) -> None:
        super().__init__(
            config=config.transformer,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=vocab_size,
            max_sequence_length=config.seq_length,
            hybrid_layer_pattern=config.hybrid_layer_pattern,
            fp16_lm_cross_entropy=config.fp16_lm_cross_entropy,
            parallel_output=config.parallel_output,
            share_embeddings_and_output_weights=config.share_embeddings_and_output_weights,
            position_embedding_type=config.position_embedding_type,
            rotary_percent=config.rotary_percent,
            rotary_base=config.rotary_base,
            scatter_embedding_sequence_parallel=False,
            seq_len_interpolation_factor=config.seq_len_interpolation_factor,
            pre_process=pre_process,
            post_process=post_process,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
        )
        self.model_config = config
        if hasattr(self, "output_layer"):
            extend_instance(self.output_layer, MuseGlimmerOutputLayerMixin)
        if pre_process:
            self.vision_tower = MuseGlimmerVisionModel(
                config.vision,
                config.transformer,
                recompute_layers=config.recompute_vision_layers,
            )
            self.vision_adapter = MuseGlimmerVisionAdapter(config)
            self.vision_projection = nn.Linear(
                config.projector_hidden_size,
                config.transformer.hidden_size,
                bias=False,
            )
            for module in (self.vision_tower, self.vision_adapter, self.vision_projection):
                module.to(dtype=config.transformer.params_dtype)
                hook_hf_module_setattr_for_tp_grad_sync(module)
        self.freeze(
            freeze_language_model=config.freeze_language_model,
            freeze_vision_model=config.freeze_vision_model,
            freeze_vision_projection=config.freeze_vision_projection,
        )

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple[tuple[int, int, int], ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return a backend-stable Muse checkpoint schema.

        Transformer Engine affine norms add empty ``_extra_state`` objects that
        the explicit local fallback does not expose. Omitting only those empty
        objects keeps checkpoints backend-interchangeable without relaxing
        strict loading for parameters or stateful Transformer Engine modules.
        """
        metadata = ensure_metadata_has_dp_cp_group(metadata)
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        replica_id = (0, get_pg_rank(self.tp_group), get_pg_rank(metadata["dp_cp_group"]))
        for key in list(sharded_state_dict):
            if key.endswith(self._CENTERED_NORM_WEIGHT_SUFFIXES):
                sharded_state_dict[key].replica_id = replica_id
            elif key.endswith(self._EMPTY_NORM_EXTRA_STATE_SUFFIXES):
                extra_state = sharded_state_dict.pop(key)
                extra_state_data = getattr(extra_state, "data", None)
                has_extra_state = (
                    extra_state_data.numel() > 0
                    if isinstance(extra_state_data, Tensor)
                    else extra_state_data is not None and bool(extra_state_data)
                )
                if has_extra_state:
                    raise ValueError(f"Muse Transformer Engine norm extra state must be empty: {key}.")
        return sharded_state_dict

    def freeze(
        self,
        *,
        freeze_language_model: bool,
        freeze_vision_model: bool,
        freeze_vision_projection: bool,
    ) -> None:
        modules: list[nn.Module] = []
        if freeze_language_model:
            modules.extend(
                module
                for module in (
                    getattr(self, "embedding", None),
                    self.decoder,
                    getattr(self, "output_layer", None),
                )
                if module is not None
            )
        if freeze_vision_model and hasattr(self, "vision_tower"):
            modules.append(self.vision_tower)
        if freeze_vision_projection and hasattr(self, "vision_adapter"):
            modules.extend([self.vision_adapter, self.vision_projection])
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad = False

    def _vision_features(self, pixel_values: Tensor, grid_thw: Tensor) -> Tensor:
        features = self.vision_tower(pixel_values, grid_thw)
        features = self.vision_projection(self.vision_adapter(features))
        return F.rms_norm(
            features,
            (features.shape[-1],),
            eps=self.config.layernorm_epsilon,
        )

    @staticmethod
    def _scatter_features(inputs_embeds: Tensor, input_ids: Tensor, features: Tensor, token_id: int) -> Tensor:
        mask = (input_ids == token_id).unsqueeze(-1).expand_as(inputs_embeds)
        if inputs_embeds[mask].numel() != features.numel():
            raise ValueError(
                f"Muse media token count mismatch for token {token_id}: "
                f"{mask[..., 0].sum().item()} text slots vs {features.shape[0]} vision features."
            )
        return inputs_embeds.masked_scatter(mask, features.to(inputs_embeds.device, inputs_embeds.dtype))

    def forward(
        self,
        input_ids: Tensor | None = None,
        position_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        pixel_values: Tensor | None = None,
        image_grid_thw: Tensor | None = None,
        pixel_values_videos: Tensor | None = None,
        video_grid_thw: Tensor | None = None,
        labels: Tensor | None = None,
        inference_context: BaseInferenceContext | None = None,
        runtime_gather_output: bool | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        *,
        inference_params: BaseInferenceContext | None = None,
        loss_mask: Tensor | None = None,
        padding_mask: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        if self.pre_process:
            if inputs_embeds is None:
                if input_ids is None:
                    raise ValueError("input_ids is required when inputs_embeds is not supplied.")
                multimodal_mask = (input_ids == self.model_config.image_token_id) | (
                    input_ids == self.model_config.video_token_id
                )
                llm_input_ids = input_ids.masked_fill(multimodal_mask, 0)
                inputs_embeds = self.embedding(input_ids=llm_input_ids, position_ids=None)
                inputs_embeds = F.rms_norm(
                    inputs_embeds,
                    (inputs_embeds.shape[-1],),
                    eps=self.config.layernorm_epsilon,
                )
                inputs_embeds = inputs_embeds.transpose(0, 1).contiguous()
            if pixel_values is not None:
                if image_grid_thw is None or input_ids is None:
                    raise ValueError("image_grid_thw and input_ids are required with pixel_values.")
                inputs_embeds = self._scatter_features(
                    inputs_embeds,
                    input_ids,
                    self._vision_features(pixel_values, image_grid_thw),
                    self.model_config.image_token_id,
                )
            if pixel_values_videos is not None:
                if video_grid_thw is None or input_ids is None:
                    raise ValueError("video_grid_thw and input_ids are required with pixel_values_videos.")
                inputs_embeds = self._scatter_features(
                    inputs_embeds,
                    input_ids,
                    self._vision_features(pixel_values_videos, video_grid_thw),
                    self.model_config.video_token_id,
                )
            inputs_embeds = inputs_embeds.transpose(0, 1).contiguous()

        inputs_embeds, labels, loss_mask, position_ids, attention_mask = slice_batch_for_context_parallel(
            inputs_embeds=inputs_embeds,
            labels=labels,
            loss_mask=loss_mask,
            position_ids=position_ids,
            attention_mask=attention_mask,
            packed_seq_params=packed_seq_params,
            pg_collection=self.pg_collection,
        )
        if self.config.sequence_parallel and inputs_embeds is not None:
            inputs_embeds = scatter_to_sequence_parallel_region(inputs_embeds, group=self.pg_collection.tp)

        output = super().forward(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=inputs_embeds,
            labels=labels,
            inference_context=inference_context,
            runtime_gather_output=runtime_gather_output,
            packed_seq_params=packed_seq_params,
            inference_params=inference_params,
            loss_mask=loss_mask,
            padding_mask=padding_mask,
        )
        if loss_mask is not None:
            return output, loss_mask
        return output


__all__ = [
    "MuseGlimmerModel",
    "MuseGlimmerVisionModel",
    "get_muse_glimmer_hybrid_stack_spec",
]
