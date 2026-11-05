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
# See the License for the specific language governing permissions and limitations
# under the License.

"""Unified bridge for GLM-5.3-Flash (model_type ``glm5_next``).

GLM-5.3-Flash is a VLM: a hybrid KDA/DSA + MoE + mHC language model paired with a
replicated HF vision tower. This bridge handles both the text and vision paths in a
single registered bridge and model — there is no separate "VL" split.
``AutoBridge.get_model_config()`` returns ``GLM53FlashModelConfig``, which
``GLM53FlashModelBuilder`` builds; ``GLM53FlashModelProvider`` is the deprecated
provider counterpart built from the same kwargs.

Each HF decoder layer maps to two physical layers: attention at 2N and FFN at 2N+1.
Checkpoint keys use ``model.language_model``; mixed FP8 weights are dequantized on
import. Vision weights (``model.visual.**``) are loaded replicated and stay BF16 even
under FP8. The VLM wrapper degrades to text-only when ``pixel_values`` is None.
"""

import types
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch
from megatron.core import parallel_state
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols
from megatron.core.pipeline_parallel.utils import (
    is_pp_first_stage,
    is_pp_last_stage,
    is_vp_first_stage,
    is_vp_last_stage,
)
from megatron.core.ssm.kda_layer_config import KDALayerConfig
from megatron.core.tensor_parallel import scatter_to_sequence_parallel_region
from megatron.core.transformer import ModuleSpec
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.module import MegatronModule
from torch import Tensor, nn
from torch.distributed._tensor import DTensor

from megatron.bridge.models.conversion import quantization_utils
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ColumnParallelMapping,
    GatedMLPMapping,
    HCAlphaMapping,
    InitOnlyMapping,
    MegatronParamMapping,
    ReplicatedMapping,
    RowParallelMapping,
    _module_uses_fsdp,
)
from megatron.bridge.models.conversion.utils import get_module_and_param_from_name
from megatron.bridge.models.glm5next.glm53flash_config import GLM53FlashModelConfig
from megatron.bridge.models.hybrid.hybrid_provider import get_default_hybrid_stack_spec
from megatron.bridge.models.hybrid_mla_provider import HybridMLAModelProvider
from megatron.bridge.training.utils.packed_seq_utils import (
    get_packed_seq_cp_partition_indices,
    get_packed_seq_q_cu_seqlens,
    preprocess_packed_seqs,
)
from megatron.bridge.utils.common_utils import hook_hf_module_setattr_for_tp_grad_sync


try:
    from transformers import Glm5NextForConditionalGeneration
except ImportError:  # pragma: no cover
    Glm5NextForConditionalGeneration = "Glm5NextForConditionalGeneration"  # type: ignore[assignment]


if TYPE_CHECKING:
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.process_groups_config import ProcessGroupCollection


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _glm5next_text_config(hf_config):
    """Return the nested text_config for the GLM-5.3-Flash VLM."""
    text_config = getattr(hf_config, "text_config", None)
    return text_config if text_config is not None else hf_config


def _glm5next_layer_types(hf_config) -> list[str]:
    """Return the normalized per-layer attention types used by the bridge.

    Transformers 5.19 normalizes the DSA layer name from
    ``deepseek_sparse_attention`` to ``indexed_attention``.  Keep both spellings
    at the config boundary; the provider mapping below uses the canonical HF
    spelling.
    """
    text_config = _glm5next_text_config(hf_config)
    layer_types = list(getattr(text_config, "layer_types", []))
    num_layers = text_config.num_hidden_layers
    if len(layer_types) != num_layers:
        raise ValueError(f"layer_types has {len(layer_types)} entries, but num_hidden_layers is {num_layers}.")
    return [
        "deepseek_sparse_attention" if layer_type == "indexed_attention" else layer_type for layer_type in layer_types
    ]


def _glm5next_mlp_layer_types(hf_config) -> list[str]:
    """Per-HF-layer MLP type: 'dense' or 'sparse'."""
    text_config = _glm5next_text_config(hf_config)
    mlp_types = getattr(text_config, "mlp_layer_types", None)
    if mlp_types is None:
        first_k_dense = getattr(text_config, "first_k_dense_replace", min(3, text_config.num_hidden_layers))
        mlp_types = ["dense"] * first_k_dense + ["sparse"] * (text_config.num_hidden_layers - first_k_dense)
    return list(mlp_types)


def _glm5next_hybrid_layout(hf_config) -> tuple[str, list[int]]:
    """Return the Hybrid layer pattern and MoE schedule of the main decoder.

    Each HF block becomes an attention layer (KDA or DSA) followed by an FFN layer (dense MLP or
    MoE). ``moe_layer_freq`` indexes those physical layers, as router replay does.
    """
    attn_sym = {"linear_attention": Symbols.KDA, "deepseek_sparse_attention": Symbols.DS_ATTENTION}
    ffn_sym = {"dense": Symbols.MLP, "sparse": Symbols.MOE}
    pattern_chars = []
    moe_layer_freq = []
    for attn_type, mlp_type in zip(
        _glm5next_layer_types(hf_config), _glm5next_mlp_layer_types(hf_config), strict=True
    ):
        if attn_type not in attn_sym:
            raise ValueError(f"Unsupported GLM-5.3-Flash attention layer type: {attn_type!r}")
        if mlp_type not in ffn_sym:
            raise ValueError(f"Unsupported GLM-5.3-Flash MLP layer type: {mlp_type!r}")
        pattern_chars.append(attn_sym[attn_type] + ffn_sym[mlp_type])
        moe_layer_freq.extend((0, 1 if mlp_type == "sparse" else 0))
    return "".join(pattern_chars), moe_layer_freq


# ---------------------------------------------------------------------------
# Packed-layout helpers
# ---------------------------------------------------------------------------


def _is_keep_mask(attention_mask: Optional[Tensor], input_ids: Tensor) -> bool:
    """Whether ``attention_mask`` is a ``[batch, seq]`` keep mask for dense ``input_ids`` rows."""
    return attention_mask is not None and attention_mask.dim() == 2 and attention_mask.shape == input_ids.shape


def _packed_position_ids(packed_seq_params: "PackedSeqParams", total_tokens: int, device: torch.device) -> Tensor:
    """Return per-sequence positions ``[1, total_tokens]`` for a packed THD stream.

    Positions restart at each physical sequence start; tokens past the last boundary get 0.
    """
    _, cu_seqlens = get_packed_seq_q_cu_seqlens(packed_seq_params)
    cu_seqlens = cu_seqlens.to(device=device, dtype=torch.long)
    starts = torch.repeat_interleave(cu_seqlens[:-1], cu_seqlens.diff())
    positions = torch.zeros(total_tokens, dtype=torch.long, device=device)
    covered = min(starts.numel(), total_tokens)
    positions[:covered] = torch.arange(covered, device=device) - starts[:covered]
    return positions.unsqueeze(0)


def _select_tokens(tensor: Optional[Tensor], index: Optional[Tensor]) -> Optional[Tensor]:
    """Select tokens along dim 1 of a ``[1, total_tokens, ...]`` stream; no-op without an index."""
    if tensor is None or index is None:
        return tensor
    return tensor.index_select(1, index.to(tensor.device))


# ---------------------------------------------------------------------------
# Custom param mappings
# ---------------------------------------------------------------------------


class _ColumnParallelConcatMapping(MegatronParamMapping[dict]):
    """Map separate HF Q/K/V tensors to rank-local [q_shard | k_shard | v_shard]."""

    def __init__(self, megatron_param: str, hf_params: list[str]):
        super().__init__(megatron_param, {str(i): p for i, p in enumerate(hf_params)})
        self._hf_order = hf_params
        self._tp_mapping = ColumnParallelMapping(megatron_param, megatron_param)

    def resolve(self, captures):
        resolved_mg, resolved_hf = self._resolve_names(captures)
        return _ColumnParallelConcatMapping(resolved_mg, list(resolved_hf.values()))

    def _shard_per_rank(self, merged: torch.Tensor, section_sizes: list[int]) -> list[torch.Tensor]:
        if any(size % self.tp_size for size in section_sizes):
            raise ValueError("Each KDA Q/K/V section must be divisible by tensor parallel size.")
        sections = merged.split(section_sizes, dim=0)
        return [
            torch.cat(shards, dim=0)
            for shards in zip(*(section.chunk(self.tp_size, dim=0) for section in sections), strict=True)
        ]

    def hf_to_megatron(self, hf_weights: dict, megatron_module: nn.Module) -> torch.Tensor:
        splits = None
        if self.tp_rank == 0:
            sections = [hf_weights[str(i)] for i in range(len(self._hf_order))]
            merged = torch.cat(sections, dim=0)
            if self.tp_size == 1:
                return merged
            splits = self._shard_per_rank(merged, [section.shape[0] for section in sections])

        normalized_param = self._tp_mapping._normalize_expert_param_name(self.megatron_param)
        _, target = get_module_and_param_from_name(megatron_module, normalized_param)
        shape = target.orig_param.shape if isinstance(target, DTensor) else target.shape
        return self._tp_mapping.scatter_to_tp_ranks(splits, shape, target.dtype, target.device)

    def megatron_to_hf(self, megatron_weights: Optional[torch.Tensor], megatron_module) -> dict:
        # Non-owning PP stages receive None; replicate the owner's tensor and layout so
        # every stage (including the saving rank) emits the HF tensors.
        owns_weights = megatron_weights is not None
        megatron_weights = self.broadcast_from_pp_rank(megatron_weights, cache_key=str(self.hf_param))
        if megatron_weights is None:
            return {}
        local_layout = None
        if owns_weights:
            config = self._get_config(megatron_module) if megatron_module is not None else None
            local_layout = (self._section_sizes(config), _module_uses_fsdp(megatron_module))
        sizes, materialized = self.broadcast_obj_from_pp_rank(local_layout, cache_key=f"{self.hf_param}:layout")
        weights = self.maybe_dequantize(megatron_weights)

        if self.tp_size == 1:
            if sizes is None:
                sizes = self._equal_sections(weights.shape[0])
            parts = weights.split(sizes, dim=0)
            return dict(zip(self._hf_order, parts, strict=True))

        # Each TP rank holds [q_r | k_r | v_r]. A Megatron-FSDP tensor is already
        # materialized as those rank pieces concatenated in rank order; plain shards
        # need a TP gather. Either way, regroup the rank pieces section by section.
        if materialized:
            shards = list(weights.chunk(self.tp_size, dim=0))
        else:
            shards = self._tp_mapping.gather_from_tp_ranks(weights)
        if sizes is None:
            local_sizes = self._equal_sections(shards[0].shape[0])
        else:
            if any(size % self.tp_size for size in sizes):
                raise ValueError("Each KDA Q/K/V section must be divisible by tensor parallel size.")
            local_sizes = [size // self.tp_size for size in sizes]
        parts = [
            torch.cat(section_shards, dim=0)
            for section_shards in zip(*(shard.split(local_sizes, dim=0) for shard in shards), strict=True)
        ]
        return dict(zip(self._hf_order, parts, strict=True))

    def _equal_sections(self, rows: int) -> list[int]:
        if rows % len(self._hf_order):
            raise ValueError("Cannot infer unequal KDA Q/K/V sections without the model config.")
        return [rows // len(self._hf_order)] * len(self._hf_order)

    def _section_sizes(self, config):
        if config is None:
            return None
        qk = config.linear_key_head_dim * config.linear_num_key_heads
        v = config.linear_value_head_dim * config.linear_num_value_heads
        return [qk, qk, v]


class _EhProjSplitMapping(MegatronParamMapping[torch.Tensor]):
    """Split HF eh_proj [hidden, 2*hidden] into column-sharded Megatron e_proj/h_proj.

    Import: split along input dim (dim 1), shard column-parallel (dim 0).
    Export: gather both halves, concat along dim 1.
    """

    _export_cache: dict = {}

    def __init__(self, megatron_param: str, hf_param: str, index: int):
        super().__init__(megatron_param=megatron_param, hf_param=hf_param)
        self._index = index
        self.allow_hf_name_mismatch = True

    def resolve(self, captures):
        resolved_mg, resolved_hf = self._resolve_names(captures)
        return _EhProjSplitMapping(resolved_mg, resolved_hf, self._index)

    def hf_to_megatron(self, hf_weights, megatron_module):
        if hf_weights is None:
            return None
        half = hf_weights.shape[1] // 2
        chunk = hf_weights[:, self._index * half : (self._index + 1) * half]
        return self._shard_column(chunk, megatron_module)

    def _shard_column(self, weight, megatron_module):
        device = next(megatron_module.parameters()).device
        weight = weight.to(device=device)
        if self.tp_size == 1:
            return weight
        shard = weight.shape[0] // self.tp_size
        start = self.tp_rank * shard
        return weight[start : start + shard].contiguous()

    def megatron_to_hf(self, megatron_weights, megatron_module):
        # MTP lives on the last PP stage; replicate it so every stage emits eh_proj.
        owns_weights = megatron_weights is not None
        megatron_weights = self.broadcast_from_pp_rank(megatron_weights, cache_key=str(self.hf_param))
        if megatron_weights is None:
            return {}
        materialized = self.broadcast_obj_from_pp_rank(
            _module_uses_fsdp(megatron_module) if owns_weights else None,
            cache_key=f"{self.hf_param}:materialized",
        )
        megatron_weights = self.maybe_dequantize(megatron_weights)
        # A Megatron-FSDP tensor is already materialized across TP; gather plain shards only.
        if self.tp_size == 1 or materialized:
            full = megatron_weights
        else:
            full = torch.cat(self.gather_from_tp_ranks(megatron_weights), dim=0)
        cache = _EhProjSplitMapping._export_cache
        key = str(self.hf_param)
        cache[f"{key}#{self._index}"] = full
        other_key = f"{key}#{1 - self._index}"
        if other_key not in cache:
            return {}
        other_w = cache.pop(other_key)
        cache.pop(f"{key}#{self._index}", None)
        parts = [None, None]
        parts[self._index] = full
        parts[1 - self._index] = other_w
        return {key: torch.cat(parts, dim=1).contiguous()}


# ---------------------------------------------------------------------------
# Stack spec
# ---------------------------------------------------------------------------


def glm5next_hybrid_stack_spec(config: HybridMLAModelProvider) -> ModuleSpec:
    """Return the Hybrid stack with MLA q/kv norms wired into the DSA spec."""
    stack_spec = deepcopy(get_default_hybrid_stack_spec(config))
    dsa_submodules = stack_spec.submodules.dsa_layer.submodules
    native_norm = dsa_submodules.input_layernorm
    attention_submodules = dsa_submodules.self_attention.submodules
    attention_submodules.q_layernorm = native_norm
    attention_submodules.kv_layernorm = native_norm
    return stack_spec


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


@dataclass
class GLM53FlashModelProvider(HybridMLAModelProvider, KDALayerConfig):
    """Provider for the unified GLM-5.3-Flash VLM: hybrid language stack + HF vision."""

    # Vision configuration (HF Glm5NextVisionConfig object, stored as-is).
    vision_config: Any = None
    spatial_merge_size: int = 2
    # Inject vision features before scattering the full sequence across TP ranks.
    scatter_embedding_sequence_parallel: bool = False
    # Required by HF's can_return_tuple-decorated feature extraction methods.
    return_dict: bool = True

    # Multimodal token IDs (from the top-level Glm5NextConfig).
    image_token_id: int = 154854
    video_token_id: int = 154855
    image_start_token_id: int = 154830
    image_end_token_id: int = 154831
    video_start_token_id: int = 154832
    video_end_token_id: int = 154833

    # Freeze options for LoRA training.
    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False

    def provide(
        self, pre_process: Optional[bool] = None, post_process: Optional[bool] = None, vp_stage: Optional[int] = None
    ) -> "GLM53FlashModel":
        """Build the VLM wrapper and apply explicitly requested freeze options.

        Omitted ``pre_process``/``post_process`` flags resolve to this rank's first/last
        pipeline stage, as in ``HybridModelProvider.provide``, so the wrapper and its
        language model agree on which stage owns the vision tower and embeddings.
        """
        vp_size = self.virtual_pipeline_model_parallel_size
        if pre_process is None:
            pre_process = is_vp_first_stage(vp_stage=vp_stage, vp_size=vp_size) and is_pp_first_stage(
                self._pg_collection.pp
            )
        if post_process is None:
            post_process = is_vp_last_stage(vp_stage=vp_stage, vp_size=vp_size) and is_pp_last_stage(
                self._pg_collection.pp
            )
        model = GLM53FlashModel(self, pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
        if self.freeze_language_model or self.freeze_vision_model or self.freeze_vision_projection:
            model.freeze(
                freeze_language_model=self.freeze_language_model,
                freeze_vision_model=self.freeze_vision_model,
                freeze_vision_projection=self.freeze_vision_projection,
            )
        return model

    def provide_language_model(
        self, pre_process: Optional[bool] = None, post_process: Optional[bool] = None, vp_stage: Optional[int] = None
    ):
        """Build the Megatron HybridModel language model (delegates to the parent)."""
        return super().provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)


# ---------------------------------------------------------------------------
# Model (VLM wrapper: replicated HF vision tower + Megatron HybridModel)
# ---------------------------------------------------------------------------


class GLM53FlashModel(MegatronModule):
    """Inject HF image/video features into the hybrid language model's embeddings.

    Args:
        config: The provider, or the transformer config of a builder-backed model config.
            Either one carries the Hugging Face vision config and the multimodal token IDs.
        pre_process: Whether to build the vision tower and the embeddings.
        post_process: Whether to build the language-model output layer.
        vp_stage: Virtual pipeline stage.
        build_language_model: Builds the language model from ``pre_process``, ``post_process``
            and ``vp_stage``. Defaults to the provider's ``provide_language_model()``.
        pg_collection: Process groups a builder passes. Defaults to the provider's.
    """

    def __init__(
        self,
        config,
        pre_process: bool = True,
        post_process: bool = True,
        vp_stage: Optional[int] = None,
        *,
        build_language_model: Optional[Callable[..., MegatronModule]] = None,
        pg_collection: Optional["ProcessGroupCollection"] = None,
    ) -> None:
        super().__init__(config=config)
        if pg_collection is not None:
            self.pg_collection = pg_collection

        self.pre_process = pre_process
        self.post_process = post_process
        self.vp_stage = vp_stage

        if pre_process:
            from transformers.models.glm5_next.modeling_glm5_next import (
                Glm5NextModel,
                Glm5NextVisionModel,
            )

            self.visual = Glm5NextVisionModel._from_config(config.vision_config)
            # A builder-built model has no Float16Module during conversion; match the language model.
            params_dtype = getattr(config, "params_dtype", None)
            if params_dtype is not None:
                self.visual.to(dtype=params_dtype)
            hook_hf_module_setattr_for_tp_grad_sync(self.visual)

            # Only the stage that owns the vision tower extracts and inserts its features.
            self.get_image_features = types.MethodType(Glm5NextModel.get_image_features, self)
            self.get_video_features = types.MethodType(Glm5NextModel.get_video_features, self)
            self.get_placeholder_mask = types.MethodType(Glm5NextModel.get_placeholder_mask, self)

        if build_language_model is None:
            build_language_model = self.config.provide_language_model
        self.language_model = build_language_model(
            pre_process=pre_process, post_process=post_process, vp_stage=vp_stage
        )

        self.share_embeddings_and_output_weights = self.language_model.share_embeddings_and_output_weights
        self.shared_embedding_or_output_weight = self.language_model.shared_embedding_or_output_weight

        self.config.spatial_merge_size = getattr(self.config.vision_config, "spatial_merge_size", 2)

    def _process_groups(self) -> Optional["ProcessGroupCollection"]:
        """Return the process groups a builder passed, or the ones the provider carries."""
        pg_collection = getattr(self, "pg_collection", None)
        if pg_collection is not None:
            return pg_collection
        return getattr(self.config, "_pg_collection", None)

    def set_input_tensor(self, input_tensor) -> None:
        """Set model chunk input tensor."""
        self.language_model.set_input_tensor(input_tensor)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        mm_token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        runtime_gather_output: Optional[bool] = None,
        packed_seq_params: Optional["PackedSeqParams"] = None,
        *,
        loss_mask: Optional[Tensor] = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Insert vision features, then run the language model on a THD token stream.

        Two input layouts are accepted:

        * Dense rows without ``packed_seq_params`` (Bridge BSHD training, generation).
          ``attention_mask`` is an optional ``[batch, seq]`` keep mask; every token is
          kept when it is omitted. Token IDs, embeddings, labels and the loss mask are
          packed with the same layout, so padding has zero loss, and the packed loss
          mask is returned with the output.
        * A ``[1, total_tokens]`` stream that is already packed, with ``packed_seq_params``
          and ``attention_mask=None`` (Bridge collate-time packing). Its boundaries are
          kept and only this context-parallel rank's tokens are selected.

        The language model applies no positional embedding, so ``position_ids`` is not
        read; per-sequence positions in the packed layout are derived for MTP instead.
        Placeholder masks need whole sequences, so vision features are inserted before
        packing or context-parallel selection; sequence-parallel scattering comes last.
        Only the pre-process stage embeds tokens and runs the vision tower. Later stages
        read activations from ``set_input_tensor`` and use token IDs only for layout.
        """
        if input_ids is None:
            raise ValueError("GLM53FlashModel requires input_ids to derive the packed token layout.")

        already_packed = packed_seq_params is not None
        return_loss_mask = False
        cp_index = None
        keep_mask = None
        if already_packed:
            if attention_mask is not None:
                raise ValueError(
                    "Packed GLM-5.3-Flash input takes its sequence boundaries from packed_seq_params; "
                    "pass attention_mask=None."
                )
            if input_ids.dim() != 2 or input_ids.size(0) != 1:
                raise ValueError(
                    "Packed GLM-5.3-Flash input must have shape [1, total_tokens]; got input_ids shape "
                    f"{tuple(input_ids.shape)}."
                )
            cp_index = self._packed_cp_index(packed_seq_params, input_ids)
            lm_input_ids = _select_tokens(input_ids, cp_index)
            lm_position_ids = _select_tokens(
                _packed_position_ids(packed_seq_params, input_ids.size(1), input_ids.device), cp_index
            )
            if cp_index is not None:
                labels = _select_tokens(labels, cp_index)
                if loss_mask is not None:
                    loss_mask = _select_tokens(loss_mask, cp_index)
                    return_loss_mask = True
        else:
            if _is_keep_mask(attention_mask, input_ids):
                keep_mask = attention_mask.bool()
            else:
                keep_mask = torch.ones(input_ids.shape, dtype=torch.bool, device=input_ids.device)
            lm_input_ids, packed_seq_params = self._pack_rows(input_ids, keep_mask)
            row_positions = (keep_mask.long().cumsum(dim=-1) - 1).clamp(min=0)
            lm_position_ids = self._pack_rows(row_positions, keep_mask)[0]
            if labels is not None:
                labels = self._pack_rows(labels, keep_mask)[0]
            if loss_mask is not None:
                loss_mask = self._pack_rows(loss_mask, keep_mask)[0]
                return_loss_mask = True

        decoder_input = None
        if self.pre_process:
            if inputs_embeds is None:
                # [seq, batch, hidden], not yet scattered for SP -> HF's [batch, seq, hidden].
                inputs_embeds = self.language_model.embedding(input_ids=input_ids, position_ids=None)
                inputs_embeds = inputs_embeds.transpose(0, 1).contiguous()
            inputs_embeds = self._insert_vision_features(
                input_ids,
                inputs_embeds,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
            )
            if already_packed:
                decoder_input = _select_tokens(inputs_embeds, cp_index)
            else:
                decoder_input = self._pack_rows(inputs_embeds, keep_mask)[0]
            # Megatron expects [seq, batch, hidden]; THD uses batch 1.
            decoder_input = decoder_input.transpose(0, 1).contiguous()
            if self.config.sequence_parallel:
                pg_collection = self._process_groups()
                tp_group = pg_collection.tp if pg_collection is not None else None
                decoder_input = scatter_to_sequence_parallel_region(decoder_input, group=tp_group)

        output = self.language_model.forward(
            input_ids=lm_input_ids,
            position_ids=lm_position_ids,
            attention_mask=None,
            decoder_input=decoder_input,
            labels=labels,
            loss_mask=loss_mask,
            runtime_gather_output=runtime_gather_output,
            packed_seq_params=packed_seq_params,
        )
        if return_loss_mask:
            return output, loss_mask
        return output

    def _insert_vision_features(
        self,
        input_ids: Tensor,
        inputs_embeds: Tensor,
        *,
        pixel_values: Optional[Tensor],
        image_grid_thw: Optional[Tensor],
        pixel_values_videos: Optional[Tensor],
        video_grid_thw: Optional[Tensor],
    ) -> Tensor:
        """Replace image and video placeholder embeddings with HF vision features."""
        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw).pooler_output
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw).pooler_output
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
        return inputs_embeds

    def _pack_rows(self, tensor: Tensor, keep_mask: Tensor) -> tuple[Tensor, "PackedSeqParams"]:
        """Pack the kept tokens of dense rows into this rank's TP/CP/FP8-aligned THD stream."""
        return preprocess_packed_seqs(
            tensor,
            keep_mask,
            pg_collection=self._process_groups(),
            use_fp8_padding=self.config.fp8 in ("e4m3", "hybrid"),
        )

    def _packed_cp_index(self, packed_seq_params: "PackedSeqParams", input_ids: Tensor) -> Optional[Tensor]:
        """Return this CP rank's token indices into a packed stream, or None without CP."""
        pg_collection = self._process_groups()
        cp_group = pg_collection.cp if pg_collection is not None else parallel_state.get_context_parallel_group()
        cp_size = cp_group.size()
        if cp_size == 1:
            return None
        return get_packed_seq_cp_partition_indices(
            packed_seq_params,
            total_tokens=input_ids.size(1),
            cp_size=cp_size,
            cp_rank=cp_group.rank(),
            device=input_ids.device,
            cp_group=cp_group,
        )

    def freeze(
        self,
        freeze_language_model: bool,
        freeze_vision_model: bool,
        freeze_vision_projection: bool,
    ):
        """Freeze model modules (set requires_grad=False)."""
        modules = []
        if freeze_language_model and hasattr(self, "language_model") and self.language_model is not None:
            modules.append(self.language_model)
        if freeze_vision_model and hasattr(self, "visual") and self.visual is not None:
            if hasattr(self.visual, "patch_embed"):
                modules.append(self.visual.patch_embed)
            if hasattr(self.visual, "blocks"):
                modules.append(self.visual.blocks)
        if freeze_vision_projection and hasattr(self, "visual") and self.visual is not None:
            if hasattr(self.visual, "merger"):
                modules.append(self.visual.merger)
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------


@MegatronModelBridge.register_bridge(
    source=Glm5NextForConditionalGeneration,
    target=GLM53FlashModel,
    provider=GLM53FlashModelProvider,
    model_type="glm5_next",
)
class GLM53FlashBridge(MegatronModelBridge):
    """Unified bridge for GLM-5.3-Flash: hybrid language model + replicated vision tower."""

    MODEL_CONFIG_CLASS = GLM53FlashModelConfig
    USE_MODEL_CONFIG_FOR_CONVERSION = True

    # FP8 source weights are dequantized on import and exported as plain weights; saved
    # artifacts drop quantization_config, so their inverse scales are not written back.
    DEQUANTIZED_SOURCE_KEY_SUFFIXES = ("_scale_inv",)

    def hf_config_to_model_config_kwargs(self, hf_config) -> dict[str, Any]:
        """Convert the Hugging Face GLM-5.3-Flash config to builder config kwargs.

        The language-model fields come from the nested ``text_config``; the vision config, the
        multimodal token IDs and ``tie_word_embeddings`` come from the top-level config.
        """
        text_config = _glm5next_text_config(hf_config)
        config_kwargs = super().hf_config_to_model_config_kwargs(text_config)
        # GLM-5.3-Flash applies no rotary embedding; its MLA rope settings are set below.
        config_kwargs.pop("_mla_rope_params", None)

        main_pattern, moe_layer_freq = _glm5next_hybrid_layout(hf_config)
        lin = text_config.linear_attn_config
        num_mtp = int(getattr(text_config, "num_nextn_predict_layers", 0) or 0)
        vision_config = getattr(hf_config, "vision_config", None)

        config_kwargs.update(
            normalization="RMSNorm",
            gated_linear_unit=True,
            add_bias_linear=False,
            share_embeddings_and_output_weights=bool(getattr(hf_config, "tie_word_embeddings", False)),
            qk_layernorm=True,
            multi_latent_attention=True,
            sequence_parallel=True,
            # HybridModelProvider defaults that differ from the builder config's.
            position_embedding_type="none",
            attention_backend=AttnBackend.flash,
            # ---- MoE ----
            moe_grouped_gemm=True,
            moe_router_pre_softmax=True,
            moe_token_dispatcher_type="alltoall",
            moe_flex_dispatcher_backend="hybridep",
            moe_flex_dispatcher_num_sms=16,
            moe_router_load_balancing_type="seq_aux_loss",
            moe_shared_expert_overlap=False,
            moe_router_score_function=text_config.scoring_func,  # "sigmoid"
            moe_router_enable_expert_bias=True,
            moe_router_dtype="fp32",
            moe_permute_fusion=True,
            moe_router_topk=text_config.num_experts_per_tok,  # 8
            moe_router_topk_scaling_factor=getattr(text_config, "routed_scaling_factor", 1.0),  # 2.5
            moe_aux_loss_coeff=getattr(text_config, "router_aux_loss_coef", 0.001),
            activation_func_clamp_value=getattr(text_config, "swiglu_limit", 0.0),  # 10.0
            moe_shared_expert_intermediate_size=text_config.moe_intermediate_size * text_config.n_shared_experts,
            # The shared expert is always on: the checkpoint has no per-token gate scalar.
            moe_shared_expert_gate=False,
            hidden_dropout=0.0,
            attention_softmax_in_fp32=False,
            make_vocab_size_divisible_by=1280,
            # ---- Hybrid layer pattern: each HF block is an attention and an FFN layer ----
            hybrid_layer_pattern=main_pattern,
            num_layers=len(main_pattern),
            hybrid_stack_spec=glm5next_hybrid_stack_spec,
            moe_layer_freq=moe_layer_freq,
            # ---- MLA geometry (NoPE): qk_rope_head_dim=0, qk_nope_head_dim=qk_head_dim ----
            qk_pos_emb_head_dim=text_config.qk_head_dim - text_config.qk_nope_head_dim,  # 0
            q_lora_rank=text_config.q_lora_rank,  # 1536
            kv_lora_rank=text_config.kv_lora_rank,  # 512
            v_head_dim=text_config.v_head_dim,  # 256
            qk_head_dim=text_config.qk_head_dim,  # 256
            num_attention_heads=text_config.num_attention_heads,  # 64
            mla_proj_use_quantization=False,
            # HF head_dim=0 is a NoPE sentinel, not a valid attention channel width.
            kv_channels=text_config.v_head_dim,  # 256
            rope_type="yarn",
            rotary_base=10000.0,
            rotary_scaling_factor=1.0,
            mscale=1.0,
            mscale_all_dim=1.0,
            apply_rope_fusion=False,
            rotary_interleaved=False,
            dsa_indexer_rope_interleaved=getattr(text_config, "indexer_rope_interleave", True),
            cp_comm_type="allgather",
            # ---- DSA indexer (kpool) ----
            experimental_attention_variant="dsa",
            dsa_indexer_head_dim=text_config.index_head_dim,  # 128
            dsa_indexer_n_heads=text_config.index_n_heads,  # 32
            dsa_indexer_topk=text_config.index_topk,  # 2048
            dsa_indexer_loss_coeff=0.0,
            dsa_indexer_use_sparse_loss=False,
            dsa_indexer_rotate_activation=False,
            dsa_indexer_qk_proj_use_quantization=False,
            dsa_indexer_kpool_use_quantization=True,
            dsa_indexer_k_norm_fp32=True,
            # vLLM computes this projection in FP32 because BF16 rounding can change
            # near-tie KPool rankings on long-context inputs.
            dsa_indexer_weights_proj_use_quantization=False,
            dsa_indexer_weights_proj_output_dtype="fp32",
            dsa_indexer_k_norm_epsilon=1e-6,
            dsa_indexer_kpool=int(getattr(text_config, "index_kpool", 1)),  # 4
            dsa_indexer_kpool_always_select_tail=bool(getattr(text_config, "index_kpool_always_select_tail", True)),
            dsa_indexer_topk_freq=1,
            dsa_indexer_skip_topk_offset=0,
            # ---- KDA two-stage gates; query, key and value share one head geometry ----
            kda_disable_fp8=True,
            kda_two_stage_gates=True,
            kda_safe_gate=True,
            kda_lower_bound=float(lin.get("gate_lower_bound", -5.0)),
            gdn_pre_gated_delta_rule_fusion=False,
            gdn_conv_pad_alignment=None,
            linear_num_key_heads=int(lin["num_heads"]),  # 64
            linear_num_value_heads=int(lin["num_heads"]),  # 64
            linear_key_head_dim=int(lin["head_dim"]),  # 128
            linear_value_head_dim=int(lin["head_dim"]),  # 128
            linear_conv_kernel_dim=int(lin["short_conv_kernel_size"]),  # 4
            # ---- mHC (multi-stream hyper-connections) ----
            enable_mhc_connections=True,
            mhc_num_residual_streams=int(getattr(text_config, "hc_mult", 4)),
            mhc_sinkhorn_iterations=int(getattr(text_config, "hc_sinkhorn_iters", 20)),
            mhc_learned_output_contract=False,
            mhc_norm_eps_inside_sqrt=True,
            mhc_keep_mappings_in_fp32=True,
            # ---- MTP: one repeated DSA + MoE layer, excluded from num_layers and moe_layer_freq ----
            # A source without MTP gets no MTP pattern: finalize() would otherwise append one
            # repeated depth and infer mtp_num_layers=1.
            mtp_num_layers=num_mtp,
            mtp_hybrid_override_pattern="DE" if num_mtp else None,
            mtp_use_repeated_layer=bool(num_mtp),
            persist_layer_norm=True,
            gradient_accumulation_fusion=True,
            bias_dropout_fusion=True,
            # ---- Vision tower (replicated HF Glm5NextVisionModel) ----
            vision_config=vision_config,
            image_token_id=getattr(hf_config, "image_token_id", 154854),
            video_token_id=getattr(hf_config, "video_token_id", 154855),
            image_start_token_id=getattr(hf_config, "image_start_token_id", 154830),
            image_end_token_id=getattr(hf_config, "image_end_token_id", 154831),
            video_start_token_id=getattr(hf_config, "video_start_token_id", 154832),
            video_end_token_id=getattr(hf_config, "video_end_token_id", 154833),
            spatial_merge_size=getattr(vision_config, "spatial_merge_size", 2),
        )
        if num_mtp:
            config_kwargs["mtp_loss_scaling_factor"] = 0.1
        return config_kwargs

    def hf_config_to_provider_kwargs(self, hf_config) -> dict[str, Any]:
        """Adapt the builder kwargs to the deprecated provider path."""
        provider_kwargs = self.hf_config_to_model_config_kwargs(hf_config)
        # Provider-only flag that nothing reads; kept so the deprecated provider is unchanged.
        provider_kwargs["keep_mtp_spec_in_bf16"] = True
        return provider_kwargs

    @classmethod
    def megatron_to_hf_config(cls, provider) -> dict:
        hf_config = super().megatron_to_hf_config(provider)
        pattern = getattr(provider, "hybrid_layer_pattern", None)
        if pattern:
            main_pattern = pattern.split(Symbols.MTP_SEPARATOR)[0]
            hf_config["num_hidden_layers"] = sum(1 for c in main_pattern if c in (Symbols.KDA, Symbols.DS_ATTENTION))
        return hf_config

    # ------------------------------------------------------------------
    # FP8 dequantization on import
    # ------------------------------------------------------------------

    def maybe_modify_loaded_hf_weight(self, hf_param, hf_state_dict: Mapping[str, torch.Tensor]):
        # InitOnlyMapping declares a synthetic HF key ("<mg>...__init_only__") that
        # never exists in the checkpoint; skip the state-dict lookup so import keeps
        # the Megatron module's random initialization.
        if isinstance(hf_param, str) and hf_param.endswith(".__init_only__"):
            return None
        hf_weights = super().maybe_modify_loaded_hf_weight(hf_param, hf_state_dict)
        if isinstance(hf_weights, dict):
            return {
                key: self._maybe_dequantize_fp8(tensor, hf_param[key], hf_state_dict)
                for key, tensor in hf_weights.items()
            }
        return self._maybe_dequantize_fp8(hf_weights, hf_param, hf_state_dict)

    @staticmethod
    def _maybe_dequantize_fp8(weight, param_name: str, hf_state_dict: Mapping[str, torch.Tensor]):
        scale_key = param_name + "_scale_inv"
        return quantization_utils.maybe_dequantize_fp8_blockwise(weight, hf_state_dict.get(scale_key))

    # ------------------------------------------------------------------
    # Weight mapping registry
    # ------------------------------------------------------------------

    def mapping_registry(self) -> MegatronMappingRegistry:
        hf_config = self.hf_config
        text_config = _glm5next_text_config(hf_config)
        attn_types = _glm5next_layer_types(hf_config)
        mlp_types = _glm5next_mlp_layer_types(hf_config)
        num_hf_layers = text_config.num_hidden_layers

        mapping_list = [
            AutoMapping(
                "embedding.word_embeddings.weight",
                "model.language_model.embed_tokens.weight",
            ),
            AutoMapping(
                "decoder.final_norm.weight",
                "model.language_model.norm.weight",
            ),
            AutoMapping(
                "output_layer.weight",
                "lm_head.weight",
            ),
        ]

        for hf_layer_idx in range(num_hf_layers):
            is_kda = attn_types[hf_layer_idx] == "linear_attention"
            is_moe = mlp_types[hf_layer_idx] == "sparse"
            attn_layer_idx = 2 * hf_layer_idx
            ffn_layer_idx = attn_layer_idx + 1
            hf_layer = f"model.language_model.layers.{hf_layer_idx}"
            attn_layer = f"decoder.layers.{attn_layer_idx}"
            ffn_layer = f"decoder.layers.{ffn_layer_idx}"

            # mHC hyper-connections wrap every non-MTP physical layer (attn + ffn).
            mapping_list.extend(self._mhc_mappings(attn_layer, hf_layer, "attn"))
            mapping_list.extend(self._mhc_mappings(ffn_layer, hf_layer, "ffn"))

            if is_kda:
                mapping_list.extend(self._kda_attn_mappings(attn_layer, hf_layer))
            else:
                mapping_list.extend(self._dsa_attn_mappings(attn_layer, hf_layer))

            if is_moe:
                mapping_list.extend(self._moe_ffn_mappings(ffn_layer, hf_layer))
            else:
                mapping_list.extend(self._dense_ffn_mappings(ffn_layer, hf_layer))

        # MTP uses split e/h projections and two inner layers without HF mHC weights.
        num_mtp = int(getattr(text_config, "num_nextn_predict_layers", 0) or 0)
        for mtp_idx in range(num_mtp):
            mtp_hf = f"model.language_model.layers.{mtp_idx + num_hf_layers}"
            mtp_mg = f"mtp.layers.{mtp_idx}"
            eh_proj = f"{mtp_hf}.eh_proj.weight"
            mapping_list.extend(
                [
                    AutoMapping(f"{mtp_mg}.enorm.weight", f"{mtp_hf}.enorm.weight"),
                    AutoMapping(f"{mtp_mg}.hnorm.weight", f"{mtp_hf}.hnorm.weight"),
                    # Split eh_proj [hidden, 2*hidden] along its input dimension.
                    _EhProjSplitMapping(f"{mtp_mg}.e_proj.weight", eh_proj, 0),
                    _EhProjSplitMapping(f"{mtp_mg}.h_proj.weight", eh_proj, 1),
                    AutoMapping(
                        f"{mtp_mg}.final_layernorm.weight",
                        f"{mtp_hf}.shared_head.norm.weight",
                    ),
                ]
            )
            # Inner layer 0 is attention; layer 1 is MoE.
            mtp_attn_layer = f"{mtp_mg}.mtp_model_layer.layers.0"
            mtp_ffn_layer = f"{mtp_mg}.mtp_model_layer.layers.1"
            mapping_list.extend(self._mhc_init_only_mappings(mtp_attn_layer))
            mapping_list.extend(self._mhc_init_only_mappings(mtp_ffn_layer))
            mapping_list.extend(self._dsa_attn_mappings(mtp_attn_layer, mtp_hf))
            mapping_list.extend(self._moe_ffn_mappings(mtp_ffn_layer, mtp_hf))

        # The language model is a submodule of the VLM wrapper, so prefix every
        # Megatron param with ``language_model.`` and append the replicated vision weights.
        return self._prefix_mapping_registry(
            MegatronMappingRegistry(*mapping_list),
            "language_model.",
            ReplicatedMapping("visual.**", "model.visual.**"),
        )

    @staticmethod
    def _mhc_init_only_mappings(mg_layer: str) -> list:
        """Init-only mappings for an mHC hyper_connection with no HF source.

        Used for the MTP layer's inner physical layers, whose ``hyper_connection``
        params (mapping_proj, bias, alpha_pre/post/res) are created by MCore but
        have no counterpart in GLM-5.3-Flash's HF checkpoint.
        """
        hc = "hyper_connection"
        return [
            InitOnlyMapping(f"{mg_layer}.{hc}.mapping_proj.weight"),
            InitOnlyMapping(f"{mg_layer}.{hc}.bias"),
            InitOnlyMapping(f"{mg_layer}.{hc}.alpha_pre"),
            InitOnlyMapping(f"{mg_layer}.{hc}.alpha_post"),
            InitOnlyMapping(f"{mg_layer}.{hc}.alpha_res"),
        ]

    # ------------------------------------------------------------------
    # Sub-mapping builders
    # ------------------------------------------------------------------

    @staticmethod
    def _mhc_mappings(mg_layer: str, hf_layer: str, kind: str) -> list:
        """mHC hyper-connection mappings for one physical layer.

        ``kind`` is 'attn' or 'ffn'. With ``enable_mhc_connections`` MCore wraps
        each physical layer in a ``HyperConnectionHybridLayer`` exposing a single
        ``hyper_connection`` submodule (NOT separate ``self_attention_hyper_connection``
        / ``mlp_hyper_connection``). The checkpoint stores ``hc_{kind}_{fn,base,scale}``
        as bare parameters (no ``.weight`` suffix) on the HF layer.
        """
        hc = "hyper_connection"
        pre = f"{mg_layer}.{hc}.alpha_pre"
        post = f"{mg_layer}.{hc}.alpha_post"
        res = f"{mg_layer}.{hc}.alpha_res"
        scale = f"{hf_layer}.hc_{kind}_scale"
        return [
            ReplicatedMapping(
                f"{mg_layer}.{hc}.mapping_proj.weight",
                f"{hf_layer}.hc_{kind}_fn",
            ),
            ReplicatedMapping(
                f"{mg_layer}.{hc}.bias",
                f"{hf_layer}.hc_{kind}_base",
            ),
            HCAlphaMapping(pre, scale, 0),
            HCAlphaMapping(post, scale, 1),
            HCAlphaMapping(res, scale, 2),
        ]

    @staticmethod
    def _kda_attn_mappings(mg_layer: str, hf_layer: str) -> list:
        """KDA (linear attention, two-stage gates) weight mappings.

        MCore KDA fuses q|k|v into ``in_proj`` (order q|k|v along dim 0) and
        q|k|v conv into a single ``conv1d``; HF stores them separately.
        f_a/g_a are replicated; f_b/g_b, in_proj, beta_proj, conv1d, A_log,
        dt_bias are TP column-sharded.
        """
        sa = f"{mg_layer}.inner_layer.self_attention"
        hf_attn = f"{hf_layer}.self_attn"
        return [
            AutoMapping(f"{mg_layer}.inner_layer.input_layernorm.weight", f"{hf_layer}.input_layernorm.weight"),
            # Fused QKV in_proj (q|k|v, column-parallel, no GQA interleave).
            _ColumnParallelConcatMapping(
                f"{sa}.in_proj.weight",
                [
                    f"{hf_attn}.q_proj.weight",
                    f"{hf_attn}.k_proj.weight",
                    f"{hf_attn}.v_proj.weight",
                ],
            ),
            # Beta projection (separate, column-parallel).
            ColumnParallelMapping(f"{sa}.beta_proj.weight", f"{hf_attn}.b_proj.weight"),
            # Two-stage low-rank forget gate: f_a replicated, f_b TP-sharded.
            ReplicatedMapping(f"{sa}.f_a_proj.weight", f"{hf_attn}.f_a_proj.weight"),
            ColumnParallelMapping(f"{sa}.f_b_proj.weight", f"{hf_attn}.f_b_proj.weight"),
            # Two-stage low-rank output gate: g_a replicated, g_b TP-sharded.
            ReplicatedMapping(f"{sa}.g_a_proj.weight", f"{hf_attn}.g_a_proj.weight"),
            ColumnParallelMapping(f"{sa}.g_b_proj.weight", f"{hf_attn}.g_b_proj.weight"),
            # Fused q|k|v conv1d (depthwise). [conv_dim, 1, kernel] -> concat dim 0.
            _ColumnParallelConcatMapping(
                f"{sa}.conv1d.weight",
                [
                    f"{hf_attn}.q_conv1d.weight",
                    f"{hf_attn}.k_conv1d.weight",
                    f"{hf_attn}.v_conv1d.weight",
                ],
            ),
            # Kernel parameters (TP-sharded along dim 0).
            ColumnParallelMapping(f"{sa}.A_log", f"{hf_attn}.A_log"),
            ColumnParallelMapping(f"{sa}.dt_bias", f"{hf_attn}.dt_bias"),
            # Output norm + projection.
            ReplicatedMapping(f"{sa}.out_norm.weight", f"{hf_attn}.o_norm.weight"),
            RowParallelMapping(f"{sa}.out_proj.weight", f"{hf_attn}.o_proj.weight"),
        ]

    @staticmethod
    def _dsa_attn_mappings(mg_layer: str, hf_layer: str) -> list:
        """DSA (MLA + kpool indexer) weight mappings."""
        sa = f"{mg_layer}.inner_layer.self_attention"
        idx = f"{sa}.core_attention.indexer"
        hf_attn = f"{hf_layer}.self_attn"
        hf_idx = f"{hf_attn}.indexer"
        return [
            AutoMapping(f"{mg_layer}.inner_layer.input_layernorm.weight", f"{hf_layer}.input_layernorm.weight"),
            # MLA projections. The down-projections (q_a/kv_a) are TELinear with
            # parallel_mode="duplicated" → replicated. The up-projections (q_b/kv_b)
            # are TEColumnParallelLinear → column-parallel.
            ReplicatedMapping(f"{sa}.linear_q_down_proj.weight", f"{hf_attn}.q_a_proj.weight"),
            ColumnParallelMapping(f"{sa}.linear_q_up_proj.weight", f"{hf_attn}.q_b_proj.weight"),
            ReplicatedMapping(f"{sa}.q_layernorm.weight", f"{hf_attn}.q_a_layernorm.weight"),
            ReplicatedMapping(f"{sa}.linear_kv_down_proj.weight", f"{hf_attn}.kv_a_proj_with_mqa.weight"),
            ColumnParallelMapping(f"{sa}.linear_kv_up_proj.weight", f"{hf_attn}.kv_b_proj.weight"),
            ReplicatedMapping(f"{sa}.kv_layernorm.weight", f"{hf_attn}.kv_a_layernorm.weight"),
            RowParallelMapping(f"{sa}.linear_proj.weight", f"{hf_attn}.o_proj.weight"),
            # DSA kpool indexer. All indexer projections are parallel_mode="duplicated"
            # (replicated) in MCore — see dsa.py build_module(..., parallel_mode="duplicated").
            ReplicatedMapping(f"{idx}.linear_wq_b.weight", f"{hf_idx}.wq_b.weight"),
            ReplicatedMapping(f"{idx}.linear_wk.weight", f"{hf_idx}.wk.weight"),
            ReplicatedMapping(f"{idx}.k_norm.weight", f"{hf_idx}.k_norm.weight"),
            ReplicatedMapping(f"{idx}.k_norm.bias", f"{hf_idx}.k_norm.bias"),
            ReplicatedMapping(f"{idx}.linear_weights_proj.weight", f"{hf_idx}.weights_proj.weight"),
            ReplicatedMapping(f"{idx}.index_kpool_compress_ape", f"{hf_idx}.index_kpool_compress_ape"),
            ReplicatedMapping(f"{idx}.index_kpool_compress_gate", f"{hf_idx}.index_kpool_compress_gate"),
        ]

    @staticmethod
    def _dense_ffn_mappings(mg_layer: str, hf_layer: str) -> list:
        """Dense MLP FFN mappings."""
        inner = f"{mg_layer}.inner_layer"
        hf_mlp = f"{hf_layer}.mlp"
        return [
            AutoMapping(
                f"{inner}.mlp.linear_fc1.layer_norm_weight",
                f"{hf_layer}.post_attention_layernorm.weight",
            ),
            RowParallelMapping(f"{inner}.mlp.linear_fc2.weight", f"{hf_mlp}.down_proj.weight"),
            GatedMLPMapping(
                f"{inner}.mlp.linear_fc1.weight",
                gate=f"{hf_mlp}.gate_proj.weight",
                up=f"{hf_mlp}.up_proj.weight",
            ),
        ]

    @staticmethod
    def _moe_ffn_mappings(mg_layer: str, hf_layer: str) -> list:
        """MoE FFN (router + shared expert + routed experts) mappings."""
        inner = f"{mg_layer}.inner_layer"
        mlp = f"{inner}.mlp"
        hf_mlp = f"{hf_layer}.mlp"
        return [
            AutoMapping(
                f"{inner}.pre_mlp_layernorm.weight",
                f"{hf_layer}.post_attention_layernorm.weight",
            ),
            AutoMapping(f"{mlp}.router.weight", f"{hf_mlp}.gate.weight"),
            AutoMapping(f"{mlp}.router.expert_bias", f"{hf_mlp}.gate.e_score_correction_bias"),
            # Shared expert (always-on; no per-token gate scalar in GLM-5.3-Flash).
            RowParallelMapping(
                f"{mlp}.shared_experts.linear_fc2.weight",
                f"{hf_mlp}.shared_experts.down_proj.weight",
            ),
            GatedMLPMapping(
                f"{mlp}.shared_experts.linear_fc1.weight",
                gate=f"{hf_mlp}.shared_experts.gate_proj.weight",
                up=f"{hf_mlp}.shared_experts.up_proj.weight",
            ),
            # Routed experts (grouped-GEMM + sequential fallback).
            RowParallelMapping(
                f"{mlp}.experts.linear_fc2.weight*",
                f"{hf_mlp}.experts.*.down_proj.weight",
            ),
            RowParallelMapping(
                f"{mlp}.experts.local_experts.*.linear_fc2.weight",
                f"{hf_mlp}.experts.*.down_proj.weight",
            ),
            GatedMLPMapping(
                f"{mlp}.experts.linear_fc1.weight*",
                gate=f"{hf_mlp}.experts.*.gate_proj.weight",
                up=f"{hf_mlp}.experts.*.up_proj.weight",
            ),
            GatedMLPMapping(
                f"{mlp}.experts.local_experts.*.linear_fc1.weight",
                gate=f"{hf_mlp}.experts.*.gate_proj.weight",
                up=f"{hf_mlp}.experts.*.up_proj.weight",
            ),
        ]
