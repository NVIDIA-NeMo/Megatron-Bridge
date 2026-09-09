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
single registered bridge + provider + model — there is no separate "VL" split.

Each HF decoder layer maps to two physical layers: attention at 2N and FFN at 2N+1.
Checkpoint keys use ``model.language_model``; mixed FP8 weights are dequantized on
import. Vision weights (``model.visual.**``) are loaded replicated and stay BF16 even
under FP8. The VLM wrapper degrades to text-only when ``pixel_values`` is None.
"""

import types
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols
from megatron.core.tensor_parallel import scatter_to_sequence_parallel_region
from megatron.core.transformer import ModuleSpec
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
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.hybrid.hybrid_provider import get_default_hybrid_stack_spec
from megatron.bridge.models.hybrid_mla_provider import HybridMLAModelProvider
from megatron.bridge.training.utils.packed_seq_utils import preprocess_packed_seqs
from megatron.bridge.utils.common_utils import hook_hf_module_setattr_for_tp_grad_sync


try:
    from transformers import Glm5NextForConditionalGeneration
except ImportError:  # pragma: no cover
    Glm5NextForConditionalGeneration = "Glm5NextForConditionalGeneration"  # type: ignore[assignment]


if TYPE_CHECKING:
    from megatron.core.packed_seq_params import PackedSeqParams


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _glm53_text_config(hf_config):
    """Return the nested text_config for the GLM-5.3-Flash VLM."""
    text_config = getattr(hf_config, "text_config", None)
    return text_config if text_config is not None else hf_config


def _glm53_layer_types(hf_config) -> list[str]:
    """Per-HF-layer attention type: 'linear_attention' (KDA) or 'deepseek_sparse_attention' (DSA)."""
    text_config = _glm53_text_config(hf_config)
    layer_types = list(getattr(text_config, "layer_types", []))
    num_layers = text_config.num_hidden_layers
    if len(layer_types) != num_layers:
        raise ValueError(f"layer_types has {len(layer_types)} entries, but num_hidden_layers is {num_layers}.")
    return layer_types


def _glm53_mlp_layer_types(hf_config) -> list[str]:
    """Per-HF-layer MLP type: 'dense' or 'sparse'."""
    text_config = _glm53_text_config(hf_config)
    mlp_types = getattr(text_config, "mlp_layer_types", None)
    if mlp_types is None:
        first_k_dense = getattr(text_config, "first_k_dense_replace", min(3, text_config.num_hidden_layers))
        mlp_types = ["dense"] * first_k_dense + ["sparse"] * (text_config.num_hidden_layers - first_k_dense)
    return list(mlp_types)


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
        if megatron_weights is None:
            return {}
        weights = self.maybe_dequantize(megatron_weights)
        config = self._get_config(megatron_module) if megatron_module is not None else None
        sizes = self._section_sizes(config)
        sharded = self.tp_size > 1 and not _module_uses_fsdp(megatron_module)
        if sizes is None:
            if weights.shape[0] % len(self._hf_order):
                raise ValueError("Cannot infer unequal KDA Q/K/V sections without the model config.")
            rows = weights.shape[0] // len(self._hf_order) * (self.tp_size if sharded else 1)
            sizes = [rows] * len(self._hf_order)

        if sharded:
            if any(size % self.tp_size for size in sizes):
                raise ValueError("Each KDA Q/K/V section must be divisible by tensor parallel size.")
            local_sizes = [size // self.tp_size for size in sizes]
            shards = self._tp_mapping.gather_from_tp_ranks(weights)
            parts = [
                torch.cat(section_shards, dim=0)
                for section_shards in zip(*(shard.split(local_sizes, dim=0) for shard in shards), strict=True)
            ]
        else:
            parts = weights.split(sizes, dim=0)
        return dict(zip(self._hf_order, parts, strict=True))

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
        if megatron_weights is None:
            return {}
        megatron_weights = self.maybe_dequantize(megatron_weights)
        if self.tp_size == 1:
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


def glm53_hybrid_stack_spec(config: HybridMLAModelProvider) -> ModuleSpec:
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
class GLM53FlashModelProvider(HybridMLAModelProvider):
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
        """Build the VLM wrapper and apply explicitly requested freeze options."""
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
    """Inject HF image/video features into the hybrid language model's embeddings."""

    def __init__(
        self, config, pre_process: bool = True, post_process: bool = True, vp_stage: Optional[int] = None
    ) -> None:
        super().__init__(config=config)

        self.pre_process = pre_process
        self.post_process = post_process
        self.vp_stage = vp_stage

        if pre_process:
            from transformers.models.glm5_next.modeling_glm5_next import (
                Glm5NextModel,
                Glm5NextVisionModel,
            )

            self.visual = Glm5NextVisionModel._from_config(config.vision_config)
            hook_hf_module_setattr_for_tp_grad_sync(self.visual)

        self.language_model = self.config.provide_language_model(
            pre_process=pre_process, post_process=post_process, vp_stage=vp_stage
        )

        self.share_embeddings_and_output_weights = config.share_embeddings_and_output_weights
        self.shared_embedding_or_output_weight = self.language_model.shared_embedding_or_output_weight

        # Monkey-patch HF image/video feature extraction + placeholder-mask logic.
        self.get_image_features = types.MethodType(Glm5NextModel.get_image_features, self)
        self.get_video_features = types.MethodType(Glm5NextModel.get_video_features, self)
        self.get_placeholder_mask = types.MethodType(Glm5NextModel.get_placeholder_mask, self)

        self.config.spatial_merge_size = getattr(self.config.vision_config, "spatial_merge_size", 2)

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
        temperature: Optional[float] = None,
        output_processor=None,
        output_processor_context=None,
        *,
        loss_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Inject vision features before packing BSHD inputs into THD layout.

        Placeholder masks require full BSHD sequences. Packing and sequence-parallel
        scattering happen afterward; NoPE attention uses only packed_seq_params.
        """
        if self.pre_process:
            if inputs_embeds is None:
                inputs_embeds = self.language_model.embedding(
                    input_ids=input_ids, position_ids=None
                )  # [seq_len, batch, hidden] (full seq, scatter_embedding_sequence_parallel=False)
                # Transpose to HF format [batch, seq_len, hidden].
                inputs_embeds = inputs_embeds.transpose(1, 0).contiguous()

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

            use_fp8_padding = self.config.fp8 in ("e4m3", "hybrid")
            inputs_embeds_thd = preprocess_packed_seqs(
                inputs_embeds,
                attention_mask,
                pg_collection=self.config._pg_collection,
                use_fp8_padding=use_fp8_padding,
            )[0]
            # Megatron expects [seq, batch, hidden] with batch=1 for THD.
            inputs_embeds_thd = inputs_embeds_thd.transpose(1, 0).contiguous()  # [total, 1, hidden]

            # MTP re-embeds input_ids, so it also needs the packed token layout.
            input_ids_thd = preprocess_packed_seqs(
                input_ids,
                attention_mask,
                pg_collection=self.config._pg_collection,
                use_fp8_padding=use_fp8_padding,
            )[0]

            if self.config.sequence_parallel:
                tp_group = self.config._pg_collection.tp if self.config._pg_collection is not None else None
                inputs_embeds_thd = scatter_to_sequence_parallel_region(inputs_embeds_thd, group=tp_group)

        # verl's output processor needs hidden states, not the MTP auxiliary loss.
        return self.language_model.forward(
            input_ids=input_ids_thd,
            position_ids=None,
            attention_mask=None,
            decoder_input=inputs_embeds_thd,
            labels=labels,
            loss_mask=loss_mask,
            runtime_gather_output=runtime_gather_output,
            packed_seq_params=packed_seq_params,
            output_processor=output_processor,
            output_processor_context=output_processor_context,
            compute_mtp_loss=False,
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

    _provider_cls = GLM53FlashModelProvider

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM):
        hf_config = hf_pretrained.config
        text_config = _glm53_text_config(hf_config)

        # Build base provider kwargs from the nested text_config (the VLM top-level
        # config lacks hidden_size/num_layers/etc. that the text sub-model owns).
        provider_kwargs = self.hf_config_to_provider_kwargs(text_config)
        provider_kwargs.pop("_mla_rope_params", None)
        valid_fields = self._provider_cls.__dataclass_fields__
        provider = self._provider_cls(**{k: v for k, v in provider_kwargs.items() if k in valid_fields})

        provider.normalization = "RMSNorm"
        provider.gated_linear_unit = True
        provider.add_bias_linear = False
        provider.share_embeddings_and_output_weights = bool(getattr(hf_config, "tie_word_embeddings", False))
        provider.qk_layernorm = True
        provider.multi_latent_attention = True
        provider.sequence_parallel = True

        # ---- MoE ----
        provider.moe_grouped_gemm = True
        provider.moe_router_pre_softmax = True
        provider.moe_token_dispatcher_type = "alltoall"
        provider.moe_flex_dispatcher_backend = "hybridep"
        provider.moe_flex_dispatcher_num_sms = 16
        provider.moe_router_load_balancing_type = "seq_aux_loss"
        provider.moe_shared_expert_overlap = False
        provider.moe_router_score_function = text_config.scoring_func  # "sigmoid"
        provider.moe_router_enable_expert_bias = True
        provider.moe_router_dtype = "fp32"
        provider.moe_permute_fusion = True
        provider.moe_router_topk = text_config.num_experts_per_tok  # 8
        provider.moe_router_topk_scaling_factor = getattr(text_config, "routed_scaling_factor", 1.0)  # 2.5
        provider.norm_topk_prob = text_config.norm_topk_prob
        provider.moe_aux_loss_coeff = getattr(text_config, "router_aux_loss_coef", 0.001)
        provider.activation_func_clamp_value = getattr(text_config, "swiglu_limit", 0.0)  # 10.0
        provider.moe_shared_expert_intermediate_size = text_config.moe_intermediate_size * text_config.n_shared_experts
        # GLM-5.3-Flash's shared expert is always-on (no per-token gate scalar in the
        # checkpoint), so disable MCore's optional shared-expert gate.
        provider.moe_shared_expert_gate = False

        provider.hidden_dropout = 0.0
        provider.attention_softmax_in_fp32 = False
        provider.make_vocab_size_divisible_by = 1280
        provider.actual_vocab_size = text_config.vocab_size

        # ---- Hybrid layer pattern ----
        attn_types = _glm53_layer_types(hf_config)
        mlp_types = _glm53_mlp_layer_types(hf_config)
        attn_sym = {
            "linear_attention": Symbols.KDA,
            "deepseek_sparse_attention": Symbols.DS_ATTENTION,
        }
        ffn_sym = {"dense": Symbols.MLP, "sparse": Symbols.MOE}
        pattern_chars = []
        # Router replay indexes physical modules: each HF block becomes attention + FFN.
        moe_layer_freq = []
        for at, mt in zip(attn_types, mlp_types, strict=True):
            if at not in attn_sym:
                raise ValueError(f"Unsupported GLM-5.3-Flash attention layer type: {at!r}")
            if mt not in ffn_sym:
                raise ValueError(f"Unsupported GLM-5.3-Flash MLP layer type: {mt!r}")
            pattern_chars.append(attn_sym[at] + ffn_sym[mt])
            moe_layer_freq.append(0)  # attention module is never MoE
            moe_layer_freq.append(1 if mt == "sparse" else 0)  # FFN module
        main_pattern = "".join(pattern_chars)
        provider.hybrid_layer_pattern = main_pattern
        provider.num_layers = len(main_pattern)
        provider.hybrid_stack_spec = glm53_hybrid_stack_spec
        provider.moe_layer_freq = moe_layer_freq

        # ---- MLA geometry (NoPE) ----
        # GLM-5.3-Flash: mla_use_nope=True -> qk_rope_head_dim=0, qk_nope_head_dim=256, qk_head_dim=256.
        provider.qk_pos_emb_head_dim = text_config.qk_head_dim - text_config.qk_nope_head_dim  # 0
        provider.q_lora_rank = text_config.q_lora_rank  # 1536
        provider.kv_lora_rank = text_config.kv_lora_rank  # 512
        provider.v_head_dim = text_config.v_head_dim  # 256
        provider.qk_head_dim = text_config.qk_head_dim  # 256
        provider.num_attention_heads = text_config.num_attention_heads  # 64
        provider.num_key_value_heads = text_config.num_key_value_heads  # 64
        # HF head_dim=0 is a NoPE sentinel, not a valid attention channel width.
        provider.kv_channels = text_config.v_head_dim  # 256
        provider.rope_type = "yarn"
        provider.rotary_base = 10000.0
        provider.rotary_scaling_factor = 1.0
        provider.mscale = 1.0
        provider.mscale_all_dim = 1.0
        provider.apply_rope_fusion = False
        provider.rotary_interleaved = False
        provider.dsa_indexer_rope_interleaved = getattr(text_config, "indexer_rope_interleave", True)
        provider.cp_comm_type = "allgather"

        # ---- DSA indexer (kpool) ----
        provider.experimental_attention_variant = "dsa"
        provider.dsa_indexer_head_dim = text_config.index_head_dim  # 128
        provider.dsa_indexer_n_heads = text_config.index_n_heads  # 32
        provider.dsa_indexer_topk = text_config.index_topk  # 2048
        provider.dsa_indexer_loss_coeff = 0.0
        provider.dsa_indexer_use_sparse_loss = False
        provider.dsa_indexer_rotate_activation = False
        provider.dsa_indexer_kpool_fp8 = True
        provider.dsa_indexer_k_norm_epsilon = 1e-6
        provider.dsa_indexer_kpool = int(getattr(text_config, "index_kpool", 1))  # 4
        provider.dsa_indexer_kpool_always_select_tail = bool(
            getattr(text_config, "index_kpool_always_select_tail", False)
        )
        provider.dsa_indexer_topk_freq = 1
        provider.dsa_indexer_skip_topk_offset = 0

        # ---- KDA two-stage gates ----
        provider.kda_two_stage_gates = True
        provider.kda_safe_gate = True
        provider.kda_lower_bound = float(text_config.linear_attn_config.get("gate_lower_bound", -5.0))
        provider.gdn_pre_gated_delta_rule_fusion = False
        provider.gdn_conv_pad_alignment = None
        lin = text_config.linear_attn_config
        # KDA uses equal query/key/value head geometry.
        provider.linear_num_key_heads = int(lin["num_heads"])  # 64
        provider.linear_num_value_heads = int(lin["num_heads"])  # 64 (qkv equal)
        provider.linear_key_head_dim = int(lin["head_dim"])  # 128
        provider.linear_value_head_dim = int(lin["head_dim"])  # 128
        provider.linear_conv_kernel_dim = int(lin["short_conv_kernel_size"])  # 4

        # ---- mHC (multi-stream Hyper-Connections) ----
        provider.enable_mhc_connections = True
        provider.mhc_num_residual_streams = int(getattr(text_config, "hc_mult", 4))
        provider.mhc_sinkhorn_iterations = int(getattr(text_config, "hc_sinkhorn_iters", 20))
        provider.mhc_learned_output_contract = False
        provider.mhc_norm_eps_inside_sqrt = True
        provider.mhc_keep_mappings_in_fp32 = True

        # ---- MTP (1 layer, DSA + MoE) ----
        # MTP has its own pattern and is excluded from num_layers/moe_layer_freq.
        num_mtp = int(getattr(text_config, "num_nextn_predict_layers", 0) or 0)
        provider.mtp_num_layers = num_mtp or None
        provider.mtp_hybrid_override_pattern = "DE"
        provider.mtp_use_repeated_layer = True
        provider.keep_mtp_spec_in_bf16 = True
        if num_mtp:
            provider.mtp_loss_scaling_factor = 0.1

        provider.persist_layer_norm = True
        provider.gradient_accumulation_fusion = True
        provider.bias_dropout_fusion = True

        # ---- Vision tower (replicated HF Glm5NextVisionModel) ----
        provider.vision_config = getattr(hf_config, "vision_config", None)
        provider.image_token_id = getattr(hf_config, "image_token_id", 154854)
        provider.video_token_id = getattr(hf_config, "video_token_id", 154855)
        provider.image_start_token_id = getattr(hf_config, "image_start_token_id", 154830)
        provider.image_end_token_id = getattr(hf_config, "image_end_token_id", 154831)
        provider.video_start_token_id = getattr(hf_config, "video_start_token_id", 154832)
        provider.video_end_token_id = getattr(hf_config, "video_end_token_id", 154833)
        provider.spatial_merge_size = getattr(provider.vision_config, "spatial_merge_size", 2)

        return provider

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
        text_config = _glm53_text_config(hf_config)
        attn_types = _glm53_layer_types(hf_config)
        mlp_types = _glm53_mlp_layer_types(hf_config)
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
                        "model.language_model.norm.weight",
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
