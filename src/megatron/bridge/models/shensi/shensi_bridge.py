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

"""Bridge between the Shensi HuggingFace model and its Megatron implementation."""

from typing import Any, Dict, List, Optional

import torch
from torch.distributed._tensor import DTensor
from transformers import PretrainedConfig

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ColumnParallelMapping,
    FusedExpertMapping,
    FusedGatedExpertMapping,
    GatedMLPMapping,
    MegatronParamMapping,
    ReplicatedMapping,
    RowParallelMapping,
    _module_uses_fsdp,
)
from megatron.bridge.models.conversion.utils import get_module_and_param_from_name
from megatron.bridge.models.shensi.configuration_shensi import (
    resolve_attn_res_block_layer_types,
    resolve_indexer_types,
    resolve_layer_types,
    resolve_mlp_layer_types,
    resolve_rosa_layer_ids,
)
from megatron.bridge.models.shensi.modeling_shensi import ShensiModel
from megatron.bridge.models.shensi.shensi_builder import ShensiModelConfig
from megatron.bridge.models.shensi.shensi_provider import (
    SHENSI_HARD_DEFAULTS,
    ShensiModelProvider,
    shensi_derived_field_values,
)


def _get(config, name: str, default=None):
    value = getattr(config, name, default)
    return default if value is None else value


def shensi_transformer_field_values(hf_config, *, mtp_num_layers: int = 0) -> dict[str, Any]:
    """Flat Megatron field mapping for a Shensi HF config."""
    fields: dict[str, Any] = dict(shensi_derived_field_values(hf_config, mtp_num_layers=mtp_num_layers))
    fields.update(SHENSI_HARD_DEFAULTS)
    fields.update(
        gated_linear_unit=True,
        moe_grouped_gemm=True,
        moe_router_pre_softmax=False,
        moe_router_score_function=hf_config.scoring_func,
        moe_router_topk=hf_config.num_experts_per_tok,
        moe_router_topk_scaling_factor=hf_config.routed_scaling_factor,
        moe_token_dispatcher_type="allgather",
        moe_router_dtype="fp32",
        moe_layer_freq=1,
        activation_func_clamp_value=hf_config.swiglu_limit,
        layernorm_epsilon=hf_config.rms_norm_eps,
        init_method_std=hf_config.initializer_range,
        attention_dropout=hf_config.attention_dropout,
        hidden_dropout=0.0,
        masked_softmax_fusion=False,
        persist_layer_norm=False,
        hidden_size=hf_config.hidden_size,
        num_layers=hf_config.num_hidden_layers,
        vocab_size=hf_config.vocab_size,
        seq_length=int(_get(hf_config, "max_position_embeddings", 4096)),
        share_embeddings_and_output_weights=bool(_get(hf_config, "tie_word_embeddings", True)),
        rotary_percent=1.0,
        num_query_groups=1,
        rope_type="rope",
        original_max_position_embeddings=4096,
    )
    fields["kv_lora_rank"] = int(hf_config.deepemb_kv_rank)
    return fields


class _StreamTokenEmbeddingMapping(ColumnParallelMapping):
    """Vocabulary-parallel per-stream token table with Megatron-Core's padded vocabulary.

    HuggingFace stores the table at the unpadded vocabulary size; Megatron-Core pads the
    vocabulary to the TP-divisible size and shards it like ``embedding.word_embeddings``. Pad on
    import and crop on export so the two layouts stay interchangeable while the table occupies
    ``1 / TP`` of its reference footprint per rank.
    """

    def __init__(self, megatron_param: str, hf_param: str, hf_vocab_size: int):
        super().__init__(megatron_param, hf_param)
        self.hf_vocab_size = int(hf_vocab_size)

    def _padded_vocab_size(self, megatron_module) -> int:
        _, target = get_module_and_param_from_name(megatron_module, self.megatron_param)
        target = target.orig_param if isinstance(target, DTensor) else target
        return int(target.shape[0]) * self.tp_size

    def hf_to_megatron(self, hf_weights: torch.Tensor, megatron_module) -> torch.Tensor:
        if hf_weights is not None:
            padded = self._padded_vocab_size(megatron_module)
            if hf_weights.shape[0] < padded:
                padding = hf_weights.new_zeros((padded - hf_weights.shape[0], *hf_weights.shape[1:]))
                hf_weights = torch.cat([hf_weights, padding], dim=0)
            elif hf_weights.shape[0] > padded:
                hf_weights = hf_weights[:padded]
        return super().hf_to_megatron(hf_weights, megatron_module)

    def megatron_to_hf(self, megatron_weights: Optional[torch.Tensor], megatron_module) -> dict:
        weights = super().megatron_to_hf(megatron_weights, megatron_module)
        return {name: weight[: self.hf_vocab_size] for name, weight in weights.items()}


# Adapted from megatron/bridge/models/conversion/param_mapping.py (QKVMapping) with uneven
# sections: KDA's in_proj shards q/k/v separately (qk_dim != v_dim), so the merge/split
# keeps the native delegate-to-the-TP-mapping shape instead of head interleaving.
class _ColumnParallelConcatMapping(MegatronParamMapping[dict]):
    """Map separate HF tensors to rank-local ``[part_0 | part_1 | ...]`` column shards."""

    def __init__(self, megatron_param: str, hf_params: List[str]):
        super().__init__(megatron_param, {str(i): p for i, p in enumerate(hf_params)})
        self._hf_order = list(hf_params)
        self._tp_mapping = ColumnParallelMapping(megatron_param, megatron_param)

    def resolve(self, captures):
        resolved_mg, resolved_hf = self._resolve_names(captures)
        return _ColumnParallelConcatMapping(resolved_mg, list(resolved_hf.values()))

    def _shard_per_rank(self, merged: torch.Tensor, section_sizes: List[int]) -> List[torch.Tensor]:
        if any(size % self.tp_size for size in section_sizes):
            raise ValueError("Each Shensi KDA section must be divisible by tensor parallel size.")
        sections = merged.split(section_sizes, dim=0)
        return [
            torch.cat(shards, dim=0)
            for shards in zip(*(section.chunk(self.tp_size, dim=0) for section in sections), strict=True)
        ]

    def hf_to_megatron(self, hf_weights: dict, megatron_module) -> torch.Tensor:
        splits = None
        if self.tp_rank == 0:
            sections = [hf_weights[str(i)] for i in range(len(self._hf_order))]
            merged = torch.cat(sections, dim=0)
            if self.tp_size == 1:
                return merged
            splits = self._shard_per_rank(merged, [section.shape[0] for section in sections])

        normalized = self._tp_mapping._normalize_expert_param_name(self.megatron_param)
        _, target = get_module_and_param_from_name(megatron_module, normalized)
        shape = target.orig_param.shape if isinstance(target, DTensor) else target.shape
        return self._tp_mapping.scatter_to_tp_ranks(splits, shape, target.dtype, target.device)

    def megatron_to_hf(self, megatron_weights: Optional[torch.Tensor], megatron_module) -> dict:
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
            return dict(zip(self._hf_order, weights.split(sizes, dim=0), strict=True))

        if materialized:
            shards = list(weights.chunk(self.tp_size, dim=0))
        else:
            shards = self._tp_mapping.gather_from_tp_ranks(weights)
        if sizes is None:
            local_sizes = self._equal_sections(shards[0].shape[0])
        else:
            if any(size % self.tp_size for size in sizes):
                raise ValueError("Each Shensi KDA section must be divisible by tensor parallel size.")
            local_sizes = [size // self.tp_size for size in sizes]
        parts = [
            torch.cat(section_shards, dim=0)
            for section_shards in zip(*(shard.split(local_sizes, dim=0) for shard in shards), strict=True)
        ]
        return dict(zip(self._hf_order, parts, strict=True))

    def _equal_sections(self, rows: int) -> List[int]:
        if rows % len(self._hf_order):
            raise ValueError("Cannot infer unequal KDA sections without the model config.")
        return [rows // len(self._hf_order)] * len(self._hf_order)

    def _section_sizes(self, config):
        if config is None:
            return None
        qk = config.linear_key_head_dim * config.linear_num_key_heads
        v = config.linear_value_head_dim * config.linear_num_value_heads
        return [qk, qk, v]


def _layer_mappings(
    megatron_prefix: str,
    hf_prefix: str,
    *,
    layer_type: str,
    mlp_type: str,
    hc_kernels: List[int],
    hc_conditional: bool,
    routed_streams: int,
    is_full_indexer: bool,
    is_first_full_indexer: bool,
    is_first_dense: bool,
    has_kv_emb: bool,
    has_rosa: bool,
    rosa_tied: bool = True,
    first_rosa: int = -1,
    mtp_head: bool = False,
) -> list:
    """Mappings of one decoder-shaped layer."""
    fp, hp = megatron_prefix, hf_prefix
    out = [
        ReplicatedMapping(fp + "input_layernorm.weight", hp + "input_layernorm.weight"),
        ReplicatedMapping(fp + "pre_mlp_layernorm.weight", hp + "post_attention_layernorm.weight"),
    ]
    for name in ("attn_hc", "ffn_hc"):
        out += [
            ReplicatedMapping(fp + f"{name}.fn", hp + f"{name}.fn"),
            ReplicatedMapping(fp + f"{name}.base", hp + f"{name}.base"),
            ReplicatedMapping(fp + f"{name}.scale", hp + f"{name}.scale"),
        ]
        if hc_conditional:
            out.append(ReplicatedMapping(fp + f"{name}.decay_logit", hp + f"{name}.decay_logit"))
            for idx in range(routed_streams):
                out += [
                    ReplicatedMapping(fp + f"{name}.adapters.{idx}.0.weight", hp + f"{name}.adapters.{idx}.0.weight"),
                    ReplicatedMapping(fp + f"{name}.adapters.{idx}.1.weight", hp + f"{name}.adapters.{idx}.1.weight"),
                ]
        if name == "ffn_hc":
            for idx in range(len(hc_kernels)):
                out += [
                    ReplicatedMapping(fp + f"{name}.convs.{idx}.weight", hp + f"{name}.convs.{idx}.weight"),
                    ReplicatedMapping(fp + f"{name}.convs.{idx}.bias", hp + f"{name}.convs.{idx}.bias"),
                ]

    for name in ("self_attention_res", "mlp_res"):
        out += [
            ReplicatedMapping(fp + f"{name}.g_proj.weight", hp + f"{name}.g_proj.weight"),
            ReplicatedMapping(fp + f"{name}.g_proj.bias", hp + f"{name}.g_proj.bias"),
            ReplicatedMapping(fp + f"{name}.q_proj.weight", hp + f"{name}.q_proj.weight"),
            ReplicatedMapping(fp + f"{name}.k_proj.weight", hp + f"{name}.k_proj.weight"),
            ReplicatedMapping(fp + f"{name}.scale", hp + f"{name}.scale"),
        ]

    if layer_type == "indexed_attention":
        out += [
            ReplicatedMapping(fp + "self_attention.q_a_proj.weight", hp + "self_attn.q_a_proj.weight"),
            ReplicatedMapping(fp + "self_attention.q_a_layernorm.weight", hp + "self_attn.q_a_layernorm.weight"),
            # The reference packs the query and the output gate into one projection.
            ColumnParallelMapping(fp + "self_attention.q_b_proj.weight", hp + "self_attn.q_b_proj.weight"),
            ReplicatedMapping(fp + "self_attention.kv.weight", hp + "self_attn.kv.weight"),
            ReplicatedMapping(fp + "self_attention.kv_up.weight", hp + "self_attn.kv_up.weight"),
            ReplicatedMapping(fp + "self_attention.kv_norm.weight", hp + "self_attn.kv_norm.weight"),
            RowParallelMapping(fp + "self_attention.o_proj.weight", hp + "self_attn.o_proj.weight"),
        ]
        if has_kv_emb:
            out.append(
                ReplicatedMapping(fp + "self_attention.kv_emb.weight", hp + "self_attn.kv_emb.weight"),
            )
        if is_full_indexer:
            out += [
                ReplicatedMapping(
                    fp + "self_attention.indexer.linear_wq_b.weight", hp + "self_attn.indexer.wq_b.weight"
                ),
                ReplicatedMapping(fp + "self_attention.indexer.linear_key.weight", hp + "self_attn.indexer.k.weight"),
                ReplicatedMapping(
                    fp + "self_attention.indexer.linear_key_up.weight", hp + "self_attn.indexer.k_up.weight"
                ),
                ReplicatedMapping(fp + "self_attention.indexer.k_norm.weight", hp + "self_attn.indexer.k_norm.weight"),
                ReplicatedMapping(fp + "self_attention.indexer.k_norm.bias", hp + "self_attn.indexer.k_norm.bias"),
                ReplicatedMapping(
                    fp + "self_attention.indexer.linear_weights_proj.weight",
                    hp + "self_attn.indexer.weights_proj.weight",
                ),
            ]
        if is_first_full_indexer:
            out.append(
                ReplicatedMapping(
                    fp + "self_attention.indexer.linear_key_emb.weight", hp + "self_attn.indexer.k_emb.weight"
                ),
            )
    else:
        out += [
            _ColumnParallelConcatMapping(
                fp + "self_attention.in_proj.weight",
                [hp + "self_attn.q_proj.weight", hp + "self_attn.k_proj.weight", hp + "self_attn.v_proj.weight"],
            ),
            ColumnParallelMapping(fp + "self_attention.conv1d.weight", hp + "self_attn.conv1d.weight"),
            ColumnParallelMapping(fp + "self_attention.beta_proj.weight", hp + "self_attn.b_proj.weight"),
            ReplicatedMapping(fp + "self_attention.f_a_proj.weight", hp + "self_attn.forget_gate.f_a_proj.weight"),
            ColumnParallelMapping(fp + "self_attention.f_b_proj.weight", hp + "self_attn.forget_gate.f_b_proj.weight"),
            ColumnParallelMapping(fp + "self_attention.A_log", hp + "self_attn.forget_gate.A_log"),
            ColumnParallelMapping(fp + "self_attention.dt_bias", hp + "self_attn.forget_gate.dt_bias"),
            ReplicatedMapping(fp + "self_attention.g_a_proj.weight", hp + "self_attn.g_a_proj.weight"),
            ColumnParallelMapping(fp + "self_attention.g_b_proj.weight", hp + "self_attn.g_b_proj.weight"),
            ReplicatedMapping(fp + "self_attention.out_norm.weight", hp + "self_attn.o_norm.weight"),
            RowParallelMapping(fp + "self_attention.out_proj.weight", hp + "self_attn.o_proj.weight"),
        ]

    if mlp_type == "dense":
        out += [
            GatedMLPMapping(
                fp + "mlp.linear_fc1.weight", gate=hp + "mlp.gate_proj.weight", up=hp + "mlp.up_proj.weight"
            ),
            RowParallelMapping(fp + "mlp.linear_fc2.weight", hp + "mlp.down_proj.weight"),
            ReplicatedMapping(fp + "mlp.token_shift", hp + "mlp.token_shift"),
        ]
        if is_first_dense or mtp_head:
            out += [
                ReplicatedMapping(fp + "mlp.deepemb.weight", hp + "mlp.deepemb.weight"),
                ReplicatedMapping(fp + "mlp.deepemb_a_proj.weight", hp + "mlp.deepemb_a_proj.weight"),
                ColumnParallelMapping(fp + "mlp.deepemb_b_proj.weight", hp + "mlp.deepemb_b_proj.weight"),
                ColumnParallelMapping(fp + "mlp.deepemb_b_proj.bias", hp + "mlp.deepemb_b_proj.bias"),
            ]
    else:
        out += [
            ReplicatedMapping(fp + "mlp.router.weight", hp + "mlp.gate.weight"),
            ReplicatedMapping(fp + "mlp.fc2_norm.weight", hp + "mlp.routed_expert_norm.weight"),
            AutoMapping(fp + "mlp.fc1_latent_proj.weight", hp + "mlp.routed_expert_down_proj.weight"),
            AutoMapping(fp + "mlp.fc2_latent_proj.weight", hp + "mlp.routed_expert_up_proj.weight"),
            FusedGatedExpertMapping(fp + "mlp.experts.linear_fc1.weight*", hp + "mlp.experts.gate_up_proj"),
            FusedExpertMapping(fp + "mlp.experts.linear_fc2.weight*", hp + "mlp.experts.down_proj"),
        ]

    if has_rosa:
        out += [
            ReplicatedMapping(fp + "rosa.embed_proj.weight", hp + "rosa.embed_proj.weight"),
            ReplicatedMapping(fp + "rosa.stream_gate.weight", hp + "rosa.stream_gate.weight"),
            ReplicatedMapping(fp + "rosa.stream_gate.bias", hp + "rosa.stream_gate.bias"),
        ]
        if not rosa_tied and first_rosa >= 0:
            out.append(
                ReplicatedMapping(
                    fp + "rosa.embed_tokens.weight",
                    hp.split("layers.")[0] + f"layers.{first_rosa}.rosa.embed_tokens.weight",
                )
            )
    return out


@MegatronModelBridge.register_bridge(
    source="ShensiForCausalLM",
    target=ShensiModel,
    provider=ShensiModelProvider,
    model_type="shensi",
)
class ShensiBridge(MegatronModelBridge):
    """Maps a Shensi HF checkpoint onto a `ShensiModelProvider` and its parameters."""

    MODEL_CONFIG_CLASS = ShensiModelConfig
    USE_MODEL_CONFIG_FOR_CONVERSION = True

    def hf_config_to_provider_kwargs(self, hf_config: PretrainedConfig) -> dict[str, Any]:
        """Return the same flat kwargs for the legacy provider API."""
        return self.hf_config_to_model_config_kwargs(hf_config)

    def hf_config_to_model_config_kwargs(self, hf_config: PretrainedConfig) -> dict[str, Any]:
        if bool(_get(hf_config, "attention_bias", False)) or bool(_get(hf_config, "mlp_bias", False)):
            # HF has separate `attention_bias` / `mlp_bias` knobs; Megatron-Core has one
            # `add_bias_linear` for every projection, so per-group biases cannot be mirrored.
            raise NotImplementedError(
                "shensi requires attention_bias=False and mlp_bias=False: Megatron-Core's single "
                "`add_bias_linear` cannot express the HuggingFace per-group bias layout."
            )
        if not bool(_get(hf_config, "norm_topk_prob", True)):
            # HF's router normalizes the top-k scores only when `norm_topk_prob` is set;
            # Megatron-Core's sigmoid/sqrtsoftplus top-k path always renormalizes and never
            # reads the field, so the off state cannot be mirrored.
            raise NotImplementedError(
                "shensi requires norm_topk_prob=True: Megatron-Core's top-k router always "
                "renormalizes the selected scores, which mirrors only the normalized "
                "HuggingFace routing."
            )
        kwargs = super().hf_config_to_model_config_kwargs(hf_config)
        mtp = int(_get(hf_config, "num_mtp_layers", 0) or 0)
        kwargs.update(shensi_transformer_field_values(hf_config, mtp_num_layers=mtp))
        return kwargs

    def mapping_registry(self) -> MegatronMappingRegistry:
        hf_config = self.hf_config
        hc_kernels = list(_get(hf_config, "hc_conv_kernels", []) or [])
        mappings = [
            AutoMapping("embedding.word_embeddings.weight", "model.embed_tokens.weight"),
            AutoMapping("output_layer.weight", "lm_head.weight"),
            ReplicatedMapping("norm.weight", "model.norm.weight"),
            ReplicatedMapping("hc_head.hc_fn", "model.hc_head.hc_fn"),
            ReplicatedMapping("hc_head.hc_base", "model.hc_head.hc_base"),
            ReplicatedMapping("hc_head.hc_scale", "model.hc_head.hc_scale"),
            ReplicatedMapping("attn_bank.basis", "model.attn_bank.basis"),
            ReplicatedMapping("attn_bank.stream_basis", "model.attn_bank.stream_basis"),
            ReplicatedMapping("attn_bank.token_down.weight", "model.attn_bank.token_down.weight"),
            ReplicatedMapping("attn_bank.token_up.weight", "model.attn_bank.token_up.weight"),
        ]
        for idx in range(len(hc_kernels)):
            mappings += [
                ReplicatedMapping(
                    f"stream_embeddings.convs.{idx}.weight", f"model.stream_embeddings.convs.{idx}.weight"
                ),
                ReplicatedMapping(f"stream_embeddings.convs.{idx}.bias", f"model.stream_embeddings.convs.{idx}.bias"),
            ]
        mappings.append(
            _StreamTokenEmbeddingMapping(
                "stream_embeddings.token_embeddings.weight",
                "model.stream_embeddings.token_embeddings.weight",
                hf_vocab_size=int(hf_config.vocab_size),
            )
        )
        for name in ("q_proj", "k_proj"):
            mappings.append(
                ReplicatedMapping(f"output_attn_res.{name}.weight", f"model.output_attn_res.{name}.weight")
            )
        mappings += [
            ReplicatedMapping("output_attn_res.g_proj.weight", "model.output_attn_res.g_proj.weight"),
            ReplicatedMapping("output_attn_res.g_proj.bias", "model.output_attn_res.g_proj.bias"),
            ReplicatedMapping("output_attn_res.scale", "model.output_attn_res.scale"),
        ]

        n_layers = int(hf_config.num_hidden_layers)
        mlp_layer_types = resolve_mlp_layer_types(
            _get(hf_config, "mlp_layer_types", None), int(_get(hf_config, "first_k_dense_replace", 3)), n_layers
        )
        attn_res_types = _get(hf_config, "attn_res_block_layer_types", None) or resolve_attn_res_block_layer_types(
            mlp_layer_types, int(_get(hf_config, "attn_res_block_size", 2)), n_layers
        )
        layer_types = _get(hf_config, "layer_types", None) or resolve_layer_types(attn_res_types)
        indexer_types = resolve_indexer_types(layer_types, _get(hf_config, "indexer_types", None))
        rosa_ids = set(resolve_rosa_layer_ids(mlp_layer_types, _get(hf_config, "rosa_layer_ids", None)))
        first_rosa = min(rosa_ids) if rosa_ids else -1
        rosa_tied = int(_get(hf_config, "rosa_embed_dim", None) or hf_config.hidden_size) == int(hf_config.hidden_size)
        first_full = indexer_types.index("full") if "full" in indexer_types else -1
        first_indexed = layer_types.index("indexed_attention")
        first_dense = mlp_layer_types.index("dense") if "dense" in mlp_layer_types else -1
        hc_mult = int(_get(hf_config, "hc_mult", 4))
        hc_fixed = int(_get(hf_config, "hc_fixed_streams", 1))
        hc_active = int(_get(hf_config, "hc_active_streams", 4))
        routed_streams = hc_mult - hc_fixed
        hc_conditional = hc_active < hc_mult and routed_streams > 0

        for i in range(n_layers):
            mappings += _layer_mappings(
                f"decoder.layers.{i}.",
                f"model.layers.{i}.",
                layer_type=layer_types[i],
                mlp_type=mlp_layer_types[i],
                hc_kernels=hc_kernels,
                hc_conditional=hc_conditional,
                routed_streams=routed_streams,
                is_full_indexer=(indexer_types[i] == "full"),
                is_first_full_indexer=(i == first_full),
                is_first_dense=(i == first_dense),
                has_kv_emb=(i == first_indexed),
                has_rosa=(i in rosa_ids),
                rosa_tied=rosa_tied,
                first_rosa=first_rosa,
            )

        for depth in range(int(_get(hf_config, "num_mtp_layers", 0) or 0)):
            ck = f"mtp.{depth}"
            mg = f"mtp.layers.{depth}"
            mappings += [
                AutoMapping(f"{mg}.enorm.weight", f"{ck}.enorm.weight"),
                AutoMapping(f"{mg}.hnorm.weight", f"{ck}.hnorm.weight"),
                AutoMapping(f"{mg}.eh_proj.weight", f"{ck}.eh_proj.weight"),
                AutoMapping(f"{mg}.final_layernorm.weight", f"{ck}.norm.weight"),
            ]
            last = n_layers - 1
            mappings += _layer_mappings(
                f"{mg}.mtp_model_layer.layers.0.",
                f"{ck}.",
                layer_type=layer_types[last],
                mlp_type=mlp_layer_types[last],
                hc_kernels=hc_kernels,
                hc_conditional=hc_conditional,
                routed_streams=routed_streams,
                is_full_indexer=(indexer_types[last] == "full"),
                is_first_full_indexer=(last == first_full),
                is_first_dense=(last == first_dense),
                has_kv_emb=(last == first_indexed),
                has_rosa=False,
                rosa_tied=rosa_tied,
                first_rosa=first_rosa,
                mtp_head=True,
            )
        return MegatronMappingRegistry(*mappings)

    @classmethod
    def megatron_to_hf_config(cls, provider: ShensiModelProvider) -> Dict:
        hf = super().megatron_to_hf_config(provider)
        hf["model_type"] = "shensi"
        hf["architectures"] = ["ShensiForCausalLM"]
        hf["hidden_size"] = provider.hidden_size
        hf["num_hidden_layers"] = provider.num_layers
        hf["num_mtp_layers"] = int(getattr(provider, "mtp_num_layers", 0) or 0)
        hf["attn_res_block_size"] = getattr(provider, "attn_res_block_size", 2)
        hf["latent_size"] = getattr(provider, "latent_size", provider.hidden_size)
        hf["hc_mult"] = getattr(provider, "hc_mult", 4)
        hf["hc_conv_kernels"] = list(getattr(provider, "hc_conv_kernels", []) or [])
        hf["layer_types"] = list(getattr(provider, "layer_types", []) or [])
        hf["mlp_layer_types"] = list(getattr(provider, "mlp_layer_types", []) or [])
        hf["indexer_types"] = list(getattr(provider, "indexer_types", []) or [])
        return hf
