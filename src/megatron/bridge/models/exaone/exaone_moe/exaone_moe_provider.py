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

from __future__ import annotations

from collections.abc import Callable
from copy import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec, get_gpt_decoder_layer_specs
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
    get_transformer_layer_offset,
)
from megatron.core.utils import get_pg_rank

from megatron.bridge.models.gpt_provider import GPTModelProvider


try:
    import transformer_engine  # type: ignore  # noqa: F401

    HAVE_TE = True
except (ImportError, ModuleNotFoundError):
    HAVE_TE = False

if TYPE_CHECKING:
    from megatron.core.transformer import ModuleSpec


class _MTPDenseLayerSpecsList(list):
    """Return a dense layer spec when MCore asks which spec to use for MTP."""

    def __init__(self, data: list["ModuleSpec"], dense_mtp_spec: "ModuleSpec") -> None:
        super().__init__(data)
        self._dense_mtp_spec = dense_mtp_spec

    def __getitem__(self, idx):
        if isinstance(idx, int) and idx < 0:
            return self._dense_mtp_spec
        return super().__getitem__(idx)


def _layer_specific_config(
    config: "GPTModelProvider",
    *,
    layer_idx: int,
    is_mtp_layer: bool,
) -> "GPTModelProvider":
    """Return a shallow config copy with EXAONE's per-layer settings applied."""
    layer_config = copy(config)

    if is_mtp_layer:
        sliding_windows = getattr(config, "mtp_sliding_windows", None) or []
        layer_types = getattr(config, "mtp_layer_types", None) or []
        swiglu_limits = []
    else:
        sliding_windows = getattr(config, "sliding_windows", None) or []
        layer_types = getattr(config, "layer_types", None) or []
        swiglu_limits = getattr(config, "swiglu_limits", None) or []

    if 0 <= layer_idx < len(sliding_windows):
        sliding_window = int(sliding_windows[layer_idx])
        layer_config.window_size = (sliding_window - 1, 0) if sliding_window > 0 else None
        # ``window_size`` already describes this exact layer, so no additional
        # global skip pattern should be applied by MCore.
        layer_config.window_attn_skip_freq = None
    elif 0 <= layer_idx < len(layer_types) and layer_types[layer_idx] == "full_attention":
        layer_config.window_size = None
        layer_config.window_attn_skip_freq = None

    if 0 <= layer_idx < len(swiglu_limits):
        swiglu_limit = float(swiglu_limits[layer_idx])
        layer_config.activation_func_clamp_value = swiglu_limit if swiglu_limit > 0.0 else None

    if is_mtp_layer and 0 <= layer_idx < len(layer_types):
        # MCore numbers each MTP decoder layer from one, independently of the
        # main decoder. Give its attention module an MTP-specific RoPE pattern
        # with the length required by TransformerConfig validation.
        no_rope = int(layer_types[layer_idx] != "sliding_attention")
        layer_config.no_rope_freq = [no_rope] * config.num_layers

    return layer_config


class ExaoneMoeDecoderLayer(TransformerLayer):
    """Transformer layer that applies EXAONE's per-layer window and SwiGLU settings."""

    def __init__(
        self,
        config: "GPTModelProvider",
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: float | None = None,
        pg_collection: ProcessGroupCollection | None = None,
        vp_stage: int | None = None,
        is_mtp_layer: bool = False,
        add_layer_offset: bool = True,
        pp_layer_offset: int | None = None,
        name: str | None = None,
    ) -> None:
        if is_mtp_layer:
            layer_idx = layer_number - 1
        elif add_layer_offset:
            if pg_collection is None:
                pg_collection = ProcessGroupCollection.use_mpu_process_groups()
            layer_idx = (
                layer_number + get_transformer_layer_offset(config, vp_stage, get_pg_rank(pg_collection.pp)) - 1
            )
        else:
            layer_idx = layer_number - 1

        layer_config = _layer_specific_config(
            config,
            layer_idx=layer_idx,
            is_mtp_layer=is_mtp_layer,
        )
        super().__init__(
            config=layer_config,
            submodules=submodules,
            layer_number=layer_number,
            hidden_dropout=hidden_dropout,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
            is_mtp_layer=is_mtp_layer,
            add_layer_offset=add_layer_offset,
            pp_layer_offset=pp_layer_offset,
            name=name,
        )


def build_exaone_moe_layer_spec(cfg: "GPTModelProvider", **kwargs) -> "ModuleSpec":
    """Build EXAONE MoE decoder specs while keeping MTP sub-layers dense."""
    block_submodules = get_gpt_decoder_block_spec(cfg, use_transformer_engine=HAVE_TE, **kwargs)

    for layer_spec in block_submodules.layer_specs:
        layer_spec.module = ExaoneMoeDecoderLayer

    if getattr(cfg, "mtp_num_layers", None):
        dense_cfg = copy(cfg)
        dense_cfg.moe_layer_freq = [0] * cfg.num_layers
        dense_cfg.num_moe_experts = None
        dense_cfg.moe_grouped_gemm = False
        dense_mtp_spec = get_gpt_decoder_layer_specs(dense_cfg, use_transformer_engine=HAVE_TE)[-1]
        dense_mtp_spec.module = ExaoneMoeDecoderLayer
        block_submodules.layer_specs = _MTPDenseLayerSpecsList(block_submodules.layer_specs, dense_mtp_spec)

    return block_submodules


@dataclass
class ExaoneMoeModelProvider(GPTModelProvider):
    """Model provider for EXAONE MoE models."""

    transformer_layer_spec: "ModuleSpec" | Callable[["GPTModelProvider"], "ModuleSpec"] = build_exaone_moe_layer_spec

    # Model
    normalization: str = "RMSNorm"
    activation_func: Callable = F.silu
    gated_linear_unit: bool = True  # swiglu
    position_embedding_type: str = "rope"
    add_bias_linear: bool = False
    seq_length: int = 4096
    rotary_base: float = 1000000.0
    rope_scaling: bool = False
    rope_scaling_factor: float = 8.0
    make_vocab_size_divisible_by: int = 128
    mtp_num_layers: int | None = None
    mtp_loss_scaling_factor: float | None = None
    kv_channels: int | None = 128
    layer_types: list[str] | None = None
    sliding_windows: list[int] | None = None
    mtp_layer_types: list[str] | None = None
    mtp_sliding_windows: list[int] | None = None
    swiglu_limits: list[float] | None = None

    # Regularization
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    qk_layernorm: bool = True

    # MoE
    moe_grouped_gemm: bool = True
    moe_router_pre_softmax: bool = True
    moe_enable_deepep: bool = False
    moe_token_dispatcher_type: str = "alltoall"
    moe_router_load_balancing_type: str = "global_aux_loss"
    moe_shared_expert_overlap: bool = True
    moe_expert_capacity_factor: float | None = None
    moe_router_dtype: str = "fp32"
    moe_aux_loss_coeff: float = 1e-2
    moe_z_loss_coeff: float = 1e-3
    moe_permute_fusion: bool = True

    # FP8
    fp8: str | None = None
    fp8_recipe: str | None = "tensorwise"
    first_last_layers_bf16: bool = False
    num_layers_at_start_in_bf16: int = 1
    num_layers_at_end_in_bf16: int = 1
    fp8_param: bool = False
    fp8_param_gather: bool = False

    # Miscellaneous
    init_method_std: float = 0.006
    layernorm_epsilon: float = 1e-5
    params_dtype: torch.dtype = torch.bfloat16
    async_tensor_model_parallel_allreduce: bool = True
    attention_softmax_in_fp32: bool = True
    persist_layer_norm: bool = True
    num_layers_in_first_pipeline_stage: int | None = None
    num_layers_in_last_pipeline_stage: int | None = None
    account_for_embedding_in_pipeline_split: bool = False
    account_for_loss_in_pipeline_split: bool = False

    # fusions
    apply_rope_fusion: bool = False
    bias_activation_fusion: bool = False
    bias_dropout_fusion: bool = False
    masked_softmax_fusion: bool = False
    gradient_accumulation_fusion: bool = False

    # Router Bias
    moe_router_topk_scaling_factor: float = 2.5
    moe_router_score_function: str = "sigmoid"
    moe_router_enable_expert_bias: bool = True
    moe_router_bias_update_rate: float = 1e-3
