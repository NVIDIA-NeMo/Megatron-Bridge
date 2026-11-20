# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Qwen4-Exp layer specs using Bridge-local model components."""

from copy import deepcopy
from dataclasses import fields
from functools import partial
from typing import TYPE_CHECKING

from megatron.core.models.backends import get_backend_from_config
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_transformer_layer_with_experimental_attention_variant_spec,
)
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from megatron.core.transformer.transformer_layer import TransformerLayerSubmodules

from megatron.bridge.models.qwen.modeling_qwen4_exp.gated_residual import (
    GatedResidualHyperConnection,
    GatedResidualOutputMixer,
)
from megatron.bridge.models.qwen.modeling_qwen4_exp.model import (
    Qwen4ExpGatedDeltaNet,
    Qwen4ExpLayerSubmodules,
    Qwen4ExpTransformerLayer,
)
from megatron.bridge.models.qwen.modeling_qwen4_exp.per_layer_embedding import PerLayerEmbedding
from megatron.bridge.models.qwen.modeling_qwen4_exp.qsa import (
    QSACoreAttention,
    QSAIndexer,
    QSAIndexerSubmodules,
    QwenSparseSelfAttention,
    QwenSparseSelfAttentionSubmodules,
)


if TYPE_CHECKING:
    from megatron.bridge.models.qwen.modeling_qwen4_exp.provider import Qwen4ExpModelProvider


def get_qwen4_exp_block_spec(
    config: "Qwen4ExpModelProvider", vp_stage: int | None = None
) -> TransformerBlockSubmodules:
    """Replace family-specific slots in Core's GDN/MoE decoder spec.

    The provider restricts this first migration to PP=CP=1. Tensor parallelism
    and sequence parallelism retain Core's usual attention/MLP implementations.
    """
    if config.pipeline_model_parallel_size != 1 or config.context_parallel_size != 1 or vp_stage is not None:
        raise NotImplementedError("Qwen4-Exp currently requires PP=CP=1 without virtual pipeline stages.")
    backend = get_backend_from_config(config)
    layer_specs = deepcopy(get_transformer_layer_with_experimental_attention_variant_spec(config, backend=backend))
    ple_layer_ids = set(config.ple_layer_ids or [])
    for layer_number, spec in enumerate(layer_specs, start=1):
        spec.module = Qwen4ExpTransformerLayer
        old = spec.submodules
        spec.submodules = Qwen4ExpLayerSubmodules(
            **{field.name: getattr(old, field.name) for field in fields(TransformerLayerSubmodules)}
        )
        sub = spec.submodules
        sub.input_layernorm = IdentityOp
        sub.pre_mlp_layernorm = IdentityOp
        sub.self_attention_hyper_connection = GatedResidualHyperConnection
        sub.mlp_hyper_connection = GatedResidualHyperConnection
        if layer_number in ple_layer_ids:
            sub.per_layer_embedding = ModuleSpec(module=PerLayerEmbedding)

        attention = sub.self_attention
        if hasattr(attention.submodules, "in_proj"):
            attention.module = Qwen4ExpGatedDeltaNet
            attention.submodules.in_proj = backend.column_parallel_linear()
        elif config.qsa_indexer_n_heads is not None:
            qk_norm = backend.layer_norm(rms_norm=True, for_qk=True) if config.qk_layernorm else IdentityOp
            sub.self_attention = ModuleSpec(
                module=QwenSparseSelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=QwenSparseSelfAttentionSubmodules(
                    linear_qkv=backend.column_parallel_linear(),
                    core_attention=partial(QSACoreAttention, dense_core_attention=backend.core_attention()),
                    linear_proj=backend.row_parallel_linear(),
                    q_layernorm=qk_norm,
                    k_layernorm=qk_norm,
                    indexer=ModuleSpec(
                        module=QSAIndexer,
                        submodules=QSAIndexerSubmodules(
                            linear_qk=backend.linear(),
                            q_layernorm=backend.layer_norm(rms_norm=True, for_qk=True),
                            k_layernorm=backend.layer_norm(rms_norm=True, for_qk=True),
                        ),
                    ),
                ),
                metainfo={"fuse_input_layernorm": False},
            )
        else:
            attention.submodules.linear_qkv = backend.column_parallel_linear()

        # A dense MLP must not apply a second input norm after GR has normalized
        # and mixed the streams. MoE uses an unfused input projection already.
        mlp_submodules = getattr(sub.mlp, "submodules", None)
        if mlp_submodules is None and isinstance(sub.mlp, partial):
            mlp_submodules = sub.mlp.keywords.get("submodules")
        if mlp_submodules is not None and hasattr(mlp_submodules, "linear_fc1"):
            mlp_submodules.linear_fc1 = backend.column_parallel_linear()

    return TransformerBlockSubmodules(layer_specs=layer_specs, layer_norm=GatedResidualOutputMixer)
