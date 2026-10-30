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

"""D1 family mappings composed from existing Bridge conversion primitives."""

from typing import TypeAlias

import torch

from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ConcatenatedQKVMapping,
    GatedMLPMapping,
    MegatronParamMapping,
    QKVMapping,
    ReplicatedMapping,
)


D1ParamMapping: TypeAlias = MegatronParamMapping[torch.Tensor] | MegatronParamMapping[dict[str, torch.Tensor]]


def backbone_mappings(*, megatron_prefix: str, hf_prefix: str) -> list[D1ParamMapping]:
    """Map native D1 language parameters to the published LFM2 names.

    Args:
        megatron_prefix: Native backbone's nonempty module path.
        hf_prefix: Published checkpoint backbone's nonempty module path.

    Returns:
        Bidirectional mappings, including GQA QKV and gated-MLP layout transforms.
    """
    if not megatron_prefix or not hf_prefix:
        raise ValueError("backbone mapping prefixes must be nonempty")
    m = f"{megatron_prefix}.layers.*"
    h = f"{hf_prefix}.layers.*"
    replicated = {
        f"{megatron_prefix}.embed_tokens.weight": f"{hf_prefix}.embed_tokens.weight",
        f"{megatron_prefix}.embedding_norm.weight": f"{hf_prefix}.embedding_norm.weight",
        f"{m}.operator_norm.weight": f"{h}.operator_norm.weight",
        f"{m}.ffn_norm.weight": f"{h}.ffn_norm.weight",
        f"{m}.conv.in_proj.weight": f"{h}.conv.in_proj.weight",
        f"{m}.conv.out_proj.weight": f"{h}.conv.out_proj.weight",
        f"{m}.conv.conv.weight": f"{h}.conv.conv.weight",
        f"{m}.self_attn.q_layernorm.weight": f"{h}.self_attn.q_layernorm.weight",
        f"{m}.self_attn.k_layernorm.weight": f"{h}.self_attn.k_layernorm.weight",
    }
    mappings: list[D1ParamMapping] = [ReplicatedMapping(native, published) for native, published in replicated.items()]
    mappings.extend(
        [
            AutoMapping(f"{m}.self_attn.linear_proj.weight", f"{h}.self_attn.out_proj.weight"),
            QKVMapping(
                f"{m}.self_attn.linear_qkv.weight",
                q=f"{h}.self_attn.q_proj.weight",
                k=f"{h}.self_attn.k_proj.weight",
                v=f"{h}.self_attn.v_proj.weight",
            ),
            GatedMLPMapping(
                f"{m}.feed_forward.linear_fc1.weight",
                gate=f"{h}.feed_forward.w1.weight",
                up=f"{h}.feed_forward.w3.weight",
            ),
            AutoMapping(f"{m}.feed_forward.linear_fc2.weight", f"{h}.feed_forward.w2.weight"),
        ]
    )
    return mappings


def decision_head_mappings(*, megatron_prefix: str, hf_prefix: str) -> list[D1ParamMapping]:
    """Map the native Omni head to the released PyTorch encoder head.

    Args:
        megatron_prefix: Native decision head's nonempty module path.
        hf_prefix: Published decision head's nonempty module path.

    Returns:
        Bidirectional mappings preserving all weights and biases, including
        PyTorch's concatenated QKV weight and bias layouts.
    """
    if not megatron_prefix or not hf_prefix:
        raise ValueError("head mapping prefixes must be nonempty")
    m = f"{megatron_prefix}.layers.*"
    h = f"{hf_prefix}.head.layers.*"
    mappings: list[D1ParamMapping] = [
        ReplicatedMapping(f"{megatron_prefix}.type_emb.weight", f"{hf_prefix}.type_emb.weight")
    ]
    replicated = {
        f"{m}.norm1": f"{h}.norm1",
        f"{m}.norm2": f"{h}.norm2",
        f"{megatron_prefix}.scorer_norm": f"{hf_prefix}.scorer.0",
        f"{megatron_prefix}.scorer_hidden": f"{hf_prefix}.scorer.1",
        f"{megatron_prefix}.scorer_output": f"{hf_prefix}.scorer.3",
    }
    projections = {
        f"{m}.self_attn.linear_proj": f"{h}.self_attn.out_proj",
        f"{m}.mlp.linear_fc1": f"{h}.linear1",
        f"{m}.mlp.linear_fc2": f"{h}.linear2",
    }
    for suffix in ("weight", "bias"):
        mappings.extend(
            ReplicatedMapping(f"{native}.{suffix}", f"{published}.{suffix}")
            for native, published in replicated.items()
        )
        mappings.extend(
            AutoMapping(f"{native}.{suffix}", f"{published}.{suffix}") for native, published in projections.items()
        )
        mappings.append(
            ConcatenatedQKVMapping(f"{m}.self_attn.linear_qkv.{suffix}", f"{h}.self_attn.in_proj_{suffix}")
        )
    return mappings
