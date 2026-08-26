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

"""MLA attention spec helpers for the DeepSeek family."""

from dataclasses import replace
from typing import Optional

from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TELayerNormColumnParallelLinear,
)
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.multi_latent_attention import MLASelfAttention
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig


class MLASelfAttentionWithoutQueryNorm(MLASelfAttention):
    """MLA self-attention that does not add a query norm when there is no query LoRA.

    MCore derives Q and KV normalization from a single ``qk_layernorm`` flag. DeepSeek
    needs it enabled for ``kv_a_layernorm``, which every checkpoint ships. When
    ``q_lora_rank`` is None, that same flag also makes MCore fuse a query normalization
    into ``linear_q_proj`` (``QKNormConfigResolver._resolve_mla_qk_layernorm``), but the
    HF architecture defines no query-side norm in that case: ``DeepseekV3Attention``
    builds a bare ``q_proj``.

    The result is a trainable parameter with no HF counterpart, which cannot be loaded
    and is silently dropped on export. This subclass keeps the KV norm and drops the
    query norm so the converted model matches the source architecture.

    Transformer Engine is required for the no-query-LoRA case. MCore builds
    ``linear_q_proj`` from the backend's fused norm+linear implementation, which only
    Transformer Engine provides, so the local backend is rejected with an explicit
    message rather than an internal one.
    """

    def __init__(self, config, submodules, *args, **kwargs):
        """Build a plain query projection with both supported MCore MLA layouts."""
        if config.q_lora_rank is not None:
            super().__init__(config, submodules, *args, **kwargs)
            return

        if config.transformer_impl != "transformer_engine":
            raise ValueError(
                "DeepSeek without a query LoRA (`q_lora_rank=None`) requires "
                f"`transformer_impl='transformer_engine'`; `{config.transformer_impl}` "
                "provides no fused norm+linear projection. DeepSeek needs `qk_layernorm` "
                "for `kv_a_layernorm`, while the query projection must remain unfused."
            )

        # Newer MCore resolves these entries in ``_resolve_qk_norm_config``. Older MCore
        # consumes the spec directly in ``__init__``. Making the spec valid for both lets
        # the parent choose its native path without importing a version-specific helper.
        submodules = replace(
            submodules,
            q_layernorm=IdentityOp,
            linear_q_proj=TEColumnParallelLinear,
        )
        super().__init__(config, submodules, *args, **kwargs)

    def _resolve_qk_norm_config(self, submodules):
        """Keep newer MCore's KV resolver while replacing only the query projection."""
        if self.config.q_lora_rank is not None:
            return super()._resolve_qk_norm_config(submodules)

        submodules = replace(
            submodules,
            q_layernorm=IdentityOp,
            linear_q_proj=TELayerNormColumnParallelLinear,
        )
        layer_classes = super()._resolve_qk_norm_config(submodules)
        layer_classes["linear_q_proj"] = TEColumnParallelLinear
        return layer_classes


def get_deepseek_decoder_block_spec(
    config: TransformerConfig,
    use_transformer_engine: bool,
    normalization: Optional[str] = None,
    qk_l2_norm: Optional[bool] = False,
    vp_stage: Optional[int] = None,
    pp_rank: Optional[int] = None,
) -> ModuleSpec:
    """Build the decoder block spec, omitting the query norm when ``q_lora_rank`` is None.

    The signature mirrors ``get_gpt_decoder_block_spec`` exactly, including ``vp_stage``
    and ``pp_rank``. ``GPTModelProvider.provide()`` inspects the callable's parameters and
    only forwards ``vp_stage`` when it is declared, so dropping it here would leave
    interleaved pipeline parallelism calling MCore's layer-offset helper without a virtual
    stage, which asserts.

    Args:
        config: The model provider / transformer config.
        use_transformer_engine: Whether to build Transformer Engine submodules.
        normalization: Optional normalization override, forwarded unchanged.
        qk_l2_norm: Optional QK L2 norm flag, forwarded unchanged.
        vp_stage: Virtual pipeline stage, forwarded unchanged.
        pp_rank: Pipeline rank, forwarded unchanged.

    Returns:
        The decoder block spec, with MLA self-attention replaced by
        :class:`MLASelfAttentionWithoutQueryNorm` when there is no query LoRA.
    """
    spec = get_gpt_decoder_block_spec(
        config,
        use_transformer_engine=use_transformer_engine,
        normalization=normalization,
        qk_l2_norm=qk_l2_norm,
        vp_stage=vp_stage,
        pp_rank=pp_rank,
    )
    return replace_mla_self_attention(config, spec)


def replace_mla_self_attention(config: TransformerConfig, spec: ModuleSpec) -> ModuleSpec:
    """Swap MLA self-attention for the query-norm-free variant, in place, on every layer.

    Shared with the MTP path: a standalone MTP pipeline stage owns no decoder layers, so
    the provider re-derives a layer spec straight from MCore and never passes through
    :func:`get_deepseek_decoder_block_spec`. Without this the MTP layer regains the query
    norm that the decoder layers just dropped.

    Accepts either a block spec (``.layer_specs``) or a single layer spec.
    """
    if getattr(config, "q_lora_rank", None) is not None:
        return spec

    layer_specs = getattr(spec, "layer_specs", None)
    for layer_spec in layer_specs if layer_specs is not None else [spec]:
        self_attention = getattr(layer_spec.submodules, "self_attention", None)
        if self_attention is not None and self_attention.module is MLASelfAttention:
            self_attention.module = MLASelfAttentionWithoutQueryNorm
    return spec
