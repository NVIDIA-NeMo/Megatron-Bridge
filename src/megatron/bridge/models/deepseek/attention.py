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

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.transformer.mla_qk_norm_config import get_backend
from megatron.core.transformer.multi_latent_attention import MLASelfAttention
from megatron.core.transformer.spec_utils import ModuleSpec


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
    """

    def _resolve_qk_norm_config(self, submodules):
        """Replace the fused query projection with a plain one when there is no query LoRA."""
        layer_classes = super()._resolve_qk_norm_config(submodules)
        if self.config.q_lora_rank is None:
            backend = get_backend(self.config.transformer_impl)
            layer_classes["linear_q_proj"] = backend.column_parallel_linear()
        return layer_classes


def get_deepseek_decoder_block_spec(config, *, use_transformer_engine: bool) -> ModuleSpec:
    """Build the decoder block spec, omitting the query norm when ``q_lora_rank`` is None.

    Args:
        config: The model provider / transformer config.
        use_transformer_engine: Whether to build Transformer Engine submodules.

    Returns:
        The decoder block spec, with MLA self-attention replaced by
        :class:`MLASelfAttentionWithoutQueryNorm` when there is no query LoRA.
    """
    spec = get_gpt_decoder_block_spec(config, use_transformer_engine=use_transformer_engine)
    if getattr(config, "q_lora_rank", None) is not None:
        return spec

    for layer_spec in spec.layer_specs:
        self_attention = layer_spec.submodules.self_attention
        if self_attention.module is MLASelfAttention:
            self_attention.module = MLASelfAttentionWithoutQueryNorm
    return spec
