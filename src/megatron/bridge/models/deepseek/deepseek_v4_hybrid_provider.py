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

"""Provider for DeepSeek-V4 expressed as a Megatron-Core ``HybridModel``.

DeepSeek-V4 is a Multi-Latent-Attention (MLA) model, so it needs every MLA
configuration field (``q_lora_rank``, ``output_projection_groups``,
``v_head_dim``, ``rope_type`` / YaRN parameters, …). Those live on
:class:`MLAModelProvider`.
It is also a *hybrid* model: each logical DeepSeek-V4 block is expressed as two
Megatron hybrid layers — an attention-only layer (``W``/``C``/``H`` symbol) and
a MoE-only layer (``E`` symbol) — driven by ``hybrid_layer_pattern`` and built
by MCore's standard hybrid stack. The hybrid ``provide()``/``finalize()`` logic
lives on :class:`HybridModelProvider`.

This provider combines both. :class:`HybridModelProvider` is listed first so its
``provide()`` (which instantiates :class:`~megatron.core.models.hybrid.hybrid_model.HybridModel`)
and ``finalize()`` (which derives ``num_layers`` from ``hybrid_layer_pattern``)
win over the GPT-model versions inherited via :class:`MLAModelProvider`, while
``issubclass(DeepSeekV4HybridModelProvider, MLAModelProvider)`` stays ``True`` so
:meth:`MegatronModelBridge.provider_bridge` keeps mapping the HF config through
the direct MLA field names.
"""

from dataclasses import dataclass

from megatron.bridge.models.hybrid.hybrid_provider import (
    HybridModelProvider,
    transformer_engine_hybrid_stack_spec,
)
from megatron.bridge.models.mla_provider import MLAModelProvider


def deepseek_v4_hybrid_stack_spec(config: HybridModelProvider):
    """Resolve the richest DSv4 hybrid stack supported by the active MCore pin."""
    try:
        from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_dsv4_stack_spec
    except ImportError:
        return transformer_engine_hybrid_stack_spec()

    from megatron.core.transformer import ModuleSpec

    if isinstance(hybrid_dsv4_stack_spec, ModuleSpec):
        return hybrid_dsv4_stack_spec
    return hybrid_dsv4_stack_spec(config)


@dataclass
class DeepSeekV4HybridModelProvider(HybridModelProvider, MLAModelProvider):
    """MLA-capable :class:`HybridModelProvider` for DeepSeek-V4.

    All configuration is supplied by :class:`DeepSeekV4Bridge.provider_bridge`;
    this class fixes the method-resolution order so a single provider is both
    an MLA config carrier and a hybrid-model builder. It also retains both names
    for the grouped-output geometry: MCore main uses the descriptive
    ``output_projection_*`` fields while the frozen dev pin still reads the
    original ``o_*`` fields directly from the config object.
    """

    output_projection_groups: int = 8
    output_projection_lora_rank: int = 1024
    o_groups: int = 8
    o_lora_rank: int = 1024
