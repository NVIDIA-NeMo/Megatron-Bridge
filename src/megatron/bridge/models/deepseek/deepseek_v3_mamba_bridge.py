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

"""DeepSeek-V3 bridge variant that targets `MambaModel` instead of `GPTModel`.

The default `DeepSeekV3Bridge` instantiates a Megatron-Core `GPTModel`.
This module adds an opt-in subclass that swaps the base model
class to `MambaModel` (the hybrid Mamba/attention stack), so the MLA
attention and MoE configuration coming from the HF checkpoint can be combined
with Mamba/hybrid layer layouts.

Note:
    This bridge is *not* registered via `@MegatronModelBridge.register_bridge`
    because the auto-dispatch table in `model_bridge.py` is keyed by the HF
    source class name alone — registering a second bridge for
    `DeepseekV3ForCausalLM` would overwrite the default GPT-based bridge.
    Use this class explicitly, e.g.:

        bridge = DeepSeekV3MambaBridge()
        bridge.hf_config = hf_pretrained.config
        provider = bridge.provider_bridge(hf_pretrained)
        provider.hybrid_layer_pattern = "M*" * hf_config.num_hidden_layers
        provider.finalize()
        model = provider.provide()
"""

from dataclasses import dataclass
from typing import Optional

from megatron.core.models.mamba import MambaModel

from megatron.bridge.models.deepseek.deepseek_v3_bridge import DeepSeekV3Bridge
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.mamba.mamba_provider import MambaModelProvider
from megatron.bridge.models.transformer_config import MLATransformerConfig


@dataclass
class DeepSeekV3MambaProvider(MambaModelProvider, MLATransformerConfig):
    """Provider that combines MLA attention config with the Mamba model backbone.

    Method-resolution order places `MambaModelProvider` first, so
    `provide` returns an `MCoreMambaModel` (the hybrid stack that can
    host Mamba, attention, MLA, and DSA layers). `MLATransformerConfig`
    contributes the MLA dataclass fields (`q_lora_rank`, `kv_lora_rank`,
    `qk_head_dim`, `qk_pos_emb_head_dim`, `v_head_dim`, …) that DSv3
    requires. `finalize()` chains through both parents via C3 linearization:
    Mamba pattern processing runs first, then MLA post-init.

    Users must supply a `hybrid_layer_pattern` (or the deprecated
    `hybrid_override_pattern`) before calling `finalize()` so the hybrid
    stack knows where to place Mamba vs. attention/MLA layers.
    """

    # MLA position embeddings must be "rope" even though MambaModelProvider
    # defaults to "none" (pure-Mamba models have no position embeddings).
    position_embedding_type: str = "rope"


class DeepSeekV3MambaBridge(DeepSeekV3Bridge):
    """DeepSeek-V3 bridge that instantiates `MambaModel` instead of `GPTModel`.

    Inherits the HF→Megatron config translation from `DeepSeekV3Bridge`
    (MLA params, MoE params, vocab sizing, …) but swaps the target model class
    to `MambaModel` via `PROVIDER_CLASS`.

    The default `provider_bridge` implementation carries over the
    DSv3-specific MLA/MoE settings, then strips GPT-only spec wiring
    (`transformer_layer_spec`) that is meaningless for the Mamba stack.
    Callers are expected to assign `hybrid_layer_pattern` (and optionally
    `mamba_stack_spec`) before calling `provider.finalize()`.
    """

    PROVIDER_CLASS = DeepSeekV3MambaProvider

    # Advertised target so downstream tooling that inspects the bridge knows
    # which MCore model class this variant produces.
    TARGET_MODEL = MambaModel

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> DeepSeekV3MambaProvider:
        """Build a `DeepSeekV3MambaProvider` from an HF DSv3 checkpoint.

        Reuses the DSv3 MLA/MoE configuration produced by the parent
        `DeepSeekV3Bridge` and then clears GPT-only fields that do not
        apply to the Mamba backbone.
        """
        provider: DeepSeekV3MambaProvider = super().provider_bridge(hf_pretrained)

        # GPT-specific spec used by GPTModel; MambaModel is built from
        # `mamba_stack_spec` instead, so drop this to avoid confusion.
        if hasattr(provider, "transformer_layer_spec"):
            try:
                provider.transformer_layer_spec = None
            except (AttributeError, TypeError):
                pass

        # MLA attention requires rope positional embeddings; the Mamba provider
        # default of "none" would disable them for the attention sub-layers.
        provider.position_embedding_type = "rope"

        return provider

    def default_hybrid_layer_pattern(self, hf_pretrained: PreTrainedCausalLM) -> Optional[str]:
        """Suggest a hybrid layer pattern equivalent to the HF DSv3 layout.

        Each DSv3 transformer layer is `[MLA attention, MLP-or-MoE]`. In the
        Mamba hybrid pattern this maps to `*-` for dense blocks and `*E` for
        MoE blocks (`*` = attention, `-` = MLP, `E` = MoE). Callers can
        use this helper to populate `provider.hybrid_layer_pattern` before
        calling `finalize()`.

        Returns `None` if `first_k_dense_replace` is missing from the HF
        config (e.g. non-DSv3 variants).
        """
        hf_config = hf_pretrained.config
        num_layers = getattr(hf_config, "num_hidden_layers", None)
        first_k_dense = getattr(hf_config, "first_k_dense_replace", None)
        if num_layers is None or first_k_dense is None:
            return None

        parts = []
        for layer_idx in range(num_layers):
            parts.append("*-" if layer_idx < first_k_dense else "*E")
        return "".join(parts)
