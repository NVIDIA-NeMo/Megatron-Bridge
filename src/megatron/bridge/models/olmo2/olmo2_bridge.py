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
"""Bridge for HuggingFace ``Olmo2ForCausalLM`` ↔ Megatron-Core ``GPTModel``."""

from megatron.core.models.gpt.gpt_model import GPTModel

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
)
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.olmo2.olmo2_provider import olmo2_layer_spec


@MegatronModelBridge.register_bridge(source="Olmo2ForCausalLM", target=GPTModel, model_type="olmo2")
class Olmo2Bridge(MegatronModelBridge):
    """Bridge for AllenAI's OLMo-2 dense causal LM family.

    Architecture summary (vs. the closest existing bridges):

    +-----------------------+-------------+-------------+-------------+-------------+
    | Property              | Llama       | Qwen3       | Gemma2      | OLMo-2      |
    +=======================+=============+=============+=============+=============+
    | Pre-attn norm         | yes         | yes         | yes         | **no**      |
    | Pre-MLP norm          | yes         | yes         | yes         | **no**      |
    | Post-attn norm        | no          | no          | yes         | **yes**     |
    | Post-MLP norm         | no          | no          | yes         | **yes**     |
    | QK-RMSNorm            | no          | yes         | no          | **yes**     |
    | Logit soft-capping    | no          | no          | yes         | no          |
    | Sliding-window attn   | no          | no          | yes (alt.)  | no          |
    +-----------------------+-------------+-------------+-------------+-------------+

    The custom layer spec (see :func:`olmo2_layer_spec`) realizes the post-norm
    placement. ``mapping_registry`` below names every weight in the HF
    state dict and routes it to the corresponding Megatron-Core parameter.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("allenai/OLMo-2-1124-7B")
        >>> provider = bridge.to_megatron_provider()
    """

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> GPTModelProvider:
        """Convert HF OLMo-2 config to a ``GPTModelProvider`` and apply the OLMo-2 layer spec."""
        provider = super().provider_bridge(hf_pretrained)
        hf_config = hf_pretrained.config

        # Pure post-norm: select the OLMo-2 specific layer spec.
        provider.transformer_layer_spec = olmo2_layer_spec

        # `head_dim` is not always present in the HF config; derive it when missing.
        provider.kv_channels = getattr(hf_config, "head_dim", None) or (
            hf_config.hidden_size // hf_config.num_attention_heads
        )

        # OLMo-2 specifics (all values match `Olmo2Config` defaults / 1B + 7B + 13B configs).
        provider.normalization = "RMSNorm"
        provider.gated_linear_unit = True
        provider.position_embedding_type = "rope"
        provider.add_bias_linear = False
        provider.add_qkv_bias = False
        provider.hidden_dropout = 0.0
        provider.attention_dropout = float(getattr(hf_config, "attention_dropout", 0.0))
        provider.qk_layernorm = True
        provider.persist_layer_norm = True
        provider.share_embeddings_and_output_weights = bool(getattr(hf_config, "tie_word_embeddings", False))

        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Weight mappings between HF ``Olmo2ForCausalLM`` and Megatron-Core ``GPTModel``.

        Notable points specific to OLMo-2:

        * ``model.layers.*.post_attention_layernorm.weight`` and
          ``model.layers.*.post_feedforward_layernorm.weight`` are *output*
          norms — they map to ``linear_proj.post_layernorm`` /
          ``linear_fc2.post_layernorm``, not to the standard Llama-style
          slot ``linear_qkv.layer_norm_weight``.
        * Q-/K-RMSNorm weights live on the per-head projections inside the
          attention block — same name pattern as Qwen3 and OLMoE.
        """
        # 1:1 renames (Megatron name → HF name). Wildcards expand per layer.
        param_mappings = {
            # Token embeddings, output projection, final norm
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "output_layer.weight": "lm_head.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
            # Attention output + post-attention norm (the post-norm folded into linear_proj)
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            "decoder.layers.*.self_attention.linear_proj.post_layernorm.weight": (
                "model.layers.*.post_attention_layernorm.weight"
            ),
            # QK-RMSNorm (per-head Q/K normalization inside attention)
            "decoder.layers.*.self_attention.q_layernorm.weight": "model.layers.*.self_attn.q_norm.weight",
            "decoder.layers.*.self_attention.k_layernorm.weight": "model.layers.*.self_attn.k_norm.weight",
            # MLP down projection + post-feedforward norm (the post-norm folded into linear_fc2)
            "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
            "decoder.layers.*.mlp.linear_fc2.post_layernorm.weight": (
                "model.layers.*.post_feedforward_layernorm.weight"
            ),
        }

        mapping_list = [
            AutoMapping(megatron_param=megatron_param, hf_param=hf_param)
            for megatron_param, hf_param in param_mappings.items()
        ]

        # Fused QKV: HF stores Q/K/V separately; Megatron uses a single packed matrix.
        # OLMo-2 has no QKV bias, so only the weight is fused.
        mapping_list.append(
            QKVMapping(
                megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                q="model.layers.*.self_attn.q_proj.weight",
                k="model.layers.*.self_attn.k_proj.weight",
                v="model.layers.*.self_attn.v_proj.weight",
            )
        )

        # Gated SwiGLU MLP: HF stores gate_proj + up_proj separately;
        # Megatron concatenates them into linear_fc1.
        mapping_list.append(
            GatedMLPMapping(
                megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                gate="model.layers.*.mlp.gate_proj.weight",
                up="model.layers.*.mlp.up_proj.weight",
            )
        )

        return MegatronMappingRegistry(*mapping_list)
