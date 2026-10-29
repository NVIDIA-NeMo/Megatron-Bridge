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

"""Megatron Bridge for LLaDA1.5 (GSAI-ML/LLaDA-*) masked-diffusion LLMs.

Converts between the trust_remote_code ``LLaDAModelLM`` HF class and a
Megatron-Core ``GPTModel``. The model is a dense LLaMA-style block with
OLMo-style parameter naming and full RoPE.

Key mapping decisions, anchored to the reference implementation
(``modeling_llada.py``):

* QKV is **separate** (``q_proj``, ``k_proj``, ``v_proj`` — see
  ``LLaDALlamaBlock.__init__``), not fused. Use :class:`QKVMapping`, not
  :class:`ConcatenatedQKVMapping`.
* The SwiGLU MLP computes ``act(ff_proj(x)) * up_proj(x)`` (see
  ``LLaDALlamaBlock.forward``), so ``ff_proj`` is the **gate** and
  ``up_proj`` is the **up** input for Megatron's fused ``linear_fc1``.
* The LM head is ``model.transformer.ff_out`` (LLaDA1.5 has
  ``weight_tying: false``).
* Attention output projection is named ``attn_out``, not ``o_proj``.
* Layer norms are ``attn_norm`` (pre-attention) and ``ff_norm`` (pre-MLP),
  fused into Megatron's TE ``linear_qkv``/``linear_fc1`` ``layer_norm_weight``.
"""

import torch
import torch.nn.functional as F
from megatron.core.models.gpt.gpt_model import GPTModel

from megatron.bridge.diffusion.models.llada15.llada15_provider import LLaDA15ModelProvider
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
)
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


# Registered under the trust_remote_code class name as a string so AutoBridge
# can resolve it without importing the HF class (which lives in a dynamic
# trust_remote_code module).
@MegatronModelBridge.register_bridge(
    source="LLaDAModelLM",
    target=GPTModel,
    provider=LLaDA15ModelProvider,
    model_type="llada",
)
class LLaDA15Bridge(MegatronModelBridge):
    """HF ``LLaDAModelLM`` ↔ Megatron ``GPTModel`` bridge."""

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> LLaDA15ModelProvider:
        hf_config = hf_pretrained.config

        # LLaDAConfig uses OLMo-style names (n_layers, d_model, n_heads, ...).
        # A few HF-standard names are exposed as @property aliases in
        # configuration_llada.py, but several are not — fill them in here.
        num_layers = hf_config.n_layers
        hidden_size = hf_config.d_model
        num_attention_heads = hf_config.n_heads
        num_kv_heads = hf_config.n_kv_heads if hf_config.n_kv_heads is not None else hf_config.n_heads
        head_dim = hidden_size // num_attention_heads
        ffn_hidden_size = (
            hf_config.mlp_hidden_size if hf_config.mlp_hidden_size is not None else hf_config.mlp_ratio * hidden_size
        )
        vocab_size = hf_config.embedding_size or hf_config.vocab_size

        provider = LLaDA15ModelProvider(
            num_layers=num_layers,
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            num_attention_heads=num_attention_heads,
            num_query_groups=num_kv_heads,
            kv_channels=head_dim,
            vocab_size=vocab_size,
            seq_length=hf_config.max_sequence_length,
            layernorm_epsilon=hf_config.rms_norm_eps,
            rotary_base=hf_config.rope_theta,
            rotary_percent=1.0,
            share_embeddings_and_output_weights=hf_config.weight_tying,
            add_bias_linear=hf_config.include_bias,
            add_qkv_bias=bool(hf_config.include_bias or hf_config.include_qkv_bias),
            normalization="RMSNorm",
            gated_linear_unit=True,
            activation_func=F.silu,
            qk_layernorm=hf_config.attention_layer_norm,
            hidden_dropout=hf_config.residual_dropout,
            attention_dropout=hf_config.attention_dropout,
            bf16=True,
            params_dtype=torch.bfloat16,
            autocast_dtype=torch.bfloat16,
            make_vocab_size_divisible_by=self.make_vocab_size_divisible_by(vocab_size),
        )
        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        # Global (non-layer) parameters.
        global_map = {
            "embedding.word_embeddings.weight": "model.transformer.wte.weight",
            "decoder.final_layernorm.weight": "model.transformer.ln_f.weight",
            "output_layer.weight": "model.transformer.ff_out.weight",
        }

        # Per-layer 1:1 mappings (layer norms + attention output + MLP down).
        # The TE-fused norms ride on linear_qkv / linear_fc1 in Megatron.
        per_layer_map = {
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.transformer.blocks.*.attn_norm.weight",
            "decoder.layers.*.self_attention.linear_proj.weight": "model.transformer.blocks.*.attn_out.weight",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.transformer.blocks.*.ff_norm.weight",
            "decoder.layers.*.mlp.linear_fc2.weight": "model.transformer.blocks.*.ff_out.weight",
        }

        mappings = []
        for meg, hf in {**global_map, **per_layer_map}.items():
            mappings.append(AutoMapping(megatron_param=meg, hf_param=hf))

        # Separate Q/K/V projections fused into Megatron's linear_qkv.
        mappings.append(
            QKVMapping(
                megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                q="model.transformer.blocks.*.q_proj.weight",
                k="model.transformer.blocks.*.k_proj.weight",
                v="model.transformer.blocks.*.v_proj.weight",
            )
        )

        # SwiGLU MLP: ff_proj is the gate (gets SiLU), up_proj is the linear up.
        # See LLaDALlamaBlock.forward: `act(ff_proj(x)) * up_proj(x)`.
        mappings.append(
            GatedMLPMapping(
                megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                gate="model.transformer.blocks.*.ff_proj.weight",
                up="model.transformer.blocks.*.up_proj.weight",
            )
        )

        return MegatronMappingRegistry(*mappings)
