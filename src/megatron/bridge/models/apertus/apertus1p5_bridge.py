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

"""Megatron Bridge for Apertus 1.5 (``Apertus1p5ForConditionalGeneration``).

The Apertus 1.5 language backbone is the Apertus 1.0 decoder (xIELU MLP with up/down
projections only, RMSNorm q/k-norm, no attention bias, llama3 RoPE), so this bridge reuses
``MCoreXIELU`` and ``get_apertus_decoder_block_spec`` from the Apertus bridge. It differs in:

* Hyperparameters are read from ``config.text_config``.
* HF decoder weights live under ``model.language_model.*``; ``lm_head`` stays top level.
* The target is a text-only ``GPTModel``: the vision and audio towers and tokenizers are not
  mapped.
* Pruned LM head: when ``output_vocab_size < vocab_size`` (the extra ids are multimodal
  tokens), Megatron's single ``vocab_size`` is set to ``output_vocab_size``. The input
  embedding is truncated to its first ``output_vocab_size`` rows on import and zero-padded
  back to ``vocab_size`` on export.
* ``megatron_to_hf_config`` nests the text config under ``text_config``; the multimodal
  sub-configs are not reconstructed.
"""

from __future__ import annotations

import torch
from megatron.core.models.gpt.gpt_model import GPTModel
from transformers import Apertus1p5ForConditionalGeneration

from megatron.bridge.models.apertus.apertus_bridge import (
    MCoreXIELU,
    get_apertus_decoder_block_spec,
    xielu_placeholder,
)
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ColumnParallelMapping,
    QKVMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.conversion.utils import unwrap_model


class _TruncatedVocabEmbeddingMapping(AutoMapping):
    """AutoMapping variant for a pruned-LM-head checkpoint's input embedding table.

    HF's `embed_tokens.weight` covers the full extended vocab_size (text plus
    multimodal token ids); Megatron-core's single `vocab_size` here is set to
    the narrower `output_vocab_size` (see provider_bridge and the module
    docstring's "Pruned LM head" section) so it matches the checkpoint's
    already-pruned `lm_head.weight`. hf_to_megatron truncates the source
    tensor to the first output_vocab_size rows before handing off to the
    normal VocabParallelEmbedding sharding logic; megatron_to_hf best-effort
    zero-pads back out to the full width (those rows carry no real multimodal
    embedding -- acceptable since this bridge's HF-export path is already
    lower-priority, see megatron_to_hf_config below).
    """

    def __init__(self, megatron_param, hf_param, output_vocab_size, full_vocab_size, permute_dims=None):
        super().__init__(megatron_param, hf_param, permute_dims)
        self._output_vocab_size = output_vocab_size
        self._full_vocab_size = full_vocab_size

    def hf_to_megatron(self, hf_weights, megatron_module):
        if hf_weights.shape[0] > self._output_vocab_size:
            hf_weights = hf_weights[: self._output_vocab_size].contiguous()
        return super().hf_to_megatron(hf_weights, megatron_module)

    def megatron_to_hf(self, megatron_weights, megatron_module):
        result = super().megatron_to_hf(megatron_weights, megatron_module)
        if not result:
            return result
        key = next(iter(result))
        value = result[key]
        if value.shape[0] < self._full_vocab_size:
            pad = value.new_zeros((self._full_vocab_size - value.shape[0], *value.shape[1:]))
            value = torch.cat([value, pad], dim=0)
        return {key: value}

    def resolve(self, captures):
        resolved_megatron_param, resolved_hf_param = self._resolve_names(captures)
        return type(self)(
            resolved_megatron_param,
            resolved_hf_param,
            self._output_vocab_size,
            self._full_vocab_size,
            self.permute_dims,
        )


_ROPE_DEFAULTS = {
    "rope_type": "llama3",
    "original_max_position_embeddings": 8192,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
}


class _TextConfigOnlyShim:
    """Minimal stand-in for an hf_pretrained wrapper, exposing only `.config`.

    MegatronModelBridge.provider_bridge() (the base implementation we delegate
    to below) only ever reads `hf_pretrained.config` -- this lets us hand it
    `Apertus1p5Config.text_config` directly instead of the top-level config
    that also carries the unrelated vision_config/audio_config blocks.
    """

    def __init__(self, config):
        self.config = config


@MegatronModelBridge.register_bridge(
    source=Apertus1p5ForConditionalGeneration, target=GPTModel, model_type="apertus1p5"
)
class Apertus1p5Bridge(MegatronModelBridge):
    """Megatron Bridge for the Apertus 1.5 language model; see the module docstring."""

    @classmethod
    def hf_to_megatron_activation(cls, hidden_act: str):
        if hidden_act != "xielu":
            return super().hf_to_megatron_activation(hidden_act)
        return xielu_placeholder

    @classmethod
    def megatron_to_hf_activation(cls, activation_func) -> str:
        if activation_func is xielu_placeholder:
            return "xielu"
        return super().megatron_to_hf_activation(activation_func)

    def provider_bridge(self, hf_pretrained):
        text_config = hf_pretrained.config.text_config
        provider = super().provider_bridge(_TextConfigOnlyShim(text_config))

        output_vocab_size = getattr(text_config, "output_vocab_size", None)
        if output_vocab_size is not None and output_vocab_size != text_config.vocab_size:
            # Pruned LM head: size vocab_size to the generatable ids; see the module docstring.
            provider.vocab_size = output_vocab_size
            provider.make_vocab_size_divisible_by = self.make_vocab_size_divisible_by(output_vocab_size)

        rope = {
            **(getattr(text_config, "rope_scaling", None) or {}),
            **(getattr(text_config, "rope_parameters", None) or {}),
        }
        rope_type = rope.get("rope_type", rope.get("type", "llama3"))
        factor = float(rope.get("factor", 1.0))
        theta = float(rope.get("rope_theta", getattr(text_config, "rope_theta", 10000.0)))
        if text_config.hidden_act != "xielu":
            raise ValueError(f"Expected hidden_act='xielu', got {text_config.hidden_act!r}")
        if text_config.attention_bias:
            raise ValueError("Apertus1p5 attention_bias=True is unsupported")
        if rope_type != "llama3":
            raise ValueError(f"Unsupported Apertus1p5 RoPE type: {rope_type!r}")

        provider.apertus_rope_scaling = {
            "rope_type": rope_type,
            "type": rope_type,
            "factor": factor,
            "original_max_position_embeddings": int(rope.get("original_max_position_embeddings", 8192)),
            "low_freq_factor": float(rope.get("low_freq_factor", 1.0)),
            "high_freq_factor": float(rope.get("high_freq_factor", 4.0)),
        }
        provider.normalization = "RMSNorm"
        provider.qk_layernorm = True
        provider.gated_linear_unit = False
        provider.use_te_activation_func = False
        provider.bias_activation_fusion = False
        provider.add_bias_linear = False
        provider.add_qkv_bias = False
        provider.hidden_dropout = 0.0
        provider.rotary_interleaved = False
        provider.position_embedding_type = "rope"
        provider.rotary_base = theta
        provider.rope_scaling = True
        provider.rope_scaling_factor = factor
        provider.transformer_layer_spec = get_apertus_decoder_block_spec
        return provider

    def load_weights_hf_to_megatron(self, hf_pretrained, megatron_model, allowed_mismatched_params=None):
        models = super().load_weights_hf_to_megatron(
            hf_pretrained, megatron_model, allowed_mismatched_params=allowed_mismatched_params
        )
        [
            m._sync_runtime_scalars()
            for model in unwrap_model(models)
            for m in model.modules()
            if isinstance(m, MCoreXIELU)
        ]
        return models

    @classmethod
    def megatron_to_hf_config(cls, provider) -> dict:
        text_config = super().megatron_to_hf_config(provider)
        theta = float(provider.rotary_base)
        rope = {
            **(getattr(provider, "apertus_rope_scaling", None) or _ROPE_DEFAULTS),
            "factor": float(provider.rope_scaling_factor),
        }
        rope["rope_type"] = rope["type"] = rope.get("rope_type", rope.get("type", "llama3"))
        text_config.update(
            hidden_act="xielu",
            attention_bias=False,
            rope_theta=theta,
            rope_scaling=rope,
            rope_parameters={**rope, "rope_theta": theta},
        )
        return {
            "model_type": "apertus1p5",
            "architectures": ["Apertus1p5ForConditionalGeneration"],
            "text_config": text_config,
        }

    def mapping_registry(self) -> MegatronMappingRegistry:
        text_config = self.hf_config.text_config
        output_vocab_size = getattr(text_config, "output_vocab_size", None)
        full_vocab_size = text_config.vocab_size

        L, H = "decoder.layers.*", "model.language_model.layers.*"
        if output_vocab_size is not None and output_vocab_size != full_vocab_size:
            embedding_mapping = _TruncatedVocabEmbeddingMapping(
                "embedding.word_embeddings.weight",
                "model.language_model.embed_tokens.weight",
                output_vocab_size,
                full_vocab_size,
            )
        else:
            embedding_mapping = AutoMapping(
                "embedding.word_embeddings.weight", "model.language_model.embed_tokens.weight"
            )

        auto = {
            "output_layer.weight": "lm_head.weight",
            "decoder.final_layernorm.weight": "model.language_model.norm.weight",
            f"{L}.self_attention.linear_proj.weight": f"{H}.self_attn.o_proj.weight",
            f"{L}.self_attention.q_layernorm.weight": f"{H}.self_attn.q_norm.weight",
            f"{L}.self_attention.k_layernorm.weight": f"{H}.self_attn.k_norm.weight",
            f"{L}.mlp.linear_fc2.weight": f"{H}.mlp.down_proj.weight",
        }
        # Pre-attention / pre-MLP RMSNorm weights (fused into linear_qkv / linear_fc1 by the
        # TE spec) and the learnable xIELU scalars are replicated, not TP-sharded.
        repl = {
            f"{L}.self_attention.linear_qkv.layer_norm_weight": f"{H}.attention_layernorm.weight",
            f"{L}.mlp.linear_fc1.layer_norm_weight": f"{H}.feedforward_layernorm.weight",
            **{f"{L}.mlp.activation_func.{n}": f"{H}.mlp.act_fn.{n}" for n in ("alpha_p", "alpha_n", "beta", "eps")},
        }
        qkv = QKVMapping(
            f"{L}.self_attention.linear_qkv.weight",
            q=f"{H}.self_attn.q_proj.weight",
            k=f"{H}.self_attn.k_proj.weight",
            v=f"{H}.self_attn.v_proj.weight",
        )
        qkv._tp_mapping = ColumnParallelMapping(qkv.megatron_param, qkv.megatron_param)
        return MegatronMappingRegistry(
            embedding_mapping,
            *(AutoMapping(m, h) for m, h in auto.items()),
            *(ReplicatedMapping(m, h) for m, h in repl.items()),
            qkv,
            ColumnParallelMapping(f"{L}.mlp.linear_fc1.weight", f"{H}.mlp.up_proj.weight"),
        )
