# Megatron-Bridge bridge for Apertus 1.5 (`Apertus1p5ForConditionalGeneration`).
#
# wqwqazwsxedc/Megatron-Bridge (apertus branch) only ever added a bridge for the
# older, text-only `ApertusForCausalLM` (see apertus/apertus_bridge.py) -- there is
# no apertus1p5 bridge anywhere in that fork (confirmed against its git history:
# the entire branch is stock NVIDIA r0.5.0 plus one commit, "add apertus", which
# touches only apertus/{__init__.py,apertus_bridge.py} and a functional test).
# AutoBridge.from_hf_pretrained() therefore raises "Model architecture
# 'Apertus1p5ForConditionalGeneration' is not yet supported" (run 3149776) -- this
# is a real, total gap, not a registration bug fixable with a one-line patch.
#
# Feasibility: swiss-ai/transformers' Apertus1p5TextModel (the actual language
# backbone) is architecturally identical to plain Apertus -- same xielu MLP
# (up_proj/down_proj only, no gate), same q_norm/k_norm RMSNorm on attention,
# same attention_bias=False, same llama3-style rope_parameters dict shape
# (rope_type/rope_theta/factor/original_max_position_embeddings/low_freq_factor/
# high_freq_factor) -- confirmed by reading modeling_apertus1p5.py at the exact
# commit this script pins (SWISS_AI_TRANSFORMERS_SHA). It only differs in two
# structural ways:
#   1. Hyperparameters live under `config.text_config` (an Apertus1p5TextConfig),
#      not the top-level Apertus1p5Config -- the top level also carries
#      vision_config/audio_config for the multimodal tokenizers this benchmark
#      never uses.
#   2. The decoder stack sits one level deeper in the HF module tree:
#      Apertus1p5ForConditionalGeneration.model.language_model.{embed_tokens,
#      layers,norm} instead of plain Apertus's Apertus...ForCausalLM.model.*.
#      `lm_head` itself stays at the top level, tied to
#      model.language_model.embed_tokens.weight (per _tied_weights_keys) --
#      exactly analogous to plain Apertus's own lm_head/output_layer tie.
#
# This bridge is therefore adapted directly from apertus/apertus_bridge.py: same
# MCoreXIELU activation and get_apertus_decoder_block_spec (imported, not
# duplicated, so behavior stays identical to the one upstream commit that
# actually exists), same rope/attention_bias/hidden_act validation, only the HF
# config source (text_config instead of the top-level config) and the HF-side
# weight key prefix (model.language_model.* instead of model.*) change.
#
# Deliberately NOT mapped: vision_tower/audio_tower/vision_tokenizer/
# audio_tokenizer weights. Megatron's target here is plain GPTModel (text-only --
# see "want to take advantage of Megatron parallelism" in the dispatching
# conversation, not multimodal fidelity), which has no parameter slots for them
# anyway, and this benchmark (text-only GSM8K GRPO) never calls
# get_image_feature/get_audio_feature -- same reasoning already established for
# the SGLang side of this script (see apertus-benchmarks/patches/
# sglang-apertus1p5-local-fixes.patch point 6 in CLAUDE.md: those weights never
# need to be numerically correct for this benchmark, only absent-is-fine). No
# strict/leftover-key check was found anywhere in model_bridge.py's weight-load
# path, so simply omitting these mappings is sufficient -- there is nothing to
# suppress.
#
# Pruned LM head: confirmed (run 3171151) that swiss-ai/Apertus-v1.5-70B does
# use a pruned head -- output_vocab_size=131072 vs the extended
# vocab_size=266752 that also covers the visual/audio token ranges. An
# earlier version of this bridge raised rather than risk mismapping shapes;
# now handled properly since this benchmark (text-only GSM8K GRPO) never
# produces or consumes a token id outside [0, output_vocab_size) anyway --
# `output_vocab_size`'s own docstring in configuration_apertus1p5.py confirms
# ids `0..output_vocab_size - 1` are exactly the retained/generatable
# (non-multimodal) ones. Megatron-core's GPTModelProvider only has one
# `vocab_size` sizing both tables, so provider_bridge sets it to
# `output_vocab_size` (not the full extended vocab_size) whenever the two
# differ, and mapping_registry truncates the *input* embedding table
# (`model.language_model.embed_tokens.weight`, shape (266752, hidden) in the
# checkpoint) down to its first `output_vocab_size` rows to match -- the
# checkpoint's `lm_head.weight` is already exactly (131072, hidden) since it
# really is a physically pruned Linear layer (see
# Apertus1p5ForConditionalGeneration.__init__), so the output-side mapping
# needs no such transform. `_TruncatedVocabEmbeddingMapping` below also
# best-effort zero-pads back out to the full vocab_size on the reverse
# (megatron_to_hf) direction, for whatever HF-format-export path might call
# it -- those padded rows carry no real multimodal embedding, same caveat as
# megatron_to_hf_config below.
#
# megatron_to_hf_config below (used for full HF-format checkpoint export, not
# for the live NCCL weight sync this benchmark actually exercises during
# training) is a best-effort adaptation nesting the generic flat CONFIG_MAPPING
# output under "text_config" to match Apertus1p5Config's real shape -- lower
# confidence than the forward (provider_bridge/mapping_registry) path, since it
# is not needed to get past this benchmark's first training step and so was not
# a priority to verify.

from __future__ import annotations

import torch
from megatron.bridge.models.apertus.apertus_bridge import MCoreXIELU, get_apertus_decoder_block_spec
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ColumnParallelMapping,
    QKVMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.conversion.utils import unwrap_model
from megatron.core.models.gpt.gpt_model import GPTModel
from transformers import Apertus1p5ForConditionalGeneration


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


@MegatronModelBridge.register_bridge(source=Apertus1p5ForConditionalGeneration, target=GPTModel, model_type="apertus1p5")
class Apertus1p5Bridge(MegatronModelBridge):
    @classmethod
    def hf_to_megatron_activation(cls, hidden_act: str):
        if hidden_act != "xielu":
            return super().hf_to_megatron_activation(hidden_act)
        return lambda _: (_ for _ in ()).throw(RuntimeError("expected MCoreXIELU"))

    def provider_bridge(self, hf_pretrained):
        text_config = hf_pretrained.config.text_config
        provider = super().provider_bridge(_TextConfigOnlyShim(text_config))

        output_vocab_size = getattr(text_config, "output_vocab_size", None)
        if output_vocab_size is not None and output_vocab_size != text_config.vocab_size:
            # Pruned LM head: size Megatron's single vocab_size to the narrower,
            # physically-real output_vocab_size (matching the checkpoint's actual
            # lm_head.weight shape) rather than the full extended vocab_size --
            # see the module docstring's "Pruned LM head" section. mapping_registry
            # truncates the embedding table to match.
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
        # Pre-attn / pre-MLP RMSNorm weights + the learnable xIELU scalars, mapped
        # replicated (not TP-sharded). This recipe's use_transformer_engine spec
        # produces the fused `linear_qkv.layer_norm_weight` / `linear_fc1.layer_norm_weight`
        # names, so those direct mappings are what load. Validated end-to-end in run
        # 3247540 (46 steps, healthy GRPO curve) -- and confirmed against fork
        # megatron-core 0.18 / megatron-bridge 0.5.0; switching these to AutoMapping
        # on stock megatron-bridge 0.6.1 loaded them as ~0 instead (run 3243271).
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
