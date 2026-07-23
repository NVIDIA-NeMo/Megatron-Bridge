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

"""Step3.7 multimodal bridge.

Registers a :class:`MegatronModelBridge` for the upstream
``Step3p7ForConditionalGeneration`` HF architecture (model_type ``step37``).
The bridge:

* Re-uses :class:`Step35Bridge`'s text-decoder logic for ``provider_bridge`` by
  delegating to a synthetic Step-3.5 HF wrapper (the Step3.7 HF config
  exposes its Step-3.5 text fields under ``hf_config.text_config``).
* Adds vision-tower configuration (vision_config, image_token_id,
  projector_bias, understand_projector_stride) on top of the resulting
  provider.
* Defines an HF↔Megatron parameter mapping registry that prefixes every
  Step-3.5 text mapping with ``language_model.`` (since
  :class:`Step37Model` wraps the text GPTModel) and adds direct
  ``vision_model.*`` AutoMappings for the PE-G/14 trunk + downsamplers, plus
  a top-level ``vit_large_projector.weight`` mapping.
"""

from __future__ import annotations

import logging
from typing import List

import torch
from transformers import AutoConfig

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVGMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.stepfun.configuration_step37 import Step37Config
from megatron.bridge.models.stepfun.modelling_step37.model import Step37Model
from megatron.bridge.models.stepfun.step35_bridge import (
    StackedExpertAutoMapping,
    StackedExpertGatedMLPMapping,
    Step35Bridge,
    build_step35_layer_spec,
)
from megatron.bridge.models.stepfun.step37_provider import Step37ModelProvider


logger = logging.getLogger(__name__)


# Register the Step3.7 multimodal config with transformers AutoConfig so that
# ``AutoConfig.from_pretrained("stepfun-ai/step3p7_flash_bf16")`` resolves the
# top-level ``model_type=step37`` without requiring ``trust_remote_code=True``
# (mirrors the same pattern used by :mod:`step35_bridge`).
#
# Why this is sufficient for the convert flow: ``AutoBridge.import_ckpt``'s
# ``PreTrainedCausalLM.from_pretrained`` is *lazy* — it only stores the path
# and reads tensors via :class:`SafeTensorsStateSource` directly from the
# safetensors files on disk. The HF model class is never instantiated, so
# ``AutoModelForCausalLM.register`` is *not* needed for the convert flow.
#
# The literal string ``"step3p7"`` is the public HF ``model_type`` field shipped
# in ``stepfun-ai/step3p7_flash_bf16/config.json``; do not rename it.
AutoConfig.register("step3p7", Step37Config, exist_ok=True)


# Megatron-side parameter prefix for the text decoder. Step3.7's
# :class:`Step37Model` wraps the Step-3.5 ``GPTModel`` under
# ``self.language_model``, so every Step-3.5 megatron_param gets this prefix
# when re-used inside :meth:`Step37Bridge.mapping_registry`.
_LM_PREFIX = "language_model."


def _lm(megatron_param: str) -> str:
    """Prefix a Step-3.5 megatron_param with the ``language_model.`` namespace."""
    return f"{_LM_PREFIX}{megatron_param}"


@MegatronModelBridge.register_bridge(
    source="Step3p7ForConditionalGeneration",
    target=Step37Model,
    provider=Step37ModelProvider,
    model_type="step3p7",
)
class Step37Bridge(MegatronModelBridge):
    """Megatron Bridge for Step3.7 (Step-3.5 text + Perception-Encoder G/14 vision).

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained(
        ...     "/path/to/step3p7_flash_bf16", trust_remote_code=True
        ... )
        >>> provider = bridge.to_megatron_provider()
    """

    # The same translation table Step35Bridge uses (so the parent
    # provider_bridge picks up the text-side renames). Step3.7's text config
    # uses the identical HF schema.
    CONFIG_MAPPING = Step35Bridge.CONFIG_MAPPING

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> Step37ModelProvider:
        """Convert a HuggingFace Step3.7 config into a :class:`Step37ModelProvider`.

        Mirrors the qwen3-vl bridge pattern:

        * Pull the nested ``text_config`` directly out of the top-level Step3.7
          ``Step37Config`` and run the framework helper
          ``self.hf_config_to_provider_kwargs(text_config)`` to populate the
          common architecture fields (``num_layers`` / ``hidden_size`` /
          ``num_attention_heads`` / ``ffn_hidden_size`` / ``vocab_size`` /
          ``rotary_base`` / etc.) via :attr:`CONFIG_MAPPING`. That helper
          uses ``hasattr`` + ``getattr(..., None)`` internally, so fields
          that are absent on the Step3.7 text config (e.g. anything Step-3.5
          carried at the top level of *its* config.json) are skipped cleanly.

        * Construct :class:`Step37ModelProvider` directly from the filtered
          kwargs (instead of delegating to :meth:`Step35Bridge.provider_bridge`
          via a wrapper — that path was fragile because Step35Bridge does
          a number of bare ``hf_config.X`` reads that crash on missing
          fields like ``zero_centered`` or ``use_qk_norm``).

        * Apply Step-3.5 text-decoder overrides with explicit
          ``getattr(text_config, name, default)`` for every field that
          may or may not be present in the released Step3.7 ``text_config``.

        * Finally attach Step3.7 vision / multimodal fields from the
          top-level ``hf_config``.
        """
        hf_config = hf_pretrained.config
        text_config = hf_config.text_config
        vision_config = hf_config.vision_config

        # ── 1. Framework-safe mapping of the text config ─────────────────────
        # ``hf_config_to_provider_kwargs`` iterates ``self.CONFIG_MAPPING``
        # (which we inherit from :class:`Step35Bridge`) and uses
        # ``hasattr`` + ``getattr(..., None)`` to skip missing HF fields.
        provider_kwargs = self.hf_config_to_provider_kwargs(text_config)
        # Filter to the multimodal provider's dataclass fields so an
        # unexpected key from ``CONFIG_MAPPING`` (or a future extension)
        # never causes the ``Step37ModelProvider(**kwargs)`` call below to
        # raise ``TypeError: unexpected keyword argument``.
        valid_fields = Step37ModelProvider.__dataclass_fields__
        provider_kwargs = {k: v for k, v in provider_kwargs.items() if k in valid_fields}

        provider = Step37ModelProvider(**provider_kwargs)

        # ── 2. Step-3.5 text-decoder overrides (safe getattr+defaults) ──────
        provider.normalization = "RMSNorm"
        provider.gated_linear_unit = True
        provider.add_bias_linear = False
        provider.add_qkv_bias = False
        provider.hidden_dropout = 0.0
        provider.attention_dropout = 0.0

        # Mirror qwen3-vl's explicit assignment (qwen3_vl_bridge.py:96). The
        # framework base ``MegatronModelBridge.provider_bridge`` would flip
        # this field to ``"rope"`` (or ``"yarn"``) via its rope_scaling branch,
        # but this bridge constructs the provider directly without calling
        # ``super().provider_bridge``, so we set it explicitly. This is the
        # field ``SelfAttention.__init__`` reads when ``rotary_base_per_layer``
        # is set; ``"learned_absolute"`` would raise NotImplementedError.
        provider.position_embedding_type = "rope"

        # ``zero_centered`` lived at the top of Step-3.5-Flash's config.json
        # (=true) but is absent from the Step3.7 text_config — default to
        # True to match the Step-3.5 RMSNorm convention.
        provider.layernorm_zero_centered_gamma = bool(getattr(text_config, "zero_centered", True))
        provider.qk_layernorm = bool(getattr(text_config, "use_qk_norm", True))

        # ``tie_word_embeddings`` exists at the top of Step-3.5-Flash's
        # config.json (=false) but is absent from the Step3.7 text_config.
        # ``hf_config_to_provider_kwargs`` uses ``hasattr`` to decide whether
        # to copy ``tie_word_embeddings -> share_embeddings_and_output_weights``,
        # so on Step3.7 the field is skipped and the provider would inherit the
        # ``GPTModelProvider`` dataclass default (``True``) — which ties the
        # output layer to the embedding inside Megatron. Step-3.5/3.7 actually
        # ship distinct ``lm_head.weight`` and ``model.embed_tokens.weight``
        # tensors, so the round-trip HF→Megatron→HF then writes the embedding
        # values into the exported ``lm_head.weight``. Mirror qwen3_vl_bridge's
        # pattern (qwen3_vl_bridge.py:93) and default to False for this family.
        provider.share_embeddings_and_output_weights = bool(getattr(text_config, "tie_word_embeddings", False))

        # Per-layer partial RoPE fractions ([0.5, 1.0, 1.0, 1.0, …]).
        rotary_percents = getattr(text_config, "partial_rotary_factors", None)
        if rotary_percents is not None:
            provider.rotary_percents = list(rotary_percents)

        # Sliding-attention shape overrides. Defaults match Step-3.5-Flash;
        # we then overlay anything the HF config provides.
        provider.sliding_attention_setting = {
            "window_size": [512, 0],
            "num_attention_heads": 96,
            "num_query_groups": 8,
            "kv_channels": 128,
        }
        sliding_window = getattr(text_config, "sliding_window", None)
        if sliding_window is not None:
            provider.sliding_attention_setting["window_size"] = [int(sliding_window), 0]
        attn_other = getattr(text_config, "attention_other_setting", None) or {}
        if isinstance(attn_other, dict) and attn_other.get("attention_type") == "sliding_attention":
            if "num_attention_heads" in attn_other:
                provider.sliding_attention_setting["num_attention_heads"] = int(attn_other["num_attention_heads"])
            if "num_attention_groups" in attn_other:
                provider.sliding_attention_setting["num_query_groups"] = int(attn_other["num_attention_groups"])
            if "head_dim" in attn_other:
                provider.sliding_attention_setting["kv_channels"] = int(attn_other["head_dim"])
        provider.attention_other_setting = attn_other if isinstance(attn_other, dict) else None

        # Per-layer RoPE base. Step-3.5 ships ``rope_theta`` as a list (one
        # entry per text layer); fall back to scalar for forward-compat.
        rope_theta = getattr(text_config, "rope_theta", None)
        if isinstance(rope_theta, list):
            provider.rotary_base = float(rope_theta[0])
            provider.rotary_base_per_layer = [float(x) for x in rope_theta]
        elif rope_theta is not None:
            provider.rotary_base = float(rope_theta)

        # Dtype handling — accept string or torch.dtype, ignore if absent.
        torch_dtype = getattr(text_config, "torch_dtype", None)
        if isinstance(torch_dtype, str):
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            if torch_dtype not in dtype_map:
                raise ValueError(f"Unknown torch dtype: {torch_dtype}")
            provider.autocast_dtype = dtype_map[torch_dtype]
        elif isinstance(torch_dtype, torch.dtype):
            provider.autocast_dtype = torch_dtype
        elif torch_dtype is not None:
            raise ValueError(f"Unknown torch dtype: {torch_dtype}")

        # MoE settings.
        provider.moe_router_enable_expert_bias = bool(getattr(text_config, "use_moe_router_bias", False))
        moe_router_activation = getattr(text_config, "moe_router_activation", None)
        if moe_router_activation is not None:
            provider.moe_router_score_function = moe_router_activation
        moe_router_scaling_factor = getattr(text_config, "moe_router_scaling_factor", None)
        if moe_router_scaling_factor is not None:
            provider.moe_router_topk_scaling_factor = float(moe_router_scaling_factor)
        provider.swiglu_limits = getattr(text_config, "swiglu_limits", None)
        provider.swiglu_limits_shared = getattr(text_config, "swiglu_limits_shared", None)
        if bool(getattr(text_config, "need_fp32_gate", False)):
            provider.moe_router_dtype = "fp32"

        provider.moe_grouped_gemm = True
        provider.moe_router_load_balancing_type = "aux_loss"
        provider.moe_aux_loss_coeff = 1e-3
        provider.moe_router_pre_softmax = False
        provider.moe_token_dispatcher_type = "alltoall"
        provider.moe_permute_fusion = True

        # Per-layer MoE / dense schedule, plus the Step-3.5 layer-spec
        # builder (handles MTP / hybrid attention / per-layer overrides).
        moe_layers_enum = getattr(text_config, "moe_layers_enum", None)
        if moe_layers_enum is not None:
            moe_layer_freq = [0] * provider.num_layers
            if isinstance(moe_layers_enum, str):
                moe_layers = [int(layer) for layer in moe_layers_enum.split(",") if layer]
            else:
                moe_layers = [int(layer) for layer in moe_layers_enum]
            for idx in moe_layers:
                if 0 <= idx < provider.num_layers:
                    moe_layer_freq[idx] = 1
            provider.moe_layer_freq = moe_layer_freq
            provider.transformer_layer_spec = build_step35_layer_spec

        # Per-layer hybrid full/sliding attention schedule.
        layer_types = getattr(text_config, "layer_types", None)
        if layer_types is not None:
            provider.layer_types = list(layer_types)

        # Head-wise attention gate (default is True for Step3.7 text config).
        head_wise_attn_gate = bool(getattr(text_config, "use_head_wise_attn_gate", True))
        provider.head_wise_attn_gate = head_wise_attn_gate

        # ── 3. Step3.7 multimodal / vision fields ───────────────────────────
        provider.vision_config = vision_config
        provider.image_token_id = int(getattr(hf_config, "image_token_id", 128001))
        provider.understand_projector_stride = int(getattr(hf_config, "understand_projector_stride", 2))
        provider.projector_bias = bool(getattr(hf_config, "projector_bias", False))

        # Long-context sequence length so Step37GPTModel sizes the rotary
        # cache to the full Step-3.5 262144-position window.
        provider.language_max_sequence_length = int(getattr(text_config, "max_position_embeddings", 262144))

        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Return the full text + vision parameter mapping registry.

        Text mappings replicate :meth:`Step35Bridge.mapping_registry` with a
        ``language_model.`` prefix on every Megatron-side path (since
        :class:`Step37Model` wraps the Step-3.5 ``GPTModel`` under
        ``self.language_model``). Vision mappings are direct AutoMappings —
        the Megatron module structure mirrors the HF safetensors layout.
        """
        mapping_list: List = []

        # ──────────────────────────── Text mappings ─────────────────────────
        text_param_mappings = {
            _lm("embedding.word_embeddings.weight"): "model.embed_tokens.weight",
            _lm("output_layer.weight"): "lm_head.weight",
            _lm("decoder.final_layernorm.weight"): "model.norm.weight",
            _lm("decoder.layers.*.input_layernorm.weight"): "model.layers.*.input_layernorm.weight",
            _lm(
                "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight"
            ): "model.layers.*.input_layernorm.weight",
            _lm("decoder.layers.*.self_attention.q_layernorm.weight"): "model.layers.*.self_attn.q_norm.weight",
            _lm("decoder.layers.*.self_attention.k_layernorm.weight"): "model.layers.*.self_attn.k_norm.weight",
            _lm("decoder.layers.*.self_attention.linear_proj.weight"): "model.layers.*.self_attn.o_proj.weight",
            _lm("decoder.layers.*.pre_mlp_layernorm.weight"): "model.layers.*.post_attention_layernorm.weight",
            _lm("decoder.layers.*.mlp.linear_fc1.layer_norm_weight"): "model.layers.*.post_attention_layernorm.weight",
            _lm("decoder.layers.*.mlp.linear_fc2.weight"): "model.layers.*.mlp.down_proj.weight",
            _lm(
                "decoder.layers.*.mlp.shared_experts.linear_fc2.weight"
            ): "model.layers.*.share_expert.down_proj.weight",
            _lm("decoder.layers.*.mlp.router.weight"): "model.layers.*.moe.gate.weight",
            _lm("decoder.layers.*.mlp.router.expert_bias"): "model.layers.*.moe.router_bias",
        }
        for megatron_param, hf_param in text_param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        mapping_list.extend(
            [
                QKVGMapping(
                    megatron_param=_lm("decoder.layers.*.self_attention.linear_qkv.weight"),
                    q="model.layers.*.self_attn.q_proj.weight",
                    k="model.layers.*.self_attn.k_proj.weight",
                    v="model.layers.*.self_attn.v_proj.weight",
                    g="model.layers.*.self_attn.g_proj.weight",
                ),
                GatedMLPMapping(
                    megatron_param=_lm("decoder.layers.*.mlp.linear_fc1.weight"),
                    gate="model.layers.*.mlp.gate_proj.weight",
                    up="model.layers.*.mlp.up_proj.weight",
                ),
                StackedExpertGatedMLPMapping(
                    megatron_param=_lm("decoder.layers.*.mlp.experts.linear_fc1.weight*"),
                    gate="model.layers.*.moe.gate_proj.weight",
                    up="model.layers.*.moe.up_proj.weight",
                ),
                GatedMLPMapping(
                    megatron_param=_lm("decoder.layers.*.mlp.shared_experts.linear_fc1.weight"),
                    gate="model.layers.*.share_expert.gate_proj.weight",
                    up="model.layers.*.share_expert.up_proj.weight",
                ),
                StackedExpertAutoMapping(
                    megatron_param=_lm("decoder.layers.*.mlp.experts.linear_fc2.weight*"),
                    hf_param="model.layers.*.moe.down_proj.weight",
                ),
            ]
        )

        # MTP layers (45–47 for Step3.7; same schema as Step-3.5).
        text_config = getattr(self.hf_config, "text_config", self.hf_config)
        if text_config is None:
            logger.warning("No HF text_config found; skipping MTP mappings.")
        else:
            mtp_num_layers = getattr(text_config, "num_nextn_predict_layers", 0)
            num_transformer_layers = text_config.num_hidden_layers

            mtp_layer_param_mappings = {
                "decoder.layers.*.input_layernorm.weight": "model.layers.*.input_layernorm.weight",
                "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
                "decoder.layers.*.pre_mlp_layernorm.weight": "model.layers.*.post_attention_layernorm.weight",
                "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.post_attention_layernorm.weight",
                "decoder.layers.*.self_attention.q_layernorm.weight": "model.layers.*.self_attn.q_norm.weight",
                "decoder.layers.*.self_attention.k_layernorm.weight": "model.layers.*.self_attn.k_norm.weight",
                "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
                "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
            }

            for mtp_layer in range(mtp_num_layers):
                hf_layer = mtp_layer + num_transformer_layers
                for layer_prefix in ("mtp_model_layer", "transformer_layer"):
                    for megatron_param, hf_param in mtp_layer_param_mappings.items():
                        megatron_param_mtp = (
                            megatron_param.replace(".*", f".*.{layer_prefix}")
                            .replace("decoder", "mtp")
                            .replace(".*", f".{mtp_layer}")
                        )
                        hf_param_mtp = hf_param.replace("layers.*", f"layers.{hf_layer}")
                        mapping_list.append(
                            AutoMapping(
                                megatron_param=_lm(megatron_param_mtp),
                                hf_param=hf_param_mtp,
                            )
                        )

                    mapping_list.extend(
                        [
                            QKVGMapping(
                                megatron_param=_lm(
                                    f"mtp.layers.{mtp_layer}.{layer_prefix}.self_attention.linear_qkv.weight"
                                ),
                                q=f"model.layers.{hf_layer}.self_attn.q_proj.weight",
                                k=f"model.layers.{hf_layer}.self_attn.k_proj.weight",
                                v=f"model.layers.{hf_layer}.self_attn.v_proj.weight",
                                g=f"model.layers.{hf_layer}.self_attn.g_proj.weight",
                            ),
                            GatedMLPMapping(
                                megatron_param=_lm(f"mtp.layers.{mtp_layer}.{layer_prefix}.mlp.linear_fc1.weight"),
                                gate=f"model.layers.{hf_layer}.mlp.gate_proj.weight",
                                up=f"model.layers.{hf_layer}.mlp.up_proj.weight",
                            ),
                            AutoMapping(
                                megatron_param=_lm(f"mtp.layers.{mtp_layer}.{layer_prefix}.mlp.linear_fc2.weight"),
                                hf_param=f"model.layers.{hf_layer}.mlp.down_proj.weight",
                            ),
                        ]
                    )

                mapping_list.extend(
                    [
                        AutoMapping(
                            megatron_param=_lm(f"mtp.layers.{mtp_layer}.enorm.weight"),
                            hf_param=f"model.layers.{hf_layer}.enorm.weight",
                        ),
                        AutoMapping(
                            megatron_param=_lm(f"mtp.layers.{mtp_layer}.hnorm.weight"),
                            hf_param=f"model.layers.{hf_layer}.hnorm.weight",
                        ),
                        AutoMapping(
                            megatron_param=_lm(f"mtp.layers.{mtp_layer}.eh_proj.weight"),
                            hf_param=f"model.layers.{hf_layer}.eh_proj.weight",
                        ),
                        AutoMapping(
                            megatron_param=_lm(f"mtp.layers.{mtp_layer}.final_layernorm.weight"),
                            hf_param=f"model.layers.{hf_layer}.transformer.shared_head.norm.weight",
                        ),
                    ]
                )

        # ─────────────────────────── Vision mappings ────────────────────────
        # All vision params live under ``vision_model.*`` in BOTH the
        # Megatron module (see :class:`Step37Model`) and the HF safetensors
        # index, so every entry is a 1:1 mapping with the same name on each
        # side.
        #
        # Design choice (vs qwen3vl):
        #   This bridge keeps the PE-G/14 trunk **HF-aligned**: bare
        #   ``nn.Parameter`` for fused QKV, plain ``nn.Linear`` for output
        #   projection / MLP, ``nn.LayerNorm`` for norms, custom
        #   ``EncoderLayerScale`` for ``ls_*.gamma``. The entire tower runs
        #   on a **TP=1 mesh** (decoupled vision encoder, see
        #   :class:`ImageInsertDecoderMixin`) so every vision param is
        #   semantically **replicated**.
        #
        #   Consequence: most owning module types are *not* in
        #   ``AutoMapping._MODULE_TYPE_REGISTRY``. Specifically:
        #     * ``EncoderVisionAttention`` (owns ``attn.in_proj_{weight,bias}``)
        #     * ``EncoderLayerScale``     (owns ``ls_{1,2}.gamma``)
        #     * ``Conv2d``                (owns ``conv1.weight`` / ``vit_downsampler*``)
        #     * ``Step37VisionModel``    (owns ``positional_embedding`` directly)
        #     * plain ``nn.Linear``       (owns ``attn.out_proj.*`` / ``mlp.c_*.*``
        #                                  / top-level ``vit_large_projector.weight``)
        #     * built-in ``nn.LayerNorm`` (owns ``ln_pre.*`` / ``ln_{1,2}.*``) IS
        #       handled by AutoMapping's "Norm/Normalization" fallback in
        #       ``_detect_parallelism_type``, so layer norms are the *only*
        #       AutoMapping-safe vision params.
        #
        #   So we route layer norms through ``AutoMapping`` (they fall back
        #   to "replicated" via the Norm-substring branch) and route
        #   *everything else* through ``ReplicatedMapping`` explicitly.
        #   Contrast with qwen3vl which uses MCore ``SelfAttention``
        #   (``linear_qkv`` / ``linear_proj`` ARE in the registry) and
        #   ``ConcatenatedQKVMapping`` to fold HF's fused QKV into MCore's
        #   ``[3*H, H]`` ``linear_qkv``.

        # Norms — AutoMapping handles these via the "Norm/Normalization"
        # substring fallback.
        vision_norm_param_mappings = {
            "vision_model.ln_pre.weight": "vision_model.ln_pre.weight",
            "vision_model.ln_pre.bias": "vision_model.ln_pre.bias",
            "vision_model.transformer.resblocks.*.ln_1.weight": "vision_model.transformer.resblocks.*.ln_1.weight",
            "vision_model.transformer.resblocks.*.ln_1.bias": "vision_model.transformer.resblocks.*.ln_1.bias",
            "vision_model.transformer.resblocks.*.ln_2.weight": "vision_model.transformer.resblocks.*.ln_2.weight",
            "vision_model.transformer.resblocks.*.ln_2.bias": "vision_model.transformer.resblocks.*.ln_2.bias",
        }
        for megatron_param, hf_param in vision_norm_param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        # Everything else in the vision tower → ReplicatedMapping. Grouped
        # by physical category for readability; one big loop at the end.
        vision_replicated_param_mappings = {
            # Patch embedding + downsampler convs (Conv2d, not in registry).
            "vision_model.conv1.weight": "vision_model.conv1.weight",
            "vision_model.vit_downsampler1.weight": "vision_model.vit_downsampler1.weight",
            "vision_model.vit_downsampler1.bias": "vision_model.vit_downsampler1.bias",
            "vision_model.vit_downsampler2.weight": "vision_model.vit_downsampler2.weight",
            "vision_model.vit_downsampler2.bias": "vision_model.vit_downsampler2.bias",
            # Top-level nn.Parameter on Step37VisionModel — AutoMapping
            # would see the top-level vision class as the owning module.
            "vision_model.positional_embedding": "vision_model.positional_embedding",
            # Attention — in_proj_* are bare nn.Parameter on
            # EncoderVisionAttention; out_proj.* are plain nn.Linear.
            "vision_model.transformer.resblocks.*.attn.in_proj_weight": "vision_model.transformer.resblocks.*.attn.in_proj_weight",
            "vision_model.transformer.resblocks.*.attn.in_proj_bias": "vision_model.transformer.resblocks.*.attn.in_proj_bias",
            "vision_model.transformer.resblocks.*.attn.out_proj.weight": "vision_model.transformer.resblocks.*.attn.out_proj.weight",
            "vision_model.transformer.resblocks.*.attn.out_proj.bias": "vision_model.transformer.resblocks.*.attn.out_proj.bias",
            # LayerScale gates — gamma is an nn.Parameter on EncoderLayerScale.
            "vision_model.transformer.resblocks.*.ls_1.gamma": "vision_model.transformer.resblocks.*.ls_1.gamma",
            "vision_model.transformer.resblocks.*.ls_2.gamma": "vision_model.transformer.resblocks.*.ls_2.gamma",
            # MLP — plain nn.Linear (c_fc, c_proj) inside EncoderMLP.
            "vision_model.transformer.resblocks.*.mlp.c_fc.weight": "vision_model.transformer.resblocks.*.mlp.c_fc.weight",
            "vision_model.transformer.resblocks.*.mlp.c_fc.bias": "vision_model.transformer.resblocks.*.mlp.c_fc.bias",
            "vision_model.transformer.resblocks.*.mlp.c_proj.weight": "vision_model.transformer.resblocks.*.mlp.c_proj.weight",
            "vision_model.transformer.resblocks.*.mlp.c_proj.bias": "vision_model.transformer.resblocks.*.mlp.c_proj.bias",
            # Vision → LM projector. On the Megatron side the projector lives
            # inside ``image_insert_embedding.align_projector``; on the HF side
            # it is the top-level ``vit_large_projector`` linear.
            "image_insert_embedding.align_projector.weight": "vit_large_projector.weight",
        }
        for megatron_param, hf_param in vision_replicated_param_mappings.items():
            mapping_list.append(ReplicatedMapping(megatron_param=megatron_param, hf_param=hf_param))

        return MegatronMappingRegistry(*mapping_list)


__all__ = ["Step37Bridge"]
