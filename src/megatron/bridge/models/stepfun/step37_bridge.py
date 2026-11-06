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

* Maps the config to a builder-backed :class:`Step37ModelConfig`: the
  language fields come from :class:`Step35Bridge` applied to
  ``hf_config.text_config`` (the Step3.7 HF config nests its Step-3.5 text
  fields there), and the vision-tower fields (vision_config, image_token_id,
  projector_bias, understand_projector_stride) come from the top-level config.
* Defines an HF↔Megatron parameter mapping registry that prefixes every
  Step-3.5 text mapping with ``language_model.`` (since
  :class:`Step37Model` wraps the text GPTModel) and adds direct
  ``vision_model.*`` AutoMappings for the PE-G/14 trunk + downsamplers, plus
  a top-level ``vit_large_projector.weight`` mapping.
"""

from __future__ import annotations

import logging
from typing import Any, List

from transformers import AutoConfig, PretrainedConfig

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVGMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.stepfun.configuration_step37 import Step37Config
from megatron.bridge.models.stepfun.modelling_step37.model import Step37Model
from megatron.bridge.models.stepfun.step35_bridge import (
    StackedExpertAutoMapping,
    StackedExpertGatedMLPMapping,
    Step35Bridge,
)
from megatron.bridge.models.stepfun.step37_config import Step37ModelConfig
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
        >>> bridge = AutoBridge.from_hf_pretrained("stepfun-ai/Step-3.7-Flash")
        >>> model_config = bridge.get_model_config()
    """

    MODEL_CONFIG_CLASS = Step37ModelConfig
    USE_MODEL_CONFIG_FOR_CONVERSION = True

    # The same translation table Step35Bridge uses (so the shared mapping
    # picks up the text-side renames). Step3.7's text config uses the
    # identical HF schema.
    CONFIG_MAPPING = Step35Bridge.CONFIG_MAPPING

    @classmethod
    def megatron_to_hf_config(cls, provider: Step37ModelConfig | Step37ModelProvider) -> dict:
        """Convert a Step3.7 model config or provider into its nested Hugging Face config."""
        hf_config = super().megatron_to_hf_config(provider)
        mtp_num_layers = hf_config.pop("num_nextn_predict_layers", None)
        if mtp_num_layers is not None:
            hf_config["text_config"] = {"num_nextn_predict_layers": mtp_num_layers}
        projector_bias = getattr(provider, "projector_bias", None)
        if projector_bias is not None:
            hf_config["projector_bias"] = projector_bias
        return hf_config

    def hf_config_to_model_config_kwargs(self, hf_config: PretrainedConfig) -> dict[str, Any]:
        """Convert a Hugging Face Step3.7 config to builder config kwargs.

        The language fields come from :class:`Step35Bridge` applied to
        ``hf_config.text_config``. Where the Step3.7 provider configured a field
        differently, it is set here as the provider set it. The vision-language
        fields come from the top-level config.
        """
        text_config = hf_config.text_config
        config_kwargs = Step35Bridge().hf_config_to_model_config_kwargs(text_config)

        # The Step3.7 text_config omits ``use_qk_norm`` and ``tie_word_embeddings``, which
        # Step-3.5-Flash's config.json carries. Step3.7 normalizes Q and K and ships a
        # separate ``lm_head.weight``, so the defaults differ from Step-3.5's.
        config_kwargs["qk_layernorm"] = bool(getattr(text_config, "use_qk_norm", True))
        config_kwargs["share_embeddings_and_output_weights"] = bool(getattr(text_config, "tie_word_embeddings", False))
        config_kwargs["head_wise_attn_gate"] = bool(getattr(text_config, "use_head_wise_attn_gate", True))
        # Step3.7 requires Megatron Core's native head-wise gate (see the example README), so the
        # output-gate fallback Step-3.5 selects for older Megatron Core does not apply.
        config_kwargs.pop("attention_output_gate", None)
        if hasattr(text_config, "attention_output_gate"):
            config_kwargs["attention_output_gate"] = text_config.attention_output_gate
        # Step3.7 builds no MTP block, so the attention types stay those of the main decoder.
        layer_types = getattr(text_config, "layer_types", None)
        if layer_types is not None:
            config_kwargs["layer_types"] = list(layer_types)
        config_kwargs["position_embedding_type"] = "rope"

        config_kwargs.update(
            vision_config=hf_config.vision_config,
            image_token_id=int(getattr(hf_config, "image_token_id", 128001)),
            understand_projector_stride=int(getattr(hf_config, "understand_projector_stride", 2)),
            projector_bias=bool(getattr(hf_config, "projector_bias", False)),
            # Size the rotary cache to the full Step-3.5 262144-position window.
            language_max_sequence_length=int(getattr(text_config, "max_position_embeddings", 262144)),
        )
        return config_kwargs

    def hf_config_to_provider_kwargs(self, hf_config: PretrainedConfig) -> dict[str, Any]:
        """Adapt the canonical builder mapping to the deprecated provider path."""
        provider_kwargs = self.hf_config_to_model_config_kwargs(hf_config)
        # The provider also records the HF sliding-attention setting, which no module reads.
        attention_other_setting = getattr(hf_config.text_config, "attention_other_setting", None) or {}
        provider_kwargs["attention_other_setting"] = (
            attention_other_setting if isinstance(attention_other_setting, dict) else None
        )
        return provider_kwargs

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
            "vision_model.ln_post.weight": "vision_model.ln_post.weight",
            "vision_model.ln_post.bias": "vision_model.ln_post.bias",
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
            "vision_model.class_embedding": "vision_model.class_embedding",
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
            "image_insert_embedding.align_projector.bias": "vit_large_projector.bias",
        }
        for megatron_param, hf_param in vision_replicated_param_mappings.items():
            mapping_list.append(ReplicatedMapping(megatron_param=megatron_param, hf_param=hf_param))

        return MegatronMappingRegistry(*mapping_list)


__all__ = ["Step37Bridge"]
