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

"""Hugging Face ↔ Megatron conversion bridge for Muse Glimmer."""

from __future__ import annotations

import math
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    MegatronParamMapping,
    ReplicatedMapping,
    RMSNorm2ZeroCenteredRMSNormMapping,
    merge_qkv_weights,
    split_qkv_weights,
)
from megatron.bridge.models.conversion.utils import remove_non_pickleables
from megatron.bridge.models.muse_glimmer.modeling_muse_glimmer import MuseGlimmerModel
from megatron.bridge.models.muse_glimmer.muse_glimmer_config import (
    MuseGlimmerModelConfig,
    MuseGlimmerTransformerConfig,
    MuseGlimmerVisionModelConfig,
)


class MuseGlimmerQKVGMapping(MegatronParamMapping[dict[str, torch.Tensor]]):
    """Fuse Muse's full-head Q/K/V/gate projections into MCore gated QKV."""

    def __init__(self, megatron_param: str, *, q: str, k: str, v: str, gate: str) -> None:
        super().__init__(megatron_param, {"q": q, "k": k, "v": v, "gate": gate})
        self._tp_mapping = AutoMapping(megatron_param, megatron_param)

    @staticmethod
    def _combine_query_and_gate(
        config: MuseGlimmerTransformerConfig,
        query: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        if query.ndim != 2 or gate.ndim != 2:
            raise ValueError("Muse Glimmer QKV/gate mapping supports weight matrices only.")
        if query.shape != gate.shape:
            raise ValueError(f"Muse query and gate shapes must match, got {query.shape} and {gate.shape}.")
        head_dim = config.kv_channels or config.hidden_size // config.num_attention_heads
        expected_rows = config.num_attention_heads * head_dim
        if query.shape != (expected_rows, config.hidden_size):
            raise ValueError(
                "Unexpected Muse query/gate shape: "
                f"expected {(expected_rows, config.hidden_size)}, got {tuple(query.shape)}."
            )
        query = query.view(config.num_attention_heads, head_dim, config.hidden_size)
        gate = gate.view(config.num_attention_heads, head_dim, config.hidden_size)
        return torch.cat((query, gate), dim=1).reshape(-1, config.hidden_size)

    def hf_to_megatron(
        self,
        hf_weights: dict[str, torch.Tensor],
        megatron_module: nn.Module,
    ) -> torch.Tensor:
        if self.tp_rank == 0:
            config = self._get_config(megatron_module)
            query_with_gate = self._combine_query_and_gate(config, hf_weights["q"], hf_weights["gate"])
            merged = merge_qkv_weights(config, query_with_gate, hf_weights["k"], hf_weights["v"])
        else:
            merged = None
        return self._tp_mapping.hf_to_megatron(merged, megatron_module)

    def megatron_to_hf(
        self,
        megatron_weights: torch.Tensor | None,
        megatron_module: nn.Module | None,
    ) -> dict[str, torch.Tensor]:
        if megatron_weights is not None:
            megatron_weights = self.maybe_dequantize(megatron_weights)

        if megatron_module is None:
            config = self.broadcast_obj_from_pp_rank(None, "muse_qkvg_config")
        else:
            config = remove_non_pickleables(self._get_config(megatron_module), max_depth=3)
            config = self.broadcast_obj_from_pp_rank(config, "muse_qkvg_config")

        packed = self._tp_mapping.megatron_to_hf(megatron_weights, megatron_module)
        if not packed:
            return {}

        query_with_gate, key, value = split_qkv_weights(config, next(iter(packed.values())))
        head_dim = config.kv_channels or config.hidden_size // config.num_attention_heads
        query_with_gate = query_with_gate.view(
            config.num_attention_heads,
            2 * head_dim,
            config.hidden_size,
        )
        query, gate = query_with_gate.split(head_dim, dim=1)
        hf_param = cast(dict[str, str], self.hf_param)
        return {
            hf_param["q"]: query.reshape(-1, config.hidden_size),
            hf_param["k"]: key,
            hf_param["v"]: value,
            hf_param["gate"]: gate.reshape(-1, config.hidden_size),
        }

    def resolve(self, captures: tuple[str, ...]) -> "MuseGlimmerQKVGMapping":
        megatron_param, hf_param = self._resolve_names(captures)
        return type(self)(
            megatron_param,
            q=hf_param["q"],
            k=hf_param["k"],
            v=hf_param["v"],
            gate=hf_param["gate"],
        )


@MegatronModelBridge.register_bridge(
    source="MuseGlimmerForConditionalGeneration",
    target=MuseGlimmerModel,
    model_type="muse_glimmer",
)
class MuseGlimmerBridge(MegatronModelBridge):
    """Builder-backed bridge for the complete Muse Glimmer multimodal model."""

    MODEL_CONFIG_CLASS = MuseGlimmerModelConfig
    USE_MODEL_CONFIG_FOR_CONVERSION = True

    @staticmethod
    def _validate_architecture(hf_config: PretrainedConfig) -> tuple[Any, Any]:
        text_config = getattr(hf_config, "text_config", None)
        vision_config = getattr(hf_config, "vision_config", None)
        if text_config is None or vision_config is None:
            raise ValueError("Muse Glimmer requires both text_config and vision_config.")

        if getattr(text_config, "hidden_activation", "silu") != "silu":
            raise ValueError("Muse Glimmer currently supports only the released SiLU text MLP.")
        if bool(getattr(text_config, "attention_bias", False)):
            raise ValueError("Muse Glimmer currently supports only the released bias-free text attention.")
        if getattr(vision_config, "hidden_act", "gelu") != "gelu":
            raise ValueError("Muse Glimmer currently supports only the released GELU vision MLP.")
        if getattr(hf_config, "projector_hidden_act", "gelu") != "gelu":
            raise ValueError("Muse Glimmer currently supports only the released GELU projector.")

        text_layers = list(text_config.layer_types)
        rope_theta = list(text_config.layer_rope_theta)
        vision_layers = list(vision_config.layer_types)
        if len(text_layers) != text_config.num_hidden_layers or len(rope_theta) != text_config.num_hidden_layers:
            raise ValueError("Muse text layer_types and layer_rope_theta must match num_hidden_layers.")
        if len(vision_layers) != vision_config.num_hidden_layers:
            raise ValueError("Muse vision layer_types must match num_hidden_layers.")
        if any(layer not in {"sliding_attention", "full_attention"} for layer in text_layers):
            raise ValueError(f"Unsupported Muse text layer pattern: {text_layers}.")
        if any(layer not in {"window_attention", "full_attention"} for layer in vision_layers):
            raise ValueError(f"Unsupported Muse vision layer pattern: {vision_layers}.")

        expected_vision_output = vision_config.hidden_size * int(vision_config.merge_size) ** 2
        if int(hf_config.out_hidden_size) != expected_vision_output:
            raise ValueError(
                f"out_hidden_size={hf_config.out_hidden_size} does not match pixel-shuffled vision width "
                f"{expected_vision_output}."
            )
        return text_config, vision_config

    def hf_config_to_model_config(self, hf_config: PretrainedConfig) -> MuseGlimmerModelConfig:
        """Translate the nested HF config into a native HybridModel config."""
        text_config, vision_config = self._validate_architecture(hf_config)
        params_dtype = self.dtype_from_hf(hf_config, default=torch.bfloat16)
        head_dim = int(text_config.head_dim)
        rope_parameters = getattr(text_config, "rope_parameters", {})
        rotary_base = float(rope_parameters.get("rope_theta", 500_000.0))
        layer_types = list(text_config.layer_types)
        layer_rope_theta = list(text_config.layer_rope_theta)
        sliding_window = getattr(text_config, "sliding_window", None)
        if not isinstance(sliding_window, int) or sliding_window <= 0:
            raise ValueError("Muse Glimmer requires a positive integer sliding_window.")

        transformer = MuseGlimmerTransformerConfig(
            num_layers=int(text_config.num_hidden_layers),
            hidden_size=int(text_config.hidden_size),
            ffn_hidden_size=int(text_config.intermediate_size),
            num_attention_heads=int(text_config.num_attention_heads),
            num_query_groups=int(text_config.num_key_value_heads),
            kv_channels=head_dim,
            normalization="RMSNorm",
            layernorm_epsilon=float(text_config.rms_norm_eps),
            post_norm_epsilon=float(text_config.post_norm_eps),
            layernorm_zero_centered_gamma=True,
            gated_linear_unit=True,
            activation_func=F.silu,
            add_bias_linear=False,
            add_qkv_bias=bool(text_config.attention_bias),
            hidden_dropout=0.0,
            attention_dropout=float(text_config.attention_dropout),
            qk_layernorm=True,
            attention_output_gate=True,
            softmax_scale=float(text_config.qk_scale_factor) / math.sqrt(head_dim),
            attention_softmax_in_fp32=True,
            window_size=(sliding_window - 1, 0),
            window_attn_skip_freq=[layer == "sliding_attention" for layer in layer_types],
            no_rope_freq=[float(theta) == 0.0 for theta in layer_rope_theta],
            init_method_std=float(text_config.initializer_range),
            params_dtype=params_dtype,
            fp16=params_dtype == torch.float16,
            bf16=params_dtype == torch.bfloat16,
            transformer_impl="local",
            output_multiplier=float(text_config.output_multiplier),
            final_logit_softcapping=float(text_config.final_logit_softcapping),
        )
        vision_rope = getattr(vision_config, "rope_parameters", {})
        vision = MuseGlimmerVisionModelConfig(
            hidden_size=int(vision_config.hidden_size),
            intermediate_size=int(vision_config.intermediate_size),
            num_hidden_layers=int(vision_config.num_hidden_layers),
            num_attention_heads=int(vision_config.num_attention_heads),
            patch_size=int(vision_config.patch_size),
            patch_temporal=int(vision_config.patch_temporal),
            merge_size=int(vision_config.merge_size),
            pos_emb_height=int(vision_config.pos_emb_height),
            pos_emb_width=int(vision_config.pos_emb_width),
            max_position_embeddings=int(vision_config.max_position_embeddings),
            layer_norm_epsilon=float(vision_config.layer_norm_eps),
            hidden_activation=str(vision_config.hidden_act),
            rotary_base=float(vision_rope.get("rope_theta", 10_000.0)),
            layer_types=list(vision_config.layer_types),
        )
        return MuseGlimmerModelConfig(
            transformer=transformer,
            vocab_size=int(text_config.vocab_size),
            make_vocab_size_divisible_by=self.make_vocab_size_divisible_by(int(text_config.vocab_size)),
            should_pad_vocab=False,
            seq_length=int(text_config.max_position_embeddings),
            parallel_output=True,
            share_embeddings_and_output_weights=bool(
                getattr(hf_config, "tie_word_embeddings", text_config.tie_word_embeddings)
            ),
            hybrid_layer_pattern="*" * int(text_config.num_hidden_layers),
            position_embedding_type="rope",
            rotary_percent=1.0,
            rotary_base=rotary_base,
            vision=vision,
            image_token_id=int(hf_config.image_token_id),
            video_token_id=int(hf_config.video_token_id),
            bos_token_id=text_config.bos_token_id,
            eos_token_id=text_config.eos_token_id,
            pad_token_id=text_config.pad_token_id,
            vision_output_size=int(hf_config.out_hidden_size),
            projector_hidden_size=int(hf_config.projector_hidden_size),
            projector_hidden_activation=str(hf_config.projector_hidden_act),
        )

    @classmethod
    def megatron_to_hf_config(cls, model_config: MuseGlimmerModelConfig) -> dict[str, Any]:
        """Translate the serializable model config back to HF's nested layout."""
        if not isinstance(model_config, MuseGlimmerModelConfig):
            raise TypeError(f"Expected MuseGlimmerModelConfig, got {type(model_config).__name__}.")
        transformer = model_config.transformer
        vision = model_config.vision
        window_pattern = transformer.window_attn_skip_freq
        if not isinstance(window_pattern, list):
            raise ValueError("Muse Glimmer export requires an explicit per-layer window_attn_skip_freq list.")
        no_rope_pattern = transformer.no_rope_freq
        if not isinstance(no_rope_pattern, list):
            raise ValueError("Muse Glimmer export requires an explicit per-layer no_rope_freq list.")
        if transformer.window_size is None or transformer.kv_channels is None:
            raise ValueError("Muse Glimmer export requires explicit window_size and kv_channels.")

        dtype = str(transformer.params_dtype).removeprefix("torch.")
        layer_types = ["sliding_attention" if enabled else "full_attention" for enabled in window_pattern]
        layer_rope_theta = [0 if disabled else float(model_config.rotary_base) for disabled in no_rope_pattern]
        text_config = {
            "model_type": "muse_glimmer_text",
            "vocab_size": model_config.vocab_size,
            "hidden_size": transformer.hidden_size,
            "intermediate_size": transformer.ffn_hidden_size,
            "num_hidden_layers": transformer.num_layers,
            "num_attention_heads": transformer.num_attention_heads,
            "num_key_value_heads": transformer.num_query_groups,
            "head_dim": transformer.kv_channels,
            "hidden_activation": cls.megatron_to_hf_activation(transformer.activation_func),
            "max_position_embeddings": model_config.seq_length,
            "rms_norm_eps": transformer.layernorm_epsilon,
            "post_norm_eps": transformer.post_norm_epsilon,
            "attention_bias": transformer.add_qkv_bias,
            "attention_dropout": transformer.attention_dropout,
            "initializer_range": transformer.init_method_std,
            "sliding_window": transformer.window_size[0] + 1,
            "layer_types": layer_types,
            "layer_rope_theta": layer_rope_theta,
            "rope_parameters": {"rope_theta": model_config.rotary_base, "rope_type": "default"},
            "qk_scale_factor": transformer.softmax_scale * math.sqrt(transformer.kv_channels),
            "output_multiplier": transformer.output_multiplier,
            "final_logit_softcapping": transformer.final_logit_softcapping,
            "tie_word_embeddings": model_config.share_embeddings_and_output_weights,
            "bos_token_id": model_config.bos_token_id,
            "eos_token_id": model_config.eos_token_id,
            "pad_token_id": model_config.pad_token_id,
        }
        vision_config = {
            "model_type": "muse_glimmer_vision",
            "hidden_size": vision.hidden_size,
            "intermediate_size": vision.intermediate_size,
            "num_hidden_layers": vision.num_hidden_layers,
            "num_attention_heads": vision.num_attention_heads,
            "patch_size": vision.patch_size,
            "patch_temporal": vision.patch_temporal,
            "merge_size": vision.merge_size,
            "pos_emb_height": vision.pos_emb_height,
            "pos_emb_width": vision.pos_emb_width,
            "max_position_embeddings": vision.max_position_embeddings,
            "layer_norm_eps": vision.layer_norm_epsilon,
            "hidden_act": vision.hidden_activation,
            "layer_types": list(vision.layer_types),
            "rope_parameters": {"rope_theta": vision.rotary_base, "rope_type": "default"},
        }
        return {
            "architectures": ["MuseGlimmerForConditionalGeneration"],
            "model_type": "muse_glimmer",
            "dtype": dtype,
            "text_config": text_config,
            "vision_config": vision_config,
            "image_token_id": model_config.image_token_id,
            "video_token_id": model_config.video_token_id,
            "out_hidden_size": model_config.vision_output_size,
            "projector_hidden_size": model_config.projector_hidden_size,
            "projector_hidden_act": model_config.projector_hidden_activation,
            "tie_word_embeddings": model_config.share_embeddings_and_output_weights,
        }

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Return complete text, vision, adapter, and projection mappings."""
        text_prefix = "model.language_model"
        mappings: list[MegatronParamMapping[Any]] = [
            AutoMapping(
                megatron_param="embedding.word_embeddings.weight",
                hf_param=f"{text_prefix}.embed_tokens.weight",
            ),
            AutoMapping(
                megatron_param="output_layer.weight",
                hf_param="lm_head.weight",
            ),
            RMSNorm2ZeroCenteredRMSNormMapping(
                megatron_param="decoder.final_norm.weight",
                hf_param=f"{text_prefix}.norm.weight",
            ),
            ReplicatedMapping(
                megatron_param="decoder.layers.*.input_layernorm.weight",
                hf_param=f"{text_prefix}.layers.*.input_layernorm.weight",
            ),
            ReplicatedMapping(
                megatron_param="decoder.layers.*.self_attention.post_layernorm.weight",
                hf_param=f"{text_prefix}.layers.*.post_attention_layernorm.weight",
            ),
            ReplicatedMapping(
                megatron_param="decoder.layers.*.pre_mlp_layernorm.weight",
                hf_param=f"{text_prefix}.layers.*.pre_feedforward_layernorm.weight",
            ),
            ReplicatedMapping(
                megatron_param="decoder.layers.*.mlp.post_layernorm.weight",
                hf_param=f"{text_prefix}.layers.*.post_feedforward_layernorm.weight",
            ),
            AutoMapping(
                megatron_param="decoder.layers.*.self_attention.linear_proj.weight",
                hf_param=f"{text_prefix}.layers.*.self_attn.o_proj.weight",
            ),
            AutoMapping(
                megatron_param="decoder.layers.*.mlp.linear_fc2.weight",
                hf_param=f"{text_prefix}.layers.*.mlp.down_proj.weight",
            ),
            MuseGlimmerQKVGMapping(
                "decoder.layers.*.self_attention.linear_qkv.weight",
                q=f"{text_prefix}.layers.*.self_attn.q_proj.weight",
                k=f"{text_prefix}.layers.*.self_attn.k_proj.weight",
                v=f"{text_prefix}.layers.*.self_attn.v_proj.weight",
                gate=f"{text_prefix}.layers.*.self_attn.gate_proj.weight",
            ),
            GatedMLPMapping(
                megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                gate=f"{text_prefix}.layers.*.mlp.gate_proj.weight",
                up=f"{text_prefix}.layers.*.mlp.up_proj.weight",
            ),
            ReplicatedMapping(megatron_param="vision_tower.**", hf_param="model.vision_tower.**"),
            ReplicatedMapping(megatron_param="vision_adapter.**", hf_param="model.vision_adapter.**"),
            ReplicatedMapping(megatron_param="vision_projection.**", hf_param="model.vision_projection.**"),
        ]
        return MegatronMappingRegistry(*mappings)


__all__ = ["MuseGlimmerBridge", "MuseGlimmerQKVGMapping"]
