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

"""HF <-> Megatron conversion for DiffusionGemma checkpoints."""

import copy
from dataclasses import fields
from types import SimpleNamespace
from typing import Dict, Optional

import torch
from torch import nn

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import ReplicatedMapping
from megatron.bridge.models.diffusion_gemma.diffusion_gemma_provider import DiffusionGemmaModelProvider
from megatron.bridge.models.diffusion_gemma.modeling_diffusion_gemma import DiffusionGemmaModel
from megatron.bridge.models.gemma_vl.gemma4_vl_bridge import Gemma4VLBridge
from megatron.bridge.models.gemma_vl.gemma4_vl_provider import Gemma4VLModelProvider


def _normalized_diffusion_config(hf_config):
    """Expose implicit DiffusionGemma text semantics through Gemma4 bridge fields."""
    if getattr(hf_config, "model_type", None) != "diffusion_gemma":
        raise ValueError(f"Expected model_type='diffusion_gemma', got {getattr(hf_config, 'model_type', None)!r}")
    config = copy.deepcopy(hf_config)
    text_config = config.text_config
    required = ("num_experts", "top_k_experts", "moe_intermediate_size")
    missing = [name for name in required if getattr(text_config, name, None) is None]
    if getattr(text_config, "num_global_key_value_heads", None) is None:
        per_layer_config = getattr(text_config, "per_layer_config", None)
        layer_types = getattr(text_config, "layer_types", ())
        full_layer = next(
            (
                per_layer_config[index]
                for index, layer_type in enumerate(layer_types)
                if layer_type == "full_attention" and per_layer_config is not None
            ),
            None,
        )
        if full_layer is None:
            missing.append("num_global_key_value_heads")
        else:
            text_config.num_global_key_value_heads = full_layer.num_key_value_heads
    if missing:
        raise ValueError(f"DiffusionGemma text config is missing required fields: {missing}")
    if getattr(text_config, "hidden_size_per_layer_input", 0):
        raise ValueError("DiffusionGemma per-layer embeddings are not supported by this bridge")
    if getattr(config, "audio_config", None) is not None:
        raise ValueError("DiffusionGemma audio inputs are not supported by this bridge")
    text_config.enable_moe_block = True
    text_config.attention_k_eq_v = True
    return config


class _TiedReplicatedMapping(ReplicatedMapping):
    """One Megatron tensor backed by equal decoder and encoder HF tensors."""

    def __init__(self, megatron_param: str, decoder: str, encoder: str):
        super().__init__(megatron_param, {"decoder": decoder, "encoder": encoder})

    def hf_to_megatron(self, hf_weights: Dict[str, torch.Tensor], megatron_module: nn.Module) -> torch.Tensor:
        decoder = hf_weights["decoder"]
        encoder = hf_weights["encoder"]
        if decoder.shape != encoder.shape or decoder.dtype != encoder.dtype or not torch.equal(decoder, encoder):
            raise ValueError(
                f"DiffusionGemma tied tensors differ: {self.hf_param['decoder']} != {self.hf_param['encoder']}"
            )
        return super().hf_to_megatron(decoder, megatron_module)

    def megatron_to_hf(
        self,
        megatron_weights: Optional[torch.Tensor],
        megatron_module: Optional[nn.Module],
    ) -> Dict[str, torch.Tensor]:
        megatron_weights = self.broadcast_from_pp_rank(megatron_weights, cache_key=str(self.hf_param))
        if megatron_weights is None:
            return {}
        megatron_weights = self.maybe_dequantize(megatron_weights)
        return {self.hf_param["decoder"]: megatron_weights, self.hf_param["encoder"]: megatron_weights}

    def resolve(self, captures):
        megatron_param, hf_param = self._resolve_names(captures)
        return type(self)(megatron_param, hf_param["decoder"], hf_param["encoder"])


@MegatronModelBridge.register_bridge(
    source="DiffusionGemmaForBlockDiffusion",
    target=DiffusionGemmaModel,
    provider=DiffusionGemmaModelProvider,
    model_type="diffusion_gemma",
)
class DiffusionGemmaBridge(Gemma4VLBridge):
    """Convert the tied DiffusionGemma checkpoint into one Megatron text stack."""

    def provider_bridge(self, hf_pretrained) -> DiffusionGemmaModelProvider:
        hf_config = _normalized_diffusion_config(hf_pretrained.config)
        self.hf_config = hf_config
        gemma_provider = super().provider_bridge(SimpleNamespace(config=hf_config))
        if not isinstance(gemma_provider, Gemma4VLModelProvider):
            raise TypeError(f"Expected a Gemma4 VL MoE provider, got {type(gemma_provider).__name__}")

        provider = DiffusionGemmaModelProvider()
        for field in fields(Gemma4VLModelProvider):
            setattr(provider, field.name, getattr(gemma_provider, field.name))
        provider.canvas_length = int(getattr(hf_config, "canvas_length", 256))
        # DiffusionGemma's encoder/decoder attention removes Gemma 4's shared
        # KV-cache path. Keeping Gemma4ModelProvider's default (18) changes
        # real 30-layer checkpoints even though tiny models may not expose it.
        provider.num_kv_shared_layers = 0
        # Keep Megatron's FP32 router projection for stable top-k choices on the
        # real 128-expert BF16 checkpoint. HF BF16 itself differs materially
        # from its FP32 reference, so this is the safer distributed recipe.
        provider.moe_router_dtype = "fp32"
        # The reference diffusion SFT objective has no load-balancing penalty.
        # Keep opt-in support, but do not inherit Gemma4's 0.001 silently.
        provider.moe_aux_loss_coeff = 0.0
        provider.audio_config = None
        provider.video_token_id = None
        provider.audio_token_id = None
        return provider

    def _text_config(self):
        hf_config = getattr(self, "hf_config", None)
        if hf_config is None:
            return None
        if getattr(hf_config, "model_type", None) == "diffusion_gemma":
            return _normalized_diffusion_config(hf_config).text_config
        return hf_config.text_config

    def _conversion_mode(self) -> str:
        return "vl"

    def _hf_layer_prefix(self) -> str:
        return "model.decoder."

    def mapping_registry(self) -> MegatronMappingRegistry:
        if self._is_dense_config():
            raise ValueError("DiffusionGemma bridge supports only the 26B-A4B MoE architecture")
        registry = self._moe_mapping_registry(megatron_prefix="language_model.")
        mappings = [
            mapping
            for mapping in registry.mappings
            if mapping.megatron_param != "language_model.decoder.layers.*.layer_scalar"
        ]
        mappings.extend(
            [
                ReplicatedMapping("vision_tower.**", "model.encoder.vision_tower.**"),
                ReplicatedMapping("embed_vision.**", "model.encoder.embed_vision.**"),
                ReplicatedMapping("self_conditioning.**", "model.decoder.self_conditioning.**"),
                _TiedReplicatedMapping(
                    "language_model.decoder.layers.*.layer_scalar",
                    decoder="model.decoder.layers.*.layer_scalar",
                    encoder="model.encoder.language_model.layers.*.layer_scalar",
                ),
            ]
        )
        return MegatronMappingRegistry(*mappings)

    @classmethod
    def megatron_to_hf_config(cls, provider: DiffusionGemmaModelProvider) -> dict:
        config = super().megatron_to_hf_config(provider)
        text_config = dict(config["text_config"])
        text_config.pop("architectures", None)
        text_config.pop("enable_moe_block", None)
        text_config.pop("attention_k_eq_v", None)
        text_config["model_type"] = "diffusion_gemma_text"
        config.pop("audio_config", None)
        config.pop("audio_token_id", None)
        config.pop("video_token_id", None)
        config.update(
            {
                "architectures": ["DiffusionGemmaForBlockDiffusion"],
                "canvas_length": provider.canvas_length,
                "model_type": "diffusion_gemma",
                "text_config": text_config,
            }
        )
        return config
