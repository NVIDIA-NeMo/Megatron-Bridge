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

"""Canonical decoder weights for the shared native DiffusionGemma text stack.

Vision weights stay in the source checkpoint and are not converted. Encoder
layer-scalar buffers remain independent; all other text tensors share the
canonical decoder representation, including the tied output embedding.
"""

from copy import copy
from dataclasses import fields
from typing import Any, Mapping

import torch

from megatron.bridge.diffusion.models.diffusion_gemma.modeling_diffusion_gemma import DiffusionGemmaModel
from megatron.bridge.diffusion.models.diffusion_gemma.provider import DiffusionGemmaModelProvider
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    FusedExpertMapping,
    FusedGatedExpertMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.gemma.gemma4_bridge import Gemma4Bridge
from megatron.bridge.models.gemma.gemma4_provider import Gemma4ModelProvider
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


class _PreTrainedDiffusionGemma(PreTrainedCausalLM):
    """Keep lazy safetensors access while loading the actual block-diffusion HF class."""

    def _load_model(self):
        from transformers import DiffusionGemmaForBlockDiffusion

        if self.model_name_or_path is None:
            raise ValueError("model_name_or_path must be provided to load model")
        model_kwargs = {"trust_remote_code": self.trust_remote_code, **self.init_kwargs, "config": self.config}
        if self.torch_dtype is not None:
            model_kwargs["torch_dtype"] = self.torch_dtype
        return DiffusionGemmaForBlockDiffusion.from_pretrained(self.model_name_or_path, **model_kwargs).to(self.device)


@MegatronModelBridge.register_bridge(
    source="DiffusionGemmaForBlockDiffusion",
    target=DiffusionGemmaModel,
    provider=DiffusionGemmaModelProvider,
    model_type="diffusion_gemma",
)
class DiffusionGemmaBridge(Gemma4Bridge):
    """Import BF16 text weights into one native shared encoder/decoder stack.

    Reuses Gemma4 MoE layouts and global K=V synthesis. Quantized and clipped
    text variants are rejected because this bridge does not dequantize them.
    The supported construction path is the native DiffusionGemma provider.
    """

    MODEL_CONFIG_CLASS = None

    def _text_config(self) -> Any | None:
        hf_config = getattr(self, "hf_config", None)
        if hf_config is None:
            return None
        text_config = copy(getattr(hf_config, "text_config", hf_config))
        # DiffusionGemma fixes these semantics intrinsically; its published
        # text config does not serialize the Gemma4 dispatch flags.
        text_config.enable_moe_block = True
        text_config.attention_k_eq_v = True
        return text_config

    def _hf_layer_prefix(self) -> str:
        return "model.decoder."

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> DiffusionGemmaModelProvider:
        """Derive native BF16 text architecture without allocating HF weights."""
        self.hf_config = hf_pretrained.config
        text_config = getattr(self.hf_config, "text_config", self.hf_config)
        for config in (self.hf_config, text_config):
            if getattr(config, "quantization_config", None):
                raise ValueError("DiffusionGemma supports BF16 checkpoints, not quantized checkpoints")
            dtype = getattr(config, "dtype", None) or getattr(config, "torch_dtype", None)
            if dtype not in (None, "bfloat16", torch.bfloat16):
                raise ValueError(f"DiffusionGemma requires BF16 checkpoint metadata, found {dtype}")
        if getattr(text_config, "use_clipped_linears", False):
            raise ValueError("DiffusionGemma clipped text linears are unsupported")
        if not getattr(self.hf_config, "tie_word_embeddings", True):
            raise ValueError("DiffusionGemma requires tied text embeddings/output weights")
        if getattr(text_config, "enable_moe_block", True) is False:
            raise ValueError("DiffusionGemma requires the MoE text architecture")

        gemma_provider = self._build_moe_provider(self._text_config())
        provider_kwargs = {
            field.name: getattr(gemma_provider, field.name) for field in fields(Gemma4ModelProvider) if field.init
        }
        provider_kwargs.update(
            attention_k_eq_v=True,
            share_embeddings_and_output_weights=True,
            moe_aux_loss_coeff=0.0,
            moe_router_load_balancing_type="none",
            scatter_embedding_sequence_parallel=False,
            apply_rope_fusion=False,
        )
        return DiffusionGemmaModelProvider(**provider_kwargs)

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Map shared decoder tensors, independent encoder scalars, and conditioning."""
        mappings = list(self._moe_mapping_registry().mappings)
        # SequentialMLP retains the expert index in its native path. The HF
        # wildcard consumes only the layer index; the fused mapping selects
        # the expert axis from local_experts.<index> and merges it on export.
        mappings.extend(
            [
                FusedGatedExpertMapping(
                    megatron_param="decoder.layers.*.mlp.experts.local_experts.*.linear_fc1.weight",
                    hf_param="model.decoder.layers.*.experts.gate_up_proj",
                ),
                FusedExpertMapping(
                    megatron_param="decoder.layers.*.mlp.experts.local_experts.*.linear_fc2.weight",
                    hf_param="model.decoder.layers.*.experts.down_proj",
                ),
            ]
        )
        mappings.append(
            ReplicatedMapping(
                megatron_param="decoder.layers.*.encoder_layer_scalar",
                hf_param="model.encoder.language_model.layers.*.layer_scalar",
            )
        )
        for name in ("pre_norm", "gate_proj", "up_proj", "down_proj"):
            mappings.append(
                ReplicatedMapping(
                    megatron_param=f"self_conditioning.{name}.weight",
                    hf_param=f"model.decoder.self_conditioning.{name}.weight",
                )
            )
        return MegatronMappingRegistry(*mappings)

    def maybe_modify_loaded_hf_weight(
        self, hf_param: str | dict[str, str], hf_state_dict: Mapping[str, torch.Tensor]
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Reject raw quantization and conflicting aliases before layout conversion."""
        names = hf_param.values() if isinstance(hf_param, dict) else (hf_param,)
        for name in names:
            clipped_name = name.removesuffix(".weight") + ".linear.weight"
            if name not in hf_state_dict and clipped_name in hf_state_dict:
                raise ValueError(f"DiffusionGemma clipped text linear is unsupported: {clipped_name}")
            if name not in hf_state_dict:
                if not self._is_synthesized_kv_projection(name):
                    raise ValueError(f"DiffusionGemma is missing a required text tensor: {name}")
                continue  # Gemma4 synthesizes only the intrinsic global V projection.
            weight = hf_state_dict[name]
            if weight.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                raise ValueError(f"DiffusionGemma does not load raw quantized tensors: {name} ({weight.dtype})")
            if name.startswith("model.decoder.") and not name.endswith(".layer_scalar"):
                alias = name.replace("model.decoder.", "model.encoder.language_model.", 1)
                if alias in hf_state_dict and not torch.equal(weight, hf_state_dict[alias]):
                    raise ValueError(f"DiffusionGemma shared text tensors must be tied: {name} != {alias}")
                if name == "model.decoder.embed_tokens.weight" and "lm_head.weight" in hf_state_dict:
                    if not torch.equal(weight, hf_state_dict["lm_head.weight"]):
                        raise ValueError("DiffusionGemma tied lm_head.weight differs from the decoder embedding")
        if isinstance(hf_param, dict) and "v" in hf_param:
            value_name = hf_param["v"]
            if self._is_synthesized_kv_projection(value_name) and value_name in hf_state_dict:
                if not torch.equal(hf_state_dict[hf_param["k"]], hf_state_dict[value_name]):
                    raise ValueError(f"DiffusionGemma global K=V projections must be tied: {value_name}")
        return super().maybe_modify_loaded_hf_weight(hf_param, hf_state_dict)
