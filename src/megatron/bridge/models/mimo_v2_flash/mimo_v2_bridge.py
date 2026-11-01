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

"""MiMo-V2.6 text checkpoint conversion using the existing MiMo-V2-Flash layers."""

import json
from collections.abc import Iterable, Mapping
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import torch
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import nn

from megatron.bridge.models.conversion import quantization_utils
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import (
    HFSourcedWeightTuple,
    HFWeightTuple,
    MegatronModel,
    MegatronModelBridge,
    WeightConversionTask,
)
from megatron.bridge.models.conversion.param_mapping import AutoMapping
from megatron.bridge.models.conversion.peft_bridge import AdapterWeight
from megatron.bridge.models.conversion.utils import unwrap_model
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.hf_pretrained.state import SafeTensorsStateSource
from megatron.bridge.models.mimo_v2_flash.mimo_v2_flash_bridge import MiMoV2FlashBridge, MiMoV2FlashQKVMapping
from megatron.bridge.models.mimo_v2_flash.mimo_v2_flash_provider import MiMoV2FlashModelProvider


@lru_cache(maxsize=8)
def _checkpoint_tp_size(index_path: Path) -> int:
    if not index_path.exists():
        return 1
    tp_size = json.loads(index_path.read_text()).get("metadata", {}).get("tp_size", 1)
    if not isinstance(tp_size, int) or isinstance(tp_size, bool) or tp_size < 1:
        raise ValueError(f"Invalid MiMo checkpoint tp_size: {tp_size!r}")
    return tp_size


def _reorder_qkv(
    weight: torch.Tensor, sizes: tuple[int, int, int], source_chunks: int, target_chunks: int
) -> torch.Tensor:
    """Repartition [Q0,K0,V0,...] without changing rows within a projection."""
    if weight.shape[0] != sum(sizes) or any(s % c for s in sizes for c in (source_chunks, target_chunks)):
        raise ValueError("MiMo fused QKV dimensions must be divisible by checkpoint and attention group counts")
    trailing = weight.shape[1:]
    parts = weight.reshape(source_chunks, -1, *trailing).split([s // source_chunks for s in sizes], dim=1)
    return torch.cat(
        [part.reshape(target_chunks, size // target_chunks, *trailing) for part, size in zip(parts, sizes)],
        dim=1,
    ).reshape(weight.shape)


class MiMoV2FusedQKVMapping(AutoMapping):
    """Convert checkpoint TP chunks to Megatron's per-KV-head QKV groups."""

    def __init__(self, megatron_param: str, hf_param: str, checkpoint_tp_size: int = 1) -> None:
        super().__init__(megatron_param, hf_param)
        self.checkpoint_tp_size = checkpoint_tp_size

    @staticmethod
    def _sizes(config: TransformerConfig) -> tuple[int, int, int]:
        return (
            config.num_attention_heads * config.kv_channels,
            config.num_query_groups * config.kv_channels,
            config.num_query_groups * config.v_head_dim,
        )

    def hf_to_megatron(self, hf_weights: torch.Tensor, megatron_module: nn.Module) -> torch.Tensor:
        """Reorder before the ordinary column-parallel scatter."""
        if self.tp_rank == 0:
            config = self._get_config(megatron_module)
            hf_weights = _reorder_qkv(
                hf_weights, self._sizes(config), self.checkpoint_tp_size, config.num_query_groups
            )
        return super().hf_to_megatron(hf_weights, megatron_module)

    def megatron_to_hf(
        self, megatron_weights: torch.Tensor | None, megatron_module: nn.Module | None
    ) -> dict[str, torch.Tensor]:
        """Gather and restore the source checkpoint's fused row order."""
        config = self._get_config(megatron_module) if megatron_module is not None else None
        shape = (self._sizes(config), config.num_query_groups) if config is not None else None
        shape = self.broadcast_obj_from_pp_rank(shape, "mimo_qkv_shape")
        result = super().megatron_to_hf(megatron_weights, megatron_module)
        return {
            name: _reorder_qkv(weight, shape[0], shape[1], self.checkpoint_tp_size) for name, weight in result.items()
        }

    def resolve(self, captures: tuple[str, ...]) -> "MiMoV2FusedQKVMapping":
        """Preserve checkpoint partitioning when resolving layer wildcards."""
        megatron_param, hf_param = self._resolve_names(captures)
        return type(self)(megatron_param, hf_param, self.checkpoint_tp_size)


@MegatronModelBridge.register_bridge(
    source="MiMoV2ForCausalLM",
    target=GPTModel,
    provider=MiMoV2FlashModelProvider,
    model_type="mimo_v2",
)
class MiMoV2Bridge(MiMoV2FlashBridge):
    """Text-backbone conversion for MiMo-V2.6-Flash-RL (309B total / 15B active).

    Attention and MoE execution reuse MiMo-V2-Flash. Source-format export
    preserves the quantized representation and untouched multimodal/MTP weights.
    """

    # The generic GPT builder cannot represent this family's custom attention.
    MODEL_CONFIG_CLASS = None

    _HF_PASSTHROUGH_PREFIXES = ("visual.", "audio_encoder.", "speech_embeddings.", "model.mtp.")

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> MiMoV2FlashModelProvider:
        """Reuse the text provider and retain the full source configuration."""
        provider = super().provider_bridge(hf_pretrained)
        if hf_pretrained.config.attention_projection_layout != "fused_qkv":
            raise ValueError("MiMoV2Bridge requires the fused_qkv checkpoint layout")
        provider.mimo_v2_hf_config = deepcopy(hf_pretrained.config.to_dict())
        provider.mtp_num_layers = 0
        # The reference router casts logits to FP32, regardless of metadata dtype.
        provider.moe_router_dtype = "fp32"
        provider.moe_router_bias_update_rate = 0.0
        return provider

    @classmethod
    def megatron_to_hf_config(cls, provider: MiMoV2FlashModelProvider) -> dict[str, object]:
        """Preserve V2 architecture, quantization, and untrained encoder metadata."""
        if provider.mimo_v2_hf_config is None:
            raise ValueError("MiMo-V2 export requires a provider created from the source HF configuration")
        return deepcopy(provider.mimo_v2_hf_config)

    def _source_tp_size(self) -> int:
        state = getattr(getattr(self, "hf_pretrained", None), "state", None)
        source = getattr(state, "source", None)
        if isinstance(source, SafeTensorsStateSource):
            return _checkpoint_tp_size(source.path / "model.safetensors.index.json")
        return 1  # In-memory HF models use canonical [Q,K,V].

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Reuse all text mappings except the newly fused QKV representation."""
        mappings = []
        for mapping in super().mapping_registry().mappings:
            if mapping.megatron_param.startswith("mtp."):
                continue
            if isinstance(mapping, MiMoV2FlashQKVMapping):
                mapping = MiMoV2FusedQKVMapping(
                    mapping.megatron_param,
                    mapping.hf_param["q"].replace(".q_proj.", ".qkv_proj."),
                    self._source_tp_size(),
                )
            mappings.append(mapping)
        return MegatronMappingRegistry(*mappings)

    @staticmethod
    def get_hf_import_param_names(
        hf_param: str | dict[str, str], available_hf_param_names: set[str] | None = None
    ) -> tuple[str, ...]:
        """Include MXFP4 scale sidecars in incremental import dependencies."""
        names = list(MiMoV2FlashBridge.get_hf_import_param_names(hf_param, available_hf_param_names))
        mapped = (hf_param,) if isinstance(hf_param, str) else hf_param.values()
        if available_hf_param_names is not None:
            names.extend(f"{name}_scale" for name in mapped if f"{name}_scale" in available_hf_param_names)
        return tuple(dict.fromkeys(names))

    def _fp8_chunks(
        self, name: str, weight: torch.Tensor, scale: torch.Tensor
    ) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
        chunks = self._source_tp_size() if name.endswith(".qkv_proj.weight") else 1
        if weight.shape[0] % chunks:
            raise ValueError(f"{name}: FP8 rows are not divisible by checkpoint tp_size={chunks}")
        block = quantization_utils.FP8_BLOCK_SIZE
        expected = (
            chunks * ((weight.shape[0] // chunks + block - 1) // block),
            (weight.shape[1] + block - 1) // block,
        )
        if tuple(scale.shape) != expected:
            raise ValueError(f"{name}: expected FP8 scales {expected}, got {tuple(scale.shape)}")
        return zip(weight.chunk(chunks, dim=0), scale.chunk(chunks, dim=0))

    def maybe_modify_loaded_hf_weight(
        self, hf_param: str | dict[str, str], hf_state_dict: Mapping[str, torch.Tensor]
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Decode MXFP4 experts and separately block-scaled FP8 QKV chunks."""
        if isinstance(hf_param, dict):
            return {role: self.maybe_modify_loaded_hf_weight(name, hf_state_dict) for role, name in hf_param.items()}
        weight = hf_state_dict[hf_param]
        if weight.dtype == torch.uint8:
            return quantization_utils.dequantize_mxfp4_e2m1_packed(weight, hf_state_dict[f"{hf_param}_scale"])
        if weight.dtype == torch.float8_e4m3fn:
            scale = hf_state_dict[f"{hf_param}_scale_inv"]
            return torch.cat(
                [
                    quantization_utils.dequantize_fp8_blockwise(w, s)
                    for w, s in self._fp8_chunks(hf_param, weight, scale)
                ]
            )
        return weight

    def maybe_modify_converted_hf_weight(
        self,
        task: WeightConversionTask,
        converted_weights_dict: dict[str, torch.Tensor],
        hf_state_dict: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Restore source quantization after conversion and any LoRA merge."""
        result = {}
        for name, weight in converted_weights_dict.items():
            mxfp4_scale, fp8_scale = f"{name}_scale", f"{name}_scale_inv"
            if task.weight_dtype is not None and (mxfp4_scale in hf_state_dict or fp8_scale in hf_state_dict):
                raise ValueError(
                    "MiMo source-format export preserves quantization; weight_dtype overrides are unsupported"
                )
            if mxfp4_scale in hf_state_dict:
                packed, scale = quantization_utils.quantize_mxfp4_e2m1_like_scale(
                    weight, hf_state_dict[mxfp4_scale], name=name
                )
                result[name], result[mxfp4_scale] = packed.view(torch.uint8), scale
            elif fp8_scale in hf_state_dict:
                chunks = [
                    quantization_utils.quantize_fp8_e4m3fn_like_scale(w, s, name=name)
                    for w, s in self._fp8_chunks(name, weight, hf_state_dict[fp8_scale])
                ]
                result[name] = torch.cat([w for w, _ in chunks])
                result[fp8_scale] = torch.cat([s for _, s in chunks])
            else:
                result[name] = weight
        return result

    def _qkv_adapter_rows(
        self, megatron_model: list[MegatronModel], name: str, weight: torch.Tensor, target_chunks: int
    ) -> torch.Tensor:
        layer = int(name.split(".")[2])
        config = unwrap_model(megatron_model)[0].config
        groups = (
            config.swa_num_query_groups
            if config.hybrid_attention_pattern[layer]
            else config.full_attn_num_query_groups
        )
        sizes = (
            config.num_attention_heads * config.kv_channels,
            groups * config.kv_channels,
            groups * config.v_head_dim,
        )
        return _reorder_qkv(weight, sizes, groups, target_chunks)

    def _get_fused_adapter_linear_out_slices(
        self,
        megatron_model: list[MegatronModel],
        base_hf_weight_names: list[str],
        linear_out_tensor: torch.Tensor,
        is_expert: bool = False,
    ) -> dict[str, torch.Tensor] | None:
        """Export adapter B in canonical HF Q/K/V order, independent of base storage."""
        if len(base_hf_weight_names) == 1 and base_hf_weight_names[0].endswith(".self_attn.qkv_proj.weight"):
            name = base_hf_weight_names[0]
            return {name: self._qkv_adapter_rows(megatron_model, name, linear_out_tensor, 1)}
        return super()._get_fused_adapter_linear_out_slices(
            megatron_model, base_hf_weight_names, linear_out_tensor, is_expert=is_expert
        )

    def _merge_lora_adapter_weights(
        self,
        megatron_model: list[MegatronModel],
        converted_weights_dict: dict[str, torch.Tensor],
        adapter_weights: list[AdapterWeight],
    ) -> dict[str, torch.Tensor]:
        """Merge fused QKV adapters in the source checkpoint's row order."""
        names = list(converted_weights_dict)
        if len(names) != 1 or not names[0].endswith(".self_attn.qkv_proj.weight"):
            return super()._merge_lora_adapter_weights(megatron_model, converted_weights_dict, adapter_weights)
        (adapter,) = adapter_weights
        name = names[0]
        linear_out = self._qkv_adapter_rows(
            megatron_model, name, adapter.linear_out_weight.weight, self._source_tp_size()
        )
        return {
            name: self._merge_single_adapter_weight(
                converted_weights_dict[name], adapter.alpha, adapter.dim, adapter.linear_in_weight.weight, linear_out
            )
        }

    def stream_weights_megatron_to_hf(
        self,
        megatron_model: MegatronModel | list[MegatronModel],
        hf_pretrained: PreTrainedCausalLM,
        cpu: bool = True,
        show_progress: bool = True,
        conversion_tasks: list[WeightConversionTask] | None = None,
        *,
        merge_adapter_weights: bool = True,
        weight_dtype: torch.dtype | None = None,
        with_megatron_names: bool = False,
    ) -> Iterable[HFWeightTuple | HFSourcedWeightTuple]:
        """Export text updates while preserving untrained encoder and MTP tensors."""
        if weight_dtype is not None and getattr(hf_pretrained.config, "quantization_config", None):
            raise ValueError(
                "MiMo source-format export preserves quantization; weight_dtype overrides are unsupported"
            )
        yield from super().stream_weights_megatron_to_hf(
            megatron_model,
            hf_pretrained,
            cpu=cpu,
            show_progress=show_progress,
            conversion_tasks=conversion_tasks,
            merge_adapter_weights=merge_adapter_weights,
            weight_dtype=weight_dtype,
            with_megatron_names=with_megatron_names,
        )
        state = hf_pretrained.state
        for name in state.source.get_all_keys():
            if name.startswith(self._HF_PASSTHROUGH_PREFIXES):
                yield from HFWeightTuple(name, state[name]).iter_finalized(
                    cpu=cpu, megatron_param_names=() if with_megatron_names else None
                )
