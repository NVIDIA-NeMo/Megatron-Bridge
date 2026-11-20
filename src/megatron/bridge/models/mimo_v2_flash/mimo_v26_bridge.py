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

"""Megatron Bridge for MiMo-V2.6-Flash (``MiMoV2ForCausalLM``).

V2.6 has the same language model as MiMo-V2-Flash (including its MTP layers) and reuses its bridge; only the
checkpoint storage differs:

- ``self_attn.qkv_proj`` is one fused FP8 tensor stored as ``num_key_value_heads`` slices of [Q_s | K_s | V_s],
  with 128x128 block scales restarting at each slice. On import it is dequantized per slice and split into Q, K and
  V. MTP layers have the same layout with sliding-window attention sizes.
- Routed experts are MXFP4 (E8M0 scale per 32 values) and dequantize exactly to BF16.
- Other FP8 weights use the MiMo-V2-Flash block dequantization.
- Vision, audio and speech weights are not mapped.

Export restores the source format: weights that are MXFP4 or FP8 in the source checkpoint are requantized with the
source scale layout, ``qkv_proj`` is rebuilt slice by slice, vision/audio/speech tensors are copied from the source,
and the source ``quantization_config`` is kept. Requantizing is lossy where the source scales cannot be reproduced.
"""

import json
import re
from pathlib import Path
from typing import Iterable, Mapping

import torch
from megatron.core.models.gpt.gpt_model import GPTModel

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import (
    HFSourcedWeightTuple,
    HFWeightTuple,
    MegatronModel,
    MegatronModelBridge,
    WeightConversionTask,
)
from megatron.bridge.models.conversion.quantization_utils import (
    FP8_BLOCK_SIZE,
    dequantize_fp8_blockwise,
    dequantize_mxfp4_e2m1_packed,
    quantize_fp8_e4m3fn_like_scale,
    quantize_mxfp4_e2m1_like_scale,
)
from megatron.bridge.models.mimo_v2_flash.mimo_v2_flash_bridge import MiMoV2FlashBridge, MiMoV2FlashQKVMapping
from megatron.bridge.models.mimo_v2_flash.mimo_v2_flash_provider import MiMoV2FlashModelProvider


_LAYER_RE = re.compile(r"model\.layers\.(\d+)\.")
_QKV_SUFFIXES = (".q_proj.weight", ".k_proj.weight", ".v_proj.weight")


def _split_fused_qkv(weight: torch.Tensor, scale: torch.Tensor, config, *, swa: bool):
    """Dequantize a fused, slice-tiled FP8 ``qkv_proj`` and return BF16 ``(q, k, v)`` with heads in order."""
    heads = config.swa_num_attention_heads if swa else config.num_attention_heads
    kv_heads = config.swa_num_key_value_heads if swa else config.num_key_value_heads
    head_dim = config.swa_head_dim if swa else config.head_dim
    v_head_dim = config.swa_v_head_dim if swa else config.v_head_dim
    slices = config.num_key_value_heads  # slice count of the stored layout, for every layer
    rows_per_slice = weight.shape[0] // slices
    scales_per_slice = -(-rows_per_slice // FP8_BLOCK_SIZE)
    q_rows, k_rows = heads // slices * head_dim, kv_heads // slices * head_dim
    if not (
        rows_per_slice * slices == weight.shape[0]
        and q_rows + k_rows + kv_heads // slices * v_head_dim == rows_per_slice
        and scale.shape[0] == slices * scales_per_slice
    ):
        raise ValueError(f"unexpected fused qkv_proj layout {tuple(weight.shape)} / {tuple(scale.shape)}")
    qs, ks, vs = [], [], []
    for i in range(slices):
        rows = slice(i * rows_per_slice, (i + 1) * rows_per_slice)
        dequant = dequantize_fp8_blockwise(weight[rows], scale[i * scales_per_slice : (i + 1) * scales_per_slice])
        qs.append(dequant[:q_rows])
        ks.append(dequant[q_rows : q_rows + k_rows])
        vs.append(dequant[q_rows + k_rows :])
    return torch.cat(qs), torch.cat(ks), torch.cat(vs)


def _fuse_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, slices: int) -> torch.Tensor:
    """Inverse of the split: tile heads-in-order Q, K and V into ``slices`` slices of [Q_s | K_s | V_s]."""
    return torch.cat([t for qkv in zip(q.chunk(slices), k.chunk(slices), v.chunk(slices)) for t in qkv])


class _FusedQKVMapping(MiMoV2FlashQKVMapping):
    """The V2-Flash Q/K/V mapping; its ``q/k/v_proj`` names do not exist in V2.6 and map to ``qkv_proj``."""

    def __init__(self, megatron_param, q, k, v):
        super().__init__(megatron_param, q=q, k=k, v=v)
        self.allow_hf_name_mismatch = True


@MegatronModelBridge.register_bridge(
    source="MiMoV2ForCausalLM",
    target=GPTModel,
    provider=MiMoV2FlashModelProvider,
    model_type="mimo_v2",
)
class MiMoV26Bridge(MiMoV2FlashBridge):
    """Megatron Bridge for MiMo-V2.6-Flash from its MXFP4 checkpoint."""

    # Source tensors without a Megatron counterpart, copied unchanged on export.
    _HF_PASSTHROUGH_PREFIXES = ("visual.", "audio_encoder.", "speech_embeddings.")

    @classmethod
    def megatron_to_hf_config(cls, provider: MiMoV2FlashModelProvider) -> dict:
        """Convert the provider to a V2.6 HuggingFace config (fused QKV projection)."""
        hf_cfg = super().megatron_to_hf_config(provider)
        hf_cfg["model_type"] = "mimo_v2"
        hf_cfg["auto_map"] = {
            "AutoConfig": "configuration_mimo_v2.MiMoV2Config",
            "AutoModel": "modeling_mimo_v2.MiMoV2Model",
            "AutoModelForCausalLM": "modeling_mimo_v2.MiMoV2ForCausalLM",
        }
        hf_cfg["attention_projection_layout"] = "fused_qkv"
        return hf_cfg

    def mapping_registry(self) -> MegatronMappingRegistry:
        mappings = []
        for mapping in super().mapping_registry().mappings:
            if type(mapping) is MiMoV2FlashQKVMapping:
                mapping = _FusedQKVMapping(mapping.megatron_param, **mapping.hf_param)
            mappings.append(mapping)
        return MegatronMappingRegistry(*mappings)

    def _checkpoint_keys(self) -> set[str]:
        if getattr(self, "_hf_keys", None) is None:
            self._hf_keys = set(self.hf_pretrained.state.source.get_all_keys())
        return self._hf_keys

    def _is_swa(self, hf_name: str) -> bool:
        """MTP layers use sliding-window attention; decoder layers follow ``hybrid_layer_pattern``."""
        if hf_name.startswith("model.mtp."):
            return True
        return self.hf_config.hybrid_layer_pattern[int(_LAYER_RE.match(hf_name)[1])] == 1

    def maybe_modify_loaded_hf_weight(
        self, hf_param: str | dict[str, str], hf_state_dict: Mapping[str, torch.Tensor]
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if (
            isinstance(hf_param, dict)
            and set(hf_param) == {"q", "k", "v"}
            and hf_param["q"] not in self._checkpoint_keys()
        ):
            fused = hf_param["q"].replace(".q_proj.", ".qkv_proj.")
            q, k, v = _split_fused_qkv(
                hf_state_dict[fused], hf_state_dict[fused + "_scale_inv"], self.hf_config, swa=self._is_swa(fused)
            )
            return {"q": q, "k": k, "v": v}
        return super().maybe_modify_loaded_hf_weight(hf_param, hf_state_dict)

    def _load_and_dequant(self, key: str, hf_state_dict: Mapping[str, torch.Tensor]) -> torch.Tensor:
        weight = hf_state_dict[key]
        if weight.dtype == torch.uint8:  # routed expert, packed E2M1 with an E8M0 scale per 32 values
            return dequantize_mxfp4_e2m1_packed(weight, hf_state_dict[key + "_scale"])
        return super()._load_and_dequant(key, hf_state_dict)

    def maybe_modify_converted_hf_weight(
        self,
        task: WeightConversionTask,
        converted_weights_dict: dict[str, torch.Tensor],
        hf_state_dict: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Fuse Q, K and V back into ``qkv_proj`` and requantize weights that are quantized in the source."""
        converted_weights_dict = super().maybe_modify_converted_hf_weight(task, converted_weights_dict, hf_state_dict)
        slices = self.hf_config.num_key_value_heads
        if isinstance(task.mapping, _FusedQKVMapping) and converted_weights_dict:
            names = [next(n for n in converted_weights_dict if n.endswith(suffix)) for suffix in _QKV_SUFFIXES]
            fused = names[0].replace(".q_proj.", ".qkv_proj.")
            converted_weights_dict = {fused: _fuse_qkv(*(converted_weights_dict[n] for n in names), slices)}

        result = {}
        for name, weight in converted_weights_dict.items():
            mxfp4_scale, fp8_scale = name + "_scale", name + "_scale_inv"
            quantized = mxfp4_scale in hf_state_dict or fp8_scale in hf_state_dict
            if quantized and task.weight_dtype is not None:
                raise ValueError(f"{name} is exported in its source quantized format; weight_dtype is not supported")
            if mxfp4_scale in hf_state_dict:
                packed, scale = quantize_mxfp4_e2m1_like_scale(weight, hf_state_dict[mxfp4_scale], name=name)
                result[name], result[mxfp4_scale] = packed.view(torch.uint8), scale
            elif fp8_scale in hf_state_dict:
                # The block scales of the fused qkv_proj restart at each slice; other FP8 weights are one slice.
                n = slices if name.endswith(".qkv_proj.weight") else 1
                pairs = [
                    quantize_fp8_e4m3fn_like_scale(w, s, name=name)
                    for w, s in zip(weight.chunk(n), hf_state_dict[fp8_scale].chunk(n))
                ]
                result[name] = torch.cat([w for w, _ in pairs])
                result[fp8_scale] = torch.cat([s for _, s in pairs])
            else:
                result[name] = weight
        return result

    def stream_weights_megatron_to_hf(
        self,
        megatron_model: MegatronModel | list[MegatronModel],
        hf_pretrained,
        cpu: bool = True,
        show_progress: bool = True,
        conversion_tasks: list[WeightConversionTask] | None = None,
        merge_adapter_weights: bool = True,
        weight_dtype: torch.dtype | None = None,
        with_megatron_names: bool = False,
        current_pp_stage_only: bool = False,
    ) -> Iterable[HFWeightTuple | HFSourcedWeightTuple]:
        """Export the language model and copy the unmapped vision, audio and speech tensors from the source."""
        yield from super().stream_weights_megatron_to_hf(
            megatron_model,
            hf_pretrained,
            cpu=cpu,
            show_progress=show_progress,
            conversion_tasks=conversion_tasks,
            merge_adapter_weights=merge_adapter_weights,
            weight_dtype=weight_dtype,
            with_megatron_names=with_megatron_names,
            current_pp_stage_only=current_pp_stage_only,
        )
        source = getattr(getattr(hf_pretrained, "state", None), "source", None)
        if source is None or not self._should_emit_hf_passthrough(
            megatron_model, current_pp_stage_only=current_pp_stage_only
        ):
            return
        for name in source.get_all_keys():
            if name.startswith(self._HF_PASSTHROUGH_PREFIXES):
                yield from HFWeightTuple(name, hf_pretrained.state[name]).iter_finalized(
                    cpu=cpu, megatron_param_names=() if with_megatron_names else None
                )

    def postprocess_hf_export_artifacts(self, path: Path) -> None:
        """Restore the source ``quantization_config``, which the saved config drops but the weights still follow."""
        from transformers.utils import cached_file

        source_file = cached_file(self.hf_config._name_or_path, "config.json")
        quantization_config = json.loads(Path(source_file).read_text()).get("quantization_config")
        if quantization_config is None:
            return
        config_file = Path(path) / "config.json"
        config = json.loads(config_file.read_text())
        config["quantization_config"] = quantization_config
        config_file.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
