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

"""Attention scale companions for the shared ModelOpt weight exporter."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch

from megatron.bridge.models.conversion.model_bridge import HFWeightTuple, WeightConversionTask
from megatron.bridge.models.conversion.quant_mapping import derive_kv_bmm_amax_map
from megatron.bridge.models.conversion.utils import get_module_and_param_from_name


def _kv_spec(attention: torch.nn.Module) -> tuple[str, bool] | None:
    quantizers = [getattr(attention, f"{name}_bmm_quantizer", None) for name in ("q", "k", "v")]
    # ModelOpt applies these transforms even when quantization is disabled.
    # Scalar scale companions cannot describe or compensate for either one.
    for name, quantizer in zip(("Q", "K", "V"), quantizers, strict=True):
        if quantizer is None:
            continue
        if getattr(quantizer, "rotate_is_enabled", False):
            raise NotImplementedError(f"ModelOpt {name} export does not support rotation")
        if getattr(quantizer, "pre_quant_scale", None) is not None:
            raise NotImplementedError(f"ModelOpt {name} export does not support pre-quant scaling")
    enabled = [quantizer is not None and quantizer.is_enabled for quantizer in quantizers]
    if not any(enabled):
        return None

    # Keep weight-only export usable with ModelOpt versions predating the KV API.
    from modelopt.torch.export.quantized_weight_export import get_kv_cache_export_spec

    spec = get_kv_cache_export_spec(attention)
    if spec not in ("FP8", "NVFP4"):
        raise NotImplementedError(f"ModelOpt export requires symmetric FP8 or NVFP4 KV caches, got {spec!r}")
    for name, quantizer, is_enabled in zip(("Q", "K", "V"), quantizers, enabled, strict=True):
        if quantizer is None or not is_enabled:
            continue
        if getattr(quantizer, "backend", None) is not None:
            raise NotImplementedError(f"ModelOpt {name} export does not support custom quantization backends")
        fp8 = name == "Q" or spec == "FP8"
        block_sizes = None if fp8 else {-1: 16, "type": "dynamic", "scale_bits": (4, 3)}
        if (
            quantizer.num_bits != ((4, 3) if fp8 else (2, 1))
            or quantizer.axis is not None
            or quantizer.block_sizes != block_sizes
            or quantizer._dynamic
            or getattr(quantizer, "bias", None) is not None
            or getattr(quantizer, "bias_value", None) is not None
        ):
            layout = "per-tensor E4M3" if fp8 else "block-16 E2M1 with dynamic E4M3 block scales"
            raise NotImplementedError(f"ModelOpt {name} export requires static symmetric {layout} quantization")
    return spec, enabled[0]


@dataclass(frozen=True)
class _KVExportHook:
    task: WeightConversionTask
    attention: torch.nn.Module | None
    hf_parent: str
    spec: tuple[str, bool] | None
    values: torch.Tensor | None = None

    def validate(self) -> None:
        if self.attention is not None and _kv_spec(self.attention) != self.spec:
            raise ValueError(f"ModelOpt KV configuration changed for {self.task.global_param_name}")

    def __call__(self, hf_name: str, tensor: torch.Tensor) -> Iterable[HFWeightTuple]:
        yield HFWeightTuple(hf_name, tensor)
        if self.spec is None:
            return
        yield from self.scales()

    def capture(self) -> torch.Tensor | None:
        self.validate()
        if self.spec is None or self.attention is None:
            return None
        from modelopt.torch.export.quantized_weight_export import capture_kv_cache_export_state

        state = capture_kv_cache_export_state(self.attention, clamp_fp8_scales=False, cpu=False)
        if state is None:
            raise ValueError("KV quantizers were disabled after export planning")
        q_scale = torch.ones_like(state.k_scale)
        if self.spec[1] and not self.attention.q_bmm_quantizer._use_constant_amax:
            from modelopt.torch.export.quant_utils import get_scaling_factor

            q_scale = get_scaling_factor(self.attention.q_bmm_quantizer)
            if q_scale is None:
                raise ValueError("Missing calibrated query scale")
        scalars = (q_scale, state.k_scale, state.v_scale)
        if any(value.numel() != 1 for value in scalars):
            raise ValueError("ModelOpt attention export requires scalar Q/K/V scales")
        return torch.stack(
            [
                value.detach().to(device=self.task.param_weight.device, dtype=torch.float32).reshape(())
                for value in scalars
            ]
        )

    def scales(self) -> Iterable[HFWeightTuple]:
        if self.spec is None:
            return
        # Imported here to avoid a cycle: the shared exporter constructs these hooks.
        from megatron.bridge.models.conversion.modelopt_utils import _stage_tensor_for_collective

        values = self.values
        mapping = self.task.mapping
        if values is not None:
            values = _stage_tensor_for_collective(values, mapping.pp_group)
        values = mapping.broadcast_from_pp_rank(values, cache_key=f"modelopt:kv:{self.task.global_param_name}")
        if values is None:
            raise RuntimeError(f"Missing ModelOpt attention scales for {self.task.global_param_name}")
        values = _stage_tensor_for_collective(values, mapping.tp_group)
        shards = mapping.gather_from_tp_ranks(values)
        if any(not torch.equal(shard[1:], shards[0][1:]) for shard in shards[1:]):
            raise ValueError(f"Inconsistent ModelOpt K/V scales across TP for {self.task.global_param_name}")
        # Q calibration is local to each TP rank's heads; deployment needs one range.
        q_scale = shards[0][0] if len(shards) == 1 else torch.stack([shard[0] for shard in shards]).amax()
        for name, value in zip(("q", "k", "v"), (q_scale, values[1], values[2]), strict=True):
            yield HFWeightTuple(f"{self.hf_parent}.{name}_scale", value.clone())


def _local_kv_hooks(tasks: list[WeightConversionTask], model: list[torch.nn.Module]) -> dict[str, _KVExportHook]:
    hooks: dict[str, _KVExportHook] = {}
    mapped_attention_ids = set()
    quantized_attention = {
        id(module): name
        for chunk in model
        for name, module in chunk.named_modules()
        if any(
            quantizer is not None and quantizer.is_enabled
            for axis in ("q", "k", "v")
            for quantizer in (getattr(module, f"{axis}_bmm_quantizer", None),)
        )
    }
    attention_with_quantizers = {
        id(module)
        for chunk in model
        for module in chunk.modules()
        if any(getattr(module, f"{axis}_bmm_quantizer", None) is not None for axis in ("q", "k", "v"))
    }
    if not attention_with_quantizers:
        return hooks
    for task in tasks:
        mappings = derive_kv_bmm_amax_map([task.mapping])
        if not mappings or task.megatron_module is None:
            continue
        local_parent = task.param_name.removesuffix(".linear_qkv.weight") + ".core_attention"
        _, attention = get_module_and_param_from_name(model, local_parent, task.vp_stage)
        if id(attention) not in attention_with_quantizers:
            continue
        mapped_attention_ids.add(id(attention))
        spec = _kv_spec(attention)
        hf_parent = mappings[0].hf_param.removesuffix(".k_bmm_quantizer._amax")
        hooks[task.global_param_name] = _KVExportHook(task, attention, hf_parent, spec)

    for module_id, name in quantized_attention.items():
        if module_id not in mapped_attention_ids:
            raise NotImplementedError(f"No supported QKV conversion owns the attention quantizers at {name}")
    return hooks


def _attach_kv_hooks(
    tasks: list[WeightConversionTask], local_hooks: dict[str, _KVExportHook], group: Any
) -> dict[str, _KVExportHook]:
    from megatron.bridge.models.conversion.modelopt_utils import _all_gather_objects

    metadata = {name: (hook.hf_parent, hook.spec) for name, hook in local_hooks.items()}
    synchronized: dict[str, tuple[str, tuple[str, bool] | None]] = {}
    for rank_metadata in _all_gather_objects(metadata, group):
        for name, spec in rank_metadata.items():
            if name in synchronized and synchronized[name] != spec:
                raise ValueError(f"Inconsistent ModelOpt attention metadata for {name}")
            synchronized[name] = spec
    hooks = {}
    for task in tasks:
        if task.global_param_name in synchronized:
            if task.megatron_module is not None and task.global_param_name not in local_hooks:
                raise ValueError(f"Inconsistent ModelOpt KV quantization for {task.global_param_name}")
            parent, spec = synchronized[task.global_param_name]
            attention = (
                local_hooks[task.global_param_name].attention if task.global_param_name in local_hooks else None
            )
            hooks[task.global_param_name] = _KVExportHook(task, attention, parent, spec)
    return hooks
