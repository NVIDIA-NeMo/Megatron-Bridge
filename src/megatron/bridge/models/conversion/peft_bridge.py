# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Union

import torch
from megatron.core import parallel_state
from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import get_pg_rank, unwrap_model

from megatron.bridge.models.conversion.param_mapping import (
    ColumnParallelMapping,
    ReplicatedMapping,
    RowParallelMapping,
    split_qkv_weights,
)
from megatron.bridge.models.conversion.utils import (
    extract_sort_key,
    get_module_and_param_from_name,
    persistent_buffers,
)
from megatron.bridge.peft.canonical_lora import ModuleDict
from megatron.bridge.peft.lora import LoRAMerge
from megatron.bridge.peft.utils import get_adapter_attributes_from_linear


if TYPE_CHECKING:
    from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
    from megatron.bridge.models.conversion.model_bridge import HFWeightTuple, MegatronWeightTuple, WeightConversionTask


MegatronModel = MegatronModule


ADAPTER_NAME_MAP = {
    # Map HF base parameter suffixes (keys) to CanonicalLoRA adapter keys (values)
    ".q_proj.weight": "adapter_q",
    ".k_proj.weight": "adapter_k",
    ".v_proj.weight": "adapter_v",
    ".gate_proj.weight": "adapter_gate",
    ".up_proj.weight": "adapter_up",
}
ADAPTER_KEY_TO_SUFFIX = {value: key for key, value in ADAPTER_NAME_MAP.items()}

# Map Megatron adapter suffixes to HuggingFace LoRA parameter suffixes
MEGATRON_TO_HF_LORA_SUFFIX = {
    ".linear_in.weight": ".lora_A.weight",
    ".linear_out.weight": ".lora_B.weight",
}


@dataclass(frozen=True)
class AdapterWeightConversionTask:
    """Task describing an adapter's LoRA weights for conversion or merging."""

    global_base_prefix: str
    adapter_key: Optional[str]
    alpha: int
    dim: int
    linear_in_task: "WeightConversionTask"
    linear_out_task: "WeightConversionTask"


@dataclass(frozen=True)
class AdapterWeight:
    """Materialized adapter weights ready for merge."""

    global_base_prefix: str
    adapter_key: Optional[str]
    alpha: int
    dim: int
    linear_in_weight: "MegatronWeightTuple"
    linear_out_weight: "MegatronWeightTuple"


def _select_hf_base_param_name(base_mapping, adapter_key: Optional[str], expected_suffix: str) -> Optional[str]:
    """Return the HF base parameter name associated with this adapter."""

    hf_param = base_mapping.hf_param
    if isinstance(hf_param, str):
        return hf_param if hf_param.endswith(expected_suffix) else None

    if isinstance(hf_param, dict):
        if adapter_key:
            target_suffix = ADAPTER_KEY_TO_SUFFIX.get(adapter_key)
            if target_suffix:
                for value in hf_param.values():
                    if value.endswith(target_suffix):
                        return value

        if len(hf_param) == 1:
            value = next(iter(hf_param.values()))
            return value if value.endswith(expected_suffix) else None

    return None


class MegatronPeftBridge:
    """Mixin providing adapter-aware utilities for Megatron model bridges."""

    def _get_lora_unwrapped_name(self, megatron_param: str) -> str:
        """Remove `.to_wrap` from LoRA parameter names."""
        return megatron_param.replace(".to_wrap.", ".")

    def _is_adapter_param_name(self, param_name: str) -> bool:
        """Return True if the parameter only belongs to a PEFT adapter."""
        return ".adapter." in param_name

    def _get_adapter_wrap_module(
        self,
        local_base_prefix: str,
        megatron_model: Union[MegatronModel, List[MegatronModel]],
        vp_stage: int,
    ) -> tuple[Optional[torch.nn.Module], Optional[torch.nn.Module]]:
        """Locate the adapter wrapper and its underlying module."""

        lora_module, _ = get_module_and_param_from_name(megatron_model, local_base_prefix, vp_stage)
        adapter = getattr(lora_module, "adapter", None)
        if adapter is None:
            lora_module, _ = get_module_and_param_from_name(megatron_model, local_base_prefix + ".to_wrap", vp_stage)
        return getattr(lora_module, "adapter", None), getattr(lora_module, "to_wrap", None)

    def _resolve_hf_adapter_param_name(
        self,
        mapping_registry: "MegatronMappingRegistry",
        global_base_prefix: str,
        megatron_suffix: str,
        adapter_key: Optional[str],
    ) -> Optional[str]:
        """
        Resolve the HuggingFace adapter parameter name by translating the base Megatron name.

        Note:
            LoRA adapters never register bias tensors for `linear_in` / `linear_out`, so callers
            only pass weight suffixes here. The bias fallback below is solely for robustness in
            case a future adapter type introduces biased projections.
        """

        hf_suffix = MEGATRON_TO_HF_LORA_SUFFIX.get(megatron_suffix)
        assert hf_suffix is not None, (
            f"Unsupported adapter suffix '{megatron_suffix}'. Update MEGATRON_TO_HF_LORA_SUFFIX."
        )

        base_suffix = ".weight"
        base_mapping = mapping_registry.megatron_to_hf_lookup(f"{global_base_prefix}{base_suffix}")
        assert base_mapping is not None, (
            f"Expected mapping for adapter base '{global_base_prefix}{base_suffix}' but none found"
        )

        hf_base_name = _select_hf_base_param_name(base_mapping, adapter_key, base_suffix)
        if hf_base_name is None or not hf_base_name.endswith(base_suffix):
            return None

        return hf_base_name[: -len(base_suffix)] + hf_suffix

    def _megatron_global_adapters_info_all_pp_ranks(
        self, megatron_model: Union[MegatronModel, List[MegatronModel]]
    ) -> List[tuple[str, str, bool, bool, int, int, int, int]]:
        """Collect adapter metadata across PP ranks."""

        if hasattr(self, "_cached_param_objects_adapter"):
            return self._cached_param_objects_adapter

        if not isinstance(megatron_model, list):
            megatron_model = [megatron_model]

        from megatron.bridge.models.conversion.model_bridge import _megatron_local_name_to_global

        pp_group = parallel_state.get_pipeline_model_parallel_group()
        pp_rank = get_pg_rank(pp_group)
        model_config = unwrap_model(megatron_model)[0].config
        global_param_objects: List[tuple[str, str, bool, bool, int, int, int, int]] = []

        for vp_stage, model in enumerate(megatron_model):
            for local_param_name, _ in itertools.chain(model.named_parameters(), persistent_buffers(model)):  # type: ignore[name-defined]
                if "_extra_state" in local_param_name:
                    continue
                local_param_name = self._unwrap_name(local_param_name)
                global_param_name = _megatron_local_name_to_global(
                    megatron_model, model_config, local_param_name, vp_stage
                )
                if not self._is_adapter_param_name(global_param_name) or not global_param_name.endswith(
                    ".linear_in.weight"
                ):
                    continue

                local_base_prefix = local_param_name.partition(".adapter.")[0]
                global_base_name = global_param_name[: -len(".linear_in.weight")]
                adapter, to_wrap = self._get_adapter_wrap_module(local_base_prefix, megatron_model, vp_stage)
                if isinstance(adapter, ModuleDict):
                    adapter_name = local_param_name.removeprefix(local_base_prefix + ".adapter.").split(".")[0]
                    adapter = adapter[adapter_name]
                input_is_parallel, _, _, _, base_linear_is_parallel = get_adapter_attributes_from_linear(to_wrap)
                global_param_objects.append(
                    (
                        global_base_name,
                        local_base_prefix,
                        input_is_parallel,
                        base_linear_is_parallel,
                        adapter.alpha,
                        adapter.dim,
                        pp_rank,
                        vp_stage,
                    )
                )

        gathered_global_param_objects = [None] * pp_group.size()
        torch.distributed.all_gather_object(gathered_global_param_objects, global_param_objects, group=pp_group)
        flattened_names = list(set(sum(gathered_global_param_objects, [])))
        gathered_global_param_objects = sorted(flattened_names, key=lambda x: extract_sort_key(x[0]))
        self._cached_param_objects_adapter = gathered_global_param_objects
        return gathered_global_param_objects

    def _construct_adapters_names(self, prefix: str, adapter_key: Optional[str]) -> tuple[str, str]:
        """Build linear_in/linear_out parameter names for an adapter."""

        linear_in_name, linear_out_name = prefix + ".adapter", prefix + ".adapter"
        if adapter_key is not None:
            linear_in_name += f".{adapter_key}"
            linear_out_name += f".{adapter_key}"
        linear_in_name += ".linear_in.weight"
        linear_out_name += ".linear_out.weight"
        return linear_in_name, linear_out_name

    def build_adapter_conversion_tasks(
        self, megatron_model: Union[MegatronModel, List[MegatronModel]]
    ) -> Dict[str, List[AdapterWeightConversionTask]]:
        """Construct adapter merge tasks keyed by their base parameter."""

        if not isinstance(megatron_model, list):
            megatron_model = [megatron_model]

        adapters_info = self._megatron_global_adapters_info_all_pp_ranks(megatron_model)
        tasks_by_base: Dict[str, List[AdapterWeightConversionTask]] = defaultdict(list)  # type: ignore[name-defined]

        from megatron.bridge.models.conversion.model_bridge import WeightConversionTask

        # `MegatronModelBridge` mixes in this class and provides `mapping_registry`.
        assert hasattr(self, "mapping_registry"), "MegatronModelBridge must define mapping_registry"
        mapping_registry = self.mapping_registry()  # type: ignore[attr-defined]

        for (
            global_base_name,
            local_base_prefix,
            input_is_parallel,
            base_linear_is_parallel,
            alpha,
            dim,
            pp_rank,
            vp_stage,
        ) in adapters_info:
            global_base_prefix, _, adapter_suffix = global_base_name.partition(".adapter")
            adapter_key = None
            if adapter_suffix:
                key_token = adapter_suffix.split(".")[-1]
                if key_token.startswith("adapter_"):
                    adapter_key = key_token

            global_linear_in_name, global_linear_out_name = self._construct_adapters_names(
                global_base_prefix, adapter_key
            )
            local_linear_in_name, local_linear_out_name = global_linear_in_name, global_linear_out_name

            hf_linear_in_name = self._resolve_hf_adapter_param_name(
                mapping_registry, global_base_prefix, ".linear_in.weight", adapter_key
            )
            hf_linear_out_name = self._resolve_hf_adapter_param_name(
                mapping_registry, global_base_prefix, ".linear_out.weight", adapter_key
            )

            linear_in_module = linear_in_weight = None
            linear_out_module = linear_out_weight = None
            if parallel_state.get_pipeline_model_parallel_rank() == pp_rank:
                adapter, _ = self._get_adapter_wrap_module(local_base_prefix, megatron_model, vp_stage)
                if isinstance(adapter, ModuleDict):
                    adapter = adapter[adapter_key]
                linear_in_module, linear_in_weight = adapter.linear_in, adapter.linear_in.weight
                linear_out_module, linear_out_weight = adapter.linear_out, adapter.linear_out.weight
                local_linear_in_name, local_linear_out_name = self._construct_adapters_names(
                    local_base_prefix, adapter_key
                )

            if base_linear_is_parallel:
                linear_in_mapping_cls = RowParallelMapping if input_is_parallel else ColumnParallelMapping
                linear_out_mapping_cls = ColumnParallelMapping
            else:
                linear_in_mapping_cls = ReplicatedMapping
                linear_out_mapping_cls = ReplicatedMapping

            linear_in_task = WeightConversionTask(
                param_name=local_linear_in_name,
                global_param_name=global_linear_in_name,
                mapping=linear_in_mapping_cls(
                    megatron_param=local_linear_in_name,
                    hf_param=hf_linear_in_name or global_linear_in_name,
                ),
                pp_rank=pp_rank,
                vp_stage=vp_stage,
                megatron_module=linear_in_module,
                param_weight=linear_in_weight,
            )

            linear_out_task = WeightConversionTask(
                param_name=local_linear_out_name,
                global_param_name=global_linear_out_name,
                mapping=linear_out_mapping_cls(
                    megatron_param=local_linear_out_name,
                    hf_param=hf_linear_out_name or global_linear_out_name,
                ),
                pp_rank=pp_rank,
                vp_stage=vp_stage,
                megatron_module=linear_out_module,
                param_weight=linear_out_weight,
            )

            tasks_by_base[global_base_prefix].append(
                AdapterWeightConversionTask(
                    global_base_prefix=global_base_prefix,
                    adapter_key=adapter_key,
                    alpha=alpha,
                    dim=dim,
                    linear_in_task=linear_in_task,
                    linear_out_task=linear_out_task,
                )
            )

        return tasks_by_base

    def materialize_adapter_weights(self, adapter_tasks: List[AdapterWeightConversionTask]) -> List[AdapterWeight]:
        """Run adapter merge tasks to gather full adapter weights."""

        from megatron.bridge.models.conversion.model_bridge import MegatronWeightTuple

        materialized: List[AdapterWeight] = []
        for adapter_task in adapter_tasks:
            linear_in_dict = adapter_task.linear_in_task.mapping.megatron_to_hf(
                adapter_task.linear_in_task.param_weight, adapter_task.linear_in_task.megatron_module
            )
            linear_in_tensor = next(iter(linear_in_dict.values()))

            linear_out_dict = adapter_task.linear_out_task.mapping.megatron_to_hf(
                adapter_task.linear_out_task.param_weight, adapter_task.linear_out_task.megatron_module
            )
            linear_out_tensor = next(iter(linear_out_dict.values()))

            materialized.append(
                AdapterWeight(
                    global_base_prefix=adapter_task.global_base_prefix,
                    adapter_key=adapter_task.adapter_key,
                    alpha=adapter_task.alpha,
                    dim=adapter_task.dim,
                    linear_in_weight=MegatronWeightTuple(
                        adapter_task.linear_in_task.param_name,
                        linear_in_tensor,
                        adapter_task.linear_in_task.vp_stage,
                    ),
                    linear_out_weight=MegatronWeightTuple(
                        adapter_task.linear_out_task.param_name,
                        linear_out_tensor,
                        adapter_task.linear_out_task.vp_stage,
                    ),
                )
            )

        return materialized

    def stream_adapter_weights_megatron_to_hf(
        self,
        megatron_model: Union[MegatronModel, List[MegatronModel]],
        cpu: bool = True,
        show_progress: bool = True,
    ) -> Iterable[HFWeightTuple]:
        """Stream only adapter weights without merging them into base tensors."""

        if not isinstance(megatron_model, list):
            megatron_model = [megatron_model]

        adapter_tasks_by_base = self.build_adapter_conversion_tasks(megatron_model)
        adapter_tasks = list(itertools.chain.from_iterable(adapter_tasks_by_base.values()))
        if not adapter_tasks:
            return

        for adapter_task in self._with_progress_tracking(adapter_tasks, "Streaming adapter weights", show_progress):
            adapter_weight = self.materialize_adapter_weights([adapter_task])[0]

            linear_in_tensor = adapter_weight.linear_in_weight.weight
            linear_out_tensor = adapter_weight.linear_out_weight.weight

            if cpu:
                linear_in_tensor = linear_in_tensor.cpu()
                linear_out_tensor = linear_out_tensor.cpu()

            linear_in_hf_name = adapter_task.linear_in_task.mapping.hf_param
            linear_out_hf_name = adapter_task.linear_out_task.mapping.hf_param

            assert isinstance(linear_in_hf_name, str) and isinstance(linear_out_hf_name, str), (
                "Adapter mappings must resolve to string HF parameter names"
            )

            yield HFWeightTuple(linear_in_hf_name, linear_in_tensor)
            yield HFWeightTuple(linear_out_hf_name, linear_out_tensor)

    def _merge_lora_adapter_weights(
        self,
        megatron_model: List[MegatronModel],
        converted_weights_dict: Dict[str, torch.Tensor],
        adapter_weights: List[AdapterWeight],
    ) -> Dict[str, torch.Tensor]:
        """Merge LoRA adapter weights back into the base tensor for HF export."""

        if len(adapter_weights) > 1 and all(
            w.adapter_key in ADAPTER_NAME_MAP.values() for w in adapter_weights if w.adapter_key
        ):
            return self._merge_canonical_adapter_from_weights(converted_weights_dict, adapter_weights)

        assert len(adapter_weights) == 1, "Expected a single adapter weight for standard LoRA merging"

        adapter_weight = adapter_weights[0]
        alpha, dim = adapter_weight.alpha, adapter_weight.dim
        linear_in_weight = adapter_weight.linear_in_weight.weight
        linear_out_weight = adapter_weight.linear_out_weight.weight

        base_weight_shape = next(iter(converted_weights_dict.values())).shape
        weight_names = converted_weights_dict.keys()
        is_fused_fc1 = (
            len(weight_names) % 2 == 0
            and all("gate_proj" in name or "up_proj" in name for name in weight_names)
            and linear_out_weight.shape[0] == 2 * base_weight_shape[0]
        )
        is_fused_qkv = len(weight_names) == 3 and all(
            "q_proj" in name or "k_proj" in name or "v_proj" in name for name in weight_names
        )

        if is_fused_qkv:
            q_out, k_out, v_out = split_qkv_weights(megatron_model[0].config, linear_out_weight)
            qkv_linear_out_weights = {"q_proj": q_out, "k_proj": k_out, "v_proj": v_out}
        else:
            qkv_linear_out_weights = None

        for hf_name, base_weight in list(converted_weights_dict.items()):
            current_linear_out_weight = linear_out_weight
            if is_fused_fc1:
                split_size = linear_out_weight.shape[0] // 2
                if "gate_proj" in hf_name:
                    current_linear_out_weight = linear_out_weight[:split_size, :]
                elif "up_proj" in hf_name:
                    current_linear_out_weight = linear_out_weight[split_size:, :]
                else:
                    raise ValueError(f"Unknown weight name: {hf_name}")
            elif is_fused_qkv and qkv_linear_out_weights is not None:
                if "q_proj" in hf_name:
                    current_linear_out_weight = qkv_linear_out_weights["q_proj"]
                elif "k_proj" in hf_name:
                    current_linear_out_weight = qkv_linear_out_weights["k_proj"]
                elif "v_proj" in hf_name:
                    current_linear_out_weight = qkv_linear_out_weights["v_proj"]
                else:
                    raise ValueError(f"Unknown weight name: {hf_name}")

            merged_weight = self._merge_single_adapter_weight(
                base_weight, alpha, dim, linear_in_weight, current_linear_out_weight
            )
            converted_weights_dict[hf_name] = merged_weight

        return converted_weights_dict

    def _merge_single_adapter_weight(
        self,
        base_weight: torch.Tensor,
        alpha: int,
        dim: int,
        linear_in_weight: torch.Tensor,
        linear_out_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Merge a single adapter's weights with base weight."""

        merger = LoRAMerge()
        base_device = base_weight.device
        return merger.merge(
            base_weight,
            linear_out_weight.to(base_device),
            linear_in_weight.to(base_device),
            alpha,
            dim,
        )

    def _merge_canonical_adapter_from_weights(
        self,
        converted_weights_dict: Dict[str, torch.Tensor],
        adapter_weights: List[AdapterWeight],
    ) -> Dict[str, torch.Tensor]:
        """Merge CanonicalLoRA adapters using pre-materialized adapter weights."""

        adapter_lookup = {aw.adapter_key: aw for aw in adapter_weights}

        for hf_name, base_weight in converted_weights_dict.items():
            target_adapter = None
            for suffix, adapter_key in ADAPTER_NAME_MAP.items():
                if hf_name.endswith(suffix):
                    target_adapter = adapter_lookup.get(adapter_key)
                    break

            if target_adapter is None:
                raise ValueError(f"Adapter name mapping not found for {hf_name}")

            merged_weight = self._merge_single_adapter_weight(
                base_weight,
                target_adapter.alpha,
                target_adapter.dim,
                target_adapter.linear_in_weight.weight,
                target_adapter.linear_out_weight.weight,
            )
            converted_weights_dict[hf_name] = merged_weight

        return converted_weights_dict
