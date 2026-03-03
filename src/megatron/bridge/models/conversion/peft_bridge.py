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
import re
from collections import defaultdict
from dataclasses import dataclass
from string import digits
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, TypeVar, Union

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
from megatron.bridge.peft.utils import ParallelLinearAdapter, get_adapter_attributes_from_linear, is_expert_linear


if TYPE_CHECKING:
    from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
    from megatron.bridge.models.conversion.model_bridge import HFWeightTuple, MegatronWeightTuple, WeightConversionTask


MegatronModel = TypeVar("MegatronModel", bound=MegatronModule)


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

        # For fused qkv/gate_up case, we just need a placeholder here
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
        megatron_adapter_suffix: str,
        base_suffix: str,
        adapter_key: Optional[str],
    ) -> Optional[str]:
        """
        Resolve the HuggingFace adapter parameter name by translating the base Megatron name.

        Note:
            LoRA adapters never register bias tensors for `linear_in` / `linear_out`, so callers
            only pass weight suffixes here. The bias fallback below is solely for robustness in
            case a future adapter type introduces biased projections.
        """

        hf_suffix = MEGATRON_TO_HF_LORA_SUFFIX.get(megatron_adapter_suffix)
        assert hf_suffix is not None, (
            f"Unsupported adapter suffix '{megatron_adapter_suffix}'. Update MEGATRON_TO_HF_LORA_SUFFIX."
        )

        base_mapping = mapping_registry.megatron_to_hf_lookup(f"{global_base_prefix}{base_suffix}")
        assert base_mapping is not None, (
            f"Expected mapping for adapter base '{global_base_prefix}{base_suffix}' but none found"
        )

        # Strip expert layers numbering
        base_suffix = base_suffix.rstrip(digits)
        hf_base_name = _select_hf_base_param_name(base_mapping, adapter_key, base_suffix)
        if hf_base_name is None or not hf_base_name.endswith(base_suffix):
            return None

        return hf_base_name[: -len(base_suffix)] + hf_suffix

    def _get_base_hf_param_names_for_adapter(
        self,
        mapping_registry: "MegatronMappingRegistry",
        global_base_prefix: str,
        adapter_key: Optional[str],
        base_suffix: str,
    ) -> List[str]:
        """Return all HF base parameter names associated with this adapter."""

        base_mapping = mapping_registry.megatron_to_hf_lookup(f"{global_base_prefix}{base_suffix}")
        if base_mapping is None:
            return []

        hf_param = base_mapping.hf_param
        if isinstance(hf_param, str):
            return [hf_param]

        values = list(hf_param.values())
        if adapter_key:
            adapter_suffix = ADAPTER_KEY_TO_SUFFIX.get(adapter_key)
            if adapter_suffix:
                filtered = [value for value in values if value.endswith(adapter_suffix)]
                if filtered:
                    return filtered
        return values

    def _make_lora_param_name(self, base_name: str, megatron_adapter_suffix: str) -> Optional[str]:
        """Translate a base HF weight name into its LoRA-specific counterpart."""

        if not base_name.endswith(".weight"):
            return None

        hf_suffix = MEGATRON_TO_HF_LORA_SUFFIX.get(megatron_adapter_suffix)
        if hf_suffix is None:
            return None

        return base_name[: -len(".weight")] + hf_suffix

    def _is_fused_qkv(self, hf_weight_names: Iterable[str]) -> bool:
        """Check whether the provided HF names correspond to a fused QKV weight."""

        names = list(hf_weight_names)
        if len(names) != 3:
            return False

        required = {"q_proj", "k_proj", "v_proj"}
        discovered = {token for name in names for token in required if token in name}
        return discovered == required

    def _is_fused_fc1_gate_up(
        self,
        base_hf_weight_names: Iterable[str],
        linear_out_tensor: torch.Tensor,
        base_weight_shape: Optional[torch.Size] = None,
    ) -> bool:
        """Detect fused FC1 (gate/up) adapters based on names and tensor shape."""

        names = list(base_hf_weight_names)
        has_gate_up = (
            bool(names)
            and len(names) % 2 == 0
            and all(("gate_proj" in name or "up_proj" in name) for name in names)
            and any("gate_proj" in name for name in names)
            and any("up_proj" in name for name in names)
        )
        if not has_gate_up:
            return False

        if linear_out_tensor.ndim != 2 or linear_out_tensor.shape[0] % 2 != 0:
            return False

        if base_weight_shape is not None and linear_out_tensor.shape[0] != 2 * base_weight_shape[0]:
            return False

        return True

    def _infer_qkv_projection_from_name(self, hf_name: str) -> Optional[str]:
        """Return q_proj/k_proj/v_proj identifier based on the HF name."""

        if "q_proj" in hf_name:
            return "q_proj"
        if "k_proj" in hf_name:
            return "k_proj"
        if "v_proj" in hf_name:
            return "v_proj"
        return None

    def _infer_hf_expert_idx(self, hf_name: str) -> Optional[int]:
        """Return the expert index embedded in an HF MoE weight name."""

        match = re.search(r"\bexperts\.(\d+)\b", hf_name)
        if match is None:
            return None
        return int(match.group(1))

    def _split_qkv_linear_out_weight(
        self,
        megatron_model: Union[MegatronModel, List[MegatronModel]],
        linear_out_weight: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Split a fused LoRA linear_out tensor for QKV adapters."""

        model = megatron_model[0] if isinstance(megatron_model, list) else megatron_model
        q_out, k_out, v_out = split_qkv_weights(model.config, linear_out_weight)
        return {"q_proj": q_out, "k_proj": k_out, "v_proj": v_out}

    def _split_fused_fc1_linear_out_weight(
        self,
        linear_out_weight: torch.Tensor,
        *,
        is_expert: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split fused FC1 LoRA linear_out into gate/up with TP-aware ordering."""

        tp_size = (
            parallel_state.get_expert_tensor_parallel_world_size()
            if is_expert
            else parallel_state.get_tensor_model_parallel_world_size()
        )
        if tp_size <= 1:
            return torch.chunk(linear_out_weight, 2, dim=0)

        shard_size = linear_out_weight.shape[0] // tp_size
        if shard_size * tp_size != linear_out_weight.shape[0] or shard_size % 2 != 0:
            return torch.chunk(linear_out_weight, 2, dim=0)

        shards = torch.split(linear_out_weight, shard_size, dim=0)
        gate_parts = []
        up_parts = []
        for shard in shards:
            gate_shard, up_shard = torch.chunk(shard, 2, dim=0)
            gate_parts.append(gate_shard)
            up_parts.append(up_shard)
        gate = torch.cat(gate_parts, dim=0)
        up = torch.cat(up_parts, dim=0)
        return gate, up

    def _gather_expert_adapter_weight(
        self,
        weight: torch.Tensor,
    ) -> Optional[List[torch.Tensor]]:
        """Gather expert-sharded adapter weights across EP ranks when needed."""
        ep_size = parallel_state.get_expert_model_parallel_world_size()
        if ep_size <= 1:
            return None
        assert weight.ndim < 3

        gathered = [torch.empty_like(weight) for _ in range(ep_size)]
        torch.distributed.all_gather(gathered, weight, group=parallel_state.get_expert_model_parallel_group())
        return gathered

    def _select_expert_adapter_weight(
        self,
        weight: torch.Tensor,
        gathered: List[torch.Tensor],
        expert_idx: int,
        num_experts: int,
    ) -> torch.Tensor:
        """Select the per-expert adapter weight slice if present."""

        assert weight.ndim < 3

        ep_size = parallel_state.get_expert_model_parallel_world_size()
        if ep_size <= 1:
            return weight

        num_experts_per_rank = num_experts // ep_size
        rank = expert_idx // num_experts_per_rank
        return gathered[rank]

    def _megatron_global_adapters_info_all_pp_ranks(
        self, megatron_model: Union[MegatronModel, List[MegatronModel]]
    ) -> List[tuple[str, str, bool, bool, int, int, int, int]]:
        """Get all adapters' information tuple:
         (global_base_name, local_base_prefix, input_is_parallel, base_linear_is_parallel, alpha, dim, pp_rank, vp_stage)
        across all pipeline parallel ranks."""
        # Cache the result after first call
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
                # only collect linear_in.weight for deduplication
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
                if isinstance(adapter, ParallelLinearAdapter):
                    input_is_parallel = adapter.input_is_parallel
                    base_linear_is_parallel = True
                else:
                    attrs = get_adapter_attributes_from_linear(to_wrap)
                    input_is_parallel = attrs.input_is_parallel
                    base_linear_is_parallel = attrs.base_linear_is_parallel
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

        # flatten the list, sort it and remove duplicates
        # the order matters here, casually re-order will cause a hang.
        flattened_names = list(set(sum(gathered_global_param_objects, [])))

        # the order cannot be changed, this sync for all ranks for conversion
        # change this might cause a hang
        gathered_global_param_objects = sorted(flattened_names, key=lambda x: extract_sort_key(x[0]))

        self._cached_param_objects_adapter = gathered_global_param_objects

        return gathered_global_param_objects

    def _construct_adapters_names(self, prefix: str, adapter_key: Optional[str]) -> tuple[str, str]:
        """Build linear_in/linear_out parameter names for an adapter.

        Args:
            prefix: Base module prefix without any adapter suffix (global or local, depending on caller).
            adapter_key: Optional adapter identifier used by CanonicalLoRA (e.g. ``adapter_q``). ``None`` for
                standard single-adapter LoRA modules.

        Returns:
            Tuple ``(linear_in_name, linear_out_name)`` containing the parameter names for the adapter's
            input and output projection weights.
        """
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
        """Construct adapter merge tasks keyed by their base parameter.

        The returned dict is keyed by the *global* LoRA-wrapped parameter name
        (e.g., ``decoder.layers.0.mlp.linear_fc1.to_wrap.weight``). Each value
        contains the adapter tasks (canonical or regular) that should be
        merged into that base weight.
        """

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
            # global_base_name example: decoder.layers.0.mlp.linear_fc1.adapter.adapter_q
            global_base_prefix, _, adapter_suffix = global_base_name.partition(".adapter")

            adapter_key = None
            if adapter_suffix:
                key_token = adapter_suffix.split(".")[-1]
                if key_token.startswith("adapter_"):
                    adapter_key = key_token

            global_linear_in_name, global_linear_out_name = self._construct_adapters_names(
                global_base_prefix, adapter_key
            )
            # In case the adapter doesn't exist locally, we use the global names
            local_linear_in_name, local_linear_out_name = global_linear_in_name, global_linear_out_name

            base_suffix = ".weight"
            if is_expert_linear(global_base_prefix) and ".local_experts." not in global_base_prefix:
                # To get expert layer hf mapping properly
                base_suffix = ".weight0"

            hf_linear_in_name = self._resolve_hf_adapter_param_name(
                mapping_registry, global_base_prefix, ".linear_in.weight", base_suffix, adapter_key
            )
            hf_linear_out_name = self._resolve_hf_adapter_param_name(
                mapping_registry, global_base_prefix, ".linear_out.weight", base_suffix, adapter_key
            )

            linear_in_module, linear_in_weight = None, None
            linear_out_module, linear_out_weight = None, None
            if parallel_state.get_pipeline_model_parallel_rank() == pp_rank:
                adapter, _ = self._get_adapter_wrap_module(local_base_prefix, megatron_model, vp_stage)
                if isinstance(adapter, ModuleDict):
                    adapter = adapter[adapter_key]
                linear_in_module, linear_in_weight = adapter.linear_in, adapter.linear_in.weight
                linear_out_module, linear_out_weight = adapter.linear_out, adapter.linear_out.weight
                local_linear_in_name, local_linear_out_name = self._construct_adapters_names(
                    local_base_prefix, adapter_key
                )

            # Pick mapping strategies based on base layer parallelism
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
                    hf_param=hf_linear_in_name,
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
                    hf_param=hf_linear_out_name,
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

        # Local import avoids circular dependency while ensuring runtime access.
        from megatron.bridge.models.conversion.model_bridge import HFWeightTuple

        if not isinstance(megatron_model, list):
            megatron_model = [megatron_model]

        num_moe_experts = megatron_model[0].config.num_moe_experts
        adapter_tasks_by_base = self.build_adapter_conversion_tasks(megatron_model)
        adapter_tasks = list(itertools.chain.from_iterable(adapter_tasks_by_base.values()))
        if not adapter_tasks:
            return

        assert hasattr(self, "mapping_registry"), "MegatronModelBridge must define mapping_registry"
        mapping_registry = self.mapping_registry()  # type: ignore[attr-defined]

        for adapter_task in self._with_progress_tracking(adapter_tasks, "Streaming adapter weights", show_progress):
            adapter_weight = self.materialize_adapter_weights([adapter_task])[0]

            linear_in_tensor = adapter_weight.linear_in_weight.weight
            linear_out_tensor = adapter_weight.linear_out_weight.weight
            is_expert = is_expert_linear(adapter_task.global_base_prefix)
            is_grouped_expert = is_expert and ".local_experts." not in adapter_task.global_base_prefix
            expert_linear_in_gathered = None
            expert_linear_out_gathered = None
            if is_grouped_expert:
                expert_linear_in_gathered = self._gather_expert_adapter_weight(
                    linear_in_tensor,
                )
                expert_linear_out_gathered = self._gather_expert_adapter_weight(
                    linear_out_tensor,
                )

            base_suffixes = [".weight"]
            if is_grouped_expert:
                base_suffixes = [f".weight{expert_num}" for expert_num in range(num_moe_experts)]

            for base_suffix in base_suffixes:
                current_linear_in_tensor = linear_in_tensor
                current_linear_out_tensor = linear_out_tensor
                if is_grouped_expert:
                    expert_idx = int(base_suffix[len(".weight") :])
                    current_linear_in_tensor = self._select_expert_adapter_weight(
                        linear_in_tensor,
                        expert_linear_in_gathered,
                        expert_idx,
                        num_moe_experts,
                    )
                    current_linear_out_tensor = self._select_expert_adapter_weight(
                        linear_out_tensor,
                        expert_linear_out_gathered,
                        expert_idx,
                        num_moe_experts,
                    )

                if cpu:
                    current_linear_in_tensor = current_linear_in_tensor.cpu()
                    current_linear_out_tensor = current_linear_out_tensor.cpu()

                base_hf_weight_names = self._get_base_hf_param_names_for_adapter(
                    mapping_registry,
                    adapter_task.global_base_prefix,
                    adapter_task.adapter_key,
                    base_suffix,
                )
                linear_in_hf_names = []
                linear_out_hf_names = []
                for base_name in base_hf_weight_names:
                    linear_in_hf_names.append(self._make_lora_param_name(base_name, ".linear_in.weight"))
                    linear_out_hf_names.append(self._make_lora_param_name(base_name, ".linear_out.weight"))
                if adapter_task.adapter_key is None:
                    # Handle fused adapters (e.g., gate/up or q/k/v) by splitting the fused tensor
                    # into per-base slices keyed by the HF weight names.
                    # Example: base_hf_weight_names = ["...gate_proj.weight", "...up_proj.weight"]
                    per_base_linear_out = self._get_fused_adapter_linear_out_slices(
                        megatron_model,
                        base_hf_weight_names,
                        current_linear_out_tensor,
                        is_expert=is_expert_linear(adapter_task.global_base_prefix),
                    )
                    if per_base_linear_out is not None:
                        for index, base_name in enumerate(base_hf_weight_names):
                            current_linear_out_tensor = per_base_linear_out.get(base_name)
                            assert current_linear_out_tensor is not None, "unknown projection name"

                            yield HFWeightTuple(linear_in_hf_names[index], current_linear_in_tensor)
                            yield HFWeightTuple(linear_out_hf_names[index], current_linear_out_tensor)
                        continue

                yield HFWeightTuple(linear_in_hf_names[0], current_linear_in_tensor)
                yield HFWeightTuple(linear_out_hf_names[0], current_linear_out_tensor)

    def _get_fused_adapter_linear_out_slices(
        self,
        megatron_model: List[MegatronModel],
        base_hf_weight_names: List[str],
        linear_out_tensor: torch.Tensor,
        is_expert: bool = False,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Return per-base-name linear_out slices for fused adapters, else None.

        This supports fused QKV adapters (split into q/k/v) and fused FC1 adapters
        (split into gate/up along dim=0). The returned dict is keyed by the HF
        base weight name (e.g. `...q_proj.weight` or `...gate_proj.weight`).
        """

        if self._is_fused_qkv(base_hf_weight_names):
            qkv_linear_out_weights = self._split_qkv_linear_out_weight(megatron_model, linear_out_tensor)
            per_base: Dict[str, torch.Tensor] = {}
            for base_name in base_hf_weight_names:
                projection_key = self._infer_qkv_projection_from_name(base_name)
                if projection_key is None:
                    continue
                per_base[base_name] = qkv_linear_out_weights[projection_key]
            return per_base

        is_fused_fc1 = self._is_fused_fc1_gate_up(base_hf_weight_names, linear_out_tensor)
        if is_fused_fc1:
            gate_weight, up_weight = self._split_fused_fc1_linear_out_weight(
                linear_out_tensor,
                is_expert=is_expert,
            )
            per_base = {}
            for base_name in base_hf_weight_names:
                if "gate_proj" in base_name:
                    per_base[base_name] = gate_weight
                elif "up_proj" in base_name:
                    per_base[base_name] = up_weight
                else:
                    raise ValueError(f"Unknown fused-fc1 base weight name: {base_name}")
            return per_base

        return None

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
            return self._merge_canonical_adapter_from_weights(megatron_model, converted_weights_dict, adapter_weights)

        assert len(adapter_weights) == 1, "Expected a single adapter weight for standard LoRA merging"

        adapter_weight = adapter_weights[0]
        alpha, dim = adapter_weight.alpha, adapter_weight.dim
        linear_in_weight = adapter_weight.linear_in_weight.weight
        linear_out_weight = adapter_weight.linear_out_weight.weight
        num_moe_experts = megatron_model[0].config.num_moe_experts
        is_expert = is_expert_linear(adapter_weight.global_base_prefix)
        is_grouped_expert = is_expert and ".local_experts." not in adapter_weight.global_base_prefix
        expert_linear_in_gathered = None
        expert_linear_out_gathered = None
        if is_grouped_expert:
            expert_linear_in_gathered = self._gather_expert_adapter_weight(linear_in_weight)
            expert_linear_out_gathered = self._gather_expert_adapter_weight(linear_out_weight)

        base_weight_shape = next(iter(converted_weights_dict.values())).shape
        weight_names = converted_weights_dict.keys()
        is_fused_fc1 = self._is_fused_fc1_gate_up(weight_names, linear_out_weight, base_weight_shape)
        is_fused_qkv = self._is_fused_qkv(weight_names) and not is_expert
        qkv_linear_out_weights = (
            self._split_qkv_linear_out_weight(megatron_model, linear_out_weight) if is_fused_qkv else None
        )
        fc1_gate_weight = fc1_up_weight = None
        if is_fused_fc1 and not is_expert:
            fc1_gate_weight, fc1_up_weight = self._split_fused_fc1_linear_out_weight(
                linear_out_weight,
                is_expert=is_expert,
            )

        for hf_name, base_weight in list(converted_weights_dict.items()):
            current_linear_in_weight = linear_in_weight
            current_linear_out_weight = linear_out_weight
            if is_grouped_expert:
                expert_idx = self._infer_hf_expert_idx(hf_name)
                current_linear_in_weight = self._select_expert_adapter_weight(
                    linear_in_weight,
                    expert_linear_in_gathered,
                    expert_idx,
                    num_moe_experts,
                )
                current_linear_out_weight = self._select_expert_adapter_weight(
                    linear_out_weight,
                    expert_linear_out_gathered,
                    expert_idx,
                    num_moe_experts,
                )
            if is_fused_fc1:
                if is_expert:
                    fc1_gate_weight, fc1_up_weight = self._split_fused_fc1_linear_out_weight(
                        current_linear_out_weight,
                        is_expert=is_expert,
                    )
                if "gate_proj" in hf_name:
                    current_linear_out_weight = fc1_gate_weight
                elif "up_proj" in hf_name:
                    current_linear_out_weight = fc1_up_weight
                else:
                    raise ValueError(f"Unknown weight name: {hf_name}")
            elif is_fused_qkv and qkv_linear_out_weights is not None:
                projection_key = self._infer_qkv_projection_from_name(hf_name)
                if projection_key is None:
                    raise ValueError(f"Unknown weight name: {hf_name}")
                current_linear_out_weight = qkv_linear_out_weights[projection_key]

            merged_weight = self._merge_single_adapter_weight(
                base_weight, alpha, dim, current_linear_in_weight, current_linear_out_weight
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
        linear_out_on_base = (
            linear_out_weight if linear_out_weight.device == base_device else linear_out_weight.to(base_device)
        )
        linear_in_on_base = (
            linear_in_weight if linear_in_weight.device == base_device else linear_in_weight.to(base_device)
        )
        return merger.merge(
            base_weight,
            linear_out_on_base,
            linear_in_on_base,
            alpha,
            dim,
        )

    def _merge_canonical_adapter_from_weights(
        self,
        megatron_model: List[MegatronModel],
        converted_weights_dict: Dict[str, torch.Tensor],
        adapter_weights: List[AdapterWeight],
    ) -> Dict[str, torch.Tensor]:
        """Merge CanonicalLoRA adapters using pre-materialized adapter weights."""

        adapter_lookup = {aw.adapter_key: aw for aw in adapter_weights}
        expert_linear_in_gathered: Dict[str, List[torch.Tensor]] = {}
        expert_linear_out_gathered: Dict[str, List[torch.Tensor]] = {}
        base_prefix = adapter_weights[0].global_base_prefix
        num_moe_experts = megatron_model[0].config.num_moe_experts
        is_expert = is_expert_linear(base_prefix)
        is_grouped_expert = is_expert and ".local_experts." not in base_prefix
        if is_grouped_expert:
            for adapter_key, adapter_weight in adapter_lookup.items():
                expert_linear_in_gathered[adapter_key] = self._gather_expert_adapter_weight(
                    adapter_weight.linear_in_weight.weight,
                )
                expert_linear_out_gathered[adapter_key] = self._gather_expert_adapter_weight(
                    adapter_weight.linear_out_weight.weight,
                )

        for hf_name, base_weight in converted_weights_dict.items():
            target_adapter = None
            target_adapter_key = None
            for suffix, adapter_key in ADAPTER_NAME_MAP.items():
                if hf_name.endswith(suffix):
                    target_adapter = adapter_lookup.get(adapter_key)
                    target_adapter_key = adapter_key
                    break

            if target_adapter is None:
                raise ValueError(f"Adapter name mapping not found for {hf_name}")

            linear_in_weight = target_adapter.linear_in_weight.weight
            linear_out_weight = target_adapter.linear_out_weight.weight
            if is_grouped_expert:
                expert_idx = self._infer_hf_expert_idx(hf_name)
                linear_in_weight = self._select_expert_adapter_weight(
                    linear_in_weight,
                    expert_linear_in_gathered.get(target_adapter_key),
                    expert_idx,
                    num_moe_experts,
                )
                linear_out_weight = self._select_expert_adapter_weight(
                    linear_out_weight,
                    expert_linear_out_gathered.get(target_adapter_key),
                    expert_idx,
                    num_moe_experts,
                )

            merged_weight = self._merge_single_adapter_weight(
                base_weight,
                target_adapter.alpha,
                target_adapter.dim,
                linear_in_weight,
                linear_out_weight,
            )
            converted_weights_dict[hf_name] = merged_weight

        return converted_weights_dict
