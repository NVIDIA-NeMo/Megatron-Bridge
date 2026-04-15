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
    _split_gdn_grouped_to_separate,
    split_gdn_linear_weights,
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
    from megatron.bridge.peft.base import PEFT


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

GDN_IN_PROJ_KEYS = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")


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
        return hf_param if hf_param.endswith(expected_suffix) or expected_suffix == ".weight" else None

    if isinstance(hf_param, dict):
        if adapter_key:
            target_suffix = ADAPTER_KEY_TO_SUFFIX.get(adapter_key)
            if target_suffix:
                for value in hf_param.values():
                    if value.endswith(target_suffix):
                        return value

        # For fused qkv/gate_up case, we just need a placeholder here
        value = next(iter(hf_param.values()))
        return value if value.endswith(expected_suffix) or expected_suffix == ".weight" else None

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
        if hf_base_name is None:
            return None

        if hf_base_name.endswith(base_suffix):
            return hf_base_name[: -len(base_suffix)] + hf_suffix

        # Some HF base names (e.g., Qwen3.5 MoE expert gate_up_proj / down_proj)
        # don't include a trailing ".weight". Allow LoRA suffix to be appended directly.
        if base_suffix == ".weight":
            return hf_base_name + hf_suffix

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

        hf_suffix = MEGATRON_TO_HF_LORA_SUFFIX.get(megatron_adapter_suffix)
        if hf_suffix is None:
            return None

        if base_name.endswith(".weight"):
            return base_name[: -len(".weight")] + hf_suffix

        # Some HF base names (e.g., Qwen3.5 MoE expert gate_up_proj) omit ".weight".
        return base_name + hf_suffix

    def _is_fused_qkv(self, hf_weight_names: Iterable[str]) -> bool:
        """Check whether the provided HF names correspond to a fused QKV weight."""

        names = list(hf_weight_names)
        if len(names) != 3:
            return False

        required = {"q_proj", "k_proj", "v_proj"}
        discovered = {token for name in names for token in required if token in name}
        return discovered == required

    def _is_gdn_in_proj_split(self, hf_weight_names: Iterable[str]) -> bool:
        """Check whether the provided HF names correspond to split GDN in_proj weights."""

        names = list(hf_weight_names)
        if len(names) != 4:
            return False
        required = set(GDN_IN_PROJ_KEYS)
        discovered = {token for name in names for token in required if token in name}
        return discovered == required and all("linear_attn" in name for name in names)

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

    def _infer_gdn_in_proj_projection_from_name(self, hf_name: str) -> Optional[str]:
        """Return in_proj_qkv/z/b/a identifier based on the HF name."""

        for projection_key in GDN_IN_PROJ_KEYS:
            if projection_key in hf_name:
                return projection_key
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
        # Pass the LoRA rank as feature_dim so split_qkv_weights doesn't
        # mistake it for an FP8 compressed hidden_size.
        feature_dim = linear_out_weight.shape[-1] if linear_out_weight.ndim == 2 else None
        q_out, k_out, v_out = split_qkv_weights(model.config, linear_out_weight, feature_dim=feature_dim)
        return {"q_proj": q_out, "k_proj": k_out, "v_proj": v_out}

    def _split_gdn_in_proj_linear_out_weight(
        self,
        megatron_model: Union[MegatronModel, List[MegatronModel]],
        linear_out_weight: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Split a fused LoRA linear_out tensor for GDN in_proj adapters."""

        model = megatron_model[0] if isinstance(megatron_model, list) else megatron_model
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        feature_dim = linear_out_weight.shape[1]
        qkvz, ba = split_gdn_linear_weights(
            model.config,
            linear_out_weight,
            tp_size=tp_size,
            feature_dim=feature_dim,
        )
        qkv, z, b, a = _split_gdn_grouped_to_separate(model.config, qkvz, ba, feature_dim=feature_dim)
        return {"in_proj_qkv": qkv, "in_proj_z": z, "in_proj_b": b, "in_proj_a": a}

    def _build_lora_hf_names(self, base_hf_weight_names: List[str]) -> tuple[List[str], List[str]]:
        """Build LoRA A/B names for a list of HF base parameter names."""

        linear_in_hf_names = [
            self._make_lora_param_name(base_name, ".linear_in.weight") for base_name in base_hf_weight_names
        ]
        linear_out_hf_names = [
            self._make_lora_param_name(base_name, ".linear_out.weight") for base_name in base_hf_weight_names
        ]
        return linear_in_hf_names, linear_out_hf_names

    def _collect_packed_expert_adapter_tensors(
        self,
        linear_in_tensor: torch.Tensor,
        linear_out_tensor: torch.Tensor,
        expert_linear_in_gathered: Optional[List[torch.Tensor]],
        expert_linear_out_gathered: Optional[List[torch.Tensor]],
        num_moe_experts: int,
    ) -> tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Collect one LoRA A/B tensor per expert for grouped expert exports."""

        per_expert_linear_in: List[torch.Tensor] = []
        per_expert_linear_out: List[torch.Tensor] = []
        if linear_in_tensor.ndim > 2 or linear_out_tensor.ndim > 2:
            # Already carries local expert dim; concatenate across EP ranks if needed.
            linear_in_all = (
                torch.cat(expert_linear_in_gathered, dim=0)
                if expert_linear_in_gathered is not None
                else linear_in_tensor
            )
            linear_out_all = (
                torch.cat(expert_linear_out_gathered, dim=0)
                if expert_linear_out_gathered is not None
                else linear_out_tensor
            )
            per_expert_linear_in = list(linear_in_all)
            per_expert_linear_out = list(linear_out_all)
            return per_expert_linear_in, per_expert_linear_out

        for expert_idx in range(num_moe_experts):
            per_expert_linear_in.append(
                self._select_expert_adapter_weight(
                    linear_in_tensor,
                    expert_linear_in_gathered,
                    expert_idx,
                    num_moe_experts,
                )
            )
            per_expert_linear_out.append(
                self._select_expert_adapter_weight(
                    linear_out_tensor,
                    expert_linear_out_gathered,
                    expert_idx,
                    num_moe_experts,
                )
            )
        return per_expert_linear_in, per_expert_linear_out

    def _build_packed_expert_linear_out_by_base(
        self,
        megatron_model: List[MegatronModel],
        base_hf_weight_names: List[str],
        per_expert_linear_out: List[torch.Tensor],
        is_expert: bool,
    ) -> Dict[str, torch.Tensor]:
        """Build per-base stacked LoRA-B tensors for packed grouped-expert export."""

        if not per_expert_linear_out:
            return {}

        # Handle fused adapters (qkv/gate_up/gdn in_proj) by splitting per-expert then stacking.
        per_base_linear_out = self._get_fused_adapter_linear_out_slices(
            megatron_model,
            base_hf_weight_names,
            per_expert_linear_out[0],
            is_expert=is_expert,
        )
        if per_base_linear_out is None:
            stacked = torch.stack(per_expert_linear_out, dim=0)
            return {base_name: stacked for base_name in base_hf_weight_names}

        per_base_stacks: Dict[str, List[torch.Tensor]] = {name: [] for name in base_hf_weight_names}
        for expert_out in per_expert_linear_out:
            per_base = self._get_fused_adapter_linear_out_slices(
                megatron_model,
                base_hf_weight_names,
                expert_out,
                is_expert=is_expert,
            )
            assert per_base is not None, "Expected fused adapter split for expert LoRA"
            for base_name in base_hf_weight_names:
                per_base_stacks[base_name].append(per_base[base_name])

        return {base_name: torch.stack(parts, dim=0) for base_name, parts in per_base_stacks.items()}

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

            # If the HF base names don't include experts.N, emit packed expert weights
            # (stacked along dim 0) once per HF name instead of duplicating per expert.
            packed_expert = False
            base_hf_weight_names: List[str] = []
            if is_grouped_expert and base_suffixes:
                base_hf_weight_names = self._get_base_hf_param_names_for_adapter(
                    mapping_registry,
                    adapter_task.global_base_prefix,
                    adapter_task.adapter_key,
                    base_suffixes[0],
                )
                if base_hf_weight_names and not any(
                    re.search(r"experts\.(\d+)", name) for name in base_hf_weight_names
                ):
                    packed_expert = True

            if packed_expert:
                linear_in_hf_names, linear_out_hf_names = self._build_lora_hf_names(base_hf_weight_names)
                per_expert_linear_in, per_expert_linear_out = self._collect_packed_expert_adapter_tensors(
                    linear_in_tensor,
                    linear_out_tensor,
                    expert_linear_in_gathered,
                    expert_linear_out_gathered,
                    num_moe_experts,
                )

                if not per_expert_linear_in or not per_expert_linear_out:
                    raise ValueError(
                        f"Expected to find per-expert adapter weights for grouped expert "
                        f"linear layer but none found, global_base_prefix={adapter_task.global_base_prefix}"
                    )
                linear_in_stacked = torch.stack(per_expert_linear_in, dim=0)
                if cpu:
                    linear_in_stacked = linear_in_stacked.cpu()

                if adapter_task.adapter_key is None:
                    linear_out_by_base = self._build_packed_expert_linear_out_by_base(
                        megatron_model,
                        base_hf_weight_names,
                        per_expert_linear_out,
                        is_expert=is_expert_linear(adapter_task.global_base_prefix),
                    )
                else:
                    shared_linear_out = torch.stack(per_expert_linear_out, dim=0)
                    linear_out_by_base = {base_name: shared_linear_out for base_name in base_hf_weight_names}

                for index, base_name in enumerate(base_hf_weight_names):
                    linear_out_stacked = linear_out_by_base[base_name]
                    if cpu:
                        linear_out_stacked = linear_out_stacked.cpu()
                    yield HFWeightTuple(linear_in_hf_names[index], linear_in_stacked)
                    yield HFWeightTuple(linear_out_hf_names[index], linear_out_stacked)

                continue

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
                linear_in_hf_names, linear_out_hf_names = self._build_lora_hf_names(base_hf_weight_names)
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

        if self._is_gdn_in_proj_split(base_hf_weight_names):
            gdn_linear_out_weights = self._split_gdn_in_proj_linear_out_weight(megatron_model, linear_out_tensor)
            per_base = {}
            for base_name in base_hf_weight_names:
                projection_key = self._infer_gdn_in_proj_projection_from_name(base_name)
                if projection_key is None:
                    raise ValueError(f"Unknown GDN in_proj base weight name: {base_name}")
                per_base[base_name] = gdn_linear_out_weights[projection_key]
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

        if not converted_weights_dict:
            # Nothing to merge on this rank (e.g., non-owning PP rank or filtered mapping).
            return converted_weights_dict

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

        base_weight = next(iter(converted_weights_dict.values()))
        base_weight_shape = base_weight.shape
        weight_names = converted_weights_dict.keys()
        if self._is_gdn_in_proj_split(weight_names):
            # GDN in_proj LoRA is defined on the fused Megatron tensor; split it into
            # the four HF tensors (qkv/z/b/a) before merging.
            config = unwrap_model(megatron_model)[0].config
            hidden_size = config.hidden_size
            qk_dim = config.linear_key_head_dim * config.linear_num_key_heads
            v_dim = config.linear_value_head_dim * config.linear_num_value_heads
            num_v_heads = config.linear_num_value_heads
            fused_dim0 = 2 * qk_dim + 2 * v_dim + 2 * num_v_heads

            base_device = base_weight.device
            linear_out_on_base = (
                linear_out_weight if linear_out_weight.device == base_device else linear_out_weight.to(base_device)
            )
            linear_in_on_base = (
                linear_in_weight if linear_in_weight.device == base_device else linear_in_weight.to(base_device)
            )
            dummy_base = torch.zeros((fused_dim0, hidden_size), device=base_device, dtype=base_weight.dtype)
            lora_weight = LoRAMerge().merge(dummy_base, linear_out_on_base, linear_in_on_base, alpha, dim)

            tp_size = parallel_state.get_tensor_model_parallel_world_size()
            qkvz, ba = split_gdn_linear_weights(config, lora_weight, tp_size=tp_size)
            qkv, z, b, a = _split_gdn_grouped_to_separate(config, qkvz, ba)
            gdn_slices = {"in_proj_qkv": qkv, "in_proj_z": z, "in_proj_b": b, "in_proj_a": a}

            for hf_name, base_tensor in list(converted_weights_dict.items()):
                projection_key = self._infer_gdn_in_proj_projection_from_name(hf_name)
                if projection_key is None:
                    raise ValueError(f"Unknown GDN in_proj weight name: {hf_name}")
                converted_weights_dict[hf_name] = base_tensor + gdn_slices[projection_key]

            return converted_weights_dict
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
                if expert_idx is not None:
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

    def _merge_grouped_export_adapter_weights(
        self,
        task: "WeightConversionTask",
        converted_weights_dict: Dict[str, torch.Tensor],
        adapter_weights: List[AdapterWeight],
        num_moe_experts: int,
    ) -> Dict[str, torch.Tensor]:
        """Merge LoRA weights into a single grouped-expert export shard.

        Grouped expert mappings bypass the standard export path and therefore
        never reach `_merge_lora_adapter_weights`. Merge the current expert's
        adapter slice into its per-expert tensor before the grouped export code
        stacks all experts back together.
        """

        if not converted_weights_dict or not adapter_weights:
            return converted_weights_dict

        if len(adapter_weights) != 1 or adapter_weights[0].adapter_key is not None:
            adapter_keys = [adapter_weight.adapter_key for adapter_weight in adapter_weights]
            raise ValueError(
                "Unsupported adapter configuration for grouped export weight merging "
                f"for parameter '{task.global_param_name}': expected exactly one "
                "non-canonical adapter with adapter_key=None, but got "
                f"{len(adapter_weights)} adapter(s) with adapter keys {adapter_keys}."
            )

        from megatron.bridge.utils.common_utils import extract_expert_number_from_param

        expert_idx = extract_expert_number_from_param(task.global_param_name)

        adapter_weight = adapter_weights[0]
        linear_in_weight = adapter_weight.linear_in_weight.weight
        linear_out_weight = adapter_weight.linear_out_weight.weight

        expert_linear_in_gathered = self._gather_expert_adapter_weight(linear_in_weight)
        expert_linear_out_gathered = self._gather_expert_adapter_weight(linear_out_weight)

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

        for hf_name, base_weight in list(converted_weights_dict.items()):
            converted_weights_dict[hf_name] = self._merge_single_adapter_weight(
                base_weight,
                adapter_weight.alpha,
                adapter_weight.dim,
                current_linear_in_weight,
                current_linear_out_weight,
            )

        return converted_weights_dict

    def _merge_single_adapter_weight(
        self,
        base_weight: torch.Tensor,
        alpha: int,
        dim: int,
        linear_in_weight: torch.Tensor,
        linear_out_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Merge a single adapter's weights with base weight.

        The merge is performed in float32 to avoid precision loss from
        bfloat16 matmul (adapter weights are often stored in bf16).
        The result is cast back to the original base weight dtype.
        """

        orig_dtype = base_weight.dtype
        merger = LoRAMerge()
        base_device = base_weight.device
        linear_out_on_base = linear_out_weight.to(device=base_device, dtype=torch.float32)
        linear_in_on_base = linear_in_weight.to(device=base_device, dtype=torch.float32)
        merged = merger.merge(
            base_weight.float(),
            linear_out_on_base,
            linear_in_on_base,
            alpha,
            dim,
        )
        return merged.to(orig_dtype)

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
                if expert_idx is not None:
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


_HF_LORA_SUFFIXES = (".lora_A.weight", ".lora_B.weight")


def infer_target_modules_from_adapter_weights(adapter_weight_names: Iterable[str]) -> List[str]:
    """Derive HF ``target_modules`` from the HF-format adapter weight names.

    Given names like ``model.layers.0.self_attn.q_proj.lora_A.weight``, this
    extracts the unique module identifiers (``q_proj``, ``gate_proj``, ...) that
    the ``peft`` library expects in ``adapter_config.json``.
    """

    modules: set[str] = set()
    for name in adapter_weight_names:
        for suffix in _HF_LORA_SUFFIXES:
            if name.endswith(suffix):
                base = name[: -len(suffix)]
                module_name = base.rsplit(".", 1)[-1]
                modules.add(module_name)
                break
    return sorted(modules)


def build_adapter_config_dict(
    peft_config: PEFT,
    target_modules: List[str],
    base_model_name_or_path: Optional[str] = None,
) -> Dict[str, object]:
    """Build an HF PEFT-compatible ``adapter_config.json`` dictionary.

    The returned dict can be serialised directly with ``json.dump`` and is
    loadable by ``peft.PeftModel.from_pretrained`` without any runtime
    dependency on the ``peft`` pip package.
    """

    from megatron.bridge.peft.dora import DoRA

    config: Dict[str, object] = {
        "peft_type": "LORA",
        "auto_mapping": None,
        "base_model_name_or_path": base_model_name_or_path or "",
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "lora_alpha": getattr(peft_config, "alpha", 32),
        "lora_dropout": getattr(peft_config, "dropout", 0.0),
        "modules_to_save": None,
        "r": getattr(peft_config, "dim", 32),
        "rank_pattern": {},
        "alpha_pattern": {},
        "target_modules": target_modules,
        "task_type": "CAUSAL_LM",
        "use_dora": isinstance(peft_config, DoRA),
        "use_rslora": False,
    }
    return config
