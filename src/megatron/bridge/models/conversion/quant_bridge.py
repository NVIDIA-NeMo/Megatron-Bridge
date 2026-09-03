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

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator, List, Mapping, Optional, Tuple, TypeVar, Union

import torch
from megatron.core.fp8_utils import get_grouped_quantized_members, is_grouped_mxfp8tensor, is_mxfp8tensor  # noqa: F401
from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import unwrap_model


if TYPE_CHECKING:
    from megatron.bridge.models.conversion.model_bridge import HFWeightTuple, WeightConversionTask
    from megatron.bridge.models.conversion.param_mapping import LocalMXFP8Param, MegatronParamMapping


MegatronModel = TypeVar("MegatronModel", bound=MegatronModule)
HFPreTrained = TypeVar("HFPreTrained")


@dataclass(frozen=True)
class _NativeMXFP8Storage:
    weight: torch.Tensor
    weight_scale: torch.Tensor


def _extract_native_mxfp8_storage(
    param: torch.Tensor,
    global_param_name: str,
) -> _NativeMXFP8Storage:
    """Return validated, unconverted native MXFP8 storage views."""
    get_metadata = getattr(param, "get_metadata", None)
    if not callable(get_metadata):
        raise ValueError(f"{global_param_name}: native MXFP8 storage is missing metadata")

    try:
        metadata = get_metadata()
    except (AttributeError, RuntimeError, TypeError) as error:
        raise ValueError(f"{global_param_name}: native MXFP8 storage metadata is unavailable") from error
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{global_param_name}: native MXFP8 storage metadata is invalid")

    logical_shape = torch.Size(param.shape)
    if not logical_shape:
        raise ValueError(f"{global_param_name}: native MXFP8 storage must have a K dimension")
    if logical_shape[-1] % 32:
        raise ValueError(f"{global_param_name}: K={logical_shape[-1]} is not divisible by 32")

    rowwise_data = metadata.get("rowwise_data")
    rowwise_scale = metadata.get("rowwise_scale_inv")
    expected_scale_shape = torch.Size((*logical_shape[:-1], logical_shape[-1] // 32))
    if (
        not isinstance(rowwise_data, torch.Tensor)
        or rowwise_data.dtype != torch.uint8
        or not rowwise_data.is_contiguous()
    ):
        raise ValueError(f"{global_param_name}: invalid native MXFP8 rowwise_data")
    if (
        not isinstance(rowwise_scale, torch.Tensor)
        or rowwise_scale.dtype != torch.uint8
        or not rowwise_scale.is_contiguous()
    ):
        raise ValueError(f"{global_param_name}: invalid native MXFP8 rowwise_scale_inv")
    if metadata.get("is_2D_scaled") is not False:
        raise ValueError(f"{global_param_name}: expected native MXFP8 rowwise scale storage")
    if getattr(metadata.get("quantizer"), "block_len", None) != 32:
        raise ValueError(f"{global_param_name}: expected an MXFP8 block length of 32")
    if rowwise_data.ndim != len(logical_shape) or rowwise_scale.ndim != len(expected_scale_shape):
        raise ValueError(f"{global_param_name}: native MXFP8 storage rank mismatch")
    if any(actual < expected for actual, expected in zip(rowwise_data.shape, logical_shape)):
        raise ValueError(f"{global_param_name}: native MXFP8 value storage is too small")
    if any(actual < expected for actual, expected in zip(rowwise_scale.shape, expected_scale_shape)):
        raise ValueError(f"{global_param_name}: native MXFP8 scale storage is too small")

    data_slices = tuple(slice(0, size) for size in logical_shape)
    scale_slices = tuple(slice(0, size) for size in expected_scale_shape)
    return _NativeMXFP8Storage(
        weight=rowwise_data[data_slices].view(torch.float8_e4m3fn),
        weight_scale=rowwise_scale[scale_slices],
    )


def _lookup_grouped_expert_mapping(
    mapping_registry: Any,
    global_param_name: str,
) -> Optional["MegatronParamMapping"]:
    """Resolve a grouped-member mapping without relying on a parameter-name suffix."""
    if mapping_registry.megatron_to_hf_lookup(global_param_name) is not None:
        return None
    mapping = mapping_registry.megatron_to_hf_lookup(f"{global_param_name}0")
    if mapping is None or not mapping.is_expert:
        return None
    return mapping


def _supports_native_grouped_mxfp8(mapping: "MegatronParamMapping") -> bool:
    from megatron.bridge.models.conversion.param_mapping import FusedExpertMapping, FusedGatedExpertMapping

    return isinstance(mapping, (FusedGatedExpertMapping, FusedExpertMapping))


class MegatronQuantizationBridge:
    """Mixin providing quantization-aware utilities for Megatron model bridges."""

    def build_export_mxfp8_tasks(
        self,
        hf_pretrained: HFPreTrained,
        megatron_model: List[MegatronModel],
    ) -> List["WeightConversionTask"]:
        """Build deterministic export tasks for native MXFP8 parameters.

        Singular grouped-expert parameters remain one task when their storage is
        native MXFP8. Grouped parameters kept in BF16 are expanded into ordinary
        per-expert tasks so existing conversion mappings continue to apply.

        Args:
            hf_pretrained: Hugging Face model metadata used for mapping validation.
            megatron_model: Virtual-pipeline model chunks on the current rank.

        Returns:
            Conversion tasks in global Megatron parameter order.
        """
        from megatron.bridge.models.conversion.model_bridge import (
            WeightConversionTask,
            _get_pg_collection_from_model,
            _get_pp_rank,
            _megatron_local_name_to_global,
            get_module_and_param_from_name,
            persistent_buffers,
            unwrap_model,
        )

        if not megatron_model:
            raise ValueError("megatron_model must contain at least one model chunk")

        has_hf_state = hasattr(hf_pretrained, "state") and hasattr(hf_pretrained.state, "source")
        self.hf_pretrained = hf_pretrained
        self.hf_config = hf_pretrained.config if hasattr(hf_pretrained, "config") else hf_pretrained

        mapping_registry = self.mapping_registry()
        mapping_registry.set_process_groups_from_pg_collection(_get_pg_collection_from_model(megatron_model))
        model_config = unwrap_model(megatron_model)[0].config
        pp_rank = _get_pp_rank(megatron_model)
        global_names = self._megatron_global_param_names_all_pp_ranks(megatron_model)
        if self._share_embeddings_and_output_weights(model_config):
            global_names = [name for name in global_names if "output_layer" not in name]

        num_experts = int(getattr(model_config, "num_moe_experts", 0) or 0)
        ep_size = int(getattr(model_config, "expert_model_parallel_size", 1) or 1)
        if num_experts and num_experts % ep_size:
            raise ValueError(
                f"num_moe_experts={num_experts} must be divisible by expert_model_parallel_size={ep_size}"
            )
        local_expert_count = num_experts // ep_size if num_experts else 0

        grouped_mappings = {
            global_name: mapping
            for global_name in global_names
            if (mapping := _lookup_grouped_expert_mapping(mapping_registry, global_name)) is not None
        }
        for global_name in grouped_mappings:
            if self._is_mtp_param(global_name):
                raise ValueError("Native MXFP8 export does not support co-trained MTP grouped experts")

        local_by_global_name: dict[str, tuple[int, str, Any, torch.Tensor]] = {}
        local_grouped_storage: dict[str, bool] = {}
        local_grouped_member_counts: dict[str, Optional[int]] = {}
        for vp_stage, model in enumerate(megatron_model):
            for local_name, _ in itertools.chain(model.named_parameters(), persistent_buffers(model)):
                if "_extra_state" in local_name or self._is_adapter_param_name(local_name):
                    continue
                local_name = self._unwrap_name(local_name)
                global_name = _megatron_local_name_to_global(megatron_model, model_config, local_name, vp_stage)
                local_module, local_weight = get_module_and_param_from_name(megatron_model, local_name, vp_stage)
                if local_weight is None:
                    continue
                if local_module is not None and not hasattr(local_module, "config"):
                    setattr(local_module, "config", model_config)
                local_by_global_name[global_name] = (vp_stage, local_name, local_module, local_weight)
                if global_name in grouped_mappings:
                    uses_native_storage = is_grouped_mxfp8tensor(local_weight)
                    local_grouped_storage[global_name] = uses_native_storage
                    if uses_native_storage and _supports_native_grouped_mxfp8(grouped_mappings[global_name]):
                        members = get_grouped_quantized_members(local_weight, create_if_missing=True)
                        local_grouped_member_counts[global_name] = None if members is None else len(tuple(members))

        native_grouped_names: set[str] = set()
        for global_name, mapping in grouped_mappings.items():
            local_uses_native = local_grouped_storage.get(global_name)
            broadcaster = getattr(mapping, "broadcast_obj_from_pp_rank", None)
            uses_native = (
                bool(
                    broadcaster(
                        local_uses_native,
                        cache_key=f"native-grouped-storage:{global_name}",
                    )
                )
                if callable(broadcaster)
                else bool(local_uses_native)
            )
            if uses_native:
                if not _supports_native_grouped_mxfp8(mapping):
                    raise ValueError(
                        f"{global_name}: native grouped MXFP8 export requires "
                        "FusedGatedExpertMapping or FusedExpertMapping"
                    )
                if local_expert_count <= 0:
                    raise ValueError(
                        f"Cannot validate grouped expert parameter {global_name!r} without num_moe_experts"
                    )
                local_member_count = local_grouped_member_counts.get(global_name)
                member_count = (
                    broadcaster(
                        local_member_count,
                        cache_key=f"native-grouped-member-count:{global_name}",
                    )
                    if callable(broadcaster)
                    else local_member_count
                )
                if member_count is None:
                    raise ValueError(f"{global_name}: missing cached grouped MXFP8 members")
                if member_count != local_expert_count:
                    raise ValueError(
                        f"{global_name}: grouped MXFP8 storage has {member_count} local members, "
                        f"expected {local_expert_count}"
                    )
                native_grouped_names.add(global_name)

        grouped_expansions: dict[str, list[str]] = {}
        ordered_names: list[str] = []
        for global_name in global_names:
            if global_name in grouped_mappings and global_name not in native_grouped_names:
                if local_expert_count <= 0:
                    raise ValueError(f"Cannot expand grouped expert parameter {global_name!r} without num_moe_experts")
                expert_offset = int(grouped_mappings[global_name].ep_rank) * local_expert_count
                expanded_names = [
                    f"{global_name}{expert_offset + local_expert_id}" for local_expert_id in range(local_expert_count)
                ]
                grouped_expansions[global_name] = expanded_names
                ordered_names.extend(expanded_names)
            else:
                ordered_names.append(global_name)

        ordinary_names = [name for name in ordered_names if name not in native_grouped_names]
        hf_keys = hf_pretrained.state.source.get_all_keys() if has_hf_state else None
        mappings = self._validate_conversion_mappings(mapping_registry, ordinary_names, hf_keys)
        tasks_by_name: dict[str, WeightConversionTask] = {}

        for global_name in native_grouped_names:
            local = local_by_global_name.get(global_name)
            tasks_by_name[global_name] = WeightConversionTask(
                pp_rank=pp_rank,
                vp_stage=local[0] if local is not None else None,
                param_name=local[1] if local is not None else global_name,
                global_param_name=global_name,
                megatron_module=local[2] if local is not None else None,
                param_weight=local[3] if local is not None else None,
                mapping=grouped_mappings[global_name],
            )

        for global_name, local in local_by_global_name.items():
            vp_stage, local_name, local_module, local_weight = local
            expanded_names = grouped_expansions.get(global_name)
            if expanded_names is not None:
                members = list(local_weight.unbind(0))
                if len(members) != len(expanded_names):
                    raise ValueError(
                        f"Grouped expert parameter {global_name!r} has {len(members)} local members, "
                        f"expected {len(expanded_names)}"
                    )
                for expert_id, expanded_name in enumerate(expanded_names):
                    tasks_by_name[expanded_name] = WeightConversionTask(
                        pp_rank=pp_rank,
                        vp_stage=vp_stage,
                        param_name=f"{local_name}{expert_id}",
                        global_param_name=expanded_name,
                        megatron_module=local_module,
                        param_weight=members[expert_id],
                        mapping=mappings[expanded_name],
                    )
            elif global_name in mappings:
                tasks_by_name[global_name] = WeightConversionTask(
                    pp_rank=pp_rank,
                    vp_stage=vp_stage,
                    param_name=local_name,
                    global_param_name=global_name,
                    megatron_module=local_module,
                    param_weight=local_weight,
                    mapping=mappings[global_name],
                )

        for global_name in ordinary_names:
            if global_name not in tasks_by_name:
                tasks_by_name[global_name] = WeightConversionTask(
                    pp_rank=pp_rank,
                    vp_stage=None,
                    param_name=global_name,
                    global_param_name=global_name,
                    megatron_module=None,
                    param_weight=None,
                    mapping=mappings[global_name],
                )
        return [tasks_by_name[name] for name in ordered_names]

    def _iter_grouped_native_mxfp8_params(
        self,
        task: "WeightConversionTask",
        members: Optional[Iterable[torch.Tensor]],
    ) -> Iterator["LocalMXFP8Param"]:
        """Project cached grouped MXFP8 members in local expert order."""
        from megatron.bridge.models.conversion.param_mapping import FusedExpertMapping, FusedGatedExpertMapping

        if members is None:
            raise ValueError(f"{task.global_param_name}: missing cached grouped MXFP8 members")
        if not isinstance(task.mapping, (FusedGatedExpertMapping, FusedExpertMapping)) or not task.mapping.is_expert:
            raise ValueError(f"{task.global_param_name}: unsupported grouped MXFP8 expert mapping")

        ep_size = int(task.mapping.ep_size)
        config = getattr(task.megatron_module, "config", None)
        num_experts = int(getattr(config, "num_moe_experts", 0) or 0)
        if num_experts <= 0 or num_experts % ep_size:
            raise ValueError(
                f"{task.global_param_name}: num_moe_experts={num_experts} must be divisible by ep_size={ep_size}"
            )
        experts_per_rank = num_experts // ep_size
        local_members = tuple(members)
        if len(local_members) != experts_per_rank:
            raise ValueError(
                f"{task.global_param_name}: grouped MXFP8 storage has {len(local_members)} local members, "
                f"expected {experts_per_rank}"
            )

        expert_offset = int(task.mapping.ep_rank) * experts_per_rank
        for local_expert_id, member in enumerate(local_members):
            global_expert_id = expert_offset + local_expert_id
            member_name = f"{task.global_param_name}{global_expert_id}"
            storage = _extract_native_mxfp8_storage(member, member_name)
            yield from task.mapping.local_mxfp8_params(
                storage.weight,
                storage.weight_scale,
                global_param_name=member_name,
                megatron_module=task.megatron_module,
            )

    @staticmethod
    def _is_mtp_param(param_name: str) -> bool:
        """Return whether a Megatron parameter belongs to a co-trained MTP module."""
        return param_name.startswith("mtp.") or ".mtp." in param_name

    def stream_weights_megatron_to_hf_quant(
        self,
        megatron_model: Union[MegatronModel, List[MegatronModel]],
        hf_pretrained: HFPreTrained,
        quantization_checker: Callable[[str], bool],
        quant_fn: Callable[..., Tuple[torch.Tensor, torch.Tensor]],
        quant_block_size: Optional[Tuple[int, int]] = None,
        cpu: bool = True,
        show_progress: bool = True,
        conversion_tasks: Optional[List["WeightConversionTask"]] = None,
        merge_adapter_weights: bool = False,
    ) -> Iterable["HFWeightTuple"]:
        """Export Megatron weights to HuggingFace format with quantization."""
        from megatron.bridge.models.conversion.model_bridge import HFWeightTuple

        assert not merge_adapter_weights, (
            "Adapter merging is not supported for quantized weights. Use merge_adapter_weights=False instead."
        )

        if not isinstance(megatron_model, list):
            megatron_model = [megatron_model]

        self.hf_pretrained = hf_pretrained
        self.hf_config = hf_pretrained.config if hasattr(hf_pretrained, "config") else hf_pretrained

        # Use provided conversion tasks or build them
        if conversion_tasks is None:
            conversion_tasks = self.build_conversion_tasks(hf_pretrained, megatron_model)

        megatron_to_hf_tasks = conversion_tasks
        unwrapped_model = unwrap_model(megatron_model)[0]
        model_config = unwrapped_model.config
        embeddings_are_tied = self._share_embeddings_and_output_weights(model_config)

        hf_state_dict: Mapping[str, torch.Tensor] = hf_pretrained.state if hasattr(hf_pretrained, "state") else {}
        grouped_buffers: dict[str, dict[int, torch.Tensor]] = {}

        for task in self._with_progress_tracking(
            megatron_to_hf_tasks, "Converting to HuggingFace (Quantized)", show_progress
        ):
            converted_weights_dict = task.mapping.megatron_to_hf_quant(
                task.param_weight, task.megatron_module, quantization_checker, quant_fn, quant_block_size
            )
            if getattr(task.mapping, "is_grouped_export", False):
                converted_weights_dict = self._accumulate_grouped_export(
                    task,
                    converted_weights_dict,
                    model_config,
                    grouped_buffers,
                    hf_state_dict,
                )
                if converted_weights_dict is None:
                    continue
            converted_weights_dict = self.maybe_modify_converted_hf_weight(
                task,
                converted_weights_dict,
                hf_state_dict,
            )
            scale_block_size = quant_block_size[0] if quant_block_size is not None else None
            converted_weights_dict = self._truncate_vocab_padding(
                task,
                converted_weights_dict,
                scale_block_size=scale_block_size,
            )

            for hf_name, tensor in converted_weights_dict.items():
                final_tensor = tensor.cpu() if cpu else tensor

                if not merge_adapter_weights and "to_wrap.weight" in task.global_param_name:
                    hf_name = hf_name[: -len("weight")] + "base_layer.weight"

                if embeddings_are_tied and hf_name == "model.embed_tokens.weight":
                    yield HFWeightTuple(hf_name, final_tensor)
                    if hasattr(hf_pretrained, "state") and hasattr(hf_pretrained.state, "source"):
                        expected_keys = hf_pretrained.state.source.get_all_keys()
                        if "lm_head.weight" in expected_keys:
                            yield HFWeightTuple("lm_head.weight", final_tensor.clone().detach())
                elif embeddings_are_tied and hf_name == "lm_head.weight":
                    raise ValueError(
                        "Encountered lm_head.weight when embeddings are tied. This indicates a mapping error."
                    )
                else:
                    yield HFWeightTuple(hf_name, final_tensor)
