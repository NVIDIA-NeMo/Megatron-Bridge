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

"""Parameter mapping for PEFT adapters with distributed training support.

This module provides parameter mapping classes that handle the conversion between
HuggingFace PEFT parameter formats and Megatron distributed parameter formats,
with support for tensor parallelism and expert parallelism.

The AdapterMappingFactory co-locates mapping creation logic with the mapping classes
for better maintainability and enables bridge-specific customization.
"""

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
import torch.nn as nn

from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    MegatronParamMapping,
    QKVMapping,
)


if TYPE_CHECKING:
    from megatron.bridge.peft.conversion.peft_bridge import MegatronPEFTBridge


class AdapterMappingFactory:
    """Factory for creating adapter parameter mappings.

    Co-locates mapping creation logic with mapping classes for better maintainability
    and enables bridge-specific customization through optional bridge context.
    """

    def __init__(self, bridge_context: Optional["MegatronPEFTBridge"] = None):
        """Initialize factory with optional bridge context for customization.

        Args:
            bridge_context: Optional bridge instance for custom mapping creation
        """
        self.bridge_context = bridge_context

    def create_adapter_mapping(
        self, base_mapping: MegatronParamMapping, adapter_megatron_param: str, hf_suffix: str
    ) -> MegatronParamMapping:
        """Create adapter mapping with optional bridge context for customization.

        Args:
            base_mapping: Base model parameter mapping to adapt
            adapter_megatron_param: Megatron adapter parameter name
            hf_suffix: HuggingFace parameter suffix (e.g., ".lora_A.weight")

        Returns:
            Appropriate adapter mapping instance

        Raises:
            ValueError: If no mapping is available for the base mapping type
        """
        # Allow bridge-specific customization first
        if self.bridge_context and hasattr(self.bridge_context, "create_custom_adapter_mapping"):
            custom_mapping = self.bridge_context.create_custom_adapter_mapping(
                base_mapping, adapter_megatron_param, hf_suffix
            )
            if custom_mapping is not None:
                return custom_mapping

        # Default factory logic using isinstance for type dispatch
        if isinstance(base_mapping, QKVMapping):
            return AdapterQKVMapping.from_base_mapping(base_mapping, adapter_megatron_param, hf_suffix)
        elif isinstance(base_mapping, AutoMapping):
            return AdapterAutoMapping.from_base_mapping(base_mapping, adapter_megatron_param, hf_suffix)
        elif isinstance(base_mapping, GatedMLPMapping):
            return AdapterGatedMLPMapping.from_base_mapping(base_mapping, adapter_megatron_param, hf_suffix)
        else:
            raise ValueError(
                f"No adapter mapping available for {type(base_mapping).__name__}. "
                f"Add from_base_mapping() class method to corresponding AdapterXXXMapping class, "
                f"or override create_custom_adapter_mapping() in your bridge."
            )


class AdapterAutoMapping(MegatronParamMapping[torch.Tensor]):
    """Auto-detecting parameter mapping for PEFT adapters with EP/TP support.

    This mapping class automatically handles tensor parallel and expert parallel
    distribution based on the base module's parallelism type and adapter structure.
    """

    @classmethod
    def from_base_mapping(
        cls, base_mapping: AutoMapping, adapter_megatron_param: str, hf_suffix: str
    ) -> "AdapterAutoMapping":
        """Create AdapterAutoMapping from base AutoMapping.

        Args:
            base_mapping: Base AutoMapping to adapt
            adapter_megatron_param: Megatron adapter parameter name
            hf_suffix: HuggingFace parameter suffix

        Returns:
            New AdapterAutoMapping instance
        """
        adapter_hf_param = base_mapping.hf_param.replace(".weight", hf_suffix)
        return cls(hf_param=adapter_hf_param, megatron_param=adapter_megatron_param)

    @property
    def is_expert(self) -> bool:
        """Check if this is an expert parameter."""
        return ".mlp.experts.linear_fc" in self.megatron_param

    def hf_to_megatron(self, hf_weights: torch.Tensor, megatron_module: nn.Module) -> torch.Tensor:
        """Convert HF weights to Megatron format with correct parallelism.

        Args:
            hf_weights: HuggingFace format weights
            megatron_module: The Megatron module to load weights into

        Returns:
            Weights distributed according to the base module's parallelism
        """
        # Use module directly to determine parallelism type
        base = megatron_module.to_wrap if hasattr(megatron_module, "to_wrap") else megatron_module
        partition_dim = getattr(base, "partition_dim", None)

        # Determine if this is A (linear_in) or B (linear_out)
        is_matrix_a = "linear_in" in self.megatron_param

        # Apply correct distribution based on base parallelism
        if partition_dim == 0:  # Column-parallel base
            if is_matrix_a:
                # A is replicated for column-parallel
                return self._replicate_to_tp_ranks(hf_weights)
            else:
                # B follows column-parallel (split dim 0)
                return self._column_parallel_scatter(hf_weights)
        elif partition_dim == 1:  # Row-parallel base
            if is_matrix_a:
                # A follows row-parallel (split dim 1 for weight, dim 0 for bias)
                dim = 1 if hf_weights.ndim == 2 else 0
                return self._row_parallel_scatter(hf_weights, dim=dim)
            else:
                # B is replicated for row-parallel
                return self._replicate_to_tp_ranks(hf_weights)
        else:  # Replicated base
            return self._replicate_to_tp_ranks(hf_weights)

    def megatron_to_hf(
        self, megatron_weights: Optional[torch.Tensor], megatron_module: Optional[nn.Module]
    ) -> Dict[str, torch.Tensor]:
        """Convert Megatron weights to HF format with proper gathering.

        Args:
            megatron_weights: Megatron format weights (potentially sharded)
            megatron_module: The Megatron module

        Returns:
            Dictionary with HF parameter name and gathered weights
        """
        if megatron_weights is None:
            return {}

        # Handle EP gathering if needed
        if self.is_expert:
            gathered = self.gather_from_ep_ranks(megatron_weights, megatron_module, self.hf_param)
            return gathered

        # Standard TP gathering
        gathered_weights = self.gather_from_tp_ranks(megatron_weights)

        # Reconstruct full weight from gathered pieces
        if len(gathered_weights) > 1:
            # Need to determine concat dimension based on base parallelism
            base = megatron_module.to_wrap if hasattr(megatron_module, "to_wrap") else megatron_module
            partition_dim = getattr(base, "partition_dim", None)
            is_matrix_a = "linear_in" in self.megatron_param

            if partition_dim == 0 and not is_matrix_a:
                # Column-parallel B matrix: concat on dim 0
                full_weight = torch.cat(gathered_weights, dim=0)
            elif partition_dim == 1 and is_matrix_a:
                # Row-parallel A matrix: concat on appropriate dim
                concat_dim = 1 if megatron_weights.ndim == 2 else 0
                full_weight = torch.cat(gathered_weights, dim=concat_dim)
            else:
                # Replicated case: just take the first (they should be identical)
                full_weight = gathered_weights[0]
        else:
            full_weight = gathered_weights[0]

        return {self.hf_param: full_weight}

    def _replicate_to_tp_ranks(self, tensor: torch.Tensor) -> torch.Tensor:
        """Replicate tensor to all TP ranks."""
        return self.broadcast_tensor_to_tp_ranks(tensor, src_rank=0)

    def _column_parallel_scatter(self, tensor: torch.Tensor) -> torch.Tensor:
        """Scatter tensor for column parallel (split dim 0)."""
        tp = self.tp_size
        if tp == 1:
            return tensor

        # Split tensor along dimension 0
        chunks = torch.chunk(tensor, tp, dim=0)
        out_shape = list(tensor.shape)
        out_shape[0] = chunks[0].shape[0]

        return self.scatter_to_tp_ranks(
            list(chunks), output_shape=torch.Size(out_shape), dtype=tensor.dtype, device=tensor.device
        )


class AdapterQKVMapping(MegatronParamMapping[Dict[str, torch.Tensor]]):
    """Mapping for fused QKV adapter weights to canonical Q, K, V adapter weights.

    This mapping handles the conversion between:
    - Megatron fused QKV adapters (single linear_qkv.adapter)
    - HuggingFace canonical adapters (separate q_proj, k_proj, v_proj adapters)

    Similar to QKVMapping but for adapter parameters specifically.
    """

    @classmethod
    def from_base_mapping(
        cls, base_mapping: QKVMapping, adapter_megatron_param: str, hf_suffix: str
    ) -> "AdapterQKVMapping":
        """Create AdapterQKVMapping from base QKVMapping.

        Args:
            base_mapping: Base QKVMapping to adapt
            adapter_megatron_param: Megatron adapter parameter name
            hf_suffix: HuggingFace parameter suffix

        Returns:
            New AdapterQKVMapping instance
        """
        adapter_hf_params = {k: v.replace(".weight", hf_suffix) for k, v in base_mapping.hf_param.items()}
        return cls(
            adapter_megatron_param, q=adapter_hf_params["q"], k=adapter_hf_params["k"], v=adapter_hf_params["v"]
        )

    def __init__(self, megatron_param: str, q: str, k: str, v: str):
        """Initialize adapter QKV mapping."""
        super().__init__(megatron_param, {"q": q, "k": k, "v": v})
        # Use base module for TP distribution (adapter follows base module parallelism)
        self._tp_mapping = AdapterAutoMapping(megatron_param, megatron_param)

    def hf_to_megatron(self, hf_weights: Dict[str, torch.Tensor], megatron_module: nn.Module) -> torch.Tensor:
        """Merge Q, K, V adapter weights into fused QKV adapter format."""
        if self.tp_rank == 0:
            from megatron.bridge.models.conversion.param_mapping import merge_qkv_weights

            config = self._get_config(megatron_module)

            # For adapter weights, we expect same shapes as base weights
            if hf_weights["q"].ndim == 1:
                # Bias case - use bias merge function
                from megatron.bridge.models.conversion.param_mapping import merge_qkv_biases

                merged = merge_qkv_biases(config, hf_weights["q"], hf_weights["k"], hf_weights["v"])
            else:
                # Weight case - use standard merge function
                merged = merge_qkv_weights(config, hf_weights["q"], hf_weights["k"], hf_weights["v"])
        else:
            merged = None

        # Delegate TP distribution to base mapping
        return self._tp_mapping.hf_to_megatron(merged, megatron_module)

    def megatron_to_hf(
        self, megatron_weights: Optional[torch.Tensor], megatron_module: Optional[nn.Module]
    ) -> Dict[str, torch.Tensor]:
        """Split fused QKV adapter weights into Q, K, V components."""
        # Get config from module or broadcast from owning rank
        if megatron_module is None:
            config = self.broadcast_obj_from_pp_rank(None)
        else:
            config = self._get_config(megatron_module)
            from megatron.bridge.models.conversion.utils import remove_non_pickleables

            config = remove_non_pickleables(config, max_depth=2)
            config = self.broadcast_obj_from_pp_rank(config)

        # Delegate TP/PP gathering
        packed_dict = self._tp_mapping.megatron_to_hf(megatron_weights, megatron_module)

        if not packed_dict:
            return {}

        packed_qkv = next(iter(packed_dict.values()))

        # Split using existing QKV splitting functions
        if packed_qkv.ndim == 1:
            from megatron.bridge.models.conversion.param_mapping import split_qkv_biases

            q, k, v = split_qkv_biases(config, packed_qkv)
        else:
            from megatron.bridge.models.conversion.param_mapping import split_qkv_weights

            q, k, v = split_qkv_weights(config, packed_qkv)

        return {
            self.hf_param["q"]: q,
            self.hf_param["k"]: k,
            self.hf_param["v"]: v,
        }

    def resolve(self, captures: Tuple[str, ...]) -> "MegatronParamMapping":
        """Return a new resolved AdapterQKVMapping instance."""
        resolved_megatron_param, resolved_hf_param = self._resolve_names(captures)

        return type(self)(
            resolved_megatron_param,
            resolved_hf_param["q"],
            resolved_hf_param["k"],
            resolved_hf_param["v"],
        )


class AdapterGatedMLPMapping(MegatronParamMapping[Dict[str, torch.Tensor]]):
    """Mapping for fused FC1 adapter weights to canonical gate/up adapter weights.

    This mapping handles the conversion between:
    - Megatron fused FC1 adapters (single linear_fc1.adapter)
    - HuggingFace canonical adapters (separate gate_proj, up_proj adapters)

    Similar to GatedMLPMapping but for adapter parameters specifically.
    """

    @classmethod
    def from_base_mapping(
        cls, base_mapping: GatedMLPMapping, adapter_megatron_param: str, hf_suffix: str
    ) -> "AdapterGatedMLPMapping":
        """Create AdapterGatedMLPMapping from base GatedMLPMapping.

        Args:
            base_mapping: Base GatedMLPMapping to adapt
            adapter_megatron_param: Megatron adapter parameter name
            hf_suffix: HuggingFace parameter suffix

        Returns:
            New AdapterGatedMLPMapping instance
        """
        adapter_hf_params = {k: v.replace(".weight", hf_suffix) for k, v in base_mapping.hf_param.items()}
        return cls(adapter_megatron_param, gate=adapter_hf_params["gate"], up=adapter_hf_params["up"])

    def __init__(self, megatron_param: str, gate: str, up: str):
        """Initialize adapter gated MLP mapping."""
        super().__init__(megatron_param, {"gate": gate, "up": up})
        self._tp_mapping = AdapterAutoMapping(megatron_param, megatron_param)

    def hf_to_megatron(self, hf_weights: Dict[str, torch.Tensor], megatron_module: nn.Module) -> torch.Tensor:
        """Merge gate and up adapter weights into fused FC1 adapter format."""
        # For single TP, just concatenate and return
        if self.tp_size == 1:
            return torch.cat([hf_weights["gate"], hf_weights["up"]], dim=0)

        # Get target parameter info from megatron module
        normalized_param = self._normalize_expert_param_name(self.megatron_param)
        from megatron.bridge.models.conversion.utils import get_module_and_param_from_name

        _, target_param = get_module_and_param_from_name(megatron_module, normalized_param)

        # On rank 0, split gate and up separately, then concatenate corresponding pieces
        if self.tp_rank == 0:
            gate = hf_weights["gate"]
            up = hf_weights["up"]

            # Verify shapes match
            assert gate.shape == up.shape, "Gate and up adapter weights must have the same shape"

            # Split gate and up separately along output dimension (dim 0)
            gate_splits = torch.chunk(gate, self.tp_size, dim=0)
            up_splits = torch.chunk(up, self.tp_size, dim=0)

            # Concatenate corresponding pieces: [gate_shard_i; up_shard_i] for each rank i
            splits = [torch.cat([gate_splits[i], up_splits[i]], dim=0) for i in range(self.tp_size)]
        else:
            splits = None

        # Scatter the concatenated shards to each rank
        return self.scatter_to_tp_ranks(
            splits,
            target_param.shape,
            target_param.dtype,
            target_param.device,
        )

    def megatron_to_hf(
        self, megatron_weights: Optional[torch.Tensor], megatron_module: Optional[nn.Module]
    ) -> Dict[str, torch.Tensor]:
        """Split fused FC1 adapter weights into gate and up components."""
        # Handle cross-PP broadcast
        megatron_weights = self.broadcast_from_pp_rank(megatron_weights)

        if megatron_weights is None:
            return {}

        # Handle TP gathering
        if self.tp_size == 1:
            # No TP, just split the concatenated tensor
            fused_adapter = megatron_weights
            gate, up = torch.chunk(fused_adapter, 2, dim=0)
        else:
            # Gather shards from all TP ranks
            gathered_shards = self.gather_from_tp_ranks(megatron_weights)

            # Split each shard back into gate and up parts
            gate_parts = []
            up_parts = []
            for shard in gathered_shards:
                # Each shard is [gate_shard; up_shard] concatenated along dim 0
                gate_shard, up_shard = torch.chunk(shard, 2, dim=0)
                gate_parts.append(gate_shard)
                up_parts.append(up_shard)

            # Concatenate all gate parts and all up parts separately
            gate = torch.cat(gate_parts, dim=0)
            up = torch.cat(up_parts, dim=0)

        if self.is_expert:
            gathered_gate_weights_dict = self.gather_from_ep_ranks(gate, megatron_module, self.hf_param["gate"])
            gathered_up_weights_dict = self.gather_from_ep_ranks(up, megatron_module, self.hf_param["up"])
            return {**gathered_gate_weights_dict, **gathered_up_weights_dict}

        return {self.hf_param["gate"]: gate, self.hf_param["up"]: up}

    def resolve(self, captures: Tuple[str, ...]) -> "MegatronParamMapping":
        """Return a new resolved AdapterGatedMLPMapping instance."""
        resolved_megatron_param, resolved_hf_param = self._resolve_names(captures)

        return type(self)(
            resolved_megatron_param,
            resolved_hf_param["gate"],
            resolved_hf_param["up"],
        )

    def _row_parallel_scatter(self, tensor: torch.Tensor, dim: int) -> torch.Tensor:
        """Scatter tensor for row parallel."""
        tp = self.tp_size
        if tp == 1:
            return tensor

        # Split tensor along specified dimension
        chunks = torch.chunk(tensor, tp, dim=dim)

        # Calculate output shape
        out_shape = list(tensor.shape)
        out_shape[dim] = chunks[0].shape[dim]

        return self.scatter_to_tp_ranks(
            list(chunks), output_shape=torch.Size(out_shape), dtype=tensor.dtype, device=tensor.device
        )
