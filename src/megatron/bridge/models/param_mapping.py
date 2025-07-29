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

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar, Union

import torch
import torch.distributed
import torch.nn as nn
from megatron.core import mpu
from megatron.core.transformer.transformer_config import TransformerConfig


WeightType = TypeVar("WeightType", torch.Tensor, Dict[str, torch.Tensor])


class MegatronParamMapping(ABC, Generic[WeightType]):
    """
    Abstract base class for weight conversion between Megatron and external formats.

    This class provides the foundation for all weight mappings, handling the complex
    conversions between Megatron-Core's distributed tensor formats and standard
    (typically HuggingFace) formats. Each concrete mapping implements specific
    transformation logic while inheriting common parallel communication patterns.

    Key responsibilities:
    - Format transformation (e.g., QKV merging/splitting, gated MLP handling)
    - Tensor parallel (TP) distribution and gathering across GPUs
    - Pipeline parallel (PP) broadcasting between pipeline stages
    - Wildcard pattern resolution for layer-wise mappings

    The mapping abstraction ensures that higher-level code doesn't need to know
    about the parallel topology or format differences - it just requests a
    conversion and the mapping handles all the complexity.

    Public helper methods for subclasses:
    - broadcast_from_pp_rank: Broadcast tensors across pipeline stages
    - broadcast_obj_from_pp_rank: Broadcast Python objects across PP ranks
    - broadcast_tensor_to_tp_ranks: Broadcast within TP group
    - scatter_to_tp_ranks: Distribute tensor shards to TP ranks
    - gather_from_tp_ranks: Collect tensor shards from TP ranks

    Example:
        .. code-block:: python

            class MyCustomMapping(MegatronParamMapping[torch.Tensor]):
                def hf_to_megatron(self, hf_weights, megatron_module):
                    # Custom transformation logic
                    transformed = hf_weights.t()  # Example: transpose
                    # Use helpers for distribution
                    return self.scatter_to_tp_ranks(...)

                def megatron_to_hf(self, megatron_weights, megatron_module):
                    # Broadcast from owning PP rank
                    weight = self.broadcast_from_pp_rank(megatron_weights)
                    # Gather from TP ranks and transform
                    gathered = self.gather_from_tp_ranks(weight)
                    return {"custom_weight": gathered[0].t()}
    """

    def __init__(self, megatron_param: str, hf_param: Union[str, Dict[str, str]]):
        """Initialize the weight mapping.

        Args:
            megatron_param (str): Megatron parameter name pattern (supports *
                wildcards).
            hf_param (Union[str, Dict[str, str]]): External format name pattern(s).
        """
        self.megatron_param = megatron_param
        self.hf_param = hf_param
        self._validate_patterns()

    def _resolve_names(self, captures: Tuple[str, ...]) -> Tuple[str, Union[str, Dict[str, str]]]:
        resolved_megatron_param = self.megatron_param
        for value in captures:
            resolved_megatron_param = resolved_megatron_param.replace("*", value, 1)

        if isinstance(self.hf_param, str):
            resolved_hf_param = self.hf_param
            for value in captures:
                resolved_hf_param = resolved_hf_param.replace("*", value, 1)
        else:
            resolved_hf_param = {}
            for k, v in self.hf_param.items():
                resolved_v = v
                for value in captures:
                    resolved_v = resolved_v.replace("*", value, 1)
                resolved_hf_param[k] = resolved_v

        return resolved_megatron_param, resolved_hf_param

    def resolve(self, captures: Tuple[str, ...]) -> "MegatronParamMapping":
        """Create a new mapping with resolved wildcards.

        This default implementation works for mappings with a
        (megatron_param, hf_param) constructor.

        Args:
            captures (Tuple[str, ...]): Captured wildcard values.

        Returns:
            MegatronParamMapping: A new mapping instance with resolved names.
        """
        resolved_megatron_param, resolved_hf_param = self._resolve_names(captures)
        return type(self)(resolved_megatron_param, resolved_hf_param)

    @abstractmethod
    def hf_to_megatron(
        self,
        hf_weights: WeightType,
        megatron_module: nn.Module,
    ) -> torch.Tensor:
        """Convert hf_weights TO Megatron format.

        This method handles:
        1. Format transformation (if needed)
        2. Tensor parallel distribution (if self.tp_size > 1)

        Args:
            hf_weights (WeightType): Source hf_weights in external format.
            megatron_module (nn.Module): Target Megatron module (for config
                access).

        Returns:
            torch.Tensor: Weight tensor ready for the current TP rank.
        """
        ...

    @abstractmethod
    def megatron_to_hf(
        self,
        megatron_weights: Optional[torch.Tensor],
        megatron_module: Optional[nn.Module],
    ) -> Dict[str, torch.Tensor]:
        """Convert weights FROM Megatron format.

        This method handles:
        1. Pipeline parallel broadcasting (if weight is on different PP rank)
        2. Tensor parallel gathering (if needed)
        3. Format transformation

        Args:
            megatron_weights (Optional[torch.Tensor]): Weight tensor from current
                rank (None if on different PP rank).
            megatron_module (Optional[nn.Module]): Module for config access
                (None if on different PP rank).

        Returns:
            Dict[str, torch.Tensor]: Converted weights (empty dict if not on
                TP rank 0).
        """
        ...

    def broadcast_from_pp_rank(self, tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Broadcast a tensor from the pipeline-parallel rank that owns it.

        Broadcasts to **all** PP ranks. This mirrors the behaviour of
        `broadcast_from_megatron_pp` in the original MMapping implementation and
        additionally keeps the tensor-parallel metadata (`tensor_model_parallel`,
        `partition_dim`) consistent on every rank.

        Args:
            tensor (Optional[torch.Tensor]): The local tensor if the current PP
                rank owns it. ``None`` otherwise.

        Returns:
            Optional[torch.Tensor]: The broadcasted tensor on every PP rank, or
                ``None`` if *no* PP rank owned the tensor (which indicates a bug
                in the calling code).
        """

        # Fast-path when we are not using pipeline parallelism.
        if self.pp_size == 1:
            return tensor

        # ------------------------------------------------------------------
        # 1.  Gather (shape, dtype, tensor_parallel flag, partition_dim) from
        #     every PP rank so that we can find the source rank.
        # ------------------------------------------------------------------
        if tensor is not None:
            shape = tensor.shape
            dtype = tensor.dtype
            tensor_parallel = getattr(tensor, "tensor_model_parallel", None)
            partition_dim = getattr(tensor, "partition_dim", None)
            tensor_spec = (shape, dtype, tensor_parallel, partition_dim)
        else:
            tensor_spec = None

        tensor_spec_output: list[Optional[tuple]] = [None] * self.pp_size
        torch.distributed.all_gather_object(tensor_spec_output, tensor_spec, group=self.pp_group)

        # ------------------------------------------------------------------
        # 2.  Identify the owning rank (the only rank with a non-None spec).
        # ------------------------------------------------------------------
        target_tensor_spec = None
        src_rank = None  # Rank *inside* the PP group.
        for rank, spec in enumerate(tensor_spec_output):
            if spec is not None:
                if target_tensor_spec is not None:
                    raise ValueError(f"Tensor exists on more than one PP rank. Found on ranks {src_rank} and {rank}.")
                target_tensor_spec = spec
                src_rank = rank

        if target_tensor_spec is None:
            # No rank had the tensor – this is an error in the caller.
            raise ValueError("Object must exist on at least one PP rank")

        # ------------------------------------------------------------------
        # 3.  Ensure every rank has an allocated tensor with the right shape
        #     and dtype before the broadcast.
        # ------------------------------------------------------------------
        if tensor is None:
            shape, dtype, tensor_parallel, partition_dim = target_tensor_spec
            # Use CPU by default, unless CUDA is available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            tensor = torch.empty(shape, dtype=dtype, device=device)
            if tensor_parallel is not None:
                tensor.tensor_model_parallel = tensor_parallel
            if partition_dim is not None:
                tensor.partition_dim = partition_dim

        # ------------------------------------------------------------------
        # 4.  Broadcast from the source PP rank to all other PP ranks.
        # ------------------------------------------------------------------
        global_src = torch.distributed.get_global_rank(group=self.pp_group, group_rank=src_rank)
        torch.distributed.broadcast(tensor, src=global_src, group=self.pp_group)

        return tensor

    def broadcast_obj_from_pp_rank(self, obj: Optional[Any]) -> Any:
        """Broadcast any Python object from the PP rank that owns it.

        This method is useful for broadcasting configuration objects or
        other metadata across pipeline parallel ranks.

        Args:
            obj (Optional[Any]): Object to broadcast (None on non-owning ranks).

        Returns:
            Any: Broadcasted object on all ranks.

        Raises:
            ValueError: If object exists on multiple ranks or no ranks.
        """
        if self.pp_size == 1:
            return obj

        # ------------------------------------------------------------------
        # 1. Gather presence flags from all PP ranks to find the source rank
        # ------------------------------------------------------------------
        has_obj = obj is not None
        obj_flags = [None] * self.pp_size
        torch.distributed.all_gather_object(obj_flags, has_obj, group=self.pp_group)

        # ------------------------------------------------------------------
        # 2. Identify the owning rank (the only rank with True flag)
        # ------------------------------------------------------------------
        src_rank = None  # Rank *inside* the PP group
        for rank, flag in enumerate(obj_flags):
            if flag:
                src_rank = rank

        if src_rank is None:
            raise ValueError("Object must exist on at least one PP rank")

        # ------------------------------------------------------------------
        # 3. Broadcast the object from the source rank to all ranks
        # ------------------------------------------------------------------
        if src_rank is None:
            raise ValueError("Could not determine source rank")

        # Use broadcast_object_list which is more robust than all_gather_object
        obj_list = [obj]
        pp_ranks = torch.distributed.get_process_group_ranks(self.pp_group)
        global_src = pp_ranks[src_rank]
        torch.distributed.broadcast_object_list(obj_list, src=global_src, group=self.pp_group)

        return obj_list[0]

    def broadcast_tensor_to_tp_ranks(self, tensor: torch.Tensor, src_rank: int = 0) -> torch.Tensor:
        """Broadcast a tensor to all TP ranks.

        Args:
            tensor (torch.Tensor): The tensor to broadcast.
            src_rank (int, optional): The source rank within the TP group.
                Defaults to 0.

        Returns:
            torch.Tensor: The broadcasted tensor.
        """
        if self.tp_size == 1:
            return tensor

        global_src = torch.distributed.get_global_rank(group=self.tp_group, group_rank=src_rank)
        torch.distributed.broadcast(tensor, src=global_src, group=self.tp_group)
        return tensor

    def scatter_to_tp_ranks(
        self,
        splits: Optional[List[torch.Tensor]],
        output_shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
        src_rank: int = 0,
    ) -> torch.Tensor:
        """Scatter tensor splits to TP ranks.

        Args:
            splits (Optional[List[torch.Tensor]]): A list of tensor shards to
                scatter. Only rank `src_rank` needs this.
            output_shape (torch.Size): The shape of the output tensor on each rank.
            dtype (torch.dtype): The data type of the output tensor.
            device (torch.device): The device for the output tensor.
            src_rank (int, optional): The source rank for the scatter operation.
                Defaults to 0.

        Returns:
            torch.Tensor: The scattered tensor shard on the current rank.
        """
        if self.tp_size == 1:
            return splits[0].to(device=device) if splits else None

        output = torch.empty(output_shape, dtype=dtype, device=device)
        global_src = torch.distributed.get_global_rank(group=self.tp_group, group_rank=src_rank)

        scatter_list = None
        if self.tp_rank == src_rank and splits:
            scatter_list = [s.to(device=device) for s in splits]

        torch.distributed.scatter(
            output,
            scatter_list,
            src=global_src,
            group=self.tp_group,
        )
        return output

    def gather_from_tp_ranks(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """Gather tensors from all TP ranks.

        Args:
            tensor (torch.Tensor): The tensor shard to be gathered from the
                current rank.

        Returns:
            List[torch.Tensor]: A list of tensor shards from all TP ranks.
        """
        if self.tp_size == 1:
            return [tensor]

        gathered = [torch.empty_like(tensor) for _ in range(self.tp_size)]
        torch.distributed.all_gather(gathered, tensor, group=self.tp_group)
        return gathered

    def _validate_patterns(self):
        """Validate wildcard consistency between patterns."""
        megatron_param_wildcards = self.megatron_param.count("*")
        if isinstance(self.hf_param, str):
            hf_param_wildcards = self.hf_param.count("*")
            if megatron_param_wildcards != hf_param_wildcards:
                raise ValueError(
                    f"Wildcard count mismatch: megatron_param='{self.megatron_param}' has "
                    f"{megatron_param_wildcards} wildcards, hf_param='{self.hf_param}' has {hf_param_wildcards}"
                )
        else:
            for key, pattern in self.hf_param.items():
                hf_param_wildcards = pattern.count("*")
                if megatron_param_wildcards != hf_param_wildcards:
                    raise ValueError(
                        f"Wildcard count mismatch: megatron_param='{self.megatron_param}' has "
                        f"{megatron_param_wildcards} wildcards, hf_param['{key}']='{pattern}' has {hf_param_wildcards}"
                    )

    def _get_config(self, module: nn.Module) -> Any:
        """Extract configuration from module hierarchy."""
        current = module
        while current is not None:
            if hasattr(current, "config"):
                return current.config
            # Try parent module
            if hasattr(current, "_parent"):
                current = current._parent
            else:
                # Walk up the module tree
                for parent_module in module.modules():
                    for child_name, child_module in parent_module.named_children():
                        if child_module is current:
                            current = parent_module
                            break
                    else:
                        continue
                    break
                else:
                    current = None

        raise ValueError(
            f"Could not find config in module hierarchy for {module.__class__.__name__}. "
            f"Ensure the module or its parent has a 'config' attribute."
        )

    @property
    def tp_size(self) -> int:
        """Get tensor parallel world size from parallel_state."""
        return mpu.get_tensor_model_parallel_world_size()

    @property
    def tp_rank(self) -> int:
        """Get current tensor parallel rank from parallel_state."""
        return mpu.get_tensor_model_parallel_rank()

    @property
    def tp_group(self) -> Optional[torch.distributed.ProcessGroup]:
        """Get tensor parallel process group from parallel_state."""
        if self.tp_size > 1:
            return mpu.get_tensor_model_parallel_group()
        return None

    @property
    def pp_size(self) -> int:
        """Get pipeline parallel world size."""
        return mpu.get_pipeline_model_parallel_world_size()

    @property
    def pp_rank(self) -> int:
        """Get current pipeline parallel rank."""
        return mpu.get_pipeline_model_parallel_rank()

    @property
    def pp_group(self) -> Optional[torch.distributed.ProcessGroup]:
        """Get pipeline parallel process group."""
        if self.pp_size > 1:
            return mpu.get_pipeline_model_parallel_group()
        return None


class DirectMapping(MegatronParamMapping[torch.Tensor]):
    """Direct 1:1 weight mapping with no transformation or tensor parallelism."""

    def hf_to_megatron(
        self,
        hf_weights: torch.Tensor,
        megatron_module: nn.Module,
    ) -> torch.Tensor:
        """Direct copy - no transformation or distribution."""
        return hf_weights

    def megatron_to_hf(
        self,
        megatron_weights: Optional[torch.Tensor],
        megatron_module: Optional[nn.Module],
    ) -> Dict[str, torch.Tensor]:
        """Direct copy with PP broadcast."""
        # Handle cross-PP broadcast
        megatron_weights = self.broadcast_from_pp_rank(megatron_weights)

        if megatron_weights is None:
            return {}

        return {str(self.hf_param): megatron_weights}


class ColumnParallelMapping(MegatronParamMapping[torch.Tensor]):
    """
    Mapping for column-parallel linear and embedding weights.

    Column-parallel layers in Megatron split the output dimension across tensor
    parallel ranks. This is used for layers where each rank computes a portion
    of the output features independently, such as:
    - Embedding layers (split vocabulary)
    - Linear layers producing hidden states (e.g., QKV projections, MLP up projections)

    The weight matrix is partitioned along dimension 0 (rows), so each TP rank
    holds a subset of output features while maintaining all input features.

    **Sharding pattern**
    -   Original weight: `[output_features, input_features]`
    -   Rank 0: `[output_features/tp_size, input_features]`
    -   Rank 1: `[output_features/tp_size, input_features]`
    -   ...

    **Forward path (HuggingFace → Megatron)**
    1.  Validate divisibility: output dimension must be divisible by tp_size
    2.  Split: Chunk tensor along dim 0 into tp_size equal parts
    3.  Scatter: Distribute chunks to respective TP ranks

    **Reverse path (Megatron → HuggingFace)**
    1.  Broadcast: Ensure all PP ranks have the tensor
    2.  Gather: Collect chunks from all TP ranks
    3.  Concatenate: Reassemble along dim 0 on rank 0

    Example:
        .. code-block:: python

            # For a weight of shape [4096, 1024] with tp_size=4:
            # Each rank gets [1024, 1024] after column-parallel split
            mapping = ColumnParallelMapping("linear.weight", "transformer.linear.weight")
            megatron_weights = mapping.hf_to_megatron(hf_weight, megatron_module)
            # megatron_weights.shape = [1024, 1024] on each rank

    Note:
        This mapping also handles bias terms, which are 1D tensors split
        along their only dimension following the same pattern.
    """

    def _get_target_param_and_shape(self, megatron_module: nn.Module) -> tuple[torch.Tensor, tuple]:
        """Get the target parameter and its shape based on the parameter name."""
        param_name_lower = self.megatron_param.lower()

        # Define parameter mapping: (param_name, expected_ndim)
        param_configs = [
            ("bias", 1),
            ("weight", 2),
        ]

        for param_name, expected_ndim in param_configs:
            if param_name in param_name_lower:
                if hasattr(megatron_module, param_name):
                    target_param = getattr(megatron_module, param_name)
                    if isinstance(target_param, torch.Tensor) and target_param.ndim == expected_ndim:
                        return target_param, target_param.shape
                    else:
                        raise ValueError(
                            f"Parameter {param_name} exists but has wrong type or dimensions (expected ndim == {expected_ndim}, got {target_param.ndim if isinstance(target_param, torch.Tensor) else 'not a tensor'})"
                        )
                else:
                    raise ValueError(
                        f"Parameter name suggests {param_name} but module {megatron_module} has no {param_name}"
                    )

        raise ValueError(f"Could not determine parameter type for {self.megatron_param}")

    def hf_to_megatron(
        self,
        hf_weights: torch.Tensor,
        megatron_module: nn.Module,
    ) -> torch.Tensor:
        """Split weight along dim 0 and distribute to TP ranks."""
        if self.tp_size == 1:
            return hf_weights

        target_param, output_shape = self._get_target_param_and_shape(megatron_module)

        # On rank 0, check for divisibility and split
        if self.tp_rank == 0:
            if hf_weights is None:
                raise ValueError("hf_weights should not be None on rank 0")

            # For bias (1D), we still split along dim 0
            # For weight (2D), we split along dim 0 (output dimension)
            full_size = hf_weights.shape[0]
            if full_size % self.tp_size != 0:
                raise ValueError(f"Cannot evenly split dimension 0 size {full_size} across {self.tp_size} TP ranks")
            splits = torch.chunk(hf_weights, self.tp_size, dim=0)

        else:
            splits = None

        # Scatter to all ranks. Each rank gets its sharded shape from its module.
        return self.scatter_to_tp_ranks(
            splits,
            output_shape,
            target_param.dtype,
            target_param.device,
        )

    def megatron_to_hf(
        self,
        megatron_weights: Optional[torch.Tensor],
        megatron_module: Optional[nn.Module],
    ) -> Dict[str, torch.Tensor]:
        """Gather from all TP ranks and concatenate."""
        # Handle cross-PP broadcast
        megatron_weights = self.broadcast_from_pp_rank(megatron_weights)

        if megatron_weights is None:
            return {}

        if self.tp_size == 1:
            return {str(self.hf_param): megatron_weights}

        # Gather from all TP ranks
        gathered = self.gather_from_tp_ranks(megatron_weights)

        # Only rank 0 concatenates and returns
        if self.tp_rank == 0:
            full_weight = torch.cat(gathered, dim=0)
            return {str(self.hf_param): full_weight}
        return {}


class RowParallelMapping(MegatronParamMapping[torch.Tensor]):
    """Mapping for **row-parallel** linear weights.

    Megatron shards row-parallel tensors along **dimension 1** (the *input*
    dimension of a linear layer).

    **Forward path (external → Megatron)**
    1.  Rank 0 validates that the *second* dimension is divisible by `tp_size`.
    2.  Rank 0 splits the tensor with `torch.chunk(..., dim=1)` producing
        `tp_size` equally-sized shards.
    3.  The shards are **scattered** so that every TP rank receives exactly one
        shard matching the shape of its local Megatron parameter.

    **Reverse path (Megatron → external)**
    1.  The local Megatron parameter (which may live on any PP rank) is
        broadcast to all PP ranks so that the gather step can be collective.
    2.  All TP ranks **gather** their shard.
    3.  Rank 0 concatenates the gathered list along dim 1 to reconstruct the
        original unsharded weight and emits it under the external (HF) name.
    """

    def hf_to_megatron(
        self,
        hf_weights: torch.Tensor,
        megatron_module: nn.Module,
    ) -> torch.Tensor:
        """Split weight along dim 1 and distribute to TP ranks."""
        if self.tp_size == 1:
            return hf_weights

        # On rank 0, check for divisibility and split
        if self.tp_rank == 0:
            full_size = hf_weights.shape[1]
            if full_size % self.tp_size != 0:
                raise ValueError(f"Cannot evenly split dimension 1 size {full_size} across {self.tp_size} TP ranks")
            splits = torch.chunk(hf_weights, self.tp_size, dim=1)
        else:
            splits = None

        # Scatter to all ranks. Each rank gets its sharded shape from its module.
        output_shape = megatron_module.weight.shape
        return self.scatter_to_tp_ranks(
            splits,
            output_shape,
            megatron_module.weight.dtype,
            megatron_module.weight.device,
        )

    def megatron_to_hf(
        self,
        megatron_weights: Optional[torch.Tensor],
        megatron_module: Optional[nn.Module],
    ) -> Dict[str, torch.Tensor]:
        """Gather from all TP ranks and concatenate."""
        # Handle cross-PP broadcast
        megatron_weights = self.broadcast_from_pp_rank(megatron_weights)

        if megatron_weights is None:
            return {}

        if self.tp_size == 1:
            merged = megatron_weights
        else:
            gathered = self.gather_from_tp_ranks(megatron_weights)
            merged = torch.cat(gathered, dim=1)

        if self.tp_rank == 0:
            return {str(self.hf_param): merged}
        else:
            return {}


class ReplicatedMapping(MegatronParamMapping[torch.Tensor]):
    """Mapping for weights that are **fully replicated** across TP ranks.

    Examples: layer-norm scales, biases, router weights in MoE, etc.

    These tensors exist in exactly the same form on *every* TP rank, so the
    mapping logic is trivial – but we still need to broadcast across TP ranks
    during *load* (HF → Megatron) and ensure we do **not** emit duplicates
    during *export* (Megatron → HF).
    """

    def hf_to_megatron(
        self,
        hf_weights: torch.Tensor,
        megatron_module: nn.Module,
    ) -> torch.Tensor:
        """Replicate weight to all TP ranks."""
        target_device = megatron_module.weight.device
        if self.tp_size == 1:
            return hf_weights.to(device=target_device)

        # All ranks need the full weight
        if self.tp_rank == 0:
            hf_weights = hf_weights.to(device=target_device)
        else:
            # Create empty tensor of correct shape
            hf_weights = torch.empty_like(megatron_module.weight)

        # Broadcast from rank 0 to all TP ranks
        return self.broadcast_tensor_to_tp_ranks(hf_weights, src_rank=0)

    def megatron_to_hf(
        self,
        megatron_weights: Optional[torch.Tensor],
        megatron_module: Optional[nn.Module],
    ) -> Dict[str, torch.Tensor]:
        """Return weight only from rank 0 to avoid duplication."""
        # Handle cross-PP broadcast
        megatron_weights = self.broadcast_from_pp_rank(megatron_weights)

        if megatron_weights is None:
            return {}

        # For replicated weights, only rank 0 returns to avoid duplicates
        if self.tp_rank == 0:
            return {str(self.hf_param): megatron_weights}
        return {}


class TPAwareMapping(MegatronParamMapping[torch.Tensor]):
    """
    Smart mapping that automatically detects and applies the correct parallelism strategy.

    This mapping eliminates the need to manually specify whether a layer is
    column-parallel, row-parallel, or replicated. It examines the Megatron
    module at runtime and delegates to the appropriate specialized mapping.

    **Detection strategy**
    1. Check module class name against a registry of known types
    2. If unknown, examine module attributes (tensor_model_parallel, partition_dim)
    3. Delegate to appropriate mapping: ColumnParallel, RowParallel, or Replicated

    This abstraction is particularly useful for model-agnostic code where you
    don't know the parallelism type ahead of time, or when working with models
    that mix different parallelism strategies.

    **Built-in module recognition**
    -   Column-parallel: `ColumnParallelLinear`, `VocabParallelEmbedding`, etc.
    -   Row-parallel: `RowParallelLinear`, `TERowParallelLinear`
    -   Replicated: `LayerNorm`, `RMSNorm`, and other normalization layers

    Example:
        .. code-block:: python

            # Automatically handles any weight type
            mapping = TPAwareMapping(
                megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                hf_param="model.layers.*.mlp.gate_proj.weight"
            )

            # Works with column-parallel layers
            megatron_weights = mapping.hf_to_megatron(hf_weight, column_parallel_module)

            # Also works with normalization layers
            norm_weight = mapping.hf_to_megatron(hf_norm, layer_norm_module)

            # Register custom module types
            TPAwareMapping.register_module_type("MyCustomLinear", "column")

    Note:
        If the parallelism type cannot be determined, the mapping will raise
        a descriptive error suggesting how to fix the issue.
    """

    # Module type registry
    _MODULE_TYPE_REGISTRY: Dict[str, set] = {
        "column": {
            "ColumnParallelLinear",
            "TEColumnParallelLinear",
            "TELayerNormColumnParallelLinear",
            "VocabParallelEmbedding",
        },
        "row": {
            "RowParallelLinear",
            "TERowParallelLinear",
        },
        "replicated": {
            # Normalization layers
            "TENorm",
            "FusedLayerNorm",
            "WrappedTorchNorm",
            "LayerNorm",
            "RMSNorm",
            "L2Norm",
            # Other non-parallel modules
            "IdentityOp",
            "DotProductAttention",
            "TEDotProductAttention",
        },
    }

    @classmethod
    def register_module_type(cls, module_name: str, parallelism_type: str):
        """Register a new module type for automatic parallelism detection.

        Args:
            module_name (str): The name of the module class (e.g.,
                'MyColumnLinear').
            parallelism_type (str): One of 'column', 'row', or 'replicated'.
        """
        if parallelism_type not in cls._MODULE_TYPE_REGISTRY:
            raise ValueError(
                f"Invalid parallelism_type '{parallelism_type}'. "
                f"Must be one of {list(cls._MODULE_TYPE_REGISTRY.keys())}"
            )
        cls._MODULE_TYPE_REGISTRY[parallelism_type].add(module_name)

    def __init__(self, megatron_param: str, hf_param: str):
        """Initialize TP-aware mapping."""
        super().__init__(megatron_param, hf_param)

        # Create delegate mappings
        self._column_mapping = ColumnParallelMapping(megatron_param, hf_param)
        self._row_mapping = RowParallelMapping(megatron_param, hf_param)
        self._replicated_mapping = ReplicatedMapping(megatron_param, hf_param)

    def _detect_parallelism_type(self, module: nn.Module) -> str:
        """Detect parallelism type from module."""
        module_type = type(module).__name__

        # Check registry first
        for parallelism, types in self._MODULE_TYPE_REGISTRY.items():
            if module_type in types:
                return parallelism

        # Fallback to inspecting module attributes
        if hasattr(module, "tensor_model_parallel"):
            if not module.tensor_model_parallel:
                return "replicated"

            # Check partition dimension
            partition_dim = getattr(module, "partition_dim", None)
            if partition_dim == 0:
                return "column"
            elif partition_dim == 1:
                return "row"

        # Fallback for normalization layers
        if any(norm in module_type for norm in ["Norm", "Normalization"]):
            return "replicated"

        # Cannot determine - raise informative error
        known_types = {p: sorted(list(t)) for p, t in self._MODULE_TYPE_REGISTRY.items()}

        raise ValueError(
            f"Cannot determine parallelism type for module '{module_type}' "
            f"at weight '{self.megatron_param}'.\n"
            f"Please use an explicit mapping type (e.g., ColumnParallelMapping) "
            f"or register the module type using:\n"
            f"  TPAwareMapping.register_module_type('{module_type}', 'column|row|replicated')\n\n"
            f"Currently known module types:\n{json.dumps(known_types, indent=2)}"
        )

    def hf_to_megatron(
        self,
        hf_weights: torch.Tensor,
        megatron_module: nn.Module,
    ) -> torch.Tensor:
        """Delegate to appropriate mapping based on module type."""
        parallelism_type = self._detect_parallelism_type(megatron_module)

        if parallelism_type == "column":
            return self._column_mapping.hf_to_megatron(hf_weights, megatron_module)
        elif parallelism_type == "row":
            return self._row_mapping.hf_to_megatron(hf_weights, megatron_module)
        elif parallelism_type == "replicated":
            return self._replicated_mapping.hf_to_megatron(hf_weights, megatron_module)

        raise ValueError(f"Unknown parallelism type: {parallelism_type}")

    def megatron_to_hf(
        self,
        megatron_weights: Optional[torch.Tensor],
        megatron_module: Optional[nn.Module],
    ) -> Dict[str, torch.Tensor]:
        """Delegate to appropriate mapping based on module type."""
        # Need to determine type even if module is None (different PP rank)
        if megatron_module is not None:
            parallelism_type = self._detect_parallelism_type(megatron_module)
            # Broadcast to other ranks
            parallelism_type = self.broadcast_obj_from_pp_rank(parallelism_type)
        else:
            # Receive from owning rank
            parallelism_type = self.broadcast_obj_from_pp_rank(None)

        if parallelism_type == "column":
            return self._column_mapping.megatron_to_hf(megatron_weights, megatron_module)
        elif parallelism_type == "row":
            return self._row_mapping.megatron_to_hf(megatron_weights, megatron_module)
        elif parallelism_type == "replicated":
            return self._replicated_mapping.megatron_to_hf(megatron_weights, megatron_module)
        else:
            raise ValueError(f"Unknown parallelism type: {parallelism_type}")


class QKVMapping(MegatronParamMapping[Dict[str, torch.Tensor]]):
    """
    Mapping for interleaved Query/Key/Value attention projection weights.

    This mapping handles the conversion between separate Q, K, V matrices used in
    standard transformers and Megatron's optimized interleaved format. The
    interleaving pattern groups queries with their corresponding key-value pairs
    to maximize GEMM efficiency during attention computation.

    **External format (HuggingFace)**
    -   Separate tensors: `q_proj`, `k_proj`, `v_proj`
    -   Each of shape `[hidden_size, hidden_size]` or `[hidden_size, head_dim * num_heads]`

    **Megatron format**
    -   Single interleaved tensor following grouped query attention (GQA) pattern
    -   Interleaving order: `[q1...qn, k1, v1, q1...qn, k2, v2, ...]`
    -   Where `n = num_attention_heads / num_query_groups`

    **Key features**
    1.  Format conversion: Handles merging/splitting with proper interleaving
    2.  Grouped Query Attention: Supports different numbers of Q and KV heads
    3.  Tensor parallelism: Delegates to TPAwareMapping for distribution

    Example:
        .. code-block:: python

            # Create mapping for attention weights
            mapping = QKVMapping(
                megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                q="model.layers.*.self_attn.q_proj.weight",
                k="model.layers.*.self_attn.k_proj.weight",
                v="model.layers.*.self_attn.v_proj.weight"
            )

            # Convert from HuggingFace to Megatron
            qkv_weights = {"q": q_tensor, "k": k_tensor, "v": v_tensor}
            megatron_qkv = mapping.hf_to_megatron(qkv_weights, megatron_module)

            # Convert from Megatron to HuggingFace
            hf_weights = mapping.megatron_to_hf(megatron_qkv, megatron_module)
            # Returns: {"q_proj.weight": ..., "k_proj.weight": ..., "v_proj.weight": ...}

    Note:
        This mapping automatically handles both regular multi-head attention
        (same number of Q, K, V heads) and grouped query attention (fewer
        KV heads than Q heads) based on the model configuration.
    """

    def __init__(self, megatron_param: str, q: str, k: str, v: str):
        """Initialize QKV mapping.

        Args:
            megatron_param (str): Megatron QKV parameter name pattern.
            q (str): Query weight name pattern.
            k (str): Key weight name pattern.
            v (str): Value weight name pattern.
        """
        super().__init__(megatron_param, {"q": q, "k": k, "v": v})
        # Delegate all tensor-parallel logic to the smart TP-aware mapping so we
        # do not hard-code the assumption that QKV projections are column-parallel.
        # This keeps the format-handling (merge/split) concerns separate from
        # TP/PP distribution mechanics.
        self._tp_mapping = TPAwareMapping(megatron_param, megatron_param)

    def hf_to_megatron(
        self,
        hf_weights: Dict[str, torch.Tensor],
        megatron_module: nn.Module,
    ) -> torch.Tensor:
        """Merge Q, K, V into interleaved format and distribute."""
        if self.tp_rank == 0:
            config = self._get_config(megatron_module)

            # Check if we're dealing with biases (1D tensors) or hf_weights (2D tensors)
            if hf_weights["q"].ndim == 1:
                # For biases, use the bias-specific merge function
                merged = merge_qkv_biases(config, hf_weights["q"], hf_weights["k"], hf_weights["v"])
            else:
                # For hf_weights, use the standard merge function
                merged = merge_qkv_weights(config, hf_weights["q"], hf_weights["k"], hf_weights["v"])
        else:
            merged = None

        # Delegate the actual sharding/broadcasting to the TP-aware mapping.
        return self._tp_mapping.hf_to_megatron(merged, megatron_module)

    def megatron_to_hf(
        self,
        megatron_weights: Optional[torch.Tensor],
        megatron_module: Optional[nn.Module],
    ) -> Dict[str, torch.Tensor]:
        """Gather QKV shards and split into Q, K, V."""
        # ------------------------------------------------------------------
        # Broadcast / retrieve the transformer configuration so that every PP
        # rank (also the ones that will early-return) participates in the
        # collective communication.
        # ------------------------------------------------------------------
        if megatron_module is None:
            config = self.broadcast_obj_from_pp_rank(None)
        else:
            cfg_local = self._get_config(megatron_module)
            config = self.broadcast_obj_from_pp_rank(cfg_local)

        # Delegate TP/PP gathering.
        packed_dict = self._tp_mapping.megatron_to_hf(megatron_weights, megatron_module)

        if not packed_dict:
            return {}

        packed_qkv = next(iter(packed_dict.values()))

        # Check if we're dealing with biases (1D) or weights (2D)
        if packed_qkv.ndim == 1:
            # Split biases
            q, k, v = split_qkv_biases(config, packed_qkv)
        else:
            # Split weights
            q, k, v = split_qkv_weights(config, packed_qkv)

        return {
            self.hf_param["q"]: q,
            self.hf_param["k"]: k,
            self.hf_param["v"]: v,
        }

    def resolve(self, captures: Tuple[str, ...]) -> "MegatronParamMapping":
        """Return a new *resolved* QKVMapping instance."""
        resolved_megatron_param, resolved_hf_param = self._resolve_names(captures)

        return type(self)(
            resolved_megatron_param,
            resolved_hf_param["q"],
            resolved_hf_param["k"],
            resolved_hf_param["v"],
        )


class GatedMLPMapping(MegatronParamMapping[Dict[str, torch.Tensor]]):
    r"""Mapping for **gated-MLP** projection weights (SwiGLU / GeGLU).

    Checkpoint formats expose two independent matrices:

    -   **G** – gate projection
    -   **U** – up projection

    Megatron concatenates them row-wise (`[G; U]`) so that a single GEMM can
    produce both activations.

    **Responsibilities handled by this mapping**
    1.  **Concatenate / split** – convert between `[G; U]` (Megatron) and the
        separate `{G, U}` matrices (external).
    2.  **Tensor-parallel distribution** – correctly splits gate and up
        projections separately before concatenating corresponding shards,
        ensuring each TP rank gets the proper [gate_shard; up_shard] format.

    **TP Distribution Strategy**
    For tensor parallelism, this mapping:
    - Splits gate and up matrices separately along output dimension (dim 0)
    - Concatenates corresponding shards: [gate_shard_i; up_shard_i] for rank i
    - This ensures each rank's concatenated tensor matches the expected shape
    """

    def __init__(self, megatron_param: str, gate: str, up: str):
        """Initialize gated MLP mapping.

        Args:
            megatron_param (str): Megatron MLP parameter name pattern.
            gate (str): Gate projection weight name pattern.
            up (str): Up projection weight name pattern.
        """
        super().__init__(megatron_param, {"gate": gate, "up": up})

    def _get_target_param_and_shape(self, megatron_module: nn.Module) -> tuple[torch.Tensor, tuple]:
        """Get the target parameter and its shape based on the parameter name."""
        param_name_lower = self.megatron_param.lower()

        # Define parameter mapping: (param_name, expected_ndim)
        param_configs = [
            ("bias", 1),
            ("weight", 2),
        ]

        for param_name, expected_ndim in param_configs:
            if param_name in param_name_lower:
                if hasattr(megatron_module, param_name):
                    target_param = getattr(megatron_module, param_name)
                    if isinstance(target_param, torch.Tensor) and target_param.ndim == expected_ndim:
                        return target_param, target_param.shape
                    else:
                        raise ValueError(
                            f"Parameter {param_name} exists but has wrong type or dimensions (expected ndim == {expected_ndim}, got {target_param.ndim if isinstance(target_param, torch.Tensor) else 'not a tensor'})"
                        )
                else:
                    raise ValueError(
                        f"Parameter name suggests {param_name} but module {megatron_module} has no {param_name}"
                    )

        raise ValueError(f"Could not determine parameter type for {self.megatron_param}")

    def hf_to_megatron(
        self,
        hf_weights: Dict[str, torch.Tensor],
        megatron_module: nn.Module,
    ) -> torch.Tensor:
        """Split gate and up separately, then concatenate corresponding shards."""
        # For single TP, just concatenate and return
        if self.tp_size == 1:
            return torch.cat([hf_weights["gate"], hf_weights["up"]], dim=0)

        # Get target parameter info from megatron module
        target_param, output_shape = self._get_target_param_and_shape(megatron_module)

        # On rank 0, split gate and up separately, then concatenate corresponding pieces
        if self.tp_rank == 0:
            gate = hf_weights["gate"]
            up = hf_weights["up"]

            # Verify shapes match
            assert gate.shape == up.shape, "Gate and up weights must have the same shape"

            # Check divisibility for TP splitting
            gate_output_size = gate.shape[0]
            if gate_output_size % self.tp_size != 0:
                raise ValueError(
                    f"Cannot evenly split gate dimension 0 size {gate_output_size} across {self.tp_size} TP ranks"
                )

            # Split gate and up separately along output dimension (dim 0)
            # This works for both bias (1D) and weight (2D) tensors
            gate_splits = torch.chunk(gate, self.tp_size, dim=0)
            up_splits = torch.chunk(up, self.tp_size, dim=0)

            # Concatenate corresponding pieces: [gate_shard_i; up_shard_i] for each rank i
            splits = [torch.cat([gate_splits[i], up_splits[i]], dim=0) for i in range(self.tp_size)]
        else:
            splits = None

        # Scatter the concatenated shards to each rank
        return self.scatter_to_tp_ranks(
            splits,
            output_shape,
            target_param.dtype,
            target_param.device,
        )

    def megatron_to_hf(
        self,
        megatron_weights: Optional[torch.Tensor],
        megatron_module: Optional[nn.Module],
    ) -> Dict[str, torch.Tensor]:
        """Gather concatenated shards and split into gate and up."""
        # Handle cross-PP broadcast first
        megatron_weights = self.broadcast_from_pp_rank(megatron_weights)

        if megatron_weights is None:
            return {}

        # Handle TP gathering
        if self.tp_size == 1:
            # No TP, just split the concatenated tensor
            fused_mlp = megatron_weights
        else:
            # Gather shards from all TP ranks
            gathered_shards = self.gather_from_tp_ranks(megatron_weights)

            if self.tp_rank == 0:
                # Split each shard back into gate and up parts
                gate_parts = []
                up_parts = []
                for shard in gathered_shards:
                    # Each shard is [gate_shard; up_shard] concatenated along dim 0
                    # This works for both bias (1D) and weight (2D) tensors
                    gate_shard, up_shard = torch.chunk(shard, 2, dim=0)
                    gate_parts.append(gate_shard)
                    up_parts.append(up_shard)

                # Concatenate all gate parts and all up parts separately
                full_gate = torch.cat(gate_parts, dim=0)
                full_up = torch.cat(up_parts, dim=0)

                # Concatenate gate and up to get the full tensor
                fused_mlp = torch.cat([full_gate, full_up], dim=0)
            else:
                return {}

        # Only rank 0 returns the split weights/biases
        if self.tp_rank == 0:
            # Split the concatenated tensor in half along dim 0
            # This works for both bias (1D) and weight (2D) tensors
            gate, up = torch.chunk(fused_mlp, 2, dim=0)

            return {self.hf_param["gate"]: gate, self.hf_param["up"]: up}
        else:
            return {}

    def resolve(self, captures: Tuple[str, ...]) -> "MegatronParamMapping":
        """Return a new *resolved* GatedMLPMapping instance."""
        resolved_megatron_param, resolved_hf_param = self._resolve_names(captures)

        return type(self)(
            resolved_megatron_param,
            resolved_hf_param["gate"],
            resolved_hf_param["up"],
        )


class MOEMapping(MegatronParamMapping[torch.Tensor]):
    """Mapping for **Mixture of Experts (MoE)** weight distribution.

    MoE models distribute expert weights across Expert Parallel (EP) ranks.
    Each EP rank owns a subset of experts, and this mapping handles the
    EP distribution while delegating TP operations to TPAwareMapping.

    **Key features handled by this mapping**
    1.  **Expert parallel distribution** – different experts on different EP ranks
    2.  **Dynamic expert IDs** – weight names contain expert indices as wildcards
    3.  **Cross-EP communication** – broadcasting weights from owning EP rank
    4.  **TP delegation** – all tensor parallel ops handled by TPAwareMapping

    **Weight naming convention**
    -   Megatron: `"mlp.experts.linear_fc1.weight*"`  (where `*` is the expert ID)
    -   External: `"model.layers.*.mlp.experts.*.gate_proj.weight"`

    The expert ID wildcard is resolved based on EP rank and configuration.
    """

    def __init__(self, megatron_param: str, hf_param: str):
        """Initialize MoE weight mapping.

        Args:
            megatron_param (str): Megatron expert weight pattern (expert ID as last
                wildcard).
            hf_param (str): External weight pattern (expert ID as last wildcard).
        """
        super().__init__(megatron_param, hf_param)

        # Create a TP mapping for handling tensor parallelism
        # This will be used after EP distribution is resolved
        self._tp_mapping = TPAwareMapping(megatron_param, hf_param)

    def hf_to_megatron(
        self,
        hf_weights: torch.Tensor,
        megatron_module: nn.Module,
    ) -> torch.Tensor:
        """
        Handle EP distribution then delegate TP operations.

        This method:
        1. Determines which EP rank should own this expert
        2. Ensures only the owning rank has the weight
        3. Delegates TP distribution to TPAwareMapping
        """
        config = self._get_config(megatron_module)

        if self.ep_size == 1:
            # No EP distribution, just delegate to TP mapping
            return self._tp_mapping.hf_to_megatron(hf_weights, megatron_module)

        # Extract expert ID from the resolved parameter name
        expert_id = self._get_expert_id_from_name(self.megatron_param)

        # Determine which EP rank owns this expert
        owning_ep_rank = self._get_expert_ownership(expert_id, config)

        # Only process on the owning rank
        if self.ep_rank == owning_ep_rank:
            # Now delegate TP distribution to the TP mapping
            return self._tp_mapping.hf_to_megatron(hf_weights, megatron_module)
        else:
            # This rank doesn't own this expert
            return None

    def megatron_to_hf(
        self,
        megatron_weights: Optional[torch.Tensor],
        megatron_module: Optional[nn.Module],
    ) -> Dict[str, torch.Tensor]:
        """
        Gather from EP ranks then delegate TP gathering.

        This method:
        1. Handles cross-PP broadcast (inherited from base)
        2. Determines which EP rank owns this expert
        3. Broadcasts from owning EP rank
        4. Delegates TP gathering to TPAwareMapping
        """
        # Handle cross-PP broadcast first
        megatron_weights = self.broadcast_from_pp_rank(megatron_weights)

        if megatron_weights is None:
            return {}

        # Get configuration
        if megatron_module is None:
            config = self.broadcast_obj_from_pp_rank(None)
        else:
            cfg_local = self._get_config(megatron_module)
            config = self.broadcast_obj_from_pp_rank(cfg_local)

        if self.ep_size == 1:
            # No EP distribution, delegate directly to TP mapping
            return self._tp_mapping.megatron_to_hf(megatron_weights, megatron_module)

        # Extract expert ID from resolved name
        expert_id = self._get_expert_id_from_name(self.megatron_param)

        # Determine owning EP rank
        owning_ep_rank = self._get_expert_ownership(expert_id, config)

        # Broadcast weight from owning EP rank to all EP ranks
        # (TPAwareMapping needs the weight on all ranks for TP gathering)
        if self.ep_rank == owning_ep_rank:
            # First gather TP shards on the owning EP rank
            tp_gathered = self._tp_mapping.megatron_to_hf(megatron_weights, megatron_module)
            # Extract the gathered weight (there should be only one key)
            if tp_gathered:
                gathered_weight = next(iter(tp_gathered.values()))
            else:
                gathered_weight = None
        else:
            gathered_weight = None

        # Broadcast the gathered weight from owning EP rank to all EP ranks
        gathered_weight = self._broadcast_from_ep_rank(gathered_weight, owning_ep_rank)

        # Only return from EP rank 0 to avoid duplicates
        if self.ep_rank == 0 and gathered_weight is not None:
            return {str(self.hf_param): gathered_weight}

        return {}

    def _get_expert_id_from_name(self, name: str) -> int:
        """
        Extract expert ID from resolved weight name.

        For patterns like "weight" followed by a number
        The expert ID is expected to be the last numeric component.
        """
        import re

        # Look for pattern like "weight" followed by a number
        match = re.search(r"weight(\d+)", name)
        if match:
            return int(match.group(1))

        # Also check for patterns like "experts.0.gate_proj"
        match = re.search(r"experts\.(\d+)\.", name)
        if match:
            return int(match.group(1))

        # Fallback: try to find any trailing number
        match = re.search(r"(\d+)[^\d]*$", name)
        if match:
            return int(match.group(1))

        raise ValueError(f"Could not extract expert ID from weight name: {name}")

    def _get_expert_ownership(self, expert_id: int, config: Any) -> int:
        """
        Determine which EP rank owns a given expert.

        Args:
            expert_id (int): Global expert index
            config (Any): Model configuration with num_moe_experts

        Returns:
            int: EP rank that owns this expert
        """
        num_experts = config.num_moe_experts
        num_experts_per_rank = num_experts // self.ep_size
        return expert_id // num_experts_per_rank

    def _broadcast_from_ep_rank(self, tensor: Optional[torch.Tensor], owning_rank: int) -> Optional[torch.Tensor]:
        """
        Broadcast a tensor from a specific EP rank to all EP ranks.

        Args:
            tensor (Optional[torch.Tensor]): Tensor to broadcast (None on non-owning ranks)
            owning_rank (int): EP rank that owns the tensor

        Returns:
            Optional[torch.Tensor]: Broadcasted tensor on all EP ranks
        """
        if self.ep_size == 1:
            return tensor

        # Allocate tensor on non-owning ranks
        if self.ep_rank != owning_rank and tensor is None:
            # Get tensor spec from owning rank
            tensor_spec = None
            if self.ep_rank == owning_rank:
                tensor_spec = (tensor.shape, tensor.dtype)

            # Broadcast spec
            tensor_spec = torch.distributed.broadcast_object_list([tensor_spec], src=owning_rank, group=self.ep_group)[
                0
            ]

            # Allocate tensor
            shape, dtype = tensor_spec
            # Use CPU by default, unless CUDA is available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            tensor = torch.empty(shape, dtype=dtype, device=device)

        # Broadcast tensor data
        global_src = torch.distributed.get_global_rank(group=self.ep_group, group_rank=owning_rank)
        torch.distributed.broadcast(tensor, src=global_src, group=self.ep_group)

        return tensor

    @property
    def ep_size(self) -> int:
        """Get expert parallel world size."""
        return mpu.get_expert_model_parallel_world_size()

    @property
    def ep_rank(self) -> int:
        """Get current expert parallel rank."""
        return mpu.get_expert_model_parallel_rank()

    @property
    def ep_group(self) -> Optional[torch.distributed.ProcessGroup]:
        """Get expert parallel process group."""
        if self.ep_size > 1:
            return mpu.get_expert_model_parallel_group()
        return None


def merge_qkv_biases(config: TransformerConfig, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Merge separate Q, K, V bias vectors into Megatron's interleaved QKV format.

    Args:
        config (TransformerConfig): Transformer configuration.
        q (torch.Tensor): Query projection biases [hidden_size].
        k (torch.Tensor): Key projection biases [kv_hidden_size].
        v (torch.Tensor): Value projection biases [kv_hidden_size].

    Returns:
        torch.Tensor: Interleaved QKV biases in Megatron format as 1D tensor.
    """
    head_num = config.num_attention_heads
    num_query_groups = config.num_query_groups
    heads_per_group = head_num // num_query_groups
    head_size = config.kv_channels or (config.hidden_size // head_num)

    # Reshape biases to expose head dimension
    q = q.view(head_num, head_size)
    k = k.view(num_query_groups, head_size)
    v = v.view(num_query_groups, head_size)

    # Interleave in Megatron pattern: [q1...qn, k1, v1, q1...qn, k2, v2, ...]
    qkv_biases = []
    for i in range(num_query_groups):
        qkv_biases.append(q[i * heads_per_group : (i + 1) * heads_per_group, :])
        qkv_biases.append(k[i : i + 1, :])
        qkv_biases.append(v[i : i + 1, :])

    # Concatenate and flatten back to 1D
    qkv = torch.cat(qkv_biases)
    return qkv.flatten()


def split_qkv_biases(config: TransformerConfig, qkv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split Megatron's interleaved QKV bias into separate Q, K, V biases.

    Args:
        config (TransformerConfig): Transformer configuration.
        qkv (torch.Tensor): Interleaved QKV biases in Megatron format (1D
            tensor).

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple of (Q, K, V) bias vectors.
    """
    head_num = config.num_attention_heads
    num_query_groups = config.num_query_groups
    heads_per_group = head_num // num_query_groups
    head_size = config.kv_channels or (config.hidden_size // head_num)
    qkv_total_dim = head_num + 2 * num_query_groups

    # Reshape to expose interleaved structure
    qkv = qkv.reshape(qkv_total_dim, head_size)

    # Extract Q, K, V from interleaved pattern
    q_slice = torch.cat(
        [
            torch.arange((heads_per_group + 2) * i, (heads_per_group + 2) * i + heads_per_group)
            for i in range(num_query_groups)
        ]
    )
    k_slice = torch.arange(heads_per_group, qkv_total_dim, heads_per_group + 2)
    v_slice = torch.arange(heads_per_group + 1, qkv_total_dim, heads_per_group + 2)

    q = qkv[q_slice].flatten()
    k = qkv[k_slice].flatten()
    v = qkv[v_slice].flatten()

    return q, k, v


def merge_qkv_weights(provider: TransformerConfig, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Merge separate Q, K, V weight matrices into Megatron's interleaved QKV format.

    Args:
        provider (TransformerConfig): Model configuration provider.
        q (torch.Tensor): Query projection weights [hidden_size, hidden_size] or
            bias [hidden_size].
        k (torch.Tensor): Key projection weights [kv_hidden_size, hidden_size]
            or bias [kv_hidden_size].
        v (torch.Tensor): Value projection weights [kv_hidden_size,
            hidden_size] or bias [kv_hidden_size].

    Returns:
        torch.Tensor: Interleaved QKV weights in Megatron format.
    """
    head_num = provider.num_attention_heads
    num_query_groups = provider.num_query_groups
    heads_per_group = head_num // num_query_groups
    head_size = provider.kv_channels or (provider.hidden_size // head_num)
    hidden_size = provider.hidden_size
    is_bias = q.ndim == 1

    # Reshape to expose head dimension
    if is_bias:
        q_reshaped = q.view(head_num, head_size)
        k_reshaped = k.view(num_query_groups, head_size)
        v_reshaped = v.view(num_query_groups, head_size)
    else:
        q_reshaped = q.view(head_num, head_size, hidden_size)
        k_reshaped = k.view(num_query_groups, head_size, hidden_size)
        v_reshaped = v.view(num_query_groups, head_size, hidden_size)

    # Interleave in Megatron pattern: [q1...qn, k1, v1, q1...qn, k2, v2, ...]
    qkv_weights = []
    for i in range(num_query_groups):
        q_group = q_reshaped[i * heads_per_group : (i + 1) * heads_per_group]
        k_group = k_reshaped[i : i + 1]
        v_group = v_reshaped[i : i + 1]
        qkv_weights.extend([q_group, k_group, v_group])

    qkv = torch.cat(qkv_weights, dim=0)

    # Final reshape
    if is_bias:
        return qkv.reshape(-1)
    else:
        return qkv.reshape([-1, hidden_size])


def split_qkv_weights(
    provider: TransformerConfig, qkv: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split Megatron's interleaved QKV tensor into separate Q, K, V matrices.

    Args:
        provider (TransformerConfig): Model configuration provider.
        qkv (torch.Tensor): Interleaved QKV weights in Megatron format.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple of (Q, K, V)
            weight matrices.
    """
    head_num = provider.num_attention_heads
    num_query_groups = provider.num_query_groups
    heads_per_group = head_num // num_query_groups
    head_size = provider.kv_channels or (provider.hidden_size // head_num)
    qkv_total_dim = head_num + 2 * num_query_groups
    is_bias = qkv.ndim == 1

    if is_bias:
        hidden_size = 1
        qkv_reshaped = qkv.view(qkv_total_dim, head_size)
    else:
        hidden_size = qkv.shape[-1]
        qkv_reshaped = qkv.view(qkv_total_dim, head_size, hidden_size)

    # Extract Q, K, V from interleaved pattern
    q_slice = torch.cat(
        [
            torch.arange((heads_per_group + 2) * i, (heads_per_group + 2) * i + heads_per_group)
            for i in range(num_query_groups)
        ]
    )
    k_slice = torch.arange(heads_per_group, qkv_total_dim, heads_per_group + 2)
    v_slice = torch.arange(heads_per_group + 1, qkv_total_dim, heads_per_group + 2)

    q = qkv_reshaped[q_slice]
    k = qkv_reshaped[k_slice]
    v = qkv_reshaped[v_slice]

    if is_bias:
        q = q.reshape(-1)
        k = k.reshape(-1)
        v = v.reshape(-1)
    else:
        q = q.reshape(-1, hidden_size)
        k = k.reshape(-1, hidden_size)
        v = v.reshape(-1, hidden_size)

    return q, k, v


def gather_tp_qkv(provider: TransformerConfig, tensors: List[torch.Tensor]) -> torch.Tensor:
    """Gather QKV weights from all tensor parallel ranks.

    Args:
        provider (TransformerConfig): Model configuration provider.
        tensors (List[torch.Tensor]): List of tensor shards from each TP rank.

    Returns:
        torch.Tensor: Full QKV weight matrix.
    """
    if provider.tensor_model_parallel_size == 1:
        return tensors[0]
    return torch.cat(tensors, dim=0)


def gather_tp_gated_mlp(provider: TransformerConfig, tensors: List[torch.Tensor]) -> torch.Tensor:
    """Gather gated MLP weights from all tensor parallel ranks.

    Args:
        provider (TransformerConfig): Model configuration provider.
        tensors (List[torch.Tensor]): List of tensor shards from each TP rank.

    Returns:
        torch.Tensor: Full gated MLP weight matrix.
    """
    if provider.tensor_model_parallel_size == 1:
        return tensors[0]

    # Split each shard into gate and up parts
    gate_parts = []
    up_parts = []
    for tensor in tensors:
        gate, up = torch.chunk(tensor, 2, dim=0)
        gate_parts.append(gate)
        up_parts.append(up)

    # Concatenate gates and ups separately, then merge
    full_gate = torch.cat(gate_parts, dim=0)
    full_up = torch.cat(up_parts, dim=0)
    return torch.cat([full_gate, full_up], dim=0)


def gather_tp_column_parallel(provider: TransformerConfig, tensors: List[torch.Tensor]) -> torch.Tensor:
    """Gather column-parallel weights from all tensor parallel ranks.

    Args:
        provider (TransformerConfig): Model configuration provider.
        tensors (List[torch.Tensor]): List of tensor shards from each TP rank.

    Returns:
        torch.Tensor: Full weight matrix.
    """
    if provider.tensor_model_parallel_size == 1:
        return tensors[0]
    return torch.cat(tensors, dim=0)


def gather_tp_row_parallel(provider: TransformerConfig, tensors: List[torch.Tensor]) -> torch.Tensor:
    """Gather row-parallel weights from all tensor parallel ranks.

    Args:
        provider (TransformerConfig): Model configuration provider.
        tensors (List[torch.Tensor]): List of tensor shards from each TP rank.

    Returns:
        torch.Tensor: Full weight matrix.
    """
    if provider.tensor_model_parallel_size == 1:
        return tensors[0]
    return torch.cat(tensors, dim=1)


def transpose_tp_row_parallel(provider: TransformerConfig, tensor: torch.Tensor) -> torch.Tensor:
    """Transpose the weights for row-parallel layers.

    Args:
        provider (TransformerConfig): Model configuration provider.
        tensor (torch.Tensor): The gathered tensor.

    Returns:
        torch.Tensor: The transposed tensor.
    """
    # Always transpose row-parallel weights when converting from Megatron to HF
    # because they have different conventions
    return torch.transpose(tensor, 0, 1).contiguous()


def transpose_for_tp_row_parallel(provider: TransformerConfig, tensor: torch.Tensor) -> torch.Tensor:
    """Transpose weights from HF to Megatron format for row-parallel layers.

    This is used as a to_target transformation when loading HF weights.
    HF stores these as (in_features, out_features) but Megatron expects
    (out_features, in_features) for row-parallel layers.

    Args:
        provider (TransformerConfig): Model configuration provider.
        tensor (torch.Tensor): The HF weight tensor.

    Returns:
        torch.Tensor: The transposed tensor ready for Megatron.
    """
    return torch.transpose(tensor, 0, 1).contiguous()


def split_tp_qkv(provider: TransformerConfig, tensor: torch.Tensor) -> List[torch.Tensor]:
    """Split QKV weights for distribution across tensor parallel ranks.

    Args:
        provider (TransformerConfig): Model configuration provider.
        tensor (torch.Tensor): Full QKV weight matrix.

    Returns:
        List[torch.Tensor]: List of tensor shards, one per TP rank.
    """
    if provider.tensor_model_parallel_size == 1:
        return [tensor]
    return list(torch.chunk(tensor, provider.tensor_model_parallel_size, dim=0))


def split_tp_gated_mlp(provider: TransformerConfig, tensor: torch.Tensor) -> List[torch.Tensor]:
    """Split gated MLP weights for distribution across tensor parallel ranks.

    Args:
        provider (TransformerConfig): Model configuration provider.
        tensor (torch.Tensor): Full gated MLP weight matrix.

    Returns:
        List[torch.Tensor]: List of tensor shards, one per TP rank.
    """
    tp_size = provider.tensor_model_parallel_size
    if tp_size == 1:
        return [tensor]

    # First split into gate and up
    gate, up = torch.chunk(tensor, 2, dim=0)

    # Then split each across TP ranks
    gate_shards = torch.chunk(gate, tp_size, dim=0)
    up_shards = torch.chunk(up, tp_size, dim=0)

    # Recombine for each rank
    return [torch.cat([gate_shards[i], up_shards[i]], dim=0) for i in range(tp_size)]


def split_tp_column_parallel(provider: TransformerConfig, tensor: torch.Tensor) -> List[torch.Tensor]:
    """Split weights for column-parallel distribution across tensor parallel ranks.

    Args:
        provider (TransformerConfig): Model configuration provider.
        tensor (torch.Tensor): Full weight matrix.

    Returns:
        List[torch.Tensor]: List of tensor shards, one per TP rank.
    """
    if provider.tensor_model_parallel_size == 1:
        return [tensor]
    return list(torch.chunk(tensor, provider.tensor_model_parallel_size, dim=0))


def split_tp_row_parallel(provider: TransformerConfig, tensor: torch.Tensor) -> List[torch.Tensor]:
    """Split weights for row-parallel distribution across tensor parallel ranks.

    Args:
        provider (TransformerConfig): Model configuration provider.
        tensor (torch.Tensor): Full weight matrix.

    Returns:
        List[torch.Tensor]: List of tensor shards, one per TP rank.
    """
    if provider.tensor_model_parallel_size == 1:
        return [tensor]
    return list(torch.chunk(tensor, provider.tensor_model_parallel_size, dim=1))


def split_tp_row_parallel_from_hf(provider: TransformerConfig, tensor: torch.Tensor) -> List[torch.Tensor]:
    """Split HF weights for row-parallel distribution across tensor parallel ranks.

    This function is used when loading from HuggingFace format. For row-parallel
    layers, Megatron stores weights as (in_features, out_features_per_rank),
    which is the same orientation as HF but split along the output dimension.

    Args:
        provider (TransformerConfig): Model configuration provider.
        tensor (torch.Tensor): Full HF weight matrix (in_features, out_features).

    Returns:
        List[torch.Tensor]: List of tensor shards, one per TP rank.
    """
    if provider.tensor_model_parallel_size == 1:
        # For single GPU, return as-is (no transpose needed)
        return [tensor]

    # For row parallel in Megatron:
    # HF stores as (in_features, out_features)
    # Megatron expects (in_features, out_features_per_rank)
    # So we just split along dim 1, no transpose needed
    return list(torch.chunk(tensor, provider.tensor_model_parallel_size, dim=1))


def is_tensor_parallel(param: torch.Tensor) -> bool:
    """Check if a parameter is tensor parallel distributed."""
    return hasattr(param, "tensor_model_parallel") and param.tensor_model_parallel


def get_partition_dim(param: torch.Tensor) -> Optional[int]:
    """Get the partition dimension of a tensor parallel parameter."""
    return getattr(param, "partition_dim", None)


def colwise_tp_split(provider: TransformerConfig, weight: torch.Tensor):
    """Split a tensor along its first dimension.

    Args:
        provider (TransformerConfig): Model configuration provider.
        weight (torch.Tensor): The tensor to split.

    Returns:
        list[torch.Tensor]: List of tensor shards, one per TP rank.
    """
    return list(torch.chunk(weight, provider.tensor_model_parallel_size, dim=0))


def colwise_tp_gather(provider: TransformerConfig, shards: list[torch.Tensor]) -> torch.Tensor:
    """Gather column-wise tensor parallel shards by concatenating along dimension 0.

    Args:
        provider (TransformerConfig): Model configuration provider.
        shards (list[torch.Tensor]): List of tensor shards from each TP rank.

    Returns:
        torch.Tensor: Full weight matrix.
    """
    return torch.cat(shards, dim=0)
