from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional
import warnings


@dataclass
class ModuleParallelismConfig:
    """Parallelism config for a single module in a MIMO model."""

    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    context_parallel_size: int = 1
    expert_tensor_parallel_size: int = 1
    data_parallel_size: Optional[int] = None
    rank_offset: int = 0

    @property
    def total_model_parallel_size(self) -> int:
        return (
            self.tensor_model_parallel_size
            * self.pipeline_model_parallel_size
            * self.context_parallel_size
            * self.expert_tensor_parallel_size
        )

    @property
    def total_ranks(self) -> int:
        if self.data_parallel_size is None:
            raise ValueError("data_parallel_size must be set before accessing total_ranks.")
        return self.total_model_parallel_size * self.data_parallel_size

    def finalize(self, world_size: Optional[int]) -> None:
        """Compute data_parallel_size if unset, and validate parallelism constraints."""
        if self.data_parallel_size is None:
            if world_size is None or world_size <= 0:
                raise ValueError("world_size must be provided to compute data_parallel_size.")
            if world_size % self.total_model_parallel_size != 0:
                raise ValueError(
                    f"world_size ({world_size}) is not divisible by total_model_parallel_size "
                    f"({self.total_model_parallel_size})."
                )
            self.data_parallel_size = world_size // self.total_model_parallel_size

        if self.data_parallel_size <= 0:
            raise ValueError("data_parallel_size must be positive.")

        if self.expert_tensor_parallel_size > 1 and self.pipeline_model_parallel_size > 1:
            warnings.warn(
                "Using expert_tensor_parallel_size > 1 with pipeline_model_parallel_size > 1 "
                "is complex and may be unsupported.",
                stacklevel=2,
            )


@dataclass
class MimoParallelismConfig:
    """Configuration for multi-module (MIMO) heterogeneous parallelism."""

    llm_module_name: str
    module_parallelisms: dict[str, ModuleParallelismConfig]
    special_token_ids: dict[str, int] = field(default_factory=dict)
    deployment_mode: Literal["colocated", "heterogeneous", "homogeneous"] = "colocated"
    # TODO: Add optional topology when supporting non-encoder-to-LLM flows.

    def get_parallelism(self, module_name: str) -> ModuleParallelismConfig:
        return self.module_parallelisms[module_name]

    @property
    def module_names(self) -> list[str]:
        return list(self.module_parallelisms.keys())

    @property
    def total_world_size(self) -> int:
        if self.deployment_mode in ("colocated", "homogeneous"):
            totals = [p.total_ranks for p in self.module_parallelisms.values()]
            return max(totals) if totals else 0
        ranges = [p.rank_offset + p.total_ranks for p in self.module_parallelisms.values()]
        return max(ranges) if ranges else 0

    def _validate_colocated(self) -> None:
        totals = []
        for parallelism in self.module_parallelisms.values():
            if parallelism.rank_offset != 0:
                raise ValueError("rank_offset must be 0 for colocated deployment.")
            totals.append(parallelism.total_ranks)
        if totals and len(set(totals)) > 1:
            raise ValueError("All modules must have the same total_ranks in colocated deployment.")

    def _validate_homogeneous(self) -> None:
        first = None
        for parallelism in self.module_parallelisms.values():
            if parallelism.rank_offset != 0:
                raise ValueError("rank_offset must be 0 for homogeneous deployment.")
            values = (
                parallelism.tensor_model_parallel_size,
                parallelism.pipeline_model_parallel_size,
                parallelism.context_parallel_size,
                parallelism.expert_tensor_parallel_size,
                parallelism.data_parallel_size,
            )
            if first is None:
                first = values
            elif values != first:
                raise ValueError("All modules must have identical parallelism in homogeneous deployment.")

    def _validate_heterogeneous(self) -> None:
        # "heterogeneous" describes rank placement across distinct modules.
        ranges = []
        for parallelism in self.module_parallelisms.values():
            if parallelism.data_parallel_size is None:
                raise ValueError("data_parallel_size must be set for heterogeneous deployment.")
            ranges.append((parallelism.rank_offset, parallelism.rank_offset + parallelism.total_ranks))

        ranges.sort()
        for idx in range(1, len(ranges)):
            prev_end = ranges[idx - 1][1]
            cur_start = ranges[idx][0]
            if cur_start < prev_end:
                raise ValueError("rank_offset ranges overlap in heterogeneous deployment.")

    def finalize(self, world_size: Optional[int]) -> None:
        if self.llm_module_name not in self.module_parallelisms:
            raise ValueError(f"LLM module '{self.llm_module_name}' not in module_parallelisms.")

        if self.deployment_mode in ("colocated", "homogeneous"):
            for parallelism in self.module_parallelisms.values():
                parallelism.finalize(world_size)
        else:
            for parallelism in self.module_parallelisms.values():
                parallelism.finalize(None)

        if self.deployment_mode == "colocated":
            self._validate_colocated()
        elif self.deployment_mode == "homogeneous":
            self._validate_homogeneous()
        else:
            self._validate_heterogeneous()

        if world_size and world_size > 1:
            expected = self.total_world_size
            if expected and world_size != expected:
                raise ValueError(f"MIMO world size mismatch: expected {expected}, got {world_size}.")
