# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional


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
    """Configuration for multi-module (MIMO) heterogeneous parallelism.

    Note: Phase 1 only supports heterogeneous deployment where each module
    can have different parallelism configurations and rank offsets.

    The LLM module must be named "llm" in module_parallelisms.
    """

    module_parallelisms: dict[str, ModuleParallelismConfig]
    special_token_ids: dict[str, int] = field(default_factory=dict)
    # TODO: Add optional topology when supporting non-encoder-to-LLM flows.

    def get_parallelism(self, module_name: str) -> ModuleParallelismConfig:
        return self.module_parallelisms[module_name]

    @property
    def module_names(self) -> list[str]:
        return list(self.module_parallelisms.keys())

    @property
    def total_world_size(self) -> int:
        """Compute total world size from module rank ranges."""
        ranges = [p.rank_offset + p.total_ranks for p in self.module_parallelisms.values()]
        return max(ranges) if ranges else 0

    def _validate_heterogeneous(self) -> None:
        """Validate heterogeneous deployment: no overlapping rank ranges."""
        ranges = []
        for name, parallelism in self.module_parallelisms.items():
            if parallelism.data_parallel_size is None:
                raise ValueError("data_parallel_size must be set for heterogeneous deployment.")
            ranges.append((parallelism.rank_offset, parallelism.rank_offset + parallelism.total_ranks, name))

        ranges.sort(key=lambda x: x[0])
        for idx in range(1, len(ranges)):
            prev_end = ranges[idx - 1][1]
            cur_start = ranges[idx][0]
            if cur_start < prev_end:
                raise ValueError("rank_offset ranges overlap in heterogeneous deployment.")

        # Check for gaps between modules (likely misconfiguration)
        # Gaps in the middle are errors; leading gaps (rank_offset > 0) are warnings
        if ranges:
            min_rank = ranges[0][0]  # Already sorted by rank_offset
            max_rank = ranges[-1][1]

            # Collect all covered ranks
            covered_ranks = set()
            for parallelism in self.module_parallelisms.values():
                start = parallelism.rank_offset
                end = start + parallelism.total_ranks
                covered_ranks.update(range(start, end))

            # Check for gaps between min and max (error - likely misconfiguration)
            expected_middle = set(range(min_rank, max_rank))
            gaps_in_middle = expected_middle - covered_ranks
            if gaps_in_middle:
                raise ValueError(
                    f"Ranks {sorted(gaps_in_middle)} are not assigned to any module in heterogeneous "
                    f"deployment. This creates a gap between modules which is not allowed."
                )

            # Check for leading gap (ranks 0 to min_rank-1 unused) - warning only
            if min_rank > 0:
                warnings.warn(
                    f"Ranks {list(range(min_rank))} (before first module) are not assigned to any "
                    f"module in heterogeneous deployment. These ranks will be idle during training.",
                    stacklevel=3,
                )

    def finalize(self, world_size: Optional[int]) -> None:
        """Finalize parallelism config: compute data_parallel_size and validate."""
        if "llm" not in self.module_parallelisms:
            raise ValueError(
                f"LLM module 'llm' must be in module_parallelisms. "
                f"Found modules: {list(self.module_parallelisms.keys())}"
            )

        # In heterogeneous mode, data_parallel_size must be pre-set (not computed from world_size)
        for parallelism in self.module_parallelisms.values():
            parallelism.finalize(None)

        self._validate_heterogeneous()

        if world_size and world_size > 1:
            expected = self.total_world_size
            if expected and world_size != expected:
                raise ValueError(f"MIMO world size mismatch: expected {expected}, got {world_size}.")
