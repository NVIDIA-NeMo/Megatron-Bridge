# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY


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

    The language module must be named MIMO_LANGUAGE_MODULE_KEY ("language") in module_parallelisms.
    """

    module_parallelisms: dict[str, ModuleParallelismConfig]
    special_token_ids: dict[str, int] = field(default_factory=dict)

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
        """
        Ensure module rank ranges do not overlap in a heterogeneous deployment.
        
        This verifies that every module has `data_parallel_size` set and that the half-open rank ranges
        defined by (`rank_offset`, `rank_offset + total_ranks`) for all modules are non-overlapping.
        Raises:
            ValueError: If a module's `data_parallel_size` is `None`.
            ValueError: If any module's rank range overlaps a previous module's range.
        """
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

    def _validate_parallelism_constraints(self) -> None:
        """
        Validate that module parallelism settings meet constraints required for cross-module communication and embedding alignment.
        
        Performs these checks:
        - Tensor-parallel (TP) sizes for every module must be powers of two.
        - Data-parallel (DP) sizes between every pair of modules (when both are set) must be divisible (one must divide the other).
        - For embedding alignment, every non-language ("encoder") module's DP (when set) must be greater than or equal to the language module's DP.
        
        Raises:
            ValueError: If any TP is not a power of two, if any pair of set DP sizes are not divisible, or if an encoder module's DP is less than the language module's DP.
        """

        def is_power_of_two(n: int) -> bool:
            """
            Check whether an integer is a power of two.
            
            Returns:
                `True` if `n` is a power of two and greater than zero, `False` otherwise.
            """
            return n > 0 and (n & (n - 1)) == 0

        # Validate TP is power of 2
        for name, p in self.module_parallelisms.items():
            tp = p.tensor_model_parallel_size
            if not is_power_of_two(tp):
                raise ValueError(
                    f"Module '{name}' has TP={tp}, but TP size must be a power of 2 "
                    f"(1, 2, 4, 8, ...) for cross-module communication compatibility."
                )

        # Validate DP sizes are pairwise divisible
        module_names = list(self.module_parallelisms.keys())
        for i, name1 in enumerate(module_names):
            for name2 in module_names[i + 1 :]:
                dp1 = self.module_parallelisms[name1].data_parallel_size
                dp2 = self.module_parallelisms[name2].data_parallel_size
                if dp1 is None or dp2 is None:
                    continue
                if dp1 % dp2 != 0 and dp2 % dp1 != 0:
                    raise ValueError(
                        f"DP sizes must be divisible between modules. "
                        f"Module '{name1}' has DP={dp1}, module '{name2}' has DP={dp2}. "
                        f"One must divide the other for BridgeCommunicator."
                    )

        # Validate encoder DP >= LLM DP for embedding alignment
        # Encoder modules produce embeddings consumed by LLM. If encoder DP < LLM DP,
        # the same encoder batch would need to align with different LLM batches, which fails.
        llm_dp = self.module_parallelisms[MIMO_LANGUAGE_MODULE_KEY].data_parallel_size
        if llm_dp is not None:
            for name, p in self.module_parallelisms.items():
                if name == MIMO_LANGUAGE_MODULE_KEY:
                    continue
                encoder_dp = p.data_parallel_size
                if encoder_dp is not None and encoder_dp < llm_dp:
                    raise ValueError(
                        f"Encoder module '{name}' has DP={encoder_dp} < LLM DP={llm_dp}. "
                        f"Encoder DP must be >= LLM DP for embedding alignment across batches."
                    )

    def finalize(self, world_size: int) -> None:
        """
        Finalize and validate all module parallelism configurations against the provided world size.
        
        Parameters:
            world_size (int): Total number of ranks in the distributed world; must match the computed total from module configurations.
        
        Raises:
            ValueError: If the language module (MIMO_LANGUAGE_MODULE_KEY) is missing from module_parallelisms.
            ValueError: If any module's configuration is invalid or inconsistent with heterogeneous constraints.
            ValueError: If the provided world_size does not equal the computed total world size when the computed total is non-zero.
        """
        if MIMO_LANGUAGE_MODULE_KEY not in self.module_parallelisms:
            raise ValueError(
                f"Language module '{MIMO_LANGUAGE_MODULE_KEY}' must be in module_parallelisms. "
                f"Found modules: {list(self.module_parallelisms.keys())}"
            )

        # In heterogeneous mode, data_parallel_size must be pre-set (not computed from world_size)
        for parallelism in self.module_parallelisms.values():
            parallelism.finalize(None)

        self._validate_heterogeneous()
        self._validate_parallelism_constraints()

        expected = self.total_world_size
        if expected and world_size != expected:
            raise ValueError(f"MIMO world size mismatch: expected {expected}, got {world_size}.")
