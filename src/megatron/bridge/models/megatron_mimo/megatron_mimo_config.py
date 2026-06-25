# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY


@dataclass
class ModuleParallelismConfig:
    """Parallelism config for a single module in a MegatronMIMO model."""

    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    context_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    expert_tensor_parallel_size: int = 1
    data_parallel_size: Optional[int] = None
    rank_offset: int = 0

    @property
    def dense_model_parallel_size(self) -> int:
        """Dense/token model-parallel size: TP * CP * PP."""
        return self.tensor_model_parallel_size * self.context_parallel_size * self.pipeline_model_parallel_size

    @property
    def total_model_parallel_size(self) -> int:
        """Backward-compatible alias for the dense/token model-parallel size."""
        return self.dense_model_parallel_size

    @property
    def total_ranks(self) -> int:
        if self.data_parallel_size is None:
            raise ValueError("data_parallel_size must be set before accessing total_ranks.")
        return self.dense_model_parallel_size * self.data_parallel_size

    @property
    def expert_data_parallel_size(self) -> int:
        """Derived expert data-parallel size over this module's rank span."""
        if self.data_parallel_size is None:
            raise ValueError("data_parallel_size must be set before accessing expert_data_parallel_size.")
        expert_model_parallel_span = (
            self.expert_tensor_parallel_size * self.expert_model_parallel_size * self.pipeline_model_parallel_size
        )
        if self.total_ranks % expert_model_parallel_span != 0:
            raise ValueError(
                "total_ranks must be divisible by expert_tensor_parallel_size * "
                "expert_model_parallel_size * pipeline_model_parallel_size."
            )
        return self.total_ranks // expert_model_parallel_span

    def finalize(self, world_size: Optional[int]) -> None:
        """Compute data_parallel_size if unset, and validate parallelism constraints."""
        self._validate_positive_sizes(validate_data_parallel=False)

        if self.data_parallel_size is None:
            if world_size is None or world_size <= 0:
                raise ValueError("world_size must be provided to compute data_parallel_size.")
            if world_size % self.dense_model_parallel_size != 0:
                raise ValueError(
                    f"world_size ({world_size}) is not divisible by total_model_parallel_size "
                    f"({self.total_model_parallel_size})."
                )
            self.data_parallel_size = world_size // self.dense_model_parallel_size

        self._validate_positive_sizes(validate_data_parallel=True)
        self._validate_expert_factorization()

    def _validate_positive_sizes(self, *, validate_data_parallel: bool) -> None:
        fields_to_validate = (
            "tensor_model_parallel_size",
            "pipeline_model_parallel_size",
            "context_parallel_size",
            "expert_model_parallel_size",
            "expert_tensor_parallel_size",
        )
        if validate_data_parallel:
            fields_to_validate += ("data_parallel_size",)

        for field_name in fields_to_validate:
            value = getattr(self, field_name)
            if value is None or value <= 0:
                raise ValueError(f"{field_name} must be positive.")

        if self.rank_offset < 0:
            raise ValueError("rank_offset must be non-negative.")

    def _validate_expert_factorization(self) -> None:
        if self.data_parallel_size is None:
            return
        dense_token_span = self.tensor_model_parallel_size * self.context_parallel_size * self.data_parallel_size
        expert_span = self.expert_tensor_parallel_size * self.expert_model_parallel_size
        if dense_token_span % expert_span != 0:
            raise ValueError(
                "TP * CP * DP must be divisible by expert_tensor_parallel_size * "
                f"expert_model_parallel_size; got TP={self.tensor_model_parallel_size}, "
                f"CP={self.context_parallel_size}, DP={self.data_parallel_size}, "
                f"ETP={self.expert_tensor_parallel_size}, EP={self.expert_model_parallel_size}."
            )


@dataclass
class MegatronMIMOParallelismConfig:
    """Configuration for multi-module (MegatronMIMO) heterogeneous parallelism.

    Supports heterogeneous deployment where each module can have different
    parallelism configurations and rank offsets.

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

    def _validate_heterogeneous(self, world_size: int) -> None:
        """Validate heterogeneous deployment: rank ranges tile [0, world_size)."""
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

        expected_start = 0
        for start, end, name in ranges:
            if start != expected_start:
                raise ValueError(
                    "rank_offset ranges must tile the distributed world with no gaps; "
                    f"expected module '{name}' to start at rank {expected_start}, got {start}."
                )
            expected_start = end

        if expected_start != world_size:
            raise ValueError(
                "rank_offset ranges must tile the distributed world with no gaps; "
                f"covered ranks [0, {expected_start}), world_size is {world_size}."
            )

    def _validate_encoder_expert_parallelism(self) -> None:
        for name, parallelism in self.module_parallelisms.items():
            if name == MIMO_LANGUAGE_MODULE_KEY:
                continue
            if parallelism.expert_model_parallel_size != 1 or parallelism.expert_tensor_parallel_size != 1:
                raise ValueError(
                    "Encoder modules must remain dense for MegatronMIMO MoE; "
                    f"module '{name}' has expert_model_parallel_size="
                    f"{parallelism.expert_model_parallel_size}, expert_tensor_parallel_size="
                    f"{parallelism.expert_tensor_parallel_size}."
                )

    def _validate_parallelism_constraints(self) -> None:
        """Validate parallelism constraints for cross-module communication.

        - TP sizes must be powers of 2
        - DP sizes must be pairwise divisible (one divides the other)
        """

        def is_power_of_two(n: int) -> bool:
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

    def finalize(self, world_size: int) -> None:
        """Finalize parallelism config: compute data_parallel_size and validate.

        Args:
            world_size: Total number of ranks in the distributed world.
                MegatronMIMO requires a distributed environment, so this must always be provided.
        """
        if MIMO_LANGUAGE_MODULE_KEY not in self.module_parallelisms:
            raise ValueError(
                f"Language module '{MIMO_LANGUAGE_MODULE_KEY}' must be in module_parallelisms. "
                f"Found modules: {list(self.module_parallelisms.keys())}"
            )

        # In heterogeneous mode, data_parallel_size must be pre-set (not computed from world_size)
        for parallelism in self.module_parallelisms.values():
            parallelism.finalize(None)

        self._validate_heterogeneous(world_size)
        self._validate_encoder_expert_parallelism()
        self._validate_parallelism_constraints()
