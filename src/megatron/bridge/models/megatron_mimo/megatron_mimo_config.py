# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY


@dataclass
class ModuleParallelismConfig:
    """Parallelism config for a single module in a MegatronMIMO model."""

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
class MegatronMIMOParallelismConfig:
    """Configuration for multi-module (MegatronMIMO) heterogeneous parallelism.

    Supports two module-placement modes — auto-detected from module rank ranges:

    * Non-colocated: each module occupies a disjoint rank range (different
      ``rank_offset`` / ``total_ranks``). Encoder and LLM run on separate
      physical ranks; cross-module communication uses the multi-module
      pipeline communicator.
    * Colocated: every module spans the same rank range (identical
      ``rank_offset`` AND identical ``total_ranks``), each with its own
      TP/DP layout. MCore's ``ColocatedBridgeCommunicator`` handles the
      encoder→LLM activation reshape.

    Partial overlap — modules whose ranges neither fully match nor are fully
    disjoint — is rejected because neither mode applies.

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

    def _validate_module_placement(self) -> None:
        """Validate module placement against the two supported modes.

        Accepts either:
          * Colocated — every module range identical (same ``rank_offset``
            AND ``total_ranks``).
          * Non-colocated — every module range pairwise disjoint.

        Rejects any other layout (partial overlap or containment — e.g. an
        LLM spanning all ranks while two encoders occupy disjoint halves).
        Such "hybrid" layouts are a legitimate pattern but not yet
        supported: MCore's ``RankRole.build`` collapses placement to a
        single mode enum and ``MimoModel.forward`` dispatches on that enum,
        so a rank in both an encoder grid and the LLM grid would run only
        one module per step. Supporting hybrid placement would require
        per-pair mode detection on the MCore side. Until then we fail
        fast here so configurations are unambiguous.
        """
        for parallelism in self.module_parallelisms.values():
            if parallelism.data_parallel_size is None:
                raise ValueError("data_parallel_size must be set for module placement.")

        placement_tuples = [(p.rank_offset, p.total_ranks) for p in self.module_parallelisms.values()]
        if len(set(placement_tuples)) == 1:
            # All modules share the same rank range → colocated. Valid.
            return

        # Non-colocated: every pair of ranges must be disjoint.
        ranges = sorted(
            (p.rank_offset, p.rank_offset + p.total_ranks, name) for name, p in self.module_parallelisms.items()
        )
        for idx in range(1, len(ranges)):
            prev_start, prev_end, prev_name = ranges[idx - 1]
            cur_start, cur_end, cur_name = ranges[idx]
            if cur_start < prev_end:
                raise ValueError(
                    f"Module rank ranges must be either all identical (colocated) "
                    f"or all pairwise disjoint (non-colocated). Got partial overlap "
                    f"(hybrid placement — valid pattern, not yet supported): "
                    f"'{prev_name}' = [{prev_start}, {prev_end}), "
                    f"'{cur_name}' = [{cur_start}, {cur_end})."
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

        # Colocated with asymmetric DP is not yet supported: the forward step
        # slices the global micro-batch by a single DP geometry, so modality
        # inputs and LLM keys cannot be routed independently.
        # TODO(liding): enable asymmetric DP.
        is_colocated = len({(p.rank_offset, p.total_ranks) for p in self.module_parallelisms.values()}) == 1
        if is_colocated and llm_dp is not None:
            for name, p in self.module_parallelisms.items():
                if name == MIMO_LANGUAGE_MODULE_KEY:
                    continue
                encoder_dp = p.data_parallel_size
                if encoder_dp is not None and encoder_dp != llm_dp:
                    raise ValueError(
                        f"Colocated MegatronMIMO requires encoder DP == LLM DP. "
                        f"Module '{name}' has DP={encoder_dp}, LLM has DP={llm_dp}."
                    )

        # Colocated with asymmetric TP is not yet supported: mcore's CUDA RNG
        # tracker has one set of named states seeded by a single tp_rank, so a
        # rank whose encoder tp_rank and LLM tp_rank differ will run one module
        # with the wrong TP-region dropout masks and weight-init RNG. Correct
        # per-module seeding requires module-scoped tracker contexts, an mcore
        # architectural change.
        # TODO(liding): enable asymmetric TP.
        llm_tp = self.module_parallelisms[MIMO_LANGUAGE_MODULE_KEY].tensor_model_parallel_size
        if is_colocated:
            for name, p in self.module_parallelisms.items():
                if name == MIMO_LANGUAGE_MODULE_KEY:
                    continue
                encoder_tp = p.tensor_model_parallel_size
                if encoder_tp != llm_tp:
                    raise ValueError(
                        f"Colocated MegatronMIMO requires encoder TP == LLM TP. "
                        f"Module '{name}' has TP={encoder_tp}, LLM has TP={llm_tp}."
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

        self._validate_module_placement()
        self._validate_parallelism_constraints()

        expected = self.total_world_size
        if expected and world_size != expected:
            raise ValueError(f"MegatronMIMO world size mismatch: expected {expected}, got {world_size}.")
