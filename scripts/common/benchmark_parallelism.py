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
"""Lightweight parallel-topology helpers shared by benchmark launchers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParallelTopology:
    """Model and expert parallel degrees that constrain a benchmark world."""

    tensor_parallel: int = 1
    pipeline_parallel: int = 1
    context_parallel: int = 1
    expert_parallel: int = 1
    expert_tensor_parallel: int | None = None
    gtp_remat: int = 1
    expert_gtp_remat: int = 1


def _first_attribute(config: Any, names: tuple[str, ...], default: Any) -> Any:
    """Return the first exposed configuration attribute from ``names``."""
    for name in names:
        if hasattr(config, name):
            return getattr(config, name)
    return default


def topology_from_config(config: Any) -> ParallelTopology:
    """Read parallel degrees from a model config or lightweight workload config."""
    return ParallelTopology(
        tensor_parallel=getattr(config, "tensor_model_parallel_size", 1),
        pipeline_parallel=getattr(config, "pipeline_model_parallel_size", 1),
        context_parallel=getattr(config, "context_parallel_size", 1),
        expert_parallel=getattr(config, "expert_model_parallel_size", 1),
        expert_tensor_parallel=getattr(config, "expert_tensor_parallel_size", None),
        gtp_remat=_first_attribute(config, ("gtp_weight_remat_size", "gtp_remat_size"), 1),
        expert_gtp_remat=_first_attribute(
            config,
            ("expert_gtp_weight_remat_size", "expert_gtp_remat_size"),
            1,
        ),
    )


def _validate_positive_sizes(sizes: dict[str, int]) -> None:
    """Require positive integer parallel degrees."""
    invalid_sizes = {name: size for name, size in sizes.items() if not isinstance(size, int) or size <= 0}
    if invalid_sizes:
        raise ValueError(f"Parallel sizes must be positive integers, got {invalid_sizes}.")


def data_parallel_size(*, num_gpus: int, topology: ParallelTopology) -> int:
    """Validate dense and expert grids, then return the dense DP degree."""
    if not isinstance(num_gpus, int) or num_gpus <= 0:
        raise ValueError(f"Number of GPUs must be a positive integer, got {num_gpus!r}.")

    expert_tensor_parallel = (
        topology.tensor_parallel if topology.expert_tensor_parallel is None else topology.expert_tensor_parallel
    )
    sizes = {
        "TP": topology.tensor_parallel,
        "PP": topology.pipeline_parallel,
        "CP": topology.context_parallel,
        "EP": topology.expert_parallel,
        "ETP": expert_tensor_parallel,
        "GTP remat": topology.gtp_remat,
        "expert GTP remat": topology.expert_gtp_remat,
    }
    _validate_positive_sizes(sizes)

    dense_factor = (
        topology.tensor_parallel * topology.pipeline_parallel * topology.context_parallel * topology.gtp_remat
    )
    if num_gpus % dense_factor != 0:
        if topology.gtp_remat == 1:
            factor_name = "TP * PP * CP"
            factor_values = f"{topology.tensor_parallel} * {topology.pipeline_parallel} * {topology.context_parallel}"
        else:
            factor_name = "TP * PP * CP * GTP remat"
            factor_values = (
                f"{topology.tensor_parallel} * {topology.pipeline_parallel} * {topology.context_parallel} "
                f"* {topology.gtp_remat}"
            )
        raise ValueError(
            f"Requested {num_gpus} GPUs is not divisible by {factor_name} "
            f"({factor_values} = {dense_factor}). Override the parallel sizes for this GPU count."
        )

    expert_factor = (
        expert_tensor_parallel * topology.expert_parallel * topology.pipeline_parallel * topology.expert_gtp_remat
    )
    if num_gpus % expert_factor != 0:
        if topology.expert_gtp_remat == 1:
            factor_name = "ETP * EP * PP"
            factor_values = f"{expert_tensor_parallel} * {topology.expert_parallel} * {topology.pipeline_parallel}"
        else:
            factor_name = "ETP * EP * PP * expert GTP remat"
            factor_values = (
                f"{expert_tensor_parallel} * {topology.expert_parallel} * {topology.pipeline_parallel} "
                f"* {topology.expert_gtp_remat}"
            )
        raise ValueError(
            f"The expert-parallel grid is incompatible with requested {num_gpus} GPUs: the world size must be "
            f"divisible by {factor_name} ({factor_values} = {expert_factor}). "
            "Override the expert or pipeline parallel sizes for this GPU count."
        )

    return num_gpus // dense_factor


def weak_scaled_global_batch_size(*, base_gbs: int, base_data_parallel: int, data_parallel: int) -> int:
    """Preserve samples per DP rank while requiring an integral GBS."""
    sizes = {
        "canonical DP": base_data_parallel,
        "requested DP": data_parallel,
        "canonical GBS": base_gbs,
    }
    invalid_sizes = {name: size for name, size in sizes.items() if not isinstance(size, int) or size <= 0}
    if invalid_sizes:
        raise ValueError(f"Weak-scaling inputs must be positive integers, got {invalid_sizes}.")

    scaled_gbs_numerator = base_gbs * data_parallel
    if scaled_gbs_numerator % base_data_parallel != 0:
        raise ValueError(
            f"Weak scaling GBS {base_gbs} from DP {base_data_parallel} to DP {data_parallel} does not produce "
            "an integer global batch size. Pass --global_batch_size explicitly."
        )
    return scaled_gbs_numerator // base_data_parallel
