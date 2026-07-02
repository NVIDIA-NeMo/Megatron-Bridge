# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Provider-neutral runtime infrastructure for heterogeneous MegatronMIMO models."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from megatron.core.process_groups_config import ProcessGroupCollection


if TYPE_CHECKING:
    from megatron.core.hyper_comm_grid import HyperCommGrid


@dataclass
class MegatronMIMOInfra:
    """Runtime process-group infrastructure for a heterogeneous MIMO model."""

    module_to_grid_map: dict[str, "HyperCommGrid"]
    topology: dict[str, list[str]]
    pg_collections: dict[str, ProcessGroupCollection | None]
    participating_modules: list[str]
    module_output_ndim: dict[str, int] = field(default_factory=dict)


__all__ = ["MegatronMIMOInfra"]
