from __future__ import annotations

from typing import Dict, Optional

from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mimo_config import MIMOConfig


def build_hypercomm_grids(cfg: ConfigContainer) -> Dict[str, "HyperCommGrid"]:
    """Create HyperCommGrid objects per module from MIMO config."""
    if cfg.mimo is None:
        raise ValueError("MIMO configuration is not set.")

    from megatron.core.hyper_comm_grid import HyperCommGrid

    grids: Dict[str, HyperCommGrid] = {}
    for module_name, parallelism in cfg.mimo.module_parallelisms.items():
        shape = [
            parallelism.tensor_parallel,
            parallelism.context_parallel,
            parallelism.expert_parallel,
            parallelism.pipeline_parallel,
            parallelism.data_parallel,
        ]
        grid = HyperCommGrid(
            shape=shape,
            dim_names=["tp", "cp", "ep", "pp", "dp"],
            rank_offset=parallelism.rank_offset,
            backend="nccl",
        )
        for dim in ("tp", "cp", "ep", "pp", "dp"):
            _ = grid.create_pg([dim])
        grids[module_name] = grid
    return grids


def _default_topology(mimo_cfg: MIMOConfig) -> Dict[str, list[str]]:
    """Infer a default multi-encoder -> LLM topology."""
    llm = mimo_cfg.llm_module_name
    return {name: [llm] for name in mimo_cfg.module_names if name != llm} | {llm: []}


def build_colocated_comm_config(
    mimo_cfg: MIMOConfig, grids: Dict[str, "HyperCommGrid"]
) -> "ColocatedCommConfig":
    """Build ColocatedCommConfig with default encoder-to-LLM topology."""
    from megatron.core.models.mimo.config.base_configs import ColocatedCommConfig

    module_to_grid_map = {name: grid for name, grid in grids.items()}
    topology = _default_topology(mimo_cfg)
    return ColocatedCommConfig(
        module_to_grid_map=module_to_grid_map,
        topology=topology,
        dim_mapping={"b": 0, "s": 1, "h": 2},
    )


def build_mimo_model_config(
    cfg: ConfigContainer,
    *,
    language_model_spec: "ModuleSpec",
    modality_submodules_spec: Dict[str, "ModuleSpec"],
    colocated_comm_config: Optional["ColocatedCommConfig"] = None,
) -> "MimoModelConfig":
    """Build MCore MimoModelConfig from Bridge config and ModuleSpecs."""
    if cfg.mimo is None:
        raise ValueError("MIMO configuration is not set.")

    from megatron.core.models.mimo.config.base_configs import MimoModelConfig

    mimo_cfg = cfg.mimo
    if colocated_comm_config is None and mimo_cfg.deployment_mode == "colocated":
        grids = build_hypercomm_grids(cfg)
        colocated_comm_config = build_colocated_comm_config(mimo_cfg, grids)

    return MimoModelConfig(
        language_model_spec=language_model_spec,
        modality_submodules_spec=modality_submodules_spec,
        special_token_ids=mimo_cfg.special_token_ids,
        colocated_comm_config=colocated_comm_config,
    )
