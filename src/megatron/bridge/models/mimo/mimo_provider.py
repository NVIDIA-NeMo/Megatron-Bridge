# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""MIMO Model Provider for heterogeneous multi-module training."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

import torch
import torch.distributed as dist

from megatron.core.models.mimo import MimoModel
from megatron.core.models.mimo.config.base_configs import MimoModelConfig, ColocatedCommConfig
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.spec_utils import ModuleSpec

from megatron.bridge.models.mimo.mimo_config import MimoParallelismConfig
from megatron.bridge.models.mimo.mimo_builder import (
    build_hypercomm_grids,
    build_colocated_comm_config,
    _default_topology,
)

if TYPE_CHECKING:
    from megatron.core.hyper_comm_grid import HyperCommGrid


@dataclass
class MimoModelProviderResult:
    """Result container for MimoModelProvider.provide().
    
    Attributes:
        model: The constructed MimoModel (None on non-participating ranks).
        module_to_grid_map: Mapping of module names to their HyperCommGrids.
        topology: DAG of module data flow.
        pg_collections: Mapping of module names to ProcessGroupCollections.
    """
    model: Optional[MimoModel]
    module_to_grid_map: Dict[str, "HyperCommGrid"]
    topology: Dict[str, List[str]]
    pg_collections: Dict[str, Optional[ProcessGroupCollection]]


@dataclass
class MimoModelProvider:
    """Unified MIMO provider with parallelism support.
    
    Bridge provider that adds parallelism infrastructure to MCore's MIMO models.
    Users provide model specs (like MCore examples), and this provider handles:
    - HyperCommGrid creation per module
    - ProcessGroupCollection extraction from grids
    - pg_collection injection into specs
    - Rank participation checking
    - Freezing logic
    
    **Per-Encoder Parallelism:**
    To use different parallelism for each encoder, treat each encoder as a 
    separate module in both `modality_submodules_spec` and `mimo_parallelism_config`:
    
    Example:
        >>> # Define different parallelism per encoder
        >>> mimo_parallelism_config = MimoParallelismConfig(
        ...     llm_module_name="llm",
        ...     module_parallelisms={
        ...         "llm": ModuleParallelismConfig(tensor_parallel=8),
        ...         "clip_encoder": ModuleParallelismConfig(tensor_parallel=2),
        ...         "dino_encoder": ModuleParallelismConfig(tensor_parallel=4),
        ...         "audio_encoder": ModuleParallelismConfig(tensor_parallel=1),
        ...     }
        ... )
        >>> 
        >>> # Create separate specs for each encoder
        >>> provider = MimoModelProvider(
        ...     language_model_spec=gpt_spec,
        ...     modality_submodules_spec={
        ...         "clip_encoder": clip_vision_spec,  # Gets TP=2
        ...         "dino_encoder": dino_vision_spec,  # Gets TP=4
        ...         "audio_encoder": audio_spec,       # Gets TP=1
        ...     },
        ...     special_token_ids={
        ...         "clip_encoder": 32000,
        ...         "dino_encoder": 32001,
        ...         "audio_encoder": 32002,
        ...     },
        ...     mimo_parallelism_config=mimo_parallelism_config,
        ... )
        >>> result = provider.provide()
    """
    
    # Model specs (user provides, like llava_vlm.py example)
    # Note: Each key in modality_submodules_spec should match a module name
    # in mimo_parallelism_config.module_parallelisms for per-encoder parallelism
    language_model_spec: ModuleSpec
    modality_submodules_spec: Dict[str, ModuleSpec] = field(default_factory=dict)
    special_token_ids: Dict[str, int] = field(default_factory=dict)
    
    # Parallelism config (Bridge's value-add)
    mimo_parallelism_config: Optional[MimoParallelismConfig] = None
    
    # Freezing options
    freeze_language_model: bool = False
    freeze_modality_encoders: Dict[str, bool] = field(default_factory=dict)
    freeze_modality_projections: Dict[str, bool] = field(default_factory=dict)
    
    @property
    def tensor_model_parallel_size(self) -> int:
        """Return LLM's tensor parallel size for compatibility with standard code paths."""
        if self.mimo_parallelism_config is None:
            return 1
        llm_parallelism = self.mimo_parallelism_config.get_parallelism(
            self.mimo_parallelism_config.llm_module_name
        )
        return llm_parallelism.tensor_parallel

    @property
    def pipeline_model_parallel_size(self) -> int:
        """Return LLM's pipeline parallel size for compatibility with standard code paths."""
        if self.mimo_parallelism_config is None:
            return 1
        llm_parallelism = self.mimo_parallelism_config.get_parallelism(
            self.mimo_parallelism_config.llm_module_name
        )
        return llm_parallelism.pipeline_parallel

    @property
    def context_parallel_size(self) -> int:
        """Return LLM's context parallel size for compatibility with standard code paths."""
        if self.mimo_parallelism_config is None:
            return 1
        llm_parallelism = self.mimo_parallelism_config.get_parallelism(
            self.mimo_parallelism_config.llm_module_name
        )
        return llm_parallelism.context_parallel

    def _get_pg_collections_from_grids(
        self,
        grids: Dict[str, "HyperCommGrid"],
    ) -> Dict[str, Optional[ProcessGroupCollection]]:
        """Get ProcessGroupCollections from HyperCommGrids.
        
        Returns None for modules this rank doesn't participate in.
        """
        pg_collections: Dict[str, Optional[ProcessGroupCollection]] = {}
        current_rank = dist.get_rank()
        
        for module_name, grid in grids.items():
            # Check if current rank is in this grid's range
            if grid.rank_offset <= current_rank < (grid.rank_offset + grid.size):
                pg_collections[module_name] = ProcessGroupCollection(
                    tp=grid.get_pg(["tp"]),
                    dp=grid.get_pg(["dp"]),
                    pp=grid.get_pg(["pp"]),
                    cp=grid.get_pg(["cp"]),
                    ep=grid.get_pg(["ep"]),
                    dp_cp=grid.get_pg(["dp", "cp"]),
                )
            else:
                pg_collections[module_name] = None
        
        return pg_collections
    
    def _inject_pg_collection_into_language_spec(
        self,
        spec: ModuleSpec,
        pg_collection: ProcessGroupCollection,
    ) -> ModuleSpec:
        """Deep copy language model spec and inject pg_collection into params."""
        spec = copy.deepcopy(spec)
        if spec.params is None:
            spec.params = {}
        spec.params["pg_collection"] = pg_collection
        return spec
    
    def _inject_pg_collection_into_modality_spec(
        self,
        spec: ModuleSpec,
        pg_collection: ProcessGroupCollection,
    ) -> ModuleSpec:
        """Inject pg_collection into encoder specs within a modality submodule.
        
        Note: For per-encoder parallelism, each modality spec should contain
        only ONE encoder in its submodules["encoders"] dict. This allows each
        encoder to get its own pg_collection based on its module name.
        
        Example:
            # Single encoder per modality spec
            modality_submodules_spec = {
                "clip_encoder": ModuleSpec(
                    submodules={"encoders": {"clip": clip_spec}}  # Only one
                ),
                "dino_encoder": ModuleSpec(
                    submodules={"encoders": {"dino": dino_spec}}  # Only one
                ),
            }
        """
        spec = copy.deepcopy(spec)
        
        # Inject into encoders
        if spec.submodules and "encoders" in spec.submodules:
            for encoder_name, encoder_spec in spec.submodules["encoders"].items():
                if encoder_spec.params is None:
                    encoder_spec.params = {}
                encoder_spec.params["pg_collection"] = pg_collection
        
        # Inject tp_group into projections
        if spec.submodules and "input_projections" in spec.submodules:
            for proj_spec in spec.submodules["input_projections"]:
                if isinstance(proj_spec, ModuleSpec):
                    if proj_spec.params is None:
                        proj_spec.params = {}
                    if "tp_group" not in proj_spec.params:
                        proj_spec.params["tp_group"] = pg_collection.tp
        
        return spec
    
    def provide(
        self,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> MimoModelProviderResult:
        """Build and return the MimoModel with parallelism infrastructure.
        
        Args:
            device: Device to move model to. Defaults to cuda.
            dtype: Data type for model parameters. Defaults to bfloat16.
            
        Returns:
            MimoModelProviderResult with model and infrastructure.
            model is None if this rank doesn't participate in any module.
        """
        if device is None:
            device = torch.device("cuda")
        
        # Step 1: Build HyperCommGrids (if config provided)
        if self.mimo_parallelism_config is not None:
            grids = build_hypercomm_grids(self.mimo_parallelism_config)
            pg_collections = self._get_pg_collections_from_grids(grids)
            topology = _default_topology(self.mimo_parallelism_config)
        else:
            # No parallelism - use global process groups (like llava_vlm.py example)
            grids = {}
            pg_collections = {}
            topology = {}
        
        # Step 2: Check rank participation
        participating_modules = [
            name for name, pg in pg_collections.items() if pg is not None
        ]
        
        if self.mimo_parallelism_config and not participating_modules:
            # This rank doesn't participate in any module
            return MimoModelProviderResult(
                model=None,
                module_to_grid_map=grids,
                topology=topology,
                pg_collections=pg_collections,
            )
        
        # Step 3: Inject pg_collection into language model spec
        language_spec = self.language_model_spec
        if self.mimo_parallelism_config:
            llm_name = self.mimo_parallelism_config.llm_module_name
            llm_pg = pg_collections.get(llm_name)
            if llm_pg is not None:
                language_spec = self._inject_pg_collection_into_language_spec(
                    language_spec, llm_pg
                )
        
        # Step 4: Inject pg_collection into modality specs
        # Each module_name should match an entry in mimo_parallelism_config.module_parallelisms
        # This enables per-encoder parallelism (e.g., clip_encoder with TP=2, dino_encoder with TP=4)
        modality_specs: Dict[str, ModuleSpec] = {}
        for module_name, spec in self.modality_submodules_spec.items():
            module_pg = pg_collections.get(module_name) if pg_collections else None
            if module_pg is not None:
                spec = self._inject_pg_collection_into_modality_spec(spec, module_pg)
            modality_specs[module_name] = spec
        
        # Step 5: Build colocated comm config if needed
        # Note: Both "colocated" and "homogeneous" use ColocatedCommConfig because
        # modules share the same ranks (rank_offset=0) and need TP resharding.
        # "heterogeneous" uses pipeline's set_input_tensor() for inter-rank communication.
        colocated_comm_config: Optional[ColocatedCommConfig] = None
        if (self.mimo_parallelism_config and 
            self.mimo_parallelism_config.deployment_mode in ("colocated", "homogeneous")):
            colocated_comm_config = build_colocated_comm_config(
                self.mimo_parallelism_config, grids
            )
        
        # Step 6: Create MimoModel using MCore's config
        mimo_model_config = MimoModelConfig(
            language_model_spec=language_spec,
            modality_submodules_spec=modality_specs,
            special_token_ids=self.special_token_ids,
            colocated_comm_config=colocated_comm_config,
        )
        
        mimo_model = MimoModel(mimo_model_config)
        
        # Step 7: Apply freezing
        self._apply_freezing(mimo_model)
        
        # Step 8: Move to device/dtype
        mimo_model.to(device).to(dtype)
        
        return MimoModelProviderResult(
            model=mimo_model,
            module_to_grid_map=grids,
            topology=topology,
            pg_collections=pg_collections,
        )
    
    def _apply_freezing(self, model: MimoModel) -> None:
        """Apply freezing based on configuration."""
        if self.freeze_language_model and hasattr(model, 'language_model'):
            for param in model.language_model.parameters():
                param.requires_grad = False
        
        if hasattr(model, 'modality_submodules'):
            for modality, should_freeze in self.freeze_modality_encoders.items():
                if should_freeze and modality in model.modality_submodules:
                    submodule = model.modality_submodules[modality]
                    if hasattr(submodule, 'encoders'):
                        for param in submodule.encoders.parameters():
                            param.requires_grad = False
            
            for modality, should_freeze in self.freeze_modality_projections.items():
                if should_freeze and modality in model.modality_submodules:
                    submodule = model.modality_submodules[modality]
                    if hasattr(submodule, 'input_projections'):
                        for param in submodule.input_projections.parameters():
                            param.requires_grad = False

    def finalize(self) -> None:
        """Finalize MIMO parallelism configuration."""
        if self.mimo_parallelism_config is not None:
            world_size = dist.get_world_size() if dist.is_initialized() else None
            self.mimo_parallelism_config.finalize(world_size)
