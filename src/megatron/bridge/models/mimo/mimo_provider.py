# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""MIMO Model Provider for heterogeneous multi-module training.

This module provides MimoModelProvider, which integrates with the standard
ModelProviderMixin interface to enable MIMO models in the training loop.

Key differences from standard providers:
- Uses HyperCommGrids for heterogeneous per-module parallelism
- Has separate build_infra() method for infrastructure metadata
- Overrides provide_distributed_model() for custom DDP handling
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, ClassVar, Dict, List, Optional, Union

import torch
import torch.distributed as dist
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.models.mimo import MimoModel
from megatron.core.models.mimo.config.base_configs import MimoModelConfig
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.utils import get_model_config

from megatron.bridge.models.mimo.mimo_builder import (
    _default_topology,
    build_hypercomm_grids,
    is_pp_first_stage,
    is_pp_last_stage,
    populate_embedding_and_position_groups,
)
from megatron.bridge.models.mimo.mimo_config import MimoParallelismConfig
from megatron.bridge.models.mimo.mimo_ddp import wrap_mimo_model_distributed
from megatron.bridge.models.model_provider import ModelProviderMixin


if TYPE_CHECKING:
    from megatron.core.hyper_comm_grid import HyperCommGrid


@dataclass
class MimoModelInfra:
    """MIMO infrastructure metadata (separate from model).

    This dataclass contains the parallelism infrastructure that MIMO builds,
    separated from the model itself to maintain the standard provide() contract.

    Attributes:
        module_to_grid_map: Mapping of module names to their HyperCommGrids.
        topology: DAG of module data flow (module_name -> list of downstream modules).
        pg_collections: Mapping of module names to ProcessGroupCollections.
            None for modules this rank doesn't participate in.
        participating_modules: List of module names this rank participates in.
    """

    module_to_grid_map: Dict[str, "HyperCommGrid"]
    topology: Dict[str, List[str]]
    pg_collections: Dict[str, Optional[ProcessGroupCollection]]
    participating_modules: List[str]


@dataclass
class MimoModelProvider(ModelProviderMixin[MimoModel]):
    """MIMO provider with heterogeneous parallelism support.

    Integrates with the standard training loop via provide_distributed_model().
    Use build_infra() to access MIMO-specific infrastructure (grids, topology, pg_collections).

    This provider handles:
    - HyperCommGrid creation per module (heterogeneous parallelism)
    - ProcessGroupCollection extraction from grids
    - pg_collection injection into specs
    - Rank participation checking
    - Freezing logic

    **Per-Encoder Parallelism:**
    To use different parallelism for each encoder, treat each encoder as a
    separate module in both `modality_submodules_spec` and `mimo_parallelism_config`:

    Example:
        >>> mimo_parallelism_config = MimoParallelismConfig(
        ...     module_parallelisms={
        ...         "llm": ModuleParallelismConfig(tensor_model_parallel_size=8),
        ...         "clip_encoder": ModuleParallelismConfig(tensor_model_parallel_size=2),
        ...     }
        ... )
        >>> provider = MimoModelProvider(
        ...     language_model_spec=gpt_spec,
        ...     modality_submodules_spec={"clip_encoder": clip_spec},
        ...     mimo_parallelism_config=mimo_parallelism_config,
        ... )
        >>> # For training loop integration:
        >>> model = provider.provide_distributed_model(ddp_config=ddp_config)
        >>> # Or for manual usage:
        >>> model = provider.provide()
        >>> infra = provider.build_infra()
    """

    LANGUAGE_MODULE_KEY: ClassVar[str] = "llm"

    # Model specs (user provides, like llava_vlm.py example)
    language_model_spec: ModuleSpec
    modality_submodules_spec: Dict[str, ModuleSpec] = field(default_factory=dict)
    special_token_ids: Dict[str, int] = field(default_factory=dict)

    mimo_parallelism_config: Optional[MimoParallelismConfig] = None

    # Cached grids after build_model() - used by data loading
    _grids: Optional[Dict[str, "HyperCommGrid"]] = field(default=None, repr=False)

    # Freezing options
    freeze_language_model: bool = False
    freeze_modality_encoders: Dict[str, bool] = field(default_factory=dict)
    freeze_modality_projections: Dict[str, bool] = field(default_factory=dict)

    # Fields required by ModelProviderMixin / get_model()
    # These have sensible defaults for MIMO
    fp16: bool = False
    bf16: bool = True
    use_cpu_initialization: bool = False
    init_model_with_meta_device: bool = False
    virtual_pipeline_model_parallel_size: Optional[int] = None

    @property
    def tensor_model_parallel_size(self) -> int:
        """Return LLM's tensor parallel size for compatibility with standard code paths."""
        if self.mimo_parallelism_config is None:
            return 1
        llm_parallelism = self.mimo_parallelism_config.get_parallelism(self.LANGUAGE_MODULE_KEY)
        return llm_parallelism.tensor_model_parallel_size

    @property
    def pipeline_model_parallel_size(self) -> int:
        """Return LLM's pipeline parallel size for compatibility with standard code paths."""
        if self.mimo_parallelism_config is None:
            return 1
        llm_parallelism = self.mimo_parallelism_config.get_parallelism(self.LANGUAGE_MODULE_KEY)
        return llm_parallelism.pipeline_model_parallel_size

    @property
    def context_parallel_size(self) -> int:
        """Return LLM's context parallel size for compatibility with standard code paths."""
        if self.mimo_parallelism_config is None:
            return 1
        llm_parallelism = self.mimo_parallelism_config.get_parallelism(self.LANGUAGE_MODULE_KEY)
        return llm_parallelism.context_parallel_size

    def build_infra(self) -> MimoModelInfra:
        """Build MIMO parallelism infrastructure.

        This method builds HyperCommGrids, ProcessGroupCollections, and topology
        for MIMO's heterogeneous parallelism. It is idempotent and does not
        mutate provider state (results are not cached).

        Can be called before or after provide(). Call finalize() first to
        validate the parallelism configuration.

        Returns:
            MimoModelInfra containing grids, topology, pg_collections,
            and the list of modules this rank participates in.
        """
        if self.mimo_parallelism_config is not None:
            grids = build_hypercomm_grids(self.mimo_parallelism_config)
            pg_collections = self._get_pg_collections_from_grids(grids)
            topology = _default_topology(self.mimo_parallelism_config)
        else:
            # No parallelism - use global process groups
            grids = {}
            pg_collections = {}
            topology = {}

        # Cache grids for later use (e.g., data loading)
        object.__setattr__(self, "_grids", grids)

        participating_modules = [name for name, pg in pg_collections.items() if pg is not None]

        return MimoModelInfra(
            module_to_grid_map=grids,
            topology=topology,
            pg_collections=pg_collections,
            participating_modules=participating_modules,
        )

    def _get_pg_collections_from_grids(
        self,
        grids: Dict[str, "HyperCommGrid"],
    ) -> Dict[str, Optional[ProcessGroupCollection]]:
        """Get ProcessGroupCollections from HyperCommGrids.

        Creates all standard process groups plus embedding groups for PP > 1.
        Returns None for modules this rank doesn't participate in.
        """
        pg_collections: Dict[str, Optional[ProcessGroupCollection]] = {}
        current_rank = dist.get_rank()

        for module_name, grid in grids.items():
            # Check if current rank is in this grid's range
            if grid.rank_offset <= current_rank < (grid.rank_offset + grid.size):
                pp_group = grid.get_pg(["pp"])

                # Create embedding groups for PP > 1 (collective operation on all PP ranks)
                pos_embd_pg, embd_pg = populate_embedding_and_position_groups(pp_group)

                # Only assign embedding groups to ranks that should have them
                first_stage = is_pp_first_stage(pp_group)
                last_stage = is_pp_last_stage(pp_group)

                pg_collections[module_name] = ProcessGroupCollection(
                    tp=grid.get_pg(["tp"]),
                    dp=grid.get_pg(["dp"]),
                    pp=pp_group,
                    cp=grid.get_pg(["cp"]),
                    ep=grid.get_pg(["ep"]),
                    dp_cp=grid.get_pg(["dp", "cp"]),
                    # Position embeddings only on first PP stage
                    pos_embd=pos_embd_pg if first_stage else None,
                    # Word embeddings on first and last PP stages (for tied embeddings)
                    embd=embd_pg if (first_stage or last_stage) else None,
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
        """Inject pg_collection into encoder specs within a modality submodule."""
        spec = copy.deepcopy(spec)

        # Inject into encoders
        if spec.submodules and "encoders" in spec.submodules:
            for _encoder_name, encoder_spec in spec.submodules["encoders"].items():
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
        pre_process: Optional[bool] = None,
        post_process: Optional[bool] = None,
        vp_stage: Optional[int] = None,
    ) -> MimoModel:
        """Build and return the MimoModel instance.

        This method follows the standard ModelProviderMixin.provide() contract,
        returning only the model instance. For infrastructure metadata (grids,
        topology, pg_collections), use build_infra() separately.

        Args:
            pre_process: Unused for MIMO (accepted for API compatibility).
            post_process: Unused for MIMO (accepted for API compatibility).
            vp_stage: Unused for MIMO (accepted for API compatibility).

        Returns:
            MimoModel instance.

        Note:
            Device/dtype handling is done by provide_distributed_model(),
            consistent with other providers. This method returns a CPU model.

        Raises:
            ValueError: If this rank doesn't participate in any module
                (indicates invalid parallelism configuration).
        """
        # Build infrastructure
        infra = self.build_infra()

        # Inject pg_collection into language model spec
        language_spec = self.language_model_spec
        if self.mimo_parallelism_config:
            llm_pg = infra.pg_collections.get(self.LANGUAGE_MODULE_KEY)
            if llm_pg is not None:
                language_spec = self._inject_pg_collection_into_language_spec(language_spec, llm_pg)

        # Inject pg_collection into modality specs
        modality_specs: Dict[str, ModuleSpec] = {}
        for module_name, spec in self.modality_submodules_spec.items():
            module_pg = infra.pg_collections.get(module_name) if infra.pg_collections else None
            if module_pg is not None:
                spec = self._inject_pg_collection_into_modality_spec(spec, module_pg)
            modality_specs[module_name] = spec

        # Create MimoModel
        grid_map = infra.module_to_grid_map if self.mimo_parallelism_config else None
        lang_key = self.LANGUAGE_MODULE_KEY if grid_map else None
        mimo_model_config = MimoModelConfig(
            language_model_spec=language_spec,
            modality_submodules_spec=modality_specs,
            special_token_ids=self.special_token_ids,
            module_to_grid_map=grid_map,
            language_module_key=lang_key,
        )

        mimo_model = MimoModel(mimo_model_config)

        # Apply freezing
        self._apply_freezing(mimo_model)

        return mimo_model

    def provide_distributed_model(
        self,
        ddp_config: Optional[DistributedDataParallelConfig] = None,
        model_type=None,
        overlap_param_gather_with_optimizer_step: bool = False,
        fp16: Optional[bool] = None,
        bf16: Optional[bool] = None,
        use_megatron_fsdp: bool = False,
        use_torch_fsdp2: bool = False,
        wrap_with_ddp: bool = True,
        data_parallel_random_init: bool = True,
        use_cpu_initialization: Optional[bool] = False,
        init_model_with_meta_device: Optional[bool] = None,
        pre_wrap_hook: Optional[
            Union[
                Callable[[List[MegatronModule]], List[MegatronModule]],
                List[Callable[[List[MegatronModule]], List[MegatronModule]]],
            ]
        ] = None,
        post_wrap_hook: Optional[Callable[[List[MegatronModule]], List[MegatronModule]]] = None,
        mixed_precision_wrapper: Optional[Callable] = None,
    ) -> List[MegatronModule]:
        """Build MIMO model with heterogeneous parallelism and DDP wrapping.

        This overrides the standard ModelProviderMixin implementation because MIMO:
        - Uses per-module HyperCommGrids instead of global mpu
        - Has different pg_collections per module
        - May have ranks that don't participate in all modules
        - Requires per-submodule DDP wrapping for correct gradient sync

        The method:
        1. Calls finalize() to validate parallelism config
        2. Calls build_infra() to create grids and pg_collections
        3. Calls provide() to build the model
        4. Applies pre-wrap hooks
        5. Moves to device
        6. Wraps each submodule with DDP using its own pg_collection
        7. Applies mixed precision (Float16Module)
        8. Applies post-wrap hooks

        Args:
            ddp_config: Configuration for distributed data parallel.
            model_type: Type of model (unused for MIMO, accepted for compatibility).
            overlap_param_gather_with_optimizer_step: Whether to overlap param gathering.
            fp16: Override FP16 setting.
            bf16: Override BF16 setting.
            use_megatron_fsdp: Use Megatron's Fully Sharded Data Parallel.
            use_torch_fsdp2: Use PyTorch FSDP2.
            wrap_with_ddp: Whether to wrap model with DDP.
            data_parallel_random_init: Initialize parameters randomly across DP ranks.
            use_cpu_initialization: Initialize model on CPU.
            init_model_with_meta_device: Initialize model on meta device.
            pre_wrap_hook: Callable(s) to modify model before wrapping.
            post_wrap_hook: Callable to modify model after wrapping.
            mixed_precision_wrapper: Wrapper for mixed precision (e.g., Float16Module).

        Returns:
            List containing the wrapped MimoModel.

        Raises:
            ValueError: If this rank doesn't participate in any module
                (indicates invalid parallelism configuration).
        """
        if wrap_with_ddp and ddp_config is None:
            raise ValueError("ddp_config is required when wrap_with_ddp is True")

        if use_megatron_fsdp or use_torch_fsdp2:
            raise NotImplementedError(
                "FSDP is not yet supported for MIMO models. Use DDP (wrap_with_ddp=True) instead."
            )

        # Finalize parallelism config
        self.finalize()

        # Build infrastructure
        infra = self.build_infra()

        # Get the model
        model = self.provide()
        model_list = [model]

        # Resolve hooks
        final_pre_wrap_hook = self._resolve_hooks(pre_wrap_hook)
        final_post_wrap_hook = post_wrap_hook or self.post_wrap_hook

        # Apply pre-wrap hooks
        if final_pre_wrap_hook:
            result = final_pre_wrap_hook(model_list)
            if result is not None:
                model_list = result

        # Move to device
        if not use_cpu_initialization and not init_model_with_meta_device:
            for m in model_list:
                m.cuda(torch.cuda.current_device())

        # Set variable_seq_lengths=True for multimodule pipeline support (required by PR 3129)
        # This must be set before the model is used in the training loop
        for m in model_list:
            model_config = get_model_config(m)
            model_config.variable_seq_lengths = True

        # Wrap submodules with DDP (before Float16Module)
        # MIMO uses per-submodule DDP for heterogeneous parallelism
        if wrap_with_ddp and ddp_config is not None and self.mimo_parallelism_config:
            model_list = [
                wrap_mimo_model_distributed(
                    mimo_model=m,
                    ddp_config=ddp_config,
                    mimo_parallelism_config=self.mimo_parallelism_config,
                    grids=infra.module_to_grid_map,
                    pg_collections=infra.pg_collections,
                )
                for m in model_list
            ]

        # Apply mixed precision: cast parameters directly instead of wrapping
        # with Float16Module. Float16Module requires a single pg_collection for
        # PP stage checks, but MimoModel has multiple modules with different PP
        # groups (and modules can be colocated). Direct casting avoids this issue;
        # input/output dtype handling is done by the data pipeline and loss fn.
        use_fp16 = fp16 if fp16 is not None else self.fp16
        use_bf16 = bf16 if bf16 is not None else self.bf16
        if use_fp16:
            model_list = [m.half() for m in model_list]
        elif use_bf16:
            model_list = [m.bfloat16() for m in model_list]

        # Apply post-wrap hooks
        if final_post_wrap_hook:
            result = final_post_wrap_hook(model_list)
            if result is not None:
                model_list = result

        return model_list

    def _resolve_hooks(
        self,
        pre_wrap_hook: Optional[
            Union[
                Callable[[List[MegatronModule]], List[MegatronModule]],
                List[Callable[[List[MegatronModule]], List[MegatronModule]]],
            ]
        ],
    ) -> Optional[Callable[[List[MegatronModule]], List[MegatronModule]]]:
        """Resolve pre-wrap hooks to a single callable."""
        if pre_wrap_hook is not None:
            if isinstance(pre_wrap_hook, list):

                def composed_hook(model: List[MegatronModule]) -> List[MegatronModule]:
                    for hook in pre_wrap_hook:
                        result = hook(model)
                        if result is not None:
                            model = result
                    return model

                return composed_hook
            return pre_wrap_hook
        return self.pre_wrap_hook

    def initialize_model_parallel(
        self,
        seed: Optional[int] = None,
        seed_kwargs: Optional[dict] = None,
        **model_parallel_kwargs,
    ) -> None:
        """MIMO uses its own parallelism via MimoParallelismConfig.

        This method is a no-op for MIMO. Parallelism is set up in build_infra()
        using HyperCommGrids, not global mpu state.

        Note:
            Call finalize() to validate the parallelism configuration, then
            build_infra() to create the HyperCommGrids.
        """
        # MIMO manages its own parallelism via HyperCommGrids
        pass

    def _apply_freezing(self, model: MimoModel) -> None:
        """Apply freezing based on configuration."""
        if self.freeze_language_model and hasattr(model, "language_model"):
            for param in model.language_model.parameters():
                param.requires_grad = False

        if hasattr(model, "modality_submodules"):
            for modality, should_freeze in self.freeze_modality_encoders.items():
                if should_freeze and modality in model.modality_submodules:
                    submodule = model.modality_submodules[modality]
                    if hasattr(submodule, "encoders"):
                        for param in submodule.encoders.parameters():
                            param.requires_grad = False

            for modality, should_freeze in self.freeze_modality_projections.items():
                if should_freeze and modality in model.modality_submodules:
                    submodule = model.modality_submodules[modality]
                    if hasattr(submodule, "input_projections"):
                        for param in submodule.input_projections.parameters():
                            param.requires_grad = False

    def finalize(self) -> None:
        """Finalize MIMO parallelism configuration.

        This validates the parallelism config and should be called before
        build_infra() or provide(). It is called automatically by
        provide_distributed_model().

        Raises:
            ValueError: If any rank doesn't participate in at least one module.
                This indicates the parallelism configuration doesn't cover all
                ranks in the world (validated by MimoParallelismConfig.finalize()).
        """
        if self.mimo_parallelism_config is not None:
            world_size = dist.get_world_size() if dist.is_initialized() else None
            self.mimo_parallelism_config.finalize(world_size)
