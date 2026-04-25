# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""MegatronMIMO Model Provider for heterogeneous multi-module training.

This module provides MegatronMIMOProvider, which integrates with the standard
ModelProviderMixin interface to enable multi-module models in the training loop.

Key differences from standard providers:
- Uses HyperCommGrids for heterogeneous per-module parallelism
- Has separate build_infra() method for infrastructure metadata
- Overrides provide_distributed_model() for custom DDP handling
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Union

import torch
import torch.distributed as dist
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.enums import ModelType
from megatron.core.models.mimo import MimoModel
from megatron.core.models.mimo.config.base_configs import MimoModelConfig
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.utils import get_model_config

from megatron.bridge.models.megatron_mimo.megatron_mimo_builder import (
    build_hypercomm_grids,
    is_pp_first_stage,
    is_pp_last_stage,
    populate_embedding_and_position_groups,
)
from megatron.bridge.models.megatron_mimo.megatron_mimo_config import MegatronMIMOParallelismConfig
from megatron.bridge.models.megatron_mimo.megatron_mimo_ddp import wrap_megatron_mimo_model_distributed
from megatron.bridge.models.model_provider import ModelProviderMixin


if TYPE_CHECKING:
    from megatron.core.hyper_comm_grid import HyperCommGrid


@dataclass
class MegatronMIMOInfra:
    """MegatronMIMO infrastructure metadata (separate from model).

    This dataclass contains the parallelism infrastructure that MegatronMIMO builds,
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
    module_output_ndim: Dict[str, int] = field(default_factory=dict)


@dataclass
class MegatronMIMOProvider(ModelProviderMixin[MimoModel]):
    """MegatronMIMO provider with heterogeneous parallelism support.

    Integrates with the standard training loop via provide_distributed_model().
    Use build_infra() to access MegatronMIMO-specific infrastructure (grids, topology, pg_collections).

    This provider handles:
    - HyperCommGrid creation per module (heterogeneous parallelism)
    - ProcessGroupCollection extraction from grids
    - pg_collection injection into specs
    - Rank participation checking
    - Freezing logic

    **Per-Encoder Parallelism:**
    To use different parallelism for each encoder, treat each encoder as a
    separate module in both `modality_submodules_spec` and `megatron_mimo_parallelism_config`:

    Example:
        >>> megatron_mimo_parallelism_config = MegatronMIMOParallelismConfig(
        ...     module_parallelisms={
        ...         "language": ModuleParallelismConfig(tensor_model_parallel_size=8),
        ...         "clip_encoder": ModuleParallelismConfig(tensor_model_parallel_size=2),
        ...     }
        ... )
        >>> provider = MegatronMIMOProvider(
        ...     language_model_spec=gpt_spec,
        ...     modality_submodules_spec={"clip_encoder": clip_spec},
        ...     megatron_mimo_parallelism_config=megatron_mimo_parallelism_config,
        ... )
        >>> # For training loop integration:
        >>> model = provider.provide_distributed_model(ddp_config=ddp_config)
        >>> # Or for manual usage:
        >>> model = provider.provide()
        >>> infra = provider.build_infra()
    """

    # Model specs (user provides, like llava_vlm.py example).
    # Optional so subclasses (e.g. LlavaMegatronMIMOProvider) can build it in __post_init__.
    language_model_spec: Optional[ModuleSpec] = None
    modality_submodules_spec: Dict[str, ModuleSpec] = field(default_factory=dict)
    special_token_ids: Dict[str, int] = field(default_factory=dict)

    megatron_mimo_parallelism_config: Optional[MegatronMIMOParallelismConfig] = None

    # Module data-flow DAG for MultiModulePipelineCommunicator.
    # If None, auto-derived as: all modality_submodules → MIMO_LANGUAGE_MODULE_KEY (terminal).
    # Set explicitly for non-standard topologies (e.g., language → generator).
    topology: Optional[Dict[str, List[str]]] = None

    # Output tensor dimensionality per module for bridge communicator routing.
    # Vision/audio encoders typically produce 2D [S, H]; language modules produce 3D [S, B, H].
    # If None, auto-derived: language module → 3, all others → 2.
    module_output_ndim: Optional[Dict[str, int]] = None

    # Cached grids after build_model() - used by data loading
    _grids: Optional[Dict[str, "HyperCommGrid"]] = field(default=None, repr=False)

    # Freezing options
    freeze_language_model: bool = False
    freeze_modality_encoders: Dict[str, bool] = field(default_factory=dict)
    freeze_modality_projections: Dict[str, bool] = field(default_factory=dict)

    # Fields required by ModelProviderMixin / get_model()
    fp16: bool = False
    bf16: bool = True
    use_cpu_initialization: bool = False
    init_model_with_meta_device: bool = False

    def build_infra(self) -> MegatronMIMOInfra:
        """Build MegatronMIMO parallelism infrastructure.

        This method builds HyperCommGrids, ProcessGroupCollections, and topology
        for MegatronMIMO's heterogeneous parallelism. It is idempotent and does not
        mutate provider state (results are not cached).

        Can be called before or after provide(). Call finalize() first to
        validate the parallelism configuration.

        Returns:
            MegatronMIMOInfra containing grids, topology, pg_collections,
            and the list of modules this rank participates in.
        """
        if self.megatron_mimo_parallelism_config is not None:
            grids = build_hypercomm_grids(self.megatron_mimo_parallelism_config)
            pg_collections = self._get_pg_collections_from_grids(grids)
        else:
            grids = {}
            pg_collections = {}

        if self.topology is not None:
            topology = self.topology
        else:
            topology = {name: [MIMO_LANGUAGE_MODULE_KEY] for name in self.modality_submodules_spec} | {
                MIMO_LANGUAGE_MODULE_KEY: []
            }

        # Cache grids for later use (e.g., data loading)
        object.__setattr__(self, "_grids", grids)

        participating_modules = [name for name, pg in pg_collections.items() if pg is not None]

        # Derive module output tensor dimensionality if not explicitly configured.
        # Language module produces 3D [S, B, H]; modality encoders produce 2D [S, H].
        if self.module_output_ndim is not None:
            output_ndim = self.module_output_ndim
        else:
            output_ndim = {name: 3 if name == MIMO_LANGUAGE_MODULE_KEY else 2 for name in grids}

        return MegatronMIMOInfra(
            module_to_grid_map=grids,
            topology=topology,
            pg_collections=pg_collections,
            participating_modules=participating_modules,
            module_output_ndim=output_ndim,
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

        for module_name, grid in grids.items():
            pp_group = grid.get_pg(["pp"])

            # dist.new_group() is a collective on the default PG — all ranks must
            # call it in the same global order regardless of module membership.
            pos_embd_pg, embd_pg = populate_embedding_and_position_groups(pp_group)

            # Only build a full PG collection for ranks that participate in this module.
            if grid.is_current_rank_in_grid():
                first_stage = is_pp_first_stage(pp_group)
                last_stage = is_pp_last_stage(pp_group)

                pg_collections[module_name] = ProcessGroupCollection(
                    tp=grid.get_pg(["tp"]),
                    dp=grid.get_pg(["dp"]),
                    pp=pp_group,
                    cp=grid.get_pg(["cp"]),
                    ep=grid.get_pg(["ep"]),
                    dp_cp=grid.get_pg(["dp", "cp"]),
                    mp=grid.get_pg(["tp", "pp"]),
                    tp_ep_pp=grid.get_pg(["tp", "ep", "pp"]),
                    pos_embd=pos_embd_pg if first_stage else None,
                    embd=embd_pg if (first_stage or last_stage) else None,
                )
            else:
                pg_collections[module_name] = None

        return pg_collections

    def _inject_pg_collection_into_language_spec(
        self,
        spec: ModuleSpec,
        pg_collection: ProcessGroupCollection,
        pre_process: Optional[bool] = None,
        post_process: Optional[bool] = None,
    ) -> ModuleSpec:
        """Deep copy language model spec and inject stage-aware params."""
        spec = copy.deepcopy(spec)
        if spec.params is None:
            spec.params = {}
        spec.params["pg_collection"] = pg_collection
        if pre_process is not None:
            spec.params["pre_process"] = pre_process
        if post_process is not None:
            spec.params["post_process"] = post_process
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
            pre_process: Unused for MegatronMIMO (accepted for API compatibility).
            post_process: Unused for MegatronMIMO (accepted for API compatibility).
            vp_stage: Unused for MegatronMIMO (accepted for API compatibility).

        Returns:
            MimoModel instance.

        Note:
            Device/dtype handling is done by provide_distributed_model(),
            consistent with other providers. This method returns a CPU model.

        Raises:
            ValueError: If language_model_spec is not set, or if this rank
                doesn't participate in any module.
        """
        if self.language_model_spec is None:
            raise ValueError(
                "language_model_spec must be set before calling provide(). "
                "Set it directly or use a subclass that populates it in __post_init__."
            )

        # Build infrastructure
        infra = self.build_infra()

        # Inject pg_collection into language model spec
        language_spec = self.language_model_spec
        llm_pg = None
        if self.megatron_mimo_parallelism_config:
            llm_pg = infra.pg_collections.get(MIMO_LANGUAGE_MODULE_KEY)
            if llm_pg is not None:
                language_spec = self._inject_pg_collection_into_language_spec(
                    language_spec,
                    llm_pg,
                    pre_process=is_pp_first_stage(llm_pg.pp),
                    post_process=is_pp_last_stage(llm_pg.pp),
                )

        # Inject pg_collection into modality specs
        modality_specs: Dict[str, ModuleSpec] = {}
        for module_name, spec in self.modality_submodules_spec.items():
            module_pg = infra.pg_collections.get(module_name) if infra.pg_collections else None
            if module_pg is not None:
                spec = self._inject_pg_collection_into_modality_spec(spec, module_pg)
            modality_specs[module_name] = spec

        # Create MimoModel
        mimo_model_config = MimoModelConfig(
            language_model_spec=language_spec,
            modality_submodules_spec=modality_specs,
            special_token_ids=self.special_token_ids,
            module_to_grid_map=(
                infra.module_to_grid_map if self.megatron_mimo_parallelism_config is not None else None
            ),
        )

        # Thread the LLM's CP and TP groups explicitly so MimoModel's PartitionAdapter
        # (built when language CP>1 or sequence_parallel=True) binds to the correct
        # groups instead of falling back to uninitialised global parallel_state.
        cp_group = llm_pg.cp if llm_pg is not None else None
        tp_group = llm_pg.tp if llm_pg is not None else None
        megatron_mimo_model = MimoModel(mimo_model_config, cp_group=cp_group, tp_group=tp_group)

        # Set model_type so mcore schedules can introspect it
        # (forward_backward_no_pipelining calls get_model_type via get_attr_wrapped_model).
        megatron_mimo_model.model_type = ModelType.encoder_or_decoder

        # Apply freezing
        self._apply_freezing(megatron_mimo_model)

        return megatron_mimo_model

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
        use_cpu_initialization: Optional[bool] = None,
        init_model_with_meta_device: Optional[bool] = None,
        pre_wrap_hook: Optional[
            Union[
                Callable[[List[MegatronModule]], List[MegatronModule]],
                List[Callable[[List[MegatronModule]], List[MegatronModule]]],
            ]
        ] = None,
        post_wrap_hook: Optional[Callable[[List[MegatronModule]], List[MegatronModule]]] = None,
    ) -> List[MegatronModule]:
        """Build MegatronMIMO model with heterogeneous parallelism and DDP wrapping.

        This overrides the standard ModelProviderMixin implementation because MegatronMIMO:
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
        7. Casts to fp16/bf16 (direct casting, not Float16Module)
        8. Applies post-wrap hooks

        Args:
            ddp_config: Configuration for distributed data parallel.
            model_type: Type of model (unused for MegatronMIMO, accepted for compatibility).
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
                "FSDP is not yet supported for MegatronMIMO models. Use DDP (wrap_with_ddp=True) instead."
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

        # Resolve initialization settings from provider defaults if not specified
        local_use_cpu_init = (
            use_cpu_initialization if use_cpu_initialization is not None else self.use_cpu_initialization
        )
        local_init_meta_device = (
            init_model_with_meta_device
            if init_model_with_meta_device is not None
            else self.init_model_with_meta_device
        )

        # Move to device
        if not local_use_cpu_init and not local_init_meta_device:
            for m in model_list:
                m.cuda(torch.cuda.current_device())

        # Set variable_seq_lengths=True for multimodule pipeline support (required by PR 3212)
        # This must be set before the model is used in the training loop
        for m in model_list:
            model_config = get_model_config(m)
            model_config.variable_seq_lengths = True

        # Dtype cast must precede DDP wrapping so hooks bind to final parameters.
        use_fp16 = fp16 if fp16 is not None else self.fp16
        use_bf16 = bf16 if bf16 is not None else self.bf16
        if use_fp16:
            model_list = [m.half() for m in model_list]
        elif use_bf16:
            model_list = [m.bfloat16() for m in model_list]

        # Ensure frozen parameters are on GPU before DDP wrapping.
        # DDP only manages requires_grad=True params, so frozen ones must be
        # moved explicitly (especially when use_cpu_initialization=True).
        for m in model_list:
            self._move_frozen_params_to_device(m)

        # Per-submodule DDP for heterogeneous parallelism
        if wrap_with_ddp and ddp_config is not None and self.megatron_mimo_parallelism_config:
            model_list = [
                wrap_megatron_mimo_model_distributed(
                    megatron_mimo_model=m,
                    ddp_config=ddp_config,
                    megatron_mimo_parallelism_config=self.megatron_mimo_parallelism_config,
                    grids=infra.module_to_grid_map,
                    pg_collections=infra.pg_collections,
                )
                for m in model_list
            ]

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
        """MegatronMIMO uses per-module HyperCommGrids, not global MPU state.

        Raises NotImplementedError to prevent accidental global MPU initialization,
        which would corrupt process groups for heterogeneous parallelism.
        Use finalize() + build_infra() instead.
        """
        raise NotImplementedError(
            "MegatronMIMO does not use global model parallelism initialization. "
            "Use finalize() to validate config and build_infra() to create HyperCommGrids."
        )

    def _apply_freezing(self, model: MimoModel) -> None:
        """Apply freezing based on configuration."""
        if self.freeze_language_model and getattr(model, "language_model", None) is not None:
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

    @staticmethod
    def _move_frozen_params_to_device(model: torch.nn.Module) -> None:
        """Move frozen parameters and buffers to the current CUDA device.

        When ``use_cpu_initialization=True`` the global ``.cuda()`` call is
        skipped, and DDP only moves parameters with ``requires_grad=True``.
        This leaves frozen parameters and buffers stranded on CPU.  Call this
        after all hooks (e.g. checkpoint loading) have run but before DDP
        wrapping.
        """
        if not torch.cuda.is_available():
            return
        device = torch.cuda.current_device()
        for param in model.parameters():
            if not param.requires_grad and param.device.type == "cpu":
                param.data = param.data.to(device)
        for buf in model.buffers():
            if buf.device.type == "cpu":
                buf.data = buf.data.to(device)

    def finalize(self) -> None:
        """Finalize MegatronMIMO parallelism configuration.

        This validates the parallelism config and should be called before
        build_infra() or provide(). It is called automatically by
        provide_distributed_model().

        Raises:
            ValueError: If any rank doesn't participate in at least one module.
                This indicates the parallelism configuration doesn't cover all
                ranks in the world (validated by MegatronMIMOParallelismConfig.finalize()).
        """
        if self.megatron_mimo_parallelism_config is not None:
            if not dist.is_initialized():
                raise RuntimeError(
                    "MegatronMIMO requires torch.distributed to be initialized before finalize(). "
                    "Call torch.distributed.init_process_group() first."
                )
            self.megatron_mimo_parallelism_config.finalize(dist.get_world_size())
