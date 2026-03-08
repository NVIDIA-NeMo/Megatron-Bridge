# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Union

import torch
import torch.distributed as dist
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.models.mimo import MimoModel
from megatron.core.models.mimo.config.base_configs import MimoModelConfig
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import Float16Module, MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.utils import get_model_config

from megatron.bridge.models.mimo.mimo_builder import (
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

_SENTINEL = object()


class _MimoConfigProxy:
    """Proxy that wraps TransformerConfig and falls back to MimoModelProvider.

    In homogeneous mode, ``get_model_config(model[0])`` returns the raw
    TransformerConfig, but the training stack also accesses provider-level
    attributes (``seq_length``, ``make_vocab_size_divisible_by``, …) on it.
    This proxy transparently resolves those by trying the TransformerConfig
    first and falling back to the provider.
    """

    def __init__(self, transformer_config, provider):
        object.__setattr__(self, "_tc", transformer_config)
        object.__setattr__(self, "_provider", provider)

    def __getattr__(self, name):
        tc = object.__getattribute__(self, "_tc")
        try:
            return getattr(tc, name)
        except AttributeError:
            return getattr(object.__getattribute__(self, "_provider"), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_tc"), name, value)

    def __repr__(self):
        tc = object.__getattribute__(self, "_tc")
        return f"_MimoConfigProxy({tc!r})"


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

    # Model specs (user provides, like llava_vlm.py example).
    # Optional so subclasses (e.g. LlavaMimoProvider) can build it in __post_init__.
    language_model_spec: Optional[ModuleSpec] = None
    modality_submodules_spec: Dict[str, ModuleSpec] = field(default_factory=dict)
    special_token_ids: Dict[str, int] = field(default_factory=dict)

    mimo_parallelism_config: Optional[MimoParallelismConfig] = None

    # Module data-flow DAG for MultiModulePipelineCommunicator.
    # If None, auto-derived as: all modality_submodules → "llm" (terminal).
    # Set explicitly for non-standard topologies (e.g., llm → generator).
    topology: Optional[Dict[str, List[str]]] = None

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

    # Stack-compatibility fields (not on TransformerConfig, so __getattr__ can't proxy them)
    vocab_size: Optional[int] = None
    seq_length: int = 1024
    make_vocab_size_divisible_by: int = 128
    should_pad_vocab: bool = False
    share_embeddings_and_output_weights: bool = True

    def __getattr__(self, name: str):
        d = self.__dict__
        mimo_cfg = d.get("mimo_parallelism_config", _SENTINEL)

        if mimo_cfg is not _SENTINEL and mimo_cfg is not None:
            raise AttributeError(
                f"'{type(self).__name__}' has no attribute '{name}'. "
                f"Heterogeneous MIMO (mimo_parallelism_config is set) must use "
                f"pretrain_mimo(), not pretrain()."
            )

        lang_spec = d.get("language_model_spec")
        if lang_spec is not None and hasattr(lang_spec, "params") and lang_spec.params:
            config = lang_spec.params.get("config")
            if config is not None and hasattr(config, name):
                return getattr(config, name)
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def __setattr__(self, name: str, value):
        object.__setattr__(self, name, value)
        d = self.__dict__
        if "mimo_parallelism_config" in d and d.get("mimo_parallelism_config") is None and not name.startswith("_"):
            lang_spec = d.get("language_model_spec")
            if lang_spec is not None and hasattr(lang_spec, "params") and lang_spec.params:
                config = lang_spec.params.get("config")
                if config is not None and hasattr(config, name):
                    setattr(config, name, value)

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
        else:
            grids = {}
            pg_collections = {}

        if self.topology is not None:
            topology = self.topology
        else:
            topology = {name: ["llm"] for name in self.modality_submodules_spec} | {"llm": []}

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
        if self.mimo_parallelism_config:
            llm_pg = infra.pg_collections.get("llm")
            if llm_pg is not None:
                language_spec = self._inject_pg_collection_into_language_spec(language_spec, llm_pg)
        elif hasattr(self, "_pg_collection") and self._pg_collection is not None:
            language_spec = self._inject_pg_collection_into_language_spec(language_spec, self._pg_collection)

        # Inject pg_collection into modality specs
        modality_specs: Dict[str, ModuleSpec] = {}
        for module_name, spec in self.modality_submodules_spec.items():
            if self.mimo_parallelism_config:
                module_pg = infra.pg_collections.get(module_name) if infra.pg_collections else None
            elif hasattr(self, "_pg_collection") and self._pg_collection is not None:
                module_pg = self._pg_collection
            else:
                module_pg = None
            if module_pg is not None:
                spec = self._inject_pg_collection_into_modality_spec(spec, module_pg)
            modality_specs[module_name] = spec

        # Create MimoModel
        mimo_model_config = MimoModelConfig(
            language_model_spec=language_spec,
            modality_submodules_spec=modality_specs,
            special_token_ids=self.special_token_ids,
            module_to_grid_map=(infra.module_to_grid_map if self.mimo_parallelism_config is not None else None),
            language_module_key="llm" if self.mimo_parallelism_config is not None else None,
        )

        mimo_model = MimoModel(mimo_model_config)

        if self.mimo_parallelism_config is None:
            mimo_model.config = _MimoConfigProxy(mimo_model.config, self)

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
        use_cpu_initialization: Optional[bool] = None,
        init_model_with_meta_device: Optional[bool] = None,
        pre_wrap_hook: Optional[
            Union[
                Callable[[List[MegatronModule]], List[MegatronModule]],
                List[Callable[[List[MegatronModule]], List[MegatronModule]]],
            ]
        ] = None,
        post_wrap_hook: Optional[Callable[[List[MegatronModule]], List[MegatronModule]]] = None,
        mixed_precision_wrapper=Float16Module,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ) -> List[MegatronModule]:
        """Build MIMO model with distributed wrapping.

        In **homogeneous** mode (mimo_parallelism_config is None), delegates to
        the parent ModelProviderMixin which uses standard Float16Module + DDP.

        In **heterogeneous** mode, uses per-module HyperCommGrids, per-submodule
        DDP wrapping, and direct dtype casting (not Float16Module).

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
            mixed_precision_wrapper: Wrapper for fp16/bf16 (used in homogeneous mode).
            pg_collection: Pre-initialized ProcessGroupCollection (homogeneous mode).

        Returns:
            List containing the wrapped MimoModel.

        Raises:
            ValueError: If this rank doesn't participate in any module
                (indicates invalid parallelism configuration).
        """
        # Homogeneous mode: delegate to standard ModelProviderMixin path
        if self.mimo_parallelism_config is None:
            self.finalize()
            return super().provide_distributed_model(
                ddp_config=ddp_config,
                model_type=model_type,
                overlap_param_gather_with_optimizer_step=overlap_param_gather_with_optimizer_step,
                fp16=fp16,
                bf16=bf16,
                use_megatron_fsdp=use_megatron_fsdp,
                use_torch_fsdp2=use_torch_fsdp2,
                wrap_with_ddp=wrap_with_ddp,
                data_parallel_random_init=data_parallel_random_init,
                use_cpu_initialization=use_cpu_initialization,
                init_model_with_meta_device=init_model_with_meta_device,
                pre_wrap_hook=pre_wrap_hook,
                post_wrap_hook=post_wrap_hook,
                mixed_precision_wrapper=mixed_precision_wrapper,
                pg_collection=pg_collection,
            )

        # --- Heterogeneous mode (existing logic) ---
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

        # Per-submodule DDP for heterogeneous parallelism
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
        """Initialize model parallelism.

        In homogeneous mode (mimo_parallelism_config is None), delegates to the
        parent which reads TP/PP/CP sizes via __getattr__ proxy.

        In heterogeneous mode, raises NotImplementedError to prevent accidental
        global MPU initialization (use finalize() + build_infra() instead).
        """
        if self.mimo_parallelism_config is not None:
            raise NotImplementedError(
                "Heterogeneous MIMO does not use global model parallelism initialization. "
                "Use finalize() to validate config and build_infra() to create HyperCommGrids."
            )
        super().initialize_model_parallel(seed=seed, seed_kwargs=seed_kwargs, **model_parallel_kwargs)

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

        In homogeneous mode, enforces pipeline_model_parallel_size == 1 (PP is
        incompatible because encoders are fully replicated, not pipelined).

        Raises:
            ValueError: If any rank doesn't participate in at least one module,
                or if homogeneous mode has PP > 1.
        """
        if self.mimo_parallelism_config is not None:
            if not dist.is_initialized():
                raise RuntimeError(
                    "MIMO requires torch.distributed to be initialized before finalize(). "
                    "Call torch.distributed.init_process_group() first."
                )
            self.mimo_parallelism_config.finalize(dist.get_world_size())
        else:
            lang_spec = self.language_model_spec
            if lang_spec is not None and hasattr(lang_spec, "params") and lang_spec.params:
                config = lang_spec.params.get("config")
                if config is not None:
                    pp_size = getattr(config, "pipeline_model_parallel_size", 1)
                    if pp_size != 1:
                        raise ValueError(
                            f"Homogeneous MIMO requires pipeline_model_parallel_size=1, "
                            f"got {pp_size}. Pipeline parallelism is incompatible with "
                            f"homogeneous mode because encoders are fully replicated."
                        )
