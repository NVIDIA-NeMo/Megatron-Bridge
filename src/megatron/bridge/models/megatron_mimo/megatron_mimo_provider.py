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

import contextlib
import copy
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Callable, ContextManager, Dict, Iterator, List, Optional, Union

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
from megatron.bridge.models.megatron_mimo.megatron_mimo_config import (
    MegatronMIMOParallelismConfig,
    ModuleParallelismConfig,
)
from megatron.bridge.models.megatron_mimo.megatron_mimo_ddp import wrap_megatron_mimo_model_distributed
from megatron.bridge.models.model_provider import ModelProviderMixin


logger = logging.getLogger(__name__)

_TRUE_ENV_VALUES = {"1", "true", "yes", "y", "on"}


if TYPE_CHECKING:
    from megatron.core.hyper_comm_grid import HyperCommGrid


class MegatronMIMORNGMode(str, Enum):
    """CUDA RNG handling mode for MegatronMIMO."""

    SINGLETON = "singleton"
    PER_MODULE = "per_module"


def get_megatron_mimo_rng_mode(
    par_cfg: Optional[MegatronMIMOParallelismConfig],
) -> MegatronMIMORNGMode:
    """Select the CUDA RNG mode for a MegatronMIMO parallel layout.

    Non-colocated layouts and colocated layouts with identical TP sizes use
    MCore's standard singleton CUDA RNG tracker. Colocated asymmetric TP needs
    per-module snapshots because one physical rank can execute modules with
    different module-local TP coordinates.
    """
    if par_cfg is None or not par_cfg._is_colocated():
        return MegatronMIMORNGMode.SINGLETON
    tp_sizes = {p.tensor_model_parallel_size for p in par_cfg.module_parallelisms.values()}
    if len(tp_sizes) > 1:
        return MegatronMIMORNGMode.PER_MODULE
    return MegatronMIMORNGMode.SINGLETON


def _get_active_module_grids(
    infra: "MegatronMIMOInfra",
    current_rank: int,
) -> Dict[str, "HyperCommGrid"]:
    """Return module grids that include ``current_rank``."""
    active_modules: Dict[str, "HyperCommGrid"] = {}
    for module_name, grid in infra.module_to_grid_map.items():
        if grid.rank_offset <= current_rank < (grid.rank_offset + grid.size):
            active_modules[module_name] = grid
    return active_modules


def _get_global_seed_and_module(
    seed: int,
    active_modules: Dict[str, "HyperCommGrid"],
) -> tuple[int, Optional[str]]:
    """Pick the rank-local CPU RNG seed and the diagnostic anchor module name.

    The returned ``seed`` is fed to ``_seed_python_numpy_torch`` to drive
    Python ``random``, ``numpy``, and ``torch.manual_seed`` for CPU init. It
    is identical on every rank so any module built with
    ``use_cpu_initialization=True`` (e.g. the LLaVA vision projector) lands
    with the same parameter values on every DP/PP rank within its module's
    DP group.

    Per-PP-stage divergence in the CUDA RNG tracker is the *separate*
    responsibility of the caller, which seeds the tracker with
    ``seed + 100 * module.pp_rank`` per active module. Folding that PP
    offset into the CPU seed (the original behavior here) caused
    vision-module CPU init to diverge across LM PP stages in colocated
    PP>1 layouts: the projector's CPU-initialized weights then differed
    across vision-DP siblings on different LM PP stages, breaking iter-1
    numerics in colocated language-PP=2 LLaVA runs.

    The returned ``global_seed_module`` is used only for diagnostic logging
    — it does not affect the seed value.
    """
    if MIMO_LANGUAGE_MODULE_KEY in active_modules:
        global_seed_module: Optional[str] = MIMO_LANGUAGE_MODULE_KEY
    elif active_modules:
        global_seed_module = next(iter(active_modules))
    else:
        global_seed_module = None

    return seed, global_seed_module


def _seed_python_numpy_torch(seed: int) -> None:
    """Seed non-TP-region RNGs shared by all MegatronMIMO RNG modes."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _env_flag(name: str, default: bool = False) -> bool:
    """Parse a boolean feature flag from the environment."""
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in _TRUE_ENV_VALUES


def seed_singleton_rng_tracker(
    seed: int,
    infra: "MegatronMIMOInfra",
    *,
    seed_kwargs: Optional[dict] = None,
    current_rank: Optional[int] = None,
) -> None:
    """Seed MCore's singleton CUDA RNG tracker for baseline MegatronMIMO modes."""
    from megatron.core import tensor_parallel

    if current_rank is None:
        current_rank = dist.get_rank() if dist.is_initialized() else 0
    manual_seed_kwargs = dict(seed_kwargs or {})
    manual_seed_kwargs.pop("tp_rank", None)
    manual_seed_kwargs.pop("ep_rank", None)
    manual_seed_kwargs.pop("etp_rank", None)

    active_modules = _get_active_module_grids(infra, current_rank)
    global_seed, global_seed_module = _get_global_seed_and_module(seed, active_modules)
    _seed_python_numpy_torch(global_seed)
    infra.cuda_rng_states_per_module.clear()

    if torch.cuda.device_count() > 0 and global_seed_module is not None:
        grid = active_modules[global_seed_module]
        tp_rank = grid.get_pg(["tp"]).rank()
        pp_rank = grid.get_pg(["pp"]).rank()
        module_seed = seed + 100 * pp_rank
        tensor_parallel.model_parallel_cuda_manual_seed(
            module_seed,
            **manual_seed_kwargs,
            tp_rank=tp_rank,
            ep_rank=0,
            etp_rank=0,
        )

    logger.info(
        f"Rank {current_rank}: Initialized MegatronMIMO singleton random seeds "
        f"(global_seed={global_seed}, global_seed_module={global_seed_module}, "
        f"active_modules={list(active_modules)})"
    )


def seed_per_module_rng_tracker(
    seed: int,
    infra: "MegatronMIMOInfra",
    *,
    seed_kwargs: Optional[dict] = None,
    current_rank: Optional[int] = None,
) -> None:
    """Seed global RNGs and snapshot MCore's CUDA RNG tracker per active module.

    MegatronMIMO does not use global MPU process groups. In colocated
    heterogeneous TP, the same physical rank can have different module-local TP
    ranks, so one singleton CUDA RNG tracker state is not enough. This helper
    seeds the tracker once per active module using that module's TP/PP
    coordinates, then stores the resulting tracker states on ``infra`` for
    ``module_rng_scope`` to swap during module construction and forward.

    Args:
        seed: Base random seed.
        infra: Built MegatronMIMO infrastructure.
        seed_kwargs: Optional kwargs forwarded to
            ``tensor_parallel.model_parallel_cuda_manual_seed``. Module-local
            ``tp_rank``, ``ep_rank``, and ``etp_rank`` are owned by this helper
            and override any values in this dict.
        current_rank: Optional rank override for callers that already abstract
            distributed state in tests. Defaults to ``dist.get_rank()`` when
            torch.distributed is initialized, otherwise 0.
    """
    from megatron.core import tensor_parallel

    if current_rank is None:
        current_rank = dist.get_rank() if dist.is_initialized() else 0
    manual_seed_kwargs = dict(seed_kwargs or {})
    manual_seed_kwargs.pop("tp_rank", None)
    manual_seed_kwargs.pop("ep_rank", None)
    manual_seed_kwargs.pop("etp_rank", None)

    active_modules = _get_active_module_grids(infra, current_rank)
    global_seed, global_seed_module = _get_global_seed_and_module(seed, active_modules)
    _seed_python_numpy_torch(global_seed)

    if torch.cuda.device_count() > 0 and active_modules:
        snapshots: Dict[str, Dict] = {}
        for module_name, grid in active_modules.items():
            tp_rank = grid.get_pg(["tp"]).rank()
            pp_rank = grid.get_pg(["pp"]).rank()
            module_seed = seed + 100 * pp_rank
            tensor_parallel.model_parallel_cuda_manual_seed(
                module_seed,
                **manual_seed_kwargs,
                tp_rank=tp_rank,
                ep_rank=0,
                etp_rank=0,
            )
            snapshots[module_name] = tensor_parallel.get_cuda_rng_tracker().get_states()

        infra.cuda_rng_states_per_module.clear()
        infra.cuda_rng_states_per_module.update(snapshots)

    logger.info(
        f"Rank {current_rank}: Initialized MegatronMIMO random seeds "
        f"(global_seed={global_seed}, global_seed_module={global_seed_module}, "
        f"active_modules={list(active_modules)})"
    )


@contextlib.contextmanager
def module_rng_scope(module_name: str, infra: "MegatronMIMOInfra") -> Iterator[None]:
    """Swap the singleton CUDA RNG tracker into ``module_name``'s saved state.

    Implements Step 3b of the colocated heterogeneous TP/DP plan: each per-module
    construction and forward boundary inside ``MimoModel`` is bracketed by a
    fresh invocation of this scope (via a factory bound in ``provide()``), so
    asymmetric-TP encoder and language modules each draw from their own
    TP-region RNG tracker state.

    Semantics:
      * On entry: load ``module_name``'s saved tracker state into the live
        singleton tracker via ``set_states()``.
      * On exit: snapshot the (possibly-advanced) live tracker back into
        ``module_name``'s slot via ``get_states()``, so any RNG draws inside
        the scope persist across context entries.

    The simpler-than-symmetric design (no explicit ``previously-active``
    bookkeeping) relies on the invariant that no MIMO code path draws from the
    tracker outside a module scope: every per-module RNG-touching call site
    inside ``MimoModel`` is wrapped in ``self._scope(name)``, which resolves to
    a factory-built ``module_rng_scope``. Between consecutive scope entries the
    live tracker may belong to whichever module was last active, but that
    module's slot was already saved at its own scope exit, so swapping in the
    next module's state is a clean snapshot/restore — no RNG history is lost.

    When ``module_name`` is not in ``infra.cuda_rng_states_per_module`` (e.g.
    a non-colocated rank that doesn't participate in the requested module, or
    a legacy path where seeding hasn't run), the scope falls through to
    ``contextlib.nullcontext`` semantics — no swap, no save.

    Args:
        module_name: The module whose RNG state should be active inside the
            scope. Must match a key in ``infra.cuda_rng_states_per_module``
            (populated by ``seed_per_module_rng_tracker``).
        infra: The shared ``MegatronMIMOInfra`` (memoized by
            ``MegatronMIMOProvider.build_infra``) carrying the per-module
            snapshot dict.
    """
    if module_name not in infra.cuda_rng_states_per_module:
        # No snapshot for this module on this rank — fall through cleanly.
        # ``MimoModel._scope`` already handles the no-factory case via
        # ``nullcontext``, but we still get called when a factory is bound
        # for a module the rank doesn't actually serve (e.g. when callers
        # bind factories defensively for every module key). Treat as no-op.
        yield
        return

    from megatron.core import tensor_parallel

    tracker = tensor_parallel.get_cuda_rng_tracker()
    tracker.set_states(infra.cuda_rng_states_per_module[module_name])
    try:
        yield
    finally:
        infra.cuda_rng_states_per_module[module_name] = tracker.get_states()


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
        cuda_rng_states_per_module: Per-module snapshots of the CUDA RNG tracker
            states, populated only for colocated asymmetric-TP layouts by
            ``MegatronMIMOProvider.initialize_model_parallel``.
            Keyed by module name; the values are dicts of {tracker-name → state}
            captured by ``get_cuda_rng_tracker().get_states()``. Read by the
            ``module_rng_scope`` factories threaded into ``MimoModel`` so that
            asymmetric-TP encoder and language modules each draw from their own
            TP-region RNG. Empty dict (default) is the singleton path used by
            non-colocated, symmetric colocated, and non-distributed layouts —
            Bridge passes ``module_rng_scopes=None`` to ``MimoModel`` in that
            case and behavior is unchanged from prior releases.
    """

    module_to_grid_map: Dict[str, "HyperCommGrid"]
    topology: Dict[str, List[str]]
    pg_collections: Dict[str, Optional[ProcessGroupCollection]]
    participating_modules: List[str]
    module_output_ndim: Dict[str, int] = field(default_factory=dict)
    rng_mode: MegatronMIMORNGMode = MegatronMIMORNGMode.SINGLETON
    cuda_rng_states_per_module: Dict[str, Dict] = field(default_factory=dict)


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
    _mimo_model_parallel_initialized: bool = field(default=False, init=False, repr=False)

    # Freezing options
    freeze_language_model: bool = False
    freeze_modality_encoders: Dict[str, bool] = field(default_factory=dict)
    freeze_modality_projections: Dict[str, bool] = field(default_factory=dict)

    # Fields required by ModelProviderMixin / get_model()
    fp16: bool = False
    bf16: bool = True
    use_cpu_initialization: bool = False
    init_model_with_meta_device: bool = False

    def _validate_specs_static(self) -> None:
        """No-dist required-spec validation. Idempotent.

        MegatronMIMO is fundamentally multi-modal: it requires the language
        model spec AND at least one modality submodule spec. Without both,
        the resulting model is either invalid (no language) or a degenerate
        non-MIMO model (no modality), neither of which is a supported shape.

        ``finalize()`` (provider-level) only runs when ``dist`` is initialized
        and only when ``megatron_mimo_parallelism_config`` is set. Callers that
        bypass ``finalize()`` — ``build_infra()``, ``provide()``, the legacy
        no-parallelism path — would otherwise silently accept malformed specs.
        This method runs from both ``build_infra()`` and ``provide()`` so that
        every infrastructure/model-build path enforces the same contract.
        """
        if self.language_model_spec is None:
            raise ValueError(
                "language_model_spec must be set on MegatronMIMOProvider. "
                "Set it directly or use a subclass that populates it in __post_init__."
            )
        if not self.modality_submodules_spec:
            raise ValueError(
                "MegatronMIMOProvider requires at least one modality submodule in "
                "modality_submodules_spec. Found none; add at least one entry "
                "(e.g. 'vision') or use a non-MegatronMIMO training path for "
                "language-only models."
            )
        if self.megatron_mimo_parallelism_config is not None:
            # Catch malformed parallelism module sets before build_hypercomm_grids
            # runs, even when called outside of finalize() / non-distributed paths.
            self.megatron_mimo_parallelism_config.validate_static()

    def _is_asymmetric_tp_colocated(self) -> bool:
        """True iff colocated mode AND modules disagree on TP size.

        Drives the recompute guard in ``_validate_asymmetric_tp_constraints``:
        activation recomputation re-runs forward inside backward, outside the
        per-module ``module_rng_scope`` context, so each module's CUDA RNG
        state isn't restored when the recomputed forward draws.

        ``use_cpu_initialization=True`` was previously rejected here too, but
        that was over-conservative — CPU init builds the full master weight
        deterministically across ranks (every rank shares the same
        ``torch.manual_seed`` state) and slices using each module's own
        ``tp_group``, so per-module TP shards are correct without any
        CUDA-tracker involvement. The standard Bridge path uses the same
        single-``torch.manual_seed`` mechanism for CPU init across arbitrarily
        complex TP/PP layouts.
        """
        par_cfg = self.megatron_mimo_parallelism_config
        return get_megatron_mimo_rng_mode(par_cfg) == MegatronMIMORNGMode.PER_MODULE

    def get_rng_mode(self) -> MegatronMIMORNGMode:
        """Return the RNG mode selected by this provider's parallelism config."""
        return get_megatron_mimo_rng_mode(self.megatron_mimo_parallelism_config)

    def uses_per_module_rng(self) -> bool:
        """Whether this provider requires module-scoped CUDA RNG snapshots."""
        return self.get_rng_mode() == MegatronMIMORNGMode.PER_MODULE

    def _walk_module_spec_for_recompute(self, spec, path: str) -> List[str]:
        """Recursively inspect a ``ModuleSpec`` tree for recompute-enabled configs.

        Returns dotted paths (e.g. ``"vision.encoders.clip"``) for every
        ``TransformerConfig``-bearing entry where ``recompute_granularity`` is
        set. Walks ``spec.params['config']`` / ``spec.params['transformer_config']``
        plus all nested entries under ``spec.submodules`` (which can be a
        ``ModuleSpec``, a dict mapping names to specs, a dict-of-dicts, or a
        list/tuple of specs).

        Required because modality submodule wrappers typically have empty
        top-level ``params`` and the real ``transformer_config`` lives inside
        ``submodules["encoders"]["<encoder>"]``. A non-recursive check would
        miss encoder recompute and let asymmetric TP + encoder recompute slip
        through silently.
        """
        offenders: List[str] = []
        params = spec.params or {}
        cfg = params.get("config") or params.get("transformer_config")
        if cfg is not None and getattr(cfg, "recompute_granularity", None) is not None:
            offenders.append(path)

        submodules = getattr(spec, "submodules", None) or {}

        def _walk(node, sub_path: str) -> None:
            if node is None:
                return
            if hasattr(node, "params") or hasattr(node, "submodules"):
                # ModuleSpec-like — recurse via the same walker.
                offenders.extend(self._walk_module_spec_for_recompute(node, sub_path))
            elif isinstance(node, dict):
                for key, child in node.items():
                    _walk(child, f"{sub_path}.{key}" if sub_path else key)
            elif isinstance(node, (list, tuple)):
                for idx, child in enumerate(node):
                    _walk(child, f"{sub_path}[{idx}]")
            # else: scalar/None — ignore.

        if isinstance(submodules, dict):
            for key, child in submodules.items():
                _walk(child, f"{path}.{key}" if path else key)
        else:
            _walk(submodules, path)

        return offenders

    def _validate_asymmetric_tp_constraints(self) -> None:
        """Block v1-unsafe combinations with asymmetric TP under colocated.

        Currently rejects only activation recomputation: recompute re-runs
        forward inside backward, outside the per-module ``module_rng_scope``
        context, so each module's CUDA RNG state isn't restored when the
        recomputed forward draws. Long-term fix is autograd-side scope
        registration; tracked separately.

        ``MIMO_ALLOW_ASYMMETRIC_TP_RECOMPUTE=1`` bypasses only this recompute
        guard for controlled no-dropout experiments. Do not use it for generic
        configs where recomputed forward may consume CUDA RNG.

        ``use_cpu_initialization=True`` is intentionally NOT rejected — CPU
        init's correctness mechanism (deterministic master weight built from
        a shared ``torch.manual_seed``, then per-module ``tp_group`` slicing)
        is independent of the CUDA RNG tracker and works correctly under
        asymmetric TP. See ``_is_asymmetric_tp_colocated`` docstring.
        """
        if not self._is_asymmetric_tp_colocated():
            return

        if _env_flag("MIMO_ALLOW_ASYMMETRIC_TP_RECOMPUTE"):
            logger.warning(
                "MIMO_ALLOW_ASYMMETRIC_TP_RECOMPUTE is enabled; allowing colocated asymmetric TP with "
                "activation recomputation. This is intended only for no-dropout/no-stochastic-forward runs."
            )
            return

        offenders: List[str] = []
        offenders.extend(self._walk_module_spec_for_recompute(self.language_model_spec, "language"))
        for modality_name, spec in self.modality_submodules_spec.items():
            offenders.extend(self._walk_module_spec_for_recompute(spec, modality_name))
        if offenders:
            raise ValueError(
                f"Colocated asymmetric TP with activation recomputation is not "
                f"supported in v1 — recompute re-runs forward inside backward, "
                f"outside the module RNG scope. Offending configs: {offenders}. "
                f"Disable recompute_granularity or use symmetric TP."
            )

    def _uses_colocated_language_pp(self) -> bool:
        """Whether this provider is configured for colocated language PP."""
        par_cfg = self.megatron_mimo_parallelism_config
        if par_cfg is None or not par_cfg._is_colocated():
            return False
        language_parallelism = par_cfg.module_parallelisms.get(MIMO_LANGUAGE_MODULE_KEY)
        return language_parallelism is not None and language_parallelism.pipeline_model_parallel_size > 1

    def _uses_colocated_language_cp(self) -> bool:
        """Whether this provider is configured for colocated language CP."""
        par_cfg = self.megatron_mimo_parallelism_config
        if par_cfg is None or not par_cfg._is_colocated():
            return False
        language_parallelism = par_cfg.module_parallelisms.get(MIMO_LANGUAGE_MODULE_KEY)
        return language_parallelism is not None and language_parallelism.context_parallel_size > 1

    def _uses_colocated_language_pp_or_cp(self) -> bool:
        """Whether colocated language PP or CP spec-level guards should run."""
        return self._uses_colocated_language_pp() or self._uses_colocated_language_cp()

    def _walk_module_spec_configs(self, spec, path: str) -> List[tuple[str, object]]:
        """Return ``(path, TransformerConfig-like object)`` entries in ``spec``."""
        configs: List[tuple[str, object]] = []
        params = spec.params or {}
        for key in ("config", "transformer_config"):
            cfg = params.get(key)
            if cfg is not None:
                configs.append((path, cfg))

        submodules = getattr(spec, "submodules", None) or {}

        def _walk(node, sub_path: str) -> None:
            if node is None:
                return
            if hasattr(node, "params") or hasattr(node, "submodules"):
                configs.extend(self._walk_module_spec_configs(node, sub_path))
            elif isinstance(node, dict):
                for key, child in node.items():
                    _walk(child, f"{sub_path}.{key}" if sub_path else key)
            elif isinstance(node, (list, tuple)):
                for idx, child in enumerate(node):
                    _walk(child, f"{sub_path}[{idx}]")

        if isinstance(submodules, dict):
            for key, child in submodules.items():
                _walk(child, f"{path}.{key}" if path else key)
        else:
            _walk(submodules, path)

        return configs

    def _walk_module_spec_for_per_token_loss(self, spec, path: str) -> List[str]:
        """Return config paths missing ``calculate_per_token_loss=True``."""
        return [
            config_path
            for config_path, cfg in self._walk_module_spec_configs(spec, path)
            if not getattr(cfg, "calculate_per_token_loss", False)
        ]

    @staticmethod
    def _cp_comm_type_uses_hcp(cp_comm_type) -> bool:
        """Return whether ``cp_comm_type`` requests hierarchical CP."""
        if cp_comm_type is None:
            return False
        if isinstance(cp_comm_type, str):
            return "a2a+p2p" in cp_comm_type
        if isinstance(cp_comm_type, (list, tuple)):
            return any(MegatronMIMOProvider._cp_comm_type_uses_hcp(item) for item in cp_comm_type)
        return False

    def _validate_colocated_language_cp_spec_constraints(self) -> None:
        """Validate CP-specific spec-level constraints for colocated language CP."""
        if not self._uses_colocated_language_cp():
            return

        hcp_comm_offenders: List[str] = []
        hcp_size_offenders: List[str] = []
        tp_overlap_offenders: List[str] = []
        sequence_parallel = False
        for config_path, cfg in self._walk_module_spec_configs(self.language_model_spec, MIMO_LANGUAGE_MODULE_KEY):
            if self._cp_comm_type_uses_hcp(getattr(cfg, "cp_comm_type", None)):
                hcp_comm_offenders.append(config_path)
            if getattr(cfg, "hierarchical_context_parallel_sizes", None) is not None:
                hcp_size_offenders.append(config_path)
            if getattr(cfg, "tp_comm_overlap", False):
                tp_overlap_offenders.append(config_path)
            sequence_parallel = sequence_parallel or bool(getattr(cfg, "sequence_parallel", False))

        if hcp_comm_offenders:
            raise ValueError(
                f"Colocated MegatronMIMO with language CP>1 does not support hierarchical CP. "
                f"Set language TransformerConfig.cp_comm_type without 'a2a+p2p'. "
                f"Offending configs: {hcp_comm_offenders}."
            )
        if hcp_size_offenders:
            raise ValueError(
                f"Colocated MegatronMIMO with language CP>1 does not support hierarchical CP. "
                f"Set language TransformerConfig.hierarchical_context_parallel_sizes=None. "
                f"Offending configs: {hcp_size_offenders}."
            )
        if tp_overlap_offenders:
            raise ValueError(
                f"Colocated MegatronMIMO with language CP>1 does not support "
                f"tp_comm_overlap=True in v1. Offending configs: {tp_overlap_offenders}."
            )

        par_cfg = self.megatron_mimo_parallelism_config
        assert par_cfg is not None
        language_parallelism = par_cfg.module_parallelisms[MIMO_LANGUAGE_MODULE_KEY]
        language_params = self.language_model_spec.params or {}
        max_sequence_length = language_params.get("max_sequence_length")
        if max_sequence_length is None:
            for _, cfg in self._walk_module_spec_configs(self.language_model_spec, MIMO_LANGUAGE_MODULE_KEY):
                max_sequence_length = getattr(cfg, "max_sequence_length", None) or getattr(cfg, "seq_length", None)
                if max_sequence_length is not None:
                    break
        if max_sequence_length is None:
            return

        shard_factor = 2 * language_parallelism.context_parallel_size
        if sequence_parallel:
            shard_factor *= language_parallelism.tensor_model_parallel_size
        if max_sequence_length % shard_factor != 0:
            raise ValueError(
                f"Colocated MegatronMIMO with language CP>1 requires language max_sequence_length "
                f"to be divisible by {shard_factor} for PartitionAdapter sharding. "
                f"Got max_sequence_length={max_sequence_length}."
            )

    def _validate_colocated_language_pp_or_cp_spec_constraints(self) -> None:
        """Validate shared spec-level constraints for colocated language PP or CP."""
        if not self._uses_colocated_language_pp_or_cp():
            return

        if len(self.modality_submodules_spec) != 1:
            raise ValueError(
                f"Colocated MegatronMIMO with language PP>1 or CP>1 supports exactly one "
                f"modality module in v1. Found modality modules: "
                f"{list(self.modality_submodules_spec.keys())}."
            )

        modality_name, modality_spec = next(iter(self.modality_submodules_spec.items()))
        modality_submodules = getattr(modality_spec, "submodules", None) or {}
        if isinstance(modality_submodules, dict):
            encoders = modality_submodules.get("encoders")
            if isinstance(encoders, dict) and len(encoders) != 1:
                raise ValueError(
                    f"Colocated MegatronMIMO with language PP>1 or CP>1 supports exactly one "
                    f"encoder tower per modality in v1. Modality '{modality_name}' "
                    f"has encoder towers: {list(encoders.keys())}."
                )

        offenders = self._walk_module_spec_for_per_token_loss(self.language_model_spec, MIMO_LANGUAGE_MODULE_KEY)
        for mod_name, spec in self.modality_submodules_spec.items():
            offenders.extend(self._walk_module_spec_for_per_token_loss(spec, mod_name))
        if offenders:
            raise ValueError(
                f"Colocated MegatronMIMO with language PP>1 or CP>1 requires "
                f"calculate_per_token_loss=True on every active module "
                f"TransformerConfig. Offending configs: {offenders}."
            )

        self._validate_colocated_language_cp_spec_constraints()

    def _validate_colocated_language_pp_spec_constraints(self) -> None:
        """Compatibility wrapper for the generalized PP-or-CP spec validator."""
        self._validate_colocated_language_pp_or_cp_spec_constraints()

    def build_infra(self) -> MegatronMIMOInfra:
        """Build MegatronMIMO parallelism infrastructure.

        This method builds HyperCommGrids, ProcessGroupCollections, and topology
        for MegatronMIMO's heterogeneous parallelism. **Memoized**: subsequent
        calls return the same ``MegatronMIMOInfra`` object so that side state
        attached to it (e.g. ``cuda_rng_states_per_module`` populated by
        ``seed_per_module_rng_tracker`` between setup and model construction)
        is shared with ``provide()``/``provide_distributed_model()``. Without
        memoization, setup-side mutations would be invisible to the model
        construction path because ``provide_distributed_model`` calls
        ``build_infra()`` again internally.

        Can be called before or after provide(). Call finalize() first to
        validate the parallelism configuration.

        Returns:
            MegatronMIMOInfra containing grids, topology, pg_collections,
            and the list of modules this rank participates in.
        """
        self._validate_specs_static()
        cached = getattr(self, "_infra", None)
        if cached is not None:
            return cached
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

        infra = MegatronMIMOInfra(
            module_to_grid_map=grids,
            topology=topology,
            pg_collections=pg_collections,
            participating_modules=participating_modules,
            module_output_ndim=output_ndim,
            rng_mode=self.get_rng_mode(),
        )
        # Memoize so setup-side mutations (e.g. cuda_rng_states_per_module)
        # are visible when provide_distributed_model() calls build_infra() again.
        object.__setattr__(self, "_infra", infra)
        return infra

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
        if self.megatron_mimo_parallelism_config is not None:
            self._inject_parallelism_into_spec_configs(
                spec,
                self.megatron_mimo_parallelism_config.module_parallelisms[MIMO_LANGUAGE_MODULE_KEY],
            )
        spec.params["pg_collection"] = pg_collection
        if pre_process is not None:
            spec.params["pre_process"] = pre_process
        if post_process is not None:
            spec.params["post_process"] = post_process
        return spec

    def _inject_parallelism_into_spec_configs(
        self,
        spec: ModuleSpec,
        parallelism: ModuleParallelismConfig,
    ) -> None:
        """Set module-local parallelism on TransformerConfig objects in ``spec``."""

        def _apply(node) -> None:
            if node is None:
                return
            if isinstance(node, ModuleSpec):
                params = node.params or {}
                for key in ("config", "transformer_config"):
                    cfg = params.get(key)
                    if cfg is not None:
                        cfg.tensor_model_parallel_size = parallelism.tensor_model_parallel_size
                        cfg.pipeline_model_parallel_size = parallelism.pipeline_model_parallel_size
                        cfg.context_parallel_size = parallelism.context_parallel_size
                        cfg.expert_tensor_parallel_size = parallelism.expert_tensor_parallel_size
                _apply(getattr(node, "submodules", None))
            elif isinstance(node, dict):
                for child in node.values():
                    _apply(child)
            elif isinstance(node, (list, tuple)):
                for child in node:
                    _apply(child)

        _apply(spec)

    def _inject_pg_collection_into_modality_spec(
        self,
        spec: ModuleSpec,
        module_name: str,
        pg_collection: ProcessGroupCollection,
    ) -> ModuleSpec:
        """Inject pg_collection into a modality submodule and its encoder specs."""
        spec = copy.deepcopy(spec)
        if spec.params is None:
            spec.params = {}
        spec.params["pg_collection"] = pg_collection

        if self.megatron_mimo_parallelism_config is not None:
            self._inject_parallelism_into_spec_configs(
                spec,
                self.megatron_mimo_parallelism_config.module_parallelisms[module_name],
            )

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
            ValueError: If language_model_spec is not set, modality_submodules_spec
                is empty, or if this rank doesn't participate in any module.
        """
        # _validate_specs_static() runs again here (build_infra calls it too)
        # so direct callers of provide() get the same contract enforcement.
        # The check is cheap and idempotent.
        self._validate_specs_static()

        # Build infrastructure
        infra = self.build_infra()

        if self.uses_per_module_rng() and not infra.cuda_rng_states_per_module:
            raise RuntimeError(
                "Colocated MegatronMIMO with asymmetric TP requires per-module RNG snapshots "
                "before raw provide() constructs the model. Call "
                "MegatronMIMOProvider.initialize_model_parallel(seed=...) first, or use "
                "provide_distributed_model(), which initializes MegatronMIMO provider state "
                "for standalone construction."
            )

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
                spec = self._inject_pg_collection_into_modality_spec(spec, module_name, module_pg)
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

        # Bind one factory per module the rank participates in. Each factory
        # captures the module name via default-arg so the lambda doesn't all
        # close over the same loop variable. MimoModel calls factory() per
        # entry, returning a fresh context manager — required because
        # construction and per-step forward enter the same scope multiple times.
        # Only bind factories in the per-module RNG mode. Non-colocated and
        # symmetric colocated layouts intentionally stay on the standard
        # singleton tracker path even when they use MegatronMIMO parallelism.
        module_rng_scopes: Optional[Dict[str, Callable[[], ContextManager[None]]]] = None
        if self.uses_per_module_rng() and infra.cuda_rng_states_per_module:
            module_rng_scopes = {
                name: (lambda n=name: module_rng_scope(n, infra)) for name in infra.participating_modules
            }

        megatron_mimo_model = MimoModel(
            mimo_model_config,
            cp_group=cp_group,
            tp_group=tp_group,
            module_rng_scopes=module_rng_scopes,
        )

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

        # Standard Bridge users typically enter through provide_distributed_model()
        # without calling initialize_model_parallel() explicitly. Mirror that
        # provider contract while keeping MIMO's setup module free to pre-seed
        # with the training config seed before it gets here.
        if self.megatron_mimo_parallelism_config is not None and not self._mimo_model_parallel_initialized:
            self.initialize_model_parallel(seed=0)
        else:
            self.finalize()

        # Note: the asymmetric-TP guard runs inside finalize() (and
        # transitively from initialize_model_parallel above). We previously
        # re-ran it here to catch the use_cpu_initialization kwarg override,
        # but CPU init is no longer rejected — it's safe under asymmetric TP
        # by construction (deterministic master weight + per-module tp_group
        # slicing). Recompute, the only remaining guard, is fixed at spec
        # construction time so finalize-time alone is sufficient.

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
        """Initialize MegatronMIMO provider state without global MPU mutation.

        This mirrors the public provider lifecycle used by standard Bridge
        providers, but MegatronMIMO cannot call
        ``parallel_state.initialize_model_parallel`` because modules may have
        independent TP/DP/PP grids. Instead, this method finalizes the
        MegatronMIMO parallelism config, builds/memoizes HyperCommGrid
        infrastructure, and, when ``seed`` is provided, seeds the per-module
        CUDA RNG tracker snapshots consumed by ``module_rng_scope``.

        Args:
            seed: Base random seed. ``None`` finalizes/builds process groups but
                leaves RNG state unchanged.
            seed_kwargs: Optional kwargs forwarded to MCore's
                ``model_parallel_cuda_manual_seed``.
            **model_parallel_kwargs: Accepted for API compatibility with
                ``ModelProviderMixin.initialize_model_parallel``. MegatronMIMO
                does not use global MPU initialization kwargs.
        """
        del model_parallel_kwargs

        if self.megatron_mimo_parallelism_config is not None and not dist.is_initialized():
            import os

            from megatron.bridge.utils.common_utils import get_local_rank_preinit

            os.environ["RANK"] = os.environ.get("RANK", "0")
            os.environ["WORLD_SIZE"] = os.environ.get("WORLD_SIZE", "1")
            os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
            os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "12355")
            torch.cuda.set_device(get_local_rank_preinit())
            dist.init_process_group("nccl")

        self.finalize()
        infra = self.build_infra()
        if seed is not None:
            if self.uses_per_module_rng():
                seed_per_module_rng_tracker(seed, infra, seed_kwargs=seed_kwargs)
            else:
                seed_singleton_rng_tracker(seed, infra, seed_kwargs=seed_kwargs)

        object.__setattr__(self, "_mimo_model_parallel_initialized", True)

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
            self._validate_specs_static()
            self.megatron_mimo_parallelism_config.finalize(dist.get_world_size())
            # After parallelism config is finalized, _is_colocated() and
            # asymmetric-TP geometry are determined. Catch v1-unsafe combos
            # (cpu init, recompute) at config-build time.
            self._validate_colocated_language_pp_or_cp_spec_constraints()
            self._validate_asymmetric_tp_constraints()
