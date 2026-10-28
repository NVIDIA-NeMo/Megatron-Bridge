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
"""TrainingEngine: an in-process, every-rank training object built from a Bridge ``ConfigContainer``.

Phase A implements construction (``from_config``), shared-runtime ownership, ``parallel_info``,
engine-local ``close()`` and named-loss dispatch. Compute, state and placement APIs raise
``NotImplementedError`` until their phase lands.
"""

from __future__ import annotations

import copy
import gc
import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import torch
from megatron.core.config import set_experimental_flag
from megatron.core.distributed import DistributedDataParallel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import MegatronModule
from megatron.core.utils import get_model_config

from megatron.bridge.engine.datum import (
    AdamParams,
    CheckpointHandle,
    Datum,
    LoraConfig,
    ModelOutput,
    NormalizationPolicy,
    validate_datums,
)
from megatron.bridge.models.model_provider import ModelProviderMixin
from megatron.bridge.training.checkpointing import CheckpointLoadContext, create_checkpoint_manager
from megatron.bridge.training.config import ConfigContainer, runtime_config_update
from megatron.bridge.training.initialize import initialize_megatron
from megatron.bridge.training.optim import (
    memory_efficient_fp32_optimizer_state_loading,
    setup_optimizer,
    sync_hybrid_device_optimizer_fp32_master_copies,
)
from megatron.bridge.training.setup import (
    _build_distributed_model,
    _register_pre_wrap_hook,
    _resolve_embedding_ranks_fn,
    _update_model_config_funcs,
    _validate_and_set_vocab_size,
)
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer
from megatron.bridge.utils.common_utils import get_world_size_safe, print_rank_0


logger = logging.getLogger(__name__)

PreWrapHook = Callable[[list[MegatronModule]], list[MegatronModule]]
LossMethod = Callable[..., tuple[torch.Tensor, dict[str, float]]]

# Public engine attributes that can never be used as a loss method name.
_RESERVED_METHODS = frozenset(
    {
        "from_config",
        "forward_only",
        "forward_backward",
        "optim_step",
        "zero_grad",
        "resolve_loss_method",
        "target_logprobs",
        "target_entropy",
        "save_state",
        "load_state",
        "export",
        "cuda",
        "cpu",
        "close",
        "config",
        "parallel_info",
        "normalization",
    }
)


@dataclass(frozen=True)
class Topology:
    """Distributed topology an engine requires from the shared runtime."""

    world_size: int
    tp: int
    pp: int
    vpp: int | None
    cp: int
    ep: int
    etp: int | None
    decentralized_pg: bool


def topology_from_config(cfg: ConfigContainer) -> Topology:
    """Read the parallel topology a config asks for."""
    model = cfg.model
    return Topology(
        world_size=get_world_size_safe(),
        tp=model.tensor_model_parallel_size,
        pp=model.pipeline_model_parallel_size,
        vpp=getattr(model, "virtual_pipeline_model_parallel_size", None),
        cp=model.context_parallel_size or 1,
        ep=getattr(model, "expert_model_parallel_size", 1) or 1,
        etp=getattr(model, "expert_tensor_parallel_size", None),
        decentralized_pg=bool(cfg.dist.use_decentralized_pg),
    )


def enforce_engine_conventions(cfg: ConfigContainer) -> None:
    """Raise if a recipe contradicts the conventions the engine relies on.

    The engine defers gradient finalization to ``optim_step`` and owns checkpoint loading and the
    learning-rate schedule, so the config must use per-token loss sums, no DP averaging in the
    collective, one packed microbatch per step, and no recipe-driven resume path.
    """
    problems: list[str] = []
    if not getattr(cfg.model, "calculate_per_token_loss", False):
        problems.append("model.calculate_per_token_loss must be True")
    if getattr(cfg.ddp, "average_in_collective", False):
        problems.append("ddp.average_in_collective must be False")
    if cfg.train.micro_batch_size != 1:
        problems.append("train.micro_batch_size must be 1 (one packed sequence per microbatch)")
    if cfg.checkpoint.load is not None:
        problems.append("checkpoint.load must be None; pass checkpoint_path to the engine instead")
    if getattr(cfg.dist, "lazy_mpu_init", False):
        problems.append("dist.lazy_mpu_init is not supported")
    if getattr(cfg.dist, "use_megatron_fsdp", False) or getattr(cfg.dist, "use_torch_fsdp2", False):
        problems.append("FSDP wrappers are not supported; use DDP with the distributed optimizer")
    if cfg.peft is not None:
        problems.append("cfg.peft must be None; LoRA is configured through the engine (phase C3)")
    if problems:
        raise ValueError("Config violates TrainingEngine conventions:\n  - " + "\n  - ".join(problems))


@dataclass
class _GradHooks:
    """Schedule hooks Bridge installed on the model config; the engine nulls them per call and restores them."""

    finalize_model_grads_func: Callable[..., None] | None
    grad_sync_func: Callable[..., None] | list[Callable[..., None]] | None
    no_sync_func: Callable[[], Any] | list[Callable[[], Any]] | None
    grad_scale_func: Callable[[torch.Tensor], torch.Tensor] | None


class _SharedRuntime:
    """Process-wide Megatron runtime (process groups, RNG, microbatch calculator) shared by compatible engines."""

    def __init__(self, topology: Topology, pg_collection: ProcessGroupCollection) -> None:
        self.topology = topology
        self.pg_collection = pg_collection
        self._engines: set[int] = set()

    def attach(self, engine: TrainingEngine) -> None:
        self._engines.add(id(engine))

    def detach(self, engine: TrainingEngine) -> None:
        self._engines.discard(id(engine))

    @property
    def num_engines(self) -> int:
        """Number of engines currently attached."""
        return len(self._engines)


_RUNTIME: _SharedRuntime | None = None


def _acquire_runtime(cfg: ConfigContainer) -> _SharedRuntime:
    """Initialize Megatron once per process, or attach to the existing runtime if the topology matches."""
    global _RUNTIME
    topology = topology_from_config(cfg)
    if _RUNTIME is None:
        set_experimental_flag(cfg.dist.enable_megatron_core_experimental)
        pg_collection = initialize_megatron(cfg=cfg, get_embedding_ranks=_resolve_embedding_ranks_fn(cfg.model, None))
        if not isinstance(pg_collection, ProcessGroupCollection):
            raise RuntimeError("initialize_megatron did not return process groups (lazy MPU init is unsupported).")
        _RUNTIME = _SharedRuntime(topology, pg_collection)
        print_rank_0(f"TrainingEngine runtime initialized: {topology}")
    elif _RUNTIME.topology != topology:
        raise ValueError(
            f"Engine topology {topology} is incompatible with the active runtime {_RUNTIME.topology}. "
            "Engines in one process must share TP/PP/CP/EP sizes; use a separate worker runtime instead."
        )
    return _RUNTIME


def active_runtime() -> _SharedRuntime | None:
    """Return the process-wide runtime, or ``None`` before the first engine is built."""
    return _RUNTIME


class TrainingEngine:
    """Tinker-like training engine over a Megatron-Bridge recipe.

    Every rank constructs the engine and enters every distributed call. The engine owns its model,
    optimizer and accumulation state; the distributed runtime is shared with other compatible engines
    in the process and outlives ``close()``.
    """

    normalization: NormalizationPolicy = NormalizationPolicy.TOKEN_MEAN

    def __init__(
        self,
        model: str,
        *,
        checkpoint_path: str | None = None,
        output_head: str = "lm",
        trainable: bool = True,
        lora: LoraConfig | None = None,
        hooks: list[PreWrapHook] | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> None:
        raise NotImplementedError(
            "Construction by model name (recipes.py) is phase D; use TrainingEngine.from_config(cfg)."
        )

    # ---- construction -------------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        cfg: ConfigContainer,
        *,
        checkpoint_path: str | None = None,
        output_head: str = "lm",
        trainable: bool = True,
        lora: LoraConfig | None = None,
        hooks: list[PreWrapHook] | None = None,
    ) -> TrainingEngine:
        """Build an engine from a recipe ``ConfigContainer``.

        The config is deep-copied so several engines can be built from one recipe object. Its dataset,
        scheduler and logging sections are ignored: the caller feeds Datums and owns the learning-rate
        schedule.

        Args:
            cfg: Recipe configuration (model, optimizer, ddp, dist, mixed precision, tokenizer).
            checkpoint_path: Megatron or HF checkpoint whose weights initialize the model, or ``None``
                for the provider's initialization.
            output_head: ``"lm"`` only for now; ``"value"`` arrives with the critic (phase D).
            trainable: Build the optimizer and gradient buffers. ``False`` yields a frozen scoring engine.
            lora: LoRA settings (phase C3).
            hooks: Extra pre-wrap hooks run after the Bridge-registered ones.

        Returns:
            A constructed engine attached to the shared runtime.
        """
        if output_head != "lm":
            raise NotImplementedError(f"output_head={output_head!r} is not supported yet (value head: phase D).")
        if lora is not None:
            raise NotImplementedError("LoRA engines are phase C3.")
        cfg = copy.deepcopy(cfg)
        cfg.checkpoint.pretrained_checkpoint = checkpoint_path
        cfg.checkpoint.load = None
        runtime_config_update(cfg)
        enforce_engine_conventions(cfg)
        runtime = _acquire_runtime(cfg)
        engine = cls.__new__(cls)
        engine._build(cfg, runtime, trainable=trainable, hooks=hooks or [])
        return engine

    def _build(
        self, cfg: ConfigContainer, runtime: _SharedRuntime, *, trainable: bool, hooks: list[PreWrapHook]
    ) -> None:
        self._cfg = cfg
        self._runtime = runtime
        self._trainable = trainable
        self._closed = False
        self._step = 0
        self._state = GlobalState()
        self._state.cfg = cfg
        pg_collection = runtime.pg_collection

        tokenizer = build_tokenizer(cfg.tokenizer)
        cfg.model.vocab_size, cfg.model.should_pad_vocab = _validate_and_set_vocab_size(
            model_vocab_size=cfg.model.vocab_size,
            tokenizer_vocab_size=tokenizer.vocab_size,
            use_tokenizer_vocab_size=getattr(cfg.tokenizer, "use_tokenizer_vocab_size", False),
        )
        for hook in hooks:
            _register_pre_wrap_hook(cfg.model, hook)

        self._optimizer = None
        if trainable:
            self._model = _build_distributed_model(cfg, pg_collection)
            self._optimizer, _unused_scheduler = setup_optimizer(
                optimizer_config=cfg.optimizer,
                scheduler_config=cfg.scheduler,
                model=self._model,
                use_gloo_process_groups=cfg.dist.use_gloo_process_groups,
                pg_collection=pg_collection if cfg.dist.use_decentralized_pg else None,
                optimizer_config_override_provider=cfg.optimizer_config_override_provider,
            )
        else:
            self._model = _build_frozen_model(cfg, pg_collection)

        if cfg.checkpoint.pretrained_checkpoint is not None:
            self._load_pretrained()

        self._grad_hooks: _GradHooks | None = None
        if trainable:
            model_config = get_model_config(self._model[0])
            _update_model_config_funcs(
                self._model,
                model_config,
                cfg.ddp,
                self._optimizer,
                align_grad_reduce=cfg.dist.align_grad_reduce,
                pg_collection=pg_collection,
                peft_enabled=False,
            )
            self._grad_hooks = _GradHooks(
                finalize_model_grads_func=model_config.finalize_model_grads_func,
                grad_sync_func=model_config.grad_sync_func,
                no_sync_func=model_config.no_sync_func,
                grad_scale_func=model_config.grad_scale_func,
            )
        runtime.attach(self)

    def _load_pretrained(self) -> None:
        cfg = self._cfg
        manager = create_checkpoint_manager(cfg.checkpoint)
        checkpoint_optimizer = self._optimizer if cfg.checkpoint.load_optim and not cfg.checkpoint.finetune else None
        with memory_efficient_fp32_optimizer_state_loading(checkpoint_optimizer):
            manager.load(
                CheckpointLoadContext(
                    state=self._state,
                    model=self._model,
                    optimizer=checkpoint_optimizer,
                    opt_param_scheduler=None,
                )
            )
        sync_hybrid_device_optimizer_fp32_master_copies(self._optimizer)

    # ---- compute ------------------------------------------------------------------------------

    def forward_only(
        self,
        datums: list[Datum],
        *,
        disable_adapters: bool = False,
        gather_output: bool = True,
        broadcast_output: bool = True,
        return_entropy: bool = False,
    ) -> ModelOutput:
        """Score a global logical batch without touching gradients (phase B1)."""
        self._require_open()
        validate_datums(datums)
        raise NotImplementedError("forward_only is implemented in phase B1.")

    def forward_backward(
        self,
        datums: list[Datum],
        *,
        loss_fn: str = "cross_entropy",
        loss_kwargs: dict[str, Any] | None = None,
        broadcast_output: bool = True,
    ) -> ModelOutput:
        """Run forward/backward with the named loss and accumulate gradients (phase B2)."""
        self._require_open()
        self._require_trainable("forward_backward")
        validate_datums(datums)
        self.resolve_loss_method(loss_fn)
        raise NotImplementedError("forward_backward is implemented in phase B2.")

    def optim_step(self, adam_params: AdamParams, *, normalize_gradients: bool = True) -> dict[str, float]:
        """Normalize and finalize accumulated gradients, then step the optimizer (phase B2)."""
        self._require_open()
        self._require_trainable("optim_step")
        raise NotImplementedError("optim_step is implemented in phase B2.")

    def zero_grad(self) -> None:
        """Abort the open accumulation window (phase B2)."""
        self._require_open()
        self._require_trainable("zero_grad")
        raise NotImplementedError("zero_grad is implemented in phase B2.")

    # ---- loss extension points ----------------------------------------------------------------

    def resolve_loss_method(self, name: str) -> LossMethod:
        """Return the bound loss method for ``forward_backward(loss_fn=name)``.

        Any public callable defined on the engine class (or a subclass) that is not part of the engine
        API is a valid loss method, so subclasses add losses by defining methods.

        Raises:
            ValueError: If ``name`` is reserved, private, or not a callable attribute.
        """
        if not isinstance(name, str) or not name or name.startswith("_") or name in _RESERVED_METHODS:
            raise ValueError(f"loss_fn={name!r} is not a loss method name.")
        attribute = getattr(type(self), name, None)
        if attribute is None or not callable(attribute):
            raise ValueError(f"{type(self).__name__} has no loss method {name!r}.")
        return getattr(self, name)

    def cross_entropy(self, datums: list[Datum], model_output: Any) -> tuple[torch.Tensor, dict[str, float]]:
        """Weighted negative log-likelihood sum over target tokens (phase B2)."""
        raise NotImplementedError("cross_entropy is implemented in phase B2.")

    def importance_sampling(self, datums: list[Datum], model_output: Any) -> tuple[torch.Tensor, dict[str, float]]:
        """Importance-sampling policy-gradient loss (phase C1)."""
        raise NotImplementedError("importance_sampling is implemented in phase C1.")

    def ppo_clip(self, datums: list[Datum], model_output: Any) -> tuple[torch.Tensor, dict[str, float]]:
        """Clipped PPO objective with optional TIS correction (phase C1)."""
        raise NotImplementedError("ppo_clip is implemented in phase C1.")

    def target_logprobs(self, model_output: Any, datums: list[Datum]) -> list[torch.Tensor]:
        """Per-Datum slices of the already computed target logprobs (phase B1)."""
        raise NotImplementedError("target_logprobs is implemented in phase B1.")

    def target_entropy(self, model_output: Any, datums: list[Datum]) -> list[torch.Tensor]:
        """Per-Datum token entropies from the sharded logits (phase C1)."""
        raise NotImplementedError("target_entropy is implemented in phase C1.")

    # ---- state --------------------------------------------------------------------------------

    def save_state(self, path: str, *, user_state: Any = None, async_save: bool = False) -> CheckpointHandle:
        """Save model, optimizer and user state (phase B3)."""
        self._require_open()
        raise NotImplementedError("save_state is implemented in phase B3.")

    def load_state(self, path: str) -> Any | None:
        """Restore model, optimizer and user state (phase B3)."""
        self._require_open()
        raise NotImplementedError("load_state is implemented in phase B3.")

    def export(self, *, path: str | None = None, streaming: bool = False) -> Iterator[tuple[str, torch.Tensor]] | None:
        """Export HF weights to a directory or as a stream (phase B3)."""
        self._require_open()
        raise NotImplementedError("export is implemented in phase B3.")

    # ---- placement ----------------------------------------------------------------------------

    def cuda(self) -> TrainingEngine:
        """Move engine state back to the GPU (phase C2)."""
        self._require_open()
        raise NotImplementedError("cuda is implemented in phase C2.")

    def cpu(self) -> TrainingEngine:
        """Offload engine state to host memory (phase C2)."""
        self._require_open()
        raise NotImplementedError("cpu is implemented in phase C2.")

    # ---- introspection ------------------------------------------------------------------------

    @property
    def config(self) -> ConfigContainer:
        """The engine's private, resolved configuration."""
        return self._cfg

    @property
    def trainable(self) -> bool:
        """Whether this engine owns an optimizer and gradient buffers."""
        return self._trainable

    @property
    def model(self) -> list[MegatronModule]:
        """The engine's model chunks (DDP-wrapped when trainable)."""
        self._require_open()
        return self._model

    @property
    def optimizer(self) -> Any | None:
        """The engine's MegatronOptimizer, or ``None`` for a frozen engine."""
        self._require_open()
        return self._optimizer

    @property
    def parallel_info(self) -> dict[str, int]:
        """Sizes and ranks of the process groups this engine runs on."""
        pg_collection = self._runtime.pg_collection
        info: dict[str, int] = {
            "world_size": torch.distributed.get_world_size(),
            "rank": torch.distributed.get_rank(),
        }
        for name in ("tp", "pp", "cp", "dp", "dp_cp", "ep", "expt_tp", "expt_dp"):
            group = getattr(pg_collection, name, None)
            if group is None:
                continue
            info[f"{name}_size"] = group.size()
            info[f"{name}_rank"] = group.rank()
        return info

    @property
    def closed(self) -> bool:
        """Whether ``close()`` has released this engine."""
        return self._closed

    def close(self) -> None:
        """Release this engine's model, optimizer and buffers. The shared runtime and other engines survive."""
        if self._closed:
            return
        self._closed = True
        self._runtime.detach(self)
        for chunk in self._model:
            _release_ddp_autograd_refs(chunk)
        if self._model:
            # The model config is the provider held by self._cfg; Bridge installed optimizer-bound
            # callables on it, which would otherwise keep optimizer -> model_chunks -> model alive.
            model_config = get_model_config(self._model[0])
            hook_names = (
                "finalize_model_grads_func",
                "grad_scale_func",
                "grad_sync_func",
                "no_sync_func",
                "param_sync_func",
            )
            for name in hook_names:
                if hasattr(model_config, name):
                    setattr(model_config, name, None)
        self._optimizer = None
        self._grad_hooks = None
        self._model = []
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- internals ----------------------------------------------------------------------------

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("TrainingEngine has been closed.")

    def _require_trainable(self, api: str) -> None:
        if not self._trainable:
            raise RuntimeError(f"{api}() requires a trainable engine; this engine was built with trainable=False.")


def _release_ddp_autograd_refs(chunk: MegatronModule) -> None:
    """Break the reference cycle that otherwise keeps a closed DDP model and its buffers on the GPU.

    MCore DDP stores every parameter's ``AccumulateGrad`` node in ``grad_accs`` and registers a hook
    closure over the DDP module on each node. The cycle parameter -> node -> hook -> DDP -> parameter
    runs through C++ autograd objects that Python's garbage collector cannot traverse, so the model is
    never freed unless these references are dropped explicitly.
    """
    if not isinstance(chunk, DistributedDataParallel):
        return
    if chunk.use_forward_hook and chunk.remove_forward_pre_hook_handles:
        chunk.disable_forward_pre_hook(param_sync=False)
    chunk.grad_accs.clear()


def _build_frozen_model(cfg: ConfigContainer, pg_collection: ProcessGroupCollection) -> list[MegatronModule]:
    """Build a model without DDP or gradient buffers and freeze its parameters."""
    provider = cfg.model
    if not isinstance(provider, ModelProviderMixin):
        raise NotImplementedError("trainable=False currently requires a ModelProviderMixin model config.")
    provider.finalize()
    model = provider.provide_distributed_model(
        ddp_config=None,
        wrap_with_ddp=False,
        data_parallel_random_init=cfg.rng.data_parallel_random_init,
        pg_collection=pg_collection,
    )
    for chunk in model:
        for parameter in chunk.parameters():
            parameter.requires_grad_(False)
    return model
