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
"""TrainingEngine: an in-process, every-rank training object built from a ``TrainingRecipe``.

Phase A implements construction (``from_recipe``), shared-runtime ownership, ``parallel_info``,
engine-local ``close()`` and named-loss dispatch. Compute, state and placement APIs raise
``NotImplementedError`` until their phase lands.
"""

from __future__ import annotations

import copy
import functools
import gc
import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import torch
from megatron.core.config import set_experimental_flag
from megatron.core.distributed import DistributedDataParallel
from megatron.core.num_microbatches_calculator import init_num_microbatches_calculator
from megatron.core.optimizer import get_megatron_optimizer, get_standard_config_overrides
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import MegatronModule
from megatron.core.utils import get_model_config

from megatron.bridge.engine import losses
from megatron.bridge.engine.accumulation import AccumulationWindow
from megatron.bridge.engine.batching import (
    MicrobatchMapping,
    check_microbatches,
    default_postprocess,
    default_preprocess,
    sequence_pad_multiple,
)
from megatron.bridge.engine.checkpoint import load_pretrained_weights
from megatron.bridge.engine.datum import (
    AdamParams,
    CheckpointHandle,
    ForwardBackwardOutput,
    ForwardOutput,
    LoraConfig,
    ModelOutput,
    NormalizationPolicy,
    PreparedMicrobatch,
    check_datums,
)
from megatron.bridge.engine.optim import finalize_and_step, zero_gradients
from megatron.bridge.engine.recipes.base import TrainingRecipe, data_parallel_size, finalize_recipe
from megatron.bridge.engine.schedule import run_forward_backward, run_forward_only
from megatron.bridge.models.model_provider import ModelProviderMixin
from megatron.bridge.training.config import RerunStateMachineConfig
from megatron.bridge.training.initialize import init_rerun_state, torch_dist_init
from megatron.bridge.training.setup import (
    _build_distributed_model,
    _register_pre_wrap_hook,
    _resolve_embedding_ranks_fn,
    _update_model_config_funcs,
)
from megatron.bridge.utils.common_utils import get_rank_safe, get_world_size_safe, print_rank_0


logger = logging.getLogger(__name__)

PreWrapHook = Callable[[list[MegatronModule]], list[MegatronModule]]
LossMethod = Callable[[PreparedMicrobatch, ModelOutput], torch.Tensor]

# Public engine attributes that can never be used as a loss method name.
_RESERVED_METHODS = frozenset(
    {
        "from_recipe",
        "forward_only",
        "forward_backward",
        "optim_step",
        "zero_grad",
        "step",
        "trainable",
        "model",
        "optimizer",
        "closed",
        "process_groups",
        "pad_multiple",
        "preprocess",
        "postprocess",
        "summarize",
        "count_samples",
        "check_microbatches",
        "resolve_loss_method",
        "target_logprobs",
        "target_entropy",
        "save_state",
        "load_state",
        "export",
        "cuda",
        "cpu",
        "close",
        "recipe",
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


def topology_from_recipe(recipe: TrainingRecipe) -> Topology:
    """Read the parallel topology a recipe asks for."""
    model = recipe.model
    return Topology(
        world_size=get_world_size_safe(),
        tp=model.tensor_model_parallel_size,
        pp=model.pipeline_model_parallel_size,
        vpp=getattr(model, "virtual_pipeline_model_parallel_size", None),
        cp=model.context_parallel_size or 1,
        ep=getattr(model, "expert_model_parallel_size", 1) or 1,
        etp=getattr(model, "expert_tensor_parallel_size", None),
        decentralized_pg=bool(recipe.dist.use_decentralized_pg),
    )


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


def _acquire_runtime(recipe: TrainingRecipe) -> _SharedRuntime:
    """Initialize Megatron once per process, or attach to the existing runtime if the topology matches."""
    global _RUNTIME
    topology = topology_from_recipe(recipe)
    if _RUNTIME is None:
        set_experimental_flag(recipe.dist.enable_megatron_core_experimental)
        # MCore globals the schedules and losses consult; the engine always runs one packed
        # sequence per microbatch and passes num_microbatches explicitly.
        dp_size = data_parallel_size(recipe)
        init_num_microbatches_calculator(get_rank_safe(), None, dp_size, 1, dp_size, False)
        init_rerun_state(RerunStateMachineConfig())
        pg_collection = torch_dist_init(
            model_config=recipe.model,
            dist_config=recipe.dist,
            rng_config=recipe.rng,
            micro_batch_size=1,
            num_distributed_optimizer_instances=recipe.ddp.num_distributed_optimizer_instances,
            get_embedding_ranks=_resolve_embedding_ranks_fn(recipe.model, None),
            get_position_embedding_ranks=None,
            skip_mpu_initialization=False,
        )
        if not isinstance(pg_collection, ProcessGroupCollection):
            raise RuntimeError("torch_dist_init did not return process groups (lazy MPU init is unsupported).")
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
    """Tinker-like training engine over a Megatron-Bridge TrainingRecipe.

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
            "Construction by model name (recipe lookup) is phase D; use TrainingEngine.from_recipe(recipe)."
        )

    # ---- construction -------------------------------------------------------------------------

    @classmethod
    def from_recipe(
        cls,
        recipe: TrainingRecipe,
        *,
        checkpoint_path: str | None = None,
        output_head: str = "lm",
        trainable: bool = True,
        lora: LoraConfig | None = None,
        hooks: list[PreWrapHook] | None = None,
    ) -> TrainingEngine:
        """Build an engine from a ``TrainingRecipe``.

        The recipe is deep-copied so several engines can be built from one recipe object.

        Args:
            recipe: Model and execution configuration.
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
        recipe = copy.deepcopy(recipe)
        finalize_recipe(recipe)
        runtime = _acquire_runtime(recipe)
        engine = cls.__new__(cls)
        engine._build(recipe, runtime, checkpoint_path=checkpoint_path, trainable=trainable, hooks=hooks or [])
        return engine

    def _build(
        self,
        recipe: TrainingRecipe,
        runtime: _SharedRuntime,
        *,
        checkpoint_path: str | None,
        trainable: bool,
        hooks: list[PreWrapHook],
    ) -> None:
        self._recipe = recipe
        self._runtime = runtime
        self._trainable = trainable
        self._closed = False
        self._step = 0
        self._window = AccumulationWindow()
        pg_collection = runtime.pg_collection
        self._pad_multiple = sequence_pad_multiple(
            cp_size=runtime.topology.cp,
            tp_size=runtime.topology.tp,
            sequence_parallel=bool(getattr(recipe.model, "sequence_parallel", False)),
        )

        for hook in hooks:
            _register_pre_wrap_hook(recipe.model, hook)

        self._optimizer = None
        if trainable:
            self._model = _build_distributed_model(recipe, pg_collection)
            self._optimizer = get_megatron_optimizer(
                config=recipe.optimizer,
                model_chunks=self._model,
                config_overrides=get_standard_config_overrides(config=recipe.optimizer) or None,
                use_gloo_process_groups=recipe.dist.use_gloo_process_groups,
                pg_collection=pg_collection if recipe.dist.use_decentralized_pg else None,
            )
        else:
            self._model = _build_frozen_model(recipe, pg_collection)

        if checkpoint_path is not None:
            load_pretrained_weights(recipe, self._model, self._optimizer, checkpoint_path)

        # Bridge installs the schedule hooks on the model config (finalize_model_grads_func, grad_scale_func,
        # and the overlap hooks). The engine keeps only the finalizer: run_forward_backward suppresses it
        # per call, and optim_step invokes it once per update (see schedule.py / optim.py).
        self._finalize_model_grads_func: Callable[..., None] | None = None
        if trainable:
            model_config = get_model_config(self._model[0])
            _update_model_config_funcs(
                self._model,
                model_config,
                recipe.ddp,
                self._optimizer,
                align_grad_reduce=recipe.dist.align_grad_reduce,
                pg_collection=pg_collection,
                peft_enabled=False,
            )
            self._finalize_model_grads_func = model_config.finalize_model_grads_func
        runtime.attach(self)

    # ---- compute ------------------------------------------------------------------------------

    def forward_only(
        self,
        data: Any,
        *,
        disable_adapters: bool = False,
        gather_output: bool = True,
        broadcast_output: bool = True,
        return_entropy: bool = False,
    ) -> ForwardOutput:
        """Score a batch and return per-token target logprobs mapped back to each sample.

        ``data`` is whatever this engine's ``preprocess`` accepts; the default accepts the same global
        ``list[Datum]`` on every rank. ``gather_output=True`` returns every Datum's complete logprobs on
        every rank in input order; ``False`` returns only what this rank computed (its DP replica's
        samples, its CP positions) with ``sample_indices`` and ``token_positions``. Gradients and any
        open accumulation window are untouched.
        """
        self._require_open()
        if return_entropy:
            raise NotImplementedError("return_entropy is implemented in phase C1 (logits path).")
        if disable_adapters:
            raise NotImplementedError("disable_adapters is implemented in phase C3 (LoRA).")
        if self._runtime.topology.pp > 1:
            raise NotImplementedError(
                "forward_only with pipeline parallelism (broadcast_output) is not supported yet."
            )
        microbatches = self.preprocess(data, training=False)
        self.check_microbatches(microbatches)
        logprobs = run_forward_only(
            self._model,
            microbatches,
            pg_collection=self._runtime.pg_collection,
            mtp_target_scope=self._recipe.mtp_target_scope,
        )
        return self.postprocess(microbatches, logprobs, gather_output=gather_output)

    def forward_backward(
        self,
        data: Any,
        *,
        loss_fn: str = "cross_entropy",
        loss_kwargs: dict[str, Any] | None = None,
        broadcast_output: bool = True,
    ) -> ForwardBackwardOutput:
        """Run forward/backward with the named loss and accumulate FP32 gradients.

        Gradients are summed, unnormalized, into the DDP buffers across calls; ``optim_step`` normalizes
        and applies them. Returns this call's detached loss sum and active-token count, reduced over
        DP x CP. ``loss_kwargs`` are bound to the loss method as keyword arguments.
        """
        self._require_open()
        self._require_trainable("forward_backward")
        loss_method = self.resolve_loss_method(loss_fn)
        if loss_kwargs:
            loss_method = functools.partial(loss_method, **loss_kwargs)
        if self._runtime.topology.pp > 1:
            # TODO(pp): PP output-stage loss, broadcast_output of the summary.
            raise NotImplementedError("forward_backward with pipeline parallelism is not supported yet.")
        if getattr(self._recipe.ddp, "overlap_grad_reduce", False):
            # TODO(overlap): defer bucket reductions across calls with DDP.no_sync + explicit start_grad_sync.
            raise NotImplementedError("forward_backward with ddp.overlap_grad_reduce=True is not supported yet.")
        microbatches = self.preprocess(data, training=True)
        self.check_microbatches(microbatches)
        num_samples = self.count_samples(microbatches)
        if num_samples is None and self.normalization == NormalizationPolicy.SAMPLE_MEAN:
            raise ValueError(
                "NormalizationPolicy.SAMPLE_MEAN needs the Datum count, but count_samples returned None; "
                "override count_samples together with preprocess when the mapping is not MicrobatchMapping."
            )
        loss_sum, num_loss_tokens = run_forward_backward(
            self._model,
            microbatches,
            loss_method,
            pg_collection=self._runtime.pg_collection,
            mtp_target_scope=self._recipe.mtp_target_scope,
        )
        summary = self.summarize(loss_sum, num_loss_tokens)
        self._window.add(
            num_loss_tokens=summary.num_loss_tokens,
            num_samples=0 if num_samples is None else num_samples,
            loss_sum=summary.loss_sum,
        )
        return summary

    # ---- data pipeline extension points (override in a subclass) -------------------------------

    def preprocess(self, data: Any, *, training: bool) -> list[PreparedMicrobatch]:
        """Turn caller input into this rank's strict microbatches (boundary B1 -> B2).

        Default: ``data`` is the same global ``list[Datum]`` on every rank; it is grouped, dealt to DP
        replicas, packed or padded per the recipe's ``batch_layout``, and CP-sharded. Override to read
        DP-local data, storage handles, or pre-built batches; the result is always validated by
        ``check_microbatches`` and every DP replica must return the same number of microbatches.
        """
        datums = check_datums(data)
        pg_collection = self._runtime.pg_collection
        return default_preprocess(
            datums,
            layout=self._recipe.batch_layout,
            seq_length=self._recipe.model.seq_length,
            pad_multiple=self._pad_multiple,
            dp_group=pg_collection.dp,
            cp_group=pg_collection.cp,
            device=torch.device("cuda", torch.cuda.current_device()),
            hybrid_model=bool(getattr(self._recipe.model, "is_hybrid_model", False)),
            pad_to_max_length=self._recipe.pad_to_max_length,
        )

    def postprocess(
        self, microbatches: list[PreparedMicrobatch], logprobs: list[torch.Tensor], *, gather_output: bool
    ) -> ForwardOutput:
        """Turn per-microbatch ``[T_local]`` logprobs into the scoring result (boundary B3 -> B5).

        Default: drop padding, reassemble across CP, collect across DP when ``gather_output`` is set, and
        restore the caller's sample order. Override together with ``preprocess`` when the mapping or the
        delivery (local shards, storage) differs.
        """
        pg_collection = self._runtime.pg_collection
        return default_postprocess(
            microbatches,
            logprobs,
            gather_output=gather_output,
            cp_group=pg_collection.cp,
            dp_group=pg_collection.dp,
        )

    def summarize(self, loss_sum: torch.Tensor, num_loss_tokens: torch.Tensor) -> ForwardBackwardOutput:
        """Reduce this call's loss sum and active-token count over DP x CP (boundary B3 -> B6)."""
        pg_collection = self._runtime.pg_collection
        totals = torch.stack([loss_sum.detach().float(), num_loss_tokens.float()]).cuda()
        torch.distributed.all_reduce(totals, group=pg_collection.dp_cp)
        return ForwardBackwardOutput(loss_sum=float(totals[0].item()), num_loss_tokens=int(totals[1].item()))

    def check_microbatches(self, microbatches: list[PreparedMicrobatch]) -> None:
        """The strict gate before Megatron: shapes, dtypes, devices, vocabulary range, layout metadata,
        and equal microbatch counts across DP replicas. Always runs, for default and custom pipelines."""
        if not microbatches:
            raise ValueError("preprocess returned no microbatches; every DP replica must run at least one.")
        check_microbatches(
            microbatches,
            layout=self._recipe.batch_layout,
            vocab_size=int(self._recipe.model.vocab_size),
            dp_group=self._runtime.pg_collection.dp,
        )

    def optim_step(self, adam_params: AdamParams, *, normalize_gradients: bool = True) -> dict[str, float]:
        """Normalize the accumulated gradients once, finalize, and step the optimizer.

        The divisor follows ``self.normalization`` (active tokens ``D``, Datums ``N``, or none);
        ``normalize_gradients=False`` bypasses it. The caller supplies the learning rate and the other
        Adam hyperparameters for this step. An update is skipped (buffers zeroed, window closed) when
        the normalization domain is empty or the optimizer finds non-finite gradients.
        """
        self._require_open()
        self._require_trainable("optim_step")
        if not self._window.is_open:
            raise RuntimeError("optim_step called without a preceding forward_backward.")
        scale: float | None = None
        if normalize_gradients:
            denominator = self._window.denominator(self.normalization)
            if denominator is not None:
                if denominator == 0:
                    zero_gradients(self._model, self._optimizer)
                    self._window.reset()
                    return {"update_successful": 0.0, "skipped_empty_window": 1.0}
                scale = 1.0 / denominator
        if scale is None and self._has_coupled_auxiliary_losses():
            # Coupled MTP / router aux gradients share the primary 1/D; bypassing it changes their semantics.
            raise NotImplementedError(
                "normalize_gradients=False (or NormalizationPolicy.NONE) with MTP or router auxiliary losses "
                "enabled is not supported yet."
            )
        update_successful, grad_norm, num_zeros = finalize_and_step(
            self._model,
            self._optimizer,
            finalize_model_grads_func=self._finalize_model_grads_func,
            scale=scale,
            adam_params=adam_params,
        )
        result = {
            "update_successful": float(update_successful),
            "grad_norm": float("nan") if grad_norm is None else grad_norm,
            "num_zeros_in_grad": float("nan") if num_zeros is None else float(num_zeros),
            "num_loss_tokens": float(self._window.num_loss_tokens),
            "num_samples": float(self._window.num_samples),
            "loss_sum": self._window.loss_sum,
            "learning_rate": adam_params.learning_rate,
        }
        zero_gradients(self._model, self._optimizer)
        self._window.reset()
        if update_successful:
            self._step += 1
        return result

    def zero_grad(self) -> None:
        """Abort the open accumulation window: zero the gradient buffers and reset the counts."""
        self._require_open()
        self._require_trainable("zero_grad")
        zero_gradients(self._model, self._optimizer)
        self._window.reset()

    @property
    def step(self) -> int:
        """Number of successful optimizer updates since construction."""
        return self._step

    def count_samples(self, microbatches: list[PreparedMicrobatch]) -> int | None:
        """Datums in this call across DP replicas (``N`` for ``SAMPLE_MEAN``), or ``None`` when unknown.

        Default: read the default ``MicrobatchMapping`` (fillers count zero) and reduce over DP; any other
        mapping makes the count unknown. Override together with ``preprocess`` when the mapping differs;
        the result must be the global count, identical on every rank.
        """
        if not all(isinstance(mb.mapping, MicrobatchMapping) for mb in microbatches):
            return None
        local = sum(len(mb.mapping.segments) for mb in microbatches if not mb.mapping.filler)
        count = torch.tensor([local], device="cuda")
        torch.distributed.all_reduce(count, group=self._runtime.pg_collection.dp)
        return int(count.item())

    def _has_coupled_auxiliary_losses(self) -> bool:
        model = self._recipe.model
        mtp = (getattr(model, "mtp_num_layers", None) or 0) > 0 and not getattr(model, "mtp_detach_heads", False)
        aux_coeff = getattr(model, "moe_aux_loss_coeff", 0.0) or 0.0
        z_coeff = getattr(model, "moe_z_loss_coeff", None) or 0.0
        return bool(mtp or aux_coeff > 0 or z_coeff > 0)

    # ---- loss extension points ----------------------------------------------------------------

    def resolve_loss_method(self, name: str) -> LossMethod:
        """Return the bound loss method for ``forward_backward(loss_fn=name)``.

        Any public callable defined on the engine class (or a subclass) that is not part of the engine
        API is a valid loss method, so subclasses add losses by defining methods.

        Raises:
            ValueError: If ``name`` is reserved, private, or not a callable attribute.
            NotImplementedError: If the method declares ``requires_logits`` (the full-logits path is not
                implemented yet).
        """
        if not isinstance(name, str) or not name or name.startswith("_") or name in _RESERVED_METHODS:
            raise ValueError(f"loss_fn={name!r} is not a loss method name.")
        attribute = getattr(type(self), name, None)
        if attribute is None or not callable(attribute):
            raise ValueError(f"{type(self).__name__} has no loss method {name!r}.")
        if getattr(attribute, "requires_logits", False):
            # TODO(logits): call the model with labels=None and hand the loss the TP-sharded logits in ModelOutput.
            raise NotImplementedError(
                f"loss method {name!r} declares requires_logits; the full-logits path is not implemented yet."
            )
        return getattr(self, name)

    def cross_entropy(self, batch: PreparedMicrobatch, output: ModelOutput) -> torch.Tensor:
        """Negative log-likelihood summed over active positions (``loss_mask & ~padding_mask``)."""
        return losses.cross_entropy(batch, output)

    def importance_sampling(self, batch: PreparedMicrobatch, output: ModelOutput) -> torch.Tensor:
        """Importance-sampling policy-gradient loss (phase C1)."""
        raise NotImplementedError("importance_sampling is implemented in phase C1.")

    def ppo_clip(self, batch: PreparedMicrobatch, output: ModelOutput) -> torch.Tensor:
        """Clipped PPO objective with optional TIS correction (phase C1)."""
        raise NotImplementedError("ppo_clip is implemented in phase C1.")

    def target_logprobs(self, batch: PreparedMicrobatch, output: ModelOutput) -> list[torch.Tensor]:
        """Per-sample slices of the already computed target logprobs (phase B2)."""
        raise NotImplementedError("target_logprobs is implemented in phase B2.")

    def target_entropy(self, batch: PreparedMicrobatch, output: ModelOutput) -> list[torch.Tensor]:
        """Per-sample token entropies from the sharded logits (phase C1, requires_logits path)."""
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
    def recipe(self) -> TrainingRecipe:
        """The engine's private, finalized recipe."""
        return self._recipe

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

    @property
    def process_groups(self) -> ProcessGroupCollection:
        """The shared runtime's process groups (``dp``, ``cp``, ``tp``, ``ep``, ...), for custom pipelines."""
        return self._runtime.pg_collection

    @property
    def pad_multiple(self) -> int:
        """Per-sequence alignment the default pipeline pads to (CP zigzag and sequence-parallel constraints)."""
        return self._pad_multiple

    def close(self) -> None:
        """Release this engine's model, optimizer and buffers. The shared runtime and other engines survive."""
        if self._closed:
            return
        self._closed = True
        self._runtime.detach(self)
        for chunk in self._model:
            _release_ddp_autograd_refs(chunk)
        if self._model:
            # The model config is the provider held by self._recipe; Bridge installed optimizer-bound
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
        self._finalize_model_grads_func = None
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


def _build_frozen_model(recipe: TrainingRecipe, pg_collection: ProcessGroupCollection) -> list[MegatronModule]:
    """Build a model without DDP or gradient buffers and freeze its parameters."""
    provider = recipe.model
    if not isinstance(provider, ModelProviderMixin):
        raise NotImplementedError("trainable=False currently requires a ModelProviderMixin model config.")
    model = provider.provide_distributed_model(
        ddp_config=None,
        wrap_with_ddp=False,
        data_parallel_random_init=recipe.rng.data_parallel_random_init,
        pg_collection=pg_collection,
    )
    for chunk in model:
        for parameter in chunk.parameters():
            parameter.requires_grad_(False)
    return model
