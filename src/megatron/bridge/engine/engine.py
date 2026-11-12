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

Construction (by model name or from a recipe), scoring, gradient accumulation, optimizer steps,
checkpoints, export and CPU/GPU placement are implemented. Modes that are not supported yet raise
``NotImplementedError`` with a ``TODO`` comment at the raise site; TODO.md in this package lists them.
"""

from __future__ import annotations

import contextlib
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
from megatron.core.transformer.moe.moe_logging import MoEMetricsTracker, set_moe_metrics_tracker
from megatron.core.utils import get_model_config

from megatron.bridge.engine import losses
from megatron.bridge.engine.accumulation import AccumulationWindow
from megatron.bridge.engine.batching import (
    check_microbatches,
    default_postprocess,
    default_preprocess,
    local_sample_count,
    sequence_pad_multiple,
)
from megatron.bridge.engine.checkpoint import (
    load_engine_state,
    load_pretrained_weights,
    save_engine_state,
    start_async_checkpoint_worker,
    stop_async_checkpoint_worker,
)
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
from megatron.bridge.engine.export import ExportState, export_hf_adapter, export_hf_weights, make_export_state
from megatron.bridge.engine.lora import build_lora, make_lora_hook
from megatron.bridge.engine.offload import HostState, check_movable, offload_to_cpu, reload_to_cuda
from megatron.bridge.engine.optim import finalize_and_step, zero_gradients
from megatron.bridge.engine.recipes.base import TrainingRecipe, data_parallel_size, finalize_recipe, is_multimodal
from megatron.bridge.engine.recipes.registry import apply_recipe_overrides, resolve_recipe
from megatron.bridge.engine.router_replay import RouterReplayState
from megatron.bridge.engine.schedule import run_forward_backward, run_forward_only
from megatron.bridge.engine.value_head import apply_value_head_model_settings, make_value_head_hook
from megatron.bridge.models.model_provider import ModelProviderMixin
from megatron.bridge.peft.lora import LoRA
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
        "postprocess_forward_only",
        "postprocess_forward_backward",
        "check_microbatches",
        "resolve_loss_method",
        "target_logprobs",
        "target_entropy",
        "save_state",
        "load_state",
        "wait_for_save",
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
        seq_length: int,
        checkpoint_path: str | None = None,
        hardware: str | None = None,
        hf_revision: str | None = None,
        overrides: dict[str, Any] | None = None,
        output_head: str = "lm",
        trainable: bool = True,
        lora: LoraConfig | None = None,
        hooks: list[PreWrapHook] | None = None,
        spawn_async_save_worker: bool = True,
    ) -> None:
        """Build an engine for an HF model from the registered recipe for this hardware and context.

        Resolution reads the model's HF config, keeps the recipes verified on ``hardware`` and takes the
        smallest context bucket covering ``seq_length`` (``recipes.resolve_recipe``). The recipe fixes the
        minimum GPU count; the world size must be a multiple of it and the rest becomes data parallelism.

        Args:
            model: HF model ID or local directory with the model config.
            seq_length: Packed token budget per microbatch.
            checkpoint_path: Megatron or HF checkpoint whose weights initialize the model, or ``None``
                for the provider's initialization.
            hardware: GPU family (``"GB300"``, ...), or ``None`` to detect it from the local device.
            hf_revision: HF revision when ``model`` is a model ID.
            overrides: Recipe fields to change, by dotted path (``{"optimizer.lr": 1e-6}``).
            output_head, trainable, lora, hooks, spawn_async_save_worker: As in ``from_recipe``.
        """
        recipe = resolve_recipe(model, seq_length=seq_length, hardware=hardware, hf_revision=hf_revision)
        apply_recipe_overrides(recipe, overrides or {})
        self._init_from_recipe(
            recipe,
            checkpoint_path=checkpoint_path,
            output_head=output_head,
            trainable=trainable,
            lora=lora,
            hooks=hooks,
            spawn_async_save_worker=spawn_async_save_worker,
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
        spawn_async_save_worker: bool = True,
    ) -> TrainingEngine:
        """Build an engine from a ``TrainingRecipe``.

        The recipe is deep-copied so several engines can be built from one recipe object. The world size
        must be a multiple of the recipe's minimum GPU count (``recipes.min_gpus``); extra GPUs scale
        data parallelism.

        Args:
            recipe: Model and execution configuration.
            checkpoint_path: Megatron or HF checkpoint whose weights initialize the model, or ``None``
                for the provider's initialization.
            output_head: ``"lm"`` scores and trains target logprobs through the language-model head.
                ``"value"`` builds a critic: the LM head is replaced by a per-token scalar head (Bridge's
                value-model hook) after ``checkpoint_path`` is loaded, MTP and embedding tying are turned
                off on the recipe copy, ``forward_only`` returns values and ``forward_backward`` defaults
                to ``value_loss``. One recipe serves the actor and the critic.
            trainable: Build the optimizer and gradient buffers. ``False`` yields a frozen scoring engine.
            lora: Train LoRA adapters on a frozen base instead of the full model (``output_head="lm"``,
                ``trainable=True`` and a ``checkpoint_path`` with the base weights are required). The base
                is loaded and LoRA applied before DDP and the optimizer are built, so both hold the
                adapters only. Routing stays the base model's: MoE router expert biases are frozen too,
                and quantile-balanced routers are rejected. ``forward_only(disable_adapters=True)`` then
                scores the base model (the usual RL reference, with no second engine), ``save_state``
                saves the adapters and their settings, and ``export`` merges them into the base weights
                unless ``adapter_only=True``.
            hooks: Extra pre-wrap hooks run after the Bridge-registered ones.
            spawn_async_save_worker: Spawn the asynchronous-save worker (one process per rank with its own
                CUDA context, alive until ``close``) before the model is built. ``False`` saves that cost
                but then ``save_state(async_save=True)`` raises; synchronous saves still work.

        Returns:
            A constructed engine attached to the shared runtime.
        """
        engine = cls.__new__(cls)
        engine._init_from_recipe(
            copy.deepcopy(recipe),
            checkpoint_path=checkpoint_path,
            output_head=output_head,
            trainable=trainable,
            lora=lora,
            hooks=hooks,
            spawn_async_save_worker=spawn_async_save_worker,
        )
        return engine

    def _init_from_recipe(
        self,
        recipe: TrainingRecipe,
        *,
        checkpoint_path: str | None,
        output_head: str,
        trainable: bool,
        lora: LoraConfig | None,
        hooks: list[PreWrapHook] | None,
        spawn_async_save_worker: bool,
    ) -> None:
        """Validate the options, finalize the recipe (owned by this engine) and build."""
        if output_head not in ("lm", "value"):
            raise ValueError(f"output_head must be 'lm' or 'value', got {output_head!r}.")
        peft = None
        if lora is not None:
            if output_head != "lm":
                raise ValueError("lora needs output_head='lm'; LoRA on a value head is not supported.")
            if not trainable:
                raise ValueError("lora needs trainable=True; a frozen engine already scores the base model.")
            if checkpoint_path is None:
                raise ValueError("lora needs checkpoint_path: the pretrained base weights the adapters train on.")
            peft = build_lora(lora, recipe)
        if output_head == "value" and is_multimodal(recipe):
            # TODO(multimodal): the value head replaces the language model's output layer inside the wrapper.
            raise NotImplementedError("output_head='value' is not implemented for multimodal models yet.")
        if output_head == "value":
            apply_value_head_model_settings(recipe.model)
        finalize_recipe(recipe)
        hooks = list(hooks or [])
        if output_head == "value":
            # Loads the checkpoint into the bare model, then swaps the head; _build skips its own load.
            hooks.append(make_value_head_hook(recipe, checkpoint_path))
        if peft is not None:
            # Loads the base into the bare model, then applies LoRA before DDP and the optimizer exist;
            # _build skips its own load.
            hooks.append(make_lora_hook(recipe, peft, checkpoint_path))
        runtime = _acquire_runtime(recipe)
        self._build(
            recipe,
            runtime,
            checkpoint_path=checkpoint_path,
            output_head=output_head,
            trainable=trainable,
            peft=peft,
            hooks=hooks,
            spawn_async_save_worker=spawn_async_save_worker,
        )

    def _build(
        self,
        recipe: TrainingRecipe,
        runtime: _SharedRuntime,
        *,
        checkpoint_path: str | None,
        output_head: str,
        trainable: bool,
        peft: LoRA | None,
        hooks: list[PreWrapHook],
        spawn_async_save_worker: bool,
    ) -> None:
        self._recipe = recipe
        self._runtime = runtime
        self._output_head = output_head
        self._trainable = trainable
        self._peft = peft
        self._closed = False
        self._step = 0
        self._window = AccumulationWindow()
        self._pending_save: CheckpointHandle | None = None
        self._export_state: ExportState | None = None
        self._warned_unscaled_aux = False
        self._host_state = HostState()
        self._resident = True  # False while cpu() has the state in host memory
        self._offloaded_bytes = 0
        # MCore's MoE metrics tracker is a process-wide singleton whose per-layer buffers are sized by the
        # first model that records into it; engines of different depth (a critic without MTP next to a
        # policy with it) would overflow each other's buffers. Each engine owns one and installs it before
        # its compute calls.
        self._moe_metrics_tracker = MoEMetricsTracker()
        # Before the model: the worker's CUDA context and IPC mappings need a GPU that is not yet full.
        self._async_checkpoint_state = start_async_checkpoint_worker(recipe) if spawn_async_save_worker else None
        pg_collection = runtime.pg_collection
        self._pad_multiple = sequence_pad_multiple(
            cp_size=runtime.topology.cp,
            tp_size=runtime.topology.tp,
            sequence_parallel=bool(getattr(recipe.model, "sequence_parallel", False)),
        )

        for hook in hooks:
            _register_pre_wrap_hook(recipe.model, hook)

        # MCore's RouterReplay instances register in a process-wide list; claim the ones this build creates.
        replay_mark = RouterReplayState.mark()
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
        self._router_replay = RouterReplayState.claim(
            replay_mark,
            sequence_parallel=runtime.topology.tp > 1 and bool(getattr(recipe.model, "sequence_parallel", False)),
        )

        if checkpoint_path is not None and output_head != "value" and peft is None:
            # The value-head and LoRA pre-wrap hooks load the checkpoint into the bare model instead.
            load_pretrained_weights(recipe, self._model, self._optimizer, checkpoint_path)

        # Bridge installs the schedule hooks on the model config (finalize_model_grads_func, grad_scale_func,
        # and the overlap hooks). The engine keeps only the finalizer: run_forward_backward suppresses it
        # per call, and optim_step invokes it once per update (see schedule.py / optim.py). With LoRA the
        # finalizer also sums, across expert-parallel ranks, the gradients of expert adapters replicated
        # there (share_expert_adapters), and the per-layer fallback hooks for that sum are removed.
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
                peft_enabled=peft is not None,
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
        """Score a batch and return per-token target logprobs (or values, on a value-head engine) per sample.

        ``data`` is whatever this engine's ``preprocess`` accepts; the default accepts the same global
        ``list[Datum]`` on every rank. ``gather_output=True`` returns every Datum's complete scores on
        every rank in input order; ``False`` returns only what this rank computed (its DP replica's
        samples, its CP positions) with ``sample_indices`` and ``token_positions``. Gradients and any
        open accumulation window are untouched. Microbatches carrying ``routed_experts`` are scored
        with those expert choices replayed in the MoE routers (recipe ``router_replay``).
        ``disable_adapters=True`` scores a LoRA engine's base model (the adapters are switched off for
        this call only); it raises on an engine without LoRA rather than return the full model's scores.
        """
        self._require_open()
        self._require_resident("forward_only")
        self._activate()
        if return_entropy:
            # TODO(logits): entropy needs the model called without labels and a chunked TP-aware reduction.
            raise NotImplementedError("return_entropy needs the full-logits path, which is not implemented yet.")
        if disable_adapters and self._peft is None:
            raise ValueError("disable_adapters=True needs an engine built with lora=...; this engine has no adapters.")
        if self._runtime.topology.pp > 1:
            raise NotImplementedError(
                "forward_only with pipeline parallelism (broadcast_output) is not supported yet."
            )
        microbatches = self.preprocess(data, training=False)
        self.check_microbatches(microbatches)
        adapters_off = self._peft.disable_adapter(self._model) if disable_adapters else contextlib.nullcontext()
        with adapters_off:
            scores = run_forward_only(
                self._model,
                microbatches,
                pg_collection=self._runtime.pg_collection,
                mtp_target_scope=self._recipe.mtp_target_scope,
                output_head=self._output_head,
                router_replay=self._router_replay,
            )
        return self.postprocess_forward_only(microbatches, scores, gather_output=gather_output)

    def forward_backward(
        self,
        data: Any,
        *,
        loss_fn: str | None = None,
        loss_kwargs: dict[str, Any] | None = None,
        broadcast_output: bool = True,
    ) -> ForwardBackwardOutput:
        """Run forward/backward with the named loss and accumulate FP32 gradients.

        Gradients are summed, unnormalized, into the DDP buffers across calls; ``optim_step`` normalizes
        and applies them. Returns this call's detached loss sum, active-token count and any additive
        metrics the loss returned, all reduced over DP x CP. ``loss_fn`` defaults to ``cross_entropy``
        on a language-model engine and ``value_loss`` on a value-head engine; ``loss_kwargs`` are bound
        to the loss method as keyword arguments (for example ``ppo_clip``'s clip bounds). Microbatches
        carrying ``routed_experts`` train with those expert choices replayed (recipe ``router_replay``).
        """
        self._require_open()
        self._require_resident("forward_backward")
        self._require_trainable("forward_backward")
        self._activate()
        if loss_fn is None:
            loss_fn = "value_loss" if self._output_head == "value" else "cross_entropy"
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
        loss_sum, num_loss_tokens, metrics = run_forward_backward(
            self._model,
            microbatches,
            loss_method,
            pg_collection=self._runtime.pg_collection,
            mtp_target_scope=self._recipe.mtp_target_scope,
            output_head=self._output_head,
            router_replay=self._router_replay,
        )
        fwd_bwd_output = self.postprocess_forward_backward(microbatches, loss_sum, num_loss_tokens, metrics)
        self._add_to_window(fwd_bwd_output)
        return fwd_bwd_output

    def _add_to_window(self, fwd_bwd_output: ForwardBackwardOutput) -> None:
        """Apply one call's output to the accumulation window, or abort the window when a count the
        normalization needs is unknown (the gradients of the whole window are discarded)."""
        if fwd_bwd_output.num_samples is None and self.normalization == NormalizationPolicy.SAMPLE_MEAN:
            zero_gradients(self._model, self._optimizer)
            self._window.reset()
            raise ValueError(
                "NormalizationPolicy.SAMPLE_MEAN needs num_samples, but postprocess_forward_backward returned "
                "None; override it together with preprocess when the mapping is not MicrobatchMapping. The "
                "accumulation window was discarded."
            )
        self._window.add(
            num_loss_tokens=fwd_bwd_output.num_loss_tokens,
            num_samples=fwd_bwd_output.num_samples or 0,
            loss_sum=fwd_bwd_output.loss_sum,
        )

    # ---- data pipeline extension points (override in a subclass) -------------------------------

    def preprocess(self, data: Any, *, training: bool) -> list[PreparedMicrobatch]:
        """Turn caller input into this rank's strict microbatches.

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

    def postprocess_forward_only(
        self, microbatches: list[PreparedMicrobatch], logprobs: list[torch.Tensor], *, gather_output: bool
    ) -> ForwardOutput:
        """Turn per-microbatch ``[T_local]`` scores into the ``forward_only`` result.

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
            field="values" if self._output_head == "value" else "logprobs",
        )

    def postprocess_forward_backward(
        self,
        microbatches: list[PreparedMicrobatch],
        loss_sum: torch.Tensor,
        num_loss_tokens: torch.Tensor,
        metrics: dict[str, torch.Tensor],
    ) -> ForwardBackwardOutput:
        """Turn this rank's raw call results into the ``forward_backward`` summary.

        The summary is what the caller receives and what the engine adds to its accumulation window, so
        this method owns the normalization inputs: ``loss_sum``, ``num_loss_tokens`` and ``metrics`` are
        reduced over DP x CP, and ``num_samples`` (Datums across DP, for ``SAMPLE_MEAN``) is derived from
        the default ``MicrobatchMapping`` and left ``None`` for any other mapping. Override together with
        ``preprocess`` when the mapping differs; every field must be identical on every rank.

        One float64 collective: counts stay exact (float32 would round above 2**24 tokens). Metric keys
        are sorted so every rank stacks them in the same order.
        """
        pg_collection = self._runtime.pg_collection
        keys = sorted(metrics or {})
        values = [loss_sum.detach().double(), num_loss_tokens.double()]
        values += [metrics[key].detach().double() for key in keys]
        totals = torch.stack(values)
        torch.distributed.all_reduce(totals, group=pg_collection.dp_cp)
        reduced = totals.tolist()
        return ForwardBackwardOutput(
            loss_sum=float(reduced[0]),
            num_loss_tokens=int(reduced[1]),
            metrics={key: float(value) for key, value in zip(keys, reduced[2:])},
            num_samples=self._global_sample_count(microbatches),
        )

    def _global_sample_count(self, microbatches: list[PreparedMicrobatch]) -> int | None:
        """Datums in this call across DP replicas from the default mapping, ``None`` for any other mapping."""
        local = local_sample_count(microbatches)
        if local is None:
            return None
        count = torch.tensor([local], device="cuda")
        torch.distributed.all_reduce(count, group=self._runtime.pg_collection.dp)
        return int(count.item())

    def check_microbatches(self, microbatches: list[PreparedMicrobatch]) -> None:
        """The strict gate before Megatron: shapes, dtypes, devices, vocabulary range, layout metadata,
        and equal microbatch counts across DP replicas. Always runs, for default and custom pipelines."""
        if not microbatches:
            raise ValueError("preprocess returned no microbatches; every DP replica must run at least one.")
        routed_shape = None
        if getattr(self._recipe, "router_replay", False) and self._router_replay.num_layers:
            routed_shape = (self._router_replay.num_layers, int(self._recipe.model.moe_router_topk))
        check_microbatches(
            microbatches,
            layout=self._recipe.batch_layout,
            vocab_size=int(self._recipe.model.vocab_size),
            dp_group=self._runtime.pg_collection.dp,
            routed_shape=routed_shape,
            num_experts=getattr(self._recipe.model, "num_moe_experts", None),
        )
        if is_multimodal(self._recipe):
            # TODO(multimodal): image and video inputs (pixel data and grid positions on the Datum).
            visual_ids = torch.tensor(
                [self._recipe.model.image_token_id, self._recipe.model.video_token_id],
                device=microbatches[0].input_ids.device,
            )
            if any(torch.isin(mb.input_ids, visual_ids).any() for mb in microbatches):
                raise NotImplementedError("Image and video tokens in input_ids are not supported yet; text only.")

    def optim_step(self, adam_params: AdamParams, *, normalize_gradients: bool = True) -> dict[str, float]:
        """Normalize the accumulated gradients once, finalize, and step the optimizer.

        The divisor follows ``self.normalization`` (active tokens ``D``, Datums ``N``, or none);
        ``normalize_gradients=False`` bypasses it. The caller supplies the learning rate and the other
        Adam hyperparameters for this step. An update is skipped (buffers zeroed, window closed) when
        the normalization domain is empty or the optimizer finds non-finite gradients.
        """
        self._require_open()
        self._require_resident("optim_step")
        self._require_trainable("optim_step")
        self._activate()
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
        if scale is None and self._has_coupled_auxiliary_losses() and not self._warned_unscaled_aux:
            # The user's normalization is respected for every gradient. MCore's own MTP and router losses
            # are attached as per-token sums that the loss method never sees; scaling them is the
            # recipe's job through model.mtp_grad_scale_func / model.moe_grad_scale_func.
            self._warned_unscaled_aux = True
            logger.warning(
                "Gradients are not normalized by the engine, but the model adds MTP or router auxiliary "
                "losses as per-token sums. Scale them yourself with recipe.model.mtp_grad_scale_func / "
                "moe_grad_scale_func, or they outweigh a normalized primary loss by the token count."
            )
        # TODO(detached-mtp): the MTP head is divided by the primary token count N like every other parameter.
        # Today, per depth k: MCore multiplies the head's loss by this rank's own ratio (targets before the
        # shift / targets after the shift) before the backward, then we divide by N. The head therefore sees
        # a rank-weighted average, not a true mean over its own targets. The goal is: head loss sum / M_k,
        # where M_k counts depth-k targets pooled over DP x CP and the whole window. Two steps:
        #   1. cancel MCore's per-rank ratio before the backward (an upstream option, or mtp_grad_scale_func
        #      returning the inverse ratio per microbatch; exact only with one MTP depth, the scale is shared);
        #   2. count M_k (active positions at least k places into their sequence, all-reduced, summed over
        #      the window) and divide the parameters tagged grad_norm_group == "mtp" by M_k instead of N.
        # Coupled heads keep today's formula on purpose: it is the SFT baseline's.
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
        self._require_resident("zero_grad")
        self._require_trainable("zero_grad")
        zero_gradients(self._model, self._optimizer)
        self._window.reset()

    @property
    def step(self) -> int:
        """Number of successful optimizer updates since construction."""
        return self._step

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

    def importance_sampling(
        self, batch: PreparedMicrobatch, output: ModelOutput, *, tis_coef: float | None = None
    ) -> losses.LossReturn:
        """Importance-weighted policy gradient with optional truncated importance sampling; see ``losses``."""
        return losses.importance_sampling(batch, output, tis_coef=tis_coef)

    def ppo_clip(
        self,
        batch: PreparedMicrobatch,
        output: ModelOutput,
        *,
        clip_low: float,
        clip_high: float,
        kl_beta: float = 0.0,
        tis_coef: float | None = None,
    ) -> losses.LossReturn:
        """Clipped PPO objective with optional KL penalty to the reference and TIS correction; see ``losses``."""
        return losses.ppo_clip(
            batch, output, clip_low=clip_low, clip_high=clip_high, kl_beta=kl_beta, tis_coef=tis_coef
        )

    def value_loss(
        self, batch: PreparedMicrobatch, output: ModelOutput, *, clip: float | None = None
    ) -> losses.LossReturn:
        """Critic regression on ``loss_inputs["returns"]``, optionally clipped around ``old_values``."""
        return losses.value_loss(batch, output, clip=clip)

    def target_logprobs(self, batch: PreparedMicrobatch, output: ModelOutput) -> list[torch.Tensor]:
        """This rank's target logprobs per Datum of the microbatch, autograd intact (for custom losses)."""
        return losses.target_logprobs(batch, output)

    def target_entropy(self, batch: PreparedMicrobatch, output: ModelOutput) -> list[torch.Tensor]:
        """Per-sample token entropies from the sharded logits (needs the full-logits path)."""
        # TODO(logits): chunked TP-aware entropy over ModelOutput.logits.
        raise NotImplementedError("target_entropy needs the full-logits path, which is not implemented yet.")

    # ---- state --------------------------------------------------------------------------------

    def save_state(self, path: str, *, user_state: Any = None, async_save: bool = False) -> CheckpointHandle:
        """Save model, optimizer, RNG, the update count and per-rank ``user_state`` as ``<path>/iter_<step>``.

        Collective. Only between updates: an open accumulation window raises. A pending asynchronous
        save is waited for first, so at most one is in flight; ``async_save`` needs the engine's save worker
        (``spawn_async_save_worker``, on by default). ``user_state`` is any picklable object; each rank's
        copy is returned to that rank by ``load_state``.
        """
        self._require_open()
        self._require_resident("save_state")
        # TODO(save-mid-window): allow saving with an open window. The snapshot would be consistent (weights
        # and optimizer state of the last completed update); only document that the pending gradients are
        # not part of it, so a resumed engine redoes the interrupted update's forward_backward calls.
        self._require_closed_window("save_state")
        self._wait_pending_save()
        handle = save_engine_state(
            self._recipe,
            self._model,
            self._optimizer,
            step=self._step,
            root=path,
            user_state=user_state,
            async_save=async_save,
            async_state=self._async_checkpoint_state,
            pg_collection=self._runtime.pg_collection,
            peft=self._peft,
        )
        if async_save:
            self._pending_save = handle
        return handle

    def load_state(self, path: str) -> Any | None:
        """Restore a ``save_state`` checkpoint (the save root's latest step, or one ``iter_<step>`` directory).

        Collective. Restores weights, optimizer state when this engine is trainable, RNG state and the
        update count; returns this rank's ``user_state`` or ``None``. Only between updates. A LoRA
        engine's checkpoint holds the adapters only; they are restored over the base weights the engine
        was built with, so build it from the same base ``checkpoint_path`` as the engine that saved.
        The checkpoint's LoRA settings (rank, alpha, dropout, targets; none without LoRA) must equal this
        engine's, or it raises before anything is read.
        """
        self._require_open()
        self._require_resident("load_state")
        self._require_closed_window("load_state")
        self._wait_pending_save()
        step, user_state = load_engine_state(self._recipe, self._model, self._optimizer, path=path, peft=self._peft)
        self._step = step
        return user_state

    def wait_for_save(self) -> None:
        """Block until the pending asynchronous save, if any, is durable on every rank.

        Collective when a save is pending (all ranks must call it); a no-op otherwise. ``save_state``,
        ``load_state`` and ``close`` call it implicitly, and the ``CheckpointHandle`` returned by
        ``save_state`` offers the same through ``wait()``.
        """
        self._require_open()
        self._wait_pending_save()

    def _require_closed_window(self, name: str) -> None:
        if self._window.is_open:
            raise RuntimeError(f"{name} needs a completed update: call optim_step or zero_grad first.")

    def _wait_pending_save(self) -> None:
        if self._pending_save is not None:
            self._pending_save.wait()
            self._pending_save = None

    def export(
        self,
        *,
        path: str | None = None,
        streaming: bool = False,
        adapter_only: bool = False,
        base_model_name_or_path: str | None = None,
    ) -> Iterator[tuple[str, torch.Tensor]] | None:
        """Export the current weights in the HF layout of ``recipe.hf_path``.

        Collective. ``path`` writes a complete HF checkpoint directory (config, tokenizer files,
        safetensors; rank 0 writes). ``streaming=True`` instead returns an iterator of
        ``(hf_name, tensor)`` with full tensors on every rank; every rank must drain it, since the
        TP/EP/PP gathers run inside. A LoRA engine's adapters are merged into the base weights (in fp32,
        then cast to the base dtype); ``adapter_only=True`` exports the adapters alone, unmerged, in the
        HF PEFT layout: ``lora_A`` / ``lora_B`` tensors when streaming, or with ``path`` a PEFT directory
        whose ``adapter_config.json`` records ``base_model_name_or_path`` as the base model.

        ``base_model_name_or_path`` is required for that PEFT directory and rejected otherwise. It names
        the HF model whose weights the adapters were trained on; the engine cannot infer it, because
        ``checkpoint_path`` may hold other weights than ``recipe.hf_path`` (a fine-tuned checkpoint, say),
        and ``recipe.hf_path`` only supplies the layout.
        """
        self._require_open()
        self._require_resident("export")
        if self._output_head == "value":
            # TODO(value-export): the HF causal-LM layout has no slot for value_head; define a critic layout.
            raise NotImplementedError("export of a value-head engine is not supported yet.")
        if adapter_only and self._peft is None:
            raise ValueError("adapter_only=True needs an engine built with lora=...; this engine has no adapters.")
        writes_peft_directory = adapter_only and not streaming and path is not None
        if writes_peft_directory and not base_model_name_or_path:
            raise ValueError(
                "export(adapter_only=True, path=...) needs base_model_name_or_path: the HF model the adapters "
                "were trained on, which adapter_config.json records. recipe.hf_path is not assumed, since "
                "checkpoint_path may hold other weights."
            )
        if base_model_name_or_path is not None and not writes_peft_directory:
            raise ValueError("base_model_name_or_path only applies to export(adapter_only=True, path=...).")
        if self._export_state is None:
            self._export_state = make_export_state(self._recipe)
        if adapter_only:
            return export_hf_adapter(
                self._export_state,
                self._model,
                self._peft,
                path=path,
                streaming=streaming,
                base_model_name_or_path=base_model_name_or_path,
            )
        return export_hf_weights(self._export_state, self._model, path=path, streaming=streaming)

    # ---- placement ----------------------------------------------------------------------------

    def cuda(self) -> TrainingEngine:
        """Move the state back to the GPU, including the gradients of an open accumulation window.

        No-op when the engine is already resident. The shared runtime (process groups) is never moved,
        so other engines can run in between.
        """
        self._require_open()
        if not self._resident:
            reload_to_cuda(self._model, self._host_state)
            self._resident = True
            self._offloaded_bytes = 0
        return self

    def cpu(self) -> TrainingEngine:
        """Offload parameters, optimizer state and any open window's gradients to pinned host memory.

        No-op when already offloaded. Waits for a pending asynchronous save first. While offloaded, the
        compute, state and export calls raise; ``cuda()`` restores everything, with the window's counts
        and gradients intact. Unsupported buffer or optimizer modes raise before anything moves.
        """
        self._require_open()
        if self._resident:
            check_movable(self._recipe, self._optimizer)
            self._wait_pending_save()
            self._offloaded_bytes = offload_to_cpu(
                self._model, self._optimizer, self._host_state, keep_gradients=self._window.is_open
            )
            self._resident = False
        return self

    @property
    def device(self) -> str:
        """``"cuda"`` while resident, ``"cpu"`` after ``cpu()``."""
        return "cuda" if self._resident else "cpu"

    @property
    def offloaded_bytes(self) -> int:
        """GPU bytes released by the last ``cpu()``; 0 while resident. Compare with ``torch.cuda.memory_allocated``."""
        return self._offloaded_bytes

    @property
    def router_replay_layers(self) -> list[int]:
        """Zero-based decoder layer indices whose routing ``routed_experts`` replays, in field order; empty
        without ``recipe.router_replay``. A generation backend that records every decoder layer must keep
        exactly these columns (dense layers have no router)."""
        return self._router_replay.layer_indices if self._recipe.router_replay else []

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
        self._wait_pending_save()
        if self._async_checkpoint_state is not None:
            stop_async_checkpoint_worker(self._async_checkpoint_state)
            self._async_checkpoint_state = None
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
        self._router_replay.release()
        self._optimizer = None
        self._peft = None
        self._finalize_model_grads_func = None
        self._export_state = None
        self._model = []
        self._host_state = HostState()
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

    def _require_resident(self, api: str) -> None:
        if not self._resident:
            raise RuntimeError(f"{api}() needs the engine on the GPU; call cuda() first (it was moved with cpu()).")

    def _activate(self) -> None:
        """Point MCore's process-wide MoE metrics tracker at this engine's own instance."""
        set_moe_metrics_tracker(self._moe_metrics_tracker)


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
