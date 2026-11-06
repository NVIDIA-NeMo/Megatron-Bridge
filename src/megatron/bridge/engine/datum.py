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
"""Data contracts for the TrainingEngine (see ``trainingengine-data-contract.md``).

Boundaries: the caller hands the engine loose input (default: a global ``list[Datum]``); preprocessing
produces strict ``PreparedMicrobatch`` tensors for Megatron; the model adapter returns ``ModelOutput``
per microbatch; postprocessing delivers ``ForwardOutput`` (scoring) or ``ForwardBackwardOutput``
(training) to the caller.

Each attribute below is marked ``strict`` or ``loose``. Strict attributes are validated by the engine
(``check_datums`` for ``Datum``, ``check_microbatches`` for ``PreparedMicrobatch``) and consumed by the
engine core or by Megatron, so their shape, dtype and meaning are fixed. Loose attributes belong to
the caller, the loss, or the pipeline that produced them: the engine transports them unchanged and
never interprets them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import torch
from megatron.core.packed_seq_params import PackedSeqParams


if TYPE_CHECKING:
    from megatron.bridge.engine.batching import MicrobatchMapping


# Token-aligned RL fields every pipeline understands; packed, padded and CP-sliced like the targets.
COMMON_FIELDS = ("advantages", "old_logprobs", "reference_logprobs", "rollout_logprobs")


@dataclass
class Datum:
    """One logical, unpadded sample.

    ``input_ids[i]`` is the model input at position ``i`` and ``target_ids[i]`` the token it must predict
    (the sequence shifted by one; never shift across a packed boundary). ``loss_mask`` is a binary 0/1
    selection of the positions that enter the primary objective.

    Attributes:
        input_ids: strict. ``[L]`` int64 model inputs.
        target_ids: strict. ``[L]`` int64 targets aligned with ``input_ids``.
        loss_mask: strict. ``[L]`` float32 binary selection of the primary-objective positions; also the
            token count used for normalization.
        advantages: strict when present. ``[L]`` float32 per-target advantage, read by ``ppo_clip`` and
            ``importance_sampling``. Optional; set for every Datum of a call or for none.
        old_logprobs: strict when present. ``[L]`` float32 target logprobs under the policy that
            produced the samples (the ratio denominator). Optional, all-or-none.
        reference_logprobs: strict when present. ``[L]`` float32 target logprobs under a reference model
            (KL penalty). Optional, all-or-none.
        rollout_logprobs: strict when present. ``[L]`` float32 target logprobs recorded by the generation
            backend (truncated importance sampling). Optional, all-or-none.
        routed_experts: strict when present. ``[L, num_moe_layers, topk]`` integer expert ids the
            generation backend routed each input position to, MoE layers in model order (rollout routing
            replay); any integer dtype, every entry a valid id (no sentinel: a Datum has no padding).
            Optional, all-or-none; the engine replays it when the recipe has ``router_replay`` set and
            rejects it otherwise.
        loss_inputs: loose values, strict transport. Custom token-aligned ``[L]`` tensors for the loss
            (for example ``returns`` and ``old_values`` for ``value_loss``); every Datum in a call carries
            the same keys. The engine only packs, pads with 0 and CP-slices them, keeping the dtype.
        sample_inputs: loose. Custom per-sample values passed through untouched (never sliced); the
            default pipeline collects them per microbatch in ``PreparedMicrobatch.sample_inputs``.
    """

    input_ids: torch.Tensor
    target_ids: torch.Tensor
    loss_mask: torch.Tensor

    advantages: torch.Tensor | None = None
    old_logprobs: torch.Tensor | None = None
    reference_logprobs: torch.Tensor | None = None
    rollout_logprobs: torch.Tensor | None = None
    routed_experts: torch.Tensor | None = None

    loss_inputs: dict[str, torch.Tensor] = field(default_factory=dict)
    sample_inputs: dict[str, Any] = field(default_factory=dict)

    @property
    def num_tokens(self) -> int:
        """Number of model-input positions."""
        return int(self.input_ids.shape[0])


@dataclass
class PreparedMicrobatch:
    """Execution input for one microbatch, CP-partitioned, on the executing rank's device.

    Every token-aligned tensor has the same shape: ``[1, T_local]`` for the packed (THD) layout and
    ``[B, S_local]`` for the padded layout. ``check_microbatches`` enforces the strict attributes for
    default and custom pipelines alike.

    Attributes:
        input_ids: strict. int64 CUDA model inputs, valid vocabulary ids.
        target_ids: strict. int64 targets aligned with ``input_ids`` (fed as ``labels``).
        position_ids: strict. int64 positions; the default pipeline uses ``0..L-1`` per sequence.
        loss_mask: strict. float32 binary primary-objective selection; with ``padding_mask`` it forms
            ``active_mask``, which every predefined loss and the token count use.
        padding_mask: strict. bool, True at artificial padding; passed to the model and excluded
            from every objective.
        packed_seq_params: strict. Megatron ``PackedSeqParams`` (``qkv_format="thd"``, int32 CUDA
            ``cu_seqlens``) exactly for the packed layout, ``None`` for the padded layout.
        advantages: strict when present. float32, aligned with the targets; see ``Datum``.
        old_logprobs: strict when present. float32; see ``Datum``.
        reference_logprobs: strict when present. float32; see ``Datum``.
        rollout_logprobs: strict when present. float32; see ``Datum``.
        routed_experts: strict when present. int64 ``[B, S_local, num_moe_layers, topk]`` /
            ``[1, T_local, num_moe_layers, topk]`` expert ids per token in the model's MoE-layer order,
            ``-1`` only at padded positions (``padding_mask`` True; the engine substitutes a valid routing
            there), a valid id everywhere else. When set, the engine replays these choices in every MoE
            router instead of the router's own top-k (rollout routing replay); it requires
            ``recipe.router_replay`` and must be set for every microbatch of a call or for none.
        loss_inputs: loose values, strict transport. The Datums' custom tensors after the same packing,
            padding (value 0) and CP split as the targets; shape is checked, dtype is the caller's.
        sample_inputs: loose. One dict per Datum in this microbatch, in the microbatch's segment order,
            copied from ``Datum.sample_inputs``; the default pipeline fills it, custom ones may not.
        mapping: loose. Bookkeeping a preprocessor leaves for its matching postprocessor; the engine
            core never reads it. The default pipeline stores a ``MicrobatchMapping`` (``batching.py``:
            per-Datum segments, the un-sharded token count, this rank's CP indices, filler flag), which
            the default ``postprocess_forward_only`` and ``postprocess_forward_backward`` read, the
            latter for the sample count. A custom pipeline may store anything or leave ``None``; the
            default count then reports ``None`` rather than guess.
    """

    input_ids: torch.Tensor
    target_ids: torch.Tensor
    position_ids: torch.Tensor
    loss_mask: torch.Tensor
    padding_mask: torch.Tensor
    packed_seq_params: PackedSeqParams | None

    advantages: torch.Tensor | None = None
    old_logprobs: torch.Tensor | None = None
    reference_logprobs: torch.Tensor | None = None
    rollout_logprobs: torch.Tensor | None = None
    routed_experts: torch.Tensor | None = None

    loss_inputs: dict[str, torch.Tensor] = field(default_factory=dict)
    sample_inputs: list[dict[str, Any]] = field(default_factory=list)
    mapping: MicrobatchMapping | Any = None

    def token_tensors(self) -> dict[str, torch.Tensor]:
        """Every token-aligned tensor by name (standard, common optional, routing, and custom fields).

        All share the token shape in their leading dims; ``routed_experts`` carries two trailing dims.
        """
        tensors = {
            "input_ids": self.input_ids,
            "target_ids": self.target_ids,
            "position_ids": self.position_ids,
            "loss_mask": self.loss_mask,
            "padding_mask": self.padding_mask,
        }
        for name in COMMON_FIELDS:
            value = getattr(self, name)
            if value is not None:
                tensors[name] = value
        if self.routed_experts is not None:
            tensors["routed_experts"] = self.routed_experts
        tensors.update(self.loss_inputs)
        return tensors

    @property
    def active_mask(self) -> torch.Tensor:
        """Positions that enter the primary objective: ``loss_mask & ~padding_mask``."""
        return self.loss_mask.bool() & ~self.padding_mask


@dataclass
class ModelOutput:
    """Native per-microbatch model output on the output stage, aligned with the prepared token axis.

    Built by the engine's model adapter and handed to the loss method together with the
    ``PreparedMicrobatch``; every tensor has the microbatch's token shape and carries autograd.

    Attributes:
        logprobs: strict. Target logprobs (``-nll`` of ``target_ids`` from the configured kernel,
            TP-reduced) on a language-model engine; ``None`` on a value-head engine.
        logits: strict, reserved. This rank's TP-sharded vocabulary slice, only when the selected loss
            declares ``requires_logits``. That path is not implemented yet: the engine rejects such
            losses and the field is always ``None``.
        vocab_start_index: strict, reserved. First vocabulary id of this rank's ``logits`` slice.
        vocab_end_index: strict, reserved. One past the last vocabulary id of this rank's slice.
        values: strict. FP32 per-token values on a value-head engine (``output_head="value"``);
            ``None`` on a language-model engine.
    """

    logprobs: torch.Tensor | None
    logits: torch.Tensor | None = None
    vocab_start_index: int | None = None
    vocab_end_index: int | None = None
    values: torch.Tensor | None = None


@dataclass
class ForwardOutput:
    """Scoring result of ``forward_only``, produced by the default postprocessing.

    Gathered (``gather_output=True``): one entry per input Datum, in the caller's order, every target
    position, identical on all ranks. Local (``gather_output=False``): entries for this rank's Datums
    and CP positions only, with ``sample_indices`` and ``token_positions`` telling where they belong.
    All tensors are detached FP32; padding is dropped. A custom postprocessor may return its own type.

    Attributes:
        logprobs: strict. Per-Datum target logprobs on a language-model engine, ``None`` on a
            value-head engine.
        entropy: strict, reserved. Per-Datum token entropies when ``return_entropy=True``; needs the
            full-logits path, which is not implemented yet, so always ``None``.
        sample_indices: strict. Local mode only: each entry's position in the caller's Datum list.
        token_positions: strict. Local mode only: each entry's target positions within its Datum, in
            ascending order (CP shards hold non-contiguous positions).
        values: strict. Per-Datum values on a value-head engine, ``None`` on a language-model engine.
    """

    logprobs: list[torch.Tensor] | None
    entropy: list[torch.Tensor] | None = None
    sample_indices: list[int] | None = None
    token_positions: list[torch.Tensor] | None = None
    values: list[torch.Tensor] | None = None


@dataclass
class ForwardBackwardOutput:
    """Per-call training summary: what ``postprocess_forward_backward`` returns and ``forward_backward``
    hands to the caller, reduced over DP x CP and identical on every rank.

    Per call, not per update. The engine adds each summary to its accumulation window, so a custom
    ``postprocess_forward_backward`` decides what the window sees: ``num_loss_tokens`` and
    ``num_samples`` are the normalization inputs, ``None`` meaning the pipeline cannot tell (the engine
    then raises when the active policy needs it).

    Attributes:
        loss_sum: strict. Sum of the loss method's returns over every microbatch of the call on all
            DP x CP ranks; it means whatever the selected loss computes (a token sum for the predefined
            losses) and is never divided by the engine.
        num_loss_tokens: strict. Active primary-objective positions (``loss_mask & ~padding_mask``) of
            the call, each counted once; the ``TOKEN_MEAN`` denominator input.
        metrics: loose keys, strict transport. Additive diagnostics a loss chose to return as its second
            value (sums and counts over the same active positions), summed over microbatches and reduced
            like ``loss_sum``; empty for losses that return only the scalar. Divide by
            ``num_loss_tokens`` for per-token means.
        num_samples: strict. Datums in the call across DP replicas, the ``SAMPLE_MEAN`` denominator
            input; the default derives it from ``MicrobatchMapping``, ``None`` when unknown.
    """

    loss_sum: float
    num_loss_tokens: int
    metrics: dict[str, float] = field(default_factory=dict)
    num_samples: int | None = None


@dataclass
class AdamParams:
    """Per-step optimizer hyperparameters supplied by the caller to ``optim_step`` (Tinker-style).

    Written into every parameter group before the step; the recipe's per-group ``lr_mult`` and
    ``wd_mult`` multipliers (for example no weight decay on biases and norms) still apply on top.

    Attributes:
        learning_rate: strict. Base learning rate for this step; the schedule is the caller's.
        beta1: strict. Adam first-moment decay.
        beta2: strict. Adam second-moment decay.
        eps: strict. Adam epsilon.
        weight_decay: strict. Base decoupled weight decay.
    """

    learning_rate: float
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.0


class NormalizationPolicy(str, Enum):
    """How the accumulated primary loss gradient is normalized at ``optim_step``."""

    TOKEN_MEAN = "token_mean"  # 1 / active tokens over the whole update
    SAMPLE_MEAN = "sample_mean"  # 1 / number of Datums over the whole update
    NONE = "none"  # caller already normalized


@dataclass
class LoraConfig:
    """LoRA adapter settings for a trainable engine. Not supported yet: ``from_recipe(lora=...)`` raises.

    Attributes:
        rank: strict. Adapter rank.
        alpha: strict. Adapter scaling.
        target_modules: strict. Megatron linear-module names the adapters attach to.
        dropout: strict. Adapter dropout.
    """

    rank: int
    alpha: float
    target_modules: list[str] = field(
        default_factory=lambda: ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]
    )
    dropout: float = 0.0


@dataclass
class CheckpointHandle:
    """Handle returned by ``save_state``; ``wait()`` blocks until the complete checkpoint is durable.

    Attributes:
        path: strict. The ``<root>/iter_<step>`` directory this save writes.
        _waiter: engine-private. The finalizer of an asynchronous save, ``None`` once waited for or
            for a synchronous save.
    """

    path: str
    _waiter: Any = field(default=None, repr=False)

    def wait(self) -> None:
        """Block until the checkpoint, including user state, is fully written."""
        if self._waiter is not None:
            self._waiter()
            self._waiter = None


def check_datums(datums: list[Datum]) -> list[Datum]:
    """Validate the default engine input: a non-empty global list of consistent Datums.

    Raises:
        TypeError: If an entry is not a ``Datum`` or a field is not a tensor.
        ValueError: If the list is ``None``/empty, shapes or dtypes violate the contract, or a common
            optional field is present for some Datums but not all.
    """
    if datums is None:
        raise ValueError("data=None is not supported by the default preprocess; every rank passes the global list.")
    if len(datums) == 0:
        raise ValueError("datums must contain at least one Datum.")
    present = {name: datums[0] is not None and getattr(datums[0], name, None) is not None for name in COMMON_FIELDS}
    routed = getattr(datums[0], "routed_experts", None)
    routed_shape = tuple(routed.shape[1:]) if isinstance(routed, torch.Tensor) else None
    loss_input_keys = set(getattr(datums[0], "loss_inputs", {}) or {})
    for index, datum in enumerate(datums):
        if not isinstance(datum, Datum):
            raise TypeError(f"datums[{index}] is {type(datum).__name__}, expected Datum.")
        n = datum.num_tokens
        _check_tensor(f"datums[{index}].input_ids", datum.input_ids, n, torch.int64)
        _check_tensor(f"datums[{index}].target_ids", datum.target_ids, n, torch.int64)
        _check_tensor(f"datums[{index}].loss_mask", datum.loss_mask, n, None)
        if n == 0:
            raise ValueError(f"datums[{index}] is empty.")
        for name in COMMON_FIELDS:
            value = getattr(datum, name)
            if (value is not None) != present[name]:
                raise ValueError(f"{name} must be set for every Datum or for none (datums[{index}] differs).")
            if value is not None:
                _check_tensor(f"datums[{index}].{name}", value, n, None)
        if (datum.routed_experts is not None) != (routed_shape is not None):
            raise ValueError(f"routed_experts must be set for every Datum or for none (datums[{index}] differs).")
        if datum.routed_experts is not None:
            _check_routed_experts(f"datums[{index}].routed_experts", datum.routed_experts, n, routed_shape)
        if set(datum.loss_inputs) != loss_input_keys:
            raise ValueError(
                f"loss_inputs keys must match on every Datum: datums[{index}] has {sorted(datum.loss_inputs)}, "
                f"datums[0] has {sorted(loss_input_keys)}."
            )
        for name, value in datum.loss_inputs.items():
            _check_tensor(f"datums[{index}].loss_inputs[{name!r}]", value, n, None)
    return datums


def _check_tensor(label: str, value: Any, length: int, dtype: torch.dtype | None) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{label} must be a tensor.")
    if value.ndim != 1 or value.shape[0] != length:
        raise ValueError(f"{label} must have shape [{length}], got {tuple(value.shape)}.")
    if dtype is not None and value.dtype != dtype:
        raise ValueError(f"{label} must be {dtype}, got {value.dtype}.")


def _check_routed_experts(label: str, value: Any, length: int, trailing: tuple[int, ...]) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{label} must be a tensor.")
    if value.ndim != 3 or value.shape[0] != length or tuple(value.shape[1:]) != trailing:
        raise ValueError(
            f"{label} must have shape [{length}, num_moe_layers, topk] with the same trailing dims as datums[0] "
            f"{list(trailing)}, got {tuple(value.shape)}."
        )
    if value.is_floating_point() or value.is_complex() or value.dtype == torch.bool:
        raise ValueError(f"{label} must hold integer expert ids, got {value.dtype}.")
    if bool((value < 0).any()):
        raise ValueError(f"{label} must hold a valid expert id at every position (Datums have no padding).")
