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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch
from megatron.core.packed_seq_params import PackedSeqParams


# Token-aligned RL fields every pipeline understands; packed, padded and CP-sliced like the targets.
COMMON_FIELDS = ("advantages", "old_logprobs", "reference_logprobs", "rollout_logprobs")


@dataclass
class Datum:
    """One logical, unpadded sample.

    ``input_ids[i]`` is the model input at position ``i`` and ``target_ids[i]`` the token it must predict
    (the sequence shifted by one; never shift across a packed boundary). ``loss_mask`` is a binary 0/1
    selection of the positions that enter the primary objective.

    Attributes:
        input_ids: ``[L]`` int64 model inputs.
        target_ids: ``[L]`` int64 targets aligned with ``input_ids``.
        loss_mask: ``[L]`` float32 binary objective selection.
        advantages: ``[L]`` float32 per-target advantage (optional).
        old_logprobs: ``[L]`` float32 target logprobs under the old policy (optional).
        reference_logprobs: ``[L]`` float32 target logprobs under a reference model (optional).
        rollout_logprobs: ``[L]`` float32 target logprobs recorded by the generation backend (optional).
        loss_inputs: Custom token-aligned ``[L]`` tensors; every Datum in a call must carry the same keys.
            They follow the standard transform with padding value 0 and keep the caller's dtype.
        sample_inputs: Custom per-sample values passed through untouched (never sliced).
    """

    input_ids: torch.Tensor
    target_ids: torch.Tensor
    loss_mask: torch.Tensor

    advantages: torch.Tensor | None = None
    old_logprobs: torch.Tensor | None = None
    reference_logprobs: torch.Tensor | None = None
    rollout_logprobs: torch.Tensor | None = None

    loss_inputs: dict[str, torch.Tensor] = field(default_factory=dict)
    sample_inputs: dict[str, Any] = field(default_factory=dict)

    @property
    def num_tokens(self) -> int:
        """Number of model-input positions."""
        return int(self.input_ids.shape[0])


@dataclass
class PreparedMicrobatch:
    """Strict execution input for one microbatch, CP-partitioned, on the executing rank's device.

    Token tensors are ``[1, T_local]`` for the packed (THD) layout and ``[B, S_local]`` for the padded
    layout. ``padding_mask`` is True at artificial padding. ``packed_seq_params`` is present exactly for
    the packed layout. ``mapping`` is opaque bookkeeping a preprocessor leaves for its matching
    postprocessor; the engine never interprets it.
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

    loss_inputs: dict[str, torch.Tensor] = field(default_factory=dict)
    sample_inputs: list[dict[str, Any]] = field(default_factory=list)
    mapping: Any = None

    def token_tensors(self) -> dict[str, torch.Tensor]:
        """Every token-aligned tensor by name (standard, common optional, and custom fields)."""
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
        tensors.update(self.loss_inputs)
        return tensors

    @property
    def active_mask(self) -> torch.Tensor:
        """Positions that enter the primary objective: ``loss_mask & ~padding_mask``."""
        return self.loss_mask.bool() & ~self.padding_mask


@dataclass
class ModelOutput:
    """Native per-microbatch model output on the output stage, aligned with the prepared token axis.

    ``logprobs`` is always present (``-nll`` of ``target_ids`` from the configured kernel, TP-reduced).
    ``logits`` is the TP-sharded vocabulary slice and exists only when the selected loss declares
    ``requires_logits``; the vocabulary interval then describes this rank's slice.
    """

    logprobs: torch.Tensor
    logits: torch.Tensor | None = None
    vocab_start_index: int | None = None
    vocab_end_index: int | None = None


@dataclass
class ForwardOutput:
    """Scoring result. Gathered: one entry per input Datum, in input order, all positions.

    Local (``gather_output=False``): entries for this rank's Datums and CP positions only, with
    ``sample_indices`` (position in the caller's list) and ``token_positions`` populated.
    """

    logprobs: list[torch.Tensor]
    entropy: list[torch.Tensor] | None = None
    sample_indices: list[int] | None = None
    token_positions: list[torch.Tensor] | None = None


@dataclass
class ForwardBackwardOutput:
    """Detached per-call training summary, reduced over DP x CP."""

    loss_sum: float
    num_loss_tokens: int


@dataclass
class AdamParams:
    """Per-step optimizer hyperparameters supplied by the caller to ``optim_step`` (Tinker-style)."""

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
    """LoRA adapter settings for a trainable engine (phase C3)."""

    rank: int
    alpha: float
    target_modules: list[str] = field(
        default_factory=lambda: ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]
    )
    dropout: float = 0.0


@dataclass
class CheckpointHandle:
    """Handle returned by ``save_state``; ``wait()`` blocks until the complete checkpoint is durable."""

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
