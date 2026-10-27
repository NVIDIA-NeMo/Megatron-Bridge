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
"""Data contracts for the TrainingEngine: Datum, ModelOutput, AdamParams, NormalizationPolicy."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch


@dataclass
class Datum:
    """One logical training or scoring sequence.

    Attributes:
        input_ids: Full token sequence (prompt + response), shape ``[T]``, dtype int64.
        loss_inputs: Token-aligned per-position fields consumed by loss methods, each of shape ``[T]``
            (for example ``target_tokens``, ``weights``, ``old_logprobs``, ``advantages``).
        sample_id: Logical identity of this Datum within a call; assigned by the engine when ``None``.
    """

    input_ids: torch.Tensor
    loss_inputs: dict[str, torch.Tensor] = field(default_factory=dict)
    sample_id: int | None = None

    @property
    def num_tokens(self) -> int:
        """Number of tokens in ``input_ids``."""
        return int(self.input_ids.shape[0])


@dataclass
class ModelOutput:
    """Application-facing result of ``forward_only`` / ``forward_backward``.

    Scoring lists (``logprobs``, ``entropy``, ``values``) are indexed by entry; ``sample_ids[i]`` names the
    Datum of entry ``i`` and ``token_positions[i]`` maps that entry to the Datum's original target positions.
    """

    loss: torch.Tensor | None = None
    logprobs: list[torch.Tensor] | None = None
    entropy: list[torch.Tensor] | None = None
    values: list[torch.Tensor] | None = None
    sample_ids: list[int] | None = None
    token_positions: list[torch.Tensor] | None = None
    metrics: dict[str, float] = field(default_factory=dict)


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

    TOKEN_MEAN = "token_mean"  # 1 / sum(weights > 0) over the whole update
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


def validate_datums(datums: list[Datum]) -> list[Datum]:
    """Check Datum contracts and assign missing ``sample_id`` values in input order.

    Args:
        datums: The global logical batch supplied to an engine call.

    Returns:
        The same list, with ``sample_id`` filled in where it was ``None``.

    Raises:
        TypeError: If an entry is not a ``Datum`` or a field is not a tensor.
        ValueError: If the batch is empty, ``None`` (designated-rank submission is not supported yet),
            ``input_ids`` is not a 1-D int64 tensor, or a ``loss_inputs`` field is not token-aligned.
    """
    if datums is None:
        raise ValueError("datums=None (designated-rank submission) is not supported; every rank passes the batch.")
    if len(datums) == 0:
        raise ValueError("datums must contain at least one Datum.")
    used_ids = {d.sample_id for d in datums if isinstance(d, Datum) and d.sample_id is not None}
    if len(used_ids) != sum(1 for d in datums if isinstance(d, Datum) and d.sample_id is not None):
        raise ValueError("Datum.sample_id values must be unique within a call.")
    next_id = 0
    for index, datum in enumerate(datums):
        if not isinstance(datum, Datum):
            raise TypeError(f"datums[{index}] is {type(datum).__name__}, expected Datum.")
        ids = datum.input_ids
        if not isinstance(ids, torch.Tensor):
            raise TypeError(f"datums[{index}].input_ids must be a tensor.")
        if ids.ndim != 1 or ids.dtype != torch.int64 or ids.numel() == 0:
            raise ValueError(
                f"datums[{index}].input_ids must be a non-empty 1-D int64 tensor, got {ids.shape} {ids.dtype}."
            )
        for name, value in datum.loss_inputs.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"datums[{index}].loss_inputs[{name!r}] must be a tensor.")
            if value.shape[:1] != ids.shape:
                raise ValueError(
                    f"datums[{index}].loss_inputs[{name!r}] has shape {tuple(value.shape)}; "
                    f"its first dimension must match input_ids length {ids.shape[0]}."
                )
        if datum.sample_id is None:
            while next_id in used_ids:
                next_id += 1
            datum.sample_id = next_id
            used_ids.add(next_id)
    return datums
