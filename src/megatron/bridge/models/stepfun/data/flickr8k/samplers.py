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

"""Model-owned looped sequential / shuffle / weighted-random samplers.

These drive :class:`MixedPackedDataloader._schedule_all`. The seeds and the
exact ``torch.randperm`` / ``heapq``-based selection order define the
packed batch sequence, so they must not be changed if reproducible packs
are required.
"""

from __future__ import annotations

import heapq
from typing import Any, Optional, Sequence, Union

import torch


class LoopedSequentialSampler:
    """size=3: [0, 1, 2, 0, 1, 2, ...]"""

    def __init__(self, size: int):
        self.size = size
        self.data_idx = 0

    def __iter__(self):
        return self

    def __next__(self) -> int:
        value = self.get()
        self.update()
        return value

    def get(self) -> int:
        return self.data_idx % self.size

    def update(self) -> None:
        self.data_idx += 1

    def state_dict(self) -> dict[str, Any]:
        return {"data_idx": self.data_idx}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.data_idx = state_dict["data_idx"]


class LoopedShuffleSampler:
    """Looped shuffle sampler.

    Yields a fresh ``torch.randperm`` permutation per epoch using
    ``seed = base_seed + epoch`` (or just ``base_seed`` if
    ``same_order_for_each_epoch`` is set).
    """

    def __init__(
        self,
        size: int = 0,
        base_seed: int = 1234,
        same_order_for_each_epoch: bool = False,
    ):
        self.size = size
        self.base_seed = base_seed
        self.same_order_for_each_epoch = same_order_for_each_epoch
        self.data_idx = 0
        self._idx_cur_epoch: list[int] = []
        self._reset_idx_cur_epoch()

    def __iter__(self):
        return self

    def __next__(self) -> int:
        value = self.get()
        self.update()
        return value

    def get(self) -> int:
        return self._idx_cur_epoch[self.data_idx % self.size]

    def update(self) -> None:
        self.data_idx += 1
        if self.data_idx % self.size == 0:
            self._reset_idx_cur_epoch()

    def state_dict(self) -> dict[str, Any]:
        return dict(size=self.size, data_idx=self.data_idx)

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if "size" in state_dict:
            assert self.size == state_dict["size"]
        self.data_idx = state_dict["data_idx"]
        self._reset_idx_cur_epoch()

    def _reset_idx_cur_epoch(self) -> None:
        epoch = self.data_idx // self.size
        seed = self.base_seed
        if not self.same_order_for_each_epoch:
            seed += epoch
        rng = torch.Generator().manual_seed(seed)
        self._idx_cur_epoch = torch.randperm(self.size, generator=rng).tolist()


class WeightedRandomSampler:
    """Heap-based balanced weighted sampler.

    Maintains a min-heap of cumulative scores ``count[i] / weight[i]`` and
    always picks the lowest-score index (ties broken by lower index, per
    Python's ``heapq`` invariant). This produces an exactly reproducible
    weighted order without floating-point randomness in the selection.
    """

    def __init__(
        self,
        size: int = 0,
        base_seed: int = 1234,
        weights: Optional[Union[Sequence[float], torch.Tensor]] = None,
    ):
        self.size = size
        self.base_seed = base_seed
        self.data_idx = 0
        self._pending_idx: Optional[int] = None
        self._rng = torch.Generator().manual_seed(int(self.base_seed))

        self._weights = (
            torch.ones(self.size, dtype=torch.float32)
            if weights is None
            else torch.as_tensor(weights, dtype=torch.float32)
        )
        weights_t = self._weights
        if weights_t.numel() != self.size:
            raise ValueError(f"weights length ({weights_t.numel()}) must match size ({self.size})")
        if torch.any(weights_t <= 0):
            raise ValueError("weights must be positive")

        self._counts = torch.zeros(self.size, dtype=torch.float32)
        self._inv_weights = (1.0 / self._weights).tolist()
        self._scores = [0.0 for _ in range(self.size)]
        self._heap = [(0.0, idx) for idx in range(self.size)]
        heapq.heapify(self._heap)

    def __iter__(self):
        return self

    def __next__(self) -> int:
        value = self.get()
        self.update()
        return value

    def _select_idx(self) -> int:
        while self._heap:
            score, idx = heapq.heappop(self._heap)
            if score == self._scores[idx]:
                return int(idx)
        # Heap should never be empty; rebuild defensively.
        self._heap = [(score, idx) for idx, score in enumerate(self._scores)]
        heapq.heapify(self._heap)
        score, idx = heapq.heappop(self._heap)
        return int(idx)

    def get(self) -> int:
        if self._pending_idx is None:
            self._pending_idx = self._select_idx()
        return self._pending_idx

    def update(self) -> None:
        idx = self._pending_idx if self._pending_idx is not None else self._select_idx()
        self._pending_idx = None
        self._counts[idx] += 1
        self._scores[idx] += self._inv_weights[idx]
        heapq.heappush(self._heap, (self._scores[idx], idx))
        self.data_idx += 1

    def state_dict(self) -> dict[str, Any]:
        return dict(
            size=self.size,
            data_idx=self.data_idx,
            counts=self._counts.tolist(),
            rng_state=self._rng.get_state().tolist(),
        )

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if "size" in state_dict:
            assert self.size == state_dict["size"]
        self.data_idx = state_dict["data_idx"]
        counts = state_dict.get("counts")
        if counts is None:
            self._counts = torch.zeros(self.size, dtype=torch.float32)
        else:
            if len(counts) != self.size:
                raise ValueError(f"counts length ({len(counts)}) must match size ({self.size})")
            self._counts = torch.tensor(counts, dtype=torch.float32)
        self._scores = [float(self._counts[i]) * self._inv_weights[i] for i in range(self.size)]
        self._heap = [(self._scores[i], i) for i in range(self.size)]
        heapq.heapify(self._heap)
        rng_state = state_dict.get("rng_state")
        if rng_state is None:
            self._rng.manual_seed(int(self.base_seed))
        else:
            self._rng.set_state(torch.tensor(rng_state, dtype=torch.uint8))
        self._pending_idx = None
