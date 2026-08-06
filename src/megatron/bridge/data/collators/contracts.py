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

"""Contracts shared by model-owned collators and data-loader integrations."""

from collections.abc import Callable, Sequence
from typing import Any, Generic, Protocol, TypeVar


class PreparedSequence(Protocol):
    """Minimum prepared-sample interface required by sequence packing."""

    @property
    def sequence_length(self) -> int:
        """Return the exact unpadded token length of the prepared sequence."""
        ...


PreparedT = TypeVar("PreparedT", bound=PreparedSequence)


class ModelCollator:
    """Adapter for a model-owned full-batch collate callable."""

    def __init__(self, collate_fn: Callable[..., dict[str, Any]]) -> None:
        """Create an adapter without changing the callable's arguments or output."""
        if not callable(collate_fn):
            raise TypeError("collate_fn must be callable.")
        self._collate_fn = collate_fn

    @property
    def collate_fn(self) -> Callable[..., dict[str, Any]]:
        """Return the wrapped callable for legacy integrations."""
        return self._collate_fn

    def collate(self, examples: list[Any], processor: Any, **kwargs: Any) -> dict[str, Any]:
        """Collate a full batch using the model-owned implementation."""
        return self._collate_fn(examples, processor, **kwargs)

    def __call__(self, examples: list[Any], processor: Any, **kwargs: Any) -> dict[str, Any]:
        """Delegate callable-style use to :meth:`collate`."""
        return self.collate(examples, processor, **kwargs)


class PreparedSequenceCollator(ModelCollator, Generic[PreparedT]):
    """Collator that can prepare samples once and later assemble a packed batch."""

    def __init__(
        self,
        collate_fn: Callable[..., dict[str, Any]],
        *,
        prepare_one: Callable[..., PreparedT],
        pack_prepared: Callable[..., dict[str, Any]],
    ) -> None:
        """Create a collator with explicit single-sample and prepared-pack hooks."""
        super().__init__(collate_fn)
        if not callable(prepare_one):
            raise TypeError("prepare_one must be callable.")
        if not callable(pack_prepared):
            raise TypeError("pack_prepared must be callable.")
        self._prepare_one = prepare_one
        self._pack_prepared = pack_prepared

    def prepare_one(self, example: dict[str, Any], processor: Any, **kwargs: Any) -> PreparedT:
        """Process one normalized example into the model-owned prepared type."""
        return self._prepare_one(example, processor, **kwargs)

    def pack_prepared(self, sequences: Sequence[PreparedT], **kwargs: Any) -> dict[str, Any]:
        """Assemble prepared samples into the model-owned packed batch."""
        return self._pack_prepared(list(sequences), **kwargs)

    @staticmethod
    def aligned_length(sequence: PreparedSequence, *, multiple: int) -> int:
        """Return the physical token length consumed after per-sequence alignment."""
        if multiple <= 0:
            raise ValueError("multiple must be greater than 0.")
        return ((sequence.sequence_length + multiple - 1) // multiple) * multiple
