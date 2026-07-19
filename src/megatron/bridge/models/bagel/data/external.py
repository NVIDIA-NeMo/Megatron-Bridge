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

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Literal, Self

from megatron.bridge.data.base import DatasetBuildContext
from megatron.bridge.data.megatron_mimo.base_provider import MegatronMIMODatasetProvider


def _reject_collate(_: object) -> object:
    """Reject collation because Megatron external loaders bypass it."""
    raise RuntimeError("BAGEL external packed batches must not be collated again")


class BagelExternalLoader:
    """Expose packed BAGEL batches through Megatron's external-loader contract."""

    def __init__(self, batches: Iterable[dict[str, object]], *, length: int) -> None:
        """Wrap a packed batch stream with the length required by loader setup."""
        if length <= 0:
            raise ValueError("BAGEL external loader length must be positive")
        self._iterator = iter(batches)
        self._length = length

    def __iter__(self) -> Self:
        """Return this next-compatible external loader."""
        return self

    def __next__(self) -> dict[str, object]:
        """Return the next already-packed BAGEL batch."""
        return next(self._iterator)

    def __len__(self) -> int:
        """Return the configured number of training batches."""
        return self._length


@dataclass(kw_only=True)
class BagelMegatronMIMODatasetProvider(MegatronMIMODatasetProvider):
    """Pass one BAGEL packed-batch iterator into MegatronMIMO unchanged."""

    train_loader: BagelExternalLoader
    dataloader_type: Literal["external"] = "external"
    num_workers: int = 0
    persistent_workers: bool = False

    def build_datasets(self, context: DatasetBuildContext) -> tuple[BagelExternalLoader | None, None, None]:
        """Return only the external training loader requested by the build context."""
        return (self.train_loader if context.train_samples > 0 else None), None, None

    def get_collate_fn(self) -> Callable:
        """Return a guard callable; external loader setup never invokes it."""
        return _reject_collate
