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

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Self

from megatron.bridge.data.base import DatasetBuildContext
from megatron.bridge.data.megatron_mimo.base_provider import MegatronMIMODatasetProvider


def _reject_collate(_: object) -> object:
    """Reject collation because Megatron external loaders bypass it."""
    raise RuntimeError("BAGEL external packed batches must not be collated again")


class BagelExternalLoader:
    """Expose packed BAGEL batches through Megatron's external-loader contract."""

    def __init__(
        self,
        batches: Iterable[dict[str, object]],
        *,
        length: int,
        stateful_loaders: Sequence[object] = (),
    ) -> None:
        """Wrap a packed batch stream with the length required by loader setup."""
        if length <= 0:
            raise ValueError("BAGEL external loader length must be positive")
        self._batches = batches
        self._iterator = iter(batches)
        self._length = length
        self._position = 0
        self._stateful_loaders = stateful_loaders

    def __iter__(self) -> Self:
        """Return this next-compatible external loader."""
        return self

    def __next__(self) -> dict[str, object]:
        """Return the next already-packed BAGEL batch."""
        batch = next(self._iterator)
        self._position += 1
        return batch

    def __len__(self) -> int:
        """Return the configured number of training batches."""
        return self._length

    def save_state(self) -> dict[str, object]:
        """Capture the Energon readers, packer, and external-loader position."""
        if not self._stateful_loaders or not hasattr(self._batches, "state_dict"):
            raise RuntimeError("BAGEL checkpointing requires stateful group loaders and packer")
        return {
            "length": self._length,
            "position": self._position,
            "packer": self._batches.state_dict(),
            "loaders": [loader.save_state_rank() for loader in self._stateful_loaders],
        }

    def restore_state(self, state: Mapping[str, object]) -> None:
        """Restore the Energon readers, packer, and external-loader position."""
        if state["length"] != self._length:
            raise ValueError("BAGEL external-loader checkpoint length differs")
        loader_states = state["loaders"]
        if len(loader_states) != len(self._stateful_loaders):
            raise ValueError("BAGEL checkpoint group count differs")
        for loader, loader_state in zip(self._stateful_loaders, loader_states):
            loader.restore_state_rank(loader_state)
        self._batches.load_state_dict(state["packer"])
        self._position = state["position"]


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
