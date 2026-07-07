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

"""Shared dataset configuration and runtime build contracts."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from megatron.core.process_groups_config import ProcessGroupCollection

from megatron.bridge.training.tokenizers.tokenizer import MegatronTokenizer


@dataclass(kw_only=True)
class DataloaderConfig:
    """Base configuration for data loading."""

    dataloader_type: Literal["single", "cyclic", "batch", "external"] | None = None
    """Dataloader type used by the training setup."""

    num_workers: int = 2
    """Number of dataloader workers."""

    data_sharding: bool = True
    """Whether data is sharded across data-parallel ranks."""

    pin_memory: bool = True
    """Whether dataloaders pin host memory."""

    drop_last: bool = True
    """Whether dataloaders drop the last incomplete batch."""

    persistent_workers: bool = True
    """Whether dataloader workers persist between iterations."""

    trust_remote_code: bool | None = None
    """Whether remote code is trusted for a configured Hugging Face path."""

    def finalize(self) -> None:
        """Finalize dataloader field constraints."""
        if self.num_workers == 0 and self.persistent_workers:
            self.persistent_workers = False


@dataclass(frozen=True)
class DatasetBuildContext:
    """Runtime metadata supplied to dataset builders by training setup."""

    train_samples: int
    valid_samples: int
    test_samples: int
    tokenizer: MegatronTokenizer | None = None
    pg_collection: ProcessGroupCollection | None = None


@dataclass
class DatasetProvider(DataloaderConfig, ABC):
    """Deprecated custom dataset-provider compatibility contract."""

    @abstractmethod
    def build_datasets(self, context: DatasetBuildContext) -> tuple[Any | None, Any | None, Any | None]:
        """Build train, validation, and test datasets.

        Args:
            context: Runtime sample counts, tokenizer, and process groups.

        Returns:
            Train, validation, and test datasets. Optional splits may be ``None``.
        """
        raise NotImplementedError


def validate_declarative_mapping(value: dict[str, Any] | None, *, field_name: str) -> None:
    """Reject runtime objects from mappings stored in serializable configs."""

    def _validate(item: Any, path: str) -> None:
        if item is None or isinstance(item, (bool, int, float, str, Path)):
            return
        if isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                _validate(child, f"{path}[{index}]")
            return
        if isinstance(item, dict):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise TypeError(f"{path} keys must be strings, got {type(key).__name__}.")
                _validate(child, f"{path}.{key}")
            return
        raise TypeError(
            f"{path} must contain only declarative values (scalars, paths, lists, tuples, or mappings), "
            f"got {type(item).__name__}."
        )

    if value is not None:
        _validate(value, field_name)
