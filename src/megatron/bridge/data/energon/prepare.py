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

"""Helpers for preparing Bridge-owned Energon WebDatasets."""

import logging
import re
from collections.abc import Mapping
from pathlib import Path


logger = logging.getLogger(__name__)


def _relative_tar_paths(dataset_path: Path) -> list[str]:
    paths = sorted((*dataset_path.rglob("*.tar"), *dataset_path.rglob("*.tgz")))
    return [str(path.relative_to(dataset_path)) for path in paths]


def _validate_split_patterns(paths: list[str], split_patterns: Mapping[str, str]) -> None:
    if not split_patterns:
        raise ValueError("split_patterns must contain at least one named split.")
    for split, pattern in split_patterns.items():
        if not split:
            raise ValueError("Energon split names must not be empty.")
        try:
            expression = re.compile(pattern)
        except re.error as error:
            raise ValueError(f"Invalid regex for Energon split {split!r}: {pattern!r}") from error
        if not any(expression.match(path) for path in paths):
            raise ValueError(f"Energon split {split!r} pattern {pattern!r} did not match any tar shards.")


def prepare_webdataset(
    dataset_path: str | Path,
    split_patterns: Mapping[str, str],
    *,
    num_workers: int = 8,
) -> None:
    """Index WebDataset shards and write deterministic Energon split metadata.

    This calls Energon's preparation API directly instead of its interactive
    command-line wrapper. The API is shared by supported Megatron-Energon 7.x
    releases even though their CLI flags differ.

    Args:
        dataset_path: Directory containing ``.tar`` or ``.tgz`` shards.
        split_patterns: Mapping from split names to shard-path regular expressions.
        num_workers: Number of parallel shard-indexing workers.

    Raises:
        FileNotFoundError: If ``dataset_path`` does not contain any tar shards.
        ValueError: If workers or split patterns are invalid.
    """
    if num_workers <= 0:
        raise ValueError("num_workers must be greater than zero.")

    root = Path(dataset_path)
    paths = _relative_tar_paths(root)
    if not paths:
        raise FileNotFoundError(f"No .tar or .tgz shards found under {root}.")
    _validate_split_patterns(paths, split_patterns)

    # This is the only preparation API available across Energon 7.0-7.4.
    # Keep the private-module dependency isolated here so examples do not need
    # version-specific CLI flags or interactive input.
    from megatron.energon.flavors.webdataset.base_webdataset import BaseWebdatasetFactory

    logger.info("Indexing %d Energon shards under %s", len(paths), root)
    BaseWebdatasetFactory.prepare_dataset(
        root,
        paths,
        split_parts_patterns=list(split_patterns.items()),
        shuffle_seed=None,
        workers=num_workers,
    )
