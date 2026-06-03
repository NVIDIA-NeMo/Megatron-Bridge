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

"""Compatibility helpers for Energon sample and batch metadata."""

from collections.abc import Sequence
from dataclasses import fields, is_dataclass
from typing import Any

from megatron.energon import Batch, Sample


def _field_names(dataclass_type: type) -> set[str]:
    """Return dataclass field names, or an empty set for non-dataclass Energon bases."""
    if not is_dataclass(dataclass_type):
        return set()
    return {field.name for field in fields(dataclass_type)}


def sample_metadata_kwargs(*, key: str, restore_key: Any = (), subflavors: Any = None) -> dict[str, Any]:
    """Return Sample metadata kwargs accepted by the installed Energon version.

    Energon 7 removed the singular ``__subflavor__`` field while Energon 6 still
    requires it. Build the kwargs from the installed base dataclass so callers can
    construct Bridge sample subclasses under either contract.
    """
    sample_fields = _field_names(Sample)
    kwargs: dict[str, Any] = {
        "__key__": key,
        "__restore_key__": restore_key,
    }
    # TODO: remove this guard when Bridge no longer supports megatron-energon 6.x.
    if "__subflavor__" in sample_fields:
        kwargs["__subflavor__"] = None
    if "__subflavors__" in sample_fields:
        kwargs["__subflavors__"] = subflavors
    return kwargs


def batch_metadata_kwargs(*, keys: Sequence[str], restore_keys: Sequence[Any] | None = None) -> dict[str, Any]:
    """Return Batch metadata kwargs accepted by the installed Energon version."""
    batch_fields = _field_names(Batch)
    # TODO: remove this guard when Bridge no longer supports megatron-energon 6.x.
    if not batch_fields:
        return {}

    kwargs: dict[str, Any] = {}
    if "__key__" in batch_fields:
        kwargs["__key__"] = list(keys)
    if "__restore_key__" in batch_fields:
        if restore_keys is None:
            restore_keys = [() for _ in keys]
        kwargs["__restore_key__"] = list(restore_keys)
    return kwargs
