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

"""Strict flat attribute access for builder-backed model configs."""

import inspect
from typing import Any


class FlatTransformerConfigMixin:
    """Route flat model-config assignments to their declared owner.

    Builder-backed configs expose a flat user API while retaining a nested
    ``transformer`` dataclass for serialization and model construction. Fields
    declared by both configs are kept synchronized because both builder and
    transformer consumers may read them.
    """

    def __setattr__(self, name: str, value: Any, /) -> None:
        """Assign a declared outer or nested field and reject phantom fields."""
        try:
            transformer = object.__getattribute__(self, "transformer")
        except AttributeError:
            # Dataclass initialization assigns base fields before ``transformer``.
            object.__setattr__(self, name, value)
            return

        model_fields = getattr(type(self), "__dataclass_fields__", {})
        transformer_fields = getattr(type(transformer), "__dataclass_fields__", {})

        if name == "transformer":
            object.__setattr__(self, name, value)
        elif name in model_fields:
            object.__setattr__(self, name, value)
            if name in transformer_fields:
                setattr(transformer, name, value)
        elif name in transformer_fields:
            setattr(transformer, name, value)
        elif hasattr(inspect.getattr_static(type(self), name, None), "__set__"):
            object.__setattr__(self, name, value)
        elif name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            raise AttributeError(
                f"Neither {type(self).__name__} nor {type(transformer).__name__} declares a field named {name!r}."
            )


__all__ = ["FlatTransformerConfigMixin"]
