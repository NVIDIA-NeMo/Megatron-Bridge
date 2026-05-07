# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import io
import pickle
from types import MappingProxyType


_BUILTIN_SAFE_TYPES = frozenset(
    {
        "list",
        "dict",
        "tuple",
        "set",
        "frozenset",
        "bytes",
        "bytearray",
        "str",
        "int",
        "float",
        "bool",
        "complex",
        "slice",
        "range",
        "NoneType",
    }
)


class _RestrictedUnpickler(pickle.Unpickler):
    """Unpickler that only allows safe built-in types to prevent arbitrary code execution."""

    _SAFE_MODULES = MappingProxyType(
        {
            "builtins": _BUILTIN_SAFE_TYPES,
            "collections": frozenset({"OrderedDict"}),
        }
    )

    def find_class(self, module: str, name: str) -> type:
        if module in self._SAFE_MODULES and name in self._SAFE_MODULES[module]:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"Restricted unpickler refused to load '{module}.{name}'. Only safe built-in types are allowed."
        )


class _NumpyRestrictedUnpickler(pickle.Unpickler):
    """Unpickler that allows safe builtins and the narrow set of numpy types needed for object array reconstruction.

    NumPy object arrays (dtype='O') are serialized via pickle inside ``.npy``
    files.  The pickle stream references ``numpy.core.multiarray._reconstruct``,
    ``numpy.ndarray``, and ``numpy.dtype`` to rebuild the array container, while
    the *elements* (dicts, lists, ints, …) use only standard builtins.

    This unpickler permits exactly those types and nothing else — in particular,
    ``os``, ``subprocess``, ``builtins.eval``, etc. are blocked, preventing
    arbitrary-code-execution attacks via crafted ``.npy`` files.
    """

    _SAFE_MODULES = MappingProxyType(
        {
            "builtins": _BUILTIN_SAFE_TYPES,
            "collections": frozenset({"OrderedDict"}),
            # numpy types required to reconstruct an ndarray from pickle
            "numpy": frozenset({"ndarray", "dtype"}),
            "numpy.core.multiarray": frozenset({"_reconstruct", "scalar"}),
            # numpy ≥ 2.0 moved internals under ``numpy._core``
            "numpy._core.multiarray": frozenset({"_reconstruct", "scalar"}),
        }
    )

    def find_class(self, module: str, name: str) -> type:
        if module in self._SAFE_MODULES and name in self._SAFE_MODULES[module]:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"Restricted unpickler refused to load '{module}.{name}'. "
            "Only safe built-in and numpy array types are allowed."
        )


def safe_pickle_load(fp) -> object:
    """Deserialize from a file using a restricted unpickler that only allows safe types."""
    return _RestrictedUnpickler(fp).load()


def safe_pickle_loads(data: bytes) -> object:
    """Deserialize pickle data using a restricted unpickler that only allows safe types."""
    return _RestrictedUnpickler(io.BytesIO(data)).load()


def safe_load_npy(data: bytes):
    """Load a ``.npy`` file from raw bytes without enabling unrestricted pickle.

    For numeric arrays the fast ``allow_pickle=False`` path is used.  For object
    arrays (packed datasets storing dicts of variable-length lists) the pickle
    payload is deserialized through :class:`_NumpyRestrictedUnpickler`, which
    blocks dangerous modules like ``os`` and ``subprocess``.

    Args:
        data: Raw bytes of a ``.npy`` file.

    Returns:
        numpy.ndarray loaded from the file.
    """
    import numpy as np
    import numpy.lib.format as _fmt

    buf = io.BytesIO(data)

    # Fast path: non-object arrays don't need pickle at all.
    try:
        return np.load(buf, allow_pickle=False)
    except ValueError:
        pass

    # Object array: read past the .npy header so the buffer is positioned
    # at the pickle payload, then deserialize through the restricted unpickler.
    buf.seek(0)
    version = _fmt.read_magic(buf)
    reader = _fmt.read_array_header_1_0 if version[0] == 1 else _fmt.read_array_header_2_0
    reader(buf)  # advances past header

    return np.asarray(_NumpyRestrictedUnpickler(buf).load(), dtype=object)
