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
import types
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
            # _codecs.encode is used by NumPy to encode raw array bytes into the pickle stream
            "_codecs": frozenset({"encode"}),
        }
    )

    def find_class(self, module: str, name: str) -> type:
        if module in self._SAFE_MODULES and name in self._SAFE_MODULES[module]:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"Restricted unpickler refused to load '{module}.{name}'. "
            "Only safe built-in and numpy array types are allowed."
        )


class _EnergonUnpickler(_NumpyRestrictedUnpickler):
    """Unpickler for Energon dataloader state files (``.pt``).

    Extends the NumPy-safe unpickler with the exact Energon dataclass types that Energon serialises
    into dataloader checkpoint files.  All other globals — including ``os``, ``subprocess``, and any
    ``__reduce__`` payload callable outside this allowlist — are blocked, preventing arbitrary code
    execution from attacker-controlled checkpoint files.

    Use via :func:`energon_pickle_load` rather than instantiating directly.
    """

    _SAFE_MODULES: MappingProxyType = MappingProxyType(
        {
            **_NumpyRestrictedUnpickler._SAFE_MODULES,
            # PyTorch tensor reconstruction — required for any .pt file containing tensors.
            # These functions only rebuild tensor objects from pre-loaded storage; they do
            # not execute arbitrary code.
            "torch._utils": frozenset({"_rebuild_tensor_v2", "_rebuild_tensor"}),
            # Energon dataloader state types — the explicit allowlist for this load site.
            "megatron.energon.state": frozenset({"FlexState"}),
            "megatron.energon.rng": frozenset({"SystemRngState"}),
            "megatron.energon.savable_loader": frozenset(
                {
                    "SavableDataLoaderState",
                    "SavableDatasetCheckpoint",
                    "SavableDatasetState",
                }
            ),
            "megatron.energon.flavors.webdataset.sample_loader": frozenset({"SliceState"}),
        }
    )


def energon_torch_load(path: str, *, map_location: str = "cpu") -> object:
    """Load an Energon dataloader state ``.pt`` file through a restricted unpickler.

    Replaces ``torch.load(..., weights_only=True)`` + ``safe_globals`` for Energon checkpoint
    files.  Security is provided by :class:`_EnergonUnpickler`: any GLOBAL opcode not in its
    allowlist raises ``pickle.UnpicklingError``, blocking ``__reduce__``-based code execution.
    Using a custom ``pickle_module`` rather than ``weights_only=True`` avoids PyTorch's internal
    SETITEM restriction, which rejects dict subclasses such as ``FlexState`` in PyTorch ≥ 2.13.

    Args:
        path: Path to the ``.pt`` file written by
            :func:`~megatron.bridge.training.checkpointing.maybe_save_dataloader_state`.
        map_location: Passed to ``torch.load``; defaults to ``"cpu"`` to avoid GPU allocation
            during restore.

    Returns:
        The deserialized object (a ``dict`` containing ``"dataloader_state_dict"``).
    """
    import torch  # local import — keeps safe_pickle importable without torch on the test path

    _energon_pickle = types.ModuleType("_energon_pickle")
    for _k in dir(pickle):
        setattr(_energon_pickle, _k, getattr(pickle, _k))
    _energon_pickle.Unpickler = _EnergonUnpickler  # type: ignore[attr-defined]
    return torch.load(path, map_location=map_location, pickle_module=_energon_pickle, weights_only=False)


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
