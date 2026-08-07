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
import zipfile
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


def _load_energon_zip(path: str, *, map_location: str) -> object:
    """Parse a torch zip-format ``.pt`` file and deserialize it through :class:`_EnergonUnpickler`.

    ``torch.save`` writes a zip archive whose directory prefix is the file stem (e.g.
    ``train_dataloader_dprank000/data.pkl``).  By opening the zip ourselves and running the
    pickle stream through :class:`_EnergonUnpickler` directly, we get the same ``find_class``
    security as ``weights_only=True`` without calling ``torch.load`` at all — so there is no
    ``weights_only=`` argument to audit.

    The ``persistent_load`` hook reconstructs each tensor storage from its raw blob; the dtype
    and layout are recovered by ``_rebuild_tensor_v2`` (already in the allowlist), which reads
    those fields from the pickle stream itself.
    """
    import torch

    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()

        # torch.save uses the file stem as the archive prefix: "<stem>/data.pkl".
        pkl_entry = next(n for n in names if n.endswith("/data.pkl"))
        prefix = pkl_entry[: -len("/data.pkl")]
        blob_prefix = f"{prefix}/data/"

        # Map storage key → raw bytes for every tensor storage blob.
        blob_map = {n[len(blob_prefix) :]: zf.read(n) for n in names if n.startswith(blob_prefix)}
        pkl_bytes = zf.read(pkl_entry)

    class _ZipLoader(_EnergonUnpickler):
        def find_class(self, module: str, name: str) -> type:
            # torch.save embeds the storage class (e.g. torch.LongStorage,
            # torch.storage.UntypedStorage) in the persistent_id tuple as a GLOBAL
            # opcode.  Storage classes hold raw bytes and are not executable; allow
            # them here without adding them to the shared _SAFE_MODULES allowlist.
            if module in ("torch", "torch.storage") and name.endswith("Storage"):
                return pickle.Unpickler.find_class(self, module, name)
            return super().find_class(module, name)

        def persistent_load(self, pid: tuple):
            # torch.save persistent_id format (zip path):
            #   ('storage', storage_cls, key, location, nbytes)
            _typename, storage_cls, key, _location, _nbytes = pid
            key = key.decode() if isinstance(key, bytes) else key
            raw = blob_map[key]
            untyped = torch.frombuffer(bytearray(raw), dtype=torch.uint8).untyped_storage()
            if map_location and map_location != _location:
                untyped = untyped.to(torch.device(map_location))
            # _rebuild_tensor_v2 reads storage.dtype to determine the tensor dtype.
            # UntypedStorage has no dtype attribute; wrap it with the typed storage class
            # (e.g. torch.LongStorage) that torch.save recorded in the persistent_id.
            if storage_cls is torch.storage.UntypedStorage:
                return untyped
            return storage_cls(wrap_storage=untyped)

    return _ZipLoader(io.BytesIO(pkl_bytes)).load()


def energon_torch_load(path: str, *, map_location: str = "cpu") -> object:
    """Load an Energon dataloader state ``.pt`` file through a restricted unpickler.

    Parses the torch zip format directly via :func:`_load_energon_zip` rather than delegating
    to ``torch.load``.  Security is enforced by :class:`_EnergonUnpickler`: any GLOBAL opcode
    whose ``(module, name)`` is not in the explicit allowlist raises ``pickle.UnpicklingError``,
    blocking ``__reduce__``-based code execution from attacker-controlled checkpoint files.

    ``torch.load(weights_only=True)`` is not used because PyTorch ≥ 2.13 restricts
    SETITEM/SETITEMS to exact ``dict``, ``OrderedDict``, and ``Counter`` types, rejecting dict
    subclasses such as Energon's ``FlexState`` — which is always present in real Energon
    checkpoints (``SavableDatasetState.dataset_state`` is typed ``FlexState``, not Optional).

    Args:
        path: Path to the ``.pt`` file written by
            :func:`~megatron.bridge.training.checkpointing.maybe_save_dataloader_state`.
        map_location: Device to map tensor storages to; defaults to ``"cpu"`` to avoid GPU
            allocation during restore.

    Returns:
        The deserialized object (a ``dict`` containing ``"dataloader_state_dict"``).
    """
    return _load_energon_zip(path, map_location=map_location)


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
