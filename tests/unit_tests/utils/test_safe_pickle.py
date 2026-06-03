#!/usr/bin/env python3
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

"""Tests for safe_pickle module."""

import io
import os
import pickle
import subprocess
from collections import OrderedDict

import numpy as np
import pytest

from megatron.bridge.utils.safe_pickle import safe_load_npy, safe_pickle_load, safe_pickle_loads


class TestSafePickleRoundTrip:
    """Verify that safe types round-trip correctly."""

    @pytest.mark.parametrize(
        "obj",
        [
            [1, 2, 3],
            {"key": "value", "num": 42},
            (1, "a", 3.14),
            {1, 2, 3},
            frozenset([4, 5, 6]),
            b"binary data",
            bytearray(b"mutable bytes"),
            "hello",
            42,
            3.14,
            True,
            complex(1, 2),
            slice(1, 10, 2),
            range(5),
            None,
            OrderedDict([("a", 1), ("b", 2)]),
        ],
        ids=lambda x: type(x).__name__,
    )
    def test_allowed_types(self, obj):
        data = pickle.dumps(obj)
        result = safe_pickle_loads(data)
        assert result == obj

    def test_nested_structures(self):
        obj = {"list": [1, 2, None], "nested": {"a": (True, 3.14)}, "bytes": b"\x00\x01"}
        data = pickle.dumps(obj)
        assert safe_pickle_loads(data) == obj

    def test_safe_pickle_load_from_file(self):
        obj = {"key": [1, 2, 3]}
        buf = io.BytesIO()
        pickle.dump(obj, buf)
        buf.seek(0)
        assert safe_pickle_load(buf) == obj


class TestSafePickleRejectsUnsafe:
    """Verify that disallowed types are rejected."""

    def test_rejects_eval(self):
        data = pickle.dumps(eval)  # noqa: S301
        with pytest.raises(pickle.UnpicklingError, match="Restricted unpickler refused"):
            safe_pickle_loads(data)

    def test_rejects_os_system(self):
        import os

        data = pickle.dumps(os.system)
        with pytest.raises(pickle.UnpicklingError, match="Restricted unpickler refused"):
            safe_pickle_loads(data)

    def test_rejects_subprocess(self):
        import subprocess

        data = pickle.dumps(subprocess.Popen)
        with pytest.raises(pickle.UnpicklingError, match="Restricted unpickler refused"):
            safe_pickle_loads(data)

    def test_rejects_builtins_type(self):
        # type(None) pickles as builtins.type — should be rejected
        data = pickle.dumps(type(None))
        with pytest.raises(pickle.UnpicklingError, match="Restricted unpickler refused"):
            safe_pickle_loads(data)


class TestAllowlistImmutability:
    """Verify the allowlist cannot be mutated at runtime."""

    def test_cannot_mutate_modules(self):
        from megatron.bridge.utils.safe_pickle import _RestrictedUnpickler

        with pytest.raises(TypeError):
            _RestrictedUnpickler._SAFE_MODULES["os"] = frozenset({"system"})

    def test_cannot_mutate_allowed_names(self):
        from megatron.bridge.utils.safe_pickle import _RestrictedUnpickler

        with pytest.raises((TypeError, AttributeError)):
            _RestrictedUnpickler._SAFE_MODULES["builtins"].add("eval")


# ---------------------------------------------------------------------------
# safe_load_npy — restricted loading of .npy files
# ---------------------------------------------------------------------------


def _make_npy_bytes(obj) -> bytes:
    """Save *obj* via ``np.save`` and return the raw .npy bytes."""
    buf = io.BytesIO()
    np.save(buf, obj)
    return buf.getvalue()


def _make_malicious_npy_bytes(payload) -> bytes:
    """Build a .npy file whose object-array pickle payload contains *payload*.

    Creates a valid .npy header for an object array, then replaces the pickle
    payload with one that will attempt to instantiate *payload* on load.
    """
    buf = io.BytesIO()
    np.save(buf, np.array([None], dtype=object))
    buf.seek(0)

    version = np.lib.format.read_magic(buf)
    reader = np.lib.format.read_array_header_1_0 if version[0] == 1 else np.lib.format.read_array_header_2_0
    reader(buf)
    header_len = buf.tell()

    buf.seek(0)
    header = buf.read(header_len)
    return header + pickle.dumps(payload)


class TestSafeLoadNpyNumericArrays:
    """Numeric arrays use the fast allow_pickle=False path and should round-trip unchanged."""

    def test_int_array(self):
        """A plain int64 array should load without invoking the restricted unpickler."""
        arr = np.array([1, 2, 3], dtype=np.int64)
        result = safe_load_npy(_make_npy_bytes(arr))
        np.testing.assert_array_equal(result, arr)

    def test_float_array(self):
        """A multi-dimensional float32 array should preserve shape and values."""
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        result = safe_load_npy(_make_npy_bytes(arr))
        np.testing.assert_array_equal(result, arr)


class TestSafeLoadNpyObjectArrays:
    """Object arrays (packed SFT datasets) should load through the restricted unpickler."""

    def test_packed_dataset_round_trip(self):
        """The exact dict-of-lists shape produced by packed_sequence.py should survive a save/load cycle."""
        output_data = [
            {"input_ids": [1, 2, 3], "loss_mask": [True, False, True], "seq_start_id": [0, 3]},
            {"input_ids": [4, 5, 6, 7], "loss_mask": [True, True, True, False], "seq_start_id": [0, 4]},
        ]
        result = safe_load_npy(_make_npy_bytes(output_data))
        assert len(result) == 2
        assert result[0]["input_ids"] == [1, 2, 3]
        assert result[1]["loss_mask"] == [True, True, True, False]

    def test_empty_packed_dataset(self):
        """An empty list saved as an object array should load as a zero-length array."""
        result = safe_load_npy(_make_npy_bytes([]))
        assert len(result) == 0


class TestSafeLoadNpyRejectsMalicious:
    """Malicious .npy files that embed dangerous pickle payloads must be rejected."""

    def test_rejects_os_system(self):
        """A pickle referencing os.system should be blocked by the numpy restricted unpickler."""
        data = _make_malicious_npy_bytes(os.system)
        with pytest.raises(pickle.UnpicklingError, match="Restricted unpickler refused"):
            safe_load_npy(data)

    def test_rejects_subprocess_popen(self):
        """A pickle referencing subprocess.Popen should be blocked."""
        data = _make_malicious_npy_bytes(subprocess.Popen)
        with pytest.raises(pickle.UnpicklingError, match="Restricted unpickler refused"):
            safe_load_npy(data)

    def test_rejects_eval(self):
        """A pickle referencing builtins.eval should be blocked."""
        data = _make_malicious_npy_bytes(eval)
        with pytest.raises(pickle.UnpicklingError, match="Restricted unpickler refused"):
            safe_load_npy(data)

    def test_rejects_reduce_exploit(self):
        """A classic __reduce__-based RCE payload (os.system via a crafted class) should be blocked."""

        class Exploit:
            def __reduce__(self):
                return (os.system, ("echo pwned",))

        data = _make_malicious_npy_bytes(Exploit())
        with pytest.raises(pickle.UnpicklingError, match="Restricted unpickler refused"):
            safe_load_npy(data)


class TestNumpyAllowlistImmutability:
    """The numpy restricted unpickler allowlist must be immutable to prevent runtime tampering."""

    def test_cannot_mutate_modules(self):
        """Adding a new module to the allowlist at runtime should raise TypeError."""
        from megatron.bridge.utils.safe_pickle import _NumpyRestrictedUnpickler

        with pytest.raises(TypeError):
            _NumpyRestrictedUnpickler._SAFE_MODULES["os"] = frozenset({"system"})

    def test_cannot_mutate_allowed_names(self):
        """Adding a name to an existing module's frozenset should raise TypeError or AttributeError."""
        from megatron.bridge.utils.safe_pickle import _NumpyRestrictedUnpickler

        with pytest.raises((TypeError, AttributeError)):
            _NumpyRestrictedUnpickler._SAFE_MODULES["builtins"].add("eval")
