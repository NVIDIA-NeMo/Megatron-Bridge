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
import pickle
from collections import OrderedDict

import pytest

from megatron.bridge.utils.safe_pickle import safe_pickle_load, safe_pickle_loads


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
