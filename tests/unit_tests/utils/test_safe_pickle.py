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

import codecs
import io
import os
import pickle
import subprocess
from collections import OrderedDict
from enum import Enum

import numpy as np
import pytest
import torch
from megatron.energon.savable_loader import SavableDataLoaderState
from megatron.energon.state import FlexState

from megatron.bridge.utils.safe_pickle import (
    energon_torch_load,
    safe_load_npy,
    safe_pickle_load,
    safe_pickle_loads,
    safe_torch_tensor_pickle_loads,
)


class _BucketKey(Enum):
    IMAGE = "image"


class _RejectingValueMap(dict):
    def __getitem__(self, key):
        raise AssertionError("Enum reconstruction consulted _value2member_map_")


class _BucketKeyWithRejectingValueMap(Enum):
    IMAGE = "image"


_BucketKeyWithRejectingValueMap._value2member_map_ = _RejectingValueMap(
    _BucketKeyWithRejectingValueMap._value2member_map_
)


_ENUM_HOOK_CALLED = {"call": False, "setstate": False}


class _BucketKeyWithCustomCall(Enum):
    IMAGE = "image"

    def __call__(self):
        _ENUM_HOOK_CALLED["call"] = True


class _BucketKeyWithCustomSetstate(Enum):
    IMAGE = "image"

    def __setstate__(self, _state):
        _ENUM_HOOK_CALLED["setstate"] = True


class _EnumMemberCallPayload:
    def __reduce__(self):
        return _BucketKeyWithCustomCall.IMAGE, ()


class _EnumMemberBuildPayload:
    def __reduce__(self):
        return _BucketKeyWithCustomSetstate, ("image",), {"checkpoint_selected_state": True}


class _BucketKeyWithCustomMissing(Enum):
    IMAGE = "image"

    @classmethod
    def _missing_(cls, _value):
        return cls.IMAGE


class _BucketKeyWithCustomHash(Enum):
    IMAGE = "image"

    def __hash__(self):
        return hash(self.value)


class _BucketKeyWithCustomEq(Enum):
    IMAGE = "image"

    def __eq__(self, other):
        return self is other

    __hash__ = Enum.__hash__


class _BucketKeyWithCustomGetattribute(Enum):
    IMAGE = "image"

    def __getattribute__(self, name):
        return object.__getattribute__(self, name)


class _CustomEnumMemberName:
    def __hash__(self):
        return 1


class _BucketKeyWithCustomMemberName(Enum):
    IMAGE = "image"

    def __init__(self, _value):
        object.__setattr__(self, "_name_", _CustomEnumMemberName())


class _BucketKeyWithCustomRepr(Enum):
    IMAGE = "image"

    def __repr__(self):
        return "image"


class _EnumReduceWithInvalidValue:
    def __reduce__(self):
        return _BucketKey, (_BucketKeyWithCustomRepr.IMAGE,)


class _HashDescriptor:
    def __get__(self, instance, owner):
        if instance is None:
            return Enum.__hash__
        return Enum.__hash__.__get__(instance, owner)


class _BucketKeyWithHashDescriptor(Enum):
    IMAGE = "image"

    __hash__ = _HashDescriptor()


class _CustomContainerMeta(type):
    def __getattribute__(cls, name):
        return super().__getattribute__(name)


class _EnumContainerWithCustomMeta(metaclass=_CustomContainerMeta):
    class BucketKey(Enum):
        IMAGE = "image"


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

    @pytest.mark.parametrize("protocol", [0, 1, 2])
    def test_pickled_video_frames_from_legacy_protocols(self, protocol):
        """Encoded video frames remain loadable when shards use older pickle protocols."""
        frames = [[b"encoded JPEG frame"]]

        result = safe_pickle_loads(pickle.dumps(frames, protocol=protocol))

        assert result == frames

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

    def test_rejects_application_codec_for_legacy_bytes(self):
        """Legacy bytes reconstruction cannot select an application codec hook."""
        hook_called = False

        def search(encoding):
            nonlocal hook_called
            if encoding == "applicationcodec":
                hook_called = True
                return codecs.CodecInfo(
                    name=encoding,
                    encode=lambda value, errors="strict": (b"encoded", len(value)),
                    decode=None,
                )
            return None

        class Payload:
            def __reduce__(self):
                return codecs.encode, ("attacker selected", "applicationcodec")

        codecs.register(search)
        try:
            with pytest.raises(pickle.UnpicklingError, match="invalid legacy bytes payload"):
                safe_pickle_loads(pickle.dumps(Payload()))
        finally:
            codecs.unregister(search)

        assert not hook_called


class TestSafeTorchTensorPickle:
    """Raw WebDataset tensor pickles allow data but reject executable globals."""

    def test_plain_tensor_container_round_trip(self):
        value = {
            "embedding": torch.arange(6, dtype=torch.float32).reshape(2, 3),
            "mask": torch.tensor([True, False]),
        }

        restored = safe_torch_tensor_pickle_loads(pickle.dumps(value))

        assert restored.keys() == value.keys()
        assert torch.equal(restored["embedding"], value["embedding"])
        assert torch.equal(restored["mask"], value["mask"])

    def test_rejects_reduce_payload_without_executing_it(self, tmp_path):
        marker = tmp_path / "pickle-executed"

        class Payload:
            def __reduce__(self):
                return os.system, (f"touch {marker}",)

        with pytest.raises(pickle.UnpicklingError, match="Restricted unpickler refused"):
            safe_torch_tensor_pickle_loads(pickle.dumps(Payload()))

        assert not marker.exists()


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


def test_energon_group_bucket_enum_key_round_trip(tmp_path):
    """Energon's supported Hashable grouping keys survive dataloader checkpoint restore."""
    state = SavableDataLoaderState(
        worker_states=[FlexState(buckets={_BucketKey.IMAGE: {"batch_size": 2}})],
        next_worker_id=0,
        micro_batch_size=2,
    )
    path = tmp_path / "dataloader-state.pt"
    torch.save({"dataloader_state_dict": state}, path)

    restored = energon_torch_load(str(path))["dataloader_state_dict"]

    bucket_key = next(iter(restored.worker_states[0]["buckets"]))
    assert bucket_key is _BucketKey.IMAGE


def test_energon_group_bucket_frozenset_key_round_trip(tmp_path):
    """Energon can reuse a restored partial bucket keyed by any supported Hashable."""
    bucket_key = frozenset({"image"})
    state = SavableDataLoaderState(
        worker_states=[FlexState(buckets={bucket_key: {"batch_size": 2}})],
        next_worker_id=0,
        micro_batch_size=2,
    )
    path = tmp_path / "dataloader-state.pt"
    torch.save({"dataloader_state_dict": state}, path)

    restored = energon_torch_load(str(path))["dataloader_state_dict"]

    restored_buckets = restored.worker_states[0]["buckets"]
    assert restored_buckets[bucket_key] == {"batch_size": 2}


def test_energon_group_bucket_enum_does_not_use_application_value_map(tmp_path):
    """Enum reconstruction does not execute a mutable application lookup mapping."""
    state = SavableDataLoaderState(
        worker_states=[FlexState(buckets={_BucketKeyWithRejectingValueMap.IMAGE: {"batch_size": 2}})],
        next_worker_id=0,
        micro_batch_size=2,
    )
    path = tmp_path / "dataloader-state.pt"
    torch.save({"dataloader_state_dict": state}, path)

    restored = energon_torch_load(str(path))["dataloader_state_dict"]

    bucket_key = next(iter(restored.worker_states[0]["buckets"]))
    assert bucket_key is _BucketKeyWithRejectingValueMap.IMAGE


def test_energon_group_bucket_enum_restores_immutable_containers_and_aliases(tmp_path):
    """Postprocessing restores Enum tokens inside immutable state without breaking aliases."""
    shared = (_BucketKey.IMAGE, "shared")
    state = SavableDataLoaderState(
        worker_states=[FlexState(payload=[shared, shared])],
        next_worker_id=0,
        micro_batch_size=2,
    )
    path = tmp_path / "dataloader-state.pt"
    torch.save({"dataloader_state_dict": state}, path)

    restored = energon_torch_load(str(path))["dataloader_state_dict"]

    first, second = restored.worker_states[0]["payload"]
    assert first is second
    assert first == (_BucketKey.IMAGE, "shared")


def test_energon_group_bucket_enum_rejects_cyclic_immutable_state(tmp_path):
    """Token traversal fails closed instead of silently breaking an immutable cycle."""
    cyclic_list = []
    cyclic_tuple = (cyclic_list,)
    cyclic_list.append(cyclic_tuple)
    state = SavableDataLoaderState(
        worker_states=[FlexState(payload=cyclic_tuple)],
        next_worker_id=0,
        micro_batch_size=2,
    )
    path = tmp_path / "dataloader-state.pt"
    torch.save({"dataloader_state_dict": state}, path)

    with pytest.raises(pickle.UnpicklingError, match="cyclic immutable container"):
        energon_torch_load(str(path))


@pytest.mark.parametrize(
    ("payload", "hook"),
    [
        (_EnumMemberCallPayload(), "call"),
        (_EnumMemberBuildPayload(), "setstate"),
    ],
)
def test_energon_group_bucket_enum_rejects_post_resolution_opcodes(tmp_path, payload, hook):
    """Later pickle opcodes cannot call or mutate a resolved application Enum member."""
    _ENUM_HOOK_CALLED[hook] = False
    state = SavableDataLoaderState(
        worker_states=[FlexState(payload=payload)],
        next_worker_id=0,
        micro_batch_size=2,
    )
    path = tmp_path / "dataloader-state.pt"
    torch.save({"dataloader_state_dict": state}, path)

    with pytest.raises(pickle.UnpicklingError, match="Restricted unpickler refused"):
        energon_torch_load(str(path))

    assert not _ENUM_HOOK_CALLED[hook]


def test_energon_group_bucket_enum_with_custom_missing_is_rejected(tmp_path):
    """Enum hooks that could execute checkpoint-selected behavior remain outside the allowlist."""
    state = SavableDataLoaderState(
        worker_states=[FlexState(buckets={_BucketKeyWithCustomMissing.IMAGE: {"batch_size": 2}})],
        next_worker_id=0,
        micro_batch_size=2,
    )
    path = tmp_path / "dataloader-state.pt"
    torch.save({"dataloader_state_dict": state}, path)

    with pytest.raises(pickle.UnpicklingError, match="Restricted unpickler refused"):
        energon_torch_load(str(path))


@pytest.mark.parametrize(
    "bucket_key",
    [
        _BucketKeyWithCustomHash.IMAGE,
        _BucketKeyWithCustomEq.IMAGE,
        _BucketKeyWithCustomGetattribute.IMAGE,
        _BucketKeyWithCustomMemberName.IMAGE,
        _BucketKeyWithHashDescriptor.IMAGE,
        _EnumContainerWithCustomMeta.BucketKey.IMAGE,
    ],
)
def test_energon_group_bucket_enum_with_custom_dict_hooks_is_rejected(tmp_path, bucket_key):
    """Enum hooks invoked by dictionary reconstruction remain outside the allowlist."""
    state = SavableDataLoaderState(
        worker_states=[FlexState(buckets={bucket_key: {"batch_size": 2}})],
        next_worker_id=0,
        micro_batch_size=2,
    )
    path = tmp_path / "dataloader-state.pt"
    torch.save({"dataloader_state_dict": state}, path)

    with pytest.raises(pickle.UnpicklingError, match="Restricted unpickler refused"):
        energon_torch_load(str(path))


def test_energon_group_bucket_enum_with_custom_repr_is_rejected_before_reduce(tmp_path):
    """A crafted REDUCE cannot invoke an application Enum's custom representation hook."""
    state = SavableDataLoaderState(
        worker_states=[FlexState(payload=_EnumReduceWithInvalidValue())],
        next_worker_id=0,
        micro_batch_size=2,
    )
    path = tmp_path / "dataloader-state.pt"
    torch.save({"dataloader_state_dict": state}, path)

    with pytest.raises(pickle.UnpicklingError, match="Restricted unpickler refused"):
        energon_torch_load(str(path))


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
        """The exact dict-of-lists shape produced by offline packing should survive a save/load cycle."""
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
