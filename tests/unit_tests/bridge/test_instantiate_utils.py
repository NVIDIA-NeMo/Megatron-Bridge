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

import logging

import pytest

from megatron.bridge.utils.instantiate_utils import InstantiationException, InstantiationMode, instantiate


class DummyTarget:
    def __init__(self, a: int, b: int = 0) -> None:
        self.a = a
        self.b = b


class KwTarget:
    def __init__(self, **kwargs) -> None:  # noqa: D401 - simple holder
        self.kwargs = dict(kwargs)


def _target_qualname(obj) -> str:
    return f"{obj.__module__}.{obj.__qualname__}"


def test_drops_unexpected_kwargs_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    config = {
        "_target_": _target_qualname(DummyTarget),
        "a": 10,
        "foo": 123,  # unexpected key that should be dropped
    }

    with caplog.at_level(logging.WARNING):
        obj = instantiate(config)

    assert isinstance(obj, DummyTarget)
    assert obj.a == 10
    # 'foo' is dropped; 'b' remains default
    assert obj.b == 0

    # Ensure a warning was emitted mentioning the dropped key
    warnings = [rec.getMessage() for rec in caplog.records if rec.levelno == logging.WARNING]
    assert any("Dropping unexpected config keys" in m for m in warnings)
    assert any("foo" in m for m in warnings)


def test_allows_kwargs_when_target_accepts_var_kwargs(caplog: pytest.LogCaptureFixture) -> None:
    config = {
        "_target_": _target_qualname(KwTarget),
        "foo": 1,
        "bar": 2,
    }

    with caplog.at_level(logging.WARNING):
        obj = instantiate(config)

    assert isinstance(obj, KwTarget)
    assert obj.kwargs == {"foo": 1, "bar": 2}

    # No warning should be emitted for **kwargs targets
    warnings = [rec.getMessage() for rec in caplog.records if rec.levelno == logging.WARNING]
    assert not any("Dropping unexpected config keys" in m for m in warnings)


def test_raises_on_unexpected_kwargs_in_strict_mode() -> None:
    config = {
        "_target_": _target_qualname(DummyTarget),
        "a": 10,
        "foo": 123,
    }

    with pytest.raises(InstantiationException):
        instantiate(config, mode=InstantiationMode.STRICT)
