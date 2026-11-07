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

import importlib
import importlib.abc
import importlib.util
import os
import sys

import pytest

import megatron.bridge


_KEY = "TRITON_CACHE_AUTOTUNING"


class _ModeloptSettingTritonEnv(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Stand-in for modelopt whose import sets TRITON_CACHE_AUTOTUNING=1 like vllm does."""

    def find_spec(self, fullname, path, target=None):
        if fullname in ("modelopt", "modelopt.torch"):
            return importlib.util.spec_from_loader(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        if module.__name__ == "modelopt.torch":
            os.environ.setdefault(_KEY, "1")


@pytest.fixture
def reload_bridge(monkeypatch):
    monkeypatch.delitem(sys.modules, "modelopt", raising=False)
    monkeypatch.delitem(sys.modules, "modelopt.torch", raising=False)
    monkeypatch.setattr(sys, "meta_path", [_ModeloptSettingTritonEnv(), *sys.meta_path])
    return lambda: importlib.reload(megatron.bridge)


def test_import_does_not_leak_triton_cache_autotuning(monkeypatch, reload_bridge):
    monkeypatch.delenv(_KEY, raising=False)
    reload_bridge()
    assert _KEY not in os.environ


def test_import_keeps_user_set_triton_cache_autotuning(monkeypatch, reload_bridge):
    monkeypatch.setenv(_KEY, "1")
    reload_bridge()
    assert os.environ[_KEY] == "1"
