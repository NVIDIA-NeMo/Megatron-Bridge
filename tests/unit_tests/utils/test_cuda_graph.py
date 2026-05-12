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

"""Unit tests for utils/cuda_graph read-side helpers.

The cuda_graph module exposes a handful of "read" functions
(``cuda_graph_module_names``, ``has_cuda_graph_module``,
``is_full_iteration_cuda_graph``, ``uses_local_cuda_graph_manager``)
that are pure attribute reads + string-set operations on a config
object. They are independent of any real ``CudaGraphScope`` /
``CudaGraphModule`` enum values, so they're easy to verify with
``SimpleNamespace`` fakes.

The "write" helpers (``set_cuda_graph_modules`` and friends) call
``CudaGraphScope[name]`` and ``CudaGraphModule[name]`` which require
real enum members — they're better covered by an integration test
and are not included here.
"""

from types import SimpleNamespace

from megatron.bridge.utils.cuda_graph import (
    cuda_graph_module_names,
    has_cuda_graph_module,
    is_full_iteration_cuda_graph,
    uses_local_cuda_graph_manager,
)


def _cfg(**overrides):
    """Build a config namespace with the cuda-graph attributes the helpers read."""
    base = {
        "cuda_graph_modules": None,
        "cuda_graph_scope": None,
        "cuda_graph_impl": "none",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class TestCudaGraphModuleNames:
    def test_empty_when_neither_set(self):
        assert cuda_graph_module_names(_cfg()) == []

    def test_returns_modules_list_when_modules_is_set(self):
        # New-MCore API: read from cuda_graph_modules and ignore cuda_graph_scope.
        cfg = _cfg(cuda_graph_modules=["attention", "mlp"], cuda_graph_scope="full")
        assert cuda_graph_module_names(cfg) == ["attention", "mlp"]

    def test_returns_modules_list_for_string_input(self):
        cfg = _cfg(cuda_graph_modules="attention,mlp")
        assert cuda_graph_module_names(cfg) == ["attention", "mlp"]

    def test_single_string_value_for_modules(self):
        cfg = _cfg(cuda_graph_modules="attention")
        assert cuda_graph_module_names(cfg) == ["attention"]

    def test_falls_back_to_scope_when_modules_unset(self):
        # Legacy API path.
        cfg = _cfg(cuda_graph_modules=None, cuda_graph_scope="attention,mlp")
        assert cuda_graph_module_names(cfg) == ["attention", "mlp"]

    def test_filters_full_iteration_from_scope(self):
        # cuda_graph_module_names *excludes* "full_iteration" and
        # "full_iteration_inference" because those are full-iter markers,
        # not per-layer modules. Other scope names pass through.
        cfg = _cfg(cuda_graph_scope="attention,full_iteration,full_iteration_inference,mlp")
        assert cuda_graph_module_names(cfg) == ["attention", "mlp"]

    def test_scope_full_string_returns_empty(self):
        # "full" is the documented "wildcard" string that maps to "no
        # explicit per-layer list" (see _as_list).
        cfg = _cfg(cuda_graph_scope="full")
        assert cuda_graph_module_names(cfg) == []


class TestHasCudaGraphModule:
    def test_false_when_neither_set(self):
        assert has_cuda_graph_module(_cfg(), "attention") is False

    def test_true_when_present_in_modules(self):
        cfg = _cfg(cuda_graph_modules=["attention", "mlp"])
        assert has_cuda_graph_module(cfg, "attention") is True
        assert has_cuda_graph_module(cfg, "mlp") is True

    def test_true_when_present_in_legacy_scope(self):
        cfg = _cfg(cuda_graph_scope=["attention"])
        assert has_cuda_graph_module(cfg, "attention") is True

    def test_module_argument_accepts_enum_like_object(self):
        # The helper extracts .name when available, so any object with
        # a .name attribute should be looked up by that name. Build a
        # tiny stand-in enum-member instead of importing the real enum.
        cfg = _cfg(cuda_graph_modules=["attention"])
        fake_enum_member = SimpleNamespace(name="attention")
        assert has_cuda_graph_module(cfg, fake_enum_member) is True

    def test_false_for_unknown_module(self):
        cfg = _cfg(cuda_graph_modules=["attention"])
        assert has_cuda_graph_module(cfg, "mlp") is False


class TestIsFullIterationCudaGraph:
    def test_false_when_impl_is_none(self):
        assert is_full_iteration_cuda_graph(_cfg(cuda_graph_impl="none")) is False

    def test_true_when_impl_is_full_iteration(self):
        # New-MCore API: cuda_graph_impl = "full_iteration".
        assert is_full_iteration_cuda_graph(_cfg(cuda_graph_impl="full_iteration")) is True

    def test_true_for_legacy_local_with_full_iteration_scope(self):
        # Old-MCore API: cuda_graph_impl = "local" + scope contains "full_iteration".
        cfg = _cfg(cuda_graph_impl="local", cuda_graph_scope=["full_iteration"])
        assert is_full_iteration_cuda_graph(cfg) is True

    def test_false_for_local_without_full_iteration_scope(self):
        cfg = _cfg(cuda_graph_impl="local", cuda_graph_scope=["attention", "mlp"])
        assert is_full_iteration_cuda_graph(cfg) is False

    def test_false_for_unknown_impl_value(self):
        # Anything that's not "full_iteration" or "local" -> not full-iter.
        assert is_full_iteration_cuda_graph(_cfg(cuda_graph_impl="something_else")) is False

    def test_default_impl_is_none(self):
        # getattr default applies when the attribute is missing entirely.
        cfg = SimpleNamespace()  # no cuda_graph_impl attribute at all
        assert is_full_iteration_cuda_graph(cfg) is False


class TestUsesLocalCudaGraphManager:
    def test_false_when_impl_is_none(self):
        assert uses_local_cuda_graph_manager(_cfg()) is False

    def test_true_for_local_impl_without_full_iteration(self):
        cfg = _cfg(cuda_graph_impl="local", cuda_graph_scope=["attention"])
        assert uses_local_cuda_graph_manager(cfg) is True

    def test_false_for_local_impl_with_full_iteration_scope(self):
        # local + full_iteration scope means full-iteration capture, not a
        # local manager — uses_local_cuda_graph_manager should return False.
        cfg = _cfg(cuda_graph_impl="local", cuda_graph_scope=["full_iteration"])
        assert uses_local_cuda_graph_manager(cfg) is False

    def test_false_for_full_iteration_impl(self):
        # New API: cuda_graph_impl="full_iteration" -> not a local manager.
        cfg = _cfg(cuda_graph_impl="full_iteration")
        assert uses_local_cuda_graph_manager(cfg) is False
