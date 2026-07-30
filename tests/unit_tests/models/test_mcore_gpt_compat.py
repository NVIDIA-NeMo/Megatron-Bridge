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

"""The vendored ``megatron.training.models.gpt`` fallback must track upstream.

The fallback is only exercised on Megatron-Core 0.18.x, where the upstream module
does not exist yet, so these checks compare it against the pinned Megatron-Core
whenever that module is importable.
"""

import dataclasses
import inspect

import pytest

from megatron.bridge.models.gpt import mcore_gpt_compat


upstream = pytest.importorskip(
    "megatron.training.models.gpt",
    reason="Megatron-Core predates megatron.training.models.gpt; the vendored copy is in use",
)


def _public_names(module) -> set[str]:
    """Public classes and functions *defined* in the module (imports excluded)."""
    return {
        name
        for name, obj in vars(module).items()
        if not name.startswith("_") and getattr(obj, "__module__", None) == module.__name__
    }


def test_vendored_module_exposes_the_symbols_bridge_imports():
    for name in ("GPTModelConfig", "GPTModelBuilder", "mtp_block_spec"):
        assert hasattr(mcore_gpt_compat, name)


def test_public_surface_matches_upstream():
    missing = _public_names(upstream) - _public_names(mcore_gpt_compat)
    assert not missing, f"vendored copy is missing upstream symbols: {sorted(missing)}"


def test_config_fields_match_upstream():
    vendored = {f.name for f in dataclasses.fields(mcore_gpt_compat.GPTModelConfig)}
    current = {f.name for f in dataclasses.fields(upstream.GPTModelConfig)}
    assert vendored == current, (
        "GPTModelConfig drifted upstream; refresh "
        "src/megatron/bridge/models/gpt/mcore_gpt_compat.py from megatron/training/models/gpt.py"
    )


def test_builder_methods_match_upstream():
    def methods(cls):
        return {name for name, _ in inspect.getmembers(cls, inspect.isfunction) if not name.startswith("_")}

    assert methods(mcore_gpt_compat.GPTModelBuilder) == methods(upstream.GPTModelBuilder)


def test_mtp_block_spec_signature_matches_upstream():
    assert inspect.signature(mcore_gpt_compat.mtp_block_spec) == inspect.signature(upstream.mtp_block_spec)
