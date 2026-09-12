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

"""Root test configuration.

Defines the *conversion-only stack*: the subset of tests that exercise
HuggingFace <-> Megatron weight mapping, model providers, and roundtrip
conversion, with no dependency on the Megatron-Bridge training loop.

Membership is a path rule, applied automatically at collection time, so a
newly added ``*_bridge.py`` / ``*_provider.py`` / ``*_conversion.py`` test is
included in ``-m conversion`` without anyone having to remember to tag it.
This is the contract NeMo-RL relies on: RL depends on MCore directly and only
needs this conversion subset of Megatron-Bridge to keep working across MCore
bumps. Run it with ``pytest -m conversion``.

To change what counts as "conversion", edit ``_is_conversion_test`` below --
that is the single source of truth.
"""

from __future__ import annotations

from pathlib import Path

import pytest


# Explicit core files under tests/unit_tests/models/ that belong to the
# conversion stack but are not covered by the *_bridge / *_provider suffixes.
_UNIT_MODEL_CORE_FILES = frozenset(
    {
        "test_param_mapping.py",
        "test_mapping_registry.py",
        "test_conversion_utils.py",
        "test_conversion_unwrap_utils.py",
        "test_gpt_provider.py",
        "test_model_provider_mixin.py",
    }
)


def _is_conversion_test(path: Path) -> bool:
    """Return True if a collected test file is part of the conversion-only stack."""
    posix = path.as_posix()
    name = path.name

    # CPU unit tier: per-family bridge/provider mapping + core mapping helpers.
    if "/tests/unit_tests/models/" in posix:
        if name.endswith("_bridge.py") or name.endswith("_provider.py"):
            return True
        if name in _UNIT_MODEL_CORE_FILES:
            return True

    # GPU functional tier: generic converter + per-family HF<->Megatron roundtrip.
    if "/tests/functional_tests/test_groups/converter/" in posix:
        return True
    if "/tests/functional_tests/test_groups/models/" in posix and name.endswith("_conversion.py"):
        return True

    return False


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-apply the ``conversion`` marker to every conversion-stack test."""
    marker = pytest.mark.conversion
    for item in items:
        try:
            path = Path(item.path)
        except (AttributeError, TypeError):
            path = Path(str(item.fspath))
        if _is_conversion_test(path):
            item.add_marker(marker)
