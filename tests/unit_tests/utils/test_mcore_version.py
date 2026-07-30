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

import re
from pathlib import Path

import pytest
from packaging.specifiers import SpecifierSet
from packaging.version import Version

from megatron.bridge.utils import mcore_version


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_window_is_derived_from_the_pin():
    pin = Version(mcore_version.MCORE_PIN_VERSION)

    assert mcore_version.min_supported_mcore_version() == Version(f"{pin.major}.{pin.minor - 1}.0")
    assert mcore_version.max_supported_mcore_version_exclusive() == Version(f"{pin.major}.{pin.minor + 1}.0")
    assert mcore_version.supported_mcore_specifier() == f">={pin.major}.{pin.minor - 1}.0,<{pin.major}.{pin.minor + 1}"


@pytest.mark.parametrize(
    "version,supported",
    [
        ("0.17.9", False),
        ("0.18.0", True),
        ("0.18.2", True),
        ("0.19.0", True),
        ("0.20.0", False),
    ],
)
def test_is_mcore_version_supported_boundaries(version, supported):
    # The parametrization is written against a 0.19.0 pin; skip if the pin has moved.
    if mcore_version.MCORE_PIN_VERSION != "0.19.0":
        pytest.skip("boundary cases are pinned to a 0.19.0 window")
    assert mcore_version.is_mcore_version_supported(Version(version)) is supported


def test_unknown_version_is_treated_as_supported():
    assert mcore_version.is_mcore_version_supported(None) is True


def test_check_mcore_version_can_raise(monkeypatch):
    monkeypatch.setattr(mcore_version, "get_mcore_version", lambda: Version("0.1.0"))

    assert mcore_version.check_mcore_version() is False
    with pytest.raises(RuntimeError, match="outside the version range"):
        mcore_version.check_mcore_version(raise_on_mismatch=True)


def test_pin_matches_megatron_lm_submodule():
    package_info = REPO_ROOT / "3rdparty" / "Megatron-LM" / "megatron" / "core" / "package_info.py"
    if not package_info.exists():
        pytest.skip("Megatron-LM submodule is not checked out")

    text = package_info.read_text()
    parts = {key: re.search(rf"^{key} = (\d+)", text, re.MULTILINE) for key in ("MAJOR", "MINOR", "PATCH")}
    if any(match is None for match in parts.values()):
        pytest.skip("could not parse the submodule version")

    submodule_version = Version(".".join(match.group(1) for match in parts.values()))
    pin = Version(mcore_version.MCORE_PIN_VERSION)
    assert (pin.major, pin.minor) == (submodule_version.major, submodule_version.minor), (
        "MCORE_PIN_VERSION is stale: update megatron.bridge.utils.mcore_version (and the "
        "megatron-core bound in pyproject.toml) when the Megatron-LM pin moves."
    )


def test_pyproject_bound_matches_declared_window():
    pyproject = (REPO_ROOT / "pyproject.toml").read_text()
    match = re.search(r'"megatron-core\[dev,mlm\](?P<spec>[^"]*)"', pyproject)
    assert match is not None, "megatron-core dependency not found in pyproject.toml"

    assert SpecifierSet(match.group("spec")) == SpecifierSet(mcore_version.supported_mcore_specifier())
