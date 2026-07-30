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

"""Declared Megatron-Core support window for pip-installed Megatron Bridge.

Bridge is developed against the Megatron-Core commit pinned in
``3rdparty/Megatron-LM``, but a pip-installed Bridge must also work with the
Megatron-Core releases users already have. The supported window is a *rule*
relative to the pin rather than a hardcoded pair of versions:

    previous minor release (N-1) of the pin  <=  megatron-core  <  next minor of the pin

With the pin at 0.19.0 that resolves to ``>=0.18.0,<0.20`` — the released 0.18.x
line through the current dev pin. When the pin moves to 0.20.x the window moves
with it; only :data:`MCORE_PIN_VERSION` needs updating, and
``tests/unit_tests/utils/test_mcore_version.py`` fails if it drifts from the
submodule or from the bound declared in ``pyproject.toml``.
"""

import logging

from packaging.version import Version


logger = logging.getLogger(__name__)


#: Megatron-Core version of the commit pinned in ``3rdparty/Megatron-LM``.
#: Update this together with the submodule pin (enforced by unit test).
MCORE_PIN_VERSION: str = "0.19.0"


def _pin() -> Version:
    return Version(MCORE_PIN_VERSION)


def min_supported_mcore_version() -> Version:
    """Return the oldest supported Megatron-Core version (N-1 minor of the pin)."""
    pin = _pin()
    if pin.minor == 0:
        raise ValueError(f"Cannot derive an N-1 minor release from pin {pin}")
    return Version(f"{pin.major}.{pin.minor - 1}.0")


def max_supported_mcore_version_exclusive() -> Version:
    """Return the exclusive upper bound of the support window (next minor of the pin)."""
    pin = _pin()
    return Version(f"{pin.major}.{pin.minor + 1}.0")


def supported_mcore_specifier() -> str:
    """Return the support window as a PEP 440 specifier, e.g. ``>=0.18.0,<0.20``."""
    upper = max_supported_mcore_version_exclusive()
    return f">={min_supported_mcore_version()},<{upper.major}.{upper.minor}"


def get_mcore_version() -> Version | None:
    """Return the installed Megatron-Core version, or ``None`` if it cannot be parsed."""
    try:
        import megatron.core

        return Version(Version(megatron.core.__version__).base_version)
    except Exception:  # noqa: BLE001 — version reporting must never break imports
        return None


def is_mcore_version_supported(version: Version | None = None) -> bool:
    """Check whether a Megatron-Core version falls inside the supported window.

    Args:
        version: Version to check. Defaults to the installed Megatron-Core.

    Returns:
        ``True`` when inside the window, or when the version cannot be determined.
    """
    version = version if version is not None else get_mcore_version()
    if version is None:
        return True
    return min_supported_mcore_version() <= version < max_supported_mcore_version_exclusive()


def check_mcore_version(*, raise_on_mismatch: bool = False) -> bool:
    """Warn (or raise) when the installed Megatron-Core is outside the supported window.

    Args:
        raise_on_mismatch: Raise ``RuntimeError`` instead of logging a warning.

    Returns:
        ``True`` when the installed version is supported.
    """
    version = get_mcore_version()
    if is_mcore_version_supported(version):
        return True

    message = (
        f"megatron-core {version} is outside the version range supported by this "
        f"Megatron Bridge release (megatron-core{supported_mcore_specifier()}). "
        "Conversion, checkpointing, or training may fail; install a supported "
        "megatron-core or upgrade Megatron Bridge."
    )
    if raise_on_mismatch:
        raise RuntimeError(message)
    logger.warning(message)
    return False
