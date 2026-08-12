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
publicly installable Megatron-Core 0.18.x line. The lower bound is therefore
explicit, while the upper bound moves with the pin:

    public compatibility floor  <=  megatron-core  <  next minor of the pin

With the pin at 0.20.0 that resolves to ``>=0.18.0,<0.21``. When the pin moves,
:data:`MCORE_PIN_VERSION`, the compatibility floor, and the bound in
``pyproject.toml`` must be reviewed together. ``tests/unit_tests/compat/test_mcore_version.py``
fails if the pin drifts from the submodule or the package bound drifts from the
declared window.
"""

import logging
from importlib import metadata

from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version


logger = logging.getLogger(__name__)


#: Oldest publicly installable Megatron-Core release supported by Bridge.
MIN_SUPPORTED_MCORE_VERSION: str = "0.18.0"

#: Megatron-Core version of the commit pinned in ``3rdparty/Megatron-LM``.
#: Update this together with the submodule pin (enforced by unit test).
MCORE_PIN_VERSION: str = "0.20.0"


def _pin() -> Version:
    return Version(MCORE_PIN_VERSION)


def min_supported_mcore_version() -> Version:
    """Return the oldest publicly installable Megatron-Core version supported by Bridge."""
    return Version(MIN_SUPPORTED_MCORE_VERSION)


def max_supported_mcore_version_exclusive() -> Version:
    """Return the exclusive upper bound of the support window (next minor of the pin)."""
    pin = _pin()
    return Version(f"{pin.major}.{pin.minor + 1}.0")


def supported_mcore_specifier() -> str:
    """Return the support window as a PEP 440 specifier, e.g. ``>=0.18.0,<0.21``."""
    upper = max_supported_mcore_version_exclusive()
    return f">={min_supported_mcore_version()},<{upper.major}.{upper.minor}"


def get_mcore_version() -> Version | None:
    """Return the installed Megatron-Core version, or ``None`` if it cannot be parsed."""
    try:
        installed_version = metadata.version("megatron-core")
    except metadata.PackageNotFoundError:
        try:
            import megatron.core

            installed_version = megatron.core.__version__
        except (AttributeError, ImportError):
            return None
    try:
        return Version(installed_version)
    except InvalidVersion:
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
    return version in SpecifierSet(supported_mcore_specifier())


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
