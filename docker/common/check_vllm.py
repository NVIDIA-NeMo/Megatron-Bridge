# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Reject outdated vLLM installations without requiring a CUDA import."""

import logging
from importlib.metadata import PackageNotFoundError, version

from packaging.version import Version


def main() -> None:
    """Validate an installed vLLM, allowing images that omit it entirely."""
    try:
        installed = version("vllm")
    except PackageNotFoundError:
        logging.getLogger(__name__).info("vLLM is not installed; skipping version check")
        return
    if Version(installed) < Version("0.30.0"):
        raise ValueError(
            f"vLLM {installed} is unsupported: rebuild with VLLM_VERSION=v0.30.0 or newer, "
            "or select a base image with a patched vLLM installation."
        )


if __name__ == "__main__":
    main()
