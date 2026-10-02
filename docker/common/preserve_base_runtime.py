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

"""Opt-in Docker guard for a source-built, CUDA-aligned Rubin base runtime."""

import argparse
import importlib.metadata
import json
import logging
import os
from pathlib import Path


logger = logging.getLogger(__name__)
REQUIRED = ("torch", "transformer-engine", "nvidia-cudnn-frontend", "nvidia-cutlass-dsl")
OPTIONAL = (
    "transformer-engine-cu13",
    "transformer-engine-torch",
    "nvidia-cutlass-dsl-libs-base",
    "nvidia-cutlass-dsl-libs-cu13",
)


def enabled() -> bool:
    """Validate and return the opt-in setting inherited by downstream images."""
    value = os.environ.get("MBRIDGE_PRESERVE_BASE_RUNTIME", "False")
    if value not in {"True", "False"}:
        raise ValueError("MBRIDGE_PRESERVE_BASE_RUNTIME must be True or False")
    return value == "True"


def package_state() -> dict[str, dict[str, str]]:
    """Record the selected distribution versions and paths without importing CUDA."""
    result = {}
    for name in REQUIRED + OPTIONAL:
        try:
            dist = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            if name in REQUIRED:
                raise
            continue
        result[name] = {"version": dist.version, "location": str(dist.locate_file(""))}
    return result


def verify_runtime(path: Path, *, record: bool) -> None:
    """Record base metadata or fail if later dependency layers replaced it.

    This checks package identity, not GPU correctness; a runtime import and
    training test is still required. Disabled builds perform no metadata reads.
    """
    if not enabled():
        return
    current = package_state()
    if record:
        path.write_text(json.dumps(current, indent=2) + "\n")
    else:
        expected = json.loads(path.read_text())
        if current != expected:
            raise RuntimeError(f"Base runtime packages changed: expected={expected}, actual={current}")
    logger.info("Preserved base runtime packages: %s", current)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("record", "check"))
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    verify_runtime(args.path, record=args.mode == "record")
