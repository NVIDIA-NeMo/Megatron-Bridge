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

"""Validate functional-test identifiers and metadata before emitting CI matrices."""

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import TypedDict


logger = logging.getLogger(__name__)
SCRIPT_PATTERN = re.compile(r"L[0-2]_[A-Za-z0-9_.-]+")
# Keep timeout values within the range accepted by the test-template action.
MAX_TIMEOUT = 2147483647


class MatrixEntry(TypedDict):
    """Validated input for one functional-test job."""

    script: str
    timeout: int
    runner: str


class Matrix(TypedDict):
    """GitHub Actions matrix with explicitly included jobs."""

    include: list[MatrixEntry]


def _read_metadata(path: Path) -> dict[str, str]:
    metadata: dict[str, str] = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            for key in ("CI_TIMEOUT", "GPU_COUNT"):
                prefix = f"# {key}="
                if line.startswith(prefix) and key not in metadata:
                    metadata[key] = line[len(prefix) :].rstrip("\r\n")
    return metadata


def _entry(path: Path, *, hardware: str, runner_prefix: str) -> MatrixEntry:
    script = path.stem
    if not SCRIPT_PATTERN.fullmatch(script):
        raise ValueError(f"Invalid functional-test script identifier: {path.name!r}")

    metadata = _read_metadata(path)
    timeout = metadata.get("CI_TIMEOUT", "30")
    if not re.fullmatch(r"[1-9][0-9]{0,9}", timeout) or int(timeout) > MAX_TIMEOUT:
        raise ValueError(f"Invalid CI_TIMEOUT in {path.name!r}: expected an integer from 1 to {MAX_TIMEOUT}")

    runner = runner_prefix
    if "GPU_COUNT" in metadata:
        gpu_count = metadata["GPU_COUNT"]
        gpu_pattern = r"[1-9][0-9]*" if hardware == "h100" else r"x[1-9][0-9]*"
        if not re.fullmatch(gpu_pattern, gpu_count):
            raise ValueError(f"Invalid GPU_COUNT in {path.name!r} for {hardware}")
        if hardware == "h100":
            runner = re.sub(r"gpu-x[0-9]+", f"gpu-x{gpu_count}", runner_prefix, count=1)
        else:
            runner = f"nemo-ci-gcp-gpu-{gpu_count}"

    return {"script": script, "timeout": int(timeout), "runner": runner}


def build_matrices(launch_root: Path, *, hardware: str, runner_prefix: str) -> dict[str, Matrix]:
    """Build all active and flaky matrices, rejecting invalid discovered inputs.

    Args:
        launch_root: Root directory containing hardware-specific launch scripts.
        hardware: Either ``h100`` or ``gb200``.
        runner_prefix: Default runner label, before any GPU_COUNT override.

    Returns:
        Matrices keyed by the existing workflow output names.

    Raises:
        ValueError: If a script identifier, metadata value, or runner is invalid.
        OSError: If a discovered script cannot be read.
    """
    if hardware not in ("h100", "gb200"):
        raise ValueError(f"Unsupported hardware: {hardware!r}")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", runner_prefix):
        raise ValueError("Runner prefix must be a nonempty runner label")

    output_prefix = "matrix_" if hardware == "h100" else "matrix_gb200_"
    matrices: dict[str, Matrix] = {}
    for tier in ("L0", "L1", "L2", "flaky"):
        pattern = "flaky/L*.sh" if tier == "flaky" else f"active/{tier}_*.sh"
        entries = [
            _entry(path, hardware=hardware, runner_prefix=runner_prefix)
            for path in sorted((launch_root / hardware).glob(pattern))
            if path.is_file()
        ]
        matrices[f"{output_prefix}{tier.lower()}"] = {"include": entries}
    return matrices


def main() -> None:
    """Append validated JSON matrices to the GitHub Actions output file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", required=True, choices=("h100", "gb200"))
    parser.add_argument("--runner-prefix")
    args = parser.parse_args()
    runner_prefix = args.runner_prefix
    if runner_prefix is None:
        if args.hardware == "h100":
            parser.error("--runner-prefix is required for H100")
        runner_prefix = "nemo-ci-gcp-gpu-x2"
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        parser.error("GITHUB_OUTPUT must be set")

    try:
        matrices = build_matrices(
            Path("tests/functional_tests/launch_scripts"), hardware=args.hardware, runner_prefix=runner_prefix
        )
        # Finish validation and serialization before publishing any matrix.
        output = "".join(f"{key}={json.dumps(matrix, separators=(',', ':'))}\n" for key, matrix in matrices.items())
        with Path(output_path).open("a", encoding="utf-8") as destination:
            destination.write(output)
    except (OSError, ValueError) as error:
        logger.error("Cannot generate functional-test matrices: %s", error)
        raise SystemExit(1) from error


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
