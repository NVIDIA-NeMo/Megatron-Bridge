#!/usr/bin/env python3
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
"""Decode launcher arguments and run the forward-logit comparison."""

from __future__ import annotations

import argparse
import base64
import json
import runpy
import sys
from pathlib import Path


def _decode_arguments(encoded: str) -> list[str]:
    """Decode the shell-safe comparison argument payload."""
    try:
        value = json.loads(base64.urlsafe_b64decode(encoded.encode()).decode())
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Invalid encoded comparison arguments.") from error
    if not isinstance(value, list) or not all(isinstance(argument, str) for argument in value):
        raise ValueError("Encoded comparison arguments must be a list of strings.")
    return value


def main() -> None:
    """Run the repository comparison implementation with decoded arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arguments-b64", required=True, help=argparse.SUPPRESS)
    launcher_args = parser.parse_args()

    compare_dir = Path(__file__).resolve().parents[2] / "examples" / "conversion" / "compare_hf_and_megatron"
    compare_script = compare_dir / "compare.py"
    sys.path.insert(0, str(compare_dir))
    sys.argv = [str(compare_script), *_decode_arguments(launcher_args.arguments_b64)]
    runpy.run_path(str(compare_script), run_name="__main__")


if __name__ == "__main__":
    main()
