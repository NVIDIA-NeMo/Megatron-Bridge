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

"""Run the generic adapter verifier with 26.06.01 PEFT compatibility."""

from __future__ import annotations

import runpy
from pathlib import Path

from peft_compat import apply_peft_weight_converter_compatibility


def main() -> None:
    """Apply the container compatibility wrapper and run adapter verification."""
    apply_peft_weight_converter_compatibility()
    repo_root = Path(__file__).resolve().parents[4]
    verifier = repo_root / "examples" / "conversion" / "adapter" / "verify_adapter.py"
    runpy.run_path(str(verifier), run_name="__main__")


if __name__ == "__main__":
    main()
