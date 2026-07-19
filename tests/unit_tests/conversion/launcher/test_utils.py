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

import importlib.util
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit

SCRIPT_DIR = Path(__file__).resolve().parents[4] / "scripts" / "conversion"
spec = importlib.util.spec_from_file_location("conversion_utils_under_test", SCRIPT_DIR / "utils.py")
assert spec is not None and spec.loader is not None
conversion_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conversion_utils)
prepare_output_directory = conversion_utils.prepare_output_directory


@pytest.mark.parametrize("relationship", ["equal", "destination-parent", "destination-child"])
def test_prepare_output_directory_rejects_source_overlap(tmp_path, relationship):
    source = tmp_path / "source"
    source.mkdir()
    source_marker = source / "checkpoint"
    source_marker.write_text("checkpoint")

    if relationship == "equal":
        destination = source
    elif relationship == "destination-parent":
        destination = tmp_path
    else:
        destination = source / "export"
        destination.mkdir()
        (destination / "partial-output").write_text("partial")

    with pytest.raises(ValueError, match="overlaps conversion source"):
        prepare_output_directory(str(destination), overwrite=True, source_paths=[str(source)])

    assert source_marker.read_text() == "checkpoint"
