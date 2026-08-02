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

import json
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit


def test_every_manifest_recipe_resolves_to_a_library_builder():
    repo_root = Path(__file__).resolve().parents[4]
    training_scripts = repo_root / "scripts" / "training"
    sys.path.insert(0, str(training_scripts))
    try:
        from recipe_runner import find_library_recipe
    finally:
        sys.path.remove(str(training_scripts))

    with (training_scripts / "text_pretrain_validation.json").open(encoding="utf-8") as manifest_file:
        targets = json.load(manifest_file)

    missing = [target["recipe"] for target in targets if find_library_recipe(target["recipe"]) is None]

    assert missing == []
