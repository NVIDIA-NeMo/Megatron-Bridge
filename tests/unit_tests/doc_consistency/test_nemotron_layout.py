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

"""Check example discovery and links without importing GPU dependencies."""

import importlib
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest


pytestmark = pytest.mark.unit
REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_ROOT = REPO_ROOT / "examples/models/nemotron"
MODEL_DIRECTORIES = (
    "nemotron_3_nano",
    "nemotron_3_nano_omni",
    "nemotron_3_super",
    "nemotron_3_ultra",
    "nemotron_3_5_lightning",
    "nemotron_3_5_super_vl",
)


def test_model_directories_are_flat() -> None:
    directories = {path.name for path in EXAMPLE_ROOT.iterdir() if path.is_dir() and path.name != "__pycache__"}
    assert directories == set(MODEL_DIRECTORIES)


@pytest.mark.parametrize("name", MODEL_DIRECTORIES)
def test_model_example_namespace_is_importable(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(REPO_ROOT))
    package = importlib.import_module(f"examples.models.nemotron.{name}")
    assert (EXAMPLE_ROOT / name).resolve() in {Path(path).resolve() for path in package.__path__}


@pytest.mark.parametrize("readme", sorted(EXAMPLE_ROOT.rglob("*.md")))
def test_example_relative_links(readme: Path) -> None:
    for target in re.findall(r"\]\(([^)]+)\)", readme.read_text()):
        url = urlsplit(target)
        if url.scheme or not url.path:
            continue
        destination = (readme.parent / unquote(url.path)).resolve()
        assert destination.exists(), f"{readme.relative_to(REPO_ROOT)}: {target}"
        if url.fragment and destination.suffix == ".md":
            headings = re.findall(r"^#+\s+(.+)$", destination.read_text(), re.MULTILINE)
            anchors = {re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-") for heading in headings}
            assert url.fragment in anchors, f"{destination}: #{url.fragment}"


def test_current_nemotron_example_references_resolve() -> None:
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=REPO_ROOT).decode().split("\0")
    references = 0
    for name in tracked:
        path = REPO_ROOT / name
        # Versioned commands describe their release checkout. Links to main
        # within those pages must still resolve against the current tree.
        if not path.is_file():
            continue
        content = path.read_bytes()
        if b"\0" in content:
            continue
        text = content.decode("utf-8", errors="replace")
        if name.startswith("docs/fern/versions/") and not name.startswith("docs/fern/versions/nightly/"):
            text = "\n".join(
                re.findall(r"https://github.com/NVIDIA-NeMo/Megatron-Bridge/(?:blob|tree)/main/[^\s)]+", text)
            )
        text = re.sub(r"https://github.com/NVIDIA-NeMo/Megatron-Bridge/(?:blob|tree)/(?!main/)[^\s)]+", "", text)
        for target in re.findall(r"examples/models/nemotron/[\w./-]+", text):
            target = target.rstrip(".")
            references += 1
            assert (REPO_ROOT / target).exists(), f"{name}: {target}"
    assert references > 20
