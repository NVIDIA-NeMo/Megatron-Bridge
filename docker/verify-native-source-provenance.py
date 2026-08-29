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

"""Verify native package sources selected by a frozen Megatron Core manifest."""

import argparse
import json
import re
from pathlib import Path
from typing import Any

_GIT_SHA1_PART = re.compile(r"[0-9a-f]{20}")


def _git_commit(value: dict[str, Any]) -> str:
    """Reconstruct and validate a split Git SHA-1 object ID."""
    assert value.get("algorithm") == "git-sha1"
    parts = value.get("parts")
    assert isinstance(parts, list) and len(parts) == 2
    assert all(
        isinstance(part, str) and _GIT_SHA1_PART.fullmatch(part) for part in parts
    )
    return "".join(parts)


def _lane(manifest: dict[str, Any], mcore_ref: str) -> dict[str, Any]:
    """Return the unique lane selected by an immutable MCore commit."""
    lanes = [
        lane
        for lane in manifest["lanes"]
        if _git_commit(lane["mcore_commit"]) == mcore_ref
    ]
    assert len(lanes) == 1, f"unknown or duplicate MCore provenance: {mcore_ref}"
    return lanes[0]


def _source_commit(lane: dict[str, Any], name: str) -> str:
    """Return one required native source commit."""
    source = lane["native_sources"][name]
    assert isinstance(source, dict), f"missing {name} source provenance"
    repository = source.get("repository")
    assert isinstance(repository, str) and repository.startswith("https://github.com/")
    return _git_commit(source["commit"])


def _verify_mcore_manifest(lane: dict[str, Any], path: Path) -> None:
    """Verify torch-memory-saver provenance against the frozen MCore manifest."""
    text = path.read_text()
    source = lane["native_sources"]["torch_memory_saver"]
    if source is None:
        assert "torch-memory-saver" not in text
        return

    repository = source["repository"]
    commit = _git_commit(source["commit"])
    expected = f'torch-memory-saver = {{ git = "{repository}", rev = "{commit}" }}'
    assert (
        expected in text
    ), "torch-memory-saver source does not match frozen MCore provenance"


def main() -> None:
    """Validate lane provenance and, when available, its MCore manifest."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mcore-ref", required=True)
    parser.add_argument("--transformer-engine-ref")
    parser.add_argument("--fast-hadamard-transform-ref")
    parser.add_argument("--mcore-pyproject", type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    lane = _lane(manifest, args.mcore_ref)

    if args.transformer_engine_ref is not None:
        assert _source_commit(lane, "transformer_engine") == args.transformer_engine_ref
    if args.fast_hadamard_transform_ref is not None:
        assert (
            _source_commit(lane, "fast_hadamard_transform")
            == args.fast_hadamard_transform_ref
        )
    if args.mcore_pyproject is not None:
        _verify_mcore_manifest(lane, args.mcore_pyproject)


if __name__ == "__main__":
    main()
