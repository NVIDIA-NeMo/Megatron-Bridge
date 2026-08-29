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

"""Verify native package sources selected by frozen Megatron Core and Bridge manifests."""

import argparse
import json
import re
from pathlib import Path
from typing import Any

_GIT_SHA1_PART = re.compile(r"[0-9a-f]{20}")


def _require(condition: bool, message: str) -> None:
    """Reject invalid provenance without relying on optimizable assertions."""
    if not condition:
        raise ValueError(message)


def _git_commit(value: dict[str, Any], label: str) -> str:
    """Reconstruct and validate a split Git SHA-1 object ID."""
    _require(value.get("algorithm") == "git-sha1", f"invalid algorithm for {label}")
    parts = value.get("parts")
    _require(isinstance(parts, list) and len(parts) == 2, f"invalid parts for {label}")
    _require(
        all(isinstance(part, str) and _GIT_SHA1_PART.fullmatch(part) for part in parts),
        f"invalid Git SHA-1 for {label}",
    )
    return "".join(parts)


def _lane(manifest: dict[str, Any], mcore_ref: str) -> dict[str, Any]:
    """Return the unique lane selected by an immutable MCore commit."""
    lanes = [
        lane
        for lane in manifest["lanes"]
        if _git_commit(lane["mcore_commit"], f"{lane['name']} MCore") == mcore_ref
    ]
    _require(len(lanes) == 1, f"unknown or duplicate MCore provenance: {mcore_ref}")
    return lanes[0]


def _source(lane: dict[str, Any], name: str) -> dict[str, Any]:
    """Return one required native source record."""
    source = lane["native_sources"].get(name)
    _require(isinstance(source, dict), f"missing {name} source provenance")
    _require(
        isinstance(source.get("repository"), str)
        and source["repository"].startswith("https://github.com/"),
        f"invalid {name} repository",
    )
    _require(isinstance(source.get("package"), str), f"invalid {name} package")
    return source


def _build_commit(lane: dict[str, Any], name: str) -> str:
    """Return one native source commit used by the build cache."""
    return _git_commit(_source(lane, name)["build_commit"], f"{name} build")


def _verify_mcore_manifest(lane: dict[str, Any], path: Path) -> None:
    """Verify every active native source against the frozen MCore manifest."""
    text = path.read_text()
    for name, source in lane["native_sources"].items():
        package = name.replace("_", "-")
        if source is None:
            _require(
                package not in text, f"unexpected {package} source in MCore manifest"
            )
            continue
        package = source["package"]
        repository = source["repository"]
        commit = _git_commit(source["mcore_commit"], f"{name} MCore")
        expected = f'{package} = {{ git = "{repository}", rev = "{commit}" }}'
        _require(
            expected in text, f"{package} source does not match frozen MCore provenance"
        )


def _verify_bridge_manifest(lane: dict[str, Any], path: Path) -> None:
    """Verify each explicit Bridge-side selector against its lane provenance."""
    text = path.read_text()
    for name, source in lane["native_sources"].items():
        if source is None or source["bridge_selector"] is None:
            continue
        selector = source["bridge_selector"]
        package = source["package"]
        repository = source["repository"]
        if selector["kind"] == "uv-source":
            value = _git_commit(selector["commit"], f"{name} Bridge selector")
            expected = f'{package} = {{ git = "{repository}", rev = "{value}" }}'
        elif selector["kind"] == "vcs-requirement":
            value = selector.get("value")
            if value is None:
                value = _git_commit(selector["commit"], f"{name} Bridge selector")
            expected = f"{package} @ git+{repository}@{value}"
        else:
            raise ValueError(f"invalid {name} Bridge selector kind")
        _require(expected in text, f"{package} source does not match Bridge provenance")


def main() -> None:
    """Validate lane provenance and the supplied source manifests."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mcore-ref", required=True)
    parser.add_argument("--transformer-engine-ref")
    parser.add_argument("--fast-hadamard-transform-ref")
    parser.add_argument("--mcore-pyproject", type=Path)
    parser.add_argument("--bridge-pyproject", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    lane = _lane(manifest, args.mcore_ref)
    if args.transformer_engine_ref is not None:
        _require(
            _build_commit(lane, "transformer_engine") == args.transformer_engine_ref,
            "TransformerEngine build source does not match provenance",
        )
    if args.fast_hadamard_transform_ref is not None:
        _require(
            _build_commit(lane, "fast_hadamard_transform")
            == args.fast_hadamard_transform_ref,
            "fast-hadamard-transform build source does not match provenance",
        )
    if args.mcore_pyproject is not None:
        _verify_mcore_manifest(lane, args.mcore_pyproject)
    if args.bridge_pyproject is not None:
        _verify_bridge_manifest(lane, args.bridge_pyproject)


if __name__ == "__main__":
    main()
