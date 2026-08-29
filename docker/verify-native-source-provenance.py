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

import tomllib
from packaging.requirements import InvalidRequirement, Requirement


_GIT_OID_PART = re.compile(r"[0-9a-f]{20}")


def _require(condition: bool, message: str) -> None:
    """Reject invalid provenance without relying on optimizable assertions."""
    if not condition:
        raise ValueError(message)


def _git_commit(value: dict[str, Any], label: str) -> str:
    """Reconstruct and validate a split Git object ID."""
    _require(
        value.get("object_format") == "40-hex-git-oid",
        f"invalid object format for {label}",
    )
    parts = value.get("parts")
    _require(isinstance(parts, list) and len(parts) == 2, f"invalid parts for {label}")
    _require(
        all(isinstance(part, str) and _GIT_OID_PART.fullmatch(part) for part in parts),
        f"invalid Git object ID for {label}",
    )
    return "".join(parts)


def _lane(manifest: dict[str, Any], mcore_ref: str) -> dict[str, Any]:
    """Return the unique lane selected by an immutable MCore commit."""
    lanes = [
        lane for lane in manifest["lanes"] if _git_commit(lane["mcore_commit"], f"{lane['name']} MCore") == mcore_ref
    ]
    _require(len(lanes) == 1, f"unknown or duplicate MCore provenance: {mcore_ref}")
    return lanes[0]


def _source(lane: dict[str, Any], name: str) -> dict[str, Any]:
    """Return one required native source record."""
    source = lane["native_sources"].get(name)
    _require(isinstance(source, dict), f"missing {name} source provenance")
    _require(
        isinstance(source.get("repository"), str) and source["repository"].startswith("https://github.com/"),
        f"invalid {name} repository",
    )
    _require(isinstance(source.get("package"), str), f"invalid {name} package")
    return source


def _build_commit(lane: dict[str, Any], name: str) -> str:
    """Return one native source commit used by the build cache."""
    return _git_commit(_source(lane, name)["build_commit"], f"{name} build")


def _uv_sources(path: Path) -> dict[str, Any]:
    """Parse effective uv source declarations from a project manifest."""
    with path.open("rb") as manifest_file:
        manifest = tomllib.load(manifest_file)
    sources = manifest.get("tool", {}).get("uv", {}).get("sources", {})
    _require(isinstance(sources, dict), "invalid tool.uv.sources table")
    return sources


def _source_entries(sources: dict[str, Any], package: str) -> list[dict[str, Any]]:
    """Return normalized source entries for one package."""
    entries = sources.get(package, [])
    if isinstance(entries, dict):
        entries = [entries]
    _require(
        isinstance(entries, list) and all(isinstance(entry, dict) for entry in entries),
        f"invalid {package} source declaration",
    )
    return entries


def _verify_exact_source(sources: dict[str, Any], package: str, repository: str, revision: str, label: str) -> None:
    """Require exactly one unconditional source at the approved repository and revision."""
    entries = _source_entries(sources, package)
    _require(len(entries) == 1, f"{label} must have exactly one source declaration")
    entry = entries[0]
    _require(set(entry) == {"git", "rev"}, f"{label} source must be unconditional")
    _require(
        entry["git"] == repository and entry["rev"] == revision,
        f"{label} source does not match provenance",
    )


def _verify_mcore_manifest(lane: dict[str, Any], path: Path) -> None:
    """Verify every active native source against parsed frozen MCore metadata."""
    sources = _uv_sources(path)
    for name, source in lane["native_sources"].items():
        package = name.replace("_", "-")
        entries = _source_entries(sources, package)
        if source is None:
            _require(not entries, f"unexpected {package} source in MCore manifest")
            continue
        _verify_exact_source(
            sources,
            source["package"],
            source["repository"],
            _git_commit(source["mcore_commit"], f"{name} MCore"),
            f"{source['package']} MCore",
        )


def _verify_bridge_manifest(lane: dict[str, Any], path: Path) -> None:
    """Verify each explicit Bridge-side selector against parsed project metadata."""
    with path.open("rb") as manifest_file:
        manifest = tomllib.load(manifest_file)
    sources = manifest.get("tool", {}).get("uv", {}).get("sources", {})
    overrides = manifest.get("tool", {}).get("uv", {}).get("override-dependencies", [])
    _require(isinstance(sources, dict), "invalid Bridge tool.uv.sources table")
    _require(isinstance(overrides, list), "invalid Bridge override dependencies")
    requirements: list[Requirement] = []
    for override in overrides:
        try:
            requirements.append(Requirement(override))
        except InvalidRequirement as error:
            raise ValueError("invalid Bridge override dependency") from error
    for name, source in lane["native_sources"].items():
        if source is None or source["bridge_selector"] is None:
            continue
        selector = source["bridge_selector"]
        package = source["package"]
        repository = source["repository"]
        if selector["kind"] == "uv-source":
            revision = _git_commit(selector["commit"], f"{name} Bridge selector")
            _verify_exact_source(sources, package, repository, revision, f"{package} Bridge")
        elif selector["kind"] == "vcs-requirement":
            revision = selector.get("value")
            if revision is None:
                revision = _git_commit(selector["commit"], f"{name} Bridge selector")
            matching = [requirement for requirement in requirements if requirement.name == package]
            _require(len(matching) == 1, f"{package} must have exactly one Bridge override")
            requirement = matching[0]
            _require(requirement.marker is None, f"{package} Bridge override must be unconditional")
            _require(
                requirement.url == f"git+{repository}@{revision}",
                f"{package} source does not match Bridge provenance",
            )
        else:
            raise ValueError(f"invalid {name} Bridge selector kind")


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
            _build_commit(lane, "fast_hadamard_transform") == args.fast_hadamard_transform_ref,
            "fast-hadamard-transform build source does not match provenance",
        )
    if args.mcore_pyproject is not None:
        _verify_mcore_manifest(lane, args.mcore_pyproject)
    if args.bridge_pyproject is not None:
        _verify_bridge_manifest(lane, args.bridge_pyproject)


if __name__ == "__main__":
    main()
