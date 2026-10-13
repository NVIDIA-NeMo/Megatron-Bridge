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

"""Check same-repository branch URLs against a Git tree, without HTTP requests.

Input is Lychee's unfiltered HTTP(S) ``--dump`` output, one URL per line.
Explicit tags, SHAs, other branches, and external destinations are written to
a separate file for Lychee's normal online check. Relative Fern routes and
fragments are outside this file/directory existence check.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlsplit


LOGGER = logging.getLogger(__name__)


def repository_target(url: str, *, repository: str, branch: str) -> tuple[str, str] | None:
    """Return the expected Git object type and path for a candidate-tree URL.

    Args:
        url: Absolute URL extracted by Lychee.
        repository: Upstream owner/repository, independent of a fork's origin.
        branch: Branch whose URLs should resolve against the checked tree.

    Returns:
        A Git type and decoded path, or None for an online-check destination.
    """
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Expected an HTTP(S) URL from Lychee, got {url!r}")
    parts = parsed.path.lstrip("/").split("/", 2)
    if len(parts) != 3 or "/".join(parts[:2]).casefold() != repository.casefold():
        return None
    remainder = parts[2]
    if parsed.hostname == "github.com":
        route, separator, remainder = remainder.partition("/")
        if not separator or route not in {"blob", "tree"}:
            return None
        # GitHub redirects tree/file and blob/directory to their canonical route.
        kind = "object"
    elif parsed.hostname == "raw.githubusercontent.com":
        kind = "blob"
    else:
        return None

    # Match only at literal URL segment boundaries. Decoding the entire string
    # first would mistake an explicit main%2Fother ref for branch main + a path.
    segments = remainder.split("/")
    for ref in (f"refs/heads/{branch}", branch):
        for end in range(1, len(segments) + 1):
            candidate = unquote("/".join(segments[:end]))
            if candidate == ref:
                return kind, unquote("/".join(segments[end:])).rstrip("/")
            if len(candidate) >= len(ref):
                break
    return None


def git_tree(repo_root: Path, revision: str) -> dict[str, str]:
    """Read files and directories from an immutable tree, including shallow clones."""
    tree_id = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{revision}^{{tree}}"], cwd=repo_root, text=True
    ).strip()
    LOGGER.info("Checking repository links against %s (tree %s)", revision, tree_id)
    entries = subprocess.check_output(["git", "ls-tree", "-r", "-t", "-z", tree_id], cwd=repo_root)
    result = {"": "tree"}
    for entry in entries.split(b"\0"):
        if entry:
            metadata, path = entry.split(b"\t", 1)
            result[path.decode("utf-8")] = metadata.split()[1].decode("ascii")
    return result


def check_links(
    links: list[str], *, tree: dict[str, str], repository: str, branch: str
) -> tuple[list[str], list[str]]:
    """Partition extracted URLs into online destinations and local errors."""
    external = []
    errors = []
    checked = 0
    for url in sorted(set(links)):
        target = repository_target(url, repository=repository, branch=branch)
        if target is None:
            external.append(url)
            continue
        checked += 1
        expected_type, path = target
        actual_type = tree.get(path)
        if actual_type is None or (expected_type != "object" and actual_type != expected_type):
            errors.append(f"{url}: expected {expected_type} at {path!r}, found {actual_type or 'missing'}")
    LOGGER.info("Checked %d candidate-tree URLs; %d URLs remain for online checking", checked, len(external))
    return external, errors


def main() -> int:
    """Validate the candidate tree and write only online destinations for Lychee."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--revision", default="HEAD")
    parser.add_argument("--repository", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--links-file", type=Path, required=True)
    parser.add_argument("--external-links-file", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    links = [line.strip() for line in args.links_file.read_text().splitlines() if line.strip()]
    if not links:
        LOGGER.error("No URLs extracted; refusing to pass an empty documentation scan")
        return 1
    external, errors = check_links(
        links, tree=git_tree(args.repo_root, args.revision), repository=args.repository, branch=args.branch
    )
    for error in errors:
        LOGGER.error("%s", error)
    if errors:
        return 1
    args.external_links_file.write_text("".join(url + "\n" for url in external))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
