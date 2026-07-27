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

"""Audit completed img2dataset shards before converting them to Energon."""

import argparse
import json
import logging
import tarfile
from collections import defaultdict
from pathlib import Path


logger = logging.getLogger(__name__)
EXPECTED_EXTENSIONS = {"jpg", "json", "txt"}


def audit_download(
    shards_dir: Path,
    *,
    expected_shards: int,
    expected_attempts: int,
    minimum_successes: int,
) -> dict[str, int]:
    """Validate shard completion, member grouping, and the usable sample count.

    Args:
        shards_dir: Directory containing raw tar and completion-stat pairs.
        expected_shards: Required number of completed shards.
        expected_attempts: Required total attempted URL count.
        minimum_successes: Minimum usable image-caption sample count.

    Returns:
        Audited counts for attempts, successes, shards, and tar samples.
    """
    stats_paths = sorted(shards_dir.glob("*_stats.json"))
    tar_paths = sorted(shards_dir.glob("*.tar"))
    if len(stats_paths) != expected_shards or len(tar_paths) != expected_shards:
        raise RuntimeError(
            f"Found {len(stats_paths)} stats files and {len(tar_paths)} tars; expected {expected_shards} each."
        )
    tar_stems = {path.stem for path in tar_paths}
    stats_stems = {path.name.removesuffix("_stats.json") for path in stats_paths}
    if tar_stems != stats_stems:
        raise RuntimeError("Completed stats files and tar shards do not have matching stems.")

    stats = [json.loads(path.read_text(encoding="utf-8")) for path in stats_paths]
    attempted = sum(row["count"] for row in stats)
    successes = sum(row["successes"] for row in stats)
    if attempted != expected_attempts:
        raise RuntimeError(f"Stats report {attempted} attempts; expected {expected_attempts}.")
    if successes < minimum_successes:
        raise RuntimeError(f"Stats report {successes} successes; require at least {minimum_successes}.")

    tar_samples = 0
    for path in tar_paths:
        members: dict[str, set[str]] = defaultdict(set)
        closed_keys = set()
        previous_key = None
        with tarfile.open(path) as archive:
            for member in archive:
                if not member.isfile():
                    continue
                key, extension = Path(member.name).name.rsplit(".", 1)
                if extension not in EXPECTED_EXTENSIONS:
                    raise RuntimeError(f"{path.name} contains unexpected extension {extension!r}.")
                if previous_key is not None and key != previous_key:
                    closed_keys.add(previous_key)
                if key in closed_keys:
                    raise RuntimeError(f"{path.name} contains non-contiguous members for sample {key}.")
                previous_key = key
                if extension in members[key]:
                    raise RuntimeError(f"{path.name} contains duplicate {extension} data for sample {key}.")
                members[key].add(extension)
        if not members or any(extensions != EXPECTED_EXTENSIONS for extensions in members.values()):
            raise RuntimeError(f"{path.name} contains an incomplete image/JSON/TXT sample.")
        tar_samples += len(members)

    if tar_samples != successes:
        raise RuntimeError(f"Tar headers contain {tar_samples} samples, but stats report {successes} successes.")
    return {
        "attempted": attempted,
        "successes": successes,
        "complete_shards": len(tar_paths),
        "tar_samples": tar_samples,
    }


def main() -> None:
    """Parse arguments and audit a raw DataComp download."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards-dir", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, default=216)
    parser.add_argument("--expected-attempts", type=int, default=2_133_334)
    parser.add_argument("--minimum-successes", type=int, default=525_000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    summary = audit_download(
        args.shards_dir,
        expected_shards=args.expected_shards,
        expected_attempts=args.expected_attempts,
        minimum_successes=args.minimum_successes,
    )
    logger.info("%s", json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
