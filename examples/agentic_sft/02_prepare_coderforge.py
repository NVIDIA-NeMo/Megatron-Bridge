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

"""Materialize a bounded, revision-pinned CoderForge trajectory subset.

Only selected Parquet shards are downloaded. This is seeded shard sampling, not uniform row
sampling over the entire dataset. No tokenization or packing is performed here. Requires
``huggingface_hub`` and ``pyarrow``.

Adapted from the Megatron-LM agentic SFT example by Zhongbo Zhu.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import random
from collections import Counter
from pathlib import Path


logger = logging.getLogger(__name__)

REPO_ID = "togethercomputer/CoderForge-Preview"
REVISION = "060fca96cf723b2ebab3181e9e59fafd273df3cb"  # pragma: allowlist secret
SOURCE_SPLIT = "SWE_Rebench"
SOURCE_ROWS = 77169
DEFAULT_WORKSPACE = Path("agentic_sft_workspace")


def canonical_record(row: dict) -> dict:
    """Decode source-specific serialization while preserving trajectory fields.

    Raises:
        ValueError: With a short reason code if the row is not a usable trajectory.
    """
    record = dict(row)
    for key in ("messages", "tools"):
        value = record.get(key)
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid_{key}_json") from error
        if key == "tools" and value is None:
            record.pop(key, None)
            continue
        if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
            raise ValueError(f"invalid_{key}_schema")
        record[key] = value
    if not record["messages"]:
        raise ValueError("empty_messages")
    for message in record["messages"]:
        if not isinstance(message.get("role"), str):
            raise ValueError("invalid_role")
        if message.get("content") is not None and not isinstance(message["content"], str):
            raise ValueError("nontext_content")
        if message.get("tool_calls") is not None and not isinstance(message["tool_calls"], list):
            raise ValueError("invalid_tool_calls")
    if not any(message["role"] == "assistant" for message in record["messages"]):
        raise ValueError("no_assistant_turn")
    if not isinstance(record.get("trajectory_id"), str) or not record["trajectory_id"]:
        raise ValueError("missing_trajectory_id")
    if "image" in record:
        record["environment_image"] = record.pop("image")
    return record


def validation_group(record: dict, task_id_field: str | None) -> tuple[str, str]:
    """Use explicit task IDs when available; never guess trajectory-ID structure."""
    keys = (task_id_field,) if task_id_field else ("task_id", "instance_id")
    for key in keys:
        if record.get(key) is not None:
            return key, str(record[key])
    if task_id_field:
        raise ValueError("missing_task_id")
    return "trajectory_id", record["trajectory_id"]


def split_name(group: tuple[str, str], seed: int, validation_fraction: float) -> str:
    """Assign a group to the training or validation split from a seeded hash."""
    payload = json.dumps([seed, *group], ensure_ascii=False).encode("utf-8")
    fraction = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / 2**64
    return "validation" if fraction < validation_fraction else "training"


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file, read in 1 MiB blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    """Select CoderForge shards and write the training and validation JSONL with a manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_WORKSPACE / "data/coderforge")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_WORKSPACE / "hf_cache")
    parser.add_argument(
        "--percentage",
        type=float,
        default=5,
        help="Percentage of source rows to inspect before any later filtering (default: 5).",
    )
    parser.add_argument("--max-samples", type=int, help="Smaller candidate limit for smoke tests.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--task-id-field", help="Verified source task ID for group-disjoint splits.")
    args = parser.parse_args()
    if not 0 < args.percentage <= 100 or not 0 < args.validation_fraction < 1:
        parser.error("percentage must be in (0, 100] and validation fraction in (0, 1)")
    budget = math.floor(SOURCE_ROWS * args.percentage / 100)
    if args.max_samples is not None:
        if args.max_samples < 2:
            parser.error("--max-samples must be at least 2")
        budget = min(budget, args.max_samples)
    if budget < 2:
        parser.error("percentage selects fewer than two candidates")

    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {name: args.output_dir / f"{name}.jsonl" for name in ("training", "validation")}
    manifest_path = args.output_dir / "manifest.json"
    if any(path.exists() for path in [*outputs.values(), manifest_path]):
        parser.error("output already exists; choose a new --output-dir to preserve provenance")

    files = HfApi().list_repo_files(REPO_ID, repo_type="dataset", revision=REVISION)
    shards = sorted(name for name in files if name.startswith(f"trajectories/{SOURCE_SPLIT}-"))
    random.Random(args.seed).shuffle(shards)
    rejected = Counter()
    counts = Counter()
    group_fields = Counter()
    selected_shards = []
    seen = set()
    candidates = 0
    temporary = {name: path.with_suffix(".jsonl.partial") for name, path in outputs.items()}
    streams = {name: path.open("w", encoding="utf-8") for name, path in temporary.items()}
    try:
        for shard in shards:
            if candidates >= budget:
                break
            logger.info("Reading %s (%d/%d candidates)", shard, candidates, budget)
            local = hf_hub_download(REPO_ID, shard, repo_type="dataset", revision=REVISION, cache_dir=args.cache_dir)
            selected_shards.append({"path": shard, "sha256": file_sha256(Path(local))})
            for batch in pq.ParquetFile(local).iter_batches(batch_size=32):
                for row in batch.to_pylist():
                    if candidates >= budget:
                        break
                    candidates += 1
                    try:
                        record = canonical_record(row)
                        if record["trajectory_id"] in seen:
                            raise ValueError("duplicate_trajectory_id")
                        group = validation_group(record, args.task_id_field)
                    except ValueError as error:
                        rejected[str(error)] += 1
                        continue
                    seen.add(record["trajectory_id"])
                    split = split_name(group, args.seed, args.validation_fraction)
                    streams[split].write(json.dumps(record, ensure_ascii=False) + "\n")
                    counts[split] += 1
                    group_fields[group[0]] += 1
                if candidates >= budget:
                    break
    finally:
        for stream in streams.values():
            stream.close()
    if not counts["training"] or not counts["validation"]:
        raise ValueError("A split is empty; increase the candidate budget or validation fraction")
    for name, path in outputs.items():
        temporary[name].replace(path)
    manifest = {
        "format_version": 1,
        "source": {
            "repo": REPO_ID,
            "revision": REVISION,
            "config": "trajectories",
            "split": SOURCE_SPLIT,
        },
        "source_rows": SOURCE_ROWS,
        "requested_percentage": args.percentage,
        "preparation_script_sha256": file_sha256(Path(__file__)),
        "candidate_budget": budget,
        "candidate_rows": candidates,
        "accepted_rows": sum(counts.values()),
        "sampling": "seeded shard order; prefix within selected shards; stop at candidate budget",
        "seed": args.seed,
        "selected_shards": selected_shards,
        "rejected_rows_by_reason": dict(rejected),
        "split": {
            "method": "sha256(seed, group field, group value)",
            "validation_fraction": args.validation_fraction,
            "group_fields": dict(group_fields),
            "task_disjoint": not group_fields["trajectory_id"],
        },
        "conversion": {
            "serialized_arrays": ["messages", "tools"],
            "image": "environment_image",
            "tokenized": False,
        },
        "outputs": {
            name: {"file": path.name, "rows": counts[name], "sha256": file_sha256(path)}
            for name, path in outputs.items()
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logger.info("Prepared %s; rejected %s; manifest: %s", dict(counts), dict(rejected), manifest_path)


if __name__ == "__main__":
    main()
