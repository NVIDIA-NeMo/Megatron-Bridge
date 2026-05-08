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

"""Build an Energon WebDataset from preprocessed Valor32k-AVQA v2.0 data.

Runs the full pipeline in one shot:
  1. Build WebDataset tar shards (one sample per QA pair: conversation + video + audio)
  2. Restructure shards to the flat Energon layout (split-prefixed filenames at the root)
  3. Create all Energon metadata (.info.yaml, index.sqlite, index.uuid, split.yaml)

After this script finishes, only one manual step remains: write .nv-meta/dataset.yaml
to declare the sample type and field mapping (see data_preparation.md Step 6).

Note: energon prepare deadlocks in all modes on this version of megatron-energon due to
a bug in AggregatorPool.close() (aggregator_process.join() never returns). This script
bypasses energon prepare entirely by indexing the tar files directly with stdlib tarfile.

Usage:
    uv run python examples/models/vlm/nemotron_3_omni/data/build_valor32k_avqa_shards.py \\
        --data_root /data/valor32k_avqa \\
        --output_dir /data/valor32k_avqa/energon \\
        --samples_per_shard 100
"""

import argparse
import json
import logging
import re
import sqlite3
import tarfile
import uuid
from collections import defaultdict
from pathlib import Path

import yaml

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# VALOR-32K filenames have the form {youtube_id}_{start.000}_{end.000}.mp4.
# The QA JSON stores only the bare youtube_id, so we need to map back.
_TS_SUFFIX_RE = re.compile(r"^(.+)_\d+\.\d+_\d+\.\d+$")


def _build_file_index(directory: Path, suffix: str) -> dict[str, Path]:
    """Map bare youtube_id -> actual file path, stripping the timestamp suffix."""
    index: dict[str, Path] = {}
    for p in directory.glob(f"*{suffix}"):
        m = _TS_SUFFIX_RE.match(p.stem)
        key = m.group(1) if m else p.stem
        index[key] = p
    return index


def build_conversation_json(question: str, answer: str) -> str:
    """Serialize a single QA turn into Energon conversation JSON."""
    conversation = [
        {"role": "user", "content": [{"type": "video"}, {"type": "text", "text": question}]},
        {"role": "assistant", "content": [{"type": "text", "text": answer}]},
    ]
    return json.dumps(conversation)


def build_shards(data_root: Path, output_dir: Path, split: str, samples_per_shard: int) -> int:
    """Write video+audio QA samples into webdataset tar shards for one split."""
    import webdataset as wds

    qa_file = data_root / f"combined_dataset_{split}_flattened.json"
    if not qa_file.exists():
        logger.warning(f"Skipping {split}: {qa_file} not found")
        return 0

    with open(qa_file) as f:
        qa_pairs = json.load(f)

    # Build youtube_id -> file path indices once per split.
    video_index = _build_file_index(data_root / "videos", ".mp4")
    audio_index = _build_file_index(data_root / "audio", ".wav")
    logger.info(f"  {split}: indexed {len(video_index)} videos, {len(audio_index)} audio files")

    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(split_dir / "shard-%06d.tar")
    written = 0
    skipped = 0

    with wds.ShardWriter(pattern, maxcount=samples_per_shard) as sink:
        items = tqdm(qa_pairs, desc=f"  {split}") if tqdm else qa_pairs
        for qa in items:
            video_id = str(qa["video_id"])
            sample_id = str(qa.get("id", written))
            video_path = video_index.get(video_id)
            audio_path = audio_index.get(video_id)

            if video_path is None or audio_path is None:
                skipped += 1
                continue

            question = qa["question"]
            answer = qa["options"][qa["correct_answer_idx"]]

            sample = {
                "__key__": f"{video_id}_{sample_id}",
                "conversation.json": build_conversation_json(question, answer).encode(),
            }
            with open(video_path, "rb") as vf:
                sample["video.mp4"] = vf.read()
            with open(audio_path, "rb") as af:
                sample["audio.wav"] = af.read()

            sink.write(sample)
            written += 1

    logger.info(f"  {split}: wrote {written} samples, skipped {skipped} (missing video/audio)")
    return written


def restructure_shards(output_dir: Path) -> dict[str, list[str]]:
    """Move shards from per-split subdirs into the Energon flat root layout.

    Before: output_dir/{split}/shard-000000.tar ...
    After:  output_dir/{split}-shard-000000.tar ...

    Returns a dict mapping each split name to its ordered list of shard filenames.
    """
    split_shards: dict[str, list[str]] = {}
    for split in ("train", "val", "test"):
        split_dir = output_dir / split
        if not split_dir.is_dir():
            continue
        tars = sorted(p.name for p in split_dir.iterdir() if p.suffix == ".tar")
        names = []
        for tar in tars:
            new_name = f"{split}-{tar}"
            (split_dir / tar).rename(output_dir / new_name)
            names.append(new_name)
        split_dir.rmdir()
        split_shards[split] = names
        logger.info(f"  {split}: moved {len(names)} shards")
    return split_shards


def build_energon_index(output_dir: Path, split_shards: dict[str, list[str]]) -> None:
    """Create all Energon metadata files from the actual tar contents.

    Bypasses `energon prepare`, which deadlocks in all modes on this version of
    megatron-energon (AggregatorPool.close() → aggregator_process.join() blocks
    indefinitely; the sqlite is always complete before the hang). Uses stdlib
    tarfile to read member byte offsets directly instead.

    Creates:
        .nv-meta/.info.yaml    — per-shard sample counts
        .nv-meta/index.sqlite  — byte-offset index for random-access loading
        .nv-meta/index.uuid    — unique dataset identifier
        .nv-meta/split.yaml    — train/val/test shard assignment
    """
    meta_dir = output_dir / ".nv-meta"
    meta_dir.mkdir(exist_ok=True)

    # Build the ordered shard list (train → val → test) and index each tar.
    ordered_shards = [name for split in ("train", "val", "test") for name in split_shards.get(split, [])]

    db_path = meta_dir / "index.sqlite"
    db_path.unlink(missing_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE samples ("
        "  id           INTEGER PRIMARY KEY,"
        "  tar_file_id  INTEGER,"
        "  sample_key   TEXT,"
        "  sample_index INTEGER,"
        "  byte_offset  INTEGER,"
        "  byte_size    INTEGER)"
    )
    conn.execute("CREATE INDEX idx_samples_sample_key ON samples(sample_key)")

    shard_counts: dict[str, int] = {}
    for tar_file_id, shard_name in enumerate(ordered_shards):
        with tarfile.open(output_dir / shard_name) as tf:
            members = tf.getmembers()

        # Group tar members by WebDataset key (everything before the first '.').
        groups: dict[str, list] = defaultdict(list)
        for m in members:
            groups[m.name.split(".", 1)[0]].append(m)

        ordered_groups = sorted(groups.items(), key=lambda kv: min(m.offset for m in kv[1]))
        rows = []
        for sample_index, (sample_key, mems) in enumerate(ordered_groups):
            mems_sorted = sorted(mems, key=lambda m: m.offset)
            byte_offset = mems_sorted[0].offset
            if sample_index + 1 < len(ordered_groups):
                byte_size = min(m.offset for m in ordered_groups[sample_index + 1][1]) - byte_offset
            else:
                last = mems_sorted[-1]
                byte_size = last.offset_data + ((last.size + 511) // 512) * 512 - byte_offset
            rows.append((tar_file_id, sample_key, sample_index, byte_offset, byte_size))

        conn.executemany(
            "INSERT INTO samples (tar_file_id, sample_key, sample_index, byte_offset, byte_size)"
            " VALUES (?,?,?,?,?)",
            rows,
        )
        conn.commit()
        shard_counts[shard_name] = len(rows)

        if tar_file_id % 200 == 0:
            total = sum(shard_counts.values())
            logger.info(f"  indexed {tar_file_id}/{len(ordered_shards)} shards ({total} samples)")

    conn.close()
    logger.info(f"  index.sqlite: {sum(shard_counts.values())} samples across {len(ordered_shards)} shards")

    with open(meta_dir / ".info.yaml", "w") as f:
        yaml.dump({"shard_counts": shard_counts}, f)

    with open(meta_dir / "index.uuid", "w") as f:
        f.write(str(uuid.uuid4()))

    with open(meta_dir / "split.yaml", "w") as f:
        yaml.dump({"split_parts": {k: v for k, v in split_shards.items()}, "exclude": []}, f)

    logger.info(f"  Energon metadata written to {meta_dir}")


def main():
    """Run the full Energon dataset build pipeline for Valor32k-AVQA."""
    parser = argparse.ArgumentParser(description="Build Energon dataset from Valor32k-AVQA")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--samples_per_shard", type=int, default=100)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Step 1/3: building WebDataset shards")
    for split in ("train", "val", "test"):
        build_shards(data_root, output_dir, split, args.samples_per_shard)

    logger.info("Step 2/3: restructuring shards to Energon flat layout")
    split_shards = restructure_shards(output_dir)

    logger.info("Step 3/3: building Energon index and metadata")
    build_energon_index(output_dir, split_shards)

    logger.info(
        f"Done. Energon dataset at: {output_dir}\n"
        "Remaining: write .nv-meta/dataset.yaml (see data_preparation.md Step 4)"
    )


if __name__ == "__main__":
    main()
