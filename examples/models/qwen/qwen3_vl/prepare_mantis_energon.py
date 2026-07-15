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

"""Convert TIGER-Lab/Mantis-Instruct into an indexed Qwen-VL Energon dataset.

Usage::

    uv run python examples/models/qwen/qwen3_vl/prepare_mantis_energon.py \
        --source-dir /path/to/Mantis-Instruct \
        --output-dir /path/to/mantis-energon \
        --max-samples-per-tar 1000 \
        --validation-fraction 0.01

The converter extracts each subset's ``train_images.zip``, reads its parquet
rows, and writes deterministic ``train-shard-*`` / ``val-shard-*`` WebDataset
files. Each sample stores a pickled image-byte list in ``.jpgs`` and its ChatML
conversation in ``.json``. By default the script then indexes the shards through
Energon's preparation API and writes the Bridge ``ChatMLWebdataset`` sample loader.
"""

import hashlib
import json
import logging
import pickle
import zipfile
from argparse import ArgumentParser
from contextlib import ExitStack
from pathlib import Path

import pandas as pd
import webdataset as wds
from tqdm import tqdm

from megatron.bridge.data.energon import prepare_webdataset


logger = logging.getLogger(__name__)
_MARKER_PREFIX = ".extracted_"
_DATASET_YAML = """\
__module__: megatron.bridge.data.energon.task_encoder_utils
__class__: ChatMLWebdataset
field_map:
  imgs: jpgs
  conversation: json
subflavors: {}
"""


def _ensure_extracted(subset_dir: str, zip_name: str) -> None:
    """Safely extract one subset archive once."""
    subset_path = Path(subset_dir).resolve()
    zip_path = subset_path / zip_name
    if not zip_path.exists():
        return
    marker = subset_path / f"{_MARKER_PREFIX}{zip_name}"
    if marker.exists():
        return

    logger.info("Extracting %s ...", zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        for member in tqdm(archive.infolist(), desc=f"extract {zip_name}", unit="file"):
            destination = (subset_path / member.filename).resolve()
            if not destination.is_relative_to(subset_path):
                raise ValueError(f"Unsafe zip member outside subset directory: {member.filename}")
            archive.extract(member, subset_path)
    marker.touch()


def _sample_split(key: str, validation_fraction: float) -> str:
    """Assign a stable train/validation split from a sample key."""
    if validation_fraction <= 0:
        return "train"
    bucket = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") / 2**64
    return "val" if bucket < validation_fraction else "train"


def _read_images(subset_dir: Path, references: list[dict]) -> list[bytes]:
    """Read every image referenced by one Mantis row."""
    images = []
    for reference in references:
        with (subset_dir / reference["path"]).open("rb") as image_file:
            images.append(image_file.read())
    return images


def convert(
    source_dir: str,
    output_dir: str,
    max_count: int,
    *,
    validation_fraction: float = 0.01,
) -> dict[str, int]:
    """Convert Mantis subsets into deterministic train and validation shards.

    Args:
        source_dir: Directory containing per-subset parquet and image archives.
        output_dir: Destination for WebDataset tar shards.
        max_count: Maximum number of samples per shard.
        validation_fraction: Stable fraction of samples assigned to ``val``.

    Returns:
        Counts written to the ``train`` and ``val`` splits.
    """
    if max_count <= 0:
        raise ValueError("max_count must be greater than 0.")
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1).")

    source_path = Path(source_dir)
    subsets = sorted(path for path in source_path.iterdir() if path.is_dir())
    if not subsets:
        raise FileNotFoundError(f"No subset directories found in {source_dir}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    counts = {"train": 0, "val": 0}
    total_skipped = 0

    with ExitStack() as stack:
        writers = {
            split: stack.enter_context(
                wds.ShardWriter(str(output_path / f"{split}-shard-%06d.tar"), maxcount=max_count)
            )
            for split in counts
        }
        for subset_path in subsets:
            _ensure_extracted(str(subset_path), "train_images.zip")
            parquet_files = sorted(subset_path.glob("train-*.parquet"))
            if not parquet_files:
                logger.debug("No train parquets in subset %s, skipping", subset_path.name)
                continue

            for parquet_path in parquet_files:
                dataframe = pd.read_parquet(parquet_path)
                for index, (_, row) in enumerate(
                    tqdm(
                        dataframe.iterrows(),
                        total=len(dataframe),
                        desc=f"{subset_path.name}/{parquet_path.name}",
                        unit="sample",
                    )
                ):
                    if row["images"] is None or len(row["images"]) == 0:
                        total_skipped += 1
                        continue

                    try:
                        images = _read_images(subset_path, row["images"])
                    except (KeyError, OSError) as error:
                        logger.warning("Skipping %s idx=%d: %s", parquet_path, index, error)
                        total_skipped += 1
                        continue

                    conversation = [dict(turn) for turn in row["conversation"]]
                    placeholders = sum(turn["content"].count("<image>") for turn in conversation)
                    if placeholders != len(images):
                        logger.warning(
                            "Skipping %s idx=%d: %d <image> placeholders but %d images",
                            parquet_path,
                            index,
                            placeholders,
                            len(images),
                        )
                        total_skipped += 1
                        continue

                    key = f"{subset_path.name}__{parquet_path.stem}__{index:06d}"
                    split = _sample_split(key, validation_fraction)
                    writers[split].write(
                        {
                            "__key__": key,
                            "jpgs": pickle.dumps(images),
                            "json": json.dumps(conversation).encode(),
                        }
                    )
                    counts[split] += 1

    for split, count in counts.items():
        if count == 0:
            for empty_shard in output_path.glob(f"{split}-shard-*.tar"):
                empty_shard.unlink()

    if sum(counts.values()) == 0:
        raise RuntimeError("No samples were written; inspect the source layout and skipped-sample warnings.")
    logger.info("Wrote %s samples (%d skipped) to %s", counts, total_skipped, output_dir)
    return counts


def prepare_energon_dataset(output_dir: str, *, counts: dict[str, int], num_workers: int) -> None:
    """Index non-empty splits and write the Bridge sample-loader YAML."""
    split_patterns = {}
    for split in ("train", "val"):
        if counts[split] > 0:
            split_patterns[split] = f"{split}-shard-.*"
    prepare_webdataset(output_dir, split_patterns, num_workers=num_workers)

    metadata_dir = Path(output_dir) / ".nv-meta"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / "dataset.yaml").write_text(_DATASET_YAML, encoding="utf-8")


def main() -> None:
    """Parse arguments and run conversion plus Energon indexing."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, help="Directory containing Mantis subset folders")
    parser.add_argument("--output-dir", required=True, help="Output directory for Energon shards")
    parser.add_argument("--max-samples-per-tar", type=int, default=1000, metavar="N")
    parser.add_argument("--validation-fraction", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=8, help="Workers used by Energon indexing")
    parser.add_argument(
        "--skip-energon-prepare",
        action="store_true",
        help="Write raw shards without indexing them or writing dataset.yaml.",
    )
    args = parser.parse_args()

    counts = convert(
        args.source_dir,
        args.output_dir,
        args.max_samples_per_tar,
        validation_fraction=args.validation_fraction,
    )
    if not args.skip_energon_prepare:
        prepare_energon_dataset(args.output_dir, counts=counts, num_workers=args.num_workers)
    logger.info("Done. Set dataset.path=%s", args.output_dir)


if __name__ == "__main__":
    main()
