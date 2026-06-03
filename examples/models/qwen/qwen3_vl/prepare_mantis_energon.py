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

"""Convert TIGER-Lab/Mantis-Instruct to WebDataset shards for Energon-based QwenVL training.

All subsets are merged into a single train split.

Usage::

    python examples/models/qwen/qwen3_vl/prepare_mantis_energon.py \\
        --source-dir /path/to/mantis_instruct/Mantis-Instruct \\
        --output-dir /path/to/mantis_energon \\
        --max-samples-per-tar 1000

Source layout::

    Mantis-Instruct/
        {subset}/
            train-*.parquet   # images: [{'bytes': None, 'path': 'subdir/foo.png'}, ...]
            train_images.zip  # image files (may include subdirectories)
            ...

Output layout::

    <output-dir>/
        shard-000000.tar
        shard-000001.tar
        ...

Each shard entry: ``{subset}_{id}.jpgs`` (pickled list of raw image bytes) + ``{subset}_{id}.json``.
Images from each subset are extracted from the corresponding zip on first run (skipped on
subsequent runs via a ``.extracted_<name>`` marker file).
"""

import json
import logging
import os
import pickle
import zipfile
from argparse import ArgumentParser

import pandas as pd
import webdataset as wds
from tqdm import tqdm


logger = logging.getLogger(__name__)

_MARKER_PREFIX = ".extracted_"


def _ensure_extracted(subset_dir: str, zip_name: str) -> None:
    zip_path = os.path.join(subset_dir, zip_name)
    if not os.path.exists(zip_path):
        return
    marker = os.path.join(subset_dir, _MARKER_PREFIX + zip_name)
    if os.path.exists(marker):
        return
    logger.info("Extracting %s ...", zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        for member in tqdm(zf.namelist(), desc=f"extract {zip_name}", unit="file"):
            dest = os.path.join(subset_dir, member)
            if not os.path.exists(dest):
                zf.extract(member, subset_dir)
    open(marker, "w").close()


def convert(source_dir: str, output_dir: str, max_count: int) -> None:
    """Convert Mantis-Instruct subsets under ``source_dir`` into WebDataset shards under ``output_dir``.

    Args:
        source_dir: Path to the ``Mantis-Instruct/`` directory containing per-subset folders.
        output_dir: Destination directory for the generated ``shard-*.tar`` files.
        max_count: Maximum number of samples per output shard.
    """
    subsets = sorted(d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d)))
    if not subsets:
        raise FileNotFoundError(f"No subset directories found in {source_dir}")

    os.makedirs(output_dir, exist_ok=True)
    shard_pattern = os.path.join(output_dir, "shard-%06d.tar")
    total_written = 0
    total_skipped = 0

    with wds.ShardWriter(shard_pattern, maxcount=max_count) as sink:
        for subset in subsets:
            subset_dir = os.path.join(source_dir, subset)
            _ensure_extracted(subset_dir, "train_images.zip")

            parquet_files = sorted(
                f for f in os.listdir(subset_dir) if f.startswith("train-") and f.endswith(".parquet")
            )
            if not parquet_files:
                logger.debug("No train parquets in subset %s, skipping", subset)
                continue

            for pf in parquet_files:
                df = pd.read_parquet(os.path.join(subset_dir, pf))
                pf_stem = pf.replace(".parquet", "")
                for idx, (_, row) in enumerate(
                    tqdm(df.iterrows(), total=len(df), desc=f"{subset}/{pf}", unit="sample")
                ):
                    if row["images"] is None or len(row["images"]) == 0:
                        total_skipped += 1
                        continue

                    try:
                        imgs = [open(os.path.join(subset_dir, ref["path"]), "rb").read() for ref in row["images"]]
                    except Exception as exc:
                        logger.warning("Skipping %s/%s idx=%d: %s", subset, pf, idx, exc)
                        total_skipped += 1
                        continue

                    conversation = [dict(t) for t in row["conversation"]]
                    n_placeholders = sum(t["content"].count("<image>") for t in conversation)
                    if n_placeholders != len(imgs):
                        logger.warning(
                            "Skipping %s/%s idx=%d: %d <image> placeholders but %d images",
                            subset,
                            pf,
                            idx,
                            n_placeholders,
                            len(imgs),
                        )
                        total_skipped += 1
                        continue

                    sink.write(
                        {
                            "__key__": f"{subset}__{pf_stem}__{idx:06d}",
                            "jpgs": pickle.dumps(imgs),
                            "json": json.dumps(conversation).encode(),
                        }
                    )
                    total_written += 1

    logger.info("Wrote %d samples (%d skipped) → %s", total_written, total_skipped, output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = ArgumentParser(description="Convert Mantis-Instruct to WebDataset Energon format.")
    p.add_argument("--source-dir", required=True, help="Path to Mantis-Instruct/ directory")
    p.add_argument("--output-dir", required=True, help="Output directory for Energon shards")
    p.add_argument("--max-samples-per-tar", type=int, default=1000, metavar="N")
    args = p.parse_args()
    convert(args.source_dir, args.output_dir, args.max_samples_per_tar)
    print(f"Done. Set  dataset.path={args.output_dir}")
