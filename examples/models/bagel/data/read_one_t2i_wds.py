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

import argparse
import hashlib
import json
import logging
from pathlib import Path

import pyarrow.parquet as pq
import webdataset as wds


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Verify one BAGEL T2I WebDataset sample")
    parser.add_argument("--tar", type=Path, required=True)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--row-group", type=int, default=0)
    parser.add_argument("--row-index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    """Read and verify one BAGEL T2I WebDataset sample."""
    args = parse_args()
    table = pq.ParquetFile(args.parquet).read_row_group(args.row_group, columns=["image", "captions"])
    row = table.slice(args.row_index, 1).to_pylist()[0]

    expected_key = f"t2i-{args.parquet.stem}-rg{args.row_group}-row{args.row_index}"
    expected_metadata = {
        "dataset_group": "t2i_pretrain",
        "dataset_name": "t2i",
        "source": {
            "parquet": args.parquet.name,
            "row_group": args.row_group,
            "row": args.row_index,
        },
        "captions": row["captions"],
    }

    samples = list(wds.WebDataset(str(args.tar), shardshuffle=False))
    if len(samples) != 1:
        raise ValueError(f"expected one sample, found {len(samples)}")
    sample = samples[0]
    metadata = json.loads(sample["json"])
    if sample["__key__"] != expected_key:
        raise ValueError("WebDataset sample key differs from the source row")
    if sample["image"] != row["image"]:
        raise ValueError("WebDataset image bytes differ from the source row")
    if metadata != expected_metadata:
        raise ValueError("WebDataset JSON metadata differs from the source row")

    result = {
        "key": sample["__key__"],
        "image_sha256": hashlib.sha256(sample["image"]).hexdigest(),
        "metadata": metadata,
    }
    logger.info("%s", json.dumps(result, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
