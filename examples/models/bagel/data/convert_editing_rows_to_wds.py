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
import io
import json
import logging
import tarfile
from pathlib import Path

import pyarrow.parquet as pq


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Convert BAGEL Editing rows to WebDataset")
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--row-group", type=int, default=0)
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--num-rows", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def add_bytes(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    """Add bytes with deterministic tar metadata."""
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    archive.addfile(info, io.BytesIO(data))


def main() -> None:
    """Convert consecutive BAGEL Editing rows to deterministic WebDataset members."""
    args = parse_args()
    table = pq.ParquetFile(args.parquet).read_row_group(
        args.row_group,
        columns=["image_list", "instruction_list"],
    )
    rows = table.slice(args.start_row, args.num_rows).to_pylist()
    if len(rows) != args.num_rows:
        raise ValueError(f"requested {args.num_rows} rows, found {len(rows)}")

    with tarfile.open(args.output, "w") as archive:
        for offset, row in enumerate(rows):
            row_index = args.start_row + offset
            key = f"editing-{args.parquet.stem}-rg{args.row_group}-row{row_index}"
            metadata = {
                "dataset_group": "unified_edit",
                "dataset_name": "seedxedit_multi",
                "source": {"parquet": args.parquet.name, "row_group": args.row_group, "row": row_index},
                "instruction_list": row["instruction_list"],
                "image_count": len(row["image_list"]),
            }
            for image_index, image in enumerate(row["image_list"]):
                add_bytes(archive, f"{key}.image{image_index}", image)
            add_bytes(
                archive,
                f"{key}.json",
                json.dumps(metadata, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
            )
    logger.info("Converted %d Editing rows to %s", len(rows), args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
