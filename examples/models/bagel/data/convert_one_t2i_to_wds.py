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
import tarfile
from pathlib import Path

import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Convert one BAGEL T2I row to WebDataset")
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--row-group", type=int, default=0)
    parser.add_argument("--row-index", type=int, default=0)
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
    """Convert and verify one BAGEL T2I row."""
    args = parse_args()
    table = pq.ParquetFile(args.parquet).read_row_group(args.row_group, columns=["image", "captions"])
    row = table.slice(args.row_index, 1).to_pylist()[0]
    image_bytes = row["image"]
    captions = row["captions"]

    key = f"t2i-{args.parquet.stem}-rg{args.row_group}-row{args.row_index}"
    source = {
        "parquet": args.parquet.name,
        "row_group": args.row_group,
        "row": args.row_index,
    }
    metadata = {
        "dataset_group": "t2i_pretrain",
        "dataset_name": "t2i",
        "source": source,
        "captions": captions,
    }
    json_bytes = json.dumps(metadata, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    with tarfile.open(args.output, "w") as archive:
        add_bytes(archive, f"{key}.image", image_bytes)
        add_bytes(archive, f"{key}.json", json_bytes)

    with tarfile.open(args.output, "r") as archive:
        stored_image = archive.extractfile(f"{key}.image").read()
        stored_metadata = json.loads(archive.extractfile(f"{key}.json").read())
    if stored_image != image_bytes:
        raise ValueError("tar image bytes differ from the Parquet row")
    if stored_metadata["captions"] != captions:
        raise ValueError("tar captions differ from the Parquet row")
    if stored_metadata["source"] != source:
        raise ValueError("tar source position differs from the requested row")


if __name__ == "__main__":
    main()
