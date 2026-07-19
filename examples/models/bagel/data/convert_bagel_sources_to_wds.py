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
    """Parse bounded source-list conversion arguments."""
    parser = argparse.ArgumentParser(description="Convert ordered BAGEL source IDs to WebDataset")
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset-group", choices=("t2i_pretrain", "unified_edit", "vlm_sft"), required=True)
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
    """Write one ordered group tar from canonical BAGEL source IDs."""
    args = parse_args()
    with args.sources.open(encoding="utf-8") as stream:
        source_ids = json.load(stream)[args.dataset_group]

    vlm_lines = None
    if args.dataset_group == "vlm_sft":
        vlm_lines = (args.data_root / "vlm" / "llava_ov_si.jsonl").read_text(encoding="utf-8").splitlines()

    cached_location = None
    cached_rows = None
    with tarfile.open(args.output, "w") as archive:
        for source_id in source_ids:
            if source_id["dataset_name"] != args.dataset_group:
                raise ValueError("source ID dataset group mismatch")
            source = source_id["source"]

            if args.dataset_group == "vlm_sft":
                row_index = source["row"]
                row = json.loads(vlm_lines[row_index])
                if "video" in row:
                    raise ValueError("VLM video conversion is not supported")
                image_names = row["image"] if isinstance(row["image"], list) else [row["image"]]
                key = f"vlm-llava_ov_si-row{row_index}"
                metadata = {
                    "dataset_group": "vlm_sft",
                    "dataset_name": "llava_ov",
                    "source": source,
                    "conversations": row["conversations"],
                    "image_names": image_names,
                    "image_count": len(image_names),
                }
                for image_index, image_name in enumerate(image_names):
                    add_bytes(
                        archive,
                        f"{key}.image{image_index}",
                        (args.data_root / "vlm" / "images" / image_name).read_bytes(),
                    )
            else:
                parquet_dir = args.data_root / (
                    "t2i" if args.dataset_group == "t2i_pretrain" else "editing/seedxedit_multi"
                )
                parquet_path = parquet_dir / source["parquet"]
                location = (parquet_path, source["row_group"])
                if location != cached_location:
                    columns = (
                        ["image", "captions"]
                        if args.dataset_group == "t2i_pretrain"
                        else ["image_list", "instruction_list"]
                    )
                    cached_rows = (
                        pq.ParquetFile(parquet_path).read_row_group(source["row_group"], columns=columns).to_pylist()
                    )
                    cached_location = location
                row = cached_rows[source["row"]]
                if args.dataset_group == "t2i_pretrain":
                    key = f"t2i-{parquet_path.stem}-rg{source['row_group']}-row{source['row']}"
                    metadata = {
                        "dataset_group": "t2i_pretrain",
                        "dataset_name": "t2i",
                        "source": source,
                        "captions": row["captions"],
                    }
                    add_bytes(archive, f"{key}.image", row["image"])
                else:
                    key = f"editing-{parquet_path.stem}-rg{source['row_group']}-row{source['row']}"
                    metadata = {
                        "dataset_group": "unified_edit",
                        "dataset_name": "seedxedit_multi",
                        "source": source,
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
    logger.info("Converted %d %s sources to %s", len(source_ids), args.dataset_group, args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
