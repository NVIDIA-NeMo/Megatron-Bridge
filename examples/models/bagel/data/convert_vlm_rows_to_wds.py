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


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Convert BAGEL VLM rows to WebDataset")
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--num-rows", type=int, default=16)
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
    """Convert consecutive BAGEL VLM rows to deterministic WebDataset members."""
    args = parse_args()
    lines = args.jsonl.read_text(encoding="utf-8").splitlines()
    rows = lines[args.start_row : args.start_row + args.num_rows]
    if len(rows) != args.num_rows:
        raise ValueError(f"requested {args.num_rows} rows, found {len(rows)}")

    with tarfile.open(args.output, "w") as archive:
        for offset, line in enumerate(rows):
            row_index = args.start_row + offset
            row = json.loads(line)
            if "video" in row:
                raise ValueError("VLM video conversion is not part of this image-only milestone")
            image_names = row["image"] if isinstance(row["image"], list) else [row["image"]]
            key = f"vlm-{args.jsonl.stem}-row{row_index}"
            metadata = {
                "dataset_group": "vlm_sft",
                "dataset_name": "llava_ov",
                "source": {"jsonl": args.jsonl.name, "row": row_index},
                "conversations": row["conversations"],
                "image_names": image_names,
                "image_count": len(image_names),
            }
            for image_index, image_name in enumerate(image_names):
                add_bytes(archive, f"{key}.image{image_index}", (args.image_dir / image_name).read_bytes())
            add_bytes(
                archive,
                f"{key}.json",
                json.dumps(metadata, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
            )
    logger.info("Converted %d VLM rows to %s", len(rows), args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
