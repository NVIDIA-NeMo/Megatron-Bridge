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
    """Parse complete BAGEL sample conversion arguments."""
    parser = argparse.ArgumentParser(description="Convert the complete BAGEL sample dataset to deterministic WDS")
    parser.add_argument("--data-root", type=Path, required=True)
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


def write_manifest(group_dir: Path, dataset_group: str, samples: list[dict[str, object]], **planning: object) -> None:
    """Write the raw-data-derived source-to-WDS-index mapping."""
    manifest = {"version": 1, "dataset_group": dataset_group, "samples": samples, "planning": planning}
    (group_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def convert_t2i(data_root: Path, output: Path) -> int:
    """Convert every T2I Parquet row in canonical file/row order."""
    group_dir = output / "t2i"
    group_dir.mkdir()
    samples = []
    row_counts = {}
    parquet_paths = sorted((data_root / "t2i").glob("*.parquet"))
    with tarfile.open(group_dir / "t2i.tar", "w") as archive:
        for parquet_path in parquet_paths:
            parquet = pq.ParquetFile(parquet_path)
            row_counts[parquet_path.name] = []
            for row_group in range(parquet.num_row_groups):
                rows = parquet.read_row_group(row_group, columns=["image", "captions"]).to_pylist()
                row_counts[parquet_path.name].append(len(rows))
                for row_index, row in enumerate(rows):
                    source = {"parquet": parquet_path.name, "row_group": row_group, "row": row_index}
                    key = f"t2i-{parquet_path.stem}-rg{row_group}-row{row_index}"
                    metadata = {
                        "dataset_group": "t2i_pretrain",
                        "dataset_name": "t2i",
                        "source": source,
                        "captions": row["captions"],
                    }
                    add_bytes(archive, f"{key}.image", row["image"])
                    add_bytes(
                        archive,
                        f"{key}.json",
                        json.dumps(metadata, ensure_ascii=False, separators=(",", ":")).encode(),
                    )
                    samples.append({"index": len(samples), "source": source})
    write_manifest(
        group_dir,
        "t2i_pretrain",
        samples,
        parquet_paths=[path.name for path in parquet_paths],
        row_counts=row_counts,
    )
    return len(samples)


def convert_editing(data_root: Path, output: Path) -> int:
    """Convert every registered Editing row group in canonical order."""
    group_dir = output / "editing"
    group_dir.mkdir()
    info_path = data_root / "editing" / "parquet_info" / "seedxedit_multi.json"
    parquet_info = json.loads(info_path.read_text(encoding="utf-8"))
    info_by_name = {Path(path).name: value for path, value in parquet_info.items()}
    samples = []
    row_groups = []
    parquet_paths = sorted((data_root / "editing" / "seedxedit_multi").glob("*.parquet"))
    with tarfile.open(group_dir / "editing.tar", "w") as archive:
        for parquet_path in parquet_paths:
            parquet = pq.ParquetFile(parquet_path)
            num_row_groups = info_by_name[parquet_path.name]["num_row_groups"]
            if parquet.num_row_groups != num_row_groups:
                raise ValueError(f"row-group metadata differs for {parquet_path.name}")
            for row_group in range(num_row_groups):
                rows = parquet.read_row_group(row_group, columns=["image_list", "instruction_list"]).to_pylist()
                row_groups.append({"parquet": parquet_path.name, "row_group": row_group, "rows": len(rows)})
                for row_index, row in enumerate(rows):
                    source = {"parquet": parquet_path.name, "row_group": row_group, "row": row_index}
                    key = f"editing-{parquet_path.stem}-rg{row_group}-row{row_index}"
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
                        json.dumps(metadata, ensure_ascii=False, separators=(",", ":")).encode(),
                    )
                    samples.append({"index": len(samples), "source": source})
    write_manifest(group_dir, "unified_edit", samples, row_groups=row_groups)
    return len(samples)


def convert_vlm(data_root: Path, output: Path) -> int:
    """Convert every VLM JSONL row in original line order."""
    group_dir = output / "vlm"
    group_dir.mkdir()
    jsonl_path = data_root / "vlm" / "llava_ov_si.jsonl"
    lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    samples = []
    with tarfile.open(group_dir / "vlm.tar", "w") as archive:
        for row_index, line in enumerate(lines):
            row = json.loads(line)
            if "video" in row:
                raise ValueError("VLM video conversion is not supported")
            image_value = row.get("image", [])
            image_names = image_value if isinstance(image_value, list) else [image_value]
            source = {"jsonl": jsonl_path.name, "row": row_index}
            key = f"vlm-{jsonl_path.stem}-row{row_index}"
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
                    archive, f"{key}.image{image_index}", (data_root / "vlm" / "images" / image_name).read_bytes()
                )
            add_bytes(
                archive,
                f"{key}.json",
                json.dumps(metadata, ensure_ascii=False, separators=(",", ":")).encode(),
            )
            samples.append({"index": len(samples), "source": source})
    write_manifest(group_dir, "vlm_sft", samples, jsonl=jsonl_path.name, lines=lines)
    return len(samples)


def main() -> None:
    """Convert the complete official sample without sampling or topology inputs."""
    args = parse_args()
    if args.output.exists():
        raise ValueError(f"output already exists: {args.output}")
    args.output.mkdir(parents=True)
    counts = {
        "t2i_pretrain": convert_t2i(args.data_root, args.output),
        "unified_edit": convert_editing(args.data_root, args.output),
        "vlm_sft": convert_vlm(args.data_root, args.output),
    }
    logger.info("Converted complete BAGEL sample: %s", counts)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
