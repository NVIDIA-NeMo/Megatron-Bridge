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

"""Audit the complete BAGEL sample WDS against its raw sources."""

import argparse
import hashlib
import json
import logging
import tarfile
from pathlib import Path

import pyarrow.parquet as pq


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse complete BAGEL WDS audit arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    """Hash one WDS tar without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_member(archive: tarfile.TarFile, expected_name: str) -> bytes:
    """Read the next member and validate deterministic tar metadata."""
    info = archive.next()
    if info is None or info.name != expected_name:
        actual = None if info is None else info.name
        raise ValueError(f"expected tar member {expected_name}, got {actual}")
    if not info.isfile() or (info.mtime, info.uid, info.gid, info.uname, info.gname) != (0, 0, 0, "", ""):
        raise ValueError(f"non-deterministic tar metadata for {expected_name}")
    stream = archive.extractfile(info)
    if stream is None:
        raise ValueError(f"cannot read tar member {expected_name}")
    return stream.read()


def check_manifest(path: Path, expected: dict[str, object]) -> None:
    """Require the manifest to equal the raw-data-derived mapping."""
    actual = json.loads(path.read_text(encoding="utf-8"))
    if actual != expected:
        raise ValueError(f"manifest differs from raw sources: {path}")


def audit_t2i(data_root: Path, dataset_root: Path) -> dict[str, object]:
    """Audit every T2I Parquet row and tar member."""
    group_dir = dataset_root / "t2i"
    tar_path = group_dir / "t2i.tar"
    parquet_paths = sorted((data_root / "t2i").glob("*.parquet"))
    samples = []
    row_counts = {}
    members = 0
    with tarfile.open(tar_path, "r:") as archive:
        for parquet_path in parquet_paths:
            parquet = pq.ParquetFile(parquet_path)
            row_counts[parquet_path.name] = []
            for row_group in range(parquet.num_row_groups):
                rows = parquet.read_row_group(row_group, columns=["image", "captions"]).to_pylist()
                row_counts[parquet_path.name].append(len(rows))
                for row_index, row in enumerate(rows):
                    source = {"parquet": parquet_path.name, "row_group": row_group, "row": row_index}
                    key = f"t2i-{parquet_path.stem}-rg{row_group}-row{row_index}"
                    if read_member(archive, f"{key}.image") != row["image"]:
                        raise ValueError(f"T2I image bytes differ at {source}")
                    metadata = json.loads(read_member(archive, f"{key}.json"))
                    expected = {
                        "dataset_group": "t2i_pretrain",
                        "dataset_name": "t2i",
                        "source": source,
                        "captions": row["captions"],
                    }
                    if metadata != expected:
                        raise ValueError(f"T2I metadata differs at {source}")
                    samples.append({"index": len(samples), "source": source})
                    members += 2
        if archive.next() is not None:
            raise ValueError("unexpected trailing T2I tar member")
    check_manifest(
        group_dir / "manifest.json",
        {
            "version": 1,
            "dataset_group": "t2i_pretrain",
            "samples": samples,
            "planning": {"parquet_paths": [path.name for path in parquet_paths], "row_counts": row_counts},
        },
    )
    return {"samples": len(samples), "members": members, "tar_sha256": file_sha256(tar_path)}


def audit_editing(data_root: Path, dataset_root: Path) -> dict[str, object]:
    """Audit every registered Editing Parquet row and image."""
    group_dir = dataset_root / "editing"
    tar_path = group_dir / "editing.tar"
    info_path = data_root / "editing" / "parquet_info" / "seedxedit_multi.json"
    parquet_info = json.loads(info_path.read_text(encoding="utf-8"))
    info_by_name = {Path(path).name: value for path, value in parquet_info.items()}
    parquet_paths = sorted((data_root / "editing" / "seedxedit_multi").glob("*.parquet"))
    samples = []
    row_groups = []
    members = 0
    with tarfile.open(tar_path, "r:") as archive:
        for parquet_path in parquet_paths:
            parquet = pq.ParquetFile(parquet_path)
            num_row_groups = info_by_name[parquet_path.name]["num_row_groups"]
            if parquet.num_row_groups != num_row_groups:
                raise ValueError(f"Editing row-group count differs for {parquet_path.name}")
            for row_group in range(num_row_groups):
                rows = parquet.read_row_group(row_group, columns=["image_list", "instruction_list"]).to_pylist()
                row_groups.append({"parquet": parquet_path.name, "row_group": row_group, "rows": len(rows)})
                for row_index, row in enumerate(rows):
                    source = {"parquet": parquet_path.name, "row_group": row_group, "row": row_index}
                    key = f"editing-{parquet_path.stem}-rg{row_group}-row{row_index}"
                    for image_index, image in enumerate(row["image_list"]):
                        if read_member(archive, f"{key}.image{image_index}") != image:
                            raise ValueError(f"Editing image bytes differ at {source}, image {image_index}")
                        members += 1
                    metadata = json.loads(read_member(archive, f"{key}.json"))
                    expected = {
                        "dataset_group": "unified_edit",
                        "dataset_name": "seedxedit_multi",
                        "source": source,
                        "instruction_list": row["instruction_list"],
                        "image_count": len(row["image_list"]),
                    }
                    if metadata != expected:
                        raise ValueError(f"Editing metadata differs at {source}")
                    samples.append({"index": len(samples), "source": source})
                    members += 1
        if archive.next() is not None:
            raise ValueError("unexpected trailing Editing tar member")
    check_manifest(
        group_dir / "manifest.json",
        {
            "version": 1,
            "dataset_group": "unified_edit",
            "samples": samples,
            "planning": {"row_groups": row_groups},
        },
    )
    return {"samples": len(samples), "members": members, "tar_sha256": file_sha256(tar_path)}


def audit_vlm(data_root: Path, dataset_root: Path) -> dict[str, object]:
    """Audit every VLM JSONL row and referenced image."""
    group_dir = dataset_root / "vlm"
    tar_path = group_dir / "vlm.tar"
    jsonl_path = data_root / "vlm" / "llava_ov_si.jsonl"
    lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    samples = []
    members = 0
    with tarfile.open(tar_path, "r:") as archive:
        for row_index, line in enumerate(lines):
            row = json.loads(line)
            if "video" in row:
                raise ValueError(f"VLM video is unsupported at row {row_index}")
            image_value = row.get("image", [])
            image_names = image_value if isinstance(image_value, list) else [image_value]
            source = {"jsonl": jsonl_path.name, "row": row_index}
            key = f"vlm-{jsonl_path.stem}-row{row_index}"
            for image_index, image_name in enumerate(image_names):
                raw_image = (data_root / "vlm" / "images" / image_name).read_bytes()
                if read_member(archive, f"{key}.image{image_index}") != raw_image:
                    raise ValueError(f"VLM image bytes differ at row {row_index}, image {image_index}")
                members += 1
            metadata = json.loads(read_member(archive, f"{key}.json"))
            expected = {
                "dataset_group": "vlm_sft",
                "dataset_name": "llava_ov",
                "source": source,
                "conversations": row["conversations"],
                "image_names": image_names,
                "image_count": len(image_names),
            }
            if metadata != expected:
                raise ValueError(f"VLM metadata differs at row {row_index}")
            samples.append({"index": len(samples), "source": source})
            members += 1
        if archive.next() is not None:
            raise ValueError("unexpected trailing VLM tar member")
    check_manifest(
        group_dir / "manifest.json",
        {
            "version": 1,
            "dataset_group": "vlm_sft",
            "samples": samples,
            "planning": {"jsonl": jsonl_path.name, "lines": lines},
        },
    )
    return {"samples": len(samples), "members": members, "tar_sha256": file_sha256(tar_path)}


def main() -> None:
    """Audit all raw records, tar members, metadata, and manifests."""
    args = parse_args()
    if args.output.exists():
        raise ValueError(f"output already exists: {args.output}")
    report = {
        "t2i_pretrain": audit_t2i(args.data_root, args.dataset_root),
        "unified_edit": audit_editing(args.data_root, args.dataset_root),
        "vlm_sft": audit_vlm(args.data_root, args.dataset_root),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    logger.info("Audited complete BAGEL sample WDS: %s", report)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
