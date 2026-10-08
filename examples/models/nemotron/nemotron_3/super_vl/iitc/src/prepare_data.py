# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rebuild the frozen IITC sample selection from the published JSONL files."""

import argparse
import csv
import hashlib
import io
import json
import logging
import pickle
import re
import tarfile
import zipfile
from pathlib import Path


DATASET = "zhourax977/VEGA"
REVISION = "417bd55ab869a872cef9ae4f12a9b2073f619ac4"
FIELDS = ("id", "title", "context", "question", "answer", "image_paths", "truth_fig_idx")
PROMPT = (
    "Based on the following known information, answer the user's questions concisely and professionally. "
    "If the answer is derived from an image, please indicate in the answer by referencing [Picture *], "
    "where * represents the image number. If the answer cannot be obtained from it, ignore the content "
    "of the text and respond to the user's question in Chinese. Known content: "
)
YAML = """__module__: megatron.bridge.data.energon.task_encoder_utils
__class__: ChatMLWebdataset
field_map:
  imgs: jpgs
  conversation: json
subflavors: {}
"""


def fingerprint(row: dict) -> str:
    """Hash the fields used by training and evaluation independently of JSON spacing."""
    payload = json.dumps({k: row[k] for k in FIELDS}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def selection(source: Path, manifest: Path) -> dict[str, list[dict]]:
    """Read only frozen zero-based source rows; fail if their content changed."""
    with manifest.open() as stream:
        entries = list(csv.DictReader(stream))
    keys = [(entry["source_split"], int(entry["source_row"])) for entry in entries]
    assert len(entries) == len(set(keys)) == 10210
    found = {}
    for split in ("4k", "8k"):
        wanted = {index for part, index in keys if part == split}
        with (source / f"IITC_{split}_train.json").open() as stream:
            for index, line in enumerate(stream):
                if index in wanted:
                    found[split, index] = json.loads(line)
    result = {"train": [], "validation": []}
    for entry, key in zip(entries, keys):
        row = found[key]
        assert fingerprint(row) == entry["content_sha256"], key
        result[entry["partition"]].append({**row, "source_split": key[0], "source_row": key[1]})
    assert len(result["train"]) == 10000 and len(result["validation"]) == 210
    return result


def messages(row: dict) -> list[dict]:
    """Keep each image at its published position among the text segments."""
    content = [{"type": "text", "text": PROMPT}]
    paths = []
    for part in re.split(r"(<img>.*?</img>)", row["context"]):
        if part.startswith("<img>"):
            paths.append(part[5:-6])
            content.append({"type": "image"})
        elif part:
            content.append({"type": "text", "text": part})
    assert paths == row["image_paths"]
    content.append({"type": "text", "text": "\nQuestion:" + row["question"]})
    return [
        {"role": "user", "content": content},
        {"role": "assistant", "content": [{"type": "text", "text": row["answer"]}]},
    ]


def extract_images(archive: Path, images: Path, paths: set[str]) -> None:
    """Extract requested image files only, rejecting paths outside the destination."""
    with zipfile.ZipFile(archive) as stream:
        names = set(stream.namelist())
        for relative in sorted(paths):
            target = (images / relative).resolve()
            if not target.is_relative_to(images.resolve()):
                raise ValueError(f"Unsafe image path: {relative}")
            member = relative if relative in names else "imgs/" + relative
            if member not in names:
                raise ValueError(f"Image missing from archive: {relative}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(stream.read(member))


def package(rows: dict[str, list[dict]], *, images: Path, output: Path) -> None:
    """Write the same 100-sample ChatML shards used by the reference run."""
    output.mkdir(parents=True, exist_ok=False)
    for partition, samples in rows.items():
        split = "val" if partition == "validation" else "train"
        for start in range(0, len(samples), 100):
            path = output / f"{split}-shard-{start // 100:05d}.tar"
            with tarfile.open(path, "w") as archive:
                for row in samples[start : start + 100]:
                    key = f"iitc-{row['source_split']}-{row['source_row']}"
                    payloads = {
                        "jpgs": pickle.dumps([(images / p).read_bytes() for p in row["image_paths"]], protocol=4),
                        "json": json.dumps(messages(row), ensure_ascii=False).encode(),
                    }
                    for extension, payload in payloads.items():
                        member = tarfile.TarInfo(f"{key}.{extension}")
                        member.size = len(payload)
                        member.mtime = 0
                        archive.addfile(member, io.BytesIO(payload))
            logging.info("Packaged %s", path.name)
    from megatron.bridge.data.energon import prepare_webdataset

    prepare_webdataset(output, {"train": "train-shard-.*", "val": "val-shard-.*"}, num_workers=4)
    (output / ".nv-meta/dataset.yaml").write_text(YAML)


def main() -> None:
    """Download pinned sources optionally, validate selection, and write Energon shards."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("selection.csv"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--check-only", action="store_true", help="Check selected source rows without reading images")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.download:
        from huggingface_hub import hf_hub_download

        args.source.mkdir(parents=True, exist_ok=True)
        for split in ("4k_train", "8k_train", "4k_test", "8k_test"):
            name = f"IITC_{split}.json"
            cached = hf_hub_download(DATASET, f"datas/{name}", repo_type="dataset", revision=REVISION)
            # Stream large JSONL sources instead of reading them into memory.
            import shutil

            shutil.copyfile(cached, args.source / name)
    rows = selection(args.source, args.manifest)
    logging.info("Verified frozen selection: 10,000 training and 210 validation rows")
    if args.check_only:
        return
    if args.download:
        test = [json.loads(line) for line in (args.source / "IITC_8k_test.json").read_text().splitlines() if line]
        assert len(test) == 658
        paths = {p for row in rows["train"] + rows["validation"] + test for p in row["image_paths"]}
        archive = hf_hub_download(DATASET, "imgs.zip", repo_type="dataset", revision=REVISION)
        extract_images(Path(archive), args.images, paths)
    package(rows, images=args.images, output=args.output)
    (args.output / "selection-manifest.json").write_text(
        json.dumps(
            {
                "dataset": DATASET,
                "revision": REVISION,
                "selection_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                "train": len(rows["train"]),
                "validation": len(rows["validation"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
