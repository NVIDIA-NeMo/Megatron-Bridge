# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Select an illustrative IITC subset and package it for Bridge training."""

import argparse
import hashlib
import heapq
import io
import json
import logging
import pickle
import re
import tarfile
import zipfile
from pathlib import Path


DATASET = "zhourax977/VEGA"
# Published dataset commit, not a credential.
REVISION = "417bd55ab869a872cef9ae4f12a9b2073f619ac4"  # pragma: allowlist secret
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


def paper_ids(row: dict) -> set[str]:
    """Group versions of the same source paper together."""
    return {re.sub(r"v\d+$", "", Path(path).parts[1]) for path in row["image_paths"]}


def selection(source: Path, *, seed: int, train_size: int, val_size: int) -> dict[str, list[dict]]:
    """Hash-rank candidates, reserving separate papers for validation and test."""
    test_papers = set()
    for split in ("4k", "8k"):
        for line in (source / f"IITC_{split}_test.json").read_text().splitlines():
            if line.strip():
                test_papers.update(paper_ids(json.loads(line)))
    heaps = {"train": [], "validation": []}
    capacities = {"train": train_size * 2, "validation": val_size * 2}
    for split in ("4k", "8k"):
        with (source / f"IITC_{split}_train.json").open() as stream:
            for index, line in enumerate(stream):
                if not line.strip():
                    continue
                row = json.loads(line)
                papers = paper_ids(row)
                if len(papers) != 1 or papers & test_papers:
                    continue
                if re.findall(r"<img>(.*?)</img>", row["context"]) != row["image_paths"]:
                    continue
                paper = next(iter(papers))
                group = int(hashlib.sha256(f"{seed}:{paper}".encode()).hexdigest(), 16)
                partition = "validation" if group % 20 == 0 else "train"
                rank = int(hashlib.sha256(f"{seed}:{split}:{index}".encode()).hexdigest(), 16)
                item = (-rank, split, index, {**row, "source_split": split, "source_row": index})
                heap = heaps[partition]
                if len(heap) < capacities[partition]:
                    heapq.heappush(heap, item)
                elif item[:3] > heap[0][:3]:
                    heapq.heapreplace(heap, item)
    return {part: [item[3] for item in sorted(heap, reverse=True)] for part, heap in heaps.items()}


def filter_length(rows: list[dict], *, images: Path, processor: object, count: int, max_tokens: int) -> list[dict]:
    """Keep complete examples that fit the native training collator's token limit."""
    from PIL import Image

    from megatron.bridge.models.nemotron_omni.data.collate_fn import nemotron_omni_expanded_collate_fn

    selected = []
    seen = set()
    for row in rows:
        key = (tuple(sorted(paper_ids(row))), " ".join(row["question"].lower().split()))
        if key in seen:
            continue
        pictures = []
        try:
            for path in row["image_paths"]:
                with Image.open(images / path) as image:
                    pictures.append(image.convert("RGB"))
            conversation = messages(row)
            picture_iter = iter(pictures)
            for turn in conversation:
                for item in turn["content"]:
                    if item["type"] == "image":
                        item["image"] = next(picture_iter)
            batch = nemotron_omni_expanded_collate_fn(
                [{"conversation": conversation}],
                processor,
                sequence_length=None,
            )
        finally:
            for image in pictures:
                image.close()
        if batch["input_ids"].shape[1] > max_tokens:
            continue
        selected.append(row)
        seen.add(key)
        if len(selected) == count:
            return selected
    raise ValueError(f"Only {len(selected)} eligible examples; request fewer samples or increase the candidate pool")


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
    """Write 100-sample ChatML shards."""
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
    """Download sources, select complete examples, and write Energon shards."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--processor", required=True, help="HF processor name or path for token-length filtering")
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--train-size", type=int, default=10000)
    parser.add_argument("--val-size", type=int, default=210)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--download", action="store_true")
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
    if min(args.train_size, args.val_size, args.max_tokens) <= 0:
        parser.error("Sample counts and token limit must be positive")
    rows = selection(args.source, seed=args.seed, train_size=args.train_size, val_size=args.val_size)
    if args.download:
        test = [json.loads(line) for line in (args.source / "IITC_8k_test.json").read_text().splitlines() if line]
        assert len(test) == 658
        paths = {p for row in rows["train"] + rows["validation"] + test for p in row["image_paths"]}
        archive = hf_hub_download(DATASET, "imgs.zip", repo_type="dataset", revision=REVISION)
        extract_images(Path(archive), args.images, paths)
    import torch
    from transformers import AutoProcessor

    torch.set_num_threads(4)
    processor = AutoProcessor.from_pretrained(args.processor, trust_remote_code=True)
    rows = {
        part: filter_length(samples, images=args.images, processor=processor, count=count, max_tokens=args.max_tokens)
        for (part, samples), count in zip(rows.items(), (args.train_size, args.val_size))
    }
    package(rows, images=args.images, output=args.output)
    (args.output / "selection-manifest.json").write_text(
        json.dumps(
            {
                "dataset": DATASET,
                "revision": REVISION,
                "seed": args.seed,
                "max_tokens": args.max_tokens,
                "train": len(rows["train"]),
                "validation": len(rows["validation"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
