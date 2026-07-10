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

"""Prepare small MedPix-VQA slices as an indexed Qwen-VL Energon dataset."""

import argparse
import io
import json
import logging
import tarfile
from pathlib import Path

from datasets import load_dataset
from PIL import Image

from megatron.bridge.data.energon import prepare_webdataset


logger = logging.getLogger(__name__)
DATASET_ID = "mmoukouba/MedPix-VQA"
DATASET_YAML = """\
__module__: megatron.bridge.data.energon.task_encoder_utils
__class__: ChatMLWebdataset
field_map:
  imgs: image.png
  conversation: conversation.json
subflavors: {}
"""


def _add_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    info.mtime = 0
    archive.addfile(info, io.BytesIO(payload))


def _encode_png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def _build_conversation(question: str, answer: str) -> bytes:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": answer}]},
    ]
    return json.dumps(messages, ensure_ascii=False).encode("utf-8")


def _write_split(output_dir: Path, split: str, rows: list[dict]) -> None:
    shard_path = output_dir / f"{split}-shard-000000.tar"
    with tarfile.open(shard_path, "w") as archive:
        for index, row in enumerate(rows):
            image = row["image_id"]
            if not isinstance(image, Image.Image):
                raise TypeError(f"Expected image_id to decode as PIL.Image.Image, got {type(image).__name__}")
            key = f"{split}-{index:06d}"
            _add_bytes(archive, f"{key}.image.png", _encode_png(image))
            _add_bytes(
                archive,
                f"{key}.conversation.json",
                _build_conversation(str(row["question"]), str(row["answer"])),
            )


def _run_energon_prepare(output_dir: Path, *, num_workers: int) -> None:
    prepare_webdataset(
        output_dir,
        {"train": "train-shard-.*", "val": "val-shard-.*"},
        num_workers=num_workers,
    )


def prepare_medpix_data(
    output_dir: Path,
    *,
    train_rows: int = 16,
    validation_rows: int = 8,
    run_prepare: bool = True,
    num_workers: int = 2,
) -> None:
    """Write fixed MedPix slices, index the shards, and declare the sample loader.

    Args:
        output_dir: Directory that will contain the tar shards and Energon metadata.
        train_rows: Number of rows selected from the beginning of the training split.
        validation_rows: Number of rows selected from the beginning of the validation split.
        run_prepare: Whether to index the shards after writing them.
        num_workers: Worker count passed to Energon's preparation API.

    Raises:
        ValueError: If either requested row count is not positive.
        TypeError: If a MedPix ``image_id`` value does not decode as a PIL image.
    """
    if train_rows <= 0 or validation_rows <= 0:
        raise ValueError("train_rows and validation_rows must be greater than zero.")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_split = f"train[:{train_rows}]"
    validation_split = f"validation[:{validation_rows}]"
    train_examples = list(load_dataset(DATASET_ID, split=train_split))
    validation_examples = list(load_dataset(DATASET_ID, split=validation_split))
    _write_split(output_dir, "train", train_examples)
    _write_split(output_dir, "val", validation_examples)

    if run_prepare:
        _run_energon_prepare(output_dir, num_workers=num_workers)

    metadata_dir = output_dir / ".nv-meta"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / "dataset.yaml").write_text(DATASET_YAML, encoding="utf-8")
    manifest = {
        "dataset": DATASET_ID,
        "train_split": train_split,
        "validation_split": validation_split,
        "train_rows": len(train_examples),
        "validation_rows": len(validation_examples),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "MedPix Energon data is ready at %s (train=%d, validation=%d)",
        output_dir,
        len(train_examples),
        len(validation_examples),
    )


def main() -> None:
    """Parse command-line arguments and prepare the MedPix dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-rows", type=int, default=16)
    parser.add_argument("--validation-rows", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--skip-prepare", action="store_true", help="Write raw shards without Energon indexes.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    prepare_medpix_data(
        args.output_dir,
        train_rows=args.train_rows,
        validation_rows=args.validation_rows,
        run_prepare=not args.skip_prepare,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
