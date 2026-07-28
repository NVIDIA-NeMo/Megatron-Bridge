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

"""Create and index a tiny Qwen-VL Energon/WebDataset tutorial dataset."""

import argparse
import io
import json
import logging
import struct
import tarfile
import zlib
from pathlib import Path

from megatron.bridge.data.energon import prepare_webdataset


logger = logging.getLogger(__name__)
IMAGE_SIZE = 448
EXAMPLES = {
    "train": [
        ((220, 40, 40), "What is the dominant color?", "The image is red."),
        ((40, 180, 80), "What is the dominant color?", "The image is green."),
        ((50, 90, 220), "What is the dominant color?", "The image is blue."),
        ((230, 200, 40), "What is the dominant color?", "The image is yellow."),
    ],
    "val": [
        ((150, 70, 190), "What is the dominant color?", "The image is purple."),
        ((230, 120, 30), "What is the dominant color?", "The image is orange."),
    ],
}
DATASET_YAML = """\
__module__: megatron.bridge.data.energon.task_encoder_utils
__class__: ChatMLWebdataset
field_map:
  imgs: image.png
  conversation: conversation.json
subflavors: {}
"""


def make_solid_png(color: tuple[int, int, int], *, size: int = IMAGE_SIZE) -> bytes:
    """Return a deterministic RGB PNG containing one solid color."""

    def _chunk(name: bytes, payload: bytes) -> bytes:
        checksum = zlib.crc32(name + payload) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + name + payload + struct.pack(">I", checksum)

    row = b"\x00" + bytes(color) * size
    pixels = row * size
    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", header) + _chunk(b"IDAT", zlib.compress(pixels)) + _chunk(b"IEND", b"")
    )


def build_conversation(question: str, answer: str) -> bytes:
    """Serialize one ChatML conversation whose image is supplied by the shard."""
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": answer}]},
    ]
    return json.dumps(conversation).encode()


def _add_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    info.mtime = 0
    archive.addfile(info, io.BytesIO(payload))


def build_shards(output_dir: Path) -> None:
    """Write deterministic train and validation WebDataset tar shards."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, examples in EXAMPLES.items():
        shard_path = output_dir / f"{split}-shard-000000.tar"
        with tarfile.open(shard_path, "w") as archive:
            for index, (color, question, answer) in enumerate(examples):
                key = f"{split}-{index:06d}"
                _add_bytes(archive, f"{key}.image.png", make_solid_png(color))
                _add_bytes(archive, f"{key}.conversation.json", build_conversation(question, answer))


def run_energon_prepare(output_dir: Path, *, num_workers: int) -> None:
    """Index the tutorial splits through Energon's version-stable API."""
    prepare_webdataset(
        output_dir,
        {"train": "train-shard-.*", "val": "val-shard-.*"},
        num_workers=num_workers,
    )


def write_dataset_yaml(output_dir: Path) -> None:
    """Declare Bridge's ChatML sample loader after Energon creates metadata."""
    metadata_dir = output_dir / ".nv-meta"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / "dataset.yaml").write_text(DATASET_YAML, encoding="utf-8")


def prepare_example_data(output_dir: Path, *, run_prepare: bool = True, num_workers: int = 2) -> None:
    """Build tutorial shards, optionally index them, and write ``dataset.yaml``."""
    build_shards(output_dir)
    if run_prepare:
        run_energon_prepare(output_dir, num_workers=num_workers)
    write_dataset_yaml(output_dir)
    logger.info("Energon tutorial dataset is ready at %s", output_dir)


def main() -> None:
    """Parse arguments and create the tutorial dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Write shards and dataset.yaml without indexing the shards (useful for inspecting raw output).",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    prepare_example_data(args.output_dir, run_prepare=not args.skip_prepare, num_workers=args.num_workers)


if __name__ == "__main__":
    main()
