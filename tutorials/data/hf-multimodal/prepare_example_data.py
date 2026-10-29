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

"""Create tiny local-image JSONL splits for the Hugging Face multimodal tutorial."""

import argparse
import json
import struct
import zlib
from pathlib import Path


IMAGE_SIZE = 448
EXAMPLES = {
    "training": [
        ("red.png", (220, 40, 40), "What is the dominant color?", "The image is red."),
        ("green.png", (40, 180, 80), "What is the dominant color?", "The image is green."),
        ("blue.png", (50, 90, 220), "What is the dominant color?", "The image is blue."),
        ("yellow.png", (230, 200, 40), "What is the dominant color?", "The image is yellow."),
    ],
    "validation": [
        ("purple.png", (150, 70, 190), "What is the dominant color?", "The image is purple."),
    ],
    "test": [
        ("orange.png", (230, 120, 30), "What is the dominant color?", "The image is orange."),
    ],
}


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


def build_qwen_row(image_path: Path, question: str, answer: str) -> dict[str, object]:
    """Build one processor-native Qwen-VL chat row."""
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path.resolve())},
                    {"type": "text", "text": question},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ]
    }


def prepare_example_data(output_dir: Path) -> None:
    """Write images and JSONL train/validation/test splits under ``output_dir``."""
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    for split, examples in EXAMPLES.items():
        split_path = output_dir / f"{split}.jsonl"
        with split_path.open("w", encoding="utf-8") as output_file:
            for filename, color, question, answer in examples:
                image_path = image_dir / filename
                image_path.write_bytes(make_solid_png(color))
                output_file.write(json.dumps(build_qwen_row(image_path, question, answer)) + "\n")


def main() -> None:
    """Parse arguments and create the tutorial dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    prepare_example_data(args.output_dir)


if __name__ == "__main__":
    main()
