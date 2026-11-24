#!/usr/bin/env python3
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

"""Rebuild the twelve public source rows consumed by the 256K ERNIE benchmark."""

import argparse
import hashlib
import json
import random
import tempfile
from itertools import islice
from pathlib import Path

import pyarrow.parquet as pq
from tokenizers import Tokenizer


TULU = (
    "allenai/tulu-3-sft-mixture",
    "b14afda60f1bbebe55d5d2fa1e4df5042f97f8be",
    "data/train-00000-of-00006.parquet",
    "41d4524345443ccca4202df2f01d84d84371afcde36f637178e0e2bcd14f79c7",
)
FINEWEB = (
    "HuggingFaceFW/fineweb-edu",
    "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
    "sample/10BT/013_00000.parquet",
    "b393f51fefab26cd6f4c8f65707c1924f6666c4961a0ebebe04bb57f7ec832de",
)
TOKENIZER_SHA256 = "20e7c81f36fdfb90db664583fc6426d5c2ac6d6bcaf631c910941a8d8f30f2d9"
ROWS_SHA256 = "5fb767243f8369b0a0c9150103848bf7be22846325b78dfe272ef9a43373d513"
ROW_LENGTH = 262_144


def verify(path: Path, expected: str) -> None:
    """Verify the bytes of a pinned input or its derived output."""
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"{path} has SHA-256 {actual}, expected {expected}")


def parquet_path(local_path: Path | None, source: tuple[str, str, str, str]) -> Path:
    """Resolve a local or Hub shard and verify its contents."""
    repo, revision, filename, sha256 = source
    if local_path is None:
        from huggingface_hub import hf_hub_download

        local_path = Path(hf_hub_download(repo, filename, repo_type="dataset", revision=revision))
    verify(local_path, sha256)
    return local_path


def token_stream(path: Path, tokenizer: Tokenizer, column: str):
    """Yield shuffled tokens with EOS between source documents."""
    table = pq.read_table(path, columns=[column])
    order = list(range(table.num_rows))
    random.Random(1234).shuffle(order)
    for start in range(0, len(order), 256):
        texts = table.take(order[start : start + 256]).column(column).to_pylist()
        if column == "messages":
            texts = [
                "\n".join(
                    f"{m.get('role', '')}: {m.get('content', '')}" for m in messages or [] if isinstance(m, dict)
                )
                for messages in texts
            ]
        texts = [text for text in texts if isinstance(text, str) and text.strip()]
        for encoded in tokenizer.encode_batch(texts, add_special_tokens=False):
            yield from encoded.ids
            yield 2  # EOS between source documents


def main() -> None:
    """Build twelve 256K rows and check their combined digest before publishing them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True, help="Pinned ERNIE-5.5-Flash-Base tokenizer.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tulu", type=Path, help="Use the validated local Tulu parquet shard")
    parser.add_argument("--fineweb", type=Path, help="Use the validated local FineWeb-Edu parquet shard")
    args = parser.parse_args()
    verify(args.tokenizer, TOKENIZER_SHA256)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    tulu = token_stream(parquet_path(args.tulu, TULU), tokenizer, "messages")
    web = token_stream(parquet_path(args.fineweb, FINEWEB), tokenizer, "text")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temp_dir:
        output = Path(temp_dir) / "rows.jsonl"
        with output.open("wb") as rows:
            for index in range(12):
                tokens = tulu if (index // 2) % 2 == 0 else web
                ids = [1, *islice(tokens, ROW_LENGTH - 1)]  # BOS at the start of each row
                if len(ids) != ROW_LENGTH:
                    raise ValueError("The source shard was exhausted")
                rows.write((json.dumps({"input_ids": ids, "labels": ids}) + "\n").encode())
        verify(output, ROWS_SHA256)
        output.replace(args.output)
    print(f"{ROWS_SHA256}  {args.output}")


if __name__ == "__main__":
    main()
