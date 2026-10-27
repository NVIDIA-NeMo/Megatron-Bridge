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

"""Mock variable-length SFT data shaped like the agentic ``mixed`` length distribution.

Each row is a prompt and a completion of space-separated token IDs. A tiny local word-level tokenizer, in which token
``n`` is the string ``"n"``, reads them back as exactly those IDs, so row lengths equal the intended token counts with no
tokenizer download and no model-dependent rendering.

The lengths follow the ``mixed`` selection of the original Megatron-LM agentic SFT example by Zhongbo Zhu: sample counts
split 50% / 30% / 20% across the minimum-CP buckets (a sample needing at most CP1, CP2 or CP4 capacity), with a length
drawn uniformly inside each bucket. This reproduces that example's reported token shares of about 19% / 36% / 45%.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np


BUCKET_RATIOS = (0.5, 0.3, 0.2)
COMPLETION_FRACTION = 0.2
MANIFEST_NAME = "manifest.json"
TOKENIZER_DIR = "tokenizer"
SPLITS = ("training", "validation")
# The last three IDs are the pad, BOS and EOS special tokens; data uses IDs 1 .. vocab_size - 4.
NUM_SPECIAL_TOKENS = 3


def bucket_bounds(seq_length: int) -> list[tuple[int, int]]:
    """Return the inclusive runtime-length range of each minimum-CP bucket (CP1, CP2, CP4).

    Raises:
        ValueError: If ``seq_length`` is not a positive multiple of 32.
    """
    if seq_length < 32 or seq_length % 32:
        raise ValueError("seq_length must be a positive multiple of 32")
    quarter, half = seq_length // 4, seq_length // 2
    return [(1, quarter), (quarter + 1, half), (half + 1, seq_length)]


def bucket_counts(count: int, ratios: Sequence[float] = BUCKET_RATIOS) -> list[int]:
    """Split ``count`` samples over the buckets with largest remainders, so the total is exact.

    Raises:
        ValueError: If ``count`` is negative or the ratios are not three non-negative values summing to 1.
    """
    if count < 0:
        raise ValueError("count must be non-negative")
    if len(ratios) != 3 or any(value < 0 for value in ratios) or not math.isclose(sum(ratios), 1.0, abs_tol=1e-8):
        raise ValueError("ratios must be three non-negative values summing to 1")
    raw = [count * value for value in ratios]
    counts = [math.floor(value) for value in raw]
    for bucket in sorted(range(3), key=lambda i: (-(raw[i] - counts[i]), i))[: count - sum(counts)]:
        counts[bucket] += 1
    return counts


def runtime_lengths(count: int, seq_length: int, seed: int, ratios: Sequence[float] = BUCKET_RATIOS) -> np.ndarray:
    """Draw ``count`` runtime lengths (tokens after the single next-token shift), in a seeded shuffled order."""
    rng = np.random.default_rng(seed)
    lengths = np.concatenate(
        [
            rng.integers(low, high + 1, size=size)
            for (low, high), size in zip(bucket_bounds(seq_length), bucket_counts(count, ratios))
        ]
        or [np.empty(0, dtype=np.int64)]
    )
    rng.shuffle(lengths)
    return lengths.astype(np.int64)


def write_mock_tokenizer(path: str | os.PathLike[str], vocab_size: int) -> None:
    """Save a word-level Hugging Face tokenizer that maps the string ``"n"`` to token ID ``n``.

    IDs ``0 .. vocab_size - 4`` are numbers, followed by pad, BOS and EOS tokens, so the vocabulary has exactly
    ``vocab_size`` entries and ``vocab_size - 1`` is the EOS ID.

    Raises:
        ValueError: If ``vocab_size`` is too small to hold the special tokens.
    """
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    if vocab_size <= NUM_SPECIAL_TOKENS + 1:
        raise ValueError("vocab_size is too small")
    numbers = vocab_size - NUM_SPECIAL_TOKENS
    pad, bos, eos = "<|pad|>", "<|bos|>", "<|eos|>"
    vocab = {str(index): index for index in range(numbers)}
    vocab.update({pad: numbers, bos: numbers + 1, eos: numbers + 2})
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token=pad))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    wrapped = PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token=pad, bos_token=bos, eos_token=eos)
    wrapped.save_pretrained(os.fspath(path))


def _row(length: int, rng: np.random.Generator, vocab_size: int) -> str:
    """Build one JSONL row holding ``length + 1`` tokens, so the shifted sequence has ``length`` positions."""
    total = length + 1
    completion = min(max(1, round(total * COMPLETION_FRACTION)), total - 1)
    # Avoid ID 0 and the special tokens at the end of the vocabulary.
    tokens = rng.integers(1, vocab_size - NUM_SPECIAL_TOKENS, size=total)
    prompt_text = " ".join(map(str, tokens[: total - completion].tolist()))
    completion_text = " ".join(map(str, tokens[total - completion :].tolist()))
    return json.dumps({"input": prompt_text, "output": completion_text})


def _identity(seq_length: int, counts: dict[str, int], vocab_size: int, seed: int) -> dict[str, Any]:
    return {
        "generator": "mock_agentic_data/v1",
        "seq_length": seq_length,
        "counts": counts,
        "vocab_size": vocab_size,
        "seed": seed,
        "bucket_ratios": list(BUCKET_RATIOS),
        "completion_fraction": COMPLETION_FRACTION,
    }


def _write_split(path: Path, lengths: np.ndarray, vocab_size: int, seed: int) -> str:
    rng = np.random.default_rng(seed)
    digest = hashlib.sha256()
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as stream:
        temporary = Path(stream.name)
        for length in lengths:
            line = _row(int(length), rng, vocab_size) + "\n"
            digest.update(line.encode("utf-8"))
            stream.write(line)
    os.replace(temporary, path)
    return digest.hexdigest()


def write_mock_dataset(
    root: str | os.PathLike[str], *, seq_length: int, num_train: int, num_validation: int, vocab_size: int, seed: int
) -> dict[str, Any]:
    """Write the rows, the tokenizer that reads them, and a manifest under ``root``.

    Files are ``training.jsonl`` (and ``validation.jsonl`` when requested), ``tokenizer/`` and ``manifest.json``.

    The result is deterministic in its arguments, so every caller produces identical rows. An existing dataset with the
    same identity is reused.

    Args:
        root: Output directory.
        seq_length: Maximum runtime sequence length; bounds the three length buckets.
        num_train: Training rows.
        num_validation: Validation rows; 0 writes no validation file.
        vocab_size: Token IDs are drawn from ``[1, vocab_size - 4]``; the last three IDs are special tokens.
        seed: Seed for lengths, order and token IDs.

    Returns:
        The manifest, including each split's runtime lengths in file order.

    Raises:
        ValueError: If a count is negative.
    """
    if num_train < 1 or num_validation < 0:
        raise ValueError("num_train must be positive and num_validation non-negative")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    counts = {"training": num_train, "validation": num_validation}
    identity = _identity(seq_length, counts, vocab_size, seed)
    manifest_path = root / MANIFEST_NAME
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text())
        if (
            previous.get("identity") == identity
            and (root / TOKENIZER_DIR / "tokenizer.json").is_file()
            and all((root / f"{split}.jsonl").is_file() for split in SPLITS if counts[split])
        ):
            return previous
    tokenizer_dir = root / TOKENIZER_DIR
    staging = Path(tempfile.mkdtemp(prefix=f".{TOKENIZER_DIR}.", dir=root))
    write_mock_tokenizer(staging, vocab_size)
    shutil.rmtree(tokenizer_dir, ignore_errors=True)
    os.replace(staging, tokenizer_dir)
    manifest: dict[str, Any] = {"identity": identity, "tokenizer": TOKENIZER_DIR, "splits": {}}
    for index, split in enumerate(SPLITS):
        if not counts[split]:
            (root / f"{split}.jsonl").unlink(missing_ok=True)
            continue
        lengths = runtime_lengths(counts[split], seq_length, seed + 1000 * index)
        digest = _write_split(root / f"{split}.jsonl", lengths, vocab_size, seed + 1000 * index + 1)
        manifest["splits"][split] = {"file": f"{split}.jsonl", "sha256": digest, "runtime_lengths": lengths.tolist()}
    temporary = root / f".{MANIFEST_NAME}.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(manifest) + "\n")
    os.replace(temporary, manifest_path)
    return manifest


def ensure_mock_dataset(
    root: str | os.PathLike[str],
    *,
    seq_length: int,
    num_train: int,
    num_validation: int,
    vocab_size: int,
    seed: int,
    timeout_s: float = 1800.0,
) -> dict[str, Any]:
    """Create the dataset once per node and let the other local ranks wait for it.

    Under ``torchrun`` every rank calls this. Local rank 0 writes; the others poll for a manifest matching the requested
    dataset. Outside ``torchrun`` the caller writes directly.

    Raises:
        TimeoutError: If a waiting rank never sees the finished dataset.
    """
    arguments = {
        "seq_length": seq_length,
        "num_train": num_train,
        "num_validation": num_validation,
        "vocab_size": vocab_size,
        "seed": seed,
    }
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        return write_mock_dataset(root, **arguments)
    expected = _identity(seq_length, {"training": num_train, "validation": num_validation}, vocab_size, seed)
    manifest_path = Path(root) / MANIFEST_NAME
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("identity") == expected:
                return manifest
        time.sleep(2.0)
    raise TimeoutError(f"Mock dataset was not written under {root} within {timeout_s:.0f} s")
