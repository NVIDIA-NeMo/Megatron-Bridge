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

"""Convert official DataComp img2dataset shards into ChatML Energon data.

The official DataComp downloader writes one JPEG, caption TXT, and metadata
JSON member per successful sample. This converter preserves the caption as the
assistant target and supplies the image in a preceding user turn. The resulting
ChatML samples are directly consumable by Bridge's ``ChatMLWebdataset``.

Usage::

    uv run python tutorials/data/datacomp/prepare_datacomp_energon.py \
        --source-dir /path/to/datacomp/shards \
        --output-dir /path/to/datacomp-energon \
        --minimum-train-samples 512000 \
        --maximum-samples 525000 \
        --max-samples-per-tar 10000 \
        --validation-fraction 0.01
"""

import argparse
import hashlib
import io
import json
import logging
import re
import tarfile
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from PIL import Image


logger = logging.getLogger(__name__)
DATACOMP_REPO_ID = "mlfoundations/datacomp_1b"
DATACOMP_REVISION = "086ebeee20d4cc3b3e7c05ae703fcf278ae3a759"  # pragma: allowlist secret
DEFAULT_PROMPT = "Describe this image."
EXPECTED_RAW_EXTENSIONS = frozenset({"jpg", "json", "txt"})
DATASET_YAML = """\
__module__: megatron.bridge.data.energon.task_encoder_utils
__class__: ChatMLWebdataset
field_map:
  imgs: image.jpg
  conversation: conversation.json
subflavors: {}
"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_split(identity: str, validation_fraction: float) -> str:
    """Assign a deterministic split from an immutable sample identity."""
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1).")
    if validation_fraction == 0:
        return "train"
    bucket = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big") / 2**64
    return "val" if bucket < validation_fraction else "train"


def _build_conversation(caption: str, source_metadata: Mapping[str, object], prompt: str) -> bytes:
    payload = {
        "conversation": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": caption}],
            },
        ],
        "source": {
            "dataset": DATACOMP_REPO_ID,
            "revision": DATACOMP_REVISION,
            "uid": source_metadata["uid"],
            "url": source_metadata.get("url"),
            "source_key": source_metadata.get("key"),
            "download_status": source_metadata.get("status"),
            "original_download_sha256": source_metadata.get("sha256"),
            "face_bbox_count": len(source_metadata["face_bboxes"]),
            "width": source_metadata.get("width"),
            "height": source_metadata.get("height"),
            "original_width": source_metadata.get("original_width"),
            "original_height": source_metadata.get("original_height"),
        },
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()


def _validate_jpeg(payload: bytes) -> None:
    """Fully decode a JPEG and reject empty or non-JPEG media."""
    with Image.open(io.BytesIO(payload)) as image:
        if image.format != "JPEG":
            raise ValueError(f"Expected JPEG input, found {image.format!r}.")
        image.load()
        if image.width <= 0 or image.height <= 0:
            raise ValueError("JPEG dimensions must be positive.")


def _validate_source_metadata(source_key: str, caption: str, metadata: Mapping[str, object]) -> str:
    """Validate img2dataset metadata and return the immutable DataComp UID."""
    uid = metadata.get("uid")
    if not isinstance(uid, str) or re.fullmatch(r"[0-9a-f]{32}", uid) is None:
        raise ValueError("uid must be a 32-character lowercase hexadecimal string")
    if metadata.get("key") != source_key:
        raise ValueError("metadata key does not match the tar member key")
    if metadata.get("status") != "success":
        raise ValueError("download status must be 'success'")
    if "caption" in metadata and metadata["caption"] != caption:
        raise ValueError("metadata caption does not match the TXT caption")
    if not isinstance(metadata.get("face_bboxes"), list):
        raise ValueError("face_bboxes must be present as a list")
    return uid


def _add_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    info.mtime = 0
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    archive.addfile(info, io.BytesIO(payload))


@dataclass
class _ShardRecord:
    path: Path
    samples: int


class _DeterministicShardWriter:
    """Write fixed-count WebDataset shards with deterministic tar metadata."""

    def __init__(self, output_dir: Path, split: str, max_count: int) -> None:
        self.output_dir = output_dir
        self.split = split
        self.max_count = max_count
        self.records: list[_ShardRecord] = []
        self._archive: tarfile.TarFile | None = None
        self._path: Path | None = None
        self._count = 0

    def _open_next(self) -> None:
        self.close_current()
        self._path = self.output_dir / f"{self.split}-shard-{len(self.records):06d}.tar"
        self._archive = tarfile.open(self._path, "w", format=tarfile.GNU_FORMAT)
        self._count = 0

    def write(self, key: str, image: bytes, conversation: bytes) -> None:
        if self._archive is None or self._count >= self.max_count:
            self._open_next()
        assert self._archive is not None
        _add_bytes(self._archive, f"{key}.image.jpg", image)
        _add_bytes(self._archive, f"{key}.conversation.json", conversation)
        self._count += 1

    def close_current(self) -> None:
        if self._archive is None:
            return
        self._archive.close()
        assert self._path is not None
        self.records.append(_ShardRecord(path=self._path, samples=self._count))
        self._archive = None
        self._path = None

    def close(self) -> None:
        self.close_current()


def _iter_tar_samples(shard_path: Path) -> Iterator[tuple[str, dict[str, bytes]]]:
    """Yield grouped raw member payloads from one official img2dataset tar."""
    current_key: str | None = None
    current_sample: dict[str, bytes] = {}
    with tarfile.open(shard_path, "r") as archive:
        for member in archive:
            if not member.isfile():
                continue
            name = Path(member.name).name
            if "." not in name:
                raise ValueError(f"Unexpected member without extension in {shard_path}: {member.name}")
            key, extension = name.rsplit(".", 1)
            if current_key is not None and key != current_key:
                yield current_key, current_sample
                current_sample = {}
            current_key = key
            if extension in current_sample:
                raise ValueError(f"Duplicate {extension!r} member for sample {key!r} in {shard_path}")
            extracted: BinaryIO | None = archive.extractfile(member)
            if extracted is None:
                raise ValueError(f"Could not read member {member.name!r} from {shard_path}")
            current_sample[extension] = extracted.read()
    if current_key is not None:
        yield current_key, current_sample


def _validate_output_dir(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be new or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def convert(
    source_dir: Path,
    output_dir: Path,
    *,
    max_count: int,
    validation_fraction: float,
    minimum_train_samples: int,
    maximum_samples: int | None = None,
    prompt: str = DEFAULT_PROMPT,
) -> dict[str, object]:
    """Convert DataComp shards and return the complete preparation manifest.

    Args:
        source_dir: Directory containing official img2dataset WebDataset tars.
        output_dir: New or empty directory for converted Energon tars.
        max_count: Maximum number of samples written to one output tar.
        validation_fraction: Fraction assigned to validation by UID hash.
        minimum_train_samples: Required post-validation training count.
        maximum_samples: Optional total number of valid samples to emit.
        prompt: User text paired with each DataComp image.

    Returns:
        Manifest containing source identity, adaptation semantics, counts, and
        output-shard checksums.

    Raises:
        FileExistsError: If the output directory is not empty.
        FileNotFoundError: If no source tars exist.
        RuntimeError: If fewer than ``minimum_train_samples`` are produced.
        ValueError: If an argument is invalid.
    """
    if max_count <= 0:
        raise ValueError("max_count must be greater than zero.")
    if minimum_train_samples < 0:
        raise ValueError("minimum_train_samples must be non-negative.")
    if maximum_samples is not None and maximum_samples <= 0:
        raise ValueError("maximum_samples must be greater than zero when set.")
    if maximum_samples is not None and maximum_samples < minimum_train_samples:
        raise ValueError("maximum_samples cannot be smaller than minimum_train_samples.")
    if not prompt.strip():
        raise ValueError("prompt must not be empty.")
    _stable_split("validation", validation_fraction)

    source_shards = sorted(source_dir.glob("*.tar"))
    if not source_shards:
        raise FileNotFoundError(f"No .tar shards found under {source_dir}")
    _validate_output_dir(output_dir)

    writers = {split: _DeterministicShardWriter(output_dir, split, max_count) for split in ("train", "val")}
    counts = Counter()
    skip_reasons = Counter()
    seen_uids: set[str] = set()
    source_shards_opened = 0
    source_shard_names_opened = []

    try:
        for source_shard in source_shards:
            source_shards_opened += 1
            source_shard_names_opened.append(source_shard.name)
            logger.info("Converting %s", source_shard)
            for source_key, sample in _iter_tar_samples(source_shard):
                if maximum_samples is not None and counts["train"] + counts["val"] >= maximum_samples:
                    break
                counts["input_samples"] += 1
                missing = EXPECTED_RAW_EXTENSIONS.difference(sample)
                if missing:
                    skip_reasons[f"missing_{'_'.join(sorted(missing))}"] += 1
                    continue
                unexpected = sample.keys() - EXPECTED_RAW_EXTENSIONS
                if unexpected:
                    skip_reasons[f"unexpected_{'_'.join(sorted(unexpected))}"] += 1
                    continue
                try:
                    caption = sample["txt"].decode("utf-8")
                    metadata = json.loads(sample["json"])
                    if not isinstance(metadata, dict):
                        raise ValueError("metadata JSON must contain an object")
                    uid = _validate_source_metadata(source_key, caption, metadata)
                    if uid in seen_uids:
                        skip_reasons["duplicate_uid"] += 1
                        continue
                    if not caption.strip():
                        skip_reasons["empty_caption"] += 1
                        continue
                    _validate_jpeg(sample["jpg"])
                except (KeyError, UnicodeDecodeError, json.JSONDecodeError, OSError, ValueError) as error:
                    logger.warning("Skipping %s/%s: %s", source_shard.name, source_key, error)
                    skip_reasons["invalid_sample"] += 1
                    continue

                seen_uids.add(uid)
                split = _stable_split(uid, validation_fraction)
                writers[split].write(
                    uid,
                    sample["jpg"],
                    _build_conversation(caption, metadata, prompt),
                )
                counts[split] += 1
            if maximum_samples is not None and counts["train"] + counts["val"] >= maximum_samples:
                break
    finally:
        for writer in writers.values():
            writer.close()

    output_shards = []
    for split, writer in writers.items():
        for record in writer.records:
            output_shards.append(
                {
                    "filename": record.path.name,
                    "split": split,
                    "samples": record.samples,
                    "size_bytes": record.path.stat().st_size,
                    "sha256": _sha256(record.path),
                }
            )

    manifest = {
        "source": {
            "repo_id": DATACOMP_REPO_ID,
            "revision": DATACOMP_REVISION,
            "raw_shard_count_available": len(source_shards),
            "raw_shard_count_opened": source_shards_opened,
            "raw_shards_opened": source_shard_names_opened,
        },
        "adaptation": {
            "objective": "causal image-conditioned caption generation",
            "prompt": prompt,
            "assistant_target": "original DataComp caption",
            "loss_mask": "assistant tokens only; applied by the selected ChatML VLM collator",
            "validation_fraction": validation_fraction,
            "split_identity": "DataComp uid",
            "split_hash": "first 64 bits of SHA-256, big endian",
            "selection_order": "lexically sorted raw tar filename, then tar member order",
        },
        "counts": {
            "input_samples": counts["input_samples"],
            "train": counts["train"],
            "val": counts["val"],
            "skipped": sum(skip_reasons.values()),
            "skip_reasons": dict(sorted(skip_reasons.items())),
        },
        "minimum_train_samples": minimum_train_samples,
        "maximum_samples": maximum_samples,
        "accepted": counts["train"] >= minimum_train_samples,
        "output_shards": output_shards,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not manifest["accepted"]:
        raise RuntimeError(
            f"Only {counts['train']} training samples were produced; require at least {minimum_train_samples}."
        )
    return manifest


def prepare_energon_dataset(output_dir: Path, *, counts: Mapping[str, int], num_workers: int) -> None:
    """Index non-empty splits and write Bridge's ChatML loader declaration.

    Args:
        output_dir: Directory containing converted train/validation tars.
        counts: Converted sample counts keyed by split name.
        num_workers: Parallel workers used by Energon's indexing pass.
    """
    from megatron.bridge.data.energon import prepare_webdataset

    split_patterns = {}
    if counts["train"] > 0:
        split_patterns["train"] = "train-shard-.*"
    if counts["val"] > 0:
        split_patterns["val"] = "val-shard-.*"
    if not split_patterns:
        raise ValueError("At least one converted split must contain samples.")
    prepare_webdataset(output_dir, split_patterns, num_workers=num_workers)
    metadata_dir = output_dir / ".nv-meta"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / "dataset.yaml").write_text(DATASET_YAML, encoding="utf-8")


def main() -> None:
    """Parse arguments, convert DataComp, and index the Energon dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples-per-tar", type=int, default=10_000, metavar="N")
    parser.add_argument("--validation-fraction", type=float, default=0.01)
    parser.add_argument("--minimum-train-samples", type=int, default=0)
    parser.add_argument("--maximum-samples", type=int)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--skip-energon-prepare",
        action="store_true",
        help="Write converted shards and manifest without Energon indexing.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    manifest = convert(
        args.source_dir,
        args.output_dir,
        max_count=args.max_samples_per_tar,
        validation_fraction=args.validation_fraction,
        minimum_train_samples=args.minimum_train_samples,
        maximum_samples=args.maximum_samples,
        prompt=args.prompt,
    )
    if not args.skip_energon_prepare:
        prepare_energon_dataset(
            args.output_dir,
            counts=manifest["counts"],
            num_workers=args.num_workers,
        )
    logger.info(
        "Prepared %d train and %d validation samples at %s",
        manifest["counts"]["train"],
        manifest["counts"]["val"],
        args.output_dir,
    )


if __name__ == "__main__":
    main()
