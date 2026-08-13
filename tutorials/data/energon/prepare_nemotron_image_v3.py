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

"""Convert selected Nemotron-Image-Training-v3 subsets to indexed Energon shards.

The Hugging Face dataset stores conversations in one JSONL file per subset and
media separately, either as local files or tar archives. This helper joins both
parts into Bridge's ``ChatMLWebdataset`` contract without downloading the full
dataset. See ``nemotron-image-v3.md`` for a pinned, self-contained ``turing``
example.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import pickle
import re
import tarfile
from contextlib import ExitStack
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

import webdataset as wds

from megatron.bridge.data.energon import prepare_webdataset


logger = logging.getLogger(__name__)
DATASET_ID = "nvidia/Nemotron-Image-Training-v3"
DATASET_REVISION = "7656391d4d4cb11ec3722b34f10d499435de0460"  # pragma: allowlist secret
PINNED_SUBSET_FILES = {
    "turing": {
        "turing/turing.jsonl": (
            3_357_202,
            "c7333e5765ba8ac3ad6dd823c97a0366cc682417cc7ebd6d5374d9f9db1ce3d5",  # pragma: allowlist secret
        ),
        "turing/media/shard_000000.tar": (
            33_269_760,
            "d3ee86c9116dce94e7e3a2c2747fda41a8ba74085a4b123d95f104021b2d3196",  # pragma: allowlist secret
        ),
    },
}
DATASET_YAML = """\
__module__: megatron.bridge.data.energon.task_encoder_utils
__class__: ChatMLWebdataset
field_map:
  imgs: jpgs
  conversation: json
subflavors: {}
"""
_SAFE_KEY = re.compile(r"[^A-Za-z0-9_-]+")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_pinned_subset_files(source_dir: Path, subsets: tuple[str, ...]) -> list[dict[str, object]]:
    """Verify known self-contained subset payloads from the pinned dataset revision."""
    verified_files: list[dict[str, object]] = []
    for subset in subsets:
        expected_files = PINNED_SUBSET_FILES.get(subset)
        if not expected_files:
            raise ValueError(
                f"No pinned source-integrity manifest is available for subset {subset!r}. "
                "Use --skip-source-integrity-check only after verifying its provenance yourself."
            )
        for relative_name, (expected_size, expected_sha256) in expected_files.items():
            path = source_dir / relative_name
            if not path.is_file():
                raise FileNotFoundError(f"Pinned source file is missing: {path}")
            size = path.stat().st_size
            if size != expected_size:
                raise RuntimeError(f"{relative_name} has {size} bytes; expected {expected_size}.")
            digest = _sha256(path)
            if digest != expected_sha256:
                raise RuntimeError(f"{relative_name} SHA-256 is {digest}; expected {expected_sha256}.")
            verified_files.append(
                {
                    "path": relative_name,
                    "size": size,
                    "sha256": digest,
                }
            )
    return verified_files


class MediaResolver:
    """Resolve subset-local media paths from files or downloaded tar archives."""

    def __init__(
        self,
        subset_dir: Path,
        *,
        allow_loose_media: bool = True,
        archive_paths: tuple[Path, ...] | None = None,
    ) -> None:
        self.subset_dir = subset_dir.resolve()
        self.allow_loose_media = allow_loose_media
        self._stack = ExitStack()
        self._members: dict[str, tuple[tarfile.TarFile, tarfile.TarInfo] | None] = {}

        archives = (
            sorted(archive_paths) if archive_paths is not None else sorted((self.subset_dir / "media").rglob("*.tar"))
        )
        try:
            for archive_path in archives:
                if not archive_path.resolve().is_relative_to(self.subset_dir):
                    raise ValueError(f"Media archive is outside subset directory {self.subset_dir}: {archive_path}")
                archive = self._stack.enter_context(tarfile.open(archive_path))
                for member in archive.getmembers():
                    if not member.isfile():
                        continue
                    normalized = self._normalize_reference(member.name)
                    self._register(normalized, archive, member)
                    self._register(PurePosixPath(normalized).name, archive, member)
        except BaseException:
            self._stack.close()
            raise

    @staticmethod
    def _normalize_reference(reference: str) -> str:
        normalized = PurePosixPath(reference.replace("\\", "/")).as_posix()
        while normalized.startswith("./"):
            normalized = normalized[2:]
        return normalized

    def _register(self, key: str, archive: tarfile.TarFile, member: tarfile.TarInfo) -> None:
        location = (archive, member)
        if key not in self._members:
            self._members[key] = location
        elif self._members[key] != location:
            self._members[key] = None

    def close(self) -> None:
        """Close all opened media archives."""
        self._stack.close()

    def __enter__(self) -> MediaResolver:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def read(self, reference: str) -> bytes:
        """Read one media reference from the subset tree or its media tars."""
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError("Image content must contain a non-empty image reference.")
        if urlparse(reference).scheme:
            raise FileNotFoundError(
                f"Remote media reference {reference!r} is not downloaded. Follow the subset README and place the "
                "media under the selected subset before conversion."
            )

        normalized = self._normalize_reference(reference)
        if self.allow_loose_media:
            for relative_path in (normalized, f"media/{normalized}"):
                candidate = (self.subset_dir / relative_path).resolve()
                if candidate.is_relative_to(self.subset_dir) and candidate.is_file():
                    return candidate.read_bytes()

        for key in (normalized, PurePosixPath(normalized).name):
            if key not in self._members:
                continue
            location = self._members[key]
            if location is None:
                raise ValueError(
                    f"Media reference {reference!r} is ambiguous across downloaded archives; use a unique relative path."
                )
            archive, member = location
            extracted = archive.extractfile(member)
            if extracted is None:
                break
            return extracted.read()

        raise FileNotFoundError(
            f"Could not resolve image {reference!r} under {self.subset_dir}. "
            "Download or lay out the subset media as described by its README."
        )


def _make_shard_writer(output_dir: Path, split: str, max_samples_per_tar: int) -> wds.ShardWriter:
    """Create the standard WebDataset writer with deterministic tar metadata."""
    return wds.ShardWriter(
        str(output_dir / f"{split}-shard-%06d.tar"),
        maxcount=max_samples_per_tar,
        maxsize=float("inf"),
        verbose=0,
        encoder=False,
        mtime=0,
        mode=0o644,
        user="",
        group="",
    )


def _write_sample(
    writer: wds.ShardWriter,
    key: str,
    images: list[bytes],
    messages: list[dict[str, Any]],
) -> None:
    writer.write(
        {
            "__key__": key,
            "jpgs": pickle.dumps(images, protocol=4),
            "json": json.dumps(messages, ensure_ascii=False).encode("utf-8"),
        }
    )


def _sample_split(key: str, validation_fraction: float) -> str:
    digest = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")
    return "val" if digest / 2**64 < validation_fraction else "train"


def _sample_key(subset: str, sample_id: str) -> str:
    normalized = _SAFE_KEY.sub("_", f"{subset}__{sample_id}").strip("_")
    if not normalized:
        raise ValueError(f"Unable to derive a WebDataset key from subset={subset!r}, id={sample_id!r}.")
    return normalized


def _normalize_messages(
    messages: Any,
    *,
    resolver: MediaResolver,
    sample_name: str,
) -> tuple[list[dict[str, Any]], list[bytes]]:
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{sample_name}: messages must be a non-empty list.")

    normalized_messages: list[dict[str, Any]] = []
    images: list[bytes] = []
    for turn_index, turn in enumerate(messages):
        if not isinstance(turn, dict) or not isinstance(turn.get("role"), str):
            raise ValueError(f"{sample_name}: message {turn_index} must contain a string role.")
        raw_content = turn.get("content")
        parts = raw_content if isinstance(raw_content, list) else [raw_content]
        normalized_parts: list[dict[str, Any]] = []
        for part_index, part in enumerate(parts):
            if isinstance(part, str):
                normalized_parts.append({"type": "text", "text": part})
                continue
            if not isinstance(part, dict):
                raise ValueError(
                    f"{sample_name}: message {turn_index} content {part_index} must be a string or dictionary."
                )

            part_type = part.get("type")
            if part_type == "text":
                text = part.get("text")
                if not isinstance(text, str):
                    raise ValueError(f"{sample_name}: text content must contain a string text field.")
                normalized_parts.append({"type": "text", "text": text})
            elif part_type == "image":
                image_reference = part.get("image")
                if not isinstance(image_reference, str) or not image_reference.strip():
                    raise ValueError(f"{sample_name}: image content must contain a non-empty image reference.")
                images.append(resolver.read(image_reference))
                normalized_parts.append({"type": "image"})
            elif part_type in {"video", "audio"}:
                raise ValueError(
                    f"{sample_name}: {part_type} content is outside this Qwen3-VL image tutorial; "
                    "select an image-only subset or extend the media writer explicitly."
                )
            else:
                raise ValueError(f"{sample_name}: unsupported content type {part_type!r}.")

        if not normalized_parts:
            raise ValueError(f"{sample_name}: message {turn_index} has no content.")
        normalized_messages.append({"role": turn["role"], "content": normalized_parts})

    if not images:
        raise ValueError(f"{sample_name}: no image content was found.")
    return normalized_messages, images


def _jsonl_path(subset_dir: Path, subset: str) -> Path:
    canonical = subset_dir / f"{subset}.jsonl"
    if canonical.is_file():
        return canonical
    candidates = sorted(subset_dir.glob("*.jsonl"))
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(f"Expected {canonical} or exactly one JSONL file under {subset_dir}.")


def _validate_output_dir(output_dir: Path) -> None:
    existing = sorted(output_dir.iterdir())
    if existing:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"Output directory is not empty ({names}); choose a fresh directory.")


def convert_subsets(
    source_dir: Path,
    output_dir: Path,
    *,
    subsets: tuple[str, ...],
    validation_fraction: float = 0.05,
    max_samples_per_tar: int = 1000,
    max_samples: int | None = None,
    allow_loose_media: bool = True,
    media_archives: dict[str, tuple[Path, ...]] | None = None,
) -> dict[str, int]:
    """Convert selected local subsets into raw Bridge ChatML WebDataset shards."""
    if not subsets or any(not subset.strip() for subset in subsets):
        raise ValueError("subsets must contain at least one non-empty subset name.")
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1).")
    if max_samples_per_tar <= 0:
        raise ValueError("max_samples_per_tar must be greater than zero.")
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be greater than zero when set.")

    source_dir = source_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _validate_output_dir(output_dir)
    counts = {"train": 0, "val": 0}
    seen_keys: set[str] = set()
    total = 0

    writers: dict[str, wds.ShardWriter] = {}
    with ExitStack() as writer_stack:
        for subset in subsets:
            subset_dir = (source_dir / subset).resolve()
            if not subset_dir.is_relative_to(source_dir) or not subset_dir.is_dir():
                raise FileNotFoundError(f"Downloaded subset directory does not exist: {subset_dir}")
            jsonl_path = _jsonl_path(subset_dir, subset)
            with (
                MediaResolver(
                    subset_dir,
                    allow_loose_media=allow_loose_media,
                    archive_paths=None if media_archives is None else media_archives.get(subset, ()),
                ) as resolver,
                jsonl_path.open(encoding="utf-8") as jsonl_file,
            ):
                for line_number, line in enumerate(jsonl_file, start=1):
                    if max_samples is not None and total >= max_samples:
                        break
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    sample_id = row.get("id")
                    if not isinstance(sample_id, str) or not sample_id:
                        raise ValueError(f"{jsonl_path}:{line_number}: row id must be a non-empty string.")
                    key = _sample_key(subset, sample_id)
                    if key in seen_keys:
                        raise ValueError(f"{jsonl_path}:{line_number}: duplicate normalized sample key {key!r}.")
                    seen_keys.add(key)
                    messages, images = _normalize_messages(
                        row.get("messages"),
                        resolver=resolver,
                        sample_name=f"{jsonl_path}:{line_number}",
                    )
                    split = _sample_split(key, validation_fraction)
                    if split not in writers:
                        writers[split] = writer_stack.enter_context(
                            _make_shard_writer(output_dir, split, max_samples_per_tar)
                        )
                    _write_sample(writers[split], key, images, messages)
                    counts[split] += 1
                    total += 1
            if max_samples is not None and total >= max_samples:
                break

    if counts["train"] == 0:
        raise RuntimeError("No training samples were written.")
    if validation_fraction > 0 and counts["val"] == 0:
        raise RuntimeError(
            "No validation samples were selected. Increase --max-samples, increase --validation-fraction, "
            "or set --validation-fraction 0 and disable validation in the training command."
        )
    logger.info("Wrote %d train and %d validation samples to %s", counts["train"], counts["val"], output_dir)
    return counts


def prepare_nemotron_image_v3(
    source_dir: Path,
    output_dir: Path,
    *,
    subsets: tuple[str, ...] = ("turing",),
    validation_fraction: float = 0.05,
    max_samples_per_tar: int = 1000,
    max_samples: int | None = None,
    num_workers: int = 2,
    run_prepare: bool = True,
    verify_source_integrity: bool = True,
) -> dict[str, int]:
    """Convert selected subsets, optionally index them, and write dataset metadata."""
    source_dir = source_dir.resolve()
    verified_files = _verify_pinned_subset_files(source_dir, subsets) if verify_source_integrity else []
    verified_media_archives = (
        {
            subset: tuple(
                source_dir / relative_name
                for relative_name in PINNED_SUBSET_FILES[subset]
                if Path(relative_name).suffix == ".tar"
            )
            for subset in subsets
        }
        if verify_source_integrity
        else None
    )
    counts = convert_subsets(
        source_dir,
        output_dir,
        subsets=subsets,
        validation_fraction=validation_fraction,
        max_samples_per_tar=max_samples_per_tar,
        max_samples=max_samples,
        allow_loose_media=not verify_source_integrity,
        media_archives=verified_media_archives,
    )
    split_patterns = {split: f"{split}-shard-.*" for split, count in counts.items() if count > 0}
    if run_prepare:
        prepare_webdataset(output_dir, split_patterns, num_workers=num_workers)

    metadata_dir = output_dir / ".nv-meta"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / "dataset.yaml").write_text(DATASET_YAML, encoding="utf-8")
    manifest = {
        "dataset": DATASET_ID,
        "revision": DATASET_REVISION,
        "subsets": list(subsets),
        "validation_fraction": validation_fraction,
        "max_samples": max_samples,
        "counts": counts,
        "source_integrity_verified": verify_source_integrity,
        "verified_source_files": verified_files,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logger.info("Nemotron Image v3 Energon dataset is ready at %s", output_dir)
    return counts


def main() -> None:
    """Parse arguments and run selected-subset conversion plus Energon indexing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True, help="Root created by the pinned hf download")
    parser.add_argument("--output-dir", type=Path, required=True, help="Fresh output directory for Energon shards")
    parser.add_argument("--subsets", nargs="+", default=["turing"], help="Downloaded subset directory names")
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--max-samples-per-tar", type=int, default=1000)
    parser.add_argument("--max-samples", type=int, default=None, help="Optional global cap across selected subsets")
    parser.add_argument("--num-workers", type=int, default=2, help="Workers used by Energon indexing")
    parser.add_argument("--skip-energon-prepare", action="store_true", help="Write raw shards without Energon indexes")
    parser.add_argument(
        "--skip-source-integrity-check",
        action="store_true",
        help="Allow unpinned subsets and loose media after verifying their provenance yourself",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    prepare_nemotron_image_v3(
        args.source_dir,
        args.output_dir,
        subsets=tuple(args.subsets),
        validation_fraction=args.validation_fraction,
        max_samples_per_tar=args.max_samples_per_tar,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
        run_prepare=not args.skip_energon_prepare,
        verify_source_integrity=not args.skip_source_integrity_check,
    )


if __name__ == "__main__":
    main()
