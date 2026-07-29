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

"""Download and validate the pinned four-file DataComp metadata slice."""

import argparse
import hashlib
import json
import logging
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


logger = logging.getLogger(__name__)
DATACOMP_REPO_ID = "mlfoundations/datacomp_1b"
DATACOMP_REVISION = "086ebeee20d4cc3b3e7c05ae703fcf278ae3a759"  # pragma: allowlist secret
EXPECTED_FILES = {
    "0035af9f90f581816acf269df5eb37ad.parquet": (
        532_229,
        130_506_429,
        "e3633f90e78b827c8b667c88b8a1dce542e72feacc85be9e27f4706ed71fe1ce",  # pragma: allowlist secret
    ),
    "003da708d909c8cab24c7dcf4d04c371.parquet": (
        517_671,
        126_593_324,
        "5d2d4b0adc840b23dd9bbca04ed351f7904a6346b310326d0b34256ea1b8b0a8",  # pragma: allowlist secret
    ),
    "00818e301428c0573aac33fb4c1b5f02.parquet": (
        542_935,
        132_871_668,
        "d3bb081586d8dcf1da4883a37becb57fa18759ad83a2a1e484528719f42be047",  # pragma: allowlist secret
    ),
    "00aa8e74b038faf4d69ac89e84a318ba.parquet": (
        540_499,
        132_277_520,
        "9138d1c135e9b3452e3273f8bdf10b95c55e86b7007234c70d8cd36a12441bc4",  # pragma: allowlist secret
    ),
}
REQUIRED_COLUMNS = {"uid", "url", "text", "face_bboxes"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_metadata(output_dir: Path) -> dict[str, object]:
    """Download the pinned metadata files, validate them, and write a manifest.

    Args:
        output_dir: Directory that receives the Parquet files and manifest.

    Returns:
        The validated metadata manifest.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    unexpected_files = sorted(path.name for path in output_dir.glob("*.parquet") if path.name not in EXPECTED_FILES)
    if unexpected_files:
        raise RuntimeError(f"Metadata directory contains unexpected Parquet files: {', '.join(unexpected_files)}.")

    records = []
    for filename, (expected_rows, expected_size, expected_sha256) in sorted(EXPECTED_FILES.items()):
        path = Path(
            hf_hub_download(
                repo_id=DATACOMP_REPO_ID,
                filename=filename,
                repo_type="dataset",
                revision=DATACOMP_REVISION,
                local_dir=output_dir,
                local_dir_use_symlinks=False,
                resume_download=True,
            )
        )
        digest = _sha256(path)
        parquet = pq.ParquetFile(path)
        columns = parquet.schema_arrow.names
        if parquet.metadata.num_rows != expected_rows:
            raise RuntimeError(f"{filename} has {parquet.metadata.num_rows} rows; expected {expected_rows}.")
        if path.stat().st_size != expected_size:
            raise RuntimeError(f"{filename} has {path.stat().st_size} bytes; expected {expected_size}.")
        if digest != expected_sha256:
            raise RuntimeError(f"{filename} SHA-256 is {digest}; expected {expected_sha256}.")
        if not REQUIRED_COLUMNS <= set(columns):
            missing = ", ".join(sorted(REQUIRED_COLUMNS - set(columns)))
            raise RuntimeError(f"{filename} is missing required columns: {missing}.")
        records.append(
            {
                "filename": filename,
                "rows": expected_rows,
                "bytes": expected_size,
                "sha256": digest,
                "columns": columns,
            }
        )

    manifest: dict[str, object] = {
        "repo_id": DATACOMP_REPO_ID,
        "revision": DATACOMP_REVISION,
        "files": records,
    }
    (output_dir / "metadata-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    """Parse arguments and download the selected DataComp metadata."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    download_metadata(args.output_dir)
    logger.info("Validated %d metadata files in %s", len(EXPECTED_FILES), args.output_dir)


if __name__ == "__main__":
    main()
