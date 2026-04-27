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

"""Prepare Valor32k-AVQA v2.0 dataset for Nemotron Omni finetuning.

Downloads QA annotations, extracts audio from user-supplied videos, and
organizes into::

    output_dir/
    ├── videos/                                  # 10s MP4 clips (user-supplied)
    ├── audio/                                   # 16 kHz mono WAV (extracted)
    ├── combined_dataset_train_flattened.json
    ├── combined_dataset_val_flattened.json
    └── combined_dataset_test_flattened.json

Video source: VALOR-32K (AudioSet 10-second clips). Videos must be downloaded
manually from YouTube or BaiduPan and placed in ``output_dir/videos/`` before running
this script. See instructions in https://github.com/CASIA-IVA-Lab/VALOR for more details.

Prerequisites:
    pip install huggingface_hub tqdm
    # ffmpeg must be on PATH for audio extraction (apt-get install -y ffmpeg)

Usage:
    # 1. Download the VALOR-32K clips from YouTube or BaiduPan and place the *.mp4
    #    files in /data/valor32k_avqa/videos/ (a subset is fine, QA pairs without
    #    a matching video will simply be reported as "not usable")
    # 2. Run this script to fetch QA annotations and extract audio:
    python examples/models/vlm/nemotron_3_omni/data/prepare_valor32k_avqa.py \\
        --output_dir /data/valor32k_avqa

    # Optional: limit the number of videos processed (useful for testing):
    python examples/models/vlm/nemotron_3_omni/data/prepare_valor32k_avqa.py \\
        --output_dir /data/valor32k_avqa --max_videos 100
"""

import argparse
import json
import logging
import shutil
import subprocess
import zipfile
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen


try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

QA_ZIP_URL = "https://github.com/inesriahi/valor32k-avqa-2/raw/refs/heads/main/data.zip"
VALOR_ANNOTATIONS_URL = "https://casia-iva-group.github.io/projects/VALOR/static/files/VALOR-32K-annotations.zip"

def download_qa_annotations(output_dir: Path):
    """Download and extract QA JSON files from the GitHub repo."""
    logger.info(f"Downloading QA annotations from {QA_ZIP_URL}...")
    resp = urlopen(QA_ZIP_URL)
    with zipfile.ZipFile(BytesIO(resp.read())) as zf:
        for member in zf.namelist():
            if member.endswith(".json"):
                filename = Path(member).name
                target = output_dir / filename
                if not target.exists():
                    with zf.open(member) as src, open(target, "wb") as dst:
                        dst.write(src.read())
                    logger.info(f"  Extracted {filename}")
                else:
                    logger.info(f"  Already exists: {filename}")


def collect_video_ids(output_dir: Path) -> set[str]:
    """Collect unique video IDs from all QA JSON files."""
    video_ids = set()
    for json_file in output_dir.glob("combined_dataset_*_flattened.json"):
        with open(json_file) as f:
            data = json.load(f)
        for qa in data:
            video_ids.add(str(qa["video_id"]))
    return video_ids


def extract_audio(video_path: Path, audio_path: Path, target_sr: int = 16000) -> tuple[bool, str]:
    """Extract audio from video as 16 kHz mono WAV.

    Returns:
        (success, error_message). ``error_message`` is empty on success.
    """
    if audio_path.exists():
        return True, ""
    cmd = [
        "ffmpeg",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(target_sr),
        "-ac",
        "1",
        str(audio_path),
        "-y",
        "-loglevel",
        "error",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except FileNotFoundError:
        return False, "ffmpeg executable not found on PATH"
    except subprocess.TimeoutExpired:
        return False, "ffmpeg timed out after 60s"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    if result.returncode != 0:
        return False, (result.stderr or result.stdout or "ffmpeg exited non-zero with no output").strip()
    if not audio_path.exists():
        return False, "ffmpeg reported success but output file was not created"
    return True, ""


def main():
    """Prepare the Valor32k-AVQA v2.0 dataset from user-supplied videos."""
    parser = argparse.ArgumentParser(description="Prepare Valor32k-AVQA v2.0 dataset")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--max_videos",
        type=int,
        default=None,
        help="Limit the number of videos to process (useful for testing with a subset).",
    )
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg is not on PATH. Install it (e.g. `apt-get install -y ffmpeg` or "
            "`conda install -c conda-forge ffmpeg`) before running this script."
        )

    output = Path(args.output_dir)
    videos_dir = output / "videos"
    audio_dir = output / "audio"
    output.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    if not videos_dir.is_dir():
        raise FileNotFoundError(
            f"Videos directory not found: {videos_dir}\n"
            f"Download the VALOR-32K clips from YouTube or BaiduPan and place the *.mp4 "
            f"files there before running this script. See instructions in https://github.com/CASIA-IVA-Lab/VALOR for more details."
        )

    all_videos = sorted(videos_dir.glob("*.mp4"))
    if not all_videos:
        raise FileNotFoundError(
            f"No *.mp4 files found in {videos_dir}.\n"
            f"Download the VALOR-32K clips from YouTube or BaiduPan and place them there. See instructions in https://github.com/CASIA-IVA-Lab/VALOR for more details."
        )

    videos_to_process = all_videos
    if args.max_videos is not None:
        videos_to_process = all_videos[: args.max_videos]
        logger.info(f"--max_videos={args.max_videos}: processing {len(videos_to_process)} of {len(all_videos)} videos")

    # Step 1: Download QA annotations
    download_qa_annotations(output)

    # Step 2: Collect video IDs from QA files
    video_ids = collect_video_ids(output)
    logger.info(f"Found {len(video_ids)} unique video IDs in QA files")
    logger.info(f"Found {len(all_videos)} video files in {videos_dir}")

    # Step 3: Extract audio
    logger.info("Extracting audio from videos...")
    audio_ok = 0
    failures: list[tuple[str, str]] = []
    pbar = tqdm(videos_to_process, desc="Extracting audio") if tqdm else videos_to_process
    for vp in pbar:
        ok, err = extract_audio(vp, audio_dir / f"{vp.stem}.wav")
        if ok:
            audio_ok += 1
        else:
            failures.append((vp.name, err))
    logger.info(f"Extracted audio for {audio_ok}/{len(videos_to_process)} videos")
    if failures:
        logger.warning(f"{len(failures)} audio extraction failures. Showing up to 5:")
        for name, err in failures[:5]:
            logger.warning(f"  {name}: {err}")

    # Step 4: Summary
    processed_stems = {vp.stem for vp in videos_to_process}
    available = processed_stems & {ap.stem for ap in audio_dir.glob("*.wav")}

    for split in ("train", "val", "test"):
        qa_file = output / f"combined_dataset_{split}_flattened.json"
        if qa_file.exists():
            with open(qa_file) as f:
                data = json.load(f)
            usable = sum(1 for qa in data if str(qa["video_id"]) in available)
            logger.info(f"  {split}: {usable}/{len(data)} QA pairs usable")

    logger.info("=" * 60)
    logger.info(f"Dataset ready at: {output}")
    logger.info(f"Usable videos: {len(available)}/{len(video_ids)}")
    if len(available) < len(video_ids):
        missing = len(video_ids) - len(available)
        logger.info(f"{missing} videos missing. For full coverage, download from YouTube or BaiduPan. See instructions in https://github.com/CASIA-IVA-Lab/VALOR for more details.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
