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

"""Convert Omni Bench parquet data into local JSONL + media assets."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import wave
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from PIL import Image


PLACEHOLDER_PATTERN = re.compile(r"<image>|<audio>|<video>")


def strip_placeholders(text: str) -> str:
    """Remove multimodal placeholders and normalize whitespace."""
    cleaned = PLACEHOLDER_PATTERN.sub(" ", text)
    return " ".join(cleaned.split())


def _as_list(value: Any) -> list[Any]:
    """Normalize parquet cell values into a Python list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.ndarray):
        return list(value.tolist() if value.dtype != object else value)
    return [value]


def save_image(image_item: Any, output_path: Path) -> Path:
    """Write one image item to disk and return the output path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(image_item, dict):
        img_bytes = image_item.get("bytes")
        img_path = image_item.get("path")
        if img_bytes:
            image = Image.open(BytesIO(img_bytes))
            image.save(output_path)
            return output_path
        if img_path:
            src = Path(img_path)
            if src.resolve() != output_path.resolve():
                shutil.copy2(src, output_path)
            return output_path

    if isinstance(image_item, (bytes, bytearray)):
        image = Image.open(BytesIO(image_item))
        image.save(output_path)
        return output_path

    raise ValueError(f"Unsupported image item type: {type(image_item)!r}")


def save_audio(audio_item: Any, output_path: Path, sample_rate: int) -> Path:
    """Write one audio waveform to a WAV file and return the output path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(audio_item, dict):
        if "array" in audio_item:
            sample_rate = int(audio_item.get("sampling_rate", sample_rate))
            audio_item = audio_item["array"]
        elif "path" in audio_item and audio_item["path"]:
            src = Path(audio_item["path"])
            if src.resolve() != output_path.resolve():
                shutil.copy2(src, output_path)
            return output_path

    audio = np.asarray(audio_item, dtype=np.float32)
    audio = np.nan_to_num(audio)
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767.0).astype(np.int16)

    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())

    return output_path


def save_video(video_item: Any, output_path: Path) -> Path:
    """Write one video item to disk and return the output path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(video_item, dict):
        if video_item.get("bytes"):
            output_path.write_bytes(video_item["bytes"])
            return output_path
        if video_item.get("path"):
            src = Path(video_item["path"])
            if src.resolve() != output_path.resolve():
                shutil.copy2(src, output_path)
            return output_path

    if isinstance(video_item, (bytes, bytearray)):
        output_path.write_bytes(video_item)
        return output_path

    raise ValueError(f"Unsupported video item type: {type(video_item)!r}")


def build_user_content(
    prompt_text: str,
    image_paths: list[str],
    audio_paths: list[str],
    video_paths: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Build multimodal user content by preserving placeholder order from the prompt."""
    video_paths = video_paths or []
    image_idx = 0
    audio_idx = 0
    video_idx = 0
    content: list[dict[str, Any]] = []
    last_end = 0

    for match in PLACEHOLDER_PATTERN.finditer(prompt_text):
        text_segment = prompt_text[last_end : match.start()]
        if text_segment.strip():
            content.append({"type": "text", "text": " ".join(text_segment.split())})

        token = match.group(0)
        if token == "<image>" and image_idx < len(image_paths):
            content.append({"type": "image", "image": image_paths[image_idx]})
            image_idx += 1
        elif token == "<audio>" and audio_idx < len(audio_paths):
            content.append({"type": "audio", "audio": audio_paths[audio_idx]})
            audio_idx += 1
        elif token == "<video>" and video_idx < len(video_paths):
            content.append({"type": "video", "video": video_paths[video_idx]})
            video_idx += 1

        last_end = match.end()

    trailing = prompt_text[last_end:]
    if trailing.strip():
        content.append({"type": "text", "text": " ".join(trailing.split())})

    if not content:
        content.append({"type": "text", "text": strip_placeholders(prompt_text)})

    return content


def convert_dataframe(
    df: pd.DataFrame,
    split: str,
    output_root: Path,
    audio_sample_rate: int,
) -> tuple[Path, int]:
    """Convert one parquet dataframe split into JSONL plus extracted media."""
    split_dir = output_root / split
    media_dir = split_dir / "media"
    jsonl_path = split_dir / f"{split}.jsonl"
    split_dir.mkdir(parents=True, exist_ok=True)
    media_dir.mkdir(parents=True, exist_ok=True)

    num_written = 0
    with jsonl_path.open("w", encoding="utf-8") as jsonl_file:
        for row_idx, (_, row) in enumerate(df.iterrows()):
            sample_id = int(row.get("extra_info", {}).get("index", row_idx))

            prompt_items = _as_list(row.get("prompt"))
            prompt_entry = prompt_items[0] if prompt_items else {}
            prompt_text = prompt_entry.get("content") if isinstance(prompt_entry, dict) else str(prompt_entry)
            prompt_text = str(prompt_text or row.get("extra_info", {}).get("question", ""))

            answer = row.get("reward_model", {}).get("ground_truth") or row.get("extra_info", {}).get("answer") or ""

            image_paths: list[str] = []
            for image_idx, image_item in enumerate(_as_list(row.get("images"))):
                image_path = media_dir / f"{split}_{sample_id:05d}_image_{image_idx}.png"
                image_paths.append(str(save_image(image_item, image_path)))

            audio_paths: list[str] = []
            for audio_idx, audio_item in enumerate(_as_list(row.get("audios"))):
                audio_path = media_dir / f"{split}_{sample_id:05d}_audio_{audio_idx}.wav"
                audio_paths.append(str(save_audio(audio_item, audio_path, audio_sample_rate)))

            video_paths: list[str] = []
            if "videos" in row:
                for video_idx, video_item in enumerate(_as_list(row.get("videos"))):
                    video_path = media_dir / f"{split}_{sample_id:05d}_video_{video_idx}.mp4"
                    video_paths.append(str(save_video(video_item, video_path)))

            example = {
                "conversation": [
                    {
                        "role": "user",
                        "content": build_user_content(prompt_text, image_paths, audio_paths, video_paths),
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": str(answer)}],
                    },
                ],
                "metadata": {
                    "data_source": row.get("data_source"),
                    "ability": row.get("ability"),
                    "reward_model": row.get("reward_model"),
                    "extra_info": row.get("extra_info"),
                },
            }
            jsonl_file.write(json.dumps(example, ensure_ascii=False) + "\n")
            num_written += 1

    return jsonl_path, num_written


def infer_splits(input_paths: Iterable[Path]) -> dict[str, Path]:
    """Infer split names from parquet filenames."""
    result: dict[str, Path] = {}
    for path in input_paths:
        split = path.stem
        result[split] = path
    return result


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("./Omni_Bench_fix_simple"),
        help="Directory containing train/test parquet files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("./omni_bench_fix_simple"),
        help="Output directory for JSONL and extracted media.",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=("train", "test"),
        help="Split names to convert. Defaults to train and test.",
    )
    parser.add_argument(
        "--audio-sample-rate",
        type=int,
        default=16000,
        help="Sample rate to use when writing WAV files from waveform arrays.",
    )
    args = parser.parse_args()

    parquet_paths = []
    for split in args.splits:
        path = args.input_root / f"{split}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing parquet split: {path}")
        parquet_paths.append(path)

    for split, path in infer_splits(parquet_paths).items():
        df = pd.read_parquet(path)
        jsonl_path, num_written = convert_dataframe(df, split, args.output_root, args.audio_sample_rate)
        print(f"[done] split={split} rows={num_written} jsonl={jsonl_path}")


if __name__ == "__main__":
    main()
