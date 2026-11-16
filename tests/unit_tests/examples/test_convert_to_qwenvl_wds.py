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

"""The Qwen-VL WebDataset converter, read back with the Energon decoder of the Qwen-VL task encoder."""

import json
import runpy
import tarfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from megatron.bridge.data.energon.task_encoder_utils import videohandler


pytest.importorskip("webdataset")
pytest.importorskip("qwen_vl_utils")
pytestmark = pytest.mark.unit

SCRIPT = Path(__file__).resolve().parents[3] / "examples/models/qwen/qwen_vl/data/convert_to_qwenvl_wds.py"


def _convert(root: Path, entry: dict) -> dict[str, list]:
    """Convert a dataset of one entry and decode its media fields as the Energon task encoder does."""
    (root / "dataset.json").write_text(json.dumps([entry]))
    output = Path(runpy.run_path(str(SCRIPT))["convert"](str(root), "dataset.json", max_count=10))
    decode = videohandler("pilrgb")
    with tarfile.open(output / "pretrain-0.tar") as archive:
        return {
            field: decode(f"sample.{field}", archive.extractfile(f"sample.{field}").read())
            for field in ("jpgs", "videos")
        }


def _conversation(placeholder: str) -> list[dict]:
    return [{"from": "human", "value": f"{placeholder}What is this?"}, {"from": "gpt", "value": "A test."}]


def test_converted_shard_keeps_the_source_images(tmp_path):
    (tmp_path / "images").mkdir()
    source = Image.new("RGB", (1000, 700), (200, 30, 30))
    source.save(tmp_path / "images" / "red.png")

    media = _convert(tmp_path, {"id": "sample", "image": "images/red.png", "conversations": _conversation("<image>")})

    # The collate resizes the image, so the shard keeps its source pixels.
    assert [image.size for image in media["jpgs"]] == [(1000, 700)]
    assert np.array_equal(np.asarray(media["jpgs"][0]), np.asarray(source))
    assert media["videos"] == []


def test_converted_shard_keeps_the_sampled_video_frames_at_the_source_resolution(tmp_path):
    torchvision_io = pytest.importorskip("torchvision.io")
    pytest.importorskip("av")
    if not hasattr(torchvision_io, "read_video"):
        pytest.skip("The torchvision video reader of qwen-vl-utils needs torchvision.io.read_video.")
    (tmp_path / "videos").mkdir()
    frames = torch.stack([torch.full((180, 320, 3), index * 20, dtype=torch.uint8) for index in range(8)])
    torchvision_io.write_video(str(tmp_path / "videos" / "clip.mp4"), frames, fps=4)

    media = _convert(tmp_path, {"id": "sample", "video": "videos/clip.mp4", "conversations": _conversation("<video>")})

    # qwen-vl-utils samples the 2-second clip at 2 fps: 4 frames, which the collate resizes.
    assert media["jpgs"] == []
    assert [[frame.size for frame in video] for video in media["videos"]] == [[(320, 180)] * 4]
