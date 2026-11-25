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

import json
import runpy
import tarfile
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).parents[3]
VALOR_BUILD_SCRIPT = REPO_ROOT / "tutorials" / "data" / "valor32k-avqa" / "build_valor32k_avqa_shards.py"


def _make_data_root(root: Path, split: str, video_ids: list[str], present_ids: list[str]) -> Path:
    """Create a tiny Valor32k-AVQA data root with media only for ``present_ids``."""
    (root / "videos").mkdir(parents=True)
    (root / "audio").mkdir(parents=True)
    for youtube_id in present_ids:
        (root / "videos" / f"{youtube_id}_0.000_10.000.mp4").write_bytes(b"\x00video")
        (root / "audio" / f"{youtube_id}_0.000_10.000.wav").write_bytes(b"\x00audio")
    qa_pairs = [
        {
            "video_id": youtube_id,
            "id": idx,
            "question": "What is making the sound?",
            "options": ["a dog", "a car"],
            "correct_answer_idx": 1,
        }
        for idx, youtube_id in enumerate(video_ids)
    ]
    (root / f"combined_dataset_{split}_flattened.json").write_text(json.dumps(qa_pairs))
    return root


def test_build_shards_leaves_no_shard_when_all_samples_skipped(tmp_path):
    module = runpy.run_path(str(VALOR_BUILD_SCRIPT))
    data_root = _make_data_root(tmp_path / "data", "test", ["missing_a", "missing_b"], present_ids=[])
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    written = module["build_shards"](data_root, output_dir, "test", 10)

    assert written == 0
    assert list(output_dir.glob("test-shard-*.tar")) == []


def test_build_shards_writes_samples_when_media_present(tmp_path):
    module = runpy.run_path(str(VALOR_BUILD_SCRIPT))
    data_root = _make_data_root(tmp_path / "data", "train", ["vid_a", "vid_b"], present_ids=["vid_a", "vid_b"])
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    written = module["build_shards"](data_root, output_dir, "train", 10)

    assert written == 2
    shard = output_dir / "train-shard-000000.tar"
    assert shard.exists()
    with tarfile.open(shard) as tar:
        names = tar.getnames()
    assert sorted(n for n in names if n.endswith("conversation.json")) == [
        "vid_a_0.conversation.json",
        "vid_b_1.conversation.json",
    ]
